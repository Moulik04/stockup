"""CI smoke test for the serving image: build a fixture, run the container, check real numbers.

Nothing built the Dockerfile in CI, so it sat broken (a dependency with no wheel for the pinned
Python, and no compiler in the slim image) until someone happened to build it. This closes that.

  prepare CONTEXT_DIR   assemble a Docker build context: the project files the Dockerfile copies,
                        plus a tiny synthetic panel and the model + calibration `make train` would
                        write for it. Never touches the repo's own data/ or models/.
  check BASE_URL        wait for /health, then hit /reorder, /forecast and /metrics and compare the
                        answers to values worked out by hand below — not by calling the code under
                        test. Exits nonzero on any mismatch.

The fixture is built so the expected numbers are derivable on paper. 70 daily rows, two series:

  A  constant demand of 2                       -> 28-day mean 2, lead-time forecast 7 x 2 = 14
  B  demand 4 on even day indexes, else 0       -> 28-day mean 2 (14 four's, 14 zeros), same 14

The calibration holds out the last 7 days (indexes 63-69) and fits on 0-62. A sold 14 in them
(residual 0). B sold 4 on indexes 64, 66, 68 = 12 (residual -2). Both series have a zero rate of at
most 0.5, so they share one bucket, and its pooled std is stdev([0, -2]) = sqrt(2). Then, at the
default 95% service level and 7-day lead time:

  safety stock  = z(0.95) * sqrt(2)                 reorder point = 14 + safety stock
  order-up-to S = reorder point + 2 * 14            (default lot multiple 2)
  /forecast     = 2 +/- z(0.90) * sqrt(2) / sqrt(7) per day, lower edge clipped at zero

This script imports `reorderpoint` (for `prepare`), which `scripts/` files normally must not: direct
invocation does not put the repo root on `sys.path`, so it is inserted below, as in
`scripts/bridges2/run_deep_backtest.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LEAD_TIME = 7
LOT_MULTIPLE = 2.0
SERVICE_LEVEL = 0.95
BAND_LEVEL = 0.90  # the band's upper edge: an 80% central interval
N_DAYS = 70
SERIES_A = "SMOKE_CONSTANT"
SERIES_B = "SMOKE_ALTERNATING"
COPIED = ("Dockerfile", "pyproject.toml", "uv.lock", ".python-version")


def expected() -> dict[str, float]:
    """The numbers the fixture must produce, from first principles."""
    sigma = statistics.stdev([0.0, -2.0])  # A's residual 0, B's residual -2
    z = statistics.NormalDist().inv_cdf
    lead_mean = LEAD_TIME * 2.0
    safety = z(SERVICE_LEVEL) * sigma
    reorder_point = lead_mean + safety
    half = z(BAND_LEVEL) * sigma / math.sqrt(LEAD_TIME)
    return {
        "safety_stock": safety,
        "reorder_point": reorder_point,
        "order_quantity_a": reorder_point + LOT_MULTIPLE * lead_mean - 3.0,  # on hand 3 < s
        "order_quantity_b": reorder_point + LOT_MULTIPLE * lead_mean,  # on hand 0
        "band_half_width": half,
    }


def write_fixture(root: Path) -> None:
    """Write the synthetic panel and the model + calibration `make train` produces for it."""
    import joblib
    import pandas as pd

    from reorderpoint import calibration as cal
    from reorderpoint import train

    dates = pd.date_range("2026-01-01", periods=N_DAYS)
    idx = range(N_DAYS)
    panel = pd.concat(
        [
            pd.DataFrame({"series_id": SERIES_A, "date": dates, "y": 2.0, "price": 1.0}),
            pd.DataFrame(
                {
                    "series_id": SERIES_B,
                    "date": dates,
                    "y": [4.0 if i % 2 == 0 else 0.0 for i in idx],
                    "price": 1.0,
                }
            ),
        ],
        ignore_index=True,
    )
    panel_path = root / "data" / "track_a" / "processed" / "panel.parquet"
    model_path = root / "models" / "production" / "model.joblib"
    calibration_path = root / "models" / "production" / "safety_stock_calibration.joblib"
    for path in (panel_path, model_path, calibration_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    joblib.dump(train.train_production_model(panel), model_path)
    cal.save(train.train_production_calibration(panel, LEAD_TIME), calibration_path)


def prepare(context: Path) -> None:
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    for name in COPIED:
        shutil.copy2(REPO_ROOT / name, context / name)
    shutil.copytree(
        REPO_ROOT / "reorderpoint",
        context / "reorderpoint",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    write_fixture(context)
    print(f"build context ready: {context}")


def check_answers(reorder: list[dict], forecast: list[dict], metrics: dict) -> list[str]:
    """Every way the container's answers differ from the hand-derived ones; empty if none."""
    want = expected()
    problems: list[str] = []

    def close(label: str, got: float, exp: float, tol: float = 1e-3) -> None:
        if not math.isclose(got, exp, abs_tol=tol):
            problems.append(f"{label}: got {got:.4f}, expected {exp:.4f}")

    by_id = {row["series_id"]: row for row in reorder}
    if set(by_id) != {SERIES_A, SERIES_B}:
        problems.append(f"/reorder returned series {sorted(by_id)}")
    else:
        for sid in (SERIES_A, SERIES_B):
            close(f"reorder_point[{sid}]", by_id[sid]["reorder_point"], want["reorder_point"])
            close(f"safety_stock[{sid}]", by_id[sid]["safety_stock"], want["safety_stock"])
        close("order_quantity[A]", by_id[SERIES_A]["order_quantity"], want["order_quantity_a"])
        close("order_quantity[B]", by_id[SERIES_B]["order_quantity"], want["order_quantity_b"])

    if len(forecast) != 3 * 2:
        problems.append(f"/forecast returned {len(forecast)} rows, expected 6")
    for row in forecast:
        close(f"p50[{row['series_id']}]", row["p50"], 2.0)
        close(f"p90[{row['series_id']}]", row["p90"], 2.0 + want["band_half_width"])
        close(f"p10[{row['series_id']}]", row["p10"], max(0.0, 2.0 - want["band_half_width"]))

    calibration = metrics.get("safety_stock_calibration", {})
    if calibration.get("model") != "MovingAverage":
        problems.append(f"/metrics model is {calibration.get('model')!r}, not 'MovingAverage'")
    if metrics.get("n_series") != 2:
        problems.append(f"/metrics n_series is {metrics.get('n_series')}, expected 2")
    return problems


def _request(url: str, payload: dict | None = None) -> tuple[int, dict | list]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode(errors="replace")}


def check(base_url: str, wait_seconds: float = 90.0) -> int:
    deadline = time.time() + wait_seconds
    while True:
        try:
            status, _ = _request(f"{base_url}/health")
            if status == 200:
                break
        except OSError:
            pass
        if time.time() > deadline:
            print(f"FAIL: {base_url}/health not answering 200 within {wait_seconds:.0f}s")
            return 1
        time.sleep(2)

    calls = {
        "/reorder": {"series_ids": [SERIES_A, SERIES_B], "on_hand": {SERIES_A: 3.0}},
        "/forecast": {"series_ids": [SERIES_A, SERIES_B], "horizon": 3},
    }
    answers: dict[str, dict | list] = {}
    for path, payload in calls.items():
        status, body = _request(f"{base_url}{path}", payload)
        if status != 200:
            print(f"FAIL: POST {path} returned {status}: {body}")
            return 1
        answers[path] = body
    status, metrics = _request(f"{base_url}/metrics")
    if status != 200:
        print(f"FAIL: GET /metrics returned {status}: {metrics}")
        return 1

    problems = check_answers(answers["/reorder"], answers["/forecast"], metrics)
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        return 1
    want = expected()
    print(
        "OK: /health 200; /reorder reorder_point "
        f"{want['reorder_point']:.3f}, safety_stock {want['safety_stock']:.3f}; /forecast band "
        f"2 +/- {want['band_half_width']:.3f}; /metrics model MovingAverage"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("context", type=Path)
    sub.add_parser("check").add_argument("base_url")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.context)
        return 0
    return check(args.base_url.rstrip("/"))


if __name__ == "__main__":
    sys.exit(main())
