"""Track A's numbers must not move when the code path they share with Track B changes.

Two layers. The synthetic golden test runs everywhere, CI included: a deterministic daily panel
goes through the real fold arithmetic, the real model factories (season length, frequency, window)
and the real cost simulation, and the results are compared to values captured on the code
**before** the period abstraction was added. The real-data test compares the full Track A headline
set (cost table, cycle service level, fill rate, the S = s versus S > s policy comparison,
held-out coverage, a live refit against the cached forecasts) to
`tests/fixtures/track_a_baseline.json`, frozen from the same pre-refactor code; it needs the
caches, so it skips where they are absent.

If either fails, Track A moved. Do not update the expected values to make it pass: find out why.
"""

import os
import platform

import numpy as np
import pandas as pd
import pytest

from reorderpoint import backtest as bt
from reorderpoint import decision as dec
from reorderpoint.config import CostParams
from reorderpoint.models import naive

COSTS = CostParams(
    holding_cost_rate=0.25 / 365,
    stockout_penalty_per_unit=1.0,
    lead_time_days=7,
    service_level_target=0.95,
    gross_margin=0.275,
    lot_multiple=2.0,
)


def _serial(model):
    model._n_jobs = 1  # a process pool per fit is the slow part on a tiny panel
    return model


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260929)
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    rows = []
    for i in range(10):
        base = [0.2, 0.5, 1.0, 2.0, 4.0][i % 5]
        weekly = 1 + 0.6 * np.sin(2 * np.pi * np.arange(len(dates)) / 7 + i)
        demand = rng.poisson(np.clip(base * weekly, 0.01, None))
        for d, y in zip(dates, demand, strict=True):
            rows.append({"series_id": f"S{i}", "date": d, "y": float(y), "price": 2.0 + 0.1 * i})
    return pd.DataFrame(rows)


@pytest.fixture
def serial_registry(monkeypatch):
    monkeypatch.setattr(
        bt,
        "MODEL_FACTORIES",
        {
            "SeasonalNaive": lambda h: _serial(naive.seasonal_naive(h)),
            "MovingAverage": lambda h: _serial(naive.moving_average(h)),
        },
    )


def _run(panel: pd.DataFrame) -> tuple[list, dict]:
    """(fold dates, per model and lot multiple: cost, its split, fill rate, CSL, cycles)."""
    folds = bt.make_folds(panel)
    fold_dates = [[f.index, str(f.train_end.date()), str(f.test_end.date())] for f in folds]
    numbers = {}
    for model in bt.MODEL_FACTORIES:
        by_lot: dict[float, list[dict]] = {0.0: [], 2.0: []}
        for fold in folds:
            rows, _, _ = dec.evaluate_fold_cost_policies(
                panel, fold, model, COSTS, policies=("csl",), lot_multiples=(0.0, 2.0)
            )
            for lot, row in zip((0.0, 2.0), rows, strict=True):
                by_lot[lot].append(row)
        for lot, rows in by_lot.items():
            numbers[f"{model}|lot{lot:g}"] = {
                "total_cost": float(sum(r["total_cost"] for r in rows)),
                "holding_cost": float(sum(r["holding_cost"] for r in rows)),
                "stockout_cost": float(sum(r["stockout_cost"] for r in rows)),
                "fill_rate": float(np.mean([r["fill_rate"] for r in rows])),
                "csl": float(np.nanmean([r["cycle_service_level"] for r in rows])),
                "cycles": int(sum(r["cycles"] for r in rows)),
            }
    return fold_dates, numbers


def _golden() -> dict:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return GOLDEN_ARM64
    if machine in {"x86_64", "amd64"}:
        return GOLDEN_X86
    pytest.skip(f"no golden values captured for {machine}")


def test_synthetic_daily_panel_reproduces_the_frozen_pre_refactor_numbers(serial_registry):
    golden = _golden()
    fold_dates, numbers = _run(_panel())
    assert fold_dates == GOLDEN_FOLDS
    assert numbers.keys() == golden.keys()
    problems = []
    for key, expected in golden.items():
        got = numbers[key]
        if got["cycles"] != expected["cycles"]:  # a count of events: exact
            problems.append((key, "cycles", got["cycles"], expected["cycles"]))
        for field, value in expected.items():
            if field != "cycles" and got[field] != pytest.approx(value, rel=FLOAT_TOLERANCE):
                problems.append((key, field, got[field], value))
    assert not problems, f"{len(problems)} numbers moved, first: {problems[:4]}"


GOLDEN_FOLDS = [
    [1, "2025-07-07", "2025-08-04"],
    [2, "2025-08-04", "2025-09-01"],
    [3, "2025-09-01", "2025-09-29"],
    [4, "2025-09-29", "2025-10-27"],
]
# Platform matters here, and the test says so instead of hiding it behind a tolerance. statsforecast
# and numpy give slightly different floats on x86 and arm64, and one float-sensitive decision (an
# order triggered when on-hand falls below the reorder point) flips for MovingAverage at lot 2: 62
# replenishment cycles on arm64, 61 on x86, with costs off by up to about 1%. So there is one set of
# golden values per architecture, each captured from the code as it was before the period
# abstraction (commit ffdff7a) and compared exactly, on its own platform. The arm64 set reproduces
# on macOS and on Linux; the x86 set was captured on linux/amd64. On each platform the code before
# and after the refactor gives identical results in every field of every entry (DECISIONS.md,
# 2026-09-30 addendum). An unknown architecture skips rather than guess.
FLOAT_TOLERANCE = 1e-6  # within one platform the results are deterministic; this is float slack

GOLDEN_ARM64 = {
    "SeasonalNaive|lot0": {
        "total_cost": 322.33548804327756,
        "holding_cost": 5.846779944622554,
        "stockout_cost": 316.488708098655,
        "fill_rate": 0.7250625471792734,
        "csl": 0.5119731800766283,
        "cycles": 115,
    },
    "SeasonalNaive|lot2": {
        "total_cost": 67.62186521275511,
        "holding_cost": 19.569311453309826,
        "stockout_cost": 48.05255375944529,
        "fill_rate": 0.955132909231963,
        "csl": 0.5741071428571429,
        "cycles": 63,
    },
    "MovingAverage|lot0": {
        "total_cost": 274.38691293464194,
        "holding_cost": 6.223121872395113,
        "stockout_cost": 268.16379106224684,
        "fill_rate": 0.7682793418586472,
        "csl": 0.5542898193760263,
        "cycles": 117,
    },
    "MovingAverage|lot2": {
        "total_cost": 34.52170979895047,
        "holding_cost": 21.694523830349375,
        "stockout_cost": 12.827185968601082,
        "fill_rate": 0.9884946178523204,
        "csl": 0.8243609943977592,
        "cycles": 62,
    },
}


GOLDEN_X86 = {
    "SeasonalNaive|lot0": {
        "total_cost": 322.33548804327756,
        "holding_cost": 5.846779944622554,
        "stockout_cost": 316.488708098655,
        "fill_rate": 0.7250625471792734,
        "csl": 0.5119731800766283,
        "cycles": 115,
    },
    "SeasonalNaive|lot2": {
        "total_cost": 67.59018316485971,
        "holding_cost": 19.537629405414425,
        "stockout_cost": 48.05255375944528,
        "fill_rate": 0.955132909231963,
        "csl": 0.5741071428571429,
        "cycles": 63,
    },
    "MovingAverage|lot0": {
        "total_cost": 274.38691293464194,
        "holding_cost": 6.223121872395113,
        "stockout_cost": 268.16379106224684,
        "fill_rate": 0.7682793418586472,
        "csl": 0.5542898193760263,
        "cycles": 117,
    },
    "MovingAverage|lot2": {
        "total_cost": 34.6608971749083,
        "holding_cost": 21.69788095315458,
        "stockout_cost": 12.963016221753712,
        "fill_rate": 0.9882322796999866,
        "csl": 0.8233193277310924,
        "cycles": 61,
    },
}


# ---- the real Track A headline set ------------------------------------------------------------

REAL_TOLERANCE = 1e-9


def _leaf_pairs(got, expected, path=""):
    """Walk two JSON-shaped trees in step, yielding (path, got, expected) at every number."""
    if isinstance(expected, dict):
        assert isinstance(got, dict) and got.keys() == expected.keys(), path
        for key in expected:
            yield from _leaf_pairs(got[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected), path
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            yield from _leaf_pairs(g, e, f"{path}[{i}]")
    else:
        yield path, got, expected


@pytest.mark.skipif(
    os.environ.get("STOCKUP_SKIP_REGRESSION") == "1", reason="STOCKUP_SKIP_REGRESSION=1"
)
def test_track_a_headline_numbers_match_the_frozen_baseline():
    import json

    from reorderpoint import track_a_baseline as base

    if not base.caches_available():
        pytest.skip("Track A panel and caches not on this machine (make data divergence croston)")
    expected = json.loads(base.BASELINE_PATH.read_text())
    got = json.loads(json.dumps(base.compute()))
    moved = []
    for path, g, e in _leaf_pairs(got, expected):
        if isinstance(e, str) or e is None or isinstance(e, bool):
            if g != e:
                moved.append((path, g, e))
        elif path.startswith("/refit_max_abs_diff"):
            if not g <= base.REFIT_TOLERANCE:
                moved.append((path, g, e))
        elif g != pytest.approx(e, rel=REAL_TOLERANCE, abs=REAL_TOLERANCE):
            moved.append((path, g, e))
    assert not moved, f"Track A moved at {len(moved)} numbers, first: {moved[:5]}"
