"""Track A's headline numbers, frozen, so that refactors of the code path they run through can be
shown not to have moved any of them.

The weekly Track B pipeline needed a period abstraction (fold arithmetic, model seasonality and
frequency, defaults) in modules Track A also runs through. Nothing about Track A was meant to
change, and this is how that is checked rather than assumed:
`python -m reorderpoint.track_a_baseline --write` computed `tests/fixtures/track_a_baseline.json` on
the code as it was **before** the refactor, and `tests/test_track_a_regression.py` recomputes it on
every run and requires the same numbers.

What it covers, all on the real 400-series, 4-fold, HOBBIES setup every Track A report uses:

- the fold dates;
- the shipped cell (legacy scheme, 95% target, `S = s + 2 x lead-time demand`) and the original
  `S = s` policy, per model: cost per fold, holding and stockout split, cycle service level, fill
  rate, orders — the cost table and the policy comparison;
- held-out coverage of the reorder point and of the displayed band (the v1.2 calibration checks);
- the divergence headline;
- a live refit of the cheaper models on the last fold, compared to the cached forecasts: it goes
  through the fold, factory, season-length and frequency code that the cached numbers bypass.

It needs the caches (`make divergence`, `make croston`) and the ingested panel, none of which are
committed; without them the test skips and the synthetic golden test in the same file still runs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import croston as cr
from reorderpoint import divergence as dv
from reorderpoint import production_model as pm
from reorderpoint.config import REPO_ROOT, load_config

BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "track_a_baseline.json"
# Refit on the last fold and compared with the cache. AutoETS and AutoTheta are left out: minutes of
# wall time each, and the season-length/frequency path they share with SeasonalNaive is covered.
REFIT_MODELS = ("SeasonalNaive", "MovingAverage", "CrostonClassic", "CrostonSBA", "TSB", "LightGBM")
REFIT_TOLERANCE = 1e-6


def caches_available() -> bool:
    return bt.PANEL_PATH.exists() and dv.FORECASTS_PATH.exists() and ab.CALIB_PATH.exists()


def _policy_table(panel, forecasts, calib, costs, lot: float) -> dict:
    _, detail, _ = ab.run_cell(
        panel,
        forecasts,
        calib,
        cr.LEGACY[0],
        cr.LEGACY[1],
        costs,
        service_level=costs.service_level_target,
        lot_multiple=lot,
    )
    models = list(bt.MODEL_FACTORIES)
    table = cr.cost_table(detail, models)
    per_fold = detail.groupby(["model", "fold"]).agg(
        holding=("holding_cost", "sum"),
        stockout=("stockout_cost", "sum"),
        orders=("orders", "sum"),
        cycles=("cycles", "sum"),
    )
    mean = per_fold.groupby("model")[["holding", "stockout", "orders", "cycles"]].mean()
    return {
        m: {
            "cost_per_fold": float(table.loc[m, "cost"]),
            "csl": float(table.loc[m, "csl"]),
            "fill_rate": float(table.loc[m, "fill"]),
            "holding_per_fold": float(mean.loc[m, "holding"]),
            "stockout_per_fold": float(mean.loc[m, "stockout"]),
            "orders_per_fold": float(mean.loc[m, "orders"]),
            "cycles_per_fold": float(mean.loc[m, "cycles"]),
        }
        for m in models
    }


def refit_check(panel: pd.DataFrame, forecasts: pd.DataFrame) -> dict:
    """Max absolute difference between a live last-fold refit and the cached forecasts, per model
    and quantile column."""
    fold = bt.make_folds(panel)[-1]
    train = panel[panel["date"] <= fold.train_end]
    test = panel[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)]
    horizon = bt.HORIZON  # the test window is exactly HORIZON periods
    out = {}
    for name in REFIT_MODELS:
        model = bt.MODEL_FACTORIES[name](horizon)
        model.fit(train)
        live = model.predict_quantiles(horizon, future_exog=test.drop(columns=["y"]))
        cached = forecasts[(forecasts["model"] == name) & (forecasts["fold"] == fold.index)]
        merged = live.merge(cached, on=["series_id", "date"], suffixes=("_live", "_cached"))
        assert len(merged) == len(cached) == len(live), f"{name}: rows differ from the cache"
        out[name] = {
            col: float((merged[f"{col}_live"] - merged[f"{col}_cached"]).abs().max())
            for col in ("p10", "p50", "p90")
        }
    return out


def compute(refit: bool = True) -> dict:
    costs = load_config().costs
    panel = bt.load_eval_panel()
    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = pd.read_parquet(ab.CALIB_PATH)
    folds = bt.make_folds(panel)

    result = {
        "folds": [
            [f.index, str(f.train_end.date()), str(f.test_start.date()), str(f.test_end.date())]
            for f in folds
        ],
        "costs": {
            "lead_time": costs.lead_time_days,
            "holding_cost_rate": costs.holding_cost_rate,
            "gross_margin": costs.gross_margin,
            "lot_multiple": costs.lot_multiple,
            "service_level_target": costs.service_level_target,
        },
        "constants": {
            "horizon": bt.HORIZON,
            "n_folds": bt.N_FOLDS,
            "season_length": bt.SEASON_LENGTH,
            "n_series_sample": bt.N_SERIES_SAMPLE,
            "panel_rows": int(len(panel)),
            "panel_series": int(panel["series_id"].nunique()),
        },
        "shipped_cell_lot_2": _policy_table(panel, forecasts, calib, costs, 2.0),
        "original_policy_lot_0": _policy_table(panel, forecasts, calib, costs, 0.0),
    }
    coverage = pm.interval_evaluation(forecasts, calib, costs, pm.INTERVAL_MODELS)
    result["interval_evaluation"] = json.loads(coverage.to_json(orient="records"))
    headline = dv.build_report(forecasts, None)[1]["headline"]
    result["divergence_headline"] = {k: float(v) for k, v in headline.items()}
    if refit:
        result["refit_max_abs_diff"] = refit_check(panel, forecasts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help=f"write {BASELINE_PATH}")
    args = parser.parse_args()
    if not caches_available():
        raise SystemExit("Track A caches or panel missing: run `make data divergence croston`")
    result = compute()
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if args.write:
        BASELINE_PATH.write_text(text)
        print(f"wrote {BASELINE_PATH}")
    else:
        print(text)
    worst = max(v for m in result["refit_max_abs_diff"].values() for v in m.values())
    print(f"largest live-refit vs cache difference: {worst:.3g}")
    assert np.isfinite(worst)


if __name__ == "__main__":
    main()
