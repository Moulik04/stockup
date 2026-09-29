"""Track B: the ladder, the decision layer and the policy comparison on UCI Online Retail II.

Runs the design registered in `docs/track_b.md` before any model existed, on one panel at a time:

    TRACK_B_PANEL=primary     REORDERPOINT_TRACK=b python -m reorderpoint.track_b run
    TRACK_B_PANEL=robustness  REORDERPOINT_TRACK=b python -m reorderpoint.track_b run
    REORDERPOINT_TRACK=b python -m reorderpoint.track_b report

(`make track-b` does all three.) `run` writes `results.json` beside the panel; `report` reads both
panels' and writes `reports/track_b_online_retail_<date>.md`, which leads with which findings
replicated. One process per panel because the panel path, and the caches beside it, are fixed at
import (`backtest.PANEL_PATH`).

Everything past the forecasts reuses Track A's code (the calibrated sizing, the (s, S) simulation,
the paired bootstrap over series), through the grain abstraction: nothing here re-implements a
decision rule. What is new is the four verdicts, read exactly as `docs/track_b.md` fixes them, and
the 4 x 3 economics grid they are recomputed over.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import time
from dataclasses import replace
from unittest import mock

import numpy as np
import pandas as pd

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import croston as cr
from reorderpoint import divergence as dv
from reorderpoint import optimal_target as opt
from reorderpoint import production_model as pm
from reorderpoint.config import load_config
from reorderpoint.grain import GRAIN

LADDER = tuple(bt.MODEL_FACTORIES)
# The seven non-naive models, "the cluster" of docs/track_b.md. SeasonalNaive is in the ladder,
# not in the cluster.
CLUSTER = (
    "MovingAverage",
    "AutoETS",
    "AutoTheta",
    "LightGBM",
    "CrostonClassic",
    "CrostonSBA",
    "TSB",
)
MARGINS = (0.20, 0.275, 0.40, 0.50)
LEADS = (1, 2, 4)  # weeks
LOTS = (0.0, 2.0)  # S = s, and the shipped S = s + 2 x lead-time demand
DEFAULT_MARGIN, DEFAULT_LEAD = 0.275, 2
TARGET = 0.95
# The thresholds of finding 2, as registered.
COVERAGE_TOLERANCE, CSL_GAP, UNDERSHOOT_MIN = 0.03, 0.03, 1 / 3

RESULTS_DIR = bt.PANEL_PATH.parent / "results"
SLIM = [
    "model",
    "fold",
    "series_id",
    "total_cost",
    "holding_cost",
    "stockout_cost",
    "units_demanded",
    "units_shipped",
    "cycles",
    "stockout_cycles",
    "orders",
    "reorder_point",
    "order_up_to",
    "on_hand_start",
]


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _nan_to_none(x):
    """Strict JSON has no NaN: an undefined figure (a share over zero stockouts) is null."""
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def cell_key(margin: float, lead: int) -> str:
    return f"m{margin:g}|L{lead}"


# ---- forecasts, calibration, one simulated cell -------------------------------------------------


def forecasts_for(panel: pd.DataFrame) -> pd.DataFrame:
    if dv.FORECASTS_PATH.exists():
        return pd.read_parquet(dv.FORECASTS_PATH)
    _log("fitting every model on every fold (the forecast cache)")
    forecasts = dv.collect_forecasts(panel)
    forecasts.to_parquet(dv.FORECASTS_PATH, index=False)
    return forecasts


def calibration_for(panel: pd.DataFrame, costs) -> pd.DataFrame:
    """Fold-1 calibration residuals for this lead time; one cache per lead time, since the residual
    is a lead-time total and changes with it."""
    path = RESULTS_DIR / f"calibration_lead{costs.lead_time_days}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(ab, "CALIB_PATH", path):
        return ab.calibration_residuals(panel, costs)


def simulate(panel, forecasts, calib, costs, lot: float) -> pd.DataFrame:
    """The shipped cell (normal form, intermittency buckets, 95%) at one lot multiple."""
    _, detail, _ = ab.run_cell(
        panel,
        forecasts,
        calib,
        cr.LEGACY[0],
        cr.LEGACY[1],
        costs,
        service_level=TARGET,
        lot_multiple=lot,
    )
    return detail[SLIM].reset_index(drop=True)


def summarise(detail: pd.DataFrame) -> dict:
    """Per model: mean cost per fold and its split, pooled CSL and fill rate, orders and cycles."""
    table = cr.cost_table(detail, list(LADDER))
    per_fold = detail.groupby(["model", "fold"]).agg(
        holding=("holding_cost", "sum"),
        stockout=("stockout_cost", "sum"),
        orders=("orders", "sum"),
        cycles=("cycles", "sum"),
    )
    mean = per_fold.groupby("model").mean()
    return {
        m: {
            "cost": float(table.loc[m, "cost"]),
            "csl": float(table.loc[m, "csl"]),
            "fill": float(table.loc[m, "fill"]),
            "holding": float(mean.loc[m, "holding"]),
            "stockout": float(mean.loc[m, "stockout"]),
            "orders": float(mean.loc[m, "orders"]),
            "cycles": float(mean.loc[m, "cycles"]),
        }
        for m in LADDER
    }


# ---- the verdicts, as docs/track_b.md fixes them ------------------------------------------------


def _ci(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = boot.bootstrap_diff(a, b, "total_cost", group_cols=["fold"])
    return {"point": out["point"], "lo": out["ci_lo"], "hi": out["ci_hi"]}


def finding_1(d0: pd.DataFrame, d2: pd.DataFrame, s2: dict) -> dict:
    """Policy over model: every one of the eight models saves money going from S = s to
    S = s + 2 x lead-time demand, with a CI excluding zero, and the mean saving exceeds the spread
    between the cheapest and dearest cluster model under the shipped policy."""
    by_model = {}
    for m in LADDER:
        by_model[m] = _ci(d2[d2["model"] == m], d0[d0["model"] == m])  # lot 2 minus lot 0
    saving = -float(np.mean([v["point"] for v in by_model.values()]))
    costs = [s2[m]["cost"] for m in CLUSTER]
    spread = max(costs) - min(costs)
    all_exclude_zero = all(v["hi"] < 0 for v in by_model.values())
    return {
        "by_model": by_model,
        "mean_saving": saving,
        "spread": spread,
        "all_exclude_zero": all_exclude_zero,
        "replicated": bool(all_exclude_zero and saving > spread),
    }


def findings_3_4(d2: pd.DataFrame) -> tuple[dict, dict]:
    """The cluster tie: all 21 pairs contain zero (literal), with the six against LightGBM (Track
    A's design) reported beside them. Finding 4: the 4-week mean against LightGBM."""
    by = {m: d2[d2["model"] == m] for m in CLUSTER}
    pairs = {}
    for a, b in itertools.combinations(CLUSTER, 2):
        pairs[f"{a} - {b}"] = _ci(by[a], by[b])
    excluding = {k: v for k, v in pairs.items() if not (v["lo"] <= 0 <= v["hi"])}
    vs_lgb = {m: _ci(by[m], by["LightGBM"]) for m in CLUSTER if m != "LightGBM"}
    f3 = {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "excluding_zero": excluding,
        "vs_lightgbm": vs_lgb,
        "replicated": len(excluding) == 0,
    }
    ma = vs_lgb["MovingAverage"]
    f4 = {"moving_average_vs_lightgbm": ma, "replicated": bool(ma["lo"] <= 0 <= ma["hi"])}
    return f3, f4


def finding_2(panel, forecasts, calib, costs, d2: pd.DataFrame) -> dict:
    """The calibrated quantile does what it says; the shortfall is the policy. Per lead time (the
    margin changes prices, not cycles or coverage)."""
    coverage = pm.interval_evaluation(forecasts, calib, costs, CLUSTER)
    covers = dict(zip(coverage["model"], coverage["reorder_point_covers"], strict=True))
    cl = d2[d2["model"].isin(CLUSTER)]
    csl = 1 - float(cl["stockout_cycles"].sum()) / float(cl["cycles"].sum())
    decomposition = opt.stockout_decomposition(cl, panel, None, costs.lead_time_days)
    mean_cover = float(np.mean(list(covers.values())))
    within = all(abs(c - TARGET) <= COVERAGE_TOLERANCE for c in covers.values())
    gap = mean_cover - csl
    return {
        "coverage_by_model": {k: float(v) for k, v in covers.items()},
        "mean_coverage": mean_cover,
        "all_within_tolerance": within,
        "csl_lot2": csl,
        "gap": gap,
        "cycles": decomposition["cycles"],
        "stockout_cycles": decomposition["stockout_cycles"],
        "undershoot_share": decomposition["undershoot_share"],
        "csl_if_sizing_only": decomposition["csl_if_sizing_only"],
        "replicated": bool(
            within and gap >= CSL_GAP and decomposition["undershoot_share"] >= UNDERSHOOT_MIN
        ),
    }


# ---- one panel, end to end ----------------------------------------------------------------------


def run() -> dict:
    base = load_config().costs
    panel = bt.load_eval_panel()
    folds = bt.make_folds(panel)
    name = bt.PANEL_PATH.parent.name
    _log(f"panel {name}: {panel['series_id'].nunique():,} series x {panel['date'].nunique()} weeks")
    forecasts = forecasts_for(panel)

    result = {
        "panel": name,
        "n_series": int(panel["series_id"].nunique()),
        "folds": [
            [f.index, str(f.train_end.date()), str(f.test_start.date()), str(f.test_end.date())]
            for f in folds
        ],
        "grain": {
            "unit": GRAIN.unit,
            "horizon": GRAIN.horizon,
            "periods_per_year": GRAIN.periods_per_year,
        },
        "economics": {
            "holding_cost_rate_per_period": base.holding_cost_rate,
            "service_level_target": base.service_level_target,
            "lot_multiple": base.lot_multiple,
        },
        "cells": {},
        "finding_2": {},
    }

    acc = cr.accuracy_table(panel, forecasts, list(LADDER))
    result["accuracy"] = {
        "by_model": json.loads(
            acc.groupby("model")[["mase", "rmsse", "coverage_80"]].mean().to_json(orient="index")
        ),
        "by_fold": json.loads(acc.to_json(orient="records")),
    }
    fallback = {}
    for m in ("SeasonalNaive",):
        fallback[m] = "52-week season in the folds; plain naive on the 50-week calibration window"
    result["fallbacks"] = fallback

    kept = {}
    for lead in LEADS:
        lead_costs = replace(base, lead_time_days=lead)
        calib = calibration_for(panel, lead_costs)
        detail_lot2_default = None
        for margin in MARGINS:
            costs = replace(lead_costs, gross_margin=margin)
            _log(f"cell margin {margin:g}, lead {lead} wk: simulating S = s and S = s + 2 x LT")
            d0 = simulate(panel, forecasts, calib, costs, 0.0)
            d2 = simulate(panel, forecasts, calib, costs, 2.0)
            s0, s2 = summarise(d0), summarise(d2)
            f1 = finding_1(d0, d2, s2)
            f3, f4 = findings_3_4(d2)
            result["cells"][cell_key(margin, lead)] = {
                "lot0": s0,
                "lot2": s2,
                "finding_1": f1,
                "finding_3": f3,
                "finding_4": f4,
            }
            if margin == DEFAULT_MARGIN:
                detail_lot2_default = d2
                if lead == DEFAULT_LEAD:
                    kept["lot0"], kept["lot2"] = d0, d2
        _log(f"finding 2 at lead {lead} wk")
        result["finding_2"][f"L{lead}"] = finding_2(
            panel, forecasts, calib, lead_costs, detail_lot2_default
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kept["lot0"].to_parquet(RESULTS_DIR / "detail_default_lot0.parquet", index=False)
    kept["lot2"].to_parquet(RESULTS_DIR / "detail_default_lot2.parquet", index=False)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(_nan_to_none(result), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    _log(f"wrote {RESULTS_DIR / 'results.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if GRAIN.name != "week":
        raise SystemExit("run with REORDERPOINT_TRACK=b")
    t0 = time.time()
    if args.command == "run":
        run()
    _log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
