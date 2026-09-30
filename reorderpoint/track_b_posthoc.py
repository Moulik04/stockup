"""Track B, POST HOC: how much do the verdicts depend on how the safety-stock buffer is calibrated?

This is **not registered** and **changes no verdict**. What the registered simulation does (it is
`ablate_safety_stock.run_cell`, unchanged): fold 1's buffer comes from the lead-time window just
before
it; fold k's from the residuals of fold k-1's *first* lead-time window, one observation per series;
either way pooled by the intermittency buckets Track A uses, one normal sigma per bucket. (An
earlier
description of this in the docs and the report said one window sized all four folds. That was wrong
for the cost simulation and is corrected in `DECISIONS.md`, 2026-09-30; it is true only of the
held-out coverage check, which the registration defined that way, as Track A's is.) After the
results
two things about the calibration were found:

- fold 1's window is late November, the autumn peak, at about twice the demand of the test folds;
- at weekly grain the 50%-zero-periods intermittency split puts about 98% of series in one bucket,
  so
  one pooled absolute sigma, driven by a handful of very large residuals, becomes every series'
  buffer:
  seven times the median series' expected lead-time demand (measured on fold 1's residuals).

This separates the window from the pooling with a grid, on the same cached forecasts (nothing
refit):

    window:  single        fold 1's window sizes every fold (what the served system does)
             first_window  fold k from fold k-1's first lead-time window (the registered simulation)
             all_windows   fold k from every complete lead-time window in fold k-1
    pooling: intermittency the registered buckets
             volume_quintile  Track A's existing alternative: five equal-sized buckets by volume

The (first_window, intermittency) cell IS the registered simulation, re-implemented here, and `run`
requires it to reproduce the registered cost table to the last digit before anything else is
reported. (The first version of this module encoded the wrong description above and failed that
check; it is what caught the error.) The other five are reported beside it under the same four
verdict rules. Nothing is tuned and no cell is preferred.

    REORDERPOINT_TRACK=b TRACK_B_PANEL=primary     python -m reorderpoint.track_b_posthoc
    REORDERPOINT_TRACK=b TRACK_B_PANEL=robustness  python -m reorderpoint.track_b_posthoc
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from reorderpoint import backtest as bt
from reorderpoint import decision as dec
from reorderpoint import divergence as dv
from reorderpoint import optimal_target as opt
from reorderpoint import safety_stock as ss
from reorderpoint import track_b as tb
from reorderpoint.config import load_config
from reorderpoint.grain import GRAIN

WINDOWS = ("single", "first_window", "all_windows")
POOLINGS = ("intermittency", "volume_quintile")
VARIANTS = [(w, p) for w in WINDOWS for p in POOLINGS]
REGISTERED = ("first_window", "intermittency")


def variant_key(window: str, pooling: str) -> str:
    return f"{window}|{pooling}"


def _complete_blocks(preds: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Per (series, block): summed p50 and realised y over each complete lead-time window of a
    fold's horizon; a trailing partial block is not a lead-time total and is dropped."""
    fc = preds.sort_values(["series_id", "date"])
    block = fc.groupby("series_id").cumcount() // lead
    fc = fc.assign(block=block)
    size = fc.groupby(["series_id", "block"])["p50"].transform("size")
    fc = fc[size == lead]
    return fc.groupby(["series_id", "block"]).agg(p50=("p50", "sum"), y=("y", "sum"))


def registered_buffers(calib_model: pd.DataFrame, target: float, pooling: str) -> pd.Series:
    """The buffer the registered design gives each series: z x the pooled sigma of the fold-1
    calibration residuals, pooled by `pooling`. (`n_boot` only feeds a CI the point estimate
    ignores.)
    """
    table = calib_model.set_index("series_id")
    prior, _ = ss.per_series_safety_stock(
        table["residual"],
        table["zero_rate"],
        table["volume"],
        target,
        form="normal",
        granularity=pooling,
        n_boot=2,
    )
    return prior


def _pooled_buffer(residual_by_obs: pd.Series, train_prev, target: float, pooling: str):
    """z x the bucket sigma over `residual_by_obs` (indexed by series id, possibly several
    observations per series), bucket labels from `train_prev`'s zero rates and volumes."""
    zero_rates, volumes = ss.train_stats(train_prev)
    buckets = ss.assign_buckets(zero_rates, volumes, pooling)
    labels = buckets.reindex(residual_by_obs.index).to_numpy()
    sigma = residual_by_obs.reset_index(drop=True).groupby(labels).std(ddof=1)
    pooled = float(residual_by_obs.std(ddof=1))
    return (norm.ppf(target) * buckets.map(sigma).fillna(pooled)).rename("safety_stock")


def first_window_buffer(preds_prev, test_prev, train_prev, lead, target, pooling) -> pd.Series:
    """The registered simulation's buffer for the next fold: the previous fold's own residuals over
    its first lead-time window, one per series (`decision.lead_time_forecast_residuals`), pooled
    with
    the same function `run_cell` uses."""
    residuals = dec.lead_time_forecast_residuals(
        preds_prev, test_prev[["series_id", "date", "y"]], lead
    )
    zero_rates, volumes = ss.train_stats(train_prev)
    prior, _ = ss.per_series_safety_stock(
        residuals, zero_rates, volumes, target, form="normal", granularity=pooling, n_boot=2
    )
    return prior


def all_windows_buffer(preds_prev, train_prev, lead, target, pooling) -> pd.Series:
    """The buffer from every complete lead-time window in the previous fold (six per series for a
    13-week fold and a 2-week lead time), each a separate observation of the same sigma."""
    obs = _complete_blocks(preds_prev, lead)
    residual = (obs["y"] - obs["p50"]).droplevel("block")
    return _pooled_buffer(residual, train_prev, target, pooling)


def buffers_for(window, pooling, model, panel, forecasts, calib, folds, lead, target):
    """{fold index: per-series buffer} for one model under one variant. Fold 1 is always the window
    before it (the fold-1 calibration residuals); `window` decides folds 2 to 4."""
    first = registered_buffers(calib[calib["model"] == model], target, pooling)
    out = {folds[0].index: first}
    for prev, cur in zip(folds[:-1], folds[1:], strict=True):
        if window == "single":
            out[cur.index] = first
            continue
        preds_prev = forecasts[(forecasts["model"] == model) & (forecasts["fold"] == prev.index)]
        train_prev = panel[panel["date"] <= prev.train_end]
        if window == "first_window":
            test_prev = panel[(panel["date"] >= prev.test_start) & (panel["date"] <= prev.test_end)]
            out[cur.index] = first_window_buffer(
                preds_prev, test_prev, train_prev, lead, target, pooling
            )
        else:
            out[cur.index] = all_windows_buffer(preds_prev, train_prev, lead, target, pooling)
    return out


def simulate(panel, forecasts, buffers, costs, lot: float, target: float):
    """`ablate_safety_stock.run_cell`'s loop with the per-(model, fold) buffers given, so the four
    variants differ in the buffer and in nothing else. Returns (slim detail, coverage rows)."""
    folds = bt.make_folds(panel)
    details, cover = [], []
    for model in tb.LADDER:
        for fold in folds:
            train = panel[panel["date"] <= fold.train_end]
            test = panel[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)]
            preds = forecasts[(forecasts["fold"] == fold.index) & (forecasts["model"] == model)]
            buffer = buffers[model][fold.index]
            stats = dec.lead_time_demand_stats(preds, costs.lead_time_days)
            decisions = dec.calibrated_reorder_decision(stats, buffer, target)
            unit_costs = dec.unit_cost_from_price(train)
            demand = {
                sid: g.sort_values("date")["y"].to_numpy()
                for sid, g in test.groupby("series_id", sort=False)
            }
            _, detail = dec._simulate_policy(
                decisions,
                demand,
                unit_costs,
                float(unit_costs.mean()),
                costs,
                fold,
                model,
                policy="csl",
                lot_multiple=lot,
                start_fraction=0.5,
            )
            details.append(detail.assign(fold=fold.index, model=model))
            if lot == 2.0:
                obs = _complete_blocks(preds, costs.lead_time_days)
                b = buffer.reindex(obs.index.get_level_values("series_id")).to_numpy()
                cover.append(
                    {
                        "model": model,
                        "fold": fold.index,
                        "covered": float((obs["y"].to_numpy() <= obs["p50"].to_numpy() + b).sum()),
                        "n": len(obs),
                    }
                )
    return pd.concat(details, ignore_index=True)[tb.SLIM], pd.DataFrame(cover)


def finding_2(panel, d2, cover, lead) -> dict:
    cluster = cover[cover["model"].isin(tb.CLUSTER)]
    by_model = cluster.groupby("model")[["covered", "n"]].sum()
    per_model = (by_model["covered"] / by_model["n"]).to_dict()
    cl = d2[d2["model"].isin(tb.CLUSTER)]
    csl = 1 - float(cl["stockout_cycles"].sum()) / float(cl["cycles"].sum())
    decomposition = opt.stockout_decomposition(cl, panel, None, lead)
    mean_cover = float(np.mean(list(per_model.values())))
    within = all(abs(c - tb.TARGET) <= tb.COVERAGE_TOLERANCE for c in per_model.values())
    gap = mean_cover - csl
    share = decomposition["undershoot_share"]
    return {
        "coverage_by_model": {k: float(v) for k, v in per_model.items()},
        "mean_coverage": mean_cover,
        "all_within_tolerance": within,
        "csl_lot2": csl,
        "gap": gap,
        "undershoot_share": share,
        "replicated": bool(within and gap >= tb.CSL_GAP and share >= tb.UNDERSHOOT_MIN),
    }


def run_variant(panel, forecasts, calib, costs, window: str, pooling: str) -> dict:
    folds = bt.make_folds(panel)
    lead, target = costs.lead_time_days, tb.TARGET
    buffers = {
        m: buffers_for(window, pooling, m, panel, forecasts, calib, folds, lead, target)
        for m in tb.LADDER
    }
    d0, _ = simulate(panel, forecasts, buffers, costs, 0.0, target)
    d2, cover = simulate(panel, forecasts, buffers, costs, 2.0, target)
    s0, s2 = tb.summarise(d0), tb.summarise(d2)
    f1 = tb.finding_1(d0, d2, s2)
    f3, f4 = tb.findings_3_4(d2)
    return {
        "lot0": s0,
        "lot2": s2,
        "finding_1": f1,
        "finding_2": finding_2(panel, d2, cover, lead),
        "finding_3": f3,
        "finding_4": f4,
    }


def check_against_registered(result: dict, registered: dict) -> None:
    """The re-implementation of the registered design must equal the registered results."""
    cell = registered["cells"][tb.cell_key(tb.DEFAULT_MARGIN, tb.DEFAULT_LEAD)]
    for lot in ("lot0", "lot2"):
        for model, want in cell[lot].items():
            got = result[lot][model]
            for field, value in want.items():
                if not np.isclose(got[field], value, rtol=1e-9, atol=1e-9):
                    raise AssertionError(
                        f"re-implementation disagrees with the registered run: "
                        f"{lot} {model} {field}: {got[field]} against {value}"
                    )


def run() -> dict:
    base = load_config().costs
    costs = replace(base, gross_margin=tb.DEFAULT_MARGIN, lead_time_days=tb.DEFAULT_LEAD)
    panel = bt.load_eval_panel()
    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = pd.read_parquet(tb.RESULTS_DIR / f"calibration_lead{tb.DEFAULT_LEAD}.parquet")
    registered = json.loads((tb.RESULTS_DIR / "results.json").read_text())

    out = {"panel": bt.PANEL_PATH.parent.name, "variants": {}}
    for window, pooling in VARIANTS:
        tb._log(f"post hoc: {window}, {pooling}")
        result = run_variant(panel, forecasts, calib, costs, window, pooling)
        if (window, pooling) == REGISTERED:
            check_against_registered(result, registered)
            tb._log("  reproduces the registered cost table exactly")
        out["variants"][variant_key(window, pooling)] = result
    path = tb.RESULTS_DIR / "post_hoc.json"
    path.write_text(
        json.dumps(tb._nan_to_none(out), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    tb._log(f"wrote {path}")
    return out


def main() -> None:
    if GRAIN.name != "week":
        raise SystemExit("run with REORDERPOINT_TRACK=b")
    run()


if __name__ == "__main__":
    main()
