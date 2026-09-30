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
from pathlib import Path
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


def _boot_ci(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = boot.bootstrap_diff(a, b, "total_cost", group_cols=["fold"])
    return {"point": out["point"], "lo": out["ci_lo"], "hi": out["ci_hi"]}


def finding_1(d0: pd.DataFrame, d2: pd.DataFrame, s2: dict) -> dict:
    """Policy over model: every one of the eight models saves money going from S = s to
    S = s + 2 x lead-time demand, with a CI excluding zero, and the mean saving exceeds the spread
    between the cheapest and dearest cluster model under the shipped policy."""
    by_model = {}
    for m in LADDER:
        by_model[m] = _boot_ci(d2[d2["model"] == m], d0[d0["model"] == m])  # lot 2 minus lot 0
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
        pairs[f"{a} - {b}"] = _boot_ci(by[a], by[b])
    excluding = {k: v for k, v in pairs.items() if not (v["lo"] <= 0 <= v["hi"])}
    vs_lgb = {m: _boot_ci(by[m], by["LightGBM"]) for m in CLUSTER if m != "LightGBM"}
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


# ---- the report ---------------------------------------------------------------------------------

FINDINGS = {
    1: "The ordering policy matters more than the model",
    2: "The calibrated quantile does what it says; the shortfall is the policy",
    3: "The model cluster is a tie on cost",
    4: "The trailing mean is enough",
}


def default_cell(res: dict) -> dict:
    return res["cells"][cell_key(DEFAULT_MARGIN, DEFAULT_LEAD)]


def verdicts(
    res: dict, margin: float = DEFAULT_MARGIN, lead: int = DEFAULT_LEAD
) -> dict[int, bool]:
    cell = res["cells"][cell_key(margin, lead)]
    return {
        1: cell["finding_1"]["replicated"],
        2: res["finding_2"][f"L{lead}"]["replicated"],
        3: cell["finding_3"]["replicated"],
        4: cell["finding_4"]["replicated"],
    }


def holds_across_grid(res: dict, k: int) -> tuple[int, int]:
    """In how many of the 12 economics cells the verdict on finding k equals the default cell's."""
    base = verdicts(res)[k]
    same = sum(verdicts(res, m, lead)[k] == base for m in MARGINS for lead in LEADS)
    return same, len(MARGINS) * len(LEADS)


def _word(flag: bool) -> str:
    return "replicated" if flag else "not replicated"


def _money(v: float) -> str:
    return f"-${-v:,.0f}" if round(v) < 0 else f"${v:,.0f}"


def _ci(c: dict) -> str:
    return f"[{_money(c['lo'])}, {_money(c['hi'])}]"


def track_a_reference() -> dict:
    """Track A's numbers for the same four questions, from the frozen baseline fixture."""
    base = json.loads(cr_baseline_path().read_text())
    lot0, lot2 = base["original_policy_lot_0"], base["shipped_cell_lot_2"]
    saving = float(np.mean([lot0[m]["cost_per_fold"] - lot2[m]["cost_per_fold"] for m in LADDER]))
    costs = [lot2[m]["cost_per_fold"] for m in CLUSTER]
    csls = [lot2[m]["csl"] for m in CLUSTER]
    cover = [r["reorder_point_covers"] for r in base["interval_evaluation"]]
    ma_minus_lgb = lot2["MovingAverage"]["cost_per_fold"] - lot2["LightGBM"]["cost_per_fold"]
    return {
        "saving": saving,
        "spread": max(costs) - min(costs),
        "csl": (min(csls), max(csls)),
        "coverage": (min(cover), max(cover)),
        "ma_minus_lgb": ma_minus_lgb,
    }


def cr_baseline_path() -> Path:
    from reorderpoint import track_a_baseline

    return track_a_baseline.BASELINE_PATH


def _headline_rows(primary: dict, robust: dict, a: dict) -> list[dict]:
    vp, vr = verdicts(primary), verdicts(robust)
    f1, f2 = default_cell(primary)["finding_1"], primary["finding_2"][f"L{DEFAULT_LEAD}"]
    f3, f4 = default_cell(primary)["finding_3"], default_cell(primary)["finding_4"]
    share = f2["undershoot_share"]
    undershoot = "n/a" if share is None or not np.isfinite(share) else f"{share:.0%}"
    detail = {
        1: (
            f"S>s saves {_money(f1['mean_saving'])} per fold per model; cluster spread "
            f"{_money(f1['spread'])}; every CI excludes zero: {f1['all_exclude_zero']}"
        ),
        2: (
            f"coverage {f2['mean_coverage']:.1%} (within 3 pts for every cluster model: "
            f"{f2['all_within_tolerance']}); realised CSL {f2['csl_lot2']:.1%}; undershoot "
            f"{undershoot} of stockouts"
        ),
        3: f"{len(f3['excluding_zero'])} of {f3['n_pairs']} pairs exclude zero",
        4: f"MovingAverage minus LightGBM {_money(f4['moving_average_vs_lightgbm']['point'])} "
        f"{_ci(f4['moving_average_vs_lightgbm'])}",
    }
    track_a = {
        1: f"saving {_money(a['saving'])}; cluster spread {_money(a['spread'])}",
        2: f"coverage {a['coverage'][0]:.1%}-{a['coverage'][1]:.1%}; CSL "
        f"{a['csl'][0]:.1%}-{a['csl'][1]:.1%}",
        3: "all six CIs against LightGBM contain zero",
        4: f"MovingAverage minus LightGBM {_money(a['ma_minus_lgb'])} [-$34, +$51]",
    }
    rows = []
    for k, title in FINDINGS.items():
        same, n = holds_across_grid(primary, k)
        rows.append(
            {
                "finding": f"{k}. {title}",
                "Track A": track_a[k],
                "Track B primary panel": f"**{_word(vp[k])}**: {detail[k]}",
                "robustness panel": "agrees" if vp[k] == vr[k] else "**disagrees**",
                "primary verdict across the 12 economics cells": f"{same} of {n}",
            }
        )
    return rows


def _cost_table(res: dict) -> pd.DataFrame:
    cell = default_cell(res)
    rows = []
    for m in LADDER:
        c0, c2 = cell["lot0"][m], cell["lot2"][m]
        ci = cell["finding_1"]["by_model"][m]
        rows.append(
            {
                "model": m + (" *" if m == "SeasonalNaive" else ""),
                "S = s, cost/fold": _money(c0["cost"]),
                "S > s, cost/fold": _money(c2["cost"]),
                "S>s minus S=s": f"{_money(ci['point'])} {_ci(ci)}",
                "CSL (S>s)": f"{c2['csl']:.1%}",
                "fill rate (S>s)": f"{c2['fill']:.1%}",
            }
        )
    return pd.DataFrame(rows)


def _cluster_table(res: dict) -> pd.DataFrame:
    f3 = default_cell(res)["finding_3"]
    cell = default_cell(res)["lot2"]
    rows = [
        {
            "model": m,
            "cost/fold (S>s)": _money(cell[m]["cost"]),
            "minus LightGBM": f"{_money(c['point'])} {_ci(c)}",
            "reading": (
                "tied" if c["lo"] <= 0 <= c["hi"] else ("cheaper" if c["point"] < 0 else "dearer")
            ),
        }
        for m, c in f3["vs_lightgbm"].items()
    ]
    return pd.DataFrame(rows)


def _accuracy_table(res: dict, res_cells_key: str = "lot2") -> pd.DataFrame:
    acc = res["accuracy"]["by_model"]
    cost = {m: default_cell(res)[res_cells_key][m]["cost"] for m in LADDER}
    mase_rank = pd.Series({m: acc[m]["mase"] for m in LADDER}).rank()
    cost_rank = pd.Series(cost).rank()
    rows = [
        {
            "model": m,
            "MASE": f"{acc[m]['mase']:.3f}",
            "rank by MASE": int(mase_rank[m]),
            "rank by cost (S>s)": int(cost_rank[m]),
            "native 80% band coverage": f"{acc[m]['coverage_80']:.1%}",
        }
        for m in LADDER
    ]
    return pd.DataFrame(rows), float(mase_rank.corr(cost_rank))


def _grid_table(res: dict) -> pd.DataFrame:
    rows = []
    for m in MARGINS:
        for lead in LEADS:
            v = verdicts(res, m, lead)
            rows.append(
                {
                    "margin": f"{m:.1%}",
                    "lead time": f"{lead} wk",
                    **{f"finding {k}": "R" if v[k] else "-" for k in FINDINGS},
                }
            )
    return pd.DataFrame(rows)


def _finding_2_table(res: dict) -> pd.DataFrame:
    rows = []
    for lead in LEADS:
        f = res["finding_2"][f"L{lead}"]
        u = f["undershoot_share"]
        u = None if u is None or not np.isfinite(u) else u
        rows.append(
            {
                "lead time": f"{lead} wk",
                "mean coverage": f"{f['mean_coverage']:.1%}",
                "range across models": f"{min(f['coverage_by_model'].values()):.1%}-"
                f"{max(f['coverage_by_model'].values()):.1%}",
                "realised CSL (S>s)": f"{f['csl_lot2']:.1%}",
                "gap": f"{f['gap'] * 100:.1f} pts",
                "undershoot share": "n/a" if u is None else f"{u:.0%}",
                "verdict": _word(f["replicated"]),
            }
        )
    return pd.DataFrame(rows)


def stratum_tables(panel_dir_robust: Path, primary_ids: set[str]) -> pd.DataFrame | None:
    """Cost (S > s, default cell) and MASE for the SKUs only the robustness panel holds."""
    detail_path = panel_dir_robust / "results" / "detail_default_lot2.parquet"
    fc_path = panel_dir_robust / "backtest_forecasts.parquet"
    if not (detail_path.exists() and fc_path.exists()):
        return None
    panel = pd.read_parquet(panel_dir_robust / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    added = set(panel["series_id"].unique()) - primary_ids
    sub = panel[panel["series_id"].isin(added)]
    forecasts = pd.read_parquet(fc_path)
    forecasts = forecasts[forecasts["series_id"].isin(added)]
    detail = pd.read_parquet(detail_path)
    detail = detail[detail["series_id"].isin(added)]
    summary = summarise(detail)
    acc = cr.accuracy_table(sub, forecasts, list(LADDER)).groupby("model")["mase"].mean()
    return pd.DataFrame(
        [
            {
                "model": m,
                "cost/fold (S>s)": _money(summary[m]["cost"]),
                "CSL": f"{summary[m]['csl']:.1%}",
                "MASE": f"{acc[m]:.3f}",
            }
            for m in LADDER
        ]
    ), len(added)


def calibration_diagnostics(panel_dir: Path, res: dict) -> dict | None:
    """Descriptive evidence on the one calibration window, from the panel's own files: the demand
    level in that window against each test fold, every model's calibration sigma against its cost
    rank, and each model's forecast bias. Post hoc and descriptive: it changes no verdict."""
    calib_path = panel_dir / "results" / f"calibration_lead{DEFAULT_LEAD}.parquet"
    fc_path = panel_dir / "backtest_forecasts.parquet"
    if not (calib_path.exists() and fc_path.exists()):
        return None
    panel = pd.read_parquet(panel_dir / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    first = pd.Timestamp(res["folds"][0][1])  # fold 1's train end: the last week of year 1
    window = panel[(panel["date"] > first - GRAIN.period * DEFAULT_LEAD) & (panel["date"] <= first)]
    calib = pd.read_parquet(calib_path)
    sigma = calib.groupby("model")["residual"].std()
    forecasts = pd.read_parquet(fc_path)
    bias = forecasts.groupby("model").apply(lambda g: g["p50"].mean() / g["y"].mean() - 1)
    cost = {m: default_cell(res)["lot2"][m]["cost"] for m in LADDER}
    rho = float(pd.Series(sigma).rank().corr(pd.Series(cost).rank()))
    folds = []
    for f in res["folds"]:
        test = panel[(panel["date"] >= f[2]) & (panel["date"] <= f[3])]
        folds.append((f[0], float(test["y"].mean())))
    return {
        "window_mean": float(window["y"].mean()),
        "window": f"{window['date'].min().date()} to {window['date'].max().date()}",
        "folds": folds,
        "sigma": {m: float(v) for m, v in sigma.items()},
        "bias": {m: float(v) for m, v in bias.items()},
        "sigma_cost_rank_corr": rho,
    }


def _diagnostic_lines(name: str, d: dict | None) -> list[str]:
    if d is None:
        return []
    rows = [
        {
            "model": m,
            "calibration sigma (units per lead time)": f"{d['sigma'][m]:.0f}",
            "forecast bias (mean p50 / mean actual - 1)": f"{d['bias'][m]:+.0%}",
        }
        for m in LADDER
    ]
    folds = ", ".join(f"fold {i}: {v:.1f}" for i, v in d["folds"])
    return [
        f"{name}: the calibration window ({d['window']}) averaged {d['window_mean']:.1f} units per "
        f"SKU-week; the test folds averaged {folds}. Rank correlation between a model's "
        f"calibration sigma and its cost rank: {d['sigma_cost_rank_corr']:.2f}.",
        "",
        cr._md(pd.DataFrame(rows)),
        "",
    ]


def render_report(
    primary: dict, robust: dict, stratum, today: str, diagnostics: tuple | None = None
) -> str:
    a = track_a_reference()
    cost_p, cost_r = _cost_table(primary), _cost_table(robust)
    acc_p, corr_p = _accuracy_table(primary)
    acc_r, corr_r = _accuracy_table(robust)
    vp, vr = verdicts(primary), verdicts(robust)
    n_rep = sum(vp.values())
    disagree = [k for k in FINDINGS if vp[k] != vr[k]]
    lines = [
        f"# Track B: do the findings replicate at a small UK online gift-ware seller? {today}",
        "",
        "UCI Online Retail II (Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning "
        "Repository. https://doi.org/10.24432/C5CG6D, CC BY 4.0), cleaned to weekly SKU series "
        "(`docs/data.md`). The design, the economics and the rule for each verdict were registered "
        "before any model ran (`docs/track_b.md`, public commits 2026-09-29); nothing below is "
        "re-read after the fact. **Verdicts are the primary panel's** "
        f"({primary['n_series']:,} SKUs sold in at least 26 of the first 52 weeks); the robustness "
        f"panel ({robust['n_series']:,} SKUs, at least 13) is reported as agrees or disagrees and "
        "never used to pick the more favourable result.",
        "",
        "## Which findings replicated",
        "",
        f"**{n_rep} of 4 replicated on the primary panel.** "
        + (
            "The robustness panel agrees on all four."
            if not disagree
            else "The robustness panel disagrees on finding"
            + ("s " if len(disagree) > 1 else " ")
            + ", ".join(str(k) for k in disagree)
            + "."
        ),
        "",
        cr._md(pd.DataFrame(_headline_rows(primary, robust, a))),
        "",
        "Default cell: margin 27.5%, lead time 2 weeks, 95% target. Track A's column is "
        "from the frozen "
        "baseline, not restated. Read with the coverage limits in `docs/track_b.md`: "
        "both panels are "
        "products with at least a year of history, both decline over the test period, "
        "and the primary "
        "panel tests mature products with muted seasonality.",
        "",
        "## Finding 1 — policy against model",
        "",
        f"Primary panel, cost per fold (USD, {primary['n_series']:,} series). `*` "
        f"SeasonalNaive is in "
        "the ladder but not the cluster.",
        "",
        cr._md(cost_p),
        "",
        f"Robustness panel ({robust['n_series']:,} series):",
        "",
        cr._md(cost_r),
        "",
        "## Finding 2 — calibrated quantile against undershoot",
        "",
        "Held-out coverage of the reorder point on fixed-origin lead-time windows inside the test "
        "folds, against realised cycle service level under `S > s`, primary panel:",
        "",
        cr._md(_finding_2_table(primary)),
        "",
        "Robustness panel:",
        "",
        cr._md(_finding_2_table(robust)),
        "",
        "## Findings 3 and 4 — the cluster and the trailing mean",
        "",
        "Each cluster model against LightGBM under `S > s` (paired bootstrap, series resampled), "
        "primary panel:",
        "",
        cr._md(_cluster_table(primary)),
        "",
        f"Of the {default_cell(primary)['finding_3']['n_pairs']} pairs among the seven "
        f"cluster models, "
        f"{len(default_cell(primary)['finding_3']['excluding_zero'])} have a CI that excludes zero"
        + (
            ": "
            + "; ".join(
                f"{k} {_money(v['point'])} {_ci(v)}"
                for k, v in default_cell(primary)["finding_3"]["excluding_zero"].items()
            )
            if default_cell(primary)["finding_3"]["excluding_zero"]
            else ""
        )
        + f". With {default_cell(primary)['finding_3']['n_pairs']} comparisons about one "
        f"exclusion is "
        "expected by chance alone even among models that are truly tied.",
        "",
        "Robustness panel:",
        "",
        cr._md(_cluster_table(robust)),
        "",
        "## Accuracy against cost",
        "",
        f"Primary panel. Rank correlation between the MASE ranking and the cost ranking: "
        f"{corr_p:.2f}.",
        "",
        cr._md(acc_p),
        "",
        f"Robustness panel: rank correlation {corr_r:.2f}.",
        "",
        cr._md(acc_r),
        "",
        "## Sensitivity to the assumed economics",
        "",
        "`R` marks a replicated verdict in that cell of the margin x lead-time grid "
        "(finding 2 depends "
        "on the lead time only). Primary panel:",
        "",
        cr._md(_grid_table(primary)),
        "",
        "Robustness panel:",
        "",
        cr._md(_grid_table(robust)),
        "",
    ]
    if stratum is not None:
        table, n_added = stratum
        lines += [
            f"## The {n_added:,} SKUs only the robustness panel holds",
            "",
            "Descriptive, no verdict: the short-season end of the catalogue (`docs/track_b.md`). "
            "Cost per fold under `S > s`, default cell, and MASE.",
            "",
            cr._md(table),
            "",
        ]
    if diagnostics is not None:
        lines += [
            "## The calibration window, descriptively",
            "",
            "Post hoc and descriptive; it changes no verdict. Every model's buffer is "
            "sized from one "
            "window, the last two weeks of the first 52, which is late November: the autumn peak. "
            "That window's demand level, each model's sigma from it, and each model's "
            "forecast bias:",
            "",
            *_diagnostic_lines("Primary panel", diagnostics[0]),
            *_diagnostic_lines("Robustness panel", diagnostics[1]),
        ]
    lines += [
        "## Limits that travel with every result above",
        "",
        "- One business, two years, one prior holiday season (2010): no model can learn an annual "
        "pattern from one cycle beyond repeating it.",
        "- Both panels are products with at least a year of history; cold start is not addressed. "
        "Both decline over the test period.",
        "- The calibration is a single lead-time window, the last weeks of November "
        "2010, applied to "
        "all four folds.",
        "- SeasonalNaive uses the 52-week season in the folds and the plain naive forecast on the "
        "50-week calibration window; AutoETS and AutoTheta are non-seasonal in every fold.",
        "- Costs, margin and lead time are assumptions; the grid above is the defence, "
        "not a proof.",
        "- `y` is units invoiced, not demand.",
        "",
    ]
    return "\n".join(lines)


def report() -> Path:
    from reorderpoint import online_retail as orr

    loaded = {}
    for name in orr.PANELS:
        path = orr.PROCESSED_DIR / name / "results" / "results.json"
        if not path.exists():
            raise SystemExit(f"{path} missing: run `track_b run` for the {name} panel first")
        loaded[name] = json.loads(path.read_text())
    primary_ids = set(
        pd.read_parquet(orr.PROCESSED_DIR / "primary" / "panel.parquet", columns=["series_id"])[
            "series_id"
        ]
    )
    stratum = stratum_tables(orr.PROCESSED_DIR / "robustness", primary_ids)
    diagnostics = tuple(
        calibration_diagnostics(orr.PROCESSED_DIR / name, loaded[name]) for name in orr.PANELS
    )
    text = render_report(
        loaded["primary"],
        loaded["robustness"],
        stratum,
        dt.date.today().isoformat(),
        diagnostics,
    )
    out = bt.REPORTS_DIR / f"track_b_online_retail_{dt.date.today().isoformat()}.md"
    out.write_text(text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["run", "report"])
    args = parser.parse_args()
    if GRAIN.name != "week":
        raise SystemExit("run with REORDERPOINT_TRACK=b")
    t0 = time.time()
    if args.command == "run":
        run()
    else:
        print(f"wrote {report()}")
    _log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
