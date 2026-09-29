"""v1.1 close-out — what the production model is for, and whether it is still the right one.

The README justified serving LightGBM "for its calibrated intervals". Since the decision layer was
repaired, sizing runs on pooled empirical residuals (`safety_stock.py`, `calibration.py`) and the
order-up-to level on the P50 lead-time mean, so the question is whether the model's own P10/P90
reach the reorder decision at all. This module answers three things, at exactly the shipped policy
(`S = s + lot_multiple x E[lead-time demand]`, 95% target, legacy safety-stock scheme):

1. **Do the native quantiles feed sizing?** The shipped cell is run twice, once on the real
   forecasts and once with every model's P10/P90 replaced by arbitrary constants (P50 untouched).
   If reorder points, order-up-to levels and cost are identical, the quantiles are not an input.
2. **Is LightGBM cheaper than the models it is competing with?** Paired bootstrap on total cost
   for every RMSSE-cluster member and the Croston family against LightGBM, pooled and by
   intermittency bucket, plus SeasonalNaive as the reference that is genuinely different.
3. **What does each model cost to run?** Fit + predict time on the backtest's 400-series last
   fold, the persisted artifact, and whether it needs the feature/exog pipeline.

Reuses the forecast and calibration caches (`make divergence`, `make croston`), so it costs
simulations and one timing pass, not a backtest.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import croston as cr
from reorderpoint import divergence as dv
from reorderpoint.config import CostParams, daily_to_annual, load_config

PRODUCTION = "LightGBM"
CLUSTER = ("MovingAverage", "AutoETS", "AutoTheta")
CANDIDATES = (*CLUSTER, *cr.NEW_MODELS)
REFERENCE = "SeasonalNaive"
# Arbitrary, deliberately implausible bounds: if sizing read them, costs would move by orders of
# magnitude rather than by rounding.
SCRAMBLED_P10 = 0.0
SCRAMBLED_P90 = 1_000.0
REPORTS_DIR = bt.REPORTS_DIR
# Columns of `_simulate_policy`'s detail that the reorder decision and its cost are made of, versus
# the one that only records the (quantile-derived) spread for diagnostics.
DECISION_COLUMNS = ("reorder_point", "order_up_to", "total_cost")
SPREAD_COLUMN = "lead_time_std"


# ---- 1. do the native quantiles reach the decision? -------------------------------------------


def scramble_quantiles(forecasts: pd.DataFrame) -> pd.DataFrame:
    out = forecasts.copy()
    out["p10"] = SCRAMBLED_P10
    out["p90"] = SCRAMBLED_P90
    return out


def quantile_independence(
    panel: pd.DataFrame, forecasts: pd.DataFrame, calib: pd.DataFrame, costs: CostParams
) -> dict:
    """Max absolute change in each detail column when every model's P10/P90 is scrambled."""
    keys = ["model", "fold", "series_id"]
    real = cr.shipped_cell(panel, forecasts, calib, costs).set_index(keys).sort_index()
    scrambled = (
        cr.shipped_cell(panel, scramble_quantiles(forecasts), calib, costs)
        .set_index(keys)
        .sort_index()
    )
    assert real.index.equals(scrambled.index), "scrambling P10/P90 changed which series simulate"
    return {
        "n_rows": len(real),
        "max_abs_change": {
            col: float((real[col] - scrambled[col]).abs().max())
            for col in (*DECISION_COLUMNS, SPREAD_COLUMN)
        },
    }


# ---- 2. paired bootstrap against the production model -----------------------------------------


def cost_vs_production(detail: pd.DataFrame, zero_rates: pd.Series) -> pd.DataFrame:
    """Each candidate minus LightGBM on total cost per fold, paired over series, with the
    bootstrap's own crosses-zero verdict — pooled and per intermittency bucket."""
    detail = detail.assign(bucket=dv.bucket_of(detail["series_id"].map(zero_rates)).to_numpy())
    rows = []
    for scope in ("all", "intermittent", "regular"):
        sub = detail if scope == "all" else detail[detail["bucket"] == scope]
        base = sub[sub["model"] == PRODUCTION]
        for model in (*CANDIDATES, REFERENCE):
            ci = boot.bootstrap_diff(
                sub[sub["model"] == model], base, "total_cost", group_cols=["fold"]
            )
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "cost_per_fold": sub[sub["model"] == model]["total_cost"].sum()
                    / sub["fold"].nunique(),
                    "diff_vs_lightgbm": ci["point"],
                    "ci_lo": ci["ci_lo"],
                    "ci_hi": ci["ci_hi"],
                    "reading": _reading(model, ci),
                }
            )
    return pd.DataFrame(rows)


def _reading(model: str, ci: dict) -> str:
    if ci["crosses_zero"]:
        return "tied"
    return f"{model} cheaper" if ci["point"] < 0 else f"{PRODUCTION} cheaper"


# ---- 3. what each model costs to run ----------------------------------------------------------


def time_models(eval_panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Fit + predict wall time on the last fold's training window, the way the backtest runs it.

    Reported as one number: statsforecast's `fit` only stores the data and the per-series fitting
    happens inside `predict_quantiles`, so timing them apart would read as "free to fit"."""
    fold = bt.make_folds(eval_panel)[-1]
    train = eval_panel[eval_panel["date"] <= fold.train_end]
    test = eval_panel[
        (eval_panel["date"] >= fold.test_start) & (eval_panel["date"] <= fold.test_end)
    ]
    horizon = (fold.test_end - fold.test_start).days + 1
    rows = []
    for name in models:
        model = bt.MODEL_FACTORIES[name](horizon)
        t0 = time.perf_counter()
        model.fit(train)
        model.predict_quantiles(horizon, future_exog=test.drop(columns=["y"]))
        t2 = time.perf_counter()
        rows.append({"model": name, "fit_predict_s": t2 - t0})
        print(f"timed {name}", flush=True)
    return pd.DataFrame(rows)


def interval_coverage(forecasts: pd.DataFrame, models: list[str]) -> pd.Series:
    """Share of test days whose actual demand falls inside the model's P10–P90 band (nominal 80%)
    — the one thing the model's own quantiles are still served for (`/forecast`, the dashboard's
    fan chart)."""
    f = forecasts[forecasts["model"].isin(models)]
    return ((f["y"] >= f["p10"]) & (f["y"] <= f["p90"])).groupby(f["model"]).mean()


# ---- report -----------------------------------------------------------------------------------


def _money(v: float) -> str:
    return f"${v:+,.0f}"


def build_report(
    audit: dict,
    ci: pd.DataFrame,
    timing: pd.DataFrame,
    coverage: pd.Series,
    costs: CostParams,
    n_series: int,
) -> str:
    changes = audit["max_abs_change"]
    independent = all(changes[c] == 0.0 for c in DECISION_COLUMNS)

    def ci_table(scope: str) -> str:
        sub = ci[ci["scope"] == scope]
        display = pd.DataFrame(
            {
                "model": sub["model"],
                "cost / fold": sub["cost_per_fold"].map(lambda v: f"${v:,.0f}"),
                "minus LightGBM": sub["diff_vs_lightgbm"].map(_money),
                "95% CI": [
                    f"[{_money(lo)}, {_money(hi)}]"
                    for lo, hi in zip(sub["ci_lo"], sub["ci_hi"], strict=True)
                ],
                "reading": sub["reading"],
            }
        )
        return cr._md(display)

    timing_table = cr._md(
        timing.assign(
            fit_predict_s=timing["fit_predict_s"].map(lambda v: f"{v:.1f}"),
            coverage_80=timing["model"].map(lambda m: f"{coverage[m]:.1%}"),
        )
    )
    lines = [
        f"# Production model — what LightGBM is for, and what it costs — {dt.date.today()}",
        "",
        f"Shipped policy throughout: `S = s + {costs.lot_multiple:g} x lead-time demand`, "
        f"{costs.service_level_target:.0%} target, legacy (normal, intermittency) safety-stock "
        f"scheme, {daily_to_annual(costs.holding_cost_rate):.0%}/yr holding, "
        f"{costs.gross_margin:.1%} lost margin. "
        f"Same {n_series}-series sample, {bt.N_FOLDS} folds, and forecast/calibration caches as "
        "`reports/croston_*.md`.",
        "",
        "## 1. Do the models' own P10/P90 feed the reorder decision?",
        "",
        f"The shipped cell was simulated twice over {audit['n_rows']:,} model×fold×series rows: "
        f"once on the real forecasts, once with every model's P10 set to {SCRAMBLED_P10:g} and "
        f"P90 to {SCRAMBLED_P90:,.0f} (P50 untouched). Largest absolute change per column:",
        "",
        cr._md(
            pd.DataFrame(
                [
                    {
                        "column": c,
                        "max |real − scrambled|": f"{changes[c]:.6g}",
                        "role": (
                            "decision" if c in DECISION_COLUMNS else "diagnostic (spread column)"
                        ),
                    }
                    for c in (*DECISION_COLUMNS, SPREAD_COLUMN)
                ]
            )
        ),
        "",
        (
            "**No. Reorder points, order-up-to levels and cost are unchanged to the last bit; the "
            "only column that moves is the quantile-derived spread, which is recorded but not "
            "read by sizing.** Sizing is P50 lead-time demand + a buffer from pooled empirical "
            "residuals; the order-up-to level is P50-derived too. The quantiles are read only as "
            "a fallback for a series the calibration has no buffer for, which does not occur on "
            "this panel."
            if independent
            else "**Yes — a decision column moved, so the native quantiles do reach the decision. "
            "Do not read the rest of this report as a P50-only comparison.**"
        ),
        "",
        "## 2. Cost against LightGBM, paired bootstrap (95% CI, series resampled)",
        "",
        "Negative = cheaper than LightGBM. Six candidates plus SeasonalNaive as the reference "
        "that genuinely differs. Across six candidates, at least one CI missing zero by chance "
        "alone is roughly a one-in-four event, so a lone marginal result would not be "
        "evidence.",
        "",
        "### All series",
        "",
        ci_table("all"),
        "",
        "### Intermittent series (>50% zero training days)",
        "",
        ci_table("intermittent"),
        "",
        "### Regular series",
        "",
        ci_table("regular"),
        "",
        "## 3. What each model costs to run",
        "",
        f"Fit + predict wall time on the last fold's training window ({n_series} series, one "
        "machine, all cores). The statistical models fit per series, so their time scales roughly "
        "linearly in series count; LightGBM is one global fit. `coverage_80` is the share of test "
        "days inside the model's own P10–P90 band — what `/forecast` and the dashboard fan chart "
        "show, and the only place the models' quantiles still reach a user.",
        "",
        timing_table,
        "",
        "Not measured here, read from the code: LightGBM needs `features.build_features` (lags, "
        "rolling stats, calendar, price, event flags) and a forward exog proxy at serve time "
        "(`exog.py`, repeat-the-recent-pattern), and persists a ~110 MB artifact; the statistical "
        "and Croston models need only each series' own demand history and persist nothing.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    costs = load_config().costs
    panel = bt.load_eval_panel()
    forecasts = cr.extend_forecasts(panel)
    calib = cr.extend_calibration(panel, costs)

    audit = quantile_independence(panel, forecasts, calib, costs)
    print(audit, flush=True)

    detail = cr.shipped_cell(panel, forecasts, calib, costs)
    ci = cost_vs_production(detail, bt._intermittency(panel))

    timing = time_models(panel, [PRODUCTION, *CANDIDATES])
    coverage = interval_coverage(forecasts, [PRODUCTION, *CANDIDATES])
    report = build_report(audit, ci, timing, coverage, costs, panel["series_id"].nunique())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"production_model_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
