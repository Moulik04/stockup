"""v1.1 close-out — what the production model is for, and whether it is still the right one.

(Updated for v1.2: the served model is now `TrailingMeanModel`, the ladder's MovingAverage rung —
sections 4 and 5 below measure the calibrated interval `/forecast` now shows and confirm the served
implementation forecasts what the ladder's MovingAverage did. Sections 1-3 are the evidence for the
switch and are unchanged.)

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
4. **Does the displayed interval match the decision?** The calibrated band `/forecast` shows
   (`SafetyStockCalibration.calibrated_interval`) against the model's own P10/P90, on held-out days
   and on held-out lead-time totals, plus how often the reorder point itself covers demand.
5. **Is the served model the evaluated one?** `TrailingMeanModel` against the ladder's
   MovingAverage, forecast for forecast.

Reuses the forecast and calibration caches (`make divergence`, `make croston`), so it costs
simulations and one timing pass, not a backtest.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import croston as cr
from reorderpoint import divergence as dv
from reorderpoint import safety_stock as ss
from reorderpoint.calibration import INTERVAL_LEVEL, SafetyStockCalibration
from reorderpoint.config import CostParams, daily_to_annual, load_config
from reorderpoint.grain import GRAIN
from reorderpoint.models.trailing_mean import TrailingMeanModel

PRODUCTION = "LightGBM"  # the model this report's comparisons are made against
SERVED = "TrailingMean (served)"  # what `train.py` now fits: the ladder's MovingAverage, no deps
INTERVAL_MODELS = ("LightGBM", "MovingAverage")
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
    horizon = GRAIN.periods_in(fold.test_start, fold.test_end)
    factories = {**bt.MODEL_FACTORIES, SERVED: lambda _horizon: TrailingMeanModel()}
    rows = []
    for name in models:
        model = factories[name](horizon)
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


# ---- 4. does the displayed interval match the decision? ----------------------------------------


def _cached_calibration(
    cache: pd.DataFrame, model: str, costs: CostParams
) -> SafetyStockCalibration:
    """The production calibration object, built from the harness's fold-1 residual cache, so the
    numbers below come from the code path `/forecast` and `/reorder` actually run."""
    c = cache[cache["model"] == model].set_index("series_id")[["residual", "zero_rate", "volume"]]
    return SafetyStockCalibration(
        form=ss.LEGACY_SCHEME[0],
        granularity=ss.LEGACY_SCHEME[1],
        lead_time_days=costs.lead_time_days,
        model_name=model,
        calibrated_through=pd.NaT,
        residuals=c,
    )


def interval_evaluation(
    forecasts: pd.DataFrame, cache: pd.DataFrame, costs: CostParams, models: tuple[str, ...]
) -> pd.DataFrame:
    """Per model, out of sample over all four folds (the calibration is fold 1's residual window):
    daily coverage and mean width of the model's own P10/P90 vs the calibrated band; coverage of
    the calibrated `INTERVAL_LEVEL` interval on lead-time totals (split into misses below and
    above); and the share of lead-time windows whose realised demand the reorder point covers
    (nominally the service-level target)."""
    lead = costs.lead_time_days
    z = norm.ppf((1 + INTERVAL_LEVEL) / 2)
    rows = []
    for model in models:
        cal = _cached_calibration(cache, model, costs)
        f = forecasts[forecasts["model"] == model].sort_values(["fold", "series_id", "date"])
        banded = pd.concat(
            [cal.calibrated_interval(g) for _, g in f.groupby("fold")], ignore_index=True
        ).rename(columns={"p10": "cal_p10", "p90": "cal_p90"})
        banded = banded.assign(p10=f["p10"].to_numpy(), p90=f["p90"].to_numpy())
        native = (banded["y"] >= banded["p10"]) & (banded["y"] <= banded["p90"])
        calibrated = (banded["y"] >= banded["cal_p10"]) & (banded["y"] <= banded["cal_p90"])

        banded["block"] = banded.groupby(["fold", "series_id"]).cumcount() // lead
        # a trailing block shorter than the lead time (a 13-week fold cut into 2-week windows leaves
        # one week) is not a lead-time total and is not scored as one; on Track A's 28 days in 7-day
        # windows there is never such a block
        size = banded.groupby(["fold", "series_id", "block"])["p50"].transform("size")
        banded = banded[size == lead]
        totals = banded.groupby(["fold", "series_id", "block"]).agg(
            p50=("p50", "sum"), y=("y", "sum")
        )
        sigma = cal.lead_time_std(totals.index.get_level_values("series_id").unique())
        sig = totals.index.get_level_values("series_id").map(sigma).to_numpy()
        lo = np.clip(totals["p50"] - z * sig, 0, None)
        hi = totals["p50"] + z * sig
        buffer = cal.safety_stock(
            costs.service_level_target, series_ids=totals.index.get_level_values("series_id")
        ).to_numpy()
        rows.append(
            {
                "model": model,
                "native_daily_coverage": native.mean(),
                "calibrated_daily_coverage": calibrated.mean(),
                "native_daily_width": (banded["p90"] - banded["p10"]).mean(),
                "calibrated_daily_width": (banded["cal_p90"] - banded["cal_p10"]).mean(),
                "lt_coverage": ((totals["y"] >= lo) & (totals["y"] <= hi)).mean(),
                "lt_below": (totals["y"] < lo).mean(),
                "lt_above": (totals["y"] > hi).mean(),
                "reorder_point_covers": (totals["y"] <= totals["p50"] + buffer).mean(),
                "n_windows": len(totals),
            }
        )
    return pd.DataFrame(rows)


# ---- 5. is the served model the evaluated one? ---------------------------------------------------


def equivalence_to_ladder(eval_panel: pd.DataFrame) -> dict:
    """`TrailingMeanModel` vs the ladder's MovingAverage on every fold's training window: do they
    forecast the same (series, date) rows, and how far apart are the P50s?"""
    worst, rows_ok = 0.0, True
    for fold in bt.make_folds(eval_panel):
        train = eval_panel[eval_panel["date"] <= fold.train_end]
        horizon = GRAIN.periods_in(fold.test_start, fold.test_end)
        ref = bt.MODEL_FACTORIES["MovingAverage"](horizon)
        ref.fit(train)
        served = TrailingMeanModel()
        served.fit(train)
        a = ref.predict_quantiles(horizon)
        b = served.predict_quantiles(horizon)
        merged = a.merge(
            b, on=["series_id", "date"], how="outer", suffixes=("_ref", "_new"), indicator=True
        )
        rows_ok &= bool((merged["_merge"] == "both").all())
        worst = max(worst, float((merged["p50_ref"] - merged["p50_new"]).abs().max()))
    return {"same_rows": rows_ok, "max_abs_p50_diff": worst, "n_folds": bt.N_FOLDS}


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
    intervals: pd.DataFrame,
    equivalence: dict,
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
            coverage_80=timing["model"].map(
                lambda m: f"{coverage[m]:.1%}" if m in coverage else "n/a (no native band)"
            ),
        )
    )
    rp = intervals["reorder_point_covers"]
    native = intervals["native_daily_coverage"]
    rp_range = f"{rp.min():.1%}–{rp.max():.1%}"
    native_range = f"{native.min():.0%}–{native.max():.0%}"
    lines = [
        f"# Production model — what LightGBM was for, what it costs, and what replaced it — "
        f"{dt.date.today()}",
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
        "showed through v1.1, before they switched to the calibrated band of section 4.",
        "",
        timing_table,
        "",
        "Not measured here, read from the code: LightGBM needs `features.build_features` (lags, "
        "rolling stats, calendar, price, event flags) and a forward exog proxy at serve time "
        "(`exog.py`, repeat-the-recent-pattern), and persists a ~110 MB artifact; the statistical "
        "and Croston models need only each series' own demand history and persist nothing. The "
        f"`{SERVED}` row is the implementation `train.py` now fits: the same forecast as "
        "MovingAverage with no statsforecast call. Timing note: it is measured here on the "
        "400-series fold like every other row; on the full 5,650-series panel it takes about two "
        "seconds.",
        "",
        "## 4. Does the interval `/forecast` shows match the decision?",
        "",
        f"`/forecast` and the dashboard's fan chart now show "
        f"`P50 ± z({INTERVAL_LEVEL:.0%} central) × σ ÷ √{costs.lead_time_days}` per day, where "
        f"σ is the same pooled residual std the reorder point's buffer is "
        f"`z({costs.service_level_target:.0%}) × σ` of "
        "(`SafetyStockCalibration.calibrated_interval`). Evaluated out of sample on all four "
        "folds with the calibration held to fold 1's residual window, through the production "
        "code path:",
        "",
        cr._md(
            pd.DataFrame(
                {
                    "model": intervals["model"],
                    "native daily coverage": intervals["native_daily_coverage"].map(
                        "{:.1%}".format
                    ),
                    "calibrated daily coverage": intervals["calibrated_daily_coverage"].map(
                        "{:.1%}".format
                    ),
                    "native / calibrated daily width": [
                        f"{a:.2f} / {b:.2f}"
                        for a, b in zip(
                            intervals["native_daily_width"],
                            intervals["calibrated_daily_width"],
                            strict=True,
                        )
                    ],
                    "lead-time total coverage": intervals["lt_coverage"].map("{:.1%}".format),
                    "missed below / above": [
                        f"{lo:.1%} / {hi:.1%}"
                        for lo, hi in zip(intervals["lt_below"], intervals["lt_above"], strict=True)
                    ],
                    f"reorder point covers ({costs.service_level_target:.0%} target)": intervals[
                        "reorder_point_covers"
                    ].map("{:.1%}".format),
                }
            )
        ),
        "",
        f"Nominal coverage is {INTERVAL_LEVEL:.0%}. Read it three ways. **The reorder point does "
        f"what it says**: realised lead-time demand stayed at or under it in {rp_range} of "
        f"{int(intervals['n_windows'].iloc[0]):,} held-out windows against a "
        f"{costs.service_level_target:.0%} target (in-stock probability, not the volume-weighted "
        "fill rate the cost tables report). **The band over-covers, which is the safe direction "
        "to be wrong in**: the lower edge is clipped at zero and most demand is zero, so any "
        "band as wide as the reorder point's spread contains nearly every zero day; the "
        "independent-days assumption in `σ ÷ √L` may add width too, which this table cannot "
        "separate. **The model's own quantiles do not describe the decision**: they cover "
        f"{native_range} of days depending on the model, and LightGBM's near-nominal figure "
        "comes from alphas tuned to hit that coverage (`DECISIONS.md`, 2026-09-03), not from "
        "anything about the buffer stocked. The calibrated band depends on the model only through "
        "its residual spread, so what is shown follows what is decided.",
        "",
        "## 5. Is the served model the evaluated one?",
        "",
        f"`TrailingMeanModel` against the ladder's MovingAverage on all {equivalence['n_folds']} "
        f"folds' training windows: "
        f"{'same (series, date) rows' if equivalence['same_rows'] else '**different rows**'}, "
        f"largest P50 difference {equivalence['max_abs_p50_diff']:.1e} (statsforecast computes in "
        "float32). The cost comparison in section 2 therefore applies to the served model as "
        "it stands.",
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

    timing = time_models(panel, [PRODUCTION, *CANDIDATES, SERVED])
    coverage = interval_coverage(forecasts, [PRODUCTION, *CANDIDATES])
    intervals = interval_evaluation(forecasts, calib, costs, INTERVAL_MODELS)
    equivalence = equivalence_to_ladder(panel)
    print(intervals.round(3).to_string(), equivalence, flush=True)
    report = build_report(
        audit, ci, timing, coverage, costs, panel["series_id"].nunique(), intervals, equivalence
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"production_model_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
