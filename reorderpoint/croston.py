"""v1.1 Task 4 — the Croston family, evaluated the way every other rung of the ladder is.

CrostonClassic, CrostonSBA and TSB (reorderpoint/models/croston.py) are added to
`bt.MODEL_FACTORIES`, so every script that loops over the registry (divergence,
ablate_safety_stock, decision, order_up_to) already treats them like any other model. This module
does three things a fresh registry entry does not get for free:

1. **Extends the two shared caches** (`divergence.FORECASTS_PATH`, the fold-1 calibration
   residuals) with just the three new models, by fitting only them — not re-fitting the five
   existing models, which stay byte-identical to what earlier reports already measured.
2. **Accuracy**: MASE/RMSSE/coverage for the three new models, computed from those forecasts the
   same way `backtest.write_report` does, next to the five-model table already published.
3. **Decision cost under the shipped policy** — `S = s + 2 x lead-time demand`, 95% target, the
   legacy (normal, intermittency) safety-stock scheme (`config.DEFAULT_LOT_MULTIPLE`,
   `config.DEFAULT_GROSS_MARGIN`, `safety_stock.LEGACY_SCHEME`) — pooled and by intermittency
   bucket, with paired-bootstrap CIs against SeasonalNaive and LightGBM.

Running `ab.run_cell` at that one (form, granularity, lot, target) cell, rather than
`order_up_to.run_sweep`'s full grid, is deliberate: it reuses the exact same simulator and safety-
stock sizing every other model was scored with, at exactly the policy actually shipped, without
paying for (or risking a numeric drift in) the 70-cell sweep and its own held-out tests, which
this task does not touch.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import decision as dec
from reorderpoint import divergence as dv
from reorderpoint import safety_stock as ss
from reorderpoint.config import CostParams, load_config
from reorderpoint.grain import GRAIN
from reorderpoint.metrics import coverage, pinball_loss

NEW_MODELS = ("CrostonClassic", "CrostonSBA", "TSB")
LEGACY = ss.LEGACY_SCHEME  # ("normal", "intermittency") — the shipped safety-stock scheme
REPORTS_DIR = bt.REPORTS_DIR


# ---- extend the two shared caches with just the new models -----------------------------------


def extend_forecasts(eval_panel: pd.DataFrame) -> pd.DataFrame:
    """Adds CrostonClassic/CrostonSBA/TSB rows to the forecast cache, without re-fitting the five
    models already in it. Returns the full (8-model) frame; also persists it to
    `dv.FORECASTS_PATH` so every other script that reads that cache sees the new models too."""
    existing = pd.read_parquet(dv.FORECASTS_PATH) if dv.FORECASTS_PATH.exists() else None
    have = set(existing["model"].unique()) if existing is not None else set()
    missing = [m for m in NEW_MODELS if m not in have]
    if not missing:
        return existing

    folds = bt.make_folds(eval_panel)
    frames = []
    for fold in folds:
        train = eval_panel[eval_panel["date"] <= fold.train_end]
        test = eval_panel[
            (eval_panel["date"] >= fold.test_start) & (eval_panel["date"] <= fold.test_end)
        ]
        horizon = GRAIN.periods_in(fold.test_start, fold.test_end)
        zero_rates = bt._intermittency(train).rename("zero_rate")
        for model_name in missing:
            model = bt.MODEL_FACTORIES[model_name](horizon)
            model.fit(train)
            preds = model.predict_quantiles(horizon, future_exog=test.drop(columns=["y"]))
            merged = test[["series_id", "date", "y"]].merge(
                preds, on=["series_id", "date"], how="inner"
            )
            merged = merged.merge(zero_rates, left_on="series_id", right_index=True, how="left")
            merged["fold"] = fold.index
            merged["model"] = model_name
            frames.append(merged)
            print(f"forecasts: fold {fold.index} {model_name}: {len(merged)} rows", flush=True)

    new = pd.concat(frames, ignore_index=True)
    out = new if existing is None else pd.concat([existing, new], ignore_index=True)
    out.to_parquet(dv.FORECASTS_PATH, index=False)
    return out


def extend_calibration(eval_panel: pd.DataFrame, costs: CostParams) -> pd.DataFrame:
    """Same idea for the fold-1 calibration residuals `ab.calibration_residuals` caches — adds
    just the new models' residuals, computed the identical way (same calib_train/calib_test
    split), and persists the extended frame to `ab.CALIB_PATH`."""
    existing = pd.read_parquet(ab.CALIB_PATH) if ab.CALIB_PATH.exists() else None
    have = set(existing["model"].unique()) if existing is not None else set()
    missing = [m for m in NEW_MODELS if m not in have]
    if not missing:
        return existing

    first_fold = bt.make_folds(eval_panel)[0]
    train = eval_panel[eval_panel["date"] <= first_fold.train_end]
    cutoff = train["date"].max() - GRAIN.period * (costs.lead_time_days - 1)
    calib_train = train[train["date"] < cutoff]
    calib_test = train[train["date"] >= cutoff]

    zero_rates, volumes = ss.train_stats(calib_train)
    frames = []
    for model_name in missing:
        model = bt.MODEL_FACTORIES[model_name](costs.lead_time_days)
        model.fit(calib_train)
        preds = model.predict_quantiles(
            costs.lead_time_days, future_exog=calib_test.drop(columns=["y"])
        )
        residuals = dec.lead_time_forecast_residuals(
            preds, calib_test[["series_id", "date", "y"]], costs.lead_time_days
        )
        frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "series_id": residuals.index,
                    "residual": residuals.to_numpy(),
                    "zero_rate": zero_rates.reindex(residuals.index).to_numpy(),
                    "volume": volumes.reindex(residuals.index).to_numpy(),
                }
            )
        )
        print(f"calibration: {model_name}", flush=True)

    new = pd.concat(frames, ignore_index=True)
    out = new if existing is None else pd.concat([existing, new], ignore_index=True)
    out.to_parquet(ab.CALIB_PATH, index=False)
    return out


# ---- accuracy ----------------------------------------------------------------------------------


def accuracy_table(panel: pd.DataFrame, forecasts: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Mean MASE/RMSSE/coverage per model, computed from cached forecasts the same way
    `backtest.evaluate_fold`/`write_report` do — no re-fitting, just re-scoring."""
    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    folds = bt.make_folds(eval_panel)
    rows = []
    for fold in folds:
        train = eval_panel[eval_panel["date"] <= fold.train_end]
        scales = bt._in_sample_scales(train, bt.SEASON_LENGTH)
        fold_fc = forecasts[forecasts["fold"] == fold.index]
        for model_name in models:
            m = fold_fc[fold_fc["model"] == model_name].merge(scales, on="series_id", how="left")
            m["abs_err"] = (m["y"] - m["p50"]).abs()
            m["sq_err"] = (m["y"] - m["p50"]) ** 2
            per_series = m.groupby("series_id").agg(
                mae=("abs_err", "mean"),
                mse=("sq_err", "mean"),
                scale_mae=("scale_mae", "first"),
                scale_mse=("scale_mse", "first"),
            )
            per_series["mase"] = per_series["mae"] / per_series["scale_mae"]
            per_series["rmsse"] = np.sqrt(per_series["mse"] / per_series["scale_mse"])
            rows.append(
                {
                    "fold": fold.index,
                    "model": model_name,
                    "mase": per_series["mase"].mean(skipna=True),
                    "rmsse": per_series["rmsse"].mean(skipna=True),
                    "pinball_p50": pinball_loss(m["y"].to_numpy(), m["p50"].to_numpy(), 0.5),
                    "coverage_80": coverage(
                        m["y"].to_numpy(), m["p10"].to_numpy(), m["p90"].to_numpy()
                    ),
                }
            )
    return pd.DataFrame(rows)


def bucket_accuracy(
    panel: pd.DataFrame, forecasts: pd.DataFrame, models: list[str]
) -> pd.DataFrame:
    """Same as `accuracy_table`, split by the same >50%-zero intermittency bucket used everywhere
    else in this project (`dv.bucket_of`)."""
    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    folds = bt.make_folds(eval_panel)
    rows = []
    for fold in folds:
        train = eval_panel[eval_panel["date"] <= fold.train_end]
        scales = bt._in_sample_scales(train, bt.SEASON_LENGTH)
        fold_fc = forecasts[forecasts["fold"] == fold.index]
        for model_name in models:
            m = fold_fc[fold_fc["model"] == model_name].merge(scales, on="series_id", how="left")
            m["abs_err"] = (m["y"] - m["p50"]).abs()
            per_series = m.groupby("series_id").agg(
                mae=("abs_err", "mean"),
                scale_mae=("scale_mae", "first"),
                zero_rate=("zero_rate", "first"),
            )
            per_series["mase"] = per_series["mae"] / per_series["scale_mae"]
            per_series["bucket"] = dv.bucket_of(per_series["zero_rate"])
            g = per_series.groupby("bucket")["mase"].agg(["mean", "count"])
            for bucket, r in g.iterrows():
                rows.append(
                    {
                        "fold": fold.index,
                        "model": model_name,
                        "bucket": bucket,
                        "mase": r["mean"],
                        "n_series": int(r["count"]),
                    }
                )
    return pd.DataFrame(rows)


# ---- decision cost under the shipped policy -----------------------------------------------------


def shipped_cell(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    calib: pd.DataFrame,
    costs: CostParams,
) -> pd.DataFrame:
    """`ab.run_cell` at the one cell the project actually ships: legacy safety-stock scheme,
    `costs.lot_multiple`/`costs.service_level_target` (25%/yr holding, 27.5% margin, lot 2.0,
    target 95% — `config.load_config` defaults). Runs every model in `bt.MODEL_FACTORIES`,
    including the three added here; returns the per-(model, fold, series) detail frame."""
    _, detail, _ = ab.run_cell(
        panel,
        forecasts,
        calib,
        LEGACY[0],
        LEGACY[1],
        costs,
        service_level=costs.service_level_target,
        lot_multiple=costs.lot_multiple,
    )
    return detail


def cost_table(detail: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Pooled cost/CSL/fill per model — same aggregation as `order_up_to.curves`/`pooled`, but for
    a single already-selected cell rather than a whole sweep."""
    d = detail[detail["model"].isin(models)]
    per_fold = d.groupby(["model", "fold"]).agg(
        cost=("total_cost", "sum"),
        cycles=("cycles", "sum"),
        bad=("stockout_cycles", "sum"),
        shipped=("units_shipped", "sum"),
        demanded=("units_demanded", "sum"),
    )
    per_fold["csl"] = 1 - per_fold["bad"] / per_fold["cycles"]
    per_fold["fill"] = per_fold["shipped"] / per_fold["demanded"]
    return per_fold.groupby("model")[["cost", "csl", "fill"]].mean().reindex(models)


def bucket_cost_table(
    detail: pd.DataFrame, models: list[str], zero_rates: pd.Series
) -> pd.DataFrame:
    """`cost_table`, split by intermittency bucket."""
    d = detail[detail["model"].isin(models)].copy()
    d["bucket"] = dv.bucket_of(d["series_id"].map(zero_rates))
    rows = []
    for bucket in ("intermittent", "regular"):
        sub = d[d["bucket"] == bucket]
        t = cost_table(sub, models)
        t = t.reset_index()
        t.insert(1, "bucket", bucket)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def cost_ci_table(detail: pd.DataFrame, against: list[str]) -> tuple[pd.DataFrame, dict]:
    """Paired-bootstrap CI, each of the three new models minus each of `against` (SeasonalNaive,
    LightGBM), on total cost, pooled over folds. Returns the display table and the raw CIs keyed
    by (model, vs), so the report text can state the numbers rather than just show the table."""
    rows = []
    raw = {}
    for new_model in NEW_MODELS:
        a = detail[detail["model"] == new_model]
        for base in against:
            b = detail[detail["model"] == base]
            ci = boot.bootstrap_diff(a, b, "total_cost", group_cols=["fold"])
            raw[(new_model, base)] = ci
            rows.append(
                {
                    "model": new_model,
                    "vs": base,
                    "cost diff": f"${ci['point']:+,.0f}",
                    "95% CI": f"[${ci['ci_lo']:+,.0f}, ${ci['ci_hi']:+,.0f}]",
                    "reading": (
                        "crosses zero — not distinguishable"
                        if ci["crosses_zero"]
                        else (f"{new_model} cheaper" if ci["point"] < 0 else f"{base} cheaper")
                    ),
                }
            )
    return pd.DataFrame(rows), raw


# ---- report --------------------------------------------------------------------------------


def _md(df: pd.DataFrame, floats: tuple[str, ...] = ()) -> str:
    display = df.copy()
    for c in floats:
        if c in display.columns:
            display[c] = display[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    header = "| " + " | ".join(str(c) for c in display.columns) + " |"
    sep = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


def build_report(
    acc: pd.DataFrame,
    acc_bucket: pd.DataFrame,
    cost_pooled: pd.DataFrame,
    cost_bucket: pd.DataFrame,
    ci: pd.DataFrame,
    costs: CostParams,
    ci_raw: dict,
) -> str:
    acc_summary = (
        acc.groupby("model")[["mase", "rmsse", "pinball_p50", "coverage_80"]]
        .mean()
        .reindex([*NEW_MODELS])
        .reset_index()
    )
    acc_bucket_summary = (
        acc_bucket.groupby(["model", "bucket"])["mase"]
        .mean()
        .reset_index()
        .sort_values(["bucket", "mase"])
    )
    cost_pooled_show = cost_pooled.reset_index().assign(
        cost=lambda d: d["cost"].map("${:,.0f}".format),
        csl=lambda d: d["csl"].map("{:.1%}".format),
        fill=lambda d: d["fill"].map("{:.1%}".format),
    )
    cost_bucket_show = cost_bucket.assign(
        cost=lambda d: d["cost"].map("${:,.0f}".format),
        csl=lambda d: d["csl"].map("{:.1%}".format),
        fill=lambda d: d["fill"].map("{:.1%}".format),
    )
    date_str = dt.date.today().isoformat()
    return "\n".join(
        [
            f"# Croston family — {date_str}",
            "",
            "v1.1 Task 4. `CrostonClassic`, `CrostonSBA` and `TSB` (`statsforecast`, conformal "
            "intervals like every other statistical baseline), added to `bt.MODEL_FACTORIES` "
            "alongside the five models every other report already covers. Same 400-series "
            "sample, same 4 rolling-origin folds, 28-day horizon as `reports/backtest_*.md`.",
            "",
            "## Accuracy",
            "",
            "Mean MASE/RMSSE/coverage across folds. For comparison, the five-model table "
            "(`reports/backtest_2026-09-10.md`): AutoTheta 1.524, MovingAverage 1.530, AutoETS "
            "1.530, LightGBM 1.587, SeasonalNaive 1.633 (MASE).",
            "",
            _md(acc_summary, floats=("mase", "rmsse", "pinball_p50", "coverage_80")),
            "",
            "By intermittency bucket (>50% zero training days = intermittent):",
            "",
            _md(acc_bucket_summary, floats=("mase",)),
            "",
            "## Decision cost — the shipped policy",
            "",
            f"`S = s + {costs.lot_multiple:g} x lead-time demand`, "
            f"{costs.service_level_target:.0%} nominal target, legacy (normal, intermittency) "
            f"safety-stock scheme, holding "
            f"{costs.holding_cost_rate * 365:.1%}/yr, lost sale = {costs.gross_margin:.1%} of "
            "price — the same cell the README's four-reversal table reports as the current "
            "default for LightGBM/SeasonalNaive. Pooled (mean cost per 28-day fold):",
            "",
            _md(cost_pooled_show),
            "",
            "By intermittency bucket:",
            "",
            _md(cost_bucket_show),
            "",
            "## Confidence intervals",
            "",
            "Paired bootstrap over series (2,000 resamples), pooled over folds, cost difference "
            "= row model minus `vs` model:",
            "",
            _md(ci),
            "",
            "## Reading",
            "",
            _reading_paragraph(acc_summary, cost_pooled, ci_raw),
        ]
    )


def _reading_paragraph(acc_summary: pd.DataFrame, cost_pooled: pd.DataFrame, ci_raw: dict) -> str:
    best_acc = acc_summary.loc[acc_summary["mase"].idxmin()]
    lgb_cost = float(cost_pooled.loc["LightGBM", "cost"])
    sn_cost = float(cost_pooled.loc["SeasonalNaive", "cost"])
    cheapest_new = min(NEW_MODELS, key=lambda m: cost_pooled.loc[m, "cost"])
    vs_sn = [ci_raw[(m, "SeasonalNaive")] for m in NEW_MODELS]
    vs_lgb = [ci_raw[(m, "LightGBM")] for m in NEW_MODELS]
    all_beat_sn = all(not c["crosses_zero"] and c["point"] < 0 for c in vs_sn)
    all_tie_lgb = all(c["crosses_zero"] for c in vs_lgb)
    return (
        f"On accuracy, the Croston family lands inside the existing four-model cluster, not "
        f"below it: {best_acc['model']} is the best of the three at {best_acc['mase']:.3f} MASE, "
        f"against AutoTheta 1.524-LightGBM 1.587 for the five models already reported — a "
        f"different method, the same RMSSE-cluster story `reports/model_divergence_2026-09-15.md` "
        "already found (a flat-ish forecast on a 77%-zero panel bunches every reasonable method "
        "together on point accuracy). "
        + (
            "On decision cost, all three beat SeasonalNaive with a CI excluding zero, and none "
            f"is distinguishable from LightGBM (cheapest new model {cheapest_new} at "
            f"${cost_pooled.loc[cheapest_new, 'cost']:,.0f}/fold vs. LightGBM ${lgb_cost:,.0f} "
            f"and SeasonalNaive ${sn_cost:,.0f}) — under the policy actually shipped, the Croston "
            "family is a real, evidenced alternative to LightGBM, not a model that loses and "
            "closes the gap only formally."
            if all_beat_sn and all_tie_lgb
            else "On decision cost the family's standing against SeasonalNaive and LightGBM is "
            "mixed rather than uniform across all three — see the confidence-interval table "
            "above for which comparisons actually clear zero."
        )
        + " Its advantage is expected to concentrate in the intermittent bucket, where "
        "Croston-style methods are designed to beat a method that keeps forecasting through "
        "zero-demand periods; the by-bucket tables above, not just the pooled ones, are where "
        "to look for that."
    )


def main() -> None:
    costs = load_config().costs
    panel = bt.load_eval_panel()

    forecasts = extend_forecasts(panel)
    calib = extend_calibration(panel, costs)

    all_models = list(bt.MODEL_FACTORIES)
    acc = accuracy_table(panel, forecasts, all_models)
    acc_bucket = bucket_accuracy(panel, forecasts, list(NEW_MODELS))

    detail = shipped_cell(panel, forecasts, calib, costs)
    cost_pooled = cost_table(detail, [*NEW_MODELS, "SeasonalNaive", "LightGBM"])
    zero_rates = bt._intermittency(panel)
    cost_bucket = bucket_cost_table(detail, [*NEW_MODELS, "SeasonalNaive", "LightGBM"], zero_rates)
    ci, ci_raw = cost_ci_table(detail, against=["SeasonalNaive", "LightGBM"])

    report = build_report(acc, acc_bucket, cost_pooled, cost_bucket, ci, costs, ci_raw)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"croston_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
