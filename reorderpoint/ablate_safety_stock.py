"""v1.1 Task 5: does the service-level gap come from distributional shape or from pooling?

Runs the full 2x2 — {normal, empirical} x {2-bucket, 6-bucket} — through the identical simulation
and reports cost, fill rate and cycle service level for each cell, plus the two diagnostics that
predict which cell should win (fill rate against volume decile, and the spread of per-series
residual std inside each bucket).

Everything runs off `divergence.FORECASTS_PATH`, the forecast cache the Task 1 run already paid
for, so four cells cost four simulations rather than four backtests. The one thing not in that
cache is fold 1's bootstrap calibration (fit on the training window minus its last `lead_time`
days), which is computed once here and cached alongside it.
"""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd

from reorderpoint import backtest as bt
from reorderpoint import decision as dec
from reorderpoint import divergence as dv
from reorderpoint import safety_stock as ss
from reorderpoint.config import CostParams, load_config

CALIB_PATH = bt.PANEL_PATH.parent / "calibration_residuals.parquet"
CELL_RESULTS_PATH = bt.PANEL_PATH.parent / "safety_stock_cells.parquet"


def _train_stats(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(zero rate, mean daily volume) per series over a training window."""
    grouped = train.groupby("series_id")["y"]
    return grouped.apply(lambda s: (s == 0).mean()), grouped.mean()


def calibration_residuals(panel: pd.DataFrame, costs: CostParams) -> pd.DataFrame:
    """Fold 1 has no previous fold to borrow residuals from, so one calibration split is carved
    out of its own training window — the same procedure `decision._bootstrap_lead_time_std` uses,
    but keeping the residuals rather than collapsing them to a std. Cached: one fit per model.
    """
    if CALIB_PATH.exists():
        return pd.read_parquet(CALIB_PATH)

    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    first_fold = bt.make_folds(eval_panel)[0]
    train = eval_panel[eval_panel["date"] <= first_fold.train_end]
    cutoff = train["date"].max() - pd.Timedelta(days=costs.lead_time_days - 1)
    calib_train = train[train["date"] < cutoff]
    calib_test = train[train["date"] >= cutoff]

    zero_rates, volumes = _train_stats(calib_train)
    frames = []
    for model_name, factory in bt.MODEL_FACTORIES.items():
        model = factory(costs.lead_time_days)
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
        print(f"calibration fit: {model_name}")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(CALIB_PATH, index=False)
    return out


def run_cell(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    calib: pd.DataFrame,
    form: str,
    granularity: str,
    costs: CostParams,
    n_boot: int = ss.N_BOOTSTRAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One cell of the 2x2, across every fold and model.

    Returns (results, detail, bucket_tables). Safety stock chains forward exactly as the
    production path does: fold 1 uses the bootstrap calibration, each later fold uses the
    previous fold's own realised residuals. No fold ever sees its own outcome.
    """
    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    folds = bt.make_folds(eval_panel)
    target = costs.service_level_target

    rows, details, bucket_rows = [], [], []
    for model_name in bt.MODEL_FACTORIES:
        model_calib = calib[calib["model"] == model_name].set_index("series_id")
        prior_ss, table = ss.per_series_safety_stock(
            model_calib["residual"],
            model_calib["zero_rate"],
            model_calib["volume"],
            target,
            form=form,
            granularity=granularity,
            n_boot=n_boot,
        )
        table = table.assign(model=model_name, fold=0, form=form, granularity=granularity)
        bucket_rows.append(table.reset_index())

        for fold in folds:
            train = eval_panel[eval_panel["date"] <= fold.train_end]
            test = eval_panel[
                (eval_panel["date"] >= fold.test_start) & (eval_panel["date"] <= fold.test_end)
            ]
            preds = forecasts[
                (forecasts["fold"] == fold.index) & (forecasts["model"] == model_name)
            ]
            if preds.empty:
                continue

            stats = dec.lead_time_demand_stats(preds, costs.lead_time_days)
            safety = prior_ss.reindex(stats.index)
            # A series the calibration never saw falls back to the quantile-derived buffer, the
            # same fallback the production path uses for an incomplete calibration set.
            fallback = dec.reorder_decision(stats, target)["safety_stock"]
            decisions = stats.copy()
            decisions["safety_stock"] = safety.where(safety.notna(), fallback)
            decisions["reorder_point"] = decisions["mean"] + decisions["safety_stock"]

            unit_costs = dec.unit_cost_from_price(train)
            demand_by_series = {
                sid: g.sort_values("date")["y"].to_numpy()
                for sid, g in test.groupby("series_id", sort=False)
            }
            row, detail = dec._simulate_policy(
                decisions,
                demand_by_series,
                unit_costs,
                float(unit_costs.mean()),
                costs,
                fold,
                model_name,
                policy="csl",
            )
            row |= {"form": form, "granularity": granularity}
            rows.append(row)
            detail = detail.assign(
                fold=fold.index, model=model_name, form=form, granularity=granularity
            )
            details.append(detail)

            # this fold's own residuals calibrate the next one
            residuals = dec.lead_time_forecast_residuals(
                preds, test[["series_id", "date", "y"]], costs.lead_time_days
            )
            zero_rates, volumes = _train_stats(train)
            prior_ss, table = ss.per_series_safety_stock(
                residuals,
                zero_rates,
                volumes,
                target,
                form=form,
                granularity=granularity,
                n_boot=n_boot,
            )
            table = table.assign(
                model=model_name, fold=fold.index, form=form, granularity=granularity
            )
            bucket_rows.append(table.reset_index())

    return (
        pd.DataFrame(rows),
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        pd.concat(bucket_rows, ignore_index=True),
    )


def fill_rate_by_volume_decile(detail: pd.DataFrame, volumes: pd.Series) -> pd.DataFrame:
    """Realised fill rate against series volume decile, per model — the diagnostic that says
    whether pooling in absolute units under-provisions the high-volume tail."""
    df = detail.copy()
    df["volume"] = df["series_id"].map(volumes)
    df = df.dropna(subset=["volume"])
    df["decile"] = pd.qcut(df["volume"].rank(method="first"), 10, labels=range(1, 11))
    grouped = df.groupby(["model", "decile"], observed=True).agg(
        units_demanded=("units_demanded", "sum"), units_shipped=("units_shipped", "sum")
    )
    grouped["fill_rate"] = grouped["units_shipped"] / grouped["units_demanded"].replace(0, np.nan)
    return grouped.reset_index()


def plot_fill_rate_by_volume(by_decile: pd.DataFrame, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, group in by_decile.groupby("model"):
        g = group.sort_values("decile")
        ax.plot(g["decile"].astype(int), g["fill_rate"], marker="o", lw=1.6, label=model)
    ax.set_xlabel("series volume decile (1 = lowest)")
    ax.set_ylabel("realised fill rate")
    ax.set_title("Fill rate by volume decile — current scheme (normal, 2-bucket)", fontsize=11)
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _cell_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["form", "granularity"])[
            ["total_cost", "holding_cost", "stockout_cost", "fill_rate", "cycle_service_level"]
        ]
        .mean()
        .reset_index()
        .sort_values("total_cost")
    )


def build_report(
    results: pd.DataFrame,
    buckets: pd.DataFrame,
    by_decile: pd.DataFrame,
    dispersion: pd.DataFrame,
    costs: CostParams,
    plot_rel_path: str | None,
) -> str:
    target = costs.service_level_target
    summary = _cell_summary(results)
    baseline = summary[
        (summary["form"] == "normal") & (summary["granularity"] == "intermittency")
    ].iloc[0]
    best = summary.iloc[0]

    per_cell_model = (
        results.groupby(["form", "granularity", "model"])[
            ["total_cost", "fill_rate", "cycle_service_level"]
        ]
        .mean()
        .reset_index()
    )

    fine = buckets[buckets["granularity"] == "intermittency_volume"]
    coarse = buckets[buckets["granularity"] == "intermittency"]
    fine_emp = fine[fine["form"] == "empirical"]
    coarse_emp = coarse[coarse["form"] == "empirical"]

    decile_note = ""
    if not by_decile.empty:
        pooled = by_decile.groupby("decile", observed=True).apply(
            lambda g: g["units_shipped"].sum() / max(g["units_demanded"].sum(), 1e-9),
            include_groups=False,
        )
        lo, hi = pooled.iloc[:3].mean(), pooled.iloc[-3:].mean()
        direction = "declines" if hi < lo else "rises"
        decile_note = (
            f"Pooled across models, fill rate {direction} from {lo:.1%} in the bottom three "
            f"volume deciles to {hi:.1%} in the top three. "
            + (
                "A declining relationship is the signature of pooling in absolute units: one "
                "buffer in units is generous for a low-volume SKU and thin for a high-volume "
                "one."
                if hi < lo
                else "The pooling hypothesis predicted a decline; this does not show one, which "
                "is evidence against absolute-unit pooling being the mechanism behind the "
                "service-level gap."
            )
        )

    lines = [
        f"# Safety-stock calibration: 2x2 ablation — {dt.date.today().isoformat()}",
        "",
        "Two independent levers behind the decision layer's safety stock, measured separately "
        "and together. Same folds, models, sample and simulation as "
        "`reports/decision_<date>.md`; the only thing that varies is how a bucket of forecast "
        "residuals becomes a buffer.",
        "",
        "- **Form** — `normal`: `z · σ_pooled`, assuming symmetric normal lead-time demand. "
        "`empirical`: the residual distribution's own quantile at the target service level, "
        "assuming nothing about its shape.",
        "- **Granularity** — `intermittency`: the current 2 buckets. `intermittency_volume`: "
        "crossed with volume terciles, 6 buckets.",
        "",
        f"Target service level {target:.0%}. Cost, fill rate and CSL are means across folds and "
        "models.",
        "",
        "## The 2x2",
        "",
        bt._markdown_table(
            summary.rename(columns={"total_cost": "mean_total_cost"}),
            float_cols=(
                "mean_total_cost",
                "holding_cost",
                "stockout_cost",
                "fill_rate",
                "cycle_service_level",
            ),
        ),
        "",
        f"Baseline (the shipped scheme) is **normal × intermittency**: "
        f"${baseline['total_cost']:,.0f} per fold, fill rate {baseline['fill_rate']:.1%}, CSL "
        f"{baseline['cycle_service_level']:.1%}. Best cell by cost is **{best['form']} × "
        f"{best['granularity']}** at ${best['total_cost']:,.0f} "
        f"({(best['total_cost'] - baseline['total_cost']) / baseline['total_cost']:+.1%}), fill "
        f"rate {best['fill_rate']:.1%}, CSL {best['cycle_service_level']:.1%}.",
        "",
        "## Per model",
        "",
        bt._markdown_table(
            per_cell_model, float_cols=("total_cost", "fill_rate", "cycle_service_level")
        ),
        "",
        "## Can the buckets support the estimate?",
        "",
        "The constraint named up front: ~400 residuals per fold, split two ways, leaves ~200 per "
        f"bucket — a {target:.0%} quantile then rests on ~10 tail observations. Split six ways it "
        "is ~3. Each bucket's statistic therefore carries a bootstrap CI "
        f"({ss.N_BOOTSTRAP:,} resamples) on the quantile itself.",
        "",
        "Median bucket sizes and CI widths (CI width as a fraction of the point estimate):",
        "",
        bt._markdown_table(
            buckets.groupby(["form", "granularity"])
            .agg(
                median_n=("n", "median"),
                min_n=("n", "min"),
                median_ci_width_frac=("ci_width_frac", "median"),
                max_ci_width_frac=("ci_width_frac", "max"),
            )
            .reset_index(),
            float_cols=("median_n", "min_n", "median_ci_width_frac", "max_ci_width_frac"),
        ),
        "",
    ]

    if not fine_emp.empty and not coarse_emp.empty:
        fine_width = fine_emp["ci_width_frac"].median()
        coarse_width = coarse_emp["ci_width_frac"].median()
        usable = fine_width < 1.0
        lines += [
            f"At 6 buckets the empirical quantile's CI spans a median "
            f"{fine_width:.0%} of the estimate itself, against {coarse_width:.0%} at 2 buckets "
            f"(minimum bucket size {int(fine_emp['n'].min())} vs "
            f"{int(coarse_emp['n'].min())}). "
            + (
                "That is wide but still informative — the finer cells' results below can be read "
                "as estimates, with that uncertainty attached."
                if usable
                else "**A CI wider than the estimate it surrounds is not a usable buffer.** The "
                "6-bucket cells below are reported for completeness, but this panel does not "
                "carry enough residuals per bucket to estimate a tail quantile at that "
                "granularity. That is a finding about the data, not a tuning problem: more "
                "buckets cannot help until there are more series per bucket."
            ),
            "",
        ]

    lines += [
        "## Diagnostic 1 — fill rate against volume decile",
        "",
        "Run on the current scheme (normal × intermittency). If pooling in absolute units is "
        "what depresses service, fill rate should fall as volume rises: the same buffer in units "
        "covers a low-volume SKU comfortably and a high-volume one barely.",
        "",
        decile_note,
        "",
    ]
    if plot_rel_path:
        lines += [f"![Fill rate by volume decile]({plot_rel_path})", ""]

    lines += [
        "## Diagnostic 2 — within-bucket dispersion of per-series residual std",
        "",
        "If a single bucket spans an order of magnitude in per-series residual std, pooling it "
        "into one absolute buffer is indefensible regardless of what the cost table says.",
        "",
        bt._markdown_table(
            dispersion.reset_index(),
            float_cols=tuple(c for c in dispersion.reset_index().columns if c != "bucket"),
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    panel = pd.read_parquet(bt.PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    config = load_config()
    costs = config.costs

    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = calibration_residuals(panel, costs)

    all_results, all_detail, all_buckets = [], [], []
    for form, granularity in itertools.product(ss.FORMS, ss.GRANULARITIES):
        print(f"cell: {form} x {granularity}")
        results, detail, buckets = run_cell(panel, forecasts, calib, form, granularity, costs)
        all_results.append(results)
        all_detail.append(detail)
        all_buckets.append(buckets)

    results = pd.concat(all_results, ignore_index=True)
    detail = pd.concat(all_detail, ignore_index=True)
    buckets = pd.concat(all_buckets, ignore_index=True)
    results.to_parquet(CELL_RESULTS_PATH, index=False)

    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    last_train = eval_panel[eval_panel["date"] <= bt.make_folds(eval_panel)[-1].train_end]
    zero_rates, volumes = _train_stats(last_train)

    baseline_detail = detail[
        (detail["form"] == "normal") & (detail["granularity"] == "intermittency")
    ]
    by_decile = fill_rate_by_volume_decile(baseline_detail, volumes)

    files_dir = bt.REPORTS_DIR / "safety_stock_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    date_str = dt.date.today().isoformat()
    plot_name = f"fill_rate_by_volume_{date_str}.png"
    plot_fill_rate_by_volume(by_decile, files_dir / plot_name)

    # per-series residual std across folds, against the current 2-bucket assignment
    residual_std = _per_series_residual_std(forecasts, eval_panel, costs)
    dispersion = ss.within_bucket_dispersion(
        residual_std, ss.assign_buckets(zero_rates, volumes, "intermittency")
    )

    report = build_report(
        results, buckets, by_decile, dispersion, costs, f"safety_stock_files/{plot_name}"
    )
    out_path = bt.REPORTS_DIR / f"safety_stock_{date_str}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(_cell_summary(results).to_string(index=False))


def _per_series_residual_std(
    forecasts: pd.DataFrame, eval_panel: pd.DataFrame, costs: CostParams
) -> pd.Series:
    """Std of each series' own lead-time residuals across folds, for the production model.

    Only 4 folds, so this is a noisy per-series estimate — which is exactly why the production
    path pools in the first place. It is used here only to show the *spread* of those estimates
    inside a bucket, not as a per-series buffer.
    """
    model = "LightGBM" if "LightGBM" in set(forecasts["model"]) else forecasts["model"].iloc[0]
    rows = []
    for fold in bt.make_folds(eval_panel):
        preds = forecasts[(forecasts["fold"] == fold.index) & (forecasts["model"] == model)]
        if preds.empty:
            continue
        test = eval_panel[
            (eval_panel["date"] >= fold.test_start) & (eval_panel["date"] <= fold.test_end)
        ]
        residuals = dec.lead_time_forecast_residuals(
            preds, test[["series_id", "date", "y"]], costs.lead_time_days
        )
        rows.append(residuals.rename(fold.index))
    wide = pd.concat(rows, axis=1)
    return wide.std(axis=1, ddof=1).rename("residual_std")


if __name__ == "__main__":
    main()
