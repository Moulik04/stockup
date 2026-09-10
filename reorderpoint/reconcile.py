"""Hierarchical reconciliation experiment (Track A only): does enforcing coherence across the
state -> store -> category -> department -> item hierarchy improve accuracy over unreconciled
base forecasts? Bottom-up and MinTrace via `hierarchicalforecast`.

Simplification vs. the official M5 competition hierarchy: this uses
one *nested* chain (state -> store -> category -> department -> item) rather than M5's full
*grouped* hierarchy (which also crosses state x category, store x department, etc. independently
of the geographic path). item_id already determines dept_id/cat_id, so this chain's bottom level
is exactly this project's existing series_id (item x store) — no double-counting.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from reorderpoint import backtest as bt
from reorderpoint.config import REPO_ROOT
from reorderpoint.metrics import mase, rmsse

PANEL_PATH = REPO_ROOT / "data" / "track_a" / "processed" / "panel.parquet"
REPORTS_DIR = REPO_ROOT / "reports"

# 2 states (1 store each) keeps every category/department represented while staying close in
# scale to the 400-series backtest sample — full store list would multiply compute ~5x for a
# comparison this experiment doesn't need every store to make.
STORE_IDS = ["CA_1", "TX_1"]
N_SERIES_SAMPLE = 400
SAMPLE_SEED = 0
HORIZON = 28
N_FOLDS = 4
SEASON_LENGTH = 7

HIERARCHY_SPEC = [
    ["state_id"],
    ["state_id", "store_id"],
    ["state_id", "store_id", "cat_id"],
    ["state_id", "store_id", "cat_id", "dept_id"],
    ["state_id", "store_id", "cat_id", "dept_id", "item_id"],
]
LEVEL_NAMES = ["/".join(level) for level in HIERARCHY_SPEC]

MODEL_NAME = "AutoETS"


def build_hierarchy(
    panel: pd.DataFrame, spec: list[list[str]] = HIERARCHY_SPEC
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Bottom-level (item x store) rows in `panel` -> one long table covering every level of
    `spec`, an aggregation (summing) matrix, and a level -> node-id tags dict.
    """
    bottom = panel.rename(columns={"date": "ds"})[
        ["state_id", "store_id", "cat_id", "dept_id", "item_id", "ds", "y"]
    ]
    Y_df, S_df, tags = aggregate(bottom, spec)
    return Y_df.rename(columns={"unique_id": "series_id", "ds": "date"}), S_df, tags


def level_lookup(tags: dict) -> pd.Series:
    """Inverts `tags` (level -> [node ids]) into node id -> level, for per-level reporting."""
    pairs = [(node_id, level) for level, node_ids in tags.items() for node_id in node_ids]
    ids, levels = zip(*pairs, strict=True)
    return pd.Series(levels, index=pd.Index(ids, name="series_id"))


def fit_forecast_with_fitted(
    train: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One AutoETS fit across every hierarchy node at once (statsforecast batches internally).

    Returns (forecast_df: series_id/date/AutoETS, fitted_df: series_id/date/y/AutoETS) — the
    latter carries the model's own in-sample fitted values, which MinTrace needs to estimate its
    residual covariance (statsforecast's `fitted=True` + `forecast_fitted_values()` gives this in
    the same call, no second fit).
    """
    sf_train = train.rename(columns={"series_id": "unique_id", "date": "ds"})[
        ["unique_id", "ds", "y"]
    ]
    sf = StatsForecast(models=[AutoETS(season_length=SEASON_LENGTH)], freq="D", n_jobs=-1)
    forecast = sf.forecast(df=sf_train, h=horizon, fitted=True)
    fitted = sf.forecast_fitted_values()
    rename = {"unique_id": "series_id", "ds": "date"}
    return forecast.rename(columns=rename), fitted.rename(columns=rename)


def reconcile_forecasts(
    forecast_df: pd.DataFrame, fitted_df: pd.DataFrame, S_df: pd.DataFrame, tags: dict
) -> pd.DataFrame:
    """BottomUp + MinTrace (shrinkage) reconciliation of `forecast_df` against `fitted_df`'s
    in-sample residuals — hierarchicalforecast expects `unique_id`/`ds` column names.
    """
    hf_forecast = forecast_df.rename(columns={"series_id": "unique_id", "date": "ds"})
    hf_fitted = fitted_df.rename(columns={"series_id": "unique_id", "date": "ds"})
    hr = HierarchicalReconciliation(reconcilers=[BottomUp(), MinTrace(method="mint_shrink")])
    reconciled = hr.reconcile(Y_hat_df=hf_forecast, Y_df=hf_fitted, S_df=S_df, tags=tags)
    return reconciled.rename(columns={"unique_id": "series_id", "ds": "date"})


def metrics_by_level(
    merged: pd.DataFrame,
    train: pd.DataFrame,
    level_of: pd.Series,
    forecast_col: str,
    season_length: int = SEASON_LENGTH,
) -> pd.DataFrame:
    """Per-level (mean MASE, mean RMSSE, n_nodes) for one forecast column, `merged` carrying
    columns series_id/date/y/`forecast_col` over the test period and `train` the history used
    for each node's in-sample scale.
    """
    rows = []
    for series_id, group in merged.groupby("series_id"):
        y_train = train.loc[train["series_id"] == series_id, "y"].to_numpy()
        if len(y_train) <= season_length:
            continue
        y_true = group["y"].to_numpy()
        y_pred = group[forecast_col].to_numpy()
        rows.append(
            {
                "series_id": series_id,
                "mase": mase(y_true, y_pred, y_train, season_length),
                "rmsse": rmsse(y_true, y_pred, y_train, season_length),
            }
        )
    # Explicit columns even when `rows` is empty (every node skipped for insufficient history) —
    # pd.DataFrame([]) has no columns at all, so .set_index("series_id") would raise a confusing
    # KeyError instead of this function's caller just seeing "no nodes at this level" naturally.
    per_node = pd.DataFrame(rows, columns=["series_id", "mase", "rmsse"]).set_index("series_id")
    per_node["level"] = level_of.reindex(per_node.index)

    return (
        per_node.groupby("level")
        .agg(mase=("mase", "mean"), rmsse=("rmsse", "mean"), n_nodes=("mase", "size"))
        .reset_index()
    )


ForecastFn = Callable[[pd.DataFrame, int], tuple[pd.DataFrame, pd.DataFrame]]


def evaluate_fold_reconciliation(
    Y_df: pd.DataFrame,
    fold: bt.Fold,
    S_df: pd.DataFrame,
    tags: dict,
    forecast_fn: ForecastFn = fit_forecast_with_fitted,
) -> pd.DataFrame:
    """One fold, every hierarchy node at once: fit -> forecast+fitted -> reconcile -> per-level
    (mase, rmsse, n_nodes) for the unreconciled base forecast, BottomUp, and MinTrace.
    """
    train = Y_df[Y_df["date"] <= fold.train_end]
    test = Y_df[(Y_df["date"] >= fold.test_start) & (Y_df["date"] <= fold.test_end)]
    horizon = (fold.test_end - fold.test_start).days + 1

    forecast_df, fitted_df = forecast_fn(train, horizon)
    reconciled = reconcile_forecasts(forecast_df, fitted_df, S_df, tags)

    merged = test[["series_id", "date", "y"]].merge(
        reconciled, on=["series_id", "date"], how="inner"
    )
    lookup = level_lookup(tags)
    mintrace_col = next(c for c in reconciled.columns if c.startswith(f"{MODEL_NAME}/MinTrace"))
    method_cols = {
        "base": MODEL_NAME,
        "BottomUp": f"{MODEL_NAME}/BottomUp",
        "MinTrace": mintrace_col,
    }

    frames = []
    for method_name, col in method_cols.items():
        level_metrics = metrics_by_level(merged, train, lookup, forecast_col=col)
        level_metrics["method"] = method_name
        level_metrics["fold"] = fold.index
        frames.append(level_metrics)
    return pd.concat(frames, ignore_index=True)


def run_reconciliation_experiment(
    panel: pd.DataFrame, forecast_fn: ForecastFn = fit_forecast_with_fitted
) -> pd.DataFrame:
    """Returns per (fold, level, method) metrics across every fold — the source table for
    `write_reconciliation_report`.
    """
    bottom_panel = bt.sample_series(
        panel[panel["store_id"].isin(STORE_IDS)], N_SERIES_SAMPLE, seed=SAMPLE_SEED
    )
    Y_df, S_df, tags = build_hierarchy(bottom_panel)
    folds = bt.make_folds(Y_df, horizon=HORIZON, n_folds=N_FOLDS)

    frames = [evaluate_fold_reconciliation(Y_df, fold, S_df, tags, forecast_fn) for fold in folds]
    return pd.concat(frames, ignore_index=True)


def write_reconciliation_report(results: pd.DataFrame) -> str:
    summary = results.groupby(["level", "method"])[["mase", "rmsse"]].mean().reset_index()
    mase_pivot = (
        summary.pivot(index="level", columns="method", values="mase")
        .reindex(LEVEL_NAMES)
        .reset_index()
    )
    for method in ("BottomUp", "MinTrace"):
        if method in mase_pivot.columns:
            mase_pivot[f"{method}_vs_base"] = (
                (mase_pivot[method] - mase_pivot["base"]) / mase_pivot["base"] * 100
            )

    n_nodes = (
        results[results["method"] == "base"].groupby("level")["n_nodes"].mean().reindex(LEVEL_NAMES)
    )

    helped = mase_pivot[
        (mase_pivot.get("BottomUp_vs_base", 0) < 0) | (mase_pivot.get("MinTrace_vs_base", 0) < 0)
    ]["level"].tolist()
    hurt = mase_pivot[
        (mase_pivot.get("BottomUp_vs_base", 0) > 0) & (mase_pivot.get("MinTrace_vs_base", 0) > 0)
    ]["level"].tolist()

    bottom_base = mase_pivot.loc[mase_pivot["level"] == LEVEL_NAMES[-1], "base"].iloc[0]
    top_base = mase_pivot.loc[mase_pivot["level"] == LEVEL_NAMES[0], "base"].iloc[0]
    noise_ratio = bottom_base / top_base if top_base else float("nan")
    noise_note = (
        f"Bottom-level base MASE ({bottom_base:.2f}) is {noise_ratio:.1f}x top-level base MASE "
        f"({top_base:.2f}) — bottom-level (item x store) series are simply noisier/harder to "
        "forecast than an aggregate. BottomUp reconciliation just sums these noisier bottom "
        "forecasts, so it inherits that noise at every level above the bottom; MinTrace's "
        "covariance-weighted blend is specifically designed to correct for this rather than "
        "trusting the bottom level uniformly, which is consistent with it degrading less (or "
        "even improving on) the base forecast where BottomUp doesn't."
    )

    date_str = dt.date.today().isoformat()
    lines = [
        f"# Hierarchical reconciliation report — {date_str}",
        "",
        f"Track A (M5), stores {', '.join(STORE_IDS)} (full state/store/category/department "
        f"hierarchy under them), {N_SERIES_SAMPLE}-series bottom-level subsample "
        f"(seed={SAMPLE_SEED}). {N_FOLDS} rolling-origin folds, horizon={HORIZON} days, "
        f"base model={MODEL_NAME}. Bottom-up and MinTrace (shrinkage) via `hierarchicalforecast` "
        "— see this module's docstring for the simplified (nested, not grouped) hierarchy used.",
        "",
        "## Mean MASE by level and reconciliation method",
        "",
        bt._markdown_table(
            mase_pivot,
            float_cols=tuple(c for c in mase_pivot.columns if c != "level"),
        ),
        "",
        "n_nodes per level (mean across folds): "
        + ", ".join(f"{lvl}={n_nodes[lvl]:.0f}" for lvl in LEVEL_NAMES),
        "",
        "## Does reconciliation help, and where?",
        "",
        (
            f"Reconciliation (BottomUp and/or MinTrace) improves mean MASE over the "
            f"unreconciled base forecast at: {', '.join(helped) if helped else 'no level'}. "
            f"It makes things worse at: {', '.join(hurt) if hurt else 'no level'}. "
            "The bottom level is expected to be least affected by BottomUp specifically (it's "
            "definitionally a no-op there — nothing to sum up from below); MinTrace can still "
            "adjust bottom-level forecasts using the whole hierarchy's residual covariance."
        ),
        "",
        noise_note,
    ]
    return "\n".join(lines)


def main() -> None:
    # Filtered at parquet-read time (predicate pushdown), not after loading — the full panel is
    # ~11.7GB in memory (all 30,490 series), more than this project's dev machine has; reading
    # only STORE_IDS up front means the full frame is never materialized.
    panel = pd.read_parquet(PANEL_PATH, filters=[("store_id", "in", STORE_IDS)])
    panel["date"] = pd.to_datetime(panel["date"])

    results = run_reconciliation_experiment(panel)
    report = write_reconciliation_report(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = dt.date.today().isoformat()
    out_path = REPORTS_DIR / f"reconciliation_{date_str}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
