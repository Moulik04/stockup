"""Quantile forecast -> reorder point, safety stock, order quantity, cost simulation.

Newsvendor-style service-level policy, documented in docs/decision.md. Implemented Phase 4.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from reorderpoint import backtest as bt
from reorderpoint.config import CostParams, load_config

Z80 = norm.ppf(0.9)  # P10/P90 is the central 80% interval everywhere in this project


def lead_time_demand_stats(forecast: pd.DataFrame, lead_time_days: int) -> pd.DataFrame:
    """`forecast`: columns series_id, date, p10, p50, p90 (one row per series per future date).

    Returns index series_id, columns mean/std of total demand over the first `lead_time_days`
    forecast rows per series (by date) — see docs/decision.md step 1.
    """
    df = forecast.sort_values(["series_id", "date"]).copy()
    df["daily_std"] = (df["p90"] - df["p10"]) / (2 * Z80)
    df = df.groupby("series_id", sort=False).head(lead_time_days)
    # skipna=False: pandas' default sum() treats an all-NaN group as 0, which would silently
    # turn a model's NaN forecast (e.g. from insufficient history) into "mean=0, std=0" -- a
    # modelling failure masquerading as "this series needs zero safety stock" -- rather than
    # propagating NaN through to reorder_point, where evaluate_fold_cost's caller can detect and
    # skip it explicitly instead of simulating a policy for a series that has no real forecast.
    out = df.groupby("series_id").agg(
        mean=("p50", lambda s: s.sum(skipna=False)),
        _sq_std=("daily_std", lambda s: (s**2).sum(skipna=False)),
    )
    out["std"] = np.sqrt(out.pop("_sq_std"))
    return out[["mean", "std"]]


def reorder_decision(lead_time_stats: pd.DataFrame, service_level: float) -> pd.DataFrame:
    """Adds safety_stock and reorder_point columns — docs/decision.md step 2."""
    z = norm.ppf(service_level)
    out = lead_time_stats.copy()
    out["safety_stock"] = z * out["std"]
    out["reorder_point"] = out["mean"] + out["safety_stock"]
    return out


def order_quantity(reorder_point: pd.Series, on_hand: pd.Series) -> pd.Series:
    """Order-up-to-S policy with S = reorder_point — docs/decision.md step 3."""
    return (reorder_point - on_hand).clip(lower=0)


def stockout_probability(on_hand: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """P(lead-time demand > on_hand) under the normal approximation — docs/decision.md step 4."""
    on_hand_arr = on_hand.to_numpy(dtype=float)
    mean_arr = mean.to_numpy(dtype=float)
    std_arr = std.to_numpy(dtype=float)
    zero_std = std_arr == 0
    safe_std = np.where(zero_std, 1.0, std_arr)
    prob = 1 - norm.cdf((on_hand_arr - mean_arr) / safe_std)
    prob = np.where(zero_std, np.where(on_hand_arr >= mean_arr, 0.0, 1.0), prob)
    return pd.Series(prob, index=on_hand.index)


def _rationale_line(
    on_hand: float, mean: float, lead_time_days: int, service_level: float, order_qty: float
) -> str:
    daily_rate = mean / lead_time_days if lead_time_days else np.nan
    days_covered = on_hand / daily_rate if daily_rate else float("inf")
    action = f"order {order_qty:.0f} units" if order_qty > 0 else "no order needed"
    return (
        f"On-hand ({on_hand:.0f}) covers {days_covered:.1f} days of expected demand "
        f"({daily_rate:.1f}/day); lead time is {lead_time_days} days at {service_level:.0%} "
        f"service level → {action}."
    )


def compute_decisions(
    forecast: pd.DataFrame, on_hand: pd.Series, lead_time_days: int, service_level: float
) -> pd.DataFrame:
    """Full per-SKU policy: mean, std, safety_stock, reorder_point, on_hand, order_quantity,
    stockout_probability, rationale — one row per series_id.
    """
    stats = lead_time_demand_stats(forecast, lead_time_days)
    out = reorder_decision(stats, service_level)
    out = out.reindex(on_hand.index)
    out["on_hand"] = on_hand
    out["order_quantity"] = order_quantity(out["reorder_point"], out["on_hand"])
    out["stockout_probability"] = stockout_probability(out["on_hand"], out["mean"], out["std"])
    out["rationale"] = [
        _rationale_line(row.on_hand, row.mean, lead_time_days, service_level, row.order_quantity)
        for row in out.itertuples()
    ]
    return out


def lead_time_forecast_residuals(
    forecast: pd.DataFrame, actual: pd.DataFrame, lead_time_days: int
) -> pd.Series:
    """Per-series (actual - predicted) total demand over the first `lead_time_days` forecast
    dates — the real, empirical counterpart to `lead_time_demand_stats`'s quantile-derived mean.
    `actual`: columns series_id, date, y, covering (at least) the same dates as `forecast`.
    """
    fc = (
        forecast.sort_values(["series_id", "date"])
        .groupby("series_id", sort=False)
        .head(lead_time_days)
    )
    predicted = fc.groupby("series_id")["p50"].sum()
    matched = fc[["series_id", "date"]].merge(actual, on=["series_id", "date"], how="left")
    realised = matched.groupby("series_id")["y"].sum()
    return (realised.reindex(predicted.index).fillna(0.0) - predicted).rename("residual")


def empirical_lead_time_std(
    residuals: pd.Series, zero_rates: pd.Series, threshold: float = 0.5
) -> pd.Series:
    """Pools per-series residuals into two intermittency buckets (same >50%-zero-days split as
    backtest.py's "where the model fails" analysis) and looks each series' own bucket-level std
    back up by series_id — far more stable than a per-series std computed from only the handful
    of historical folds this project's backtester produces (see docs/decision.md).
    """
    df = pd.DataFrame({"residual": residuals, "zero_rate": zero_rates.reindex(residuals.index)})
    df["bucket"] = np.where(df["zero_rate"] > threshold, "intermittent", "regular")
    bucket_std = df.groupby("bucket")["residual"].std(ddof=1)
    overall_std = df["residual"].std(ddof=1)
    bucket_std = bucket_std.fillna(overall_std)
    return df["bucket"].map(bucket_std).rename("lead_time_std")


def unit_cost_from_price(train: pd.DataFrame) -> pd.Series:
    """Mean training-window price per series, standing in for unit cost (docs/decision.md —
    Track A has no separate COGS field). Missing/all-NaN series fall back to the global mean.
    """
    per_series = train.groupby("series_id")["price"].mean()
    global_mean = train["price"].mean()
    return per_series.fillna(global_mean)


@dataclass(frozen=True)
class SimResult:
    holding_cost: float
    stockout_cost: float
    total_cost: float
    units_demanded: float
    units_shipped: float
    fill_rate: float
    avg_on_hand: float
    turns: float


def simulate_series(
    demand: np.ndarray,
    s: float,
    S: float,
    on_hand_start: float,
    lead_time_days: int,
    unit_cost: float,
    costs: CostParams,
) -> SimResult:
    """(s, S) continuous-review simulation over realised demand, lost-sales stockouts, a single
    outstanding order at a time — docs/decision.md "Cost simulation".
    """
    on_hand = on_hand_start
    pending: tuple[int, float] | None = None
    holding_cost = 0.0
    stockout_cost = 0.0
    shipped = 0.0
    demanded = 0.0
    on_hand_sum = 0.0
    n = len(demand)

    for day in range(n):
        if pending is not None and pending[0] == day:
            on_hand += pending[1]
            pending = None

        d = float(demand[day])
        demanded += d
        ship = min(on_hand, d)
        shipped += ship
        lost = d - ship
        stockout_cost += lost * costs.stockout_penalty_per_unit
        on_hand -= ship

        holding_cost += costs.holding_cost_rate * unit_cost * on_hand
        on_hand_sum += on_hand

        if on_hand < s and pending is None:
            qty = S - on_hand
            if qty > 0:
                pending = (day + lead_time_days, qty)

    avg_on_hand = on_hand_sum / n if n else 0.0
    fill_rate = shipped / demanded if demanded > 0 else 1.0
    turns = shipped / avg_on_hand if avg_on_hand > 0 else np.nan

    return SimResult(
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        total_cost=holding_cost + stockout_cost,
        units_demanded=demanded,
        units_shipped=shipped,
        fill_rate=fill_rate,
        avg_on_hand=avg_on_hand,
        turns=turns,
    )


def evaluate_fold_cost(
    panel: pd.DataFrame,
    fold: bt.Fold,
    model_name: str,
    costs: CostParams,
    lead_time_std_override: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """One model, one fold: fit -> forecast -> reorder decision -> (s, S) simulation over the
    fold's realised demand. Steady-state assumption: on_hand_start = reorder_point (see
    docs/decision.md) — every model is judged from the same "already following its own policy"
    starting point.

    `lead_time_std_override`, when given, replaces the quantile-derived safety-stock std with an
    empirical, pooled-by-intermittency-bucket std from an earlier fold's own realised forecast
    errors (see `lead_time_forecast_residuals`/`empirical_lead_time_std` — a model's self-reported
    P10/P90 width isn't comparable across model families, so sizing safety stock from it let a
    narrower-but-equally-calibrated model lose on cost for no good reason).
    A series missing from the override (or with a NaN bucket std) falls back to the quantile
    std so a small/incomplete calibration set never breaks the policy.

    Returns (row, detail, next_lead_time_std) — `next_lead_time_std` is this fold's own residual
    calibration, for the *next* fold to use as its override (chaining forward, never using a
    fold's own future outcomes to size its own safety stock).
    """
    train = panel[panel["date"] <= fold.train_end]
    test = panel[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)]
    horizon = (fold.test_end - fold.test_start).days + 1

    model = bt.MODEL_FACTORIES[model_name](horizon)
    model.fit(train)
    future_exog = test.drop(columns=["y"])
    preds = model.predict_quantiles(horizon, future_exog=future_exog)

    stats = lead_time_demand_stats(preds, costs.lead_time_days)
    if lead_time_std_override is not None:
        overridden = lead_time_std_override.reindex(stats.index)
        stats = stats.copy()
        stats["std"] = overridden.where(overridden.notna(), stats["std"])
    decisions = reorder_decision(stats, costs.service_level_target)
    unit_costs = unit_cost_from_price(train)
    global_unit_cost = unit_costs.mean()

    sim_rows = []
    for series_id, drow in decisions.iterrows():
        demand = test.loc[test["series_id"] == series_id].sort_values("date")["y"].to_numpy()
        if len(demand) == 0:
            continue
        s = float(drow["reorder_point"])
        if not np.isfinite(s):
            # A NaN/inf reorder_point (e.g. from a model producing a NaN forecast) makes
            # `on_hand < s` False on every day in simulate_series -- silently "never reorders"
            # rather than erroring or simulating something meaningful. Skip this series the same
            # way a series with no test-period demand is already skipped above, rather than let
            # a NaN silently masquerade as a valid (if oddly conservative) simulated outcome.
            continue
        unit_cost = float(unit_costs.get(series_id, global_unit_cost))
        result = simulate_series(
            demand,
            s=s,
            S=s,
            on_hand_start=s,
            lead_time_days=costs.lead_time_days,
            unit_cost=unit_cost,
            costs=costs,
        )
        sim_rows.append({"series_id": series_id, **result.__dict__})

    detail = pd.DataFrame(sim_rows)
    total_demanded = detail["units_demanded"].sum()
    total_shipped = detail["units_shipped"].sum()
    total_avg_on_hand = detail["avg_on_hand"].sum()
    row = {
        "fold": fold.index,
        "model": model_name,
        "n_series": len(detail),
        "holding_cost": detail["holding_cost"].sum(),
        "stockout_cost": detail["stockout_cost"].sum(),
        "total_cost": detail["total_cost"].sum(),
        "fill_rate": total_shipped / total_demanded if total_demanded > 0 else 1.0,
        "turns": total_shipped / total_avg_on_hand if total_avg_on_hand > 0 else np.nan,
    }

    residuals = lead_time_forecast_residuals(
        preds, test[["series_id", "date", "y"]], costs.lead_time_days
    )
    zero_rates = bt._intermittency(train)
    next_lead_time_std = empirical_lead_time_std(residuals, zero_rates)

    return row, detail, next_lead_time_std


def _bootstrap_lead_time_std(
    panel: pd.DataFrame, first_fold: bt.Fold, model_name: str, costs: CostParams
) -> pd.Series:
    """Fold-1 fallback: no prior fold exists yet to borrow residuals from, so this carves one
    calibration split out of the training window itself — fit on everything except the last
    `lead_time_days`, forecast that held-back stretch, measure residuals there. One extra fit
    per model up front, not one per fold (see docs/decision.md).
    """
    train = panel[panel["date"] <= first_fold.train_end]
    cutoff = train["date"].max() - pd.Timedelta(days=costs.lead_time_days - 1)
    calib_train = train[train["date"] < cutoff]
    calib_test = train[train["date"] >= cutoff]

    model = bt.MODEL_FACTORIES[model_name](costs.lead_time_days)
    model.fit(calib_train)
    preds = model.predict_quantiles(
        costs.lead_time_days, future_exog=calib_test.drop(columns=["y"])
    )
    residuals = lead_time_forecast_residuals(
        preds, calib_test[["series_id", "date", "y"]], costs.lead_time_days
    )
    zero_rates = bt._intermittency(calib_train)
    return empirical_lead_time_std(residuals, zero_rates)


def run_decision_backtest(
    panel: pd.DataFrame, costs: CostParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (results, per_series_detail) across every fold x model — same folds/sample/model
    set as backtest.run_backtest, so the two reports are directly comparable.

    Safety-stock sizing chains forward fold-to-fold per model: fold 1 uses a one-off bootstrap
    calibration (`_bootstrap_lead_time_std`), and every later fold uses the *previous* fold's own
    realised residuals — never a fold's own future outcomes (see `evaluate_fold_cost`).
    """
    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    folds = bt.make_folds(eval_panel, horizon=bt.HORIZON, n_folds=bt.N_FOLDS)

    prior_std = {
        model_name: _bootstrap_lead_time_std(eval_panel, folds[0], model_name, costs)
        for model_name in bt.MODEL_FACTORIES
    }

    results = []
    detail_frames = []
    for fold in folds:
        for model_name in bt.MODEL_FACTORIES:
            row, detail, next_std = evaluate_fold_cost(
                eval_panel, fold, model_name, costs, lead_time_std_override=prior_std[model_name]
            )
            prior_std[model_name] = next_std
            results.append(row)
            detail["fold"] = fold.index
            detail["model"] = model_name
            detail_frames.append(detail)

    results_df = pd.DataFrame(results)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    return results_df, detail_df


def write_decision_report(results: pd.DataFrame, costs: CostParams) -> str:
    metric_cols = ["holding_cost", "stockout_cost", "total_cost", "fill_rate", "turns"]
    summary = results.groupby("model")[metric_cols].mean().sort_values("total_cost").reset_index()
    per_fold = results.sort_values(["fold", "total_cost"])[
        ["fold", "model", *metric_cols]
    ].reset_index(drop=True)

    best_model = summary.iloc[0]["model"]
    best_cost = summary.iloc[0]["total_cost"]

    acceptance_lines: list[str] = []
    if "SeasonalNaive" in summary["model"].values:
        naive_cost = summary.loc[summary["model"] == "SeasonalNaive", "total_cost"].iloc[0]
        savings = naive_cost - best_cost
        acceptance_lines = [
            "",
            "## Phase 4 acceptance check",
            "",
            f"- Best model (**{best_model}**) mean simulated cost: ${best_cost:,.2f} per fold "
            f"vs. SeasonalNaive's ${naive_cost:,.2f} "
            f"({'PASS' if savings > 0 else 'FAIL'} — bar is lower than SeasonalNaive).",
            (
                f"- Savings: ${savings:,.2f} per fold "
                f"({savings / naive_cost:.1%} of SeasonalNaive's cost)."
                if naive_cost
                else ""
            ),
        ]

    date_str = dt.date.today().isoformat()
    lines = [
        f"# Decision report — {date_str}",
        "",
        f"Reorder policy simulated cost, same folds/models as the backtest report. "
        f"lead_time_days={costs.lead_time_days}, "
        f"service_level_target={costs.service_level_target:.0%}, "
        f"holding_cost_rate={costs.holding_cost_rate:.1%}, "
        f"stockout_penalty_per_unit=${costs.stockout_penalty_per_unit:.2f}.",
        "",
        "## Summary (mean total cost per fold, across folds)",
        "",
        bt._markdown_table(summary, float_cols=tuple(metric_cols)),
        "",
        "## Per-fold breakdown",
        "",
        bt._markdown_table(per_fold, float_cols=tuple(metric_cols)),
        *acceptance_lines,
    ]
    return "\n".join(lines)


def main() -> None:
    panel = pd.read_parquet(bt.PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    config = load_config()
    results, _detail = run_decision_backtest(panel, config.costs)
    report = write_decision_report(results, config.costs)

    bt.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = dt.date.today().isoformat()
    out_path = bt.REPORTS_DIR / f"decision_{date_str}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
