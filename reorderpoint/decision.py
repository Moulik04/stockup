"""Quantile forecast -> reorder point, safety stock, order quantity, cost simulation.

Newsvendor-style service-level policy, documented in docs/decision.md. Implemented Phase 4.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from reorderpoint import backtest as bt
from reorderpoint.config import CostParams, describe_holding_rate, load_config
from reorderpoint.grain import GRAIN

Z80 = norm.ppf(0.9)  # P10/P90 is the central 80% interval everywhere in this project

RESULTS_PATH = bt.PANEL_PATH.parent / "decision_results.parquet"
DETAIL_PATH = bt.PANEL_PATH.parent / "decision_detail.parquet"


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


def calibrated_reorder_decision(
    lead_time_stats: pd.DataFrame, safety_stock: pd.Series, service_level: float
) -> pd.DataFrame:
    """Reorder point from a *calibrated* per-series safety stock (`safety_stock.py`) instead of
    `z * std` off the model's own interval width.

    The single sizing rule shared by the backtest harness (`ablate_safety_stock.run_cell`) and the
    served path (`serve.size_decisions`), so they cannot drift apart. A series the calibration
    never saw falls back to the quantile-derived buffer, so a thin calibration degrades to the
    old method for that series instead of dropping it from the policy."""
    fallback = reorder_decision(lead_time_stats, service_level)["safety_stock"]
    safety = safety_stock.reindex(lead_time_stats.index)
    out = lead_time_stats.copy()
    out["safety_stock"] = safety.where(safety.notna(), fallback)
    out["reorder_point"] = out["mean"] + out["safety_stock"]
    return out


def order_quantity(
    reorder_point: pd.Series, on_hand: pd.Series, order_up_to: pd.Series | None = None
) -> pd.Series:
    """Order up to S when on-hand is below the reorder point s — docs/decision.md step 3.

    `order_up_to=None` is the original `S = s` (order the deficit below s). With `S > s` the order
    is `S - on_hand`, placed only while on-hand is below s."""
    if order_up_to is None:
        return (reorder_point - on_hand).clip(lower=0)
    return (order_up_to - on_hand).where(on_hand < reorder_point, 0.0).clip(lower=0)


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
    forecast: pd.DataFrame,
    on_hand: pd.Series,
    lead_time_days: int,
    service_level: float,
    safety_stock: pd.Series | None = None,
    lot_multiple: float = 0.0,
) -> pd.DataFrame:
    """Full per-SKU policy: mean, std, safety_stock, reorder_point, on_hand, order_quantity,
    stockout_probability, rationale — one row per series_id.

    `safety_stock=None` sizes from the model's own P10/P90 width (`z * std`) — the method Phase 4
    showed loses, kept for the fallback and for comparison. Pass the per-series calibrated buffer
    to size the way the backtest harness does (see `calibrated_reorder_decision`).
    """
    stats = lead_time_demand_stats(forecast, lead_time_days)
    if safety_stock is None:
        out = reorder_decision(stats, service_level)
    else:
        out = calibrated_reorder_decision(stats, safety_stock, service_level)
        z = norm.ppf(service_level)
        if z > 0:
            # the stockout probability needs a spread; the one implied by the calibrated buffer,
            # so it is exactly 1 - service_level when on-hand sits at the reorder point
            out["std"] = out["safety_stock"] / z
    out = out.reindex(on_hand.index)
    out["on_hand"] = on_hand
    # `lot_multiple` sets S = s + lot_multiple x E[lead-time demand] (0: the original S = s), the
    # same rule `_simulate_policy` simulates
    out["order_up_to"] = out["reorder_point"] + lot_multiple * out["mean"].clip(lower=0)
    out["order_quantity"] = order_quantity(
        out["reorder_point"], out["on_hand"], out["order_up_to"] if lot_multiple else None
    )
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


def standard_loss(k: float | np.ndarray) -> float | np.ndarray:
    """Standardised normal loss function G(k) = φ(k) − k·(1 − Φ(k)).

    Expected shortfall, in std units, of a normal lead-time demand against a reorder point k
    std above its mean. Strictly decreasing in k, →0 as k→∞. This is the quantity that connects
    safety stock to *fill rate*, where `z = Φ⁻¹(service_level)` connects it to cycle service
    level — the two are different targets (see docs/decision.md, "Two service measures").
    """
    k = np.asarray(k, dtype=float)
    return norm.pdf(k) - k * (1 - norm.cdf(k))


def fill_rate_k(target_fill_rate: float, order_quantity: float, lead_time_std: float) -> float:
    """Safety factor k that delivers `target_fill_rate` on lead-time demand, by inverting

        fill_rate = 1 − σ_LT · G(k) / Q      =>      G(k) = (1 − fill_rate) · Q / σ_LT

    `order_quantity` is the expected demand per replenishment cycle — the Q in that formula.
    Because this project sets S = s (docs/decision.md step 3), Q is only the deficit consumed
    since the last order, which is *small*; small Q demands a larger k for the same fill rate,
    which is precisely the mechanism that makes a 95% cycle service level deliver a far lower
    fill rate here.

    Returns a k clamped to [-6, 6]: beyond that the normal approximation carries no information,
    and an unclamped solve on a series with σ_LT ≫ Q would return an arbitrarily large number.
    """
    if not 0 < target_fill_rate < 1:
        raise ValueError(f"target_fill_rate must be in (0, 1), got {target_fill_rate}")
    if not np.isfinite(lead_time_std) or lead_time_std <= 0:
        # No spread to protect against: any k gives the same (deterministic) outcome.
        return 0.0
    if not np.isfinite(order_quantity) or order_quantity <= 0:
        # No demand per cycle to fill; nothing for a fill-rate target to bite on.
        return 0.0

    required_loss = (1 - target_fill_rate) * order_quantity / lead_time_std
    lo, hi = -6.0, 6.0
    if required_loss >= standard_loss(lo):
        return lo
    if required_loss <= standard_loss(hi):
        return hi
    return float(brentq(lambda k: standard_loss(k) - required_loss, lo, hi))


def reorder_decision_fill_rate(
    lead_time_stats: pd.DataFrame, target_fill_rate: float
) -> pd.DataFrame:
    """Fill-rate-targeted counterpart to `reorder_decision`: sizes safety stock from the loss
    function so the *fraction of demand met* hits the target, rather than the probability of
    surviving a cycle. Q is estimated per series as its expected lead-time demand.
    """
    out = lead_time_stats.copy()
    k = [
        fill_rate_k(target_fill_rate, order_quantity=row.mean, lead_time_std=row.std)
        for row in out.itertuples()
    ]
    out["k"] = k
    out["safety_stock"] = out["k"] * out["std"]
    out["reorder_point"] = out["mean"] + out["safety_stock"]
    return out


POLICIES = {"csl": reorder_decision, "fill_rate": reorder_decision_fill_rate}


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
    cycles: int
    stockout_cycles: int
    orders: int
    order_qty_total: float
    order_qty_sq_total: float
    orders_below_one_unit: int


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

    Also counts **replenishment cycles** for the realised cycle service level: a cycle opens when
    an order is placed and closes when it lands, and counts as a stockout cycle if any demand
    goes unmet in between. That window — after the order is placed, before it arrives — is
    exactly the exposure `z · σ_LT` is sized to protect, so it is the like-for-like counterpart
    to the `service_level_target` the policy is given. Lost sales outside any open cycle (a
    series whose reorder point is so low it never triggers an order) are still costed and still
    depress fill rate, but belong to no cycle and so cannot appear in the CSL — a real asymmetry
    between the two measures, not a bookkeeping choice.
    """
    on_hand = on_hand_start
    pending: tuple[int, float] | None = None
    holding_cost = 0.0
    stockout_cost = 0.0
    shipped = 0.0
    demanded = 0.0
    on_hand_sum = 0.0
    cycles = 0
    stockout_cycles = 0
    cycle_open = False
    cycle_had_stockout = False
    # Realised order quantities. The fill-rate formula treats Q as a fixed lot size; under S = s
    # it is actually the variable undershoot below the reorder point, so its dispersion is
    # measured rather than assumed (see docs/decision.md, "How well does Q behave?").
    orders = 0
    order_qty_total = 0.0
    order_qty_sq_total = 0.0
    orders_below_one_unit = 0
    n = len(demand)

    for day in range(n):
        if pending is not None and pending[0] == day:
            on_hand += pending[1]
            pending = None
            if cycle_open:
                cycles += 1
                stockout_cycles += int(cycle_had_stockout)
                cycle_open = False
                cycle_had_stockout = False

        d = float(demand[day])
        demanded += d
        ship = min(on_hand, d)
        shipped += ship
        lost = d - ship
        if lost > 0 and cycle_open:
            cycle_had_stockout = True
        stockout_cost += lost * costs.stockout_penalty_per_unit
        on_hand -= ship

        holding_cost += costs.holding_cost_rate * unit_cost * on_hand
        on_hand_sum += on_hand

        if on_hand < s and pending is None:
            qty = S - on_hand
            if qty > 0:
                pending = (day + lead_time_days, qty)
                cycle_open = True
                cycle_had_stockout = False
                orders += 1
                order_qty_total += qty
                order_qty_sq_total += qty * qty
                orders_below_one_unit += int(qty < 1.0)

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
        cycles=cycles,
        stockout_cycles=stockout_cycles,
        orders=orders,
        order_qty_total=order_qty_total,
        order_qty_sq_total=order_qty_sq_total,
        orders_below_one_unit=orders_below_one_unit,
    )


def evaluate_fold_cost(
    panel: pd.DataFrame,
    fold: bt.Fold,
    model_name: str,
    costs: CostParams,
    lead_time_std_override: pd.Series | None = None,
    policy: str = "csl",
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """One model, one fold: fit -> forecast -> reorder decision -> (s, S) simulation over the
    fold's realised demand. Steady-state assumption: on_hand_start = reorder_point (see
    docs/decision.md) — every model is judged from the same "already following its own policy"
    starting point.

    `policy` selects how safety stock is sized from the same lead-time stats: `"csl"` is the
    original `z = Φ⁻¹(service_level_target)` cycle-service-level policy, `"fill_rate"` targets
    the same number as a *fill rate* through the loss function. Both read
    `costs.service_level_target`; they differ in what they take that number to mean.

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
    rows, details, next_std = evaluate_fold_cost_policies(
        panel, fold, model_name, costs, lead_time_std_override, policies=(policy,)
    )
    return rows[0], details[0], next_std


def evaluate_fold_cost_policies(
    panel: pd.DataFrame,
    fold: bt.Fold,
    model_name: str,
    costs: CostParams,
    lead_time_std_override: pd.Series | None = None,
    policies: tuple[str, ...] = ("csl",),
    lot_multiples: tuple[float, ...] = (0.0,),
) -> tuple[list[dict], list[pd.DataFrame], pd.Series]:
    """`evaluate_fold_cost` for several policies (and, orthogonally, several order-up-to lot
    multiples) at once, fitting the model only once.

    The fit and the forecast are what cost real time here — for a deep model like NBEATS, the only
    real time. Neither the sizing policy nor `lot_multiple` changes the forecast, only how it is
    turned into a reorder point / order-up-to level, so every (policy, lot) combination is
    simulated off the one shared fit. Returns `len(policies) x len(lot_multiples)` rows/details, in
    that nested order (policy outer, lot inner) — a caller wanting a single (policy, lot) still
    gets exactly one row back.
    """
    train = panel[panel["date"] <= fold.train_end]
    test = panel[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)]
    horizon = GRAIN.periods_in(fold.test_start, fold.test_end)

    model = bt.MODEL_FACTORIES[model_name](horizon)
    model.fit(train)
    future_exog = test.drop(columns=["y"])
    preds = model.predict_quantiles(horizon, future_exog=future_exog)

    stats = lead_time_demand_stats(preds, costs.lead_time_days)
    if lead_time_std_override is not None:
        overridden = lead_time_std_override.reindex(stats.index)
        stats = stats.copy()
        stats["std"] = overridden.where(overridden.notna(), stats["std"])
    unit_costs = unit_cost_from_price(train)
    global_unit_cost = unit_costs.mean()

    demand_by_series = {
        series_id: group.sort_values("date")["y"].to_numpy()
        for series_id, group in test.groupby("series_id", sort=False)
    }

    rows, details = [], []
    for policy in policies:
        decisions = POLICIES[policy](stats, costs.service_level_target)
        for lot in lot_multiples:
            row, detail = _simulate_policy(
                decisions,
                demand_by_series,
                unit_costs,
                global_unit_cost,
                costs,
                fold,
                model_name,
                policy,
                lot_multiple=lot,
            )
            rows.append(row)
            details.append(detail)

    residuals = lead_time_forecast_residuals(
        preds, test[["series_id", "date", "y"]], costs.lead_time_days
    )
    zero_rates = bt._intermittency(train)
    next_lead_time_std = empirical_lead_time_std(residuals, zero_rates)

    return rows, details, next_lead_time_std


def _simulate_policy(
    decisions: pd.DataFrame,
    demand_by_series: dict[str, np.ndarray],
    unit_costs: pd.Series,
    global_unit_cost: float,
    costs: CostParams,
    fold: bt.Fold,
    model_name: str,
    policy: str,
    lot_multiple: float = 0.0,
    start_fraction: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    """Simulate every series' policy over its test window.

    `lot_multiple` sets the order-up-to level `S = s + Q` with `Q = lot_multiple x` the series'
    expected lead-time demand (`decisions["mean"]`). 0 is the original `S = s`. The trigger stays
    "on-hand < s, no order in transit"; only how much is ordered changes. A series starts the
    window at `s + start_fraction x Q`: 0.5 (the default) is the steady-state average of its cycle
    (`s + Q/2`), 0 starts at the reorder point, 1 at the order-up-to level. It is `s` whatever the
    fraction when Q = 0, so the original policy is reproduced exactly."""
    sim_rows = []
    for series_id, drow in decisions.iterrows():
        demand = demand_by_series.get(series_id)
        if demand is None or len(demand) == 0:
            continue
        s = float(drow["reorder_point"])
        if not np.isfinite(s):
            # A NaN/inf reorder_point (e.g. from a model producing a NaN forecast) makes
            # `on_hand < s` False on every day in simulate_series -- silently "never reorders"
            # rather than erroring or simulating something meaningful. Skip this series the same
            # way a series with no test-period demand is already skipped above, rather than let
            # a NaN silently masquerade as a valid (if oddly conservative) simulated outcome.
            continue
        # `unit_costs` holds the mean training-window *price* per series; the economics turn it
        # into the value stock is held at and the cost of a lost sale (config.CostParams).
        price = float(unit_costs.get(series_id, global_unit_cost))
        lot = lot_multiple * max(float(drow["mean"]), 0.0) if lot_multiple else 0.0
        start = s + start_fraction * lot
        result = simulate_series(
            demand,
            s=s,
            S=s + lot,
            on_hand_start=start,
            lead_time_days=costs.lead_time_days,
            unit_cost=costs.unit_cost(price),
            costs=replace(costs, stockout_penalty_per_unit=costs.stockout_penalty(price)),
        )
        sim_rows.append(
            {
                "series_id": series_id,
                "unit_price": price,
                **result.__dict__,
                "lead_time_std": float(drow["std"]),
                "reorder_point": s,
                "order_up_to": s + lot,
                # the exact starting stock, so a replica of the simulation can reproduce it to the
                # last bit (a recomputed (s + S) / 2 can differ and flip a tie on `on_hand < s`)
                "on_hand_start": start,
            }
        )

    detail = pd.DataFrame(sim_rows)
    # Stamp the rate this run was simulated at. Holding cost is linear in it and the policy never
    # reads it, so a stored run can be re-priced exactly — but only against the rate it actually
    # used, which the config default no longer tells you (holding_breakeven.simulated_rate).
    detail["holding_cost_rate"] = costs.holding_cost_rate
    # ...and the economics it was priced under, so `economics.reprice` needs no assumptions.
    detail["gross_margin"] = np.nan if costs.gross_margin is None else costs.gross_margin
    detail["stockout_penalty_flat"] = costs.stockout_penalty_per_unit
    detail["lot_multiple"] = lot_multiple
    total_demanded = detail["units_demanded"].sum()
    total_shipped = detail["units_shipped"].sum()
    total_avg_on_hand = detail["avg_on_hand"].sum()
    total_cycles = detail["cycles"].sum()
    row = {
        "fold": fold.index,
        "model": model_name,
        "policy": policy,
        "n_series": len(detail),
        "holding_cost": detail["holding_cost"].sum(),
        "stockout_cost": detail["stockout_cost"].sum(),
        "total_cost": detail["total_cost"].sum(),
        "fill_rate": total_shipped / total_demanded if total_demanded > 0 else 1.0,
        # pooled over cycles, not averaged over series, for the same reason fill_rate pools over
        # units: a series with one cycle shouldn't weigh as much as one with a dozen.
        "cycle_service_level": (
            1 - detail["stockout_cycles"].sum() / total_cycles if total_cycles > 0 else np.nan
        ),
        "cycles": int(total_cycles),
        "turns": total_shipped / total_avg_on_hand if total_avg_on_hand > 0 else np.nan,
        # The realised safety factor: safety_stock / σ_LT, averaged over series. For the CSL
        # policy this is just z; for the fill-rate policy it varies per series with σ_LT/Q, and
        # comparing the two is what says which policy actually provisions more.
        "mean_safety_factor": _mean_safety_factor(decisions),
        **_order_quantity_stats(detail),
    }
    return row, detail


def _mean_safety_factor(decisions: pd.DataFrame) -> float:
    std = decisions["std"].replace(0, np.nan)
    return float((decisions["safety_stock"] / std).mean(skipna=True))


def _order_quantity_stats(detail: pd.DataFrame) -> dict[str, float]:
    """Realised Q: the undershoot actually ordered, pooled over every order placed.

    The fill-rate relationship `β = 1 − σ_LT·G(k)/Q` is derived for a *fixed* lot size Q. Under
    S = s there is no lot size — Q is whatever deficit accumulated since the last order — so how
    badly that assumption is violated is an empirical question, and these are the numbers that
    answer it. `q_cv` is the coefficient of variation of Q across orders; a CV anywhere near 1
    means the formula's Q is a summary of a distribution too spread out for a single number to
    stand in for.
    """
    total_orders = float(detail["orders"].sum())
    if total_orders <= 0:
        return {
            "q_mean": np.nan,
            "q_cv": np.nan,
            "q_frac_below_one": np.nan,
            "sigma_over_q": np.nan,
        }

    q_mean = detail["order_qty_total"].sum() / total_orders
    q_var = detail["order_qty_sq_total"].sum() / total_orders - q_mean**2
    q_cv = np.sqrt(max(q_var, 0.0)) / q_mean if q_mean > 0 else np.nan

    ordered = detail[detail["orders"] > 0]
    per_series_q = ordered["order_qty_total"] / ordered["orders"]
    ratio = (ordered["lead_time_std"] / per_series_q.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    return {
        "q_mean": float(q_mean),
        "q_cv": float(q_cv),
        "q_frac_below_one": float(detail["orders_below_one_unit"].sum() / total_orders),
        # median, not mean: a handful of near-zero-Q series would otherwise dominate the average
        "sigma_over_q": float(ratio.median(skipna=True)),
    }


def _bootstrap_lead_time_std(
    panel: pd.DataFrame, first_fold: bt.Fold, model_name: str, costs: CostParams
) -> pd.Series:
    """Fold-1 fallback: no prior fold exists yet to borrow residuals from, so this carves one
    calibration split out of the training window itself — fit on everything except the last
    `lead_time_days`, forecast that held-back stretch, measure residuals there. One extra fit
    per model up front, not one per fold (see docs/decision.md).
    """
    train = panel[panel["date"] <= first_fold.train_end]
    cutoff = train["date"].max() - GRAIN.period * (costs.lead_time_days - 1)
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
    panel: pd.DataFrame, costs: CostParams, lot_multiples: tuple[float, ...] = (0.0,)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (results, per_series_detail) across every fold x model — same folds/sample/model
    set as backtest.run_backtest, so the two reports are directly comparable.

    Safety-stock sizing chains forward fold-to-fold per model: fold 1 uses a one-off bootstrap
    calibration (`_bootstrap_lead_time_std`), and every later fold uses the *previous* fold's own
    realised residuals — never a fold's own future outcomes (see `evaluate_fold_cost`).

    `lot_multiples` sets `S = s + m x` expected lead-time demand for every model in this run,
    for each `m` given ((0.0,), the default: only the original `S = s`). This is the only argument
    `evaluate_fold_cost_policies` doesn't need a fresh fit for, so passing more than one value —
    e.g. to compare `S = s` against `S = s + 2 x lead-time demand` for a model expensive to fit
    (NBEATS) — costs nothing beyond the extra resimulation.
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
            # Both policies see the identical forecast and the identical residual calibration —
            # they differ only in how safety stock is sized from it, which is the comparison.
            # The chained std comes from the model's residuals, not from any policy's outcome,
            # so it's the same for both and is advanced once per model per fold.
            rows, details, next_std = evaluate_fold_cost_policies(
                eval_panel,
                fold,
                model_name,
                costs,
                lead_time_std_override=prior_std[model_name],
                policies=tuple(POLICIES),
                lot_multiples=lot_multiples,
            )
            # nested order from evaluate_fold_cost_policies: policy outer, lot_multiple inner
            combos = [(policy, lot) for policy in POLICIES for lot in lot_multiples]
            for row, detail, (policy, lot) in zip(rows, details, combos, strict=True):
                row["lot_multiple"] = lot
                results.append(row)
                detail["fold"] = fold.index
                detail["model"] = model_name
                detail["policy"] = policy
                detail_frames.append(detail)
            prior_std[model_name] = next_std

    results_df = pd.DataFrame(results)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    return results_df, detail_df


def write_decision_report(results: pd.DataFrame, costs: CostParams) -> str:
    metric_cols = [
        "holding_cost",
        "stockout_cost",
        "total_cost",
        "fill_rate",
        "cycle_service_level",
        "turns",
    ]
    if "policy" not in results.columns:
        results = results.assign(policy="csl")

    csl_results = results[results["policy"] == "csl"]
    summary = (
        csl_results.groupby("model")[metric_cols].mean().sort_values("total_cost").reset_index()
    )
    per_fold = csl_results.sort_values(["fold", "total_cost"])[
        ["fold", "model", *metric_cols]
    ].reset_index(drop=True)

    best_model = summary.iloc[0]["model"]
    best_cost = summary.iloc[0]["total_cost"]

    service_lines = _service_measure_section(results, costs)

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
        f"holding_cost_rate={describe_holding_rate(costs.holding_cost_rate)}, "
        f"stockout penalty="
        + (
            f"${costs.stockout_penalty_per_unit:.2f}/unit (flat)"
            if costs.gross_margin is None
            else f"{costs.gross_margin:.1%} of price (lost gross margin)"
        )
        + ".",
        "",
        "Two service measures are reported throughout: **fill rate** (units shipped ÷ units "
        "demanded) and **cycle service level** (share of replenishment cycles with no stockout "
        "between placing an order and receiving it). The policy's `service_level_target` is a "
        "cycle service level, so CSL is the measure it is actually aiming at; fill rate is what "
        "the business feels. They are different quantities and a 95% CSL does not imply a 95% "
        "fill rate.",
        "",
        "## Summary — CSL policy (mean per fold, across folds)",
        "",
        bt._markdown_table(summary, float_cols=tuple(metric_cols)),
        "",
        "## Per-fold breakdown — CSL policy",
        "",
        bt._markdown_table(per_fold, float_cols=tuple(metric_cols)),
        "",
        *service_lines,
        *acceptance_lines,
    ]
    return "\n".join(lines)


def _service_measure_section(results: pd.DataFrame, costs: CostParams) -> list[str]:
    """Fill rate vs cycle service level, and the CSL policy vs the fill-rate-targeted one.

    The conclusion sentence is derived from the measured gap rather than written once, so a
    rerun that lands somewhere else has to say so.
    """
    target = costs.service_level_target
    agg_cols = [
        c
        for c in [
            "fill_rate",
            "cycle_service_level",
            "total_cost",
            "holding_cost",
            "stockout_cost",
            "mean_safety_factor",
        ]
        if c in results.columns
    ]
    by_policy = results.groupby(["policy", "model"])[agg_cols].mean()

    csl_rows = by_policy.loc["csl"].sort_values("total_cost")
    mean_fill = csl_rows["fill_rate"].mean()
    mean_csl = csl_rows["cycle_service_level"].mean()

    lines = [
        "## Fill rate vs cycle service level",
        "",
        f"Under the CSL policy (`z = Φ⁻¹({target:.2f})`), averaged across models: realised "
        f"**cycle service level {mean_csl:.1%}**, realised **fill rate {mean_fill:.1%}**, against "
        f"a {target:.0%} target.",
        "",
        bt._markdown_table(
            csl_rows.reset_index()[
                ["model", "cycle_service_level", "fill_rate", "total_cost"]
            ].rename(columns={"total_cost": "mean_total_cost"}),
            float_cols=("cycle_service_level", "fill_rate", "mean_total_cost"),
        ),
        "",
    ]

    csl_gap = mean_csl - target
    fill_gap = mean_fill - target
    if abs(csl_gap) <= 0.05 and fill_gap < -0.05:
        lines += [
            f"**The policy is hitting its own target; the fill-rate shortfall is a metric "
            f"mismatch.** CSL lands within 5 points of the {target:.0%} target while fill rate "
            f"sits {abs(fill_gap) * 100:.0f} percentage points below it. Reading that gap as "
            f"'every model misses the service-level target' measures the policy against a "
            f"quantity it was never sizing for.",
        ]
    elif csl_gap < -0.05:
        lines += [
            f"**The policy misses its own target.** Realised CSL is {abs(csl_gap) * 100:.0f} "
            f"percentage points "
            f"below the {target:.0%} it was sized for, so the under-provisioning reading is "
            f"correct on its own terms — this is not only a metric mismatch.",
        ]
    else:
        lines += [
            f"Realised CSL ({mean_csl:.1%}) and fill rate ({mean_fill:.1%}) both sit near the "
            f"{target:.0%} target; neither measure shows a systematic shortfall.",
        ]
    lines += [""]

    if "fill_rate" in results["policy"].unique():
        fr_rows = by_policy.loc["fill_rate"]
        compare = (
            csl_rows[["fill_rate", "cycle_service_level", "total_cost"]]
            .join(
                fr_rows[["fill_rate", "cycle_service_level", "total_cost"]],
                lsuffix="_csl",
                rsuffix="_fillrate",
            )
            .reset_index()
        )
        mean_fr_fill = fr_rows["fill_rate"].mean()
        mean_fr_cost = fr_rows["total_cost"].mean()
        mean_csl_cost = csl_rows["total_cost"].mean()
        cost_delta = mean_fr_cost - mean_csl_cost
        z = norm.ppf(target)
        fr_k = (
            by_policy.loc["fill_rate", "mean_safety_factor"].mean()
            if "mean_safety_factor" in by_policy.columns
            else np.nan
        )
        direction = "less" if fr_k < z else "more"
        lines += [
            "## Second policy: target the fill rate directly",
            "",
            f"Same forecasts, same residual calibration, same simulation — the only change is "
            f"that safety stock is sized from the loss function to hit a {target:.0%} *fill "
            f"rate* (`G(k) = (1 − FR) · Q / σ_LT`, with Q the expected demand per cycle) instead "
            f"of a {target:.0%} cycle service level.",
            "",
            bt._markdown_table(
                compare, float_cols=tuple(c for c in compare.columns if c != "model")
            ),
            "",
            f"Averaged across models, targeting fill rate directly moves realised fill rate "
            f"{mean_fill:.1%} → {mean_fr_fill:.1%} and mean total cost "
            f"${mean_csl_cost:,.0f} → ${mean_fr_cost:,.0f} per fold "
            f"({cost_delta:+,.0f}, {cost_delta / mean_csl_cost:+.1%}).",
            "",
            f"**Why it lands where it does.** The fill-rate policy's mean realised safety factor "
            f"is {fr_k:.2f}, against the CSL policy's z = {z:.2f} — it provisions **{direction}** "
            f"stock, because at this panel's σ_LT/Q the loss function says a {target:.0%} fill "
            f"rate is the {'weaker' if fr_k < z else 'stronger'} of the two targets. The formula "
            f"predicts {target:.0%}; the simulation delivers {mean_fr_fill:.1%}. That "
            f"{(target - mean_fr_fill) * 100:.0f}-percentage-point gap between promise and "
            f"delivery is the finding — but see the next section before attributing all of it to "
            f"distributional shape.",
            "",
            "Both policies are kept and both are reported — a negative result on the more "
            "theoretically correct policy is the informative part, not something to tune away. "
            "The CSL policy stays the default for `make decide` and the serving path.",
            "",
            *_order_quantity_section(results, target, mean_fr_fill),
        ]
    return lines


def _order_quantity_section(results: pd.DataFrame, target: float, mean_fr_fill: float) -> list[str]:
    """How badly the fixed-lot-size assumption behind the fill-rate formula is violated here."""
    if "q_cv" not in results.columns:
        return []
    fr = results[results["policy"] == "fill_rate"]
    q_mean = fr["q_mean"].mean()
    q_cv = fr["q_cv"].mean()
    q_low = fr["q_frac_below_one"].mean()
    ratio = fr["sigma_over_q"].mean()

    dispersed = q_cv >= 0.5
    spread = "more than its own mean" if q_cv >= 1 else "a large fraction of its own mean"
    sub_unit = (
        f", and {q_low:.0%} of orders are for less than a single unit" if q_low >= 0.005 else ""
    )
    verdict = (
        (
            f"Q is **not** a well-behaved lot size here: its coefficient of variation across "
            f"orders is {q_cv:.2f}{sub_unit}. "
            f"A formula derived for a fixed Q is being handed a quantity that varies by "
            f"{spread}, so **part of the "
            f"{(target - mean_fr_fill) * 100:.0f}-point miss is misapplication of the formula, "
            f"not evidence about the shape of the demand distribution.** How much of it splits "
            f"which way is not separable from these runs alone."
        )
        if dispersed
        else (
            f"Q is reasonably well-behaved (CV {q_cv:.2f}), so the fixed-lot-size assumption is "
            f"not badly violated and the "
            f"{(target - mean_fr_fill) * 100:.0f}-point miss is better attributed to "
            f"distributional shape than to formula misapplication."
        )
    )

    return [
        "### How well does Q behave?",
        "",
        "`β = 1 − σ_LT·G(k)/Q` is derived for a **fixed lot size** Q. This project sets S = s, so "
        "there is no lot size — Q is whatever deficit accumulated since the last order. Measured "
        "over every order the fill-rate policy placed:",
        "",
        f"- mean realised Q: **{q_mean:.2f} units**",
        f"- coefficient of variation of Q: **{q_cv:.2f}**",
        f"- orders for less than one unit: **{q_low:.0%}**",
        f"- median realised σ_LT/Q: **{ratio:.2f}**",
        "",
        verdict,
        "",
        "The cycle-service-level result in the previous section is unaffected by any of this — "
        "`z · σ_LT` makes no lot-size assumption — and remains the primary evidence that the "
        "policy misses the target it was sized for.",
        "",
    ]


def main() -> None:
    panel = pd.read_parquet(bt.PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    config = load_config()
    results, detail = run_decision_backtest(panel, config.costs)

    # Per-series cost detail is expensive to regenerate (one model fit per model per fold) and is
    # what a paired-bootstrap CI over series needs, so it's persisted rather than discarded.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_PATH, index=False)
    detail.to_parquet(DETAIL_PATH, index=False)
    print(f"Wrote {RESULTS_PATH} and {DETAIL_PATH}")

    report = write_decision_report(results, config.costs)

    bt.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = dt.date.today().isoformat()
    out_path = bt.REPORTS_DIR / f"decision_{date_str}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
