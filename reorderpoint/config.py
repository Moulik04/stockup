"""Central config: data track switch, paths, cost parameters — all from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

DAYS_PER_YEAR = 365

# Track A holding-cost assumption, as an ANNUAL fraction of unit cost. Source: Goetschalckx,
# "Logistics Systems Design", ch. 5 "Inventory Systems" (Georgia Tech course notes, 2002): "A
# holding cost rate of 25% of the unit value per year is a widely quoted average for the US
# industry." That is a cross-industry average, not a retail-specific measurement — see
# docs/decision.md, "Holding cost rate", for what it does and does not establish. Track B (UCI
# Online Retail II) has no cost data either, so it uses this same rate.
DEFAULT_ANNUAL_HOLDING_RATE = 0.25

# The range this project treats as plausible for a carrying cost, used to shade the cost-vs-rate
# chart. The 25% above is the only figure in it that has a named source; the bounds are the
# project's working range (docs/decision.md). Where inside it the rate sits does not change the
# LightGBM/SeasonalNaive ranking (crossover ~132%/yr at the default target); the lost-sale cost and
# the service target do (reports/penalty_sensitivity_*.md).
PLAUSIBLE_ANNUAL_HOLDING_RATE = (0.15, 0.30)


# Track A stockout economics: a lost sale costs the unit's gross margin (price minus cost), and the
# inventory the holding rate applies to is valued at cost, (1 - margin) x price. One parameter ties
# both sides of the trade to the same unit. Source: Walmart Inc., Form 10-K for the fiscal year
# ended 2026-01-31 (SEC accession 0000104169-26-000055), Walmart U.S. segment: net sales
# $482,975M, gross profit $132,615M (net sales less cost of sales) -> gross profit rate 27.5%
# (27.2% and 26.8% the two prior years). The M5 data are Walmart U.S. stores, so the U.S. segment is
# the matching figure; the consolidated rate (24.2%, net sales $706,413M, cost of sales $535,395M)
# blends in Sam's Club and International and is kept only as a sweep point. Neither is the hobby
# category, Walmart notes its cost of sales omits some distribution costs, and gross margin is a
# *lower bound* on what a stockout costs — lost goodwill, and demand that does not come back, are
# extra — which is why reports sweep the margin rather than trust this value (docs/decision.md).
# Order-up-to level: S = s + Q, Q = this x expected lead-time demand. 2.0 was chosen by a rule
# fixed before the results were read (reports/order_up_to_*.md, DECISIONS.md): the multiple with
# the lowest cost at its own cost-optimal target, its CI against S = s excluding zero. 0 is the
# original S = s, which the historical reports keep (they pin lot_multiple=0 explicitly). The
# simulation prices no fixed cost per order, so this is the *service* case for a larger lot, not
# an EOQ.
DEFAULT_LOT_MULTIPLE = 2.0
DEFAULT_GROSS_MARGIN = 0.275
CONSOLIDATED_GROSS_MARGIN = (
    0.242  # Walmart Inc. consolidated, FY2026; a sweep point, not the default
)


def annual_to_daily(annual_rate: float) -> float:
    """The simulation accrues holding per day (decision.simulate_series); simple, not compounded."""
    return annual_rate / DAYS_PER_YEAR


def daily_to_annual(daily_rate: float) -> float:
    return daily_rate * DAYS_PER_YEAR


def describe_holding_rate(daily_rate: float) -> str:
    """'0.0685%/day (25.0%/yr)' — a bare '0.1%' hides what the unit is and rounds away the value."""
    return f"{daily_rate:.4%}/day ({daily_to_annual(daily_rate):.1%}/yr)"


@dataclass(frozen=True)
class CostParams:
    # Fraction of unit cost charged per DAY on hand (accrued daily in decision.simulate_series).
    # Default is DEFAULT_ANNUAL_HOLDING_RATE / 365. Until 2026-09-20 it was 0.02 (~730%/yr) — see
    # .env.example and reports/holding_breakeven_*.md.
    holding_cost_rate: float
    # Flat $/unit lost sale. Used only when `gross_margin` is None — the pre-2026-09-20 model, kept
    # so old runs can be reproduced and re-priced; not sourced (see docs/decision.md).
    stockout_penalty_per_unit: float
    lead_time_days: int
    service_level_target: float
    # Unit economics. When set: a lost sale costs `gross_margin x price` and stock is valued at
    # `(1 - gross_margin) x price`. When None: flat penalty, and stock is valued at price itself.
    gross_margin: float | None = None
    # Order-up-to level above the reorder point, as a multiple of expected lead-time demand:
    # S = s + lot_multiple x E[lead-time demand]. 0 is the original S = s. See order_up_to.py.
    lot_multiple: float = 0.0

    def unit_cost(self, price):
        """Per-unit value the holding rate applies to."""
        return price if self.gross_margin is None else price * (1 - self.gross_margin)

    def stockout_penalty(self, price):
        """Per-unit cost of a lost sale."""
        return (
            self.stockout_penalty_per_unit
            if self.gross_margin is None
            else (self.gross_margin * price)
        )

    def flat(self) -> CostParams:
        """The same parameters priced the legacy way (flat penalty, cost = price)."""
        return replace(self, gross_margin=None)


@dataclass(frozen=True)
class Config:
    track: str  # "a" or "b"
    costs: CostParams


def load_config() -> Config:
    track = os.environ.get("REORDERPOINT_TRACK", "a").lower()
    if track not in {"a", "b"}:
        raise ValueError(f"REORDERPOINT_TRACK must be 'a' or 'b', got {track!r}")

    flat_penalty = os.environ.get("STOCKOUT_PENALTY_PER_UNIT")
    margin = os.environ.get("GROSS_MARGIN")
    if flat_penalty is not None and margin is not None:
        raise ValueError(
            "set GROSS_MARGIN (unit economics) or STOCKOUT_PENALTY_PER_UNIT (flat legacy), not both"
        )
    # A flat penalty in the environment is an explicit request for the legacy model; otherwise the
    # penalty follows unit economics.
    gross_margin = None if flat_penalty is not None else float(margin or DEFAULT_GROSS_MARGIN)

    return Config(
        track=track,
        costs=CostParams(
            holding_cost_rate=float(
                os.environ.get("HOLDING_COST_RATE", annual_to_daily(DEFAULT_ANNUAL_HOLDING_RATE))
            ),
            stockout_penalty_per_unit=float(flat_penalty if flat_penalty is not None else 5.00),
            lead_time_days=int(os.environ.get("LEAD_TIME_DAYS", 7)),
            service_level_target=float(os.environ.get("SERVICE_LEVEL_TARGET", 0.95)),
            gross_margin=gross_margin,
            lot_multiple=float(os.environ.get("LOT_MULTIPLE", DEFAULT_LOT_MULTIPLE)),
        ),
    )
