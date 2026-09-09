"""Central config: data track switch, paths, cost parameters — all from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CostParams:
    holding_cost_rate: float
    stockout_penalty_per_unit: float
    lead_time_days: int
    service_level_target: float


@dataclass(frozen=True)
class Config:
    track: str  # "a" or "b"
    track_b_data_path: str | None
    costs: CostParams


def load_config() -> Config:
    track = os.environ.get("REORDERPOINT_TRACK", "a").lower()
    if track not in {"a", "b"}:
        raise ValueError(f"REORDERPOINT_TRACK must be 'a' or 'b', got {track!r}")

    return Config(
        track=track,
        track_b_data_path=os.environ.get("TRACK_B_DATA_PATH") or None,
        costs=CostParams(
            holding_cost_rate=float(os.environ.get("HOLDING_COST_RATE", 0.02)),
            stockout_penalty_per_unit=float(os.environ.get("STOCKOUT_PENALTY_PER_UNIT", 5.00)),
            lead_time_days=int(os.environ.get("LEAD_TIME_DAYS", 7)),
            service_level_target=float(os.environ.get("SERVICE_LEVEL_TARGET", 0.95)),
        ),
    )
