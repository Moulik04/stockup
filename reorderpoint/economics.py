"""Exact re-pricing of stored simulation runs under any (holding rate, gross margin).

A run's inventory trajectory — orders, stock on hand, lost units — depends on the forecast and the
service target only, never on what holding costs or what a lost sale costs (`docs/decision.md`;
verified by re-simulation in `reports/holding_breakeven_*.md`). So a stored per-series row can be
re-priced exactly: holding scales with the rate and with the unit-cost basis the rate applies to,
stockout with the per-unit penalty.

Two cost models exist and every row is priced under one of them:

- **legacy** — flat $5 per lost unit, stock valued at the retail price (`gross_margin` NaN);
- **unit economics** — a lost sale costs `gross_margin x price`, stock is valued at
  `(1 - gross_margin) x price` (`config.CostParams`).

New runs stamp what they were priced under (`decision._simulate_policy`); older parquets carry no
stamp and are read as legacy at 0.02/day, with each series' price recovered from the panel
(`unit_prices`). Everything that re-prices goes through here, so no analysis can quietly assume the
current config describes a stored run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reorderpoint.config import annual_to_daily

# Every detail parquet written before 2026-09-20 was simulated at this rate and carries no stamp.
LEGACY_SIMULATED_RATE = 0.02
LEGACY_FLAT_PENALTY = 5.0
RATE_STAMP = "holding_cost_rate"


def simulated_rate(df: pd.DataFrame) -> float:
    """The daily holding rate the rows in `df` were *simulated* at — not the current config value.

    Read from the `holding_cost_rate` column when the run stamped one, else the legacy 0.02.
    Mixed stamps mean two runs were concatenated; re-pricing that with one base rate would be
    wrong, so it raises.
    """
    if RATE_STAMP not in df.columns:
        return LEGACY_SIMULATED_RATE
    rates = df[RATE_STAMP].dropna().unique()
    if len(rates) != 1:
        raise ValueError(f"rows were simulated at {len(rates)} different holding rates: {rates}")
    return float(rates[0])


def unit_prices(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """(fold, series_id) -> the mean training-window price the simulation used for that fold.

    Cached beside the other artifacts; rebuilt from the evaluation panel when absent."""
    from reorderpoint import backtest as bt
    from reorderpoint import decision as dec

    path = bt.PANEL_PATH.parent / "unit_prices.parquet"
    if path.exists():
        return pd.read_parquet(path)
    eval_panel = (
        bt.sample_series(panel, bt.N_SERIES_SAMPLE) if panel is not None else (bt.load_eval_panel())
    )
    frames = []
    for fold in bt.make_folds(eval_panel):
        train = eval_panel[eval_panel["date"] <= fold.train_end]
        price = dec.unit_cost_from_price(train)
        frames.append(
            pd.DataFrame(
                {"fold": fold.index, "series_id": price.index, "unit_price": price.to_numpy()}
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def with_basis(detail: pd.DataFrame) -> pd.DataFrame:
    """`detail` with the columns re-pricing needs: `unit_price`, `gross_margin` (NaN = legacy flat
    penalty), `stockout_penalty_flat`. Unstamped legacy rows get the legacy values."""
    d = detail.copy()
    if "unit_price" not in d.columns:
        d = d.merge(unit_prices(), on=["fold", "series_id"], how="left", validate="m:1")
        if d["unit_price"].isna().any():
            raise ValueError("no training-window price for some (fold, series) rows")
    if "gross_margin" not in d.columns:
        d["gross_margin"] = np.nan
    if "stockout_penalty_flat" not in d.columns:
        d["stockout_penalty_flat"] = LEGACY_FLAT_PENALTY
    return d


def reprice(
    detail: pd.DataFrame,
    annual_holding: float,
    gross_margin: float | None,
    flat_penalty: float = LEGACY_FLAT_PENALTY,
) -> pd.DataFrame:
    """The rows of `detail` re-priced at `annual_holding` (fraction of unit cost per year) under
    `gross_margin` (None = the legacy flat `flat_penalty`, stock valued at price).

    Same schema back — `holding_cost`, `stockout_cost`, `total_cost` replaced — so the bootstrap
    and report code that consumes a detail frame runs on it unchanged. Exact, per series, because
    both scale factors are per-row."""
    d = with_basis(detail)
    price = d["unit_price"].to_numpy(dtype=float)
    used_margin = d["gross_margin"].to_numpy(dtype=float)
    legacy_rows = np.isnan(used_margin)
    cost_used = np.where(legacy_rows, price, price * (1 - np.nan_to_num(used_margin)))
    pen_used = np.where(
        legacy_rows, d["stockout_penalty_flat"].to_numpy(dtype=float), used_margin * price
    )
    if gross_margin is None:
        cost_new, pen_new = price, np.full_like(price, flat_penalty)
    else:
        cost_new, pen_new = price * (1 - gross_margin), gross_margin * price
    rate_used = simulated_rate(d)
    holding = (
        d["holding_cost"].to_numpy(dtype=float)
        * (annual_to_daily(annual_holding) / rate_used)
        * (cost_new / cost_used)
    )
    stockout = d["stockout_cost"].to_numpy(dtype=float) * (pen_new / pen_used)
    return d.assign(
        holding_cost=holding,
        stockout_cost=stockout,
        total_cost=holding + stockout,
        gross_margin=np.nan if gross_margin is None else gross_margin,
        stockout_penalty_flat=flat_penalty,
        **{RATE_STAMP: annual_to_daily(annual_holding)},
    )
