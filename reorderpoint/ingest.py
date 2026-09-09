"""Raw data (Track A or Track B) -> canonical long table (series_id, date, y, price, exog...).

Track A (M5): sales_train_validation.csv (wide, one row per series) + calendar.csv + sell_prices.csv
melted and joined into one long table.

Track B (business): one CSV/Excel with sku/date/units_sold[/unit_price][/on_hand], renamed onto
the same core columns (series_id, date, y, price) so the rest of the pipeline is track-agnostic.
See data/track_b/README.md for the schema contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reorderpoint.config import REPO_ROOT, load_config

CORE_COLUMNS = ["series_id", "date", "y", "price"]

TRACK_A_RAW_DIR = REPO_ROOT / "data" / "track_a" / "raw"
TRACK_A_PROCESSED = REPO_ROOT / "data" / "track_a" / "processed" / "panel.parquet"
TRACK_B_PROCESSED = REPO_ROOT / "data" / "track_b" / "processed" / "panel.parquet"


def load_track_a(
    raw_dir: Path,
    store_ids: list[str] | None = None,
    cat_ids: list[str] | None = None,
    sales_file: str = "sales_train_validation.csv",
) -> pd.DataFrame:
    """Melt M5's wide per-series rows into a long table and attach calendar + price.

    `store_ids` / `cat_ids` restrict to a subset (see docs/data.md for the default subset
    used through Phase 4 — the full dataset is only used starting Phase 5).
    """
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    sales = pd.read_csv(raw_dir / sales_file)

    if store_ids:
        sales = sales[sales["store_id"].isin(store_ids)]
    if cat_ids:
        sales = sales[sales["cat_id"].isin(cat_ids)]

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="y")

    long = long.merge(calendar, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long["snap"] = np.select(
        [long["state_id"] == "CA", long["state_id"] == "TX", long["state_id"] == "WI"],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]],
        default=0,
    )

    long = long.rename(columns={"id": "series_id", "sell_price": "price"})

    keep = CORE_COLUMNS + [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap",
    ]
    return long[keep].sort_values(["series_id", "date"]).reset_index(drop=True)


def load_track_b(path: Path) -> pd.DataFrame:
    """Read the business export and rename it onto the canonical core schema."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    required = {"sku", "date", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track B data missing required columns: {sorted(missing)}")

    df = df.rename(columns={"sku": "series_id", "units_sold": "y", "unit_price": "price"})
    if "price" not in df.columns:
        df["price"] = pd.NA

    keep = list(CORE_COLUMNS)
    if "on_hand" in df.columns:
        keep.append("on_hand")
    return df[keep].sort_values(["series_id", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw data into the canonical panel.")
    parser.add_argument("--store-ids", nargs="*", default=None)
    parser.add_argument("--cat-ids", nargs="*", default=["HOBBIES"])
    args = parser.parse_args()

    config = load_config()
    if config.track == "a":
        panel = load_track_a(TRACK_A_RAW_DIR, store_ids=args.store_ids, cat_ids=args.cat_ids)
        out_path = TRACK_A_PROCESSED
    else:
        if not config.track_b_data_path:
            raise SystemExit("REORDERPOINT_TRACK=b but TRACK_B_DATA_PATH is not set in .env")
        panel = load_track_b(Path(config.track_b_data_path))
        out_path = TRACK_B_PROCESSED

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"Wrote {len(panel):,} rows, {panel['series_id'].nunique():,} series -> {out_path}")


if __name__ == "__main__":
    main()
