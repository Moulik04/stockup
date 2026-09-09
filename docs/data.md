# Data

## Canonical schema

Every track is ingested (`reorderpoint/ingest.py`) into the same core long-table shape:

| column | type | meaning |
|---|---|---|
| `series_id` | string | one time series (Track A: `item_id_store_id_validation`; Track B: SKU) |
| `date` | date | one row per series per period |
| `y` | number | units sold — the demand proxy (see caveat below) |
| `price` | number, nullable | unit price at that date, if known |

Track A additionally carries exogenous columns straight through: `item_id`, `dept_id`, `cat_id`,
`store_id`, `state_id`, `wday`, `month`, `year`, `event_name_1/2`, `event_type_1/2`, `snap`
(SNAP-eligibility flag for that series' state, resolved from the calendar's per-state `snap_CA` /
`snap_TX` / `snap_WI` columns). Track B carries `on_hand` when the export includes it.

## Track A — M5 Forecasting Accuracy (Kaggle)

Three raw files, joined in `load_track_a`:

- `sales_train_validation.csv` — one row per series (`item_id × store_id`), wide: one column per
  day (`d_1` … `d_n`). Melted to long.
- `calendar.csv` — maps each `d_i` to a real `date`, plus calendar/event/SNAP exogenous features.
- `sell_prices.csv` — `store_id × item_id × wm_yr_wk → sell_price`, joined onto the melted panel.

Hierarchical structure: item × store × dept × category × state. Grain is **daily**. Forecast
horizon is **28 days**, matching the competition's own validation/evaluation split.

**Subset for fast iteration (Phase 0–4):** filtered to `cat_id == "HOBBIES"` (the smallest of the
three top-level categories) via `load_track_a(..., cat_ids=["HOBBIES"])`, the default in
`reorderpoint/ingest.py` (`make data`). Full dataset (`make data-full`) is used starting Phase 5.

Actual subset, from the real download: **10,808,450 rows, 5,650 series**, all 10 stores / 3 states
(CA, TX, WI), **2011-01-29 to 2016-04-24**. Full EDA in `notebooks/eda.ipynb`, exported to
[`reports/eda.md`](../reports/eda.md).

**Full dataset** (`make data-full`, all 3 categories): **58,327,370 rows, 30,490 series**, same 10
stores / 3 states and date range. 11.7GB in memory with default pandas dtypes — more than this
project's 8.6GB dev machine has, so nothing reads the full panel back into memory unfiltered;
every full-scale consumer (currently `reorderpoint/reconcile.py`) uses parquet predicate pushdown
to read only the slice it needs, rather than loading the full frame and filtering afterward.

**Status:** ingest is implemented, schema-tested against fixtures shaped exactly like the real
files (`tests/fixtures/m5_sample/`, `tests/test_ingest.py`), and now run end-to-end against the
real Kaggle download (`make data`). Headline EDA findings:

- **Zero-inflated demand is the norm, not the exception:** 77.3% of series-days have zero sales;
  89.4% of series are zero on more than half their days. This is why the Phase 3 LightGBM point
  model uses Tweedie loss rather than plain L2.
- **Seasonality is real but modest at the category level:** ~1.3x peak/trough across both month
  and day-of-week — enough to matter, not enough to dominate on its own.
- **Price effects exist but are noisy:** among series with any price variation, the median
  within-series price/sales correlation is slightly negative (-0.028) with the expected sign in
  65% of series — a feature worth including, not a strong standalone signal.
- **New items are common:** ~54% of series (3,056 / 5,650) have their first nonzero sale more than
  90 days into the panel — real cold-start cases the model needs to handle.

## Track B — business data

Not yet available. Expected schema documented in `data/track_b/README.md`; `load_track_b` is
implemented and schema-tested against a fixture matching that contract
(`tests/fixtures/track_b_sample.csv`). Real ingestion happens once the business export is available.

## What "demand" means here

`y` is **units sold**, used as a proxy for true demand. This is a known limitation: if a SKU was
out of stock, sold units under-count actual demand for that period (stockout censoring). No
correction is applied in v1 — flagged here so the reader knows the forecasts are trained against
realised sales, not an unbiased demand signal. Track B's `on_hand` column, once available,
could support a future censoring correction — not attempted in v1.
