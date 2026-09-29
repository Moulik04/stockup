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
`snap_TX` / `snap_WI` columns). Track B carries only the core columns, at weekly grain.

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

## Track B — UCI Online Retail II

Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CG6D>. Licence **CC BY 4.0** (attribution required); taken from the UCI
repository page only, not from mirrors that carry other licence labels. A UK-based, registered,
non-store online retailer of all-occasion gift-ware; many of its customers are wholesalers.
1,067,371 invoice lines, 2009-12-01 to 2011-12-09, in one workbook of two sheets. Columns: invoice
number (a leading `C` marks a cancellation), stock code, description, quantity, invoice date and
time, unit price in pounds sterling, customer id (22.8% missing), country.

`make data-b` downloads it (`scripts/download_online_retail.py`, standard library only) and runs the
cleaning (`reorderpoint/online_retail.py`). The raw file is never committed. `REORDERPOINT_TRACK=b`
selects the resulting panels (`TRACK_B_PANEL=primary|robustness`). Grain is **weekly** (Monday to Sunday), because two years is too short
for monthly folds with annual seasonality. The design, economics and what will count as a replication
of the Track A findings are in [`track_b.md`](track_b.md), registered before any model was run.

### Cleaning rules, in the order they run

Counts are from `reports/track_b_cleaning_2026-09-29.md`, which `make data-b` regenerates. "Sold" and
"returned" are units removed by that rule.

| # | rule | rows | sold units | returned units |
|---|---|---|---|---|
| 1 | stock codes normalised (trim whitespace, upper-case) | 0 (3,472 rows changed, 174 codes merged) | 0 | 0 |
| 2 | sheet overlap kept once | 22,523 | 182,448 | 15,800 |
| 3 | non-product stock codes dropped | 5,992 | 25,002 | 14,367 |
| 4 | negative quantity that is not a cancellation dropped | 3,362 | 0 | 564,030 |
| 5 | zero or negative price dropped | 2,576 | 243,450 | 0 |
| 6 | cancellations netted against the sale they reverse | 23,581 | 432,448 | 469,881 |
| 7 | partial first and last weeks dropped | 31,422 | 306,054 | 0 |

1. **Stock codes are normalised.** The same product is sometimes entered as `15056BL` and `15056bl`
   (or with a trailing space); left alone each spelling becomes its own series and splits the
   product's demand. 174 codes merged.
2. **The two sheets overlap.** Sheet "Year 2009-2010" runs to 2010-12-09 and sheet "Year 2010-2011"
   starts 2010-12-01, so nine days are in both. The two copies were checked to be the same rows
   (an error is raised if not) and the earlier sheet's copy dropped. Without this every unit in those
   days is counted twice.
3. **Non-product stock codes are dropped.** A product code is five digits with an optional one or
   two letter suffix. The 61 distinct codes that are not (after rule 1) are postage and carriage
   (`POST`, `DOT`, `C2`, `C3`), manual entries, adjustments, discounts and samples (`M`, `ADJUST`,
   `ADJUST2`, `B` bad debt, `D`, `S`), fees and commissions (`BANK CHARGES`, `AMAZONFEE`, `CRUK`), gift
   vouchers (`GIFT`, `GIFT_0001_*`), test entries (`TEST001`, `TEST002`), 34 `DCGS` items from what
   appears to be a side line, and two one-offs (`PADS`, `SP1002`). None is a product
   the seller holds stock of against a reorder point. A stricter or looser rule would move a few hundred rows.
4. **Negative quantities that are not cancellations are dropped.** All 3,362 have price zero and
   no customer: stock write-offs, "damaged", "lost", "check". They are inventory adjustments, not
   customer demand, and they carry 564,030 units, so leaving them in would erase real sales.
5. **Zero-price lines are dropped.** 2,576 lines, 243,450 units (2.2% of the units), almost all
   without a customer: samples, give-aways and adjustments with no revenue. Unit value cannot be
   computed for them and they are not sales. Price is never negative after rule 3.
6. **Cancellations are netted against the sale they reverse, not against the week they are
   recorded in.** 17,973 cancellation lines carry 469,881 units, 4.2% of what was sold. Each is taken
   off the same customer's earlier sales of the same SKU, most recent first, within 90 days; that
   matches 92.0% of cancelled units (432,448), and the other 37,433, with no customer or no sale to
   reverse, are dropped rather than subtracted from an unrelated week. Reasons: a cancellation is
   not negative demand. The two largest lines in the file, 80,995 and 74,215 units (each a single
   product ordered and cancelled by the same customer within minutes), are mistakes, and netting
   them in the cancellation's own week would only work when both fall in the same week; an order on
   Friday cancelled on Monday would leave a spike standing and a negative week behind it. Netting
   against the original sale removes the spike where it was recorded, and no week can go negative.
7. **Partial weeks are dropped.** The first week the data touches (Monday 2009-11-30, data from
   Tuesday 12-01) and the last (Monday 2011-12-05, data to Friday 12-09 midday) are incomplete, which
   would read as demand falling. That leaves **104 complete weeks, 2009-12-07 to 2011-11-28**.

### What is deliberately not cleaned

- **Bulk orders stay.** The largest surviving SKU-weeks are 10,000-11,000 units against a typical
  non-zero week of 17 (median); the top 1% of non-zero SKU-weeks hold 19.3% of all units. Wholesale
  is the business. Removing them would be choosing a smoother company than the one in the data; how
  the models and the buffer cope with them is a result, not something to clean away first.
- **Every country stays.** 82.1% of units sold (before netting) go to the UK; the rest is export wholesale (Netherlands,
  Ireland, France, Denmark, Germany...). One stock serves all of it, and a replenishment decision is
  for total demand, so restricting to UK customers would forecast a demand the seller does not
  replenish against. Prices are sterling for every country.
- **Exact duplicate lines within a sheet stay.** 11,814 lines (1.1% of rows, 0.30% of positive units,
  after the sheet overlap was removed) are identical in every column. They may be entry duplicates or the same item on two order lines;
  the data cannot say, and the effect is below any weekly noise.
- **Rows without a customer id stay** (22.8% of the file): they are sales. Only netting needs a
  customer, and a customerless cancellation is dropped, not netted.
- **Price outliers stay in demand.** A line priced more than 10× or less than 1/10 of its SKU's
  median (870 lines, 165,892 units, measured on sales lines before cancellations were netted) is left
  out of the unit value only; its quantity counts.

### Weekly panel and series selection

`y` is net units per SKU per week (zero where nothing sold), `price` is the weekly median line price
in **USD** at one fixed rate (1.5770 USD/GBP; source in [`track_b.md`](track_b.md)), carried forward
over weeks with no sale. Weeks with no trading at all (2009-12-28 and 2010-12-27, the Christmas
closures) are kept as zero weeks.

A SKU is kept if it sold in at least 26 of the first 52 weeks, a rule fixed before any model was run.
**4,667 SKUs sold at least once; 1,742 qualify** (at a threshold of 13 weeks 2,666 would, at 39 weeks
942), carrying 71.8% of the cleaned units. 181,168 rows (1,742 series × 104 weeks). The rule has a
known cost: it keeps year-round sellers and cannot select a SKU first sold in year 2 (652 of them, 19.3%
of year-2 units) or a short-season one, so the selected panel shrinks to 73% of its year-1 units in
year 2 while the company is flat (94%), and its autumn build-up is weaker than the company's
(Sep-Nov over Mar-Jul: 1.34 for the selected SKUs, 3.77 for the rest, in year 1). Results are results
for the steady sellers.

Weekly zero share of the selected panel: 30.1% of SKU-weeks (28.8% excluding the two closed weeks;
23.2% in year 1, 37.1% in year 2), against Track A's 77% daily. The zero share is intermittency at a
different grain, not less intermittent demand.

## What "demand" means here

`y` is **units sold**, used as a proxy for true demand. This is a known limitation: if a SKU was
out of stock, sold units under-count actual demand for that period (stockout censoring). No
correction is applied in v1 — flagged here so the reader knows the forecasts are trained against
realised sales, not an unbiased demand signal. Neither track has an on-hand column, so a censoring
correction could not be built from either — not attempted in v1.
