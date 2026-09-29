# Track B — UCI Online Retail II

This directory never holds the raw data (see the repo `.gitignore`); `make data-b` downloads it here.

**Source:** Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CG6D>. **Licence: CC BY 4.0 — attribution required.**
Page: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>. Only this page is used as the
source; mirrors on other sites carry other licence labels and are not.

A UK-based and registered non-store online retailer of all-occasion gift-ware; many of its customers
are wholesalers. 1,067,371 invoice lines between 2009-12-01 and 2011-12-09.

## Files

- `raw/online_retail_II.xlsx` — the UCI workbook (two sheets), fetched by
  `scripts/download_online_retail.py`. Not committed.
- `raw/online_retail_II.parquet` — a cache of it, so the 30-second xlsx read happens once. Not committed.
- `processed/primary/panel.parquet` and `processed/robustness/panel.parquet` — the cleaned weekly SKU
  panels (`series_id`, `date` = the Monday, `y` = net units, `price` = USD unit value, `month`,
  `weekofyear`), written by `python -m reorderpoint.online_retail`. The primary panel is the SKUs that
  sold in at least 26 of the first 52 weeks; the robustness panel, a superset, at least 13
  (`docs/track_b.md`). Select one with `TRACK_B_PANEL=primary|robustness` (default primary). Each panel's
  forecast and calibration caches are written beside it. Not committed.

## What is done to it

Cleaning rules, each with what it removed and why: [`docs/data.md`](../../docs/data.md). The design,
the economics (USD at one fixed rate; margin, holding cost and lead time as documented assumptions
with sensitivity) and what will count as a replication of the Track A findings, registered before any
model was run: [`docs/track_b.md`](../../docs/track_b.md). The state of the data:
`reports/track_b_cleaning_<date>.md`.
