"""Track B: UCI "Online Retail II" -> a weekly SKU x week panel on the canonical schema.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D — CC BY 4.0, attribution required. A UK-based online seller of
all-occasion gift-ware, many of whose customers are wholesalers; transactions 2009-12-01 to
2011-12-09, two xlsx sheets. `scripts/download_online_retail.py` fetches it.

Every cleaning rule is one step here, and every step records what it removed, so `docs/data.md` and
`reports/track_b_cleaning_*.md` state counts that come from the run rather than from memory. The
rules and the reasons for each are in `docs/data.md`; the choices that were not obvious (netting a
cancellation against the sale it reverses, keeping bulk orders, keeping every country) are also in
`DECISIONS.md`.

`y` is net units invoiced, a demand proxy exactly as in Track A: stock-outs are not observed, so a
week with no sales may be a week with no stock.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from reorderpoint.config import REPO_ROOT

TRACK_B_RAW_DIR = REPO_ROOT / "data" / "track_b" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "track_b" / "processed"
PANELS = ("primary", "robustness")  # each in PROCESSED_DIR / <name> / panel.parquet
RAW_XLSX = TRACK_B_RAW_DIR / "online_retail_II.xlsx"
REPORTS_DIR = REPO_ROOT / "reports"

CITATION = (
    "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5CG6D."
)

# Prices are sterling. One fixed rate for the whole period, so that no cost result depends on when
# a sale happened: the mean of the 25 monthly averages December 2009 - December 2011 of the Federal
# Reserve's H.10 US dollars per pound, as published in FRED series EXUSUK (1.5770).
GBP_TO_USD = 1.577

# A product stock code is five digits with an optional one- or two-letter variant suffix. Everything
# else in the file is postage, a fee, a manual adjustment, a voucher or a test entry (see
# `docs/data.md` for the list) or one of a few dozen "DCGS" side-line items.
PRODUCT_CODE = re.compile(r"^\d{5}[A-Za-z]{0,2}$")

# A cancellation is matched to the same customer's earlier sales of the same SKU inside this window.
# At 90 days the window holds 93% of cancelled units that have any earlier sale to that customer.
NETTING_LOOKBACK_DAYS = 90
# Lines priced beyond this factor from their SKU's median price are left out of the unit value,
# never out of the demand: the quantity on such a line is as real as any other.
PRICE_OUTLIER_RATIO = 10.0

# The panel is the 104 complete Monday-to-Sunday weeks inside the data. The first week the file
# touches (Monday 2009-11-30, data from Tuesday 12-01) and the last (Monday 2011-12-05, data to
# Friday 12-09 midday) are partial and dropped.
FIRST_WEEK = pd.Timestamp("2009-12-07")
N_WEEKS = 104
LAST_WEEK = FIRST_WEEK + pd.Timedelta(weeks=N_WEEKS - 1)  # 2011-11-28

# Series selection, fixed before any model was run: a SKU is in if it sold in at least
# MIN_ACTIVE_WEEKS of the first SELECTION_WEEKS weeks. It uses first-year data only, so it looks at
# nothing in the years being forecast, at the price of excluding every SKU launched after that.
SELECTION_WEEKS = 52
MIN_ACTIVE_WEEKS = 26
# The robustness panel, added (before any model was run) alongside the primary: a strict superset,
# every SKU that sold in at least this many of the first SELECTION_WEEKS weeks. Verdicts are the
# primary panel's; this one is reported as agrees or disagrees (`docs/track_b.md`).
ROBUSTNESS_MIN_ACTIVE_WEEKS = 13
MIN_ACTIVE_BY_PANEL = {"primary": MIN_ACTIVE_WEEKS, "robustness": ROBUSTNESS_MIN_ACTIVE_WEEKS}

LINE_COLUMNS = [
    "invoice",
    "stock_code",
    "invoice_date",
    "quantity",
    "price_gbp",
    "customer",
    "country",
]


@dataclass
class Step:
    rule: str
    rows_removed: int
    sold_units_removed: int = 0
    returned_units_removed: int = 0
    note: str = ""


@dataclass
class Cleaning:
    steps: list[Step] = field(default_factory=list)
    netting: dict = field(default_factory=dict)

    def add(self, before: pd.DataFrame, after: pd.DataFrame, rule: str, note: str = "") -> None:
        gone = before.loc[~before.index.isin(after.index)]
        self.steps.append(
            Step(
                rule,
                rows_removed=len(gone),
                sold_units_removed=int(gone.loc[gone["quantity"] > 0, "quantity"].sum()),
                returned_units_removed=int(-gone.loc[gone["quantity"] < 0, "quantity"].sum()),
                note=note,
            )
        )


# ---- loading --------------------------------------------------------------------------------


def load_raw(path: Path = RAW_XLSX, use_cache: bool = True) -> pd.DataFrame:
    """The two sheets of the UCI workbook stacked, on standardised column names. Reading the xlsx
    takes about half a minute, so a parquet copy next to it is reused when present."""
    cache = path.with_suffix(".parquet")
    if use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        raw = pd.read_parquet(cache)
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        raw = pd.concat(sheets.values(), keys=sheets.keys(), names=["sheet"])
        raw = raw.reset_index(level=0).reset_index(drop=True)
        raw["Description"] = raw["Description"].map(
            lambda v: v if isinstance(v, str) or pd.isna(v) else str(v)
        )
        raw.to_parquet(cache, index=False)
    return raw.rename(
        columns={
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price_gbp",
            "Customer ID": "customer",
            "Country": "country",
        }
    )


# ---- cleaning rules -------------------------------------------------------------------------


def drop_sheet_overlap(raw: pd.DataFrame) -> pd.DataFrame:
    """The two sheets both hold 2010-12-01 to 2010-12-09. Keep the later sheet's copy, after
    checking the two copies are the same rows — a silent drop of differing rows would be a guess."""
    sheets = raw.groupby("sheet")["invoice_date"].min().sort_values().index.tolist()
    if len(sheets) < 2:
        return raw
    first_sheet, second_sheet = sheets
    a = raw[raw["sheet"] == first_sheet]
    b = raw[raw["sheet"] == second_sheet]
    lo, hi = b["invoice_date"].min(), a["invoice_date"].max()
    if lo > hi:
        return raw
    key = [c for c in LINE_COLUMNS if c != "invoice_date"] + ["invoice_date"]
    in_a = a[a["invoice_date"] >= lo]
    in_b = b[b["invoice_date"] <= hi]

    def canon(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[key].fillna(-1).sort_values(key).reset_index(drop=True)

    if not canon(in_a).equals(canon(in_b)):
        raise ValueError("the two sheets disagree inside their overlap window; not dropping either")
    return raw.drop(in_a.index)


def net_cancellations(lines: pd.DataFrame, lookback_days: int = NETTING_LOOKBACK_DAYS):
    """Remove each cancellation's quantity from the same customer's earlier sales of the same SKU,
    most recent sale first, within `lookback_days`. Returns (net sales lines, stats).

    A cancellation with no customer, or with nothing left to reverse inside the window, is dropped
    without netting: it has no sale to attribute to, and putting it in the week it was recorded
    would subtract from demand that was never counted there.
    """
    is_cancel = lines["invoice"].str.startswith("C") & (lines["quantity"] < 0)
    sales = lines[~is_cancel].copy()
    cancels = lines[is_cancel].sort_values("invoice_date")
    remaining = sales["quantity"].to_numpy(dtype=float).copy()

    by_key: dict[tuple, np.ndarray] = {}
    keyed = sales.reset_index(drop=True)
    keyed = keyed[keyed["customer"].notna()]
    for key, idx in keyed.groupby(["customer", "stock_code"]).indices.items():
        pos = keyed.index.to_numpy()[idx]
        order = np.argsort(keyed["invoice_date"].to_numpy()[idx], kind="stable")
        by_key[key] = pos[order]

    sale_dates = sales["invoice_date"].to_numpy()
    window = np.timedelta64(lookback_days, "D")
    matched = unmatched = 0.0
    for c in cancels.itertuples(index=False):
        need = float(-c.quantity)
        if pd.isna(c.customer):
            unmatched += need
            continue
        pos = by_key.get((c.customer, c.stock_code))
        if pos is not None:
            for p in pos[::-1]:
                if sale_dates[p] > np.datetime64(c.invoice_date):
                    continue
                if sale_dates[p] < np.datetime64(c.invoice_date) - window:
                    break
                take = min(need, remaining[p])
                remaining[p] -= take
                need -= take
                matched += take
                if need <= 0:
                    break
        unmatched += max(need, 0.0)

    sales = sales.assign(quantity=remaining)
    sales = sales[sales["quantity"] > 0]
    stats = {
        "cancellation_rows": len(cancels),
        "cancelled_units": int(matched + unmatched),
        "matched_units": int(matched),
        "unmatched_units": int(unmatched),
    }
    return sales, stats


def clean_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, Cleaning]:
    """Raw workbook rows -> net sales lines, one recorded `Step` per rule, in this order."""
    log = Cleaning()
    lines = raw[["sheet", *LINE_COLUMNS, "description"]].copy()
    log.steps.append(Step("input rows (both sheets)", 0, note=f"{len(lines):,} rows"))

    # One product is sometimes entered under two spellings ("15056BL" and "15056bl", or a trailing
    # space), which would split its demand across two series. Rows are kept; the code is normalised.
    normal = lines["stock_code"].str.strip().str.upper()
    changed = int((normal != lines["stock_code"]).sum())
    merged = lines["stock_code"].nunique() - normal.nunique()
    lines["stock_code"] = normal
    log.steps.append(
        Step(
            "stock codes normalised (whitespace trimmed, upper-cased)",
            0,
            note=f"{changed:,} rows changed; {merged:,} codes merged into another spelling",
        )
    )

    after = drop_sheet_overlap(lines)
    log.add(lines, after, "sheet overlap: the same 2010-12-01..09 rows in both sheets, kept once")
    lines = after

    after = lines[lines["stock_code"].str.match(PRODUCT_CODE)]
    log.add(lines, after, "non-product stock codes (postage, fees, adjustments, vouchers, tests)")
    lines = after

    is_cancel = lines["invoice"].str.startswith("C")
    after = lines[is_cancel | (lines["quantity"] > 0)]
    log.add(lines, after, "negative quantity that is not a cancellation (stock write-offs)")
    lines = after

    after = lines[lines["price_gbp"] > 0]
    log.add(lines, after, "zero or negative price")
    lines = after

    netted, stats = net_cancellations(lines)
    log.netting = stats
    n_before = len(lines)
    log.steps.append(
        Step(
            "cancellations netted against the customer's earlier sale of the same SKU (90 days)",
            rows_removed=n_before - len(netted),
            sold_units_removed=stats["matched_units"],
            returned_units_removed=stats["cancelled_units"],
            note=(
                f"{stats['matched_units']:,} of {stats['cancelled_units']:,} cancelled units "
                f"matched; {stats['unmatched_units']:,} unmatched and dropped"
            ),
        )
    )
    lines = netted

    week = lines["invoice_date"].dt.to_period("W-SUN").dt.start_time
    keep = (week >= FIRST_WEEK) & (week <= LAST_WEEK)
    after = lines[keep]
    log.add(lines, after, "partial first and last weeks (outside the 104 complete weeks)")
    lines = after.assign(week=week[keep])
    return lines, log


# ---- weekly panel ---------------------------------------------------------------------------


def unit_values(lines: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekly unit value in USD per SKU: the median line price of the week, ignoring lines beyond
    `PRICE_OUTLIER_RATIO` of the SKU's median price, carried forward over weeks with no sales (and
    backward before the first). Returns a wide frame, weeks x SKUs."""
    median = lines.groupby("stock_code")["price_gbp"].transform("median")
    ratio = lines["price_gbp"] / median
    ok = lines[(ratio <= PRICE_OUTLIER_RATIO) & (ratio >= 1 / PRICE_OUTLIER_RATIO)]
    weekly = ok.groupby(["week", "stock_code"])["price_gbp"].median().unstack("stock_code")
    return weekly.reindex(weeks).ffill().bfill() * GBP_TO_USD


def weekly_units(lines: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Net units per week per SKU, weeks x SKUs, zero where nothing sold."""
    wide = lines.groupby(["week", "stock_code"])["quantity"].sum().unstack("stock_code")
    return wide.reindex(weeks).fillna(0.0)


def select_series(units: pd.DataFrame, min_active_weeks: int = MIN_ACTIVE_WEEKS) -> pd.Index:
    """SKUs that sold in at least `min_active_weeks` of the first SELECTION_WEEKS weeks."""
    active = (units.iloc[:SELECTION_WEEKS] > 0).sum()
    return active[active >= min_active_weeks].index


def build_panel(
    raw: pd.DataFrame, min_active_weeks: int = MIN_ACTIVE_WEEKS
) -> tuple[pd.DataFrame, dict]:
    """(panel, report). The panel has one row per selected SKU per week, columns series_id, date
    (the Monday), y (net units), price (USD unit value), month and weekofyear (calendar columns,
    known in advance); the report is everything the cleaning and selection documentation quotes."""
    lines, log = clean_lines(raw)
    weeks = pd.date_range(FIRST_WEEK, periods=N_WEEKS, freq="W-MON")
    units = weekly_units(lines, weeks)
    prices = unit_values(lines, weeks)
    keep = select_series(units, min_active_weeks)

    y = units[keep]
    long = (
        y.rename_axis(index="date")
        .reset_index()
        .melt(id_vars="date", var_name="series_id", value_name="y")
    )
    price = (
        prices[keep]
        .rename_axis(index="date")
        .reset_index()
        .melt(id_vars="date", var_name="series_id", value_name="price")
    )
    panel = long.merge(price, on=["date", "series_id"], how="left")
    panel["month"] = panel["date"].dt.month
    panel["weekofyear"] = panel["date"].dt.isocalendar().week.astype(int)
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)

    total = units.sum(axis=1)
    closed = total[total < 0.05 * total.median()].index
    descriptions = (
        lines.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iloc[0])
    )
    report = {
        "log": log,
        "lines": lines,
        "units_all": units,
        "units_selected": y,
        "descriptions": descriptions,
        "closed_weeks": [w.date() for w in closed],
        "n_skus_ever_sold": units.shape[1],
        "qualifying": {
            n: int(((units.iloc[:SELECTION_WEEKS] > 0).sum() >= n).sum()) for n in (13, 26, 39)
        },
        "generated": dt.date.today().isoformat(),
    }
    return panel, report


# ---- checkpoint summary: zero share, seasonality, bulk-order spikes ---------------------------

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
YEAR_1, YEAR_2 = "#2a78d6", "#eb6834"  # validated pair (dataviz validate_palette.js, light surface)


def zero_shares(units: pd.DataFrame, closed_weeks: list[dt.date]) -> dict:
    """Share of SKU x week cells with no sale, and how it splits. `closed_weeks` are weeks with no
    trading at all, which say nothing about any one SKU's demand."""
    open_units = units[~units.index.isin(pd.to_datetime(closed_weeks))]
    per_sku = (open_units == 0).mean()
    half = SELECTION_WEEKS
    active_tail = (units.iloc[-13:] > 0).any()
    nonzero = units.to_numpy()[units.to_numpy() > 0]
    return {
        "all_weeks": float((units == 0).to_numpy().mean()),
        "open_weeks": float((open_units == 0).to_numpy().mean()),
        "year_1": float((units.iloc[:half] == 0).to_numpy().mean()),
        "year_2": float((units.iloc[half:] == 0).to_numpy().mean()),
        "per_sku_quartiles": [float(q) for q in per_sku.quantile([0.25, 0.5, 0.75])],
        "skus_silent_in_final_13_weeks": int((~active_tail).sum()),
        "n_skus": units.shape[1],
        "mean_nonzero_units": float(nonzero.mean()),
        "median_nonzero_units": float(np.median(nonzero)),
    }


def bulk_order_tables(lines: pd.DataFrame, units: pd.DataFrame, descriptions: pd.Series) -> dict:
    """The largest SKU-week totals and the largest single invoice lines that survive cleaning, and
    how concentrated demand is in them."""
    selected = lines[lines["stock_code"].isin(units.columns)]
    stacked = units.rename_axis(index="week", columns="stock_code").stack()
    top_weeks = stacked.sort_values(ascending=False).head(12).rename("units").reset_index()
    line_units = selected.groupby(["week", "stock_code"])["quantity"].agg(["max", "size"])
    nonzero_median = units.where(units > 0).median()
    top_weeks = top_weeks.join(line_units, on=["week", "stock_code"])
    top_weeks["largest_line_share"] = top_weeks["max"] / top_weeks["units"]
    top_weeks["x_typical_week"] = top_weeks["units"] / top_weeks["stock_code"].map(nonzero_median)
    top_weeks["description"] = top_weeks["stock_code"].map(descriptions)
    top_lines = selected.sort_values("quantity", ascending=False).head(10).copy()
    top_lines["description"] = top_lines["stock_code"].map(descriptions)
    values = units.to_numpy().ravel()
    values = np.sort(values[values > 0])[::-1]
    return {
        "top_weeks": top_weeks,
        "top_lines": top_lines,
        "share_top_1pct_cells": float(values[: max(1, len(values) // 100)].sum() / values.sum()),
        "share_units_in_lines_1000_plus": float(
            selected.loc[selected["quantity"] >= 1000, "quantity"].sum()
            / selected["quantity"].sum()
        ),
    }


def selection_effect(units_all: pd.DataFrame, selected: pd.Index) -> dict:
    """What the selection rule does to the panel, against the SKUs it leaves out. The rule looks at
    year one only, so year one is where the selected SKUs are guaranteed to be active."""
    h = SELECTION_WEEKS
    rest = units_all.drop(columns=selected)
    first_sale = (units_all > 0).idxmax()
    launched_y2 = first_sale[(units_all > 0).any() & (first_sale >= units_all.index[h])].index

    def totals(df: pd.DataFrame) -> tuple[float, float]:
        return float(df.iloc[:h].sum().sum()), float(df.iloc[h:].sum().sum())

    def peak(df: pd.DataFrame, start: int) -> float:
        weekly = df.sum(axis=1).iloc[start : start + h]
        month = weekly.index.month
        return float(
            weekly[(month >= 9) & (month <= 11)].mean() / weekly[(month >= 3) & (month <= 7)].mean()
        )

    groups = {"all cleaned SKUs": units_all, "selected": units_all[selected], "not selected": rest}
    rows = []
    for name, df in groups.items():
        y1, y2 = totals(df)
        rows.append(
            {
                "group": f"{name} ({df.shape[1]:,})",
                "units, year 1": y1,
                "units, year 2": y2,
                "year 2 / year 1": y2 / y1,
                "Sep-Nov / Mar-Jul, year 1": peak(df, 0),
                "Sep-Nov / Mar-Jul, year 2": peak(df, h),
            }
        )
    all_y2 = totals(units_all)[1]
    return {
        "table": pd.DataFrame(rows),
        "launched_year_2": len(launched_y2),
        "launched_year_2_share": float(units_all[launched_y2].iloc[h:].sum().sum() / all_y2),
    }


def plot_seasonality(
    units_all: pd.DataFrame, units: pd.DataFrame, closed_weeks: list[dt.date], path: Path
) -> None:
    """Two years overlaid on one 52-week calendar, as small multiples (never a dual axis): units per
    week for every cleaned SKU, units per week for the selected SKUs, and the share of selected SKUs
    that sold at all. The first two side by side are the point: the selection rule removes the
    seasonal and the new SKUs, so the selected panel is a different picture from the company's."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = SELECTION_WEEKS
    x = np.arange(half)
    starts = units.index[:half]
    ticks = [i for i in range(half) if starts[i].day <= 7]
    labels = [starts[i].strftime("%b") for i in ticks]
    closed = pd.to_datetime(closed_weeks)
    panels = (
        (units_all.sum(axis=1).to_numpy() / 1000, "All cleaned SKUs: units per week (thousands)"),
        (units.sum(axis=1).to_numpy() / 1000, "Selected SKUs: units per week (thousands)"),
        ((units > 0).mean(axis=1).to_numpy() * 100, "Selected SKUs with any sale (%)"),
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=160, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (series, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.plot(x, series[:half], color=YEAR_1, lw=1.8, label="Dec 2009 – Nov 2010")
        ax.plot(x, series[half:], color=YEAR_2, lw=1.8, label="Dec 2010 – Nov 2011")
        for w in closed:
            ax.axvline(int((w - units.index[0]).days // 7) % half, color=INK_2, lw=0.8, ls=":")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    axes[0].text(
        0.155,
        0.93,
        "dotted line: closed for Christmas",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    axes[2].set_xticks(ticks, labels)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def cleaning_report(report: dict, panel: pd.DataFrame, plot_rel_path: str) -> str:
    log: Cleaning = report["log"]
    units: pd.DataFrame = report["units_selected"]
    zs = zero_shares(units, report["closed_weeks"])
    bulk = bulk_order_tables(report["lines"], units, report["descriptions"])
    all_units = report["units_all"]
    covered = float(units.to_numpy().sum() / all_units.to_numpy().sum())

    steps = pd.DataFrame(
        [
            {
                "rule": s.rule,
                "rows removed": f"{s.rows_removed:,}",
                "sold units": f"{s.sold_units_removed:,}",
                "returned units": f"{s.returned_units_removed:,}",
                "detail": s.note if s.rule.startswith("stock codes") else "",
            }
            for s in log.steps[1:]
        ]
    )
    weeks = bulk["top_weeks"]
    top_weeks = pd.DataFrame(
        {
            "SKU": weeks["stock_code"],
            "item": weeks["description"].fillna("").str.title().str.slice(0, 34),
            "week of": weeks["week"].dt.strftime("%Y-%m-%d"),
            "units": weeks["units"].map("{:,.0f}".format),
            "x SKU's typical week": weeks["x_typical_week"].map("{:.0f}x".format),
            "largest line": weeks["max"].map("{:,.0f}".format),
            "invoice lines": weeks["size"].astype(int),
        }
    )
    lines = bulk["top_lines"]
    top_lines = pd.DataFrame(
        {
            "SKU": lines["stock_code"],
            "item": lines["description"].fillna("").str.title().str.slice(0, 34),
            "date": lines["invoice_date"].dt.strftime("%Y-%m-%d"),
            "units": lines["quantity"].map("{:,.0f}".format),
            "country": lines["country"],
            "unit price (GBP)": lines["price_gbp"].map("{:.2f}".format),
        }
    )
    eff = selection_effect(all_units, units.columns)
    t = eff["table"]
    ratio_all, ratio_sel, ratio_rest = (float(v) for v in t["year 2 / year 1"])
    peak_sel = float(t["Sep-Nov / Mar-Jul, year 1"].iloc[1])
    peak_rest = float(t["Sep-Nov / Mar-Jul, year 1"].iloc[2])
    effect_table = eff["table"].assign(
        **{
            "units, year 1": lambda d: d["units, year 1"].map("{:,.0f}".format),
            "units, year 2": lambda d: d["units, year 2"].map("{:,.0f}".format),
            "year 2 / year 1": lambda d: d["year 2 / year 1"].map("{:.2f}".format),
            "Sep-Nov / Mar-Jul, year 1": lambda d: d["Sep-Nov / Mar-Jul, year 1"].map(
                "{:.2f}".format
            ),
            "Sep-Nov / Mar-Jul, year 2": lambda d: d["Sep-Nov / Mar-Jul, year 2"].map(
                "{:.2f}".format
            ),
        }
    )
    q = report["qualifying"]
    quartiles = zs["per_sku_quartiles"]
    share_top = bulk["share_top_1pct_cells"]
    share_big = bulk["share_units_in_lines_1000_plus"]
    out = [
        f"# Track B cleaning — UCI Online Retail II — {report['generated']}",
        "",
        f"Data: {CITATION} Licence CC BY 4.0. A UK online seller of gift-ware with many wholesale "
        "customers; 2009-12-01 to 2011-12-09.",
        "",
        "Nothing here has run a model. This is the state of the data the models will see.",
        "",
        "## What each cleaning rule removed",
        "",
        f"Input: {log.steps[0].note}. Units are counted where the rule removed them; a rule that "
        "removes sales and returns lists both.",
        "",
        _md(steps),
        "",
        f"Cancellations: {log.netting['cancellation_rows']:,} rows, "
        f"{log.netting['cancelled_units']:,} units; {log.netting['matched_units']:,} "
        f"({log.netting['matched_units'] / log.netting['cancelled_units']:.1%}) reversed a sale "
        "by the same customer of the same SKU in the previous 90 days and were netted against it, "
        f"{log.netting['unmatched_units']:,} could not be matched and were dropped.",
        "",
        "## Series selection (fixed before any model was run)",
        "",
        f"A SKU is kept if it sold in at least {MIN_ACTIVE_WEEKS} of the first {SELECTION_WEEKS} "
        f"weeks. Of {report['n_skus_ever_sold']:,} SKUs that sold at least once in the 104 weeks, "
        f"**{units.shape[1]:,} qualify**. For context only, the same count at 13 weeks is "
        f"{q[13]:,} and at 39 weeks is {q[39]:,}. The selected SKUs carry {covered:.1%} of the "
        f"cleaned units. The rule looks only at year one, so a SKU first sold in year two cannot "
        f"be selected: launches are out of scope, which is a limitation and a source of "
        f"selection bias (survivors are steadier than the catalogue).",
        "",
        f"Panel: {panel['series_id'].nunique():,} series × {panel['date'].nunique()} weeks "
        f"({panel['date'].min().date()} to {panel['date'].max().date()}, weeks start Monday), "
        f"{len(panel):,} rows. Weeks with no trading at all (Christmas closure): "
        f"{', '.join(str(w) for w in report['closed_weeks'])}.",
        "",
        "## Zero share",
        "",
        _md(
            pd.DataFrame(
                [
                    {
                        "scope": "all 104 weeks",
                        "share of SKU-weeks with no sale": f"{zs['all_weeks']:.1%}",
                    },
                    {
                        "scope": "excluding the closed weeks",
                        "share of SKU-weeks with no sale": f"{zs['open_weeks']:.1%}",
                    },
                    {
                        "scope": "year 1 (first 52 weeks)",
                        "share of SKU-weeks with no sale": f"{zs['year_1']:.1%}",
                    },
                    {"scope": "year 2", "share of SKU-weeks with no sale": f"{zs['year_2']:.1%}"},
                ]
            )
        ),
        "",
        f"Across SKUs the open-week zero share has quartiles {quartiles[0]:.0%} / "
        f"{quartiles[1]:.0%} / {quartiles[2]:.0%}. {zs['skus_silent_in_final_13_weeks']:,} of the "
        f"{zs['n_skus']:,} selected SKUs sold nothing in the final 13 weeks (discontinued or "
        f"stocked out; the data cannot tell which). When a SKU does sell in a week, the mean is "
        f"{zs['mean_nonzero_units']:.0f} units and the median {zs['median_nonzero_units']:.0f}. "
        "Track A's daily zero share is 77%, so this is intermittent at a different grain: weekly "
        "aggregation removed most of the zeros, and what is left is lumpier per non-zero period.",
        "",
        "## Seasonality, and what the selection rule does to it",
        "",
        f"![Weekly units and active SKUs, two years overlaid]({plot_rel_path})",
        "",
        f"Year 2 as a share of year 1: {ratio_all:.0%} of units company-wide, {ratio_sel:.0%} for "
        f"the selected SKUs, {ratio_rest:.0%} for the SKUs the rule leaves out. The Sep-Nov to "
        f"Mar-Jul ratio is {peak_sel:.2f} for the selected SKUs against {peak_rest:.2f} for the "
        "rest (year 1). By construction the rule keeps SKUs that sold across most of year 1 and "
        "cannot select a SKU first sold in year 2, so it drops short-season and new products.",
        "",
        _md(effect_table),
        "",
        f"{eff['launched_year_2']:,} SKUs sold for the first time in year 2 and carry "
        f"{eff['launched_year_2_share']:.1%} of year-2 units. The rule also guarantees year-1 "
        "activity and nothing after it, so the selected panel's zero share rises from year 1 to "
        "year 2 partly by construction.",
        "",
        "## Bulk orders",
        "",
        "Wholesale customers place very large orders. After cancellations are netted (which "
        "removes the two largest lines in the file, 80,995 and 74,215 units, each cancelled by "
        "the same customer within minutes), the largest surviving SKU-weeks are:",
        "",
        _md(top_weeks),
        "",
        "The largest surviving single invoice lines:",
        "",
        _md(top_lines),
        "",
        f"The top 1% of non-zero SKU-weeks hold {share_top:.1%} of all units, and invoice lines of "
        f"1,000 units or more account for {share_big:.1%}. These are kept as demand: they are the "
        "business, and removing them would be choosing a smoother company than the one in the "
        "data.",
        "",
    ]
    return "\n".join(out)


def write_panels(raw: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Both panels from one cleaning pass, written to PROCESSED_DIR/<name>/panel.parquet. Returns
    ({name: panel}, the primary panel's report)."""
    panels, primary_report = {}, None
    for name in PANELS:
        panel, report = build_panel(raw, MIN_ACTIVE_BY_PANEL[name])
        path = PROCESSED_DIR / name / "panel.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path, index=False)
        print(f"Wrote {len(panel):,} rows, {panel['series_id'].nunique():,} series -> {path}")
        panels[name] = panel
        if name == "primary":
            primary_report = report
    return panels, primary_report


def main() -> None:
    raw = load_raw()
    panels, report = write_panels(raw)
    panel = panels["primary"]
    stamp = report["generated"]
    plot_rel = f"track_b_cleaning_files/seasonality_{stamp}.png"
    plot_seasonality(
        report["units_all"],
        report["units_selected"],
        report["closed_weeks"],
        REPORTS_DIR / plot_rel,
    )
    out = REPORTS_DIR / f"track_b_cleaning_{stamp}.md"
    out.write_text(cleaning_report(report, panel, plot_rel))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
