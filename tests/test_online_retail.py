"""Track B cleaning rules on hand-built transaction frames — nothing here reads the real workbook.

Each rule is one small case whose result can be checked on paper: what a rule removes, what it must
leave alone, and where the boundary is.
"""

import numpy as np
import pandas as pd
import pytest

from reorderpoint import online_retail as orr

T0 = pd.Timestamp("2010-03-01 10:00")  # a Monday inside the panel


def _raw(rows, sheet="Year 2010-2011"):
    """rows: (invoice, stock_code, qty, when, price, customer, country) -> load_raw's layout."""
    df = pd.DataFrame(
        rows,
        columns=[
            "invoice",
            "stock_code",
            "quantity",
            "invoice_date",
            "price_gbp",
            "customer",
            "country",
        ],
    )
    df["sheet"] = sheet
    df["description"] = "ITEM " + df["stock_code"]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def _sale(inv, sku, qty, when=T0, price=2.0, cust=1.0, country="United Kingdom"):
    return (inv, sku, qty, when, price, cust, country)


def _clean(rows):
    return orr.clean_lines(_raw(rows))


def _step(log, prefix):
    return next(s for s in log.steps if s.rule.startswith(prefix))


# ---- rules that drop rows ---------------------------------------------------------------------


def test_the_overlap_between_sheets_is_kept_once_and_only_if_the_copies_agree():
    overlap = [
        _sale("500001", "10001", 3, "2010-12-02 09:00"),
        _sale("500002", "10002", 4, "2010-12-03 09:00"),
    ]
    first = _raw(
        [_sale("400001", "10001", 1, "2010-06-01 09:00"), *overlap], sheet="Year 2009-2010"
    )
    second = _raw(
        [*overlap, _sale("600001", "10001", 9, "2011-03-01 09:00")], sheet="Year 2010-2011"
    )
    both = pd.concat([first, second], ignore_index=True)
    kept = orr.drop_sheet_overlap(both)
    assert len(kept) == len(both) - 2
    assert (kept["quantity"].sum()) == 1 + 3 + 4 + 9

    changed = second.copy()
    changed.loc[0, "quantity"] = 99  # the copies now disagree
    with pytest.raises(ValueError, match="disagree"):
        orr.drop_sheet_overlap(pd.concat([first, changed], ignore_index=True))


def test_stock_codes_that_differ_only_by_case_or_whitespace_are_one_product():
    lines, log = _clean(
        [
            _sale("500001", "15056BL", 5),
            _sale("500002", "15056bl", 3),
            _sale("500003", "47503J ", 2),
            _sale("500004", "m", 1),  # a manual entry stays non-product once upper-cased
        ]
    )
    assert sorted(lines["stock_code"]) == ["15056BL", "15056BL", "47503J"]
    step = _step(log, "stock codes normalised")
    assert "3 rows changed" in step.note  # 15056bl, "47503J ", m
    assert "1 codes merged" in step.note  # 15056bl into 15056BL ("47503J " and m gain no twin)


def test_only_five_digit_codes_with_a_short_letter_suffix_are_products():
    codes = [
        "85048",
        "79323P",
        "15056BL",
        "POST",
        "DOT",
        "M",
        "BANK CHARGES",
        "gift_0001_20",
        "DCGS0058",
        "TEST001",
    ]
    lines, log = _clean([_sale(f"5000{i:02d}", c, 1) for i, c in enumerate(codes)])
    assert sorted(lines["stock_code"]) == ["15056BL", "79323P", "85048"]
    assert _step(log, "non-product").rows_removed == 7


def test_a_negative_quantity_that_is_not_a_cancellation_is_a_write_off_and_dropped():
    lines, log = _clean(
        [_sale("500001", "10001", 5), _sale("500002", "10001", -7, cust=np.nan, price=0.0)]
    )
    assert lines["quantity"].sum() == 5
    assert _step(log, "negative quantity").returned_units_removed == 7


def test_zero_price_lines_are_dropped_not_counted_as_demand():
    lines, log = _clean([_sale("500001", "10001", 5), _sale("500002", "10001", 100, price=0.0)])
    assert lines["quantity"].sum() == 5
    step = _step(log, "zero or negative price")
    assert (step.rows_removed, step.sold_units_removed) == (1, 100)


# ---- cancellations ----------------------------------------------------------------------------


def _net(rows):
    lines = _raw(rows)
    return orr.net_cancellations(lines)


def test_a_same_day_reversal_removes_the_order_entirely():
    net, stats = _net(
        [
            _sale("500001", "10001", 80_000, "2010-03-01 09:15"),
            (
                "C500002",
                "10001",
                -80_000,
                pd.Timestamp("2010-03-01 09:27"),
                2.0,
                1.0,
                "United Kingdom",
            ),
            _sale("500003", "10001", 5, "2010-03-02 09:00", cust=2.0),
        ]
    )
    assert net["quantity"].tolist() == [5.0]
    assert stats["matched_units"] == 80_000 and stats["unmatched_units"] == 0


def test_a_partial_return_reduces_the_original_sale_not_the_week_it_was_recorded_in():
    net, _ = _net(
        [
            _sale("500001", "10001", 10, "2010-03-01 09:00"),
            ("C500002", "10001", -4, pd.Timestamp("2010-03-16 09:00"), 2.0, 1.0, "United Kingdom"),
        ]
    )
    assert net["quantity"].tolist() == [6.0]
    assert net["invoice_date"].dt.date.astype(str).tolist() == ["2010-03-01"]  # the sale's date


def test_a_cancellation_takes_from_the_most_recent_earlier_sale_first():
    net, _ = _net(
        [
            _sale("500001", "10001", 10, "2010-03-01 09:00"),
            _sale("500002", "10001", 10, "2010-03-08 09:00"),
            ("C500003", "10001", -12, pd.Timestamp("2010-03-09 09:00"), 2.0, 1.0, "United Kingdom"),
        ]
    )
    # the March 8 sale is gone entirely; the other 2 of the 12 come off March 1's 10
    assert net["quantity"].tolist() == [8.0]
    assert net["invoice_date"].dt.strftime("%Y-%m-%d").tolist() == ["2010-03-01"]


def test_unmatched_cancellations_are_dropped_not_netted_into_their_own_week():
    rows = [
        _sale("500001", "10001", 10, "2010-01-01 09:00"),
        (
            "C500002",
            "10001",
            -3,
            pd.Timestamp("2010-06-01 09:00"),
            2.0,
            1.0,
            "United Kingdom",
        ),  # 151 days on
        (
            "C500003",
            "10001",
            -2,
            pd.Timestamp("2010-03-01 09:00"),
            2.0,
            2.0,
            "United Kingdom",
        ),  # other customer
        (
            "C500004",
            "10001",
            -1,
            pd.Timestamp("2010-03-01 09:00"),
            2.0,
            np.nan,
            "United Kingdom",
        ),  # no customer
        (
            "C500005",
            "10002",
            -5,
            pd.Timestamp("2010-03-01 09:00"),
            2.0,
            1.0,
            "United Kingdom",
        ),  # never sold
    ]
    net, stats = _net(rows)
    assert net["quantity"].tolist() == [10.0]  # nothing was reversed
    assert stats["matched_units"] == 0 and stats["unmatched_units"] == 3 + 2 + 1 + 5


def test_a_cancellation_larger_than_the_sale_reverses_it_and_drops_the_rest():
    net, stats = _net(
        [
            _sale("500001", "10001", 4, "2010-03-01 09:00"),
            ("C500002", "10001", -10, pd.Timestamp("2010-03-02 09:00"), 2.0, 1.0, "United Kingdom"),
        ]
    )
    assert net.empty
    assert (stats["matched_units"], stats["unmatched_units"]) == (4, 6)


# ---- weekly panel -----------------------------------------------------------------------------


def _year(sku_weeks, qty=5, price=2.0):
    """One sale per listed panel week (0-based) for each SKU, on that week's Wednesday."""
    rows = []
    for sku, weeks in sku_weeks.items():
        for w in weeks:
            when = orr.FIRST_WEEK + pd.Timedelta(weeks=w, days=2, hours=10)
            rows.append(_sale(f"5{w:03d}{sku}", sku, qty, when, price))
    return _raw(rows)


def test_weeks_start_on_monday_and_partial_weeks_at_the_edges_are_excluded():
    before = _sale("500001", "10001", 7, "2009-12-03 09:00")  # the week before FIRST_WEEK
    inside = _sale("500002", "10001", 5, "2009-12-13 23:00")  # a Sunday: still the first week
    monday = _sale("500003", "10001", 2, "2009-12-14 00:30")  # Monday: the second week
    after = _sale("500004", "10001", 9, "2011-12-05 09:00")  # the week after LAST_WEEK
    lines, log = _clean([before, inside, monday, after])
    assert sorted(lines["week"].dt.strftime("%Y-%m-%d")) == ["2009-12-07", "2009-12-14"]
    assert _step(log, "partial").rows_removed == 2
    assert orr.LAST_WEEK == pd.Timestamp("2011-11-28") and orr.N_WEEKS == 104


def test_the_panel_is_dense_with_zeros_for_weeks_with_no_sale():
    raw = _year({"10001": range(0, 104, 2)})
    panel, _ = orr.build_panel(raw)
    assert len(panel) == 104 and panel["date"].nunique() == 104
    assert panel["y"].sum() == 5 * 52 and int((panel["y"] == 0).sum()) == 52
    assert panel["date"].dt.dayofweek.eq(0).all()


def test_selection_needs_half_of_the_first_year_and_ignores_the_second():
    raw = _year(
        {
            "10001": range(0, 26),  # 26 of the first 52 weeks: in
            "10002": range(0, 25),  # 25: out
            "10003": range(52, 104),  # a launch in year 2: out
            "10004": range(0, 104, 2),  # 26 in year 1: in
        }
    )
    panel, report = orr.build_panel(raw)
    assert sorted(panel["series_id"].unique()) == ["10001", "10004"]
    assert report["qualifying"][26] == 2


def test_price_is_the_weekly_median_in_dollars_and_outlier_lines_never_touch_demand():
    normal = [_sale(f"5000{i}", "10001", 1, "2010-03-02 09:00", price=2.0) for i in range(3)]
    outlier = _sale("500009", "10001", 40, "2010-03-02 10:00", price=200.0)  # 100x the median
    raw = pd.concat([_year({"10001": range(0, 104)}), _raw([*normal, outlier])], ignore_index=True)
    panel, _ = orr.build_panel(raw)
    week = panel[panel["date"] == pd.Timestamp("2010-03-01")].iloc[0]
    assert week["price"] == pytest.approx(2.0 * orr.GBP_TO_USD)  # the 200 line is not in the price
    assert week["y"] == 5 + 3 + 40  # but its 40 units are demand


def test_price_carries_forward_over_weeks_with_no_sale():
    raw = _year({"10001": [0, 1, 2] + list(range(3, 104, 2))}, price=3.0)
    panel, _ = orr.build_panel(raw)
    assert panel["price"].notna().all()
    assert panel["price"].nunique() == 1 and panel["price"].iloc[0] == pytest.approx(
        3.0 * orr.GBP_TO_USD
    )


def test_closed_weeks_are_the_weeks_where_nothing_sold_anywhere():
    raw = _year({"10001": [w for w in range(104) if w not in (3, 55)]})
    _, report = orr.build_panel(raw)
    assert report["closed_weeks"] == [
        (orr.FIRST_WEEK + pd.Timedelta(weeks=3)).date(),
        (orr.FIRST_WEEK + pd.Timedelta(weeks=55)).date(),
    ]


# ---- the ingest hook --------------------------------------------------------------------------


def test_load_track_b_returns_the_canonical_columns(tmp_path):
    from reorderpoint.ingest import CORE_COLUMNS, load_track_b

    workbook = tmp_path / "online_retail_II.xlsx"
    workbook.write_bytes(b"")  # only its mtime matters: the parquet cache beside it is read
    original = _year({"10001": range(0, 104)}).rename(
        columns={
            "invoice": "Invoice",
            "stock_code": "StockCode",
            "description": "Description",
            "quantity": "Quantity",
            "invoice_date": "InvoiceDate",
            "price_gbp": "Price",
            "customer": "Customer ID",
            "country": "Country",
        }
    )
    original.to_parquet(workbook.with_suffix(".parquet"), index=False)
    panel = load_track_b(workbook)
    assert list(panel.columns) == CORE_COLUMNS
    assert str(panel["date"].dtype).startswith("datetime64")
    assert pd.api.types.is_numeric_dtype(panel["y"]) and pd.api.types.is_numeric_dtype(
        panel["price"]
    )
    assert panel["series_id"].nunique() == 1 and len(panel) == 104


def test_the_fixed_exchange_rate_is_the_documented_one():
    assert orr.GBP_TO_USD == 1.577
