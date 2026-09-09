"""Schema tests for both ingest tracks — run against fixtures shaped like the real files.

Track A fixtures mirror M5's actual column layout (calendar.csv, sell_prices.csv,
sales_train_validation.csv) so this exercises the real melt/join logic, not a simplification.
"""

from pathlib import Path

import pandas as pd
import pytest

from reorderpoint.ingest import CORE_COLUMNS, load_track_a, load_track_b

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def track_a_panel() -> pd.DataFrame:
    return load_track_a(FIXTURES / "m5_sample")


@pytest.fixture
def track_b_panel() -> pd.DataFrame:
    return load_track_b(FIXTURES / "track_b_sample.csv")


def test_track_a_has_core_columns(track_a_panel):
    assert set(CORE_COLUMNS).issubset(track_a_panel.columns)


def test_track_a_shape_matches_series_times_days(track_a_panel):
    # 2 series x 5 days in the fixture
    assert len(track_a_panel) == 10
    assert track_a_panel["series_id"].nunique() == 2


def test_track_a_price_join_is_correct(track_a_panel):
    prices = track_a_panel.set_index("series_id")["price"].to_dict()
    assert prices["HOBBIES_1_001_CA_1_validation"] == pytest.approx(5.98)
    assert prices["HOBBIES_1_002_CA_1_validation"] == pytest.approx(3.50)


def test_track_a_snap_matches_state(track_a_panel):
    # fixture is all CA_1 / state CA; snap_CA is 0,0,1,1,1 across d_1..d_5
    one_series = track_a_panel[
        track_a_panel["series_id"] == "HOBBIES_1_001_CA_1_validation"
    ].sort_values("date")
    assert list(one_series["snap"]) == [0, 0, 1, 1, 1]


def test_track_a_cat_filter():
    filtered = load_track_a(FIXTURES / "m5_sample", cat_ids=["HOBBIES"])
    assert (filtered["cat_id"] == "HOBBIES").all()

    empty = load_track_a(FIXTURES / "m5_sample", cat_ids=["FOODS"])
    assert empty.empty


def test_track_b_has_core_columns(track_b_panel):
    assert set(CORE_COLUMNS).issubset(track_b_panel.columns)


def test_track_b_shape(track_b_panel):
    assert len(track_b_panel) == 4
    assert track_b_panel["series_id"].nunique() == 2


def test_track_b_missing_required_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("sku,date\nFAN-1,2023-01-01\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_track_b(bad)


def test_tracks_produce_compatible_core_dtypes(track_a_panel, track_b_panel):
    for col in CORE_COLUMNS:
        a_dtype = track_a_panel[col].dtype
        b_dtype = track_b_panel[col].dtype
        if col == "date":
            assert str(a_dtype).startswith("datetime64")
            assert str(b_dtype).startswith("datetime64")
        elif col in {"y", "price"}:
            assert pd.api.types.is_numeric_dtype(a_dtype)
            assert pd.api.types.is_numeric_dtype(b_dtype) or track_b_panel[col].isna().all()
