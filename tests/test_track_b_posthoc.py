"""The post hoc calibration variants, on inputs whose answers are known.

`reorderpoint/track_b_posthoc.py` re-implements how the registered simulation sizes a buffer and
varies it. These tests pin the pieces, and the check that the re-implementation reproduces the
registered run: its first version encoded a wrong description of the simulation and failed
that check.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from reorderpoint import track_b_posthoc as ph

TARGET = 0.95
Z = norm.ppf(TARGET)


def _fold_forecasts(n_series: int, weeks: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-03", periods=weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        scale = 5.0 * (i + 1)  # series of very different size
        for d in dates:
            rows.append(
                {
                    "series_id": f"s{i}",
                    "date": d,
                    "p50": scale,
                    "y": scale + rng.normal(0, scale / 5),
                }
            )
    return pd.DataFrame(rows)


def _train(n_series: int, weeks: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2010-01-04", periods=weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        for j, d in enumerate(dates):
            # every series sells in most weeks except the last two, which are intermittent
            rows.append({"series_id": f"s{i}", "date": d, "y": float((j % 4 != 0) * (i + 1))})
    return pd.DataFrame(rows)


def test_only_complete_lead_time_windows_are_used():
    fc = _fold_forecasts(n_series=3, weeks=13)
    blocks = ph._complete_blocks(fc, lead=2)
    assert len(blocks) == 3 * 6  # six full 2-week windows per series; week 13 is dropped
    one = blocks.loc["s0"]
    assert one["p50"].tolist() == pytest.approx([10.0] * 6)


def test_all_windows_buffer_is_z_times_the_pooled_sigma_of_every_window():
    fc = _fold_forecasts(n_series=6, weeks=13)
    blocks = ph._complete_blocks(fc, lead=2)
    residual = (blocks["y"] - blocks["p50"]).droplevel("block")
    train = _train(6)
    out = ph.all_windows_buffer(fc, train, lead=2, target=TARGET, pooling="volume_quintile")
    assert set(out.index) == {f"s{i}" for i in range(6)}
    # each series' buffer is z times the std of the windows of the series in its own bucket
    from reorderpoint import safety_stock as ss

    zr, vol = ss.train_stats(train)
    buckets = ss.assign_buckets(zr, vol, "volume_quintile")
    for sid in out.index:
        peers = buckets[buckets == buckets[sid]].index
        expected = Z * residual.loc[residual.index.isin(peers)].std(ddof=1)
        assert out[sid] == pytest.approx(expected)


def test_volume_quintiles_scale_the_buffer_with_series_size_where_one_pool_does_not():
    fc = _fold_forecasts(n_series=10, weeks=13)
    train = _train(10)
    pooled = ph.all_windows_buffer(fc, train, lead=2, target=TARGET, pooling="intermittency")
    by_size = ph.all_windows_buffer(fc, train, lead=2, target=TARGET, pooling="volume_quintile")
    # in one absolute pool the smallest and largest series get the same buffer (same bucket) ...
    assert pooled["s0"] == pytest.approx(pooled["s9"])
    # ... with volume buckets the larger series get a larger buffer
    assert by_size["s9"] > 3 * by_size["s0"]


def test_single_window_gives_every_fold_the_fold_one_buffer():
    calib = pd.DataFrame(
        {
            "model": "MovingAverage",
            "series_id": [f"s{i}" for i in range(20)],
            "residual": np.random.default_rng(0).normal(0, 10, 20),
            "zero_rate": 0.1,
            "volume": np.arange(1.0, 21.0),
        }
    )
    first = ph.registered_buffers(calib, TARGET, "intermittency")
    assert first.notna().all()
    # one pool: every series gets z x the std of the twenty residuals
    assert first.iloc[0] == pytest.approx(Z * calib["residual"].std(ddof=1))


def test_the_check_against_the_registered_run_passes_on_equal_and_raises_on_unequal():
    cell = {"lot0": {"A": {"cost": 100.0, "csl": 0.9}}, "lot2": {"A": {"cost": 80.0, "csl": 0.95}}}
    registered = {"cells": {ph.tb.cell_key(ph.tb.DEFAULT_MARGIN, ph.tb.DEFAULT_LEAD): cell}}
    ph.check_against_registered({k: dict(v) for k, v in cell.items()}, registered)
    wrong = {"lot0": {"A": {"cost": 101.0, "csl": 0.9}}, "lot2": cell["lot2"]}
    with pytest.raises(AssertionError, match="disagrees with the registered run"):
        ph.check_against_registered(wrong, registered)


def test_the_grid_is_three_windows_by_two_poolings_with_the_registered_cell_marked():
    assert len(ph.VARIANTS) == 6
    assert ph.REGISTERED == ("first_window", "intermittency")
    assert ph.REGISTERED in ph.VARIANTS
