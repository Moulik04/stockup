"""Calibration-scheme tests: bucketing, both sizing forms, and the bootstrap CI."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from reorderpoint import safety_stock as ss


def _series(values, name=None):
    idx = pd.Index([f"S{i}" for i in range(len(values))], name="series_id")
    return pd.Series(values, index=idx, name=name)


def test_intermittency_buckets_reproduce_the_two_way_split():
    zero_rates = _series([0.1, 0.9, 0.5, 0.51])
    buckets = ss.assign_buckets(zero_rates, granularity="intermittency")
    assert list(buckets) == ["regular", "intermittent", "regular", "intermittent"]


def test_intermittency_volume_crosses_with_terciles():
    zero_rates = _series([0.9] * 6)
    volumes = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    buckets = ss.assign_buckets(zero_rates, volumes, granularity="intermittency_volume")
    assert set(buckets) == {"intermittent_lo", "intermittent_mid", "intermittent_hi"}
    assert buckets.iloc[0] == "intermittent_lo"
    assert buckets.iloc[-1] == "intermittent_hi"


def test_tied_volumes_still_split_into_three_groups():
    # Mostly-zero demand makes ties the norm; plain qcut would refuse to cut these.
    zero_rates = _series([0.9] * 9)
    volumes = _series([0.0] * 9)
    buckets = ss.assign_buckets(zero_rates, volumes, granularity="intermittency_volume")
    assert len(set(buckets)) == 3


def test_intermittency_volume_requires_volumes():
    with pytest.raises(ValueError):
        ss.assign_buckets(_series([0.9]), None, granularity="intermittency_volume")


def test_unknown_granularity_and_form_rejected():
    with pytest.raises(ValueError):
        ss.assign_buckets(_series([0.9]), granularity="nope")
    with pytest.raises(ValueError):
        ss.bucket_safety_stock(_series([1.0]), _series(["a"]), 0.95, form="nope")


def test_empirical_form_returns_the_residual_quantile_itself():
    residuals = _series(list(np.arange(0.0, 100.0)))
    buckets = _series(["a"] * 100)
    out = ss.bucket_safety_stock(residuals, buckets, 0.95, form="empirical", n_boot=200)
    assert out.loc["a", "safety_stock"] == pytest.approx(np.quantile(np.arange(0.0, 100.0), 0.95))


def test_normal_form_returns_z_times_std():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    residuals = _series(list(values))
    buckets = _series(["a"] * 5)
    out = ss.bucket_safety_stock(residuals, buckets, 0.95, form="normal", n_boot=200)
    assert out.loc["a", "safety_stock"] == pytest.approx(norm.ppf(0.95) * values.std(ddof=1))


def test_the_two_forms_disagree_on_a_right_skewed_distribution():
    """The whole premise of lever A: on a right-skewed residual distribution — which is what
    intermittent demand produces — the empirical quantile and z*std are materially different."""
    rng = np.random.default_rng(0)
    skewed = np.concatenate([np.zeros(180), rng.uniform(10, 60, size=20)])
    residuals = _series(list(skewed))
    buckets = _series(["a"] * len(skewed))

    normal = ss.bucket_safety_stock(residuals, buckets, 0.95, form="normal", n_boot=200)
    empirical = ss.bucket_safety_stock(residuals, buckets, 0.95, form="empirical", n_boot=200)
    assert normal.loc["a", "safety_stock"] != pytest.approx(
        empirical.loc["a", "safety_stock"], rel=0.05
    )


def test_bootstrap_ci_brackets_the_point_estimate_and_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small = rng.normal(0, 1, size=30)
    large = rng.normal(0, 1, size=3000)

    p_small, lo_small, hi_small = ss.bootstrap_quantile_ci(small, 0.95, n_boot=500)
    p_large, lo_large, hi_large = ss.bootstrap_quantile_ci(large, 0.95, n_boot=500)

    assert lo_small <= p_small <= hi_small
    assert lo_large <= p_large <= hi_large
    # the Task 5 sample-size point: a tail quantile from 30 points is far less pinned down
    assert (hi_small - lo_small) > (hi_large - lo_large)


def test_bootstrap_ci_handles_degenerate_inputs():
    assert all(np.isnan(v) for v in ss.bootstrap_quantile_ci(np.array([]), 0.95))
    point, lo, hi = ss.bootstrap_quantile_ci(np.array([5.0]), 0.95)
    assert (point, lo, hi) == (5.0, 5.0, 5.0)


def test_finer_granularity_leaves_fewer_observations_per_bucket():
    rng = np.random.default_rng(2)
    n = 300
    zero_rates = _series(list(rng.uniform(0, 1, size=n)))
    volumes = _series(list(rng.uniform(0, 10, size=n)))
    residuals = _series(list(rng.normal(0, 5, size=n)))

    _, coarse = ss.per_series_safety_stock(
        residuals, zero_rates, volumes, 0.95, "empirical", "intermittency", n_boot=200
    )
    _, fine = ss.per_series_safety_stock(
        residuals, zero_rates, volumes, 0.95, "empirical", "intermittency_volume", n_boot=200
    )
    assert len(coarse) == 2
    assert len(fine) == 6
    assert fine["n"].min() < coarse["n"].min()
    # and the finer buckets' quantile estimates are correspondingly less certain
    assert fine["ci_width_frac"].median() > coarse["ci_width_frac"].median()


def test_per_series_safety_stock_maps_each_series_to_its_bucket():
    zero_rates = _series([0.1, 0.1, 0.9, 0.9])
    residuals = _series([1.0, 2.0, 30.0, 40.0])
    mapped, table = ss.per_series_safety_stock(
        residuals, zero_rates, None, 0.95, "empirical", "intermittency", n_boot=200
    )
    assert mapped["S0"] == mapped["S1"] == table.loc["regular", "safety_stock"]
    assert mapped["S2"] == mapped["S3"] == table.loc["intermittent", "safety_stock"]
    assert mapped["S2"] > mapped["S0"]


def test_within_bucket_dispersion_exposes_an_order_of_magnitude_spread():
    stds = _series([1.0, 2.0, 5.0, 50.0])
    buckets = _series(["a", "a", "a", "a"])
    out = ss.within_bucket_dispersion(stds, buckets)
    assert out.loc["a", "n"] == 4
    assert out.loc["a", "p90_over_p10"] > 10
