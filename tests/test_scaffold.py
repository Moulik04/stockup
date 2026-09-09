"""Phase 0 smoke test: the package imports and config loads with defaults."""

from reorderpoint.config import load_config


def test_default_config_is_track_a():
    config = load_config()
    assert config.track == "a"


def test_cost_params_have_sane_defaults():
    config = load_config()
    assert config.costs.service_level_target == 0.95
    assert config.costs.lead_time_days > 0
