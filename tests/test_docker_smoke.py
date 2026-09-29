"""The CI image check's fixture and expectations, verified without Docker.

`scripts/docker_smoke.py` compares a running container to numbers worked out by hand. If those
numbers were wrong, CI would fail for a reason that has nothing to do with the image. This runs the
same fixture through the real serving code in-process and requires it to agree, and requires the
comparison to notice when it does not.
"""

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reorderpoint import calibration as cal
from reorderpoint import serve

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "docker_smoke.py"
spec = importlib.util.spec_from_file_location("docker_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def answers(tmp_path, monkeypatch):
    smoke.write_fixture(tmp_path)
    monkeypatch.setattr(serve, "MODEL_PATH", tmp_path / "models/production/model.joblib")
    monkeypatch.setattr(serve, "PANEL_PATH", tmp_path / "data/track_a/processed/panel.parquet")
    calibration_path = tmp_path / "models/production/safety_stock_calibration.joblib"
    monkeypatch.setattr(serve, "_load_calibration", lambda: cal.load(calibration_path))
    for loader in (serve._load_model, serve._load_panel):
        loader.cache_clear()
    monkeypatch.setattr(serve, "get_config", lambda: _config())
    serve.app.dependency_overrides[serve.get_config] = _config
    client = TestClient(serve.app)
    reorder = client.post(
        "/reorder",
        json={"series_ids": [smoke.SERIES_A, smoke.SERIES_B], "on_hand": {smoke.SERIES_A: 3.0}},
    ).json()
    forecast = client.post(
        "/forecast", json={"series_ids": [smoke.SERIES_A, smoke.SERIES_B], "horizon": 3}
    ).json()
    metrics = client.get("/metrics").json()
    yield reorder, forecast, metrics
    serve.app.dependency_overrides.clear()
    for loader in (serve._load_model, serve._load_panel):
        loader.cache_clear()


def _config():
    from reorderpoint.config import Config, CostParams

    return Config(
        track="a",
        track_b_data_path=None,
        costs=CostParams(
            holding_cost_rate=0.25 / 365,
            stockout_penalty_per_unit=1.0,
            lead_time_days=smoke.LEAD_TIME,
            service_level_target=smoke.SERVICE_LEVEL,
            lot_multiple=smoke.LOT_MULTIPLE,
        ),
    )


def test_the_hand_derived_numbers_match_the_real_serving_code(answers):
    assert smoke.check_answers(*answers) == []


def test_the_comparison_notices_a_wrong_reorder_point(answers):
    reorder, forecast, metrics = answers
    reorder[0]["reorder_point"] += 0.5
    problems = smoke.check_answers(reorder, forecast, metrics)
    assert any("reorder_point" in p for p in problems)


def test_the_comparison_notices_an_unclipped_or_wrong_band(answers):
    reorder, forecast, metrics = answers
    forecast[0]["p10"] = -1.0
    forecast[1]["p90"] = 9.0
    problems = smoke.check_answers(reorder, forecast, metrics)
    assert any("p10" in p for p in problems) and any("p90" in p for p in problems)


def test_the_comparison_notices_the_wrong_model(answers):
    reorder, forecast, metrics = answers
    metrics["safety_stock_calibration"]["model"] = "LightGBM"
    assert any("model" in p for p in smoke.check_answers(reorder, forecast, metrics))


def test_the_expected_numbers_are_the_ones_worked_out_in_the_docstring():
    want = smoke.expected()
    assert want["safety_stock"] == pytest.approx(2.3262, abs=1e-3)  # z(0.95) * sqrt(2)
    assert want["reorder_point"] == pytest.approx(16.3262, abs=1e-3)  # 14 + safety stock
    assert want["band_half_width"] == pytest.approx(0.6851, abs=1e-3)  # z(0.90) sqrt(2) / sqrt(7)
