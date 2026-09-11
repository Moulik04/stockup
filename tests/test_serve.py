"""FastAPI wiring tests. Uses a dummy model + a tiny in-memory panel via dependency overrides —
no real LightGBM fit, no real file I/O, same "don't exercise the slow real thing" convention as
test_backtest.py's dummy model.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from reorderpoint import serve
from reorderpoint.config import Config, CostParams


class _DummyModel:
    """Constant P10=1/P50=2/P90=3 forecast for every requested series/date."""

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict_quantiles(self, horizon: int, future_exog: pd.DataFrame | None = None):
        dates = sorted(future_exog["date"].unique())
        rows = [
            {"series_id": sid, "date": d, "p10": 1.0, "p50": 2.0, "p90": 3.0}
            for sid in future_exog["series_id"].unique()
            for d in dates
        ]
        return pd.DataFrame(rows)


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for sid in ["A", "B"]:
        for i, d in enumerate(dates):
            rows.append({"series_id": sid, "date": d, "y": float(i % 5), "price": 10.0})
    return pd.DataFrame(rows)


def _config() -> Config:
    return Config(
        track="a",
        track_b_data_path=None,
        costs=CostParams(
            holding_cost_rate=0.02,
            stockout_penalty_per_unit=5.0,
            lead_time_days=7,
            service_level_target=0.95,
        ),
    )


@pytest.fixture
def client():
    app = serve.app
    app.dependency_overrides[serve.get_model] = lambda: _DummyModel()
    app.dependency_overrides[serve.get_history] = _panel
    app.dependency_overrides[serve.get_config] = _config
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_forecast_returns_quantiles_for_requested_series(client):
    resp = client.post("/forecast", json={"series_ids": ["A"], "horizon": 7})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 7
    assert all(row["series_id"] == "A" for row in body)
    assert body[0]["p10"] == 1.0
    assert body[0]["p50"] == 2.0
    assert body[0]["p90"] == 3.0


def test_forecast_unknown_series_is_404(client):
    resp = client.post("/forecast", json={"series_ids": ["NOPE"], "horizon": 7})
    assert resp.status_code == 404


def test_reorder_returns_decision_fields(client):
    resp = client.post("/reorder", json={"series_ids": ["A"], "on_hand": {"A": 5.0}})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    row = body[0]
    assert row["series_id"] == "A"
    for field in ["reorder_point", "safety_stock", "order_quantity", "stockout_probability"]:
        assert field in row
    assert "order" in row["rationale"].lower() or "no order" in row["rationale"].lower()


def test_reorder_defaults_on_hand_to_zero_when_not_given(client):
    resp = client.post("/reorder", json={"series_ids": ["B"]})
    assert resp.status_code == 200
    assert resp.json()[0]["series_id"] == "B"


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "n_series" in body
    assert body["n_series"] == 2


def test_future_exog_wday_matches_m5_convention():
    # M5's wday is Saturday=1..Friday=7 (verified against the real calendar.csv), not pandas'
    # Monday=0 dayofweek. 2011-01-28 is a real M5 Friday (wday=7); the day after it (panel's
    # "last known date") should roll over to Saturday (wday=1) as the first future date.
    dates = pd.date_range("2011-01-01", "2011-01-28", freq="D")  # ends on a Friday
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(dates),
            "date": dates,
            "y": [1.0] * len(dates),
            "wday": [0] * len(dates),  # deliberately wrong placeholder — must be overwritten
        }
    )
    future = serve.future_exog_from_trailing_window(panel, ["A"], horizon=7)
    future = future.sort_values("date").reset_index(drop=True)

    # 2011-01-29 (Sat) .. 2011-02-04 (Fri) -> wday 1,2,3,4,5,6,7
    assert future["wday"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_future_exog_day_one_is_consistent_across_horizons():
    # Regression test: an earlier version sized its reference window to `horizon` itself
    # (group.tail(horizon)), so day 1's proxy price silently changed depending on how many days
    # were requested -- a different historical day became "day 1 of the template" for every
    # horizon length. Each day gets a distinct price so any such shift is directly observable.
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(dates),
            "date": dates,
            "y": [1.0] * len(dates),
            "price": [float(i) for i in range(len(dates))],  # every day has a unique price
        }
    )

    for horizon in (7, 14, 28):
        future = serve.future_exog_from_trailing_window(panel, ["A"], horizon)
        future = future.sort_values("date")
        day_one_price = future.iloc[0]["price"]
        if horizon == 7:
            reference_price = day_one_price
        else:
            assert day_one_price == reference_price, (
                f"day 1 price changed with horizon={horizon}: "
                f"{day_one_price} != {reference_price}"
            )
