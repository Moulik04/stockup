"""CI smoke test: a real (not dummy-model) end-to-end backtest run on a small synthetic panel.

Uses only the naive models (SeasonalNaive/MovingAverage) — cheap, no LightGBM/libomp dependency —
so this stays fast enough to run on every CI push while still exercising the real
`run_backtest`/`write_report` path, not a mock of it.

The panel needs >= 4 folds x 28-day horizon (112 days) of test coverage plus enough leading
history for the earliest fold's train window — `run_backtest` calls `make_folds(eval_panel)` with
no explicit horizon/n_folds, so it always uses the real HORIZON=28/N_FOLDS=4 constants regardless
of any `monkeypatch.setattr(bt, "HORIZON", ...)` (those are `make_folds`'s *default* argument
values, bound once at module-import time — monkeypatching the module attribute afterward doesn't
touch an already-bound default). Learned this by direct verification, not assumption: an earlier
draft of this test used a 90-day panel with monkeypatched HORIZON=7/N_FOLDS=2, which silently kept
using the real 28/4 and produced a fold whose train window started before the panel's first date
(0 groups) — surfacing as a statsforecast internal error.

Also builds the models with `n_jobs=1` rather than the production default (`n_jobs=-1`): with
n_jobs=-1, every one of the 8 StatsForecast() calls here (4 folds x 2 models) spins up a fresh
multiprocessing Pool — on macOS's `spawn` start method each worker re-imports the whole
environment (pandas/numpy/statsforecast), so 8 pool startups measurably dominate the runtime for
what's otherwise 5 trivial series (confirmed directly: real work, not a hang — but far slower than
the actual computation). `n_jobs=1` skips pool creation entirely, which is simply the right choice
for a workload this small regardless of environment.
"""

import pandas as pd
from statsforecast.models import SeasonalNaive as _SeasonalNaive
from statsforecast.models import WindowAverage as _WindowAverage
from statsforecast.utils import ConformalIntervals

from reorderpoint import backtest as bt
from reorderpoint.models.base import StatsForecastQuantileModel
from reorderpoint.models.naive import SEASON_LENGTH, WINDOW_SIZE


def _seasonal_naive_single_threaded(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _SeasonalNaive(season_length=SEASON_LENGTH, prediction_intervals=ci),
        name="SeasonalNaive",
        n_jobs=1,
    )


def _moving_average_single_threaded(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _WindowAverage(window_size=WINDOW_SIZE, prediction_intervals=ci),
        name="WindowAverage",
        n_jobs=1,
    )


def _synthetic_panel(n_series: int = 5, n_days: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for i in range(n_series):
        for j, d in enumerate(dates):
            rows.append(
                {
                    "series_id": f"S{i}",
                    "date": d,
                    "y": float((j + i) % 7),
                    "price": 10.0 + i,
                    "dept_id": f"DEPT_{i % 2}",
                }
            )
    return pd.DataFrame(rows)


def test_smoke_backtest_runs_end_to_end(monkeypatch):
    monkeypatch.setattr(
        bt,
        "MODEL_FACTORIES",
        {
            "SeasonalNaive": _seasonal_naive_single_threaded,
            "MovingAverage": _moving_average_single_threaded,
        },
    )
    monkeypatch.setattr(bt, "N_SERIES_SAMPLE", 5)

    panel = _synthetic_panel()

    results, detail = bt.run_backtest(panel)
    assert not results.empty
    assert set(results["model"]) == {"SeasonalNaive", "MovingAverage"}
    assert {"mase", "rmsse", "coverage_80"}.issubset(results.columns)

    dept_lookup = panel.drop_duplicates("series_id").set_index("series_id")["dept_id"]
    report = bt.write_report(results, detail, dept_lookup)
    assert "Backtest report" in report
    assert "SeasonalNaive" in report
