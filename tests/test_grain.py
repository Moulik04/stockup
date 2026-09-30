"""The period abstraction: Track A holds the old constants, Track B the registered ones.

The grain is fixed per process from `REORDERPOINT_TRACK` at import, so the weekly checks run in a
fresh interpreter. `tests/test_track_a_regression.py` proves Track A's numbers did not move; this
pins the constants and, for Track B, the fold dates `docs/track_b.md` registered.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest

from reorderpoint import backtest as bt
from reorderpoint.grain import DAILY, WEEKLY, for_track


def test_the_daily_grain_is_exactly_what_was_hard_coded_before():
    assert (DAILY.horizon, DAILY.n_folds, DAILY.mase_lag) == (28, 4, 7)
    assert (DAILY.naive_season, DAILY.model_season, DAILY.ma_window) == (7, 7, 28)
    assert (DAILY.default_lead_time, DAILY.n_series_sample, DAILY.freq) == (7, 400, "D")
    assert DAILY.lags == (1, 7, 14, 28) and DAILY.rolling_windows == (7, 28)
    assert (DAILY.price_change_lag, DAILY.lgb_history, DAILY.periods_per_year) == (7, 90, 365)
    assert DAILY.period == pd.Timedelta(days=1)


def test_the_weekly_grain_is_the_registered_design():
    assert (WEEKLY.horizon, WEEKLY.n_folds, WEEKLY.default_lead_time) == (13, 4, 2)
    assert WEEKLY.naive_season == 52 and WEEKLY.model_season == 1 and WEEKLY.ma_window == 4
    assert WEEKLY.mase_lag == 1 and WEEKLY.n_series_sample is None and WEEKLY.freq == "W-MON"
    assert WEEKLY.periods_per_year == 52 and WEEKLY.period == pd.Timedelta(weeks=1)


def test_track_a_is_daily_here_and_the_backtest_constants_follow_the_grain():
    assert for_track("a") is DAILY and for_track("b") is WEEKLY
    with pytest.raises(ValueError):
        for_track("c")
    assert (bt.HORIZON, bt.N_FOLDS, bt.SEASON_LENGTH, bt.N_SERIES_SAMPLE) == (28, 4, 7, 400)
    assert bt.PERIOD == pd.Timedelta(days=1)


def test_period_arithmetic():
    start, end = pd.Timestamp("2011-09-05"), pd.Timestamp("2011-11-28")
    assert WEEKLY.periods_in(start, end) == 13
    assert DAILY.periods_in(pd.Timestamp("2016-03-28"), pd.Timestamp("2016-04-24")) == 28
    assert WEEKLY.before(end, 12) == start
    assert WEEKLY.annual_to_period(0.25) == pytest.approx(0.25 / 52)
    assert DAILY.annual_to_period(0.25) == 0.25 / 365  # identical to the old annual_to_daily


def _weekly(code: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"REORDERPOINT_TRACK": "b", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_track_b_folds_are_the_registered_ones_and_cover_the_final_52_weeks():
    code = (
        "import json, pandas as pd\n"
        "from reorderpoint import backtest as bt\n"
        "w = pd.date_range('2009-12-07', periods=104, freq='W-MON')\n"
        "panel = pd.DataFrame({'series_id': 'a', 'date': w, 'y': 1.0})\n"
        "f = bt.make_folds(panel)\n"
        "consts = [bt.HORIZON, bt.N_FOLDS, bt.SEASON_LENGTH, bt.N_SERIES_SAMPLE]\n"
        "print(json.dumps({'consts': consts,\n"
        " 'folds': [[x.index, str(x.train_end.date()), str(x.test_start.date()),\n"
        "            str(x.test_end.date())] for x in f]}))\n"
    )
    got = _weekly(code)
    assert got["consts"] == [13, 4, 1, None]
    assert got["folds"] == [
        [1, "2010-11-29", "2010-12-06", "2011-02-28"],
        [2, "2011-02-28", "2011-03-07", "2011-05-30"],
        [3, "2011-05-30", "2011-06-06", "2011-08-29"],
        [4, "2011-08-29", "2011-09-05", "2011-11-28"],
    ]


def test_track_b_defaults_are_two_weeks_and_a_weekly_holding_rate():
    code = (
        "import json\n"
        "from reorderpoint.config import load_config\n"
        "c = load_config().costs\n"
        "print(json.dumps([c.lead_time_days, c.lead_time_periods, c.holding_cost_rate,"
        " c.service_level_target, c.lot_multiple, c.gross_margin]))\n"
    )
    lead, periods, holding, target, lot, margin = _weekly(code)
    assert lead == periods == 2
    assert holding == pytest.approx(0.25 / 52)
    assert (target, lot, margin) == (0.95, 2.0, 0.275)
