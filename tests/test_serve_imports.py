"""Serving must not load the model ladder.

The served model is a trailing mean; lightgbm and statsforecast belong to the backtest ladder. They
used to be dragged in anyway (`serve` -> `decision` -> `backtest` -> every model module), which is
what kept the container image carrying an OpenMP runtime the served path never uses. Checked in a
fresh interpreter: this process has already imported both through other tests.
"""

import subprocess
import sys

import pytest

HEAVY = ("lightgbm", "statsforecast", "numba")


def _loaded_after_importing(module: str) -> list[str]:
    code = (
        "import sys, importlib; "
        f"importlib.import_module({module!r}); "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(",") if m]


@pytest.mark.parametrize(
    "module",
    [
        "reorderpoint.serve",
        "reorderpoint.train",
        "reorderpoint.calibration",
        "reorderpoint.decision",
    ],
)
def test_importing_the_serving_path_leaves_the_ladder_libraries_unloaded(module):
    assert _loaded_after_importing(module) == []


def test_the_registry_still_builds_every_model_on_demand():
    from reorderpoint import backtest as bt

    assert list(bt.MODEL_FACTORIES) == [
        "SeasonalNaive",
        "MovingAverage",
        "AutoETS",
        "AutoTheta",
        "LightGBM",
        "CrostonClassic",
        "CrostonSBA",
        "TSB",
    ]
    model = bt.MODEL_FACTORIES["MovingAverage"](28)  # imports statsforecast only now
    assert hasattr(model, "fit") and hasattr(model, "predict_quantiles")
