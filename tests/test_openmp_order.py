"""LightGBM must be imported before hierarchicalforecast in any process that fits both.

Known crash (`DECISIONS.md`, 2026-09-29 v1.2 addendum): `import hierarchicalforecast.core`, then
any LightGBM fit, segfaults on macOS arm64 (exit 139). Importing lightgbm first is fine. Only
`reorderpoint/reconcile.py` imports hierarchicalforecast and it never fits LightGBM, so nothing
in the package hits it; `tests/conftest.py` pins the safe order for the pytest process, which
hides the crash from every other test. This runs the supported order in a fresh interpreter, so the
order stays known to work, on whatever platform CI runs it, without relying on that pin.

Deliberately does not run the crashing order: that would assert a segfault on one platform's
behaviour, which is not a property this project controls.
"""

import subprocess
import sys

FIT = (
    "import numpy as np; import lightgbm as lgb; "
    "lgb.LGBMRegressor(n_estimators=20, verbose=-1)"
    ".fit(np.random.rand(500, 5), np.random.rand(500)); print('fit ok')"
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_lightgbm_first_then_hierarchicalforecast_then_a_fit_does_not_crash():
    result = _run(f"import lightgbm; import hierarchicalforecast.core; {FIT}")
    assert result.returncode == 0, (result.returncode, result.stderr[-500:])
    assert "fit ok" in result.stdout


def test_the_reconcile_module_imported_after_lightgbm_leaves_a_fit_working():
    # reconcile.py is the one module in the package that imports hierarchicalforecast
    result = _run(f"import lightgbm; import reorderpoint.reconcile; {FIT}")
    assert result.returncode == 0, (result.returncode, result.stderr[-500:])
