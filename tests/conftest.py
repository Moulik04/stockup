"""Test-session setup.

lightgbm must be imported before hierarchicalforecast, whichever test module happens to be
collected first. Otherwise a later LightGBM fit segfaults (exit 139) on macOS arm64. Reproduction,
outside pytest and independent of this project's code:

    python -c "import hierarchicalforecast.core; import numpy as np, lightgbm as lgb; \\
               lgb.LGBMRegressor(n_estimators=20).fit(np.random.rand(500, 5), np.random.rand(500))"

exits 139 every time; put `import lightgbm` first and it exits 0. Importing sklearn, statsforecast,
numba or scipy.sparse.linalg first does not trigger it — it is specific to hierarchicalforecast —
and the mechanism is not known. (An OpenMP-runtime clash was the first guess; sklearn's bundled
runtime, the obvious suspect, was ruled out.) `DECISIONS.md`, 2026-09-29, has the record.

The ordering used to hold by accident: importing `reorderpoint.backtest` imported every model, and
so lightgbm, first. The model registry is lazy now (so serving does not load the ladder), which
made the order depend on test-file collection order. Nothing in `reorderpoint/` both imports
hierarchicalforecast (`reconcile.py`) and fits LightGBM, so production scripts are unaffected. This
pins the order for the one process that runs both, which hides the crash from every other test;
`tests/test_openmp_order.py` checks the supported order in a fresh interpreter instead.
"""

import lightgbm  # noqa: F401
