"""Test-session setup.

lightgbm must be imported before hierarchicalforecast, whichever test module happens to be
collected first. Otherwise a later LightGBM fit segfaults on macOS — reproducible outside pytest:
import `hierarchicalforecast.core` (or `.methods`, `.utils`), then fit any LGBMRegressor. Likely
two OpenMP runtimes in one process (a dependency's bundled libomp loaded before lightgbm's), which
is a guess about the mechanism; the ordering that avoids it is verified.

The ordering used to hold by accident: importing `reorderpoint.backtest` imported every model, and
so lightgbm, first. The model registry is lazy now (so serving does not load the ladder), which
made the order depend on test-file collection order. Nothing in `reorderpoint/` both imports
hierarchicalforecast (`reconcile.py`) and fits LightGBM, so production scripts are unaffected;
this pins the order for the one process that runs both.
"""

import lightgbm  # noqa: F401
