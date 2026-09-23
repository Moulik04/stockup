"""The Croston family via statsforecast — the models built for intermittent demand.

v1.1 Task 4: on a 77%-zero panel, the modelling ladder never included the models designed for
exactly this problem. `CrostonClassic` and `CrostonSBA` (Syntetos-Boylan bias-corrected) split
demand into inter-demand interval and non-zero size, each smoothed separately; `TSB`
(Teunter-Syntetos-Babai) replaces the interval with a demand *probability*, smoothed the same
way, which — unlike Croston — can adapt to demand that stops altogether (obsolescence).

None of the three produce native prediction intervals (`only_conformal_intervals=True` on TSB;
Croston has none either), so — like every other rung in this ladder — they go through the same
conformal-interval wrapper (`StatsForecastQuantileModel`) the ETS/Theta baselines use, rather than
a bespoke interval scheme just for this family.
"""

from __future__ import annotations

from statsforecast.models import TSB as _TSB
from statsforecast.models import CrostonClassic as _CrostonClassic
from statsforecast.models import CrostonSBA as _CrostonSBA
from statsforecast.utils import ConformalIntervals

from reorderpoint.models.base import StatsForecastQuantileModel

# TSB has no `Auto` variant and statsforecast ships no default — 0.1/0.1 is the smoothing pair
# used in the original Teunter-Syntetos-Babai (2011) worked examples and in most reference
# implementations (e.g. R's tsintermittent). Not tuned against this panel; stated, not fitted,
# matching how this project treats every other unsourced default (docs/decision.md).
TSB_ALPHA_D = 0.1
TSB_ALPHA_P = 0.1


def croston_classic(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _CrostonClassic(prediction_intervals=ci),
        name="CrostonClassic",
    )


def croston_sba(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _CrostonSBA(prediction_intervals=ci),
        name="CrostonSBA",
    )


def tsb(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _TSB(alpha_d=TSB_ALPHA_D, alpha_p=TSB_ALPHA_P, prediction_intervals=ci),
        name="TSB",
    )
