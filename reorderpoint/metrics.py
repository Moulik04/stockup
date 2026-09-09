"""MASE, RMSSE, (simplified) WRMSSE, pinball loss, and quantile coverage.

All functions operate on one series' arrays at a time; backtest.py aggregates across series.
Formulas follow the M5 competition definitions (Hyndman & Koehler's scale-free errors), except
WRMSSE, which is a documented simplification (see `weighted_average` below).
"""

from __future__ import annotations

import numpy as np


def _in_sample_scale(y_train: np.ndarray, season_length: int, squared: bool) -> float:
    """Mean (squared) seasonal one-step-ahead naive error over the training window."""
    if len(y_train) <= season_length:
        raise ValueError(
            f"y_train has {len(y_train)} points, need > season_length ({season_length})"
        )
    diffs = y_train[season_length:] - y_train[:-season_length]
    errors = diffs**2 if squared else np.abs(diffs)
    scale = errors.mean()
    return scale if scale > 0 else np.nan


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season_length: int) -> float:
    scale = _in_sample_scale(y_train, season_length, squared=False)
    if np.isnan(scale):
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season_length: int) -> float:
    scale = _in_sample_scale(y_train, season_length, squared=True)
    if np.isnan(scale):
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Revenue-weighted average of a per-series metric (e.g. RMSSE) — this project's WRMSSE.

    Not the official M5 12-level hierarchical WRMSSE; a single bottom-up (series-level)
    weighting by each series' training-window revenue share.
    """
    mask = ~np.isnan(values)
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
