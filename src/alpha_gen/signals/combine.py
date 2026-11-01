"""Combine several factor panels into one signal."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from alpha_gen.evaluation.metrics import information_coefficient
from alpha_gen.signals.transforms import cross_sectional_zscore, winsorize

Normalizer = Callable[[pd.DataFrame], pd.DataFrame]


def _normalize_all(
    factors: Mapping[str, pd.DataFrame], normalize: Normalizer, clip: float | None
) -> dict[str, pd.DataFrame]:
    if not factors:
        raise ValueError("Need at least one factor to combine")
    out = {}
    for name, panel in factors.items():
        z = normalize(panel)
        out[name] = winsorize(z, clip) if clip is not None else z
    return out


def _weighted_sum(
    scores: Mapping[str, pd.DataFrame], weights: Mapping[str, float | pd.Series]
) -> pd.DataFrame:
    """NaN-aware weighted average: a missing factor drops out instead of zeroing the signal."""
    template = next(iter(scores.values()))
    num = pd.DataFrame(0.0, index=template.index, columns=template.columns)
    den = num.copy()
    for name, z in scores.items():
        w = weights[name]
        if isinstance(w, pd.Series):
            w_frame = pd.DataFrame(
                np.repeat(w.reindex(z.index).fillna(0.0).to_numpy()[:, None], z.shape[1], axis=1),
                index=z.index,
                columns=z.columns,
            )
        else:
            w_frame = pd.DataFrame(float(w), index=z.index, columns=z.columns)
        num += (w_frame * z).fillna(0.0)
        den += w_frame.abs().where(z.notna(), 0.0)
    return num / den.replace(0.0, np.nan)


def combine(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
    *,
    normalize: Normalizer = cross_sectional_zscore,
    clip: float | None = 3.0,
) -> pd.DataFrame:
    """Static-weight blend of normalised factors (equal weights by default)."""
    scores = _normalize_all(factors, normalize, clip)
    w: dict[str, float | pd.Series] = {
        name: (1.0 if weights is None else float(weights.get(name, 0.0))) for name in scores
    }
    if all(v == 0 for v in w.values()):
        raise ValueError("All combination weights are zero")
    return _weighted_sum(scores, w)


def ic_weighted_combine(
    factors: Mapping[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    horizon: int,
    *,
    window: int = 24 * 30,
    min_periods: int | None = None,
    normalize: Normalizer = cross_sectional_zscore,
    clip: float | None = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weight each factor by its trailing mean IC, without look-ahead.

    The IC measured at ``t`` uses returns up to ``t + horizon``, so it is only usable from
    ``t + horizon`` onwards; weights are shifted accordingly. A factor whose trailing IC turns
    negative gets a negative weight (it is flipped).

    Returns:
        ``(signal, weights)`` where ``weights`` has one column per factor.
    """
    scores = _normalize_all(factors, normalize, clip)
    ic_weights = {}
    for name, z in scores.items():
        ic = information_coefficient(z, forward_returns).shift(horizon)
        ic_weights[name] = ic.rolling(window, min_periods=min_periods or window // 4).mean()
    weight_frame = pd.DataFrame(ic_weights)
    signal = _weighted_sum(scores, {k: weight_frame[k] for k in scores})
    return signal, weight_frame
