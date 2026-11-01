"""Cross-sectional and time-series normalisation of factor panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(panel: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """Z-score each row across symbols. Rows with fewer than ``min_count`` values become NaN."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = panel.sub(mean, axis=0).div(std, axis=0)
    z.loc[panel.notna().sum(axis=1) < min_count] = np.nan
    return z


def cross_sectional_rank(panel: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """Rank each row across symbols, scaled to ``[-0.5, 0.5]`` (robust to outliers)."""
    ranks = panel.rank(axis=1, method="average")
    count = panel.notna().sum(axis=1)
    scaled = ranks.sub(1.0).div((count - 1).replace(0, np.nan), axis=0) - 0.5
    scaled.loc[count < min_count] = np.nan
    return scaled


def winsorize(panel: pd.DataFrame, limit: float = 3.0) -> pd.DataFrame:
    """Clip values to ``[-limit, limit]``; apply after z-scoring."""
    return panel.clip(lower=-limit, upper=limit)


def time_series_zscore(
    panel: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    """Z-score each symbol against its own trailing window (uses data up to and including ``t``)."""
    rolling = panel.rolling(window, min_periods=min_periods or max(2, window // 2))
    return (panel - rolling.mean()) / rolling.std().replace(0.0, np.nan)
