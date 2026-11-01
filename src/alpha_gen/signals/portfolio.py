"""Turn signals into portfolio weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_weights(
    signal: pd.DataFrame,
    *,
    gross: float = 1.0,
    market_neutral: bool = True,
    volatility: pd.DataFrame | None = None,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Scale a signal into weights with ``sum(|w|) == gross`` per row.

    Args:
        signal: Scores per bar and symbol (higher = longer).
        gross: Gross exposure per row (1.0 = 100% of capital, long + short).
        market_neutral: Demean across symbols so longs and shorts balance.
        volatility: Optional per-symbol volatility; scores are divided by it (risk weighting)
            before neutralising.
        max_weight: Optional cap on absolute weight per symbol, applied after scaling (rows are
            not renormalised, so gross can end up below target).
    """
    scores = signal.copy()
    if volatility is not None:
        scores = scores / volatility.reindex_like(scores).replace(0.0, np.nan)
    if market_neutral:
        scores = scores.sub(scores.mean(axis=1), axis=0)
    total = scores.abs().sum(axis=1).replace(0.0, np.nan)
    weights = scores.div(total, axis=0) * gross
    if max_weight is not None:
        weights = weights.clip(lower=-max_weight, upper=max_weight)
    return weights.fillna(0.0)


def volatility_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    target: float,
    periods_per_year: float,
    window: int = 24 * 30,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Rescale weights so the strategy's trailing realised volatility tracks ``target``.

    Realised vol at ``t`` uses strategy returns up to ``t`` (positions from ``t - 1``), so the
    scale applied at ``t`` is known at ``t``. Before the window warms up the scale is 1.
    """
    strategy_returns = (weights.shift(1) * returns).sum(axis=1)
    realised = strategy_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(
        periods_per_year
    )
    scale = (target / realised.replace(0.0, np.nan)).clip(upper=max_leverage).fillna(1.0)
    return weights.mul(scale, axis=0)


def rebalance(weights: pd.DataFrame, every: str) -> pd.DataFrame:
    """Only trade on bars aligned to ``every`` (e.g. ``"8h"``, ``"1D"``); hold in between.

    A bar is a rebalance bar when its timestamp is a multiple of ``every`` since the epoch, so
    ``"8h"`` on hourly data trades at 00:00, 08:00 and 16:00 UTC (the Binance funding times).
    """
    index = weights.index
    assert isinstance(index, pd.DatetimeIndex)
    on_schedule = index == index.floor(every)
    held = weights.where(pd.Series(on_schedule, index=index), axis=0)
    return held.ffill().fillna(0.0)


def smooth(weights: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Exponentially smooth weights through time to cut turnover (uses past rows only)."""
    return weights.ewm(halflife=halflife, adjust=False).mean()


def time_series_trend(
    close: pd.DataFrame, lookback: int, *, weight: float = 0.5, symbols: list[str] | None = None
) -> pd.DataFrame:
    """Directional trend: ``weight * sign(close[t] / close[t - lookback] - 1)`` per symbol.

    Unlike the cross-sectional books this is *not* market neutral: it is long assets in an
    uptrend and short those in a downtrend.
    """
    panel = close[symbols] if symbols is not None else close
    trend = np.sign(panel / panel.shift(lookback) - 1.0)
    return (trend * weight).reindex(columns=close.columns).fillna(0.0)
