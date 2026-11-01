"""Synthetic markets with *planted* alpha, for tests, demos and sanity-checking the pipeline.

If a factor cannot find alpha you deliberately planted, the evaluation code is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.timeframes import timeframe_to_freq


def _ar1(rng: np.random.Generator, shape: tuple[int, int], phi: float, scale: float) -> np.ndarray:
    out = np.empty(shape)
    out[0] = rng.normal(0.0, scale / np.sqrt(1 - phi**2), shape[1])
    shocks = rng.normal(0.0, scale, shape)
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _cs_z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def make_synthetic_market(
    n_symbols: int = 20,
    n_periods: int = 24 * 120,
    timeframe: str = "1h",
    *,
    funding_alpha: float = 0.5,
    momentum_alpha: float = 0.0,
    funding_interval: int = 8,
    start: str = "2024-01-01",
    seed: int = 0,
) -> MarketData:
    """Generate perp/spot prices, funding and open interest.

    Next-bar expected returns are ``-funding_alpha * 0.05 * vol * z(funding)``
    plus ``momentum_alpha * 0.05 * vol * z(trailing 24-bar return)``, where ``z`` is
    a cross-sectional z-score. With the defaults, high funding predicts lower returns (the
    crowded-long effect that :class:`~alpha_gen.factors.carry.FundingCarry` targets).
    """
    rng = np.random.default_rng(seed)
    shape = (n_periods, n_symbols)
    index = pd.date_range(start, periods=n_periods, freq=timeframe_to_freq(timeframe), tz="UTC")
    symbols = [f"SYN{i:02d}/USDT:USDT" for i in range(n_symbols)]

    vol = rng.uniform(0.004, 0.015, n_symbols)
    funding_latent = 1e-4 + _ar1(rng, shape, phi=0.98, scale=2e-5)
    noise = rng.normal(0.0, 1.0, shape) * vol

    returns = np.zeros(shape)
    for t in range(1, n_periods):
        drift = -funding_alpha * 0.05 * vol * _cs_z(funding_latent[t - 1 : t])[0]
        if momentum_alpha and t > 24:
            trailing = returns[t - 24 : t].sum(axis=0, keepdims=True)
            drift += momentum_alpha * 0.05 * vol * _cs_z(trailing)[0]
        returns[t] = drift + noise[t]
    close = 100.0 * np.exp(np.cumsum(np.log1p(returns), axis=0))

    funding = np.full(shape, np.nan)
    funding[::funding_interval] = funding_latent[::funding_interval]
    basis = 5e-4 + _ar1(rng, shape, phi=0.95, scale=1e-4)
    open_interest = 1e7 * np.exp(np.cumsum(rng.normal(0.0, 0.01, shape), axis=0))
    volume = rng.lognormal(mean=12.0, sigma=0.5, size=shape)

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=index, columns=symbols)

    return MarketData(
        close=frame(close),
        volume=frame(volume),
        funding_rate=frame(funding),
        open_interest=frame(open_interest),
        spot_close=frame(close / (1.0 + basis)),
        timeframe=timeframe,
    )
