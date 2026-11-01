"""Price-only factors: momentum, reversal, low volatility."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.factors.base import Factor, _check_positive, register


@register
@dataclass(frozen=True)
class Momentum(Factor):
    """Trailing return from ``t - lookback`` to ``t - skip``."""

    name = "momentum"
    description = "close[t-skip] / close[t-lookback] - 1"

    lookback: int = 168
    skip: int = 0

    def __post_init__(self) -> None:
        _check_positive(lookback=self.lookback)
        if not 0 <= self.skip < self.lookback:
            raise ValueError("skip must satisfy 0 <= skip < lookback")

    def compute(self, data: MarketData) -> pd.DataFrame:
        return data.close.shift(self.skip) / data.close.shift(self.lookback) - 1.0


@register
@dataclass(frozen=True)
class ShortTermReversal(Factor):
    """Negative trailing return: bet on short-horizon mean reversion."""

    name = "reversal"
    description = "-(close[t] / close[t-lookback] - 1)"

    lookback: int = 24

    def __post_init__(self) -> None:
        _check_positive(lookback=self.lookback)

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -data.returns(self.lookback)


@register
@dataclass(frozen=True)
class LowVolatility(Factor):
    """Negative trailing realised volatility of bar returns."""

    name = "low_vol"
    description = "-std(returns) over `lookback` bars"

    lookback: int = 168

    def __post_init__(self) -> None:
        _check_positive(lookback=self.lookback)
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2 to compute a standard deviation")

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -data.returns().rolling(self.lookback, min_periods=self.lookback // 2).std()
