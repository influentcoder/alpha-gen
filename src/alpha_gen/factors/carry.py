"""Carry factors from perpetual-futures funding and the perp/spot basis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.factors.base import Factor, _check_positive, register


@register
@dataclass(frozen=True)
class FundingCarry(Factor):
    """Negative trailing mean funding rate.

    High positive funding means longs are crowded and paying shorts; the factor is bearish on it
    (and a short position collects the funding).
    """

    name = "funding_carry"
    requires = ("close", "funding_rate")
    description = "Short crowded longs: -mean(funding) over `lookback` bars"

    lookback: int = 72

    def __post_init__(self) -> None:
        _check_positive(lookback=self.lookback)

    def compute(self, data: MarketData) -> pd.DataFrame:
        funding = data.require("funding_rate").ffill()
        return -funding.rolling(self.lookback, min_periods=1).mean()


@register
@dataclass(frozen=True)
class BasisCarry(Factor):
    """Negative trailing mean perp premium over spot (``perp / spot - 1``)."""

    name = "basis_carry"
    requires = ("close", "spot_close")
    description = "Fade rich perps: -mean(perp/spot - 1) over `lookback` bars"

    lookback: int = 24

    def __post_init__(self) -> None:
        _check_positive(lookback=self.lookback)

    def compute(self, data: MarketData) -> pd.DataFrame:
        basis = data.close / data.require("spot_close") - 1.0
        return -basis.rolling(self.lookback, min_periods=1).mean()
