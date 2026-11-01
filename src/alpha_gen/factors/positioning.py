"""Positioning factors built on open interest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.factors.base import Factor, _check_positive, register


@register
@dataclass(frozen=True)
class OpenInterestTrend(Factor):
    """Open-interest change signed by the price move over the same window.

    Rising OI with rising price (new longs) or falling price (new shorts) is read as trend
    confirmation; falling OI (positions closing) as exhaustion. Hypothesis, not established fact:
    validate it with :func:`alpha_gen.evaluation.evaluate_factor` before use.
    """

    name = "oi_trend"
    requires = ("close", "open_interest")
    description = "OI % change x sign(price change) over `lookback` bars"

    lookback: int = 24

    def __post_init__(self) -> None:
        _check_positive(lookback=self.lookback)

    def compute(self, data: MarketData) -> pd.DataFrame:
        oi = data.require("open_interest")
        oi_change = oi / oi.shift(self.lookback) - 1.0
        price_change = data.returns(self.lookback)
        return oi_change * np.sign(price_change)
