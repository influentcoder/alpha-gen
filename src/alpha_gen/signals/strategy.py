"""A named, reusable bundle of factors + combination + sizing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.factors.base import Factor
from alpha_gen.signals.combine import combine
from alpha_gen.signals.portfolio import to_weights


@dataclass
class Strategy:
    """Static-weight multi-factor strategy.

    Example::

        Strategy(
            name="carry_momo",
            factors={"carry": FundingCarry(), "momo": Momentum(lookback=72)},
            weights={"carry": 2.0, "momo": 1.0},
        )
    """

    name: str
    factors: Mapping[str, Factor]
    weights: Mapping[str, float] | None = None
    gross: float = 1.0
    market_neutral: bool = True
    max_weight: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def factor_scores(self, data: MarketData) -> dict[str, pd.DataFrame]:
        return {label: factor(data) for label, factor in self.factors.items()}

    def signal(self, data: MarketData) -> pd.DataFrame:
        return combine(self.factor_scores(data), self.weights)

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        return to_weights(
            self.signal(data),
            gross=self.gross,
            market_neutral=self.market_neutral,
            max_weight=self.max_weight,
        )

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "factors": {k: f.label for k, f in self.factors.items()},
            "weights": dict(self.weights) if self.weights else "equal",
            "gross": self.gross,
            "market_neutral": self.market_neutral,
            "max_weight": self.max_weight,
            **self.metadata,
        }
