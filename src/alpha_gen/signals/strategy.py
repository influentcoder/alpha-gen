"""A named, reusable bundle: universe -> factors -> combination -> sizing -> schedule.

The same :class:`Strategy` object drives research backtests and the live runner, so a paper or
live signal is computed by exactly the code that was backtested.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.data.universe import (
    DEFAULT_EXCLUDE,
    apply_universe,
    listing_age_mask,
    top_n_membership,
)
from alpha_gen.factors.base import Factor
from alpha_gen.signals.combine import combine
from alpha_gen.signals.portfolio import rebalance, to_weights
from alpha_gen.signals.transforms import cross_sectional_rank, cross_sectional_zscore

NORMALIZERS = {"rank": cross_sectional_rank, "zscore": cross_sectional_zscore}


@dataclass
class Strategy:
    """Multi-factor, dollar-neutral cross-sectional strategy.

    Args:
        name: Identifier used in signal logs. Change it whenever the logic changes.
        factors: Label -> factor.
        weights: Label -> combination weight (equal weights if omitted).
        normalize: ``"rank"`` (robust, default) or ``"zscore"`` (winsorised at 3).
        universe_top_n: If set, only trade the top N symbols by the *previous calendar month's*
            quote volume (``data.volume`` must be loaded).
        universe_exclude: Symbols never traded (stablecoins, index perps by default).
        min_listing_age_days: If set, skip symbols that have traded for fewer days.
        vol_lookback: If set, divide scores by trailing volatility over this many bars
            (risk weighting) before neutralising.
        rebalance_every: Trade only on bars aligned to this frequency (e.g. ``"1D"``).

    Example::

        Strategy(
            name="momentum_v0",
            factors={"momo": Momentum(lookback=14)},
            universe_top_n=30,
            rebalance_every="1D",
        )
    """

    name: str
    factors: Mapping[str, Factor]
    weights: Mapping[str, float] | None = None
    normalize: str = "rank"
    gross: float = 1.0
    market_neutral: bool = True
    max_weight: float | None = None
    universe_top_n: int | None = None
    universe_exclude: frozenset[str] = DEFAULT_EXCLUDE
    min_listing_age_days: int | None = None
    vol_lookback: int | None = None
    rebalance_every: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("A strategy needs at least one factor")
        if self.normalize not in NORMALIZERS:
            raise ValueError(f"normalize must be one of {sorted(NORMALIZERS)}")
        if self.weights is not None:
            unknown = set(self.weights) - set(self.factors)
            if unknown:
                raise ValueError(f"Weights for unknown factors: {sorted(unknown)}")

    # ------------------------------------------------------------------ pipeline
    def tradable(self, data: MarketData) -> pd.DataFrame | None:
        """Boolean mask of symbols the strategy may trade at each bar (None = everything)."""
        mask: pd.DataFrame | None = None
        if self.universe_top_n is not None:
            monthly = data.require("volume").resample("MS").sum(min_count=1)
            membership = top_n_membership(
                monthly, self.universe_top_n, exclude=self.universe_exclude
            )
            ones = pd.DataFrame(1.0, index=data.index, columns=data.symbols)
            mask = apply_universe(ones, membership).notna()
        if self.min_listing_age_days is not None:
            age = listing_age_mask(data.close, pd.Timedelta(days=self.min_listing_age_days))
            mask = age if mask is None else mask & age
        return mask

    def factor_scores(self, data: MarketData) -> dict[str, pd.DataFrame]:
        mask = self.tradable(data)
        scores = {label: factor(data) for label, factor in self.factors.items()}
        if mask is not None:
            scores = {label: s.where(mask) for label, s in scores.items()}
        return scores

    def signal(self, data: MarketData) -> pd.DataFrame:
        normalizer = NORMALIZERS[self.normalize]
        return combine(
            self.factor_scores(data),
            self.weights,
            normalize=normalizer,
            clip=3.0 if self.normalize == "zscore" else None,
        )

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        volatility = None
        if self.vol_lookback is not None:
            volatility = (
                data.returns().rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
            )
        weights = to_weights(
            self.signal(data),
            gross=self.gross,
            market_neutral=self.market_neutral,
            volatility=volatility,
            max_weight=self.max_weight,
        )
        if self.rebalance_every is not None:
            weights = rebalance(weights, self.rebalance_every)
        return weights

    def is_rebalance_bar(self, timestamp: pd.Timestamp) -> bool:
        if self.rebalance_every is None:
            return True
        return bool(timestamp == timestamp.floor(self.rebalance_every))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "factors": {k: f.label for k, f in self.factors.items()},
            "weights": dict(self.weights) if self.weights else "equal",
            "normalize": self.normalize,
            "gross": self.gross,
            "market_neutral": self.market_neutral,
            "max_weight": self.max_weight,
            "universe_top_n": self.universe_top_n,
            "min_listing_age_days": self.min_listing_age_days,
            "vol_lookback": self.vol_lookback,
            "rebalance_every": self.rebalance_every,
            **self.metadata,
        }
