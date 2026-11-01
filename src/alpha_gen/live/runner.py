"""Produce the latest target weights for a strategy and (optionally) log them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from alpha_gen.data.base import DataSource
from alpha_gen.live.signal_log import SignalLog, SignalRecord
from alpha_gen.signals.strategy import Strategy
from alpha_gen.timeframes import timeframe_to_offset


class SignalRunner:
    """Fetch recent history, compute the strategy and emit weights for the last complete bar.

    Schedule :meth:`run_once` with cron/systemd/Airflow right after each bar closes.
    """

    def __init__(
        self,
        strategy: Strategy,
        source: DataSource,
        symbols: Sequence[str],
        *,
        timeframe: str = "1h",
        history_bars: int = 24 * 30,
        log: SignalLog | None = None,
    ) -> None:
        self.strategy = strategy
        self.source = source
        self.symbols = list(symbols)
        self.timeframe = timeframe
        self.history_bars = history_bars
        self.log = log

    def latest_weights(self, now: Any = None) -> tuple[pd.Timestamp, pd.Series]:
        end = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        end = end.tz_localize("UTC") if end.tz is None else end
        start = end - timeframe_to_offset(self.timeframe) * self.history_bars
        data = self.source.fetch(self.symbols, start, end, self.timeframe)
        weights = self.strategy.target_weights(data)
        timestamp = pd.Timestamp(weights.index[-1])
        return timestamp, weights.iloc[-1]

    def run_once(self, now: Any = None) -> SignalRecord | tuple[pd.Timestamp, pd.Series]:
        """Compute the latest weights; append them to the log if one is configured."""
        timestamp, weights = self.latest_weights(now)
        if self.log is None:
            return timestamp, weights
        return self.log.append(
            timestamp,
            self.strategy.name,
            weights.to_dict(),
            metadata={"timeframe": self.timeframe, "strategy": self.strategy.describe()},
        )
