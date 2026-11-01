"""Produce the latest target weights for a strategy and append them to a signal log.

Designed to be run by a scheduler (cron, GitHub Actions) shortly after each bar closes:

* **Idempotent**: if the latest complete bar is already logged, it does nothing.
* **Safe**: refuses to emit a signal from stale data (exchange outage, wrong clock).
* **Self-reporting**: each new record carries the paper P&L realised by the previous weights since
  the previous record, so the log *is* the track record.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from alpha_gen.data.base import DataSource
from alpha_gen.evaluation.backtest import CostModel
from alpha_gen.live.signal_log import SignalLog, SignalRecord
from alpha_gen.live.track import realized_returns, realized_summary
from alpha_gen.signals.strategy import Strategy
from alpha_gen.timeframes import timeframe_to_offset

log = logging.getLogger(__name__)


class StaleDataError(RuntimeError):
    """The newest bar returned by the source is too old to trade on."""


@dataclass(frozen=True)
class RunResult:
    status: Literal["logged", "already_logged", "not_rebalance_bar"]
    timestamp: pd.Timestamp
    weights: pd.Series
    record: SignalRecord | None = None


class SignalRunner:
    """Run a :class:`Strategy` on recent data from a :class:`DataSource`.

    Args:
        strategy: What to compute.
        source: Where bars come from. If ``symbols`` is None the source must offer
            ``list_symbols()`` (e.g. :class:`~alpha_gen.data.ccxt_source.CCXTSource`).
        symbols: Fixed universe to fetch, or None to discover symbols every run.
        timeframe: Bar length.
        history_bars: How much history to fetch. Must cover the longest factor lookback and, for a
            volume universe, the whole previous calendar month.
        log: Where to append signals. Without it, :meth:`run_once` only returns the weights.
        pnl_source: Source used to price the previous weights (needs funding). Defaults to
            ``source``.
        costs: Cost model for the realised paper P&L.
        max_staleness_bars: Raise :class:`StaleDataError` if the newest complete bar is older.
    """

    def __init__(
        self,
        strategy: Strategy,
        source: DataSource,
        symbols: Sequence[str] | None = None,
        *,
        timeframe: str = "1h",
        history_bars: int = 24 * 30,
        log: SignalLog | None = None,
        pnl_source: DataSource | None = None,
        costs: CostModel | None = None,
        max_staleness_bars: int = 2,
    ) -> None:
        self.strategy = strategy
        self.source = source
        self.symbols = list(symbols) if symbols is not None else None
        self.timeframe = timeframe
        self.history_bars = history_bars
        self.log = log
        self.pnl_source = pnl_source or source
        self.costs = costs or CostModel()
        self.max_staleness_bars = max_staleness_bars

    # ------------------------------------------------------------------ helpers
    def _now(self, now: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")

    def _universe(self) -> list[str]:
        if self.symbols is not None:
            return self.symbols
        lister = getattr(self.source, "list_symbols", None)
        if lister is None:
            raise ValueError("symbols=None requires a source with list_symbols()")
        return list(lister())

    def latest_weights(self, now: Any = None) -> tuple[pd.Timestamp, pd.Series]:
        """Target weights decided at the close of the latest complete bar."""
        end = self._now(now)
        bar = timeframe_to_offset(self.timeframe)
        start = end - bar * self.history_bars
        data = self.source.fetch(self._universe(), start, end, self.timeframe)
        complete = data.index[data.index + bar <= end]
        if len(complete) == 0:
            raise StaleDataError(f"No complete {self.timeframe} bar before {end}")
        latest = pd.Timestamp(complete[-1])
        if latest + bar * (self.max_staleness_bars + 1) <= end:
            raise StaleDataError(f"Newest complete bar {latest} is stale at {end}")
        weights = self.strategy.target_weights(data.slice(end=latest))
        return latest, weights.iloc[-1]

    def _realized(
        self, previous: list[SignalRecord], until: pd.Timestamp, now: pd.Timestamp
    ) -> dict[str, Any] | None:
        if not previous:
            return None
        last = pd.Timestamp(previous[-1].timestamp)
        held = sorted({s for r in previous[-2:] for s, w in r.weights.items() if w != 0.0})
        if not held:
            return {"from": str(last), "to": str(until), **_zero_pnl()}
        bar = timeframe_to_offset(self.timeframe)
        data = self.pnl_source.fetch(held, last - bar, now, self.timeframe).slice(end=until)
        frame = realized_returns(previous, data, self.costs, start=last, end=until)
        return {"from": str(last), "to": str(until), **realized_summary(frame)}

    # ------------------------------------------------------------------ public API
    def run_once(self, now: Any = None) -> RunResult:
        """Compute the latest weights and, if a log is configured, append them once."""
        end = self._now(now)
        timestamp, weights = self.latest_weights(end)
        if not self.strategy.is_rebalance_bar(timestamp):
            return RunResult("not_rebalance_bar", timestamp, weights)
        if self.log is None:
            return RunResult("logged", timestamp, weights)
        previous = [r for r in self.log.records() if r.strategy == self.strategy.name]
        if previous and pd.Timestamp(previous[-1].timestamp) >= timestamp:
            return RunResult("already_logged", timestamp, weights)
        metadata: dict[str, Any] = {
            "timeframe": self.timeframe,
            "strategy": self.strategy.describe(),
        }
        realized = self._realized(previous, timestamp, end)
        if realized is not None:
            metadata["realized"] = realized
        nonzero = {k: float(v) for k, v in weights.items() if v != 0.0}
        record = self.log.append(timestamp, self.strategy.name, nonzero, metadata)
        log.info("Logged %s @ %s (%d positions)", self.strategy.name, timestamp, len(nonzero))
        return RunResult("logged", timestamp, weights, record)


def _zero_pnl() -> dict[str, float | int]:
    return {"gross": 0.0, "costs": 0.0, "funding": 0.0, "net": 0.0, "bars": 0}
