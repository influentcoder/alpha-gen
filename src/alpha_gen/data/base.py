"""The data-source protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from alpha_gen.data.market import MarketData


@runtime_checkable
class DataSource(Protocol):
    """Anything that can produce a :class:`MarketData` for a universe and a time range."""

    @property
    def cache_key(self) -> str:
        """Stable identifier used by :class:`~alpha_gen.data.cache.CachedSource`."""
        ...

    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str = "1h",
    ) -> MarketData:
        """Load bars whose open time is in ``[start, end)``."""
        ...
