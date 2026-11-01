"""The :class:`MarketData` container that every other layer consumes.

All panels are *wide* DataFrames: a UTC :class:`~pandas.DatetimeIndex` of bar open times and one
column per symbol. A value labelled ``t`` is assumed to be known at the **close** of bar ``t``;
factors computed at ``t`` may use it, and positions decided at ``t`` earn the return of bar
``t + 1`` (see :func:`alpha_gen.evaluation.backtest.run_backtest`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

PANEL_FIELDS = ("close", "volume", "funding_rate", "open_interest", "spot_close")


class MissingDataError(KeyError):
    """Raised when a factor needs a panel that the :class:`MarketData` does not carry."""


def _validate_panel(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(panel).__name__}")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be indexed by a DatetimeIndex")
    index = panel.index
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out = panel.set_axis(index, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.columns = out.columns.astype(str)
    return out.astype(float)


@dataclass(frozen=True)
class MarketData:
    """Aligned panels for a universe of perpetual-futures symbols.

    Attributes:
        close: Perp close prices. Defines the index and the symbol universe.
        volume: Bar volume in **quote currency** (USDT), comparable across symbols (optional).
        funding_rate: Total funding rate settled **within each bar** (summed when a bar holds
            several settlements, NaN in bars without one). Factors usually forward-fill it; the
            backtester charges it only where it is set.
        open_interest: Open interest, ideally in quote currency (optional).
        spot_close: Spot close prices for the same underlyings, used for basis (optional).
        timeframe: Bar length as an exchange timeframe string, e.g. ``"1h"``.
    """

    close: pd.DataFrame
    volume: pd.DataFrame | None = None
    funding_rate: pd.DataFrame | None = None
    open_interest: pd.DataFrame | None = None
    spot_close: pd.DataFrame | None = None
    timeframe: str = "1h"

    def __post_init__(self) -> None:
        close = _validate_panel(self.close, "close")
        object.__setattr__(self, "close", close)
        for name in PANEL_FIELDS[1:]:
            panel = getattr(self, name)
            if panel is not None:
                aligned = _validate_panel(panel, name).reindex(
                    index=close.index, columns=close.columns
                )
                object.__setattr__(self, name, aligned)

    # ------------------------------------------------------------------ accessors
    @property
    def index(self) -> pd.DatetimeIndex:
        idx = self.close.index
        assert isinstance(idx, pd.DatetimeIndex)
        return idx

    @property
    def symbols(self) -> list[str]:
        return [str(c) for c in self.close.columns]

    def has(self, field: str) -> bool:
        return getattr(self, field, None) is not None

    def require(self, field: str) -> pd.DataFrame:
        """Return a panel or raise :class:`MissingDataError` if it is not loaded."""
        if field not in PANEL_FIELDS:
            raise ValueError(f"Unknown panel {field!r}; expected one of {PANEL_FIELDS}")
        panel = getattr(self, field)
        if panel is None:
            raise MissingDataError(f"MarketData has no {field!r} panel")
        assert isinstance(panel, pd.DataFrame)
        return panel

    # ------------------------------------------------------------------ derived panels
    def returns(self, periods: int = 1) -> pd.DataFrame:
        """Simple returns ending at each bar: ``close[t] / close[t - periods] - 1``."""
        return self.close / self.close.shift(periods) - 1.0

    def forward_returns(self, horizon: int = 1) -> pd.DataFrame:
        """Returns realised *after* each bar: ``close[t + horizon] / close[t] - 1``.

        Only for evaluation: using these inside a factor is look-ahead.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self.close.shift(-horizon) / self.close - 1.0

    def funding_per_bar(self) -> pd.DataFrame:
        """Funding actually settled in each bar (0 where there was no settlement)."""
        if self.funding_rate is None:
            return pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)
        return self.funding_rate.fillna(0.0)

    # ------------------------------------------------------------------ subsetting
    def _map(self, fn: Any) -> MarketData:
        updates = {
            name: fn(getattr(self, name))
            for name in PANEL_FIELDS
            if getattr(self, name) is not None
        }
        return replace(self, **updates)

    def slice(self, start: Any = None, end: Any = None) -> MarketData:
        """Time slice (inclusive on both ends, like ``DataFrame.loc``)."""
        return self._map(lambda p: p.loc[start:end])

    def select(self, symbols: list[str]) -> MarketData:
        """Restrict the universe to ``symbols``."""
        missing = sorted(set(symbols) - set(self.symbols))
        if missing:
            raise KeyError(f"Unknown symbols: {missing}")
        return self._map(lambda p: p[symbols])

    def __len__(self) -> int:
        return len(self.close)

    def __repr__(self) -> str:
        loaded = [n for n in PANEL_FIELDS if self.has(n)]
        span = f"{self.index[0]} -> {self.index[-1]}" if len(self) else "empty"
        return (
            f"MarketData({len(self.symbols)} symbols, {len(self)} x {self.timeframe} bars, "
            f"{span}, panels={loaded})"
        )
