"""Exchange data via `ccxt <https://github.com/ccxt/ccxt>`_ (install with ``alpha-gen[exchange]``).

Symbols use ccxt's unified perp notation, e.g. ``"BTC/USDT:USDT"``. The matching spot symbol
(``"BTC/USDT"``) is derived automatically when ``include_spot=True``.

Notes:
    * Some exchanges only serve limited open-interest history (Binance: ~30 days). Missing data
      is left as NaN rather than raising, so factors degrade gracefully.
    * The still-forming last candle is dropped by default so live signals never use partial bars.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.timeframes import timeframe_seconds, timeframe_to_freq

log = logging.getLogger(__name__)

Row = Any


def _to_ms(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def spot_symbol(perp_symbol: str) -> str:
    """``"BTC/USDT:USDT"`` -> ``"BTC/USDT"``."""
    return perp_symbol.split(":", 1)[0]


def paginate(
    fetch_page: Callable[[int], list[Row]],
    since_ms: int,
    until_ms: int,
    ts_of: Callable[[Row], int],
) -> list[Row]:
    """Walk a ``since``-paginated ccxt endpoint until ``until_ms`` or until it stops advancing."""
    rows: dict[int, Row] = {}
    cursor = since_ms
    while cursor < until_ms:
        page = [r for r in fetch_page(cursor) if ts_of(r) is not None and ts_of(r) >= cursor]
        if not page:
            break
        for row in page:
            ts = ts_of(row)
            if ts < until_ms:
                rows[ts] = row
        cursor = max(ts_of(r) for r in page) + 1
    return [rows[k] for k in sorted(rows)]


class CCXTSource:
    """Fetch perp OHLCV, funding, open interest and (optionally) spot closes from an exchange."""

    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        *,
        spot_exchange_id: str | None = "binance",
        include_funding: bool = True,
        include_open_interest: bool = True,
        include_spot: bool = False,
        drop_incomplete: bool = True,
        limit: int = 1000,
        exchange: Any = None,
        spot_exchange: Any = None,
        exchange_config: dict[str, Any] | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.spot_exchange_id = spot_exchange_id
        self.include_funding = include_funding
        self.include_open_interest = include_open_interest
        self.include_spot = include_spot
        self.drop_incomplete = drop_incomplete
        self.limit = limit
        self._config = {"enableRateLimit": True, **(exchange_config or {})}
        self._exchange = exchange
        self._spot_exchange = spot_exchange

    # ------------------------------------------------------------------ plumbing
    @property
    def cache_key(self) -> str:
        parts = [
            f"ccxt-{self.exchange_id}",
            f"f{int(self.include_funding)}",
            f"oi{int(self.include_open_interest)}",
            f"spot-{self.spot_exchange_id}" if self.include_spot else "nospot",
        ]
        return "_".join(parts)

    @staticmethod
    def _make(exchange_id: str, config: dict[str, Any]) -> Any:
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("CCXTSource needs ccxt: pip install 'alpha-gen[exchange]'") from exc
        return getattr(ccxt, exchange_id)(config)

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            self._exchange = self._make(self.exchange_id, self._config)
        return self._exchange

    @property
    def spot_exchange(self) -> Any:
        if self._spot_exchange is None:
            if self.spot_exchange_id is None:
                raise ValueError("include_spot=True requires spot_exchange_id or spot_exchange")
            self._spot_exchange = self._make(self.spot_exchange_id, self._config)
        return self._spot_exchange

    # ------------------------------------------------------------------ endpoints
    def _ohlcv(
        self, exchange: Any, symbol: str, timeframe: str, since: int, until: int
    ) -> pd.DataFrame:
        rows = paginate(
            lambda c: exchange.fetch_ohlcv(symbol, timeframe, since=c, limit=self.limit),
            since,
            until,
            ts_of=lambda r: r[0],
        )
        frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        frame.index = pd.to_datetime(frame.pop("ts"), unit="ms", utc=True)
        return frame

    def _funding(self, symbol: str, since: int, until: int, freq: str) -> pd.Series:
        rows = paginate(
            lambda c: self.exchange.fetch_funding_rate_history(symbol, since=c, limit=self.limit),
            since,
            until,
            ts_of=lambda r: r.get("timestamp"),
        )
        return self._series(rows, "fundingRate", freq, how="last")

    def _open_interest(
        self, symbol: str, timeframe: str, since: int, until: int, freq: str
    ) -> pd.Series:
        rows = paginate(
            lambda c: self.exchange.fetch_open_interest_history(
                symbol, timeframe, since=c, limit=min(self.limit, 500)
            ),
            since,
            until,
            ts_of=lambda r: r.get("timestamp"),
        )
        key = "openInterestValue"
        if rows and rows[0].get(key) is None:
            key = "openInterestAmount"
        return self._series(rows, key, freq, how="last")

    @staticmethod
    def _series(rows: list[dict[str, Any]], key: str, freq: str, how: str) -> pd.Series:
        if not rows:
            return _empty_series()
        idx = pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True).floor(freq)
        values = pd.Series([r.get(key) for r in rows], index=idx, dtype=float)
        return values.groupby(level=0).agg(how)

    def _spot_close(self, symbol: str, timeframe: str, since: int, until: int) -> pd.Series:
        bars = self._ohlcv(self.spot_exchange, spot_symbol(symbol), timeframe, since, until)
        return bars["close"]

    def _safe(self, what: str, symbol: str, fn: Callable[..., pd.Series], *args: Any) -> pd.Series:
        try:
            return fn(symbol, *args)
        except Exception as exc:  # exchange may not support the endpoint for this symbol
            log.warning("Could not fetch %s for %s: %s", what, symbol, exc)
            return _empty_series()

    # ------------------------------------------------------------------ public API
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str = "1h",
    ) -> MarketData:
        since, until = _to_ms(start), _to_ms(end)
        if self.drop_incomplete:
            now_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
            until = min(until, now_ms - timeframe_seconds(timeframe) * 1000 + 1)
        freq = timeframe_to_freq(timeframe)

        close: dict[str, pd.Series] = {}
        volume: dict[str, pd.Series] = {}
        funding: dict[str, pd.Series] = {}
        oi: dict[str, pd.Series] = {}
        spot: dict[str, pd.Series] = {}
        for symbol in symbols:
            bars = self._ohlcv(self.exchange, symbol, timeframe, since, until)
            close[symbol], volume[symbol] = bars["close"], bars["volume"]
            if self.include_funding:
                funding[symbol] = self._safe("funding", symbol, self._funding, since, until, freq)
            if self.include_open_interest:
                oi[symbol] = self._safe(
                    "open interest", symbol, self._open_interest, timeframe, since, until, freq
                )
            if self.include_spot:
                spot[symbol] = self._safe("spot", symbol, self._spot_close, timeframe, since, until)

        def panel(d: dict[str, pd.Series]) -> pd.DataFrame | None:
            return pd.DataFrame(d) if d else None

        close_panel = pd.DataFrame(close)
        if close_panel.empty:
            raise ValueError(f"No OHLCV returned for {list(symbols)} in [{start}, {end})")
        return MarketData(
            close=close_panel,
            volume=panel(volume),
            funding_rate=panel(funding),
            open_interest=panel(oi),
            spot_close=panel(spot),
            timeframe=timeframe,
        )
