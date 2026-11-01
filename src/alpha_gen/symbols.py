"""Symbol-name helpers shared by data sources and universes."""

from __future__ import annotations


def to_binance_id(symbol: str) -> str:
    """``"BTC/USDT:USDT"`` (ccxt unified) or ``"BTCUSDT"`` -> ``"BTCUSDT"``."""
    return symbol.split(":", 1)[0].replace("/", "").upper()
