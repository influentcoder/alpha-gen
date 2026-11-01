import pandas as pd

from alpha_gen.data.ccxt_source import CCXTSource, paginate, spot_symbol

H = 3_600_000
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)


class FakeExchange:
    """Serves 10 hourly bars, at most 3 per page, like a rate-limited exchange API."""

    def __init__(self, page_size: int = 3) -> None:
        self.page_size = page_size
        self.bars = [[T0 + i * H, 1, 1, 1, 100 + i, 10] for i in range(10)]

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [b for b in self.bars if b[0] >= since][: self.page_size]

    def fetch_funding_rate_history(self, symbol, since=None, limit=None):
        rows = [{"timestamp": T0 + i * 8 * H + 5, "fundingRate": 0.0001} for i in range(2)]
        return [r for r in rows if r["timestamp"] >= since]

    def fetch_open_interest_history(self, symbol, timeframe, since=None, limit=None):
        raise RuntimeError("not supported")


def test_paginate_stops_when_exchange_ignores_since():
    calls = []

    def page(cursor):
        calls.append(cursor)
        return [{"t": 1}, {"t": 2}]

    rows = paginate(page, 0, 100, ts_of=lambda r: r["t"])
    assert [r["t"] for r in rows] == [1, 2] and len(calls) == 2


def test_fetch_builds_market_data():
    source = CCXTSource(exchange=FakeExchange(), drop_incomplete=False)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    data = source.fetch(["BTC/USDT:USDT"], start, start + pd.Timedelta(hours=8))
    assert len(data) == 8
    assert data.close["BTC/USDT:USDT"].iloc[-1] == 107
    # funding at 00:00:00.005 is floored onto the 00:00 bar; the 08:00 one is out of range
    assert data.funding_rate["BTC/USDT:USDT"].notna().sum() == 1
    assert data.funding_rate["BTC/USDT:USDT"].iloc[0] == 0.0001
    # unsupported OI endpoint degrades to NaN instead of failing
    assert data.open_interest["BTC/USDT:USDT"].isna().all()


def test_spot_symbol():
    assert spot_symbol("BTC/USDT:USDT") == "BTC/USDT"
