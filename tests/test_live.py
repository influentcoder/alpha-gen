import json

import pandas as pd
import pytest

from alpha_gen import Strategy
from alpha_gen.data import MarketData
from alpha_gen.factors import FundingCarry
from alpha_gen.live import GENESIS_HASH, LogIntegrityError, SignalLog, SignalRunner


def test_log_chain_and_tamper_detection(tmp_path):
    log = SignalLog(tmp_path / "s.jsonl")
    first = log.append("2024-01-01T00:00", "s", {"A": 0.5, "B": -0.5})
    second = log.append("2024-01-01T01:00", "s", {"A": -0.5, "B": 0.5})
    assert first.prev_hash == GENESIS_HASH and second.prev_hash == first.hash
    assert log.verify() == 2

    lines = log.path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["weights"]["A"] = 1.0
    log.path.write_text(json.dumps(tampered) + "\n" + lines[1] + "\n")
    with pytest.raises(LogIntegrityError, match="Record 0"):
        log.verify()


def test_log_rejects_non_increasing_timestamps(tmp_path):
    log = SignalLog(tmp_path / "s.jsonl")
    log.append("2024-01-01T01:00", "s", {"A": 1.0})
    with pytest.raises(ValueError):
        log.append("2024-01-01T01:00", "s", {"A": 1.0})
    log.append("2024-01-01T01:00", "other", {"A": 1.0})  # separate strategy is fine


class FakeSource:
    cache_key = "fake"

    def __init__(self, data: MarketData) -> None:
        self.data = data
        self.calls: list[tuple] = []

    def fetch(self, symbols, start, end, timeframe="1h"):
        self.calls.append((tuple(symbols), start, end, timeframe))
        return self.data.select(list(symbols)).slice(start, end)


def test_runner_logs_latest_bar(market, tmp_path):
    source = FakeSource(market)
    strategy = Strategy(name="carry", factors={"carry": FundingCarry(lookback=8)})
    log = SignalLog(tmp_path / "s.jsonl")
    runner = SignalRunner(strategy, source, market.symbols[:5], history_bars=200, log=log)
    now = market.index[500] + pd.Timedelta(hours=1)  # bar 500 has just closed
    result = runner.run_once(now)
    record = result.record
    assert result.status == "logged"
    assert pd.Timestamp(record.timestamp) == market.index[500]
    assert set(record.weights) <= set(market.symbols[:5])  # zero weights are not logged
    assert len(record.weights) >= 4
    assert sum(abs(w) for w in record.weights.values()) == pytest.approx(1.0)
    assert source.calls[0][1] == now - pd.Timedelta(hours=200)
    assert log.verify() == 1
