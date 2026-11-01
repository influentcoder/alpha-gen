"""Opt-in checks against the live exchange used for paper trading (``pytest -m network``)."""

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.network

CONFIG = Path(__file__).resolve().parents[1] / "strategies" / "momentum_v0.toml"


def test_okx_universe_is_crypto_only():
    from alpha_gen.data.ccxt_source import CCXTSource

    symbols = CCXTSource("okx").list_symbols()
    assert "BTC/USDT:USDT" in symbols and len(symbols) > 100
    assert not {"TSLA/USDT:USDT", "AAPL/USDT:USDT", "XAU/USDT:USDT"} & set(symbols)


def test_okx_daily_bars_and_funding():
    from alpha_gen.data.ccxt_source import CCXTSource

    end = pd.Timestamp.now(tz="UTC").floor("D")
    source = CCXTSource("okx", include_open_interest=False)
    data = source.fetch(["BTC/USDT:USDT"], end - pd.Timedelta(days=10), end, "1d")
    assert len(data) >= 9 and data.index[-1] >= end - pd.Timedelta(days=2)
    assert data.volume["BTC/USDT:USDT"].iloc[-1] > 1e8  # quote volume in USDT
    # Daily funding = sum of that day's settlements (OKX settles every 8h).
    day = data.index[-2]
    raw = source.exchange.fetch_funding_rate_history(
        "BTC/USDT:USDT", since=int(day.value // 1_000_000), limit=10
    )
    same_day = [r["fundingRate"] for r in raw if pd.Timestamp(r["datetime"]).floor("D") == day]
    assert len(same_day) >= 3
    assert data.funding_rate["BTC/USDT:USDT"].loc[day] == pytest.approx(sum(same_day))


def test_paper_strategy_runs_end_to_end(tmp_path):
    from alpha_gen.cli import main

    log = tmp_path / "signals.jsonl"
    yesterday = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
    for now in (yesterday + pd.Timedelta(minutes=20), yesterday + pd.Timedelta(days=1, minutes=20)):
        main(["run", str(CONFIG), "--log", str(log), "--now", now.isoformat()])
    main(["track", "--log", str(log), "--out", str(tmp_path / "site")])
    assert (tmp_path / "site" / "index.html").exists()
