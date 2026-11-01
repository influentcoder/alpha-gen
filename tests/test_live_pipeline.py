"""Live pipeline: config -> runner -> hash-chained log with realised P&L -> track record.

The key guarantee tested here: running the strategy day by day ("live") produces exactly the
signals and P&L of a one-shot backtest over the same data.
"""

import textwrap

import numpy as np
import pandas as pd
import pytest

from alpha_gen import Strategy
from alpha_gen.data import MarketData, listing_age_mask, make_synthetic_market
from alpha_gen.evaluation import CostModel, run_backtest
from alpha_gen.factors import Momentum
from alpha_gen.live import (
    SignalLog,
    SignalRunner,
    StaleDataError,
    load_config,
    realized_returns,
    track_record,
    track_summary,
)
from alpha_gen.live.signal_log import SignalRecord
from alpha_gen.signals import rebalance, time_series_trend, to_weights

DAY = pd.Timedelta(days=1)
COSTS = CostModel(fee_bps=5, slippage_bps=3)


class FakeSource:
    """Serves slices of a fixed market, like an exchange that only knows the past."""

    cache_key = "fake"

    def __init__(self, data: MarketData) -> None:
        self.data = data
        self.calls = 0

    def list_symbols(self) -> list[str]:
        return self.data.symbols

    def fetch(self, symbols, start, end, timeframe="1d"):
        self.calls += 1
        sub = self.data.select(list(symbols))
        return sub.slice(start, pd.Timestamp(end) - pd.Timedelta(1))


@pytest.fixture(scope="module")
def daily_market() -> MarketData:
    return make_synthetic_market(
        n_symbols=12, n_periods=240, timeframe="1d", funding_interval=1, seed=11, momentum_alpha=1.0
    )


def momentum_strategy(**kw) -> Strategy:
    return Strategy(
        name="momo_test",
        factors={"momo": Momentum(lookback=14)},
        universe_top_n=8,
        rebalance_every="1D",
        **kw,
    )


def run_daily(strategy, data, log, days, history_bars=100):
    runner = SignalRunner(
        strategy, FakeSource(data), timeframe="1d", history_bars=history_bars, log=log, costs=COSTS
    )
    return [runner.run_once(data.index[i] + DAY) for i in days]


# ------------------------------------------------------------------ live == backtest
def test_live_run_reconciles_with_backtest(daily_market, tmp_path):
    strategy = momentum_strategy()
    log = SignalLog(tmp_path / "signals.jsonl")
    days = range(120, 200)
    results = run_daily(strategy, daily_market, log, days)
    assert all(r.status == "logged" for r in results)
    assert log.verify() == len(days)

    # 1. Signals equal the full-history backtest weights, bar for bar.
    full = strategy.target_weights(daily_market)
    for record in log.records():
        expected = full.loc[pd.Timestamp(record.timestamp)]
        expected = expected[expected != 0].to_dict()
        assert record.weights == pytest.approx(expected, abs=1e-8)

    # 2. Realised P&L in the log equals the backtest P&L (from the 3rd record on, once the
    #    paper book's starting position matches the backtest's).
    bt = run_backtest(full, daily_market, COSTS)
    record = track_record(log.records(), strategy.name)
    compare = record.index[2:]
    assert len(compare) > 70 and record.loc[compare, "net"].abs().min() > 0  # non-trivial
    assert record.loc[compare, "costs"].sum() > 0 and record.loc[compare, "funding"].abs().sum() > 0
    np.testing.assert_allclose(record.loc[compare, "net"], bt.net_returns.loc[compare], atol=1e-7)
    np.testing.assert_allclose(record.loc[compare, "funding"], bt.funding.loc[compare], atol=1e-9)
    assert (record["bars"] == 1).all()


def test_realized_returns_matches_backtest_on_hourly_data(market):
    """Two independent P&L implementations must agree (weights held between rebalances)."""
    weights = rebalance(to_weights(Momentum(lookback=24)(market)), "8h")
    stamps = [t for t in weights.index if t == t.floor("8h")]
    records = [
        SignalRecord(
            timestamp=str(t),
            strategy="s",
            weights={k: float(v) for k, v in weights.loc[t].items() if v},
            published_at=str(t),
            prev_hash="",
        )
        for t in stamps
    ]
    bt = run_backtest(weights, market, COSTS)
    paper = realized_returns(records, market, COSTS, start=stamps[1])
    np.testing.assert_allclose(paper["net"], bt.net_returns.loc[paper.index], atol=1e-10)
    np.testing.assert_allclose(paper["costs"], bt.trading_costs.loc[paper.index], atol=1e-12)


# ------------------------------------------------------------------ runner behaviour
def test_runner_is_idempotent(daily_market, tmp_path):
    log = SignalLog(tmp_path / "s.jsonl")
    first, second = run_daily(momentum_strategy(), daily_market, log, [150, 150])
    assert (first.status, second.status) == ("logged", "already_logged")
    assert len(log.records()) == 1


def test_missed_runs_accumulate_pnl_over_several_bars(daily_market, tmp_path):
    log = SignalLog(tmp_path / "s.jsonl")
    run_daily(momentum_strategy(), daily_market, log, [150, 151, 154])
    realized = log.records()[-1].metadata["realized"]
    assert realized["bars"] == 3  # the weights from day 151 were held for 3 days
    held = log.records()[1].weights
    rets = daily_market.returns().loc[daily_market.index[152] : daily_market.index[154]]
    gross = sum((rets[s] * w).sum() for s, w in held.items())
    assert realized["gross"] == pytest.approx(gross)


def test_runner_skips_non_rebalance_bars(market, tmp_path):
    strategy = Strategy(name="h", factors={"m": Momentum(lookback=24)}, rebalance_every="1D")
    log = SignalLog(tmp_path / "s.jsonl")
    runner = SignalRunner(strategy, FakeSource(market), market.symbols, timeframe="1h", log=log)
    at_five = market.index[600].floor("D") + pd.Timedelta(hours=6)  # 05:00 bar just closed
    assert runner.run_once(at_five).status == "not_rebalance_bar"
    midnight = market.index[600].floor("D") + pd.Timedelta(hours=1)  # 00:00 bar just closed
    assert runner.run_once(midnight).status == "logged"
    assert len(log.records()) == 1


def test_runner_refuses_stale_data(daily_market, tmp_path):
    runner = SignalRunner(
        momentum_strategy(), FakeSource(daily_market), timeframe="1d", history_bars=100
    )
    with pytest.raises(StaleDataError):
        runner.run_once(daily_market.index[-1] + 5 * DAY)


def test_runner_needs_symbols_or_lister(daily_market):
    class NoLister:
        cache_key = "x"

        def fetch(self, *a, **k):
            raise AssertionError("not reached")

    runner = SignalRunner(momentum_strategy(), NoLister(), timeframe="1d")
    with pytest.raises(ValueError, match="list_symbols"):
        runner.run_once(daily_market.index[-1])


# ------------------------------------------------------------------ strategy features
def test_strategy_universe_uses_previous_month_volume(daily_market):
    strategy = momentum_strategy()
    mask = strategy.tradable(daily_market)
    monthly = daily_market.volume.resample("MS").sum()
    march = pd.Timestamp("2024-03-15", tz="UTC")
    expected = set(monthly.loc["2024-02-01"].nlargest(8).index)
    assert set(mask.columns[mask.loc[march]]) == expected
    assert not mask.loc[:"2024-01-31"].any().any()  # no prior month yet


def test_listing_age_mask_and_strategy_filter(daily_market):
    close = daily_market.close.copy()
    close.iloc[:100, 0] = np.nan  # symbol 0 lists on day 100
    age = listing_age_mask(close, "30D")
    assert not age.iloc[129, 0] and age.iloc[130, 0]
    assert age.iloc[30:, 1].all() and not age.iloc[29, 1]
    data = MarketData(close=close, volume=daily_market.volume, timeframe="1d")
    strategy = momentum_strategy(min_listing_age_days=30)
    weights = strategy.target_weights(data)
    assert (weights.iloc[:130, 0] == 0).all()


def test_vol_scaling_tilts_weights_to_low_vol(daily_market):
    plain = momentum_strategy().target_weights(daily_market)
    scaled = momentum_strategy(vol_lookback=30).target_weights(daily_market)
    vol = daily_market.returns().std()
    calm, wild = vol.idxmin(), vol.idxmax()
    ratio = scaled.abs().mean() / plain.abs().mean().replace(0, np.nan)
    assert ratio[calm] > ratio[wild]


def test_time_series_trend(daily_market):
    w = time_series_trend(daily_market.close, 30, weight=0.5, symbols=daily_market.symbols[:2])
    assert set(np.unique(w.iloc[31:, :2])) <= {-0.5, 0.5}
    assert (w.iloc[:, 2:] == 0).all().all() and (w.iloc[:30] == 0).all().all()


# ------------------------------------------------------------------ config + track record
CONFIG = textwrap.dedent(
    """
    name = "momentum_v0"
    description = "test"
    timeframe = "1d"
    history_bars = 100

    [source]
    exchange = "okx"

    [universe]
    top_n = 8

    [[factors]]
    label = "momo"
    name = "momentum"
    params = { lookback = 14 }

    [portfolio]
    rebalance_every = "1D"

    [costs]
    fee_bps = 5.0
    slippage_bps = 3.0
    """
)


def test_load_config(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text(CONFIG)
    cfg = load_config(path)
    assert cfg.strategy.name == "momentum_v0" and cfg.exchange == "okx"
    assert cfg.strategy.factors["momo"].label == "momentum(lookback=14,skip=0)"
    assert cfg.strategy.universe_top_n == 8 and cfg.strategy.rebalance_every == "1D"
    assert cfg.costs.total_bps == 8.0 and len(cfg.sha256) == 64
    assert cfg.strategy.describe()["config_sha256"] == cfg.sha256


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("history_bars = 100", "history_bars = 40"), "62 days"),
        (("lookback = 14", "lookback = 150"), "needs more than"),
        (('name = "momentum"', 'name = "nope"'), "Unknown factor"),
    ],
)
def test_config_validation(tmp_path, edit, message):
    path = tmp_path / "s.toml"
    path.write_text(CONFIG.replace(*edit))
    with pytest.raises((ValueError, KeyError), match=message):
        load_config(path)


def test_track_record_and_summary(daily_market, tmp_path):
    log = SignalLog(tmp_path / "s.jsonl")
    run_daily(momentum_strategy(), daily_market, log, range(150, 180))
    record = track_record(log.records(), "momo_test")
    assert len(record) == 29  # the first record has nothing realised yet
    assert record["equity"].iloc[-1] == pytest.approx((1 + record["net"]).prod())
    summary = track_summary(record, "1d")
    assert summary["periods"] == 29 and np.isfinite(summary["sharpe"])
    assert track_record(log.records(), "other").empty
