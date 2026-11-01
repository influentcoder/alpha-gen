import numpy as np
import pandas as pd
import pytest

from alpha_gen.evaluation import (
    CostModel,
    evaluate_factor,
    information_coefficient,
    max_drawdown,
    run_backtest,
    turnover,
    walk_forward_select,
    walk_forward_splits,
)
from alpha_gen.factors import FundingCarry


def test_ic_perfect_and_inverse():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    fwd = pd.DataFrame({"A": [0.1, 0.3], "B": [0.2, 0.2], "C": [0.3, 0.1]}, index=idx)
    sig = pd.DataFrame({"A": [1, 1], "B": [2, 2], "C": [3, 3]}, index=idx, dtype=float)
    assert information_coefficient(sig, fwd).tolist() == pytest.approx([1.0, -1.0])


def test_ic_ignores_unpaired_values():
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    fwd = pd.DataFrame({"A": [0.1], "B": [0.2], "C": [0.3], "D": [np.nan]}, index=idx)
    sig = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [-100.0]}, index=idx)
    assert information_coefficient(sig, fwd, method="pearson").iloc[0] == pytest.approx(1.0)


def test_backtest_pnl_costs_and_funding(tiny):
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0], "B": 0.0}, index=tiny.index)
    result = run_backtest(weights, tiny, CostModel(fee_bps=10, slippage_bps=0), lag=1)
    # Long A from bar 0's close: earns +10% in bar 1 and +10% in bar 2, then exits.
    assert result.gross_returns.tolist() == pytest.approx([0.0, 0.10, 0.10, 0.0])
    assert result.trading_costs.tolist() == pytest.approx([0.0, 0.001, 0.0, 0.001])
    # Funding at bar 0 is charged on the position held during bar 0, which is flat.
    assert result.funding.abs().sum() == 0


def test_backtest_long_pays_funding(tiny):
    weights = pd.DataFrame({"A": [1.0] * 4, "B": 0.0}, index=tiny.index)
    shifted = tiny.funding_rate.shift(1)  # move the settlement into a bar where A is held
    data = type(tiny)(close=tiny.close, funding_rate=shifted, timeframe="1h")
    result = run_backtest(weights, data, CostModel(fee_bps=0, slippage_bps=0))
    assert result.funding.iloc[1] == pytest.approx(-0.01)


def test_backtest_rejects_lookahead(tiny):
    with pytest.raises(ValueError):
        run_backtest(tiny.close * 0, tiny, lag=0)


def test_turnover_and_drawdown():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    w = pd.DataFrame({"A": [0.5, -0.5, -0.5], "B": [-0.5, 0.5, 0.5]}, index=idx)
    assert turnover(w).tolist() == [1.0, 2.0, 0.0]
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_walk_forward_splits():
    idx = pd.RangeIndex(20)
    splits = list(walk_forward_splits(idx, train_size=10, test_size=4, gap=1))
    assert [(len(tr), te[0], te[-1]) for tr, te in splits] == [(10, 11, 14), (10, 15, 18)]
    expanding = list(walk_forward_splits(idx, 10, 4, expanding=True))
    assert len(expanding[1][0]) == 14 and expanding[1][0][0] == 0


def test_evaluate_factor_report(market):
    report = evaluate_factor(FundingCarry(lookback=8), market, decay_horizons=(1, 4))
    assert report.ic.mean > 0
    assert list(report.decay.index) == [1, 4]
    assert report.to_dict()["backtest"]["gross_sharpe"] > 0
    assert "funding_carry" in str(report)


def test_walk_forward_select_uses_only_training_data():
    idx = pd.date_range("2024-01-01", periods=40, freq="1D", tz="UTC")
    noise = np.random.default_rng(0).normal(0, 0.001, 40)
    # "early" is great for 20 days then terrible; "late" is the reverse.
    early = pd.Series(np.r_[np.full(20, 0.01), np.full(20, -0.01)] + noise, index=idx)
    late = pd.Series(np.r_[np.full(20, -0.01), np.full(20, 0.01)] + noise, index=idx)
    splits = walk_forward_splits(idx, train_size=10, test_size=10)
    result = walk_forward_select({"early": early, "late": late}, splits, periods_per_year=365)
    assert result.choices["chosen"].tolist() == ["early", "early", "late"]
    # The switch lags the regime change by one window: the test after it loses money.
    assert result.returns.loc["2024-01-21":"2024-01-30"].mean() < 0
    assert result.returns.index.is_monotonic_increasing and len(result.returns) == 30


def test_evaluate_factor_with_universe_and_rebalance(market):
    idx = pd.date_range(market.index[0].floor("D"), market.index[-1], freq="MS")
    member = pd.DataFrame(False, index=idx, columns=market.symbols)
    member[market.symbols[:5]] = True
    report = evaluate_factor(
        FundingCarry(lookback=8), market, universe=member, rebalance_every="8h"
    )
    traded = report.backtest.positions.abs().sum()
    assert traded[market.symbols[5:]].sum() == 0
    assert (traded[market.symbols[:5]] > 0).all()
    hours = report.backtest.positions.diff().abs().sum(axis=1)
    assert set(hours[hours > 0].index.hour) <= {1, 9, 17}  # held from the bar after 0/8/16h
