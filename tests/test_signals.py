import numpy as np
import pandas as pd
import pytest

from alpha_gen.factors import FundingCarry, Momentum
from alpha_gen.signals import (
    Strategy,
    combine,
    cross_sectional_rank,
    cross_sectional_zscore,
    ic_weighted_combine,
    rebalance,
    smooth,
    to_weights,
)


@pytest.fixture
def panel():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"A": [1.0, 2.0, np.nan], "B": [2.0, 2.0, 1.0], "C": [3.0, 5.0, np.nan]}, index=idx
    )


def test_zscore(panel):
    z = cross_sectional_zscore(panel)
    assert z.iloc[0].tolist() == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)
    assert z.iloc[2].isna().all()  # single observation


def test_rank(panel):
    r = cross_sectional_rank(panel)
    assert r.iloc[0].tolist() == [-0.5, 0.0, 0.5]
    assert r.iloc[1].tolist() == [-0.25, -0.25, 0.5]


def test_combine_skips_missing_factor(panel):
    other = panel * np.nan
    other.iloc[0] = [3.0, 2.0, 1.0]
    out = combine({"a": panel, "b": other})
    assert out.iloc[0].abs().max() == pytest.approx(0.0)  # opposite factors cancel
    pd.testing.assert_series_equal(
        out.iloc[1], cross_sectional_zscore(panel).iloc[1], check_names=False
    )


def test_to_weights_neutral_and_gross(panel):
    w = to_weights(panel, gross=2.0)
    assert w.iloc[:2].sum(axis=1).abs().max() < 1e-12
    assert w.iloc[:2].abs().sum(axis=1).tolist() == pytest.approx([2.0, 2.0])
    capped = to_weights(panel, max_weight=0.3)
    assert capped.abs().max().max() <= 0.3


def test_ic_weighted_combine_has_no_lookahead(market):
    factors = {"carry": FundingCarry(lookback=8)(market), "momo": Momentum(lookback=24)(market)}
    horizon = 4
    fwd = market.forward_returns(horizon)
    _, weights = ic_weighted_combine(factors, fwd, horizon, window=100)
    cutoff = market.index[700]
    _, weights_trunc = ic_weighted_combine(
        {k: v.loc[:cutoff] for k, v in factors.items()},
        market.slice(end=cutoff).forward_returns(horizon),
        horizon,
        window=100,
    )
    np.testing.assert_allclose(
        weights.loc[:cutoff].to_numpy(), weights_trunc.to_numpy(), equal_nan=True
    )
    assert weights["carry"].iloc[-1] > 0


def test_strategy(market):
    strategy = Strategy(
        name="s", factors={"carry": FundingCarry(), "momo": Momentum()}, weights={"carry": 1.0}
    )
    weights = strategy.target_weights(market)
    assert weights.shape == market.close.shape
    assert weights.iloc[-1].abs().sum() == pytest.approx(1.0)
    assert strategy.describe()["factors"]["carry"] == "funding_carry(lookback=72)"


def test_rebalance_holds_between_schedule_bars():
    idx = pd.date_range("2024-01-01 06:00", periods=12, freq="1h", tz="UTC")
    w = pd.DataFrame({"A": np.arange(12.0)}, index=idx)
    held = rebalance(w, "8h")
    assert held["A"].iloc[:2].tolist() == [0.0, 0.0]  # before the first 08:00 rebalance
    assert held["A"].iloc[2:].tolist() == [2.0] * 8 + [10.0, 10.0]  # 08:00 and 16:00


def test_rebalance_and_smooth_do_not_look_ahead(market):
    w = to_weights(Momentum(lookback=24)(market))
    cutoff = market.index[300]
    for fn in (lambda x: rebalance(x, "1D"), lambda x: smooth(x, 12)):
        pd.testing.assert_frame_equal(fn(w).loc[:cutoff], fn(w.loc[:cutoff]))
