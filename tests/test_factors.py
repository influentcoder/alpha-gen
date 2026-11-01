import numpy as np
import pytest

from alpha_gen.evaluation import ic_summary, information_coefficient
from alpha_gen.factors import FundingCarry, Momentum, available_factors, get_factor


@pytest.mark.parametrize("name", sorted(available_factors()))
def test_no_lookahead(name, market):
    """Scores up to t must not change when future data is removed."""
    factor = get_factor(name)
    full = factor(market)
    cutoff = market.index[len(market) // 2]
    truncated = factor(market.slice(end=cutoff))
    np.testing.assert_allclose(
        full.loc[:cutoff].to_numpy(), truncated.to_numpy(), equal_nan=True, rtol=1e-9
    )


@pytest.mark.parametrize("name", sorted(available_factors()))
def test_shape_and_finite(name, market):
    scores = get_factor(name)(market)
    assert scores.shape == market.close.shape
    assert list(scores.columns) == market.symbols
    assert np.isfinite(scores.iloc[-1]).all()


def test_funding_carry_finds_planted_alpha(market):
    ic = information_coefficient(FundingCarry(lookback=8)(market), market.forward_returns(1))
    summary = ic_summary(ic)
    assert summary.mean > 0
    assert summary.t_stat > 3


def test_momentum_values(tiny):
    scores = Momentum(lookback=2)(tiny)
    assert scores["A"].iloc[2] == pytest.approx(0.21)
    assert scores["A"].iloc[:2].isna().all()


def test_param_validation():
    with pytest.raises(ValueError):
        Momentum(lookback=5, skip=5)
    with pytest.raises(ValueError):
        FundingCarry(lookback=0)
    with pytest.raises(KeyError):
        get_factor("nope")


def test_label():
    assert Momentum(lookback=10, skip=1).label == "momentum(lookback=10,skip=1)"
