import pandas as pd
import pytest

from alpha_gen.data import MarketData, MissingDataError, load_market_data, save_market_data


def test_aligns_optional_panels_and_localizes_utc():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h")  # naive
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    oi = pd.DataFrame({"A": [5.0], "B": [6.0]}, index=idx[1:2])
    data = MarketData(close=close, open_interest=oi)
    assert str(data.index.tz) == "UTC"
    assert data.open_interest.shape == (3, 1)
    assert data.open_interest["A"].isna().tolist() == [True, False, True]


def test_returns_and_forward_returns(tiny):
    assert tiny.returns()["A"].iloc[1] == pytest.approx(0.10)
    assert tiny.forward_returns(1)["A"].iloc[0] == pytest.approx(0.10)
    assert tiny.forward_returns(2)["B"].iloc[0] == pytest.approx(-0.19)
    with pytest.raises(ValueError):
        tiny.forward_returns(0)


def test_require_missing(tiny):
    with pytest.raises(MissingDataError):
        tiny.require("open_interest")


def test_funding_per_bar(tiny):
    assert tiny.funding_per_bar()["A"].tolist() == [0.01, 0, 0, 0]


def test_slice_and_select(market):
    sub = market.slice(market.index[10], market.index[19]).select(market.symbols[:3])
    assert len(sub) == 10 and sub.symbols == market.symbols[:3]
    assert sub.funding_rate.shape == (10, 3)


def test_parquet_roundtrip(market, tmp_path):
    save_market_data(market, tmp_path)
    loaded = load_market_data(tmp_path)
    pd.testing.assert_frame_equal(loaded.close, market.close, check_freq=False)
    pd.testing.assert_frame_equal(loaded.funding_rate, market.funding_rate, check_freq=False)
    assert loaded.timeframe == market.timeframe
