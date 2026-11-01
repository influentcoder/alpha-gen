import pandas as pd
import pytest

from alpha_gen.data import MarketData, make_synthetic_market


@pytest.fixture(scope="session")
def market() -> MarketData:
    return make_synthetic_market(n_symbols=12, n_periods=24 * 60, seed=7, funding_alpha=1.0)


@pytest.fixture
def tiny() -> MarketData:
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    close = pd.DataFrame({"A": [100, 110, 121, 121], "B": [100, 90, 81, 81]}, index=idx)
    funding = pd.DataFrame({"A": [0.01, None, None, None], "B": [None] * 4}, index=idx)
    return MarketData(close=close, funding_rate=funding, timeframe="1h")
