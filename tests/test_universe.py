import numpy as np
import pandas as pd
import pytest

from alpha_gen.data import apply_universe, candidates, expand_membership, top_n_membership


@pytest.fixture
def monthly_volume():
    idx = pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")
    return pd.DataFrame(
        {
            "BTCUSDT": [10.0, 10.0, 10.0],
            "ETHUSDT": [5.0, 1.0, 1.0],
            "SOLUSDT": [1.0, 5.0, np.nan],  # delisted in March
            "USDCUSDT": [99.0, 99.0, 99.0],  # stablecoin: excluded
        },
        index=idx,
    )


def test_membership_uses_previous_period(monthly_volume):
    member = top_n_membership(monthly_volume, top_n=2)
    assert not member.iloc[0].any()  # nothing known before the first period
    assert member.iloc[1][["BTCUSDT", "ETHUSDT"]].all() and not member.iloc[1]["SOLUSDT"]
    assert member.iloc[2][["BTCUSDT", "SOLUSDT"]].all() and not member.iloc[2]["ETHUSDT"]
    assert not member["USDCUSDT"].any()
    assert candidates(member) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_membership_does_not_look_ahead(monthly_volume):
    full = top_n_membership(monthly_volume, top_n=1)
    changed = monthly_volume.copy()
    changed.iloc[-1] = [0.0, 100.0, 100.0, 0.0]  # last period's volume must not matter
    pd.testing.assert_frame_equal(full, top_n_membership(changed, top_n=1))


def test_expand_and_apply(monthly_volume):
    member = top_n_membership(monthly_volume, top_n=2)
    bars = pd.date_range("2024-01-31 22:00", periods=4, freq="1h", tz="UTC")
    expanded = expand_membership(member, bars)
    assert not expanded.iloc[1].any()  # Jan 31 23:00 -> January membership (empty)
    assert expanded.iloc[2]["BTCUSDT"]  # Feb 1 00:00 -> February membership
    scores = pd.DataFrame(1.0, index=bars, columns=["BTCUSDT", "SOLUSDT", "XRPUSDT"])
    masked = apply_universe(scores, member)
    assert masked.iloc[3].tolist()[:2] == [1.0, 1.0] or np.isnan(masked.iloc[3]["SOLUSDT"])
    assert np.isnan(masked.iloc[3]["XRPUSDT"])  # not in the volume table at all
    assert masked.iloc[0].isna().all()


def test_validation(monthly_volume):
    with pytest.raises(ValueError):
        top_n_membership(monthly_volume, top_n=0)
