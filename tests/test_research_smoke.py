"""End-to-end smoke test of the research protocol on synthetic data with a planted edge."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from alpha_gen.data import make_synthetic_market

RESEARCH = Path(__file__).resolve().parents[1] / "research"


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(RESEARCH))
    sys.modules.pop("real_data_v1", None)
    import real_data_v1

    data = make_synthetic_market(
        n_symbols=12, n_periods=24 * 480, start="2022-01-01", seed=5, funding_alpha=1.5
    )
    data = type(data)(
        close=data.close.rename(columns={data.symbols[0]: "BTCUSDT"}),
        volume=data.volume,
        funding_rate=data.funding_rate.rename(columns={data.symbols[0]: "BTCUSDT"}),
        spot_close=data.spot_close.rename(columns={data.symbols[0]: "BTCUSDT"}),
        timeframe="1h",
    )
    months = pd.date_range(data.index[0], data.index[-1], freq="MS")
    membership = pd.DataFrame(True, index=months, columns=data.symbols)
    monkeypatch.setattr(real_data_v1, "load", lambda cache_dir, with_oi: (membership, data, None))
    monkeypatch.setattr(real_data_v1, "END", data.index[-1])
    return real_data_v1


def test_protocol_finds_planted_edge_and_writes_report(protocol, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["real_data_v1", "--out", str(tmp_path)])
    protocol.main()
    report = (tmp_path / "real_data_v1.md").read_text()
    assert "## Verdict" in report and "## Where the PnL comes from" in report
    assert "price 100%, funding 100%" in report
    import json

    results = json.loads((tmp_path / "real_data_v1.json").read_text())
    carry = results["oos"]["funding_carry"]["performance"]
    assert carry["sharpe"] > 2  # the planted funding edge survives costs out of sample
    assert abs(results["variants"][0]["risk"]["beta_ew"]) < 0.3
