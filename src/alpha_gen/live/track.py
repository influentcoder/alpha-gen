"""Paper-trading P&L from a signal log, and the track record built from it.

Each new log record carries the P&L realised by the *previous* records' weights since the last
record (``metadata["realized"]``). The track record is therefore entirely contained in the
hash-chained log: anyone can re-verify both the signals and the claimed returns.

:func:`realized_returns` is written independently of
:func:`alpha_gen.evaluation.backtest.run_backtest`; the test-suite checks that they agree, which
is the "live equals backtest" guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.evaluation.backtest import CostModel
from alpha_gen.evaluation.metrics import max_drawdown
from alpha_gen.live.signal_log import SignalRecord
from alpha_gen.timeframes import periods_per_year, timeframe_to_offset


def _held_weights(records: Sequence[SignalRecord], data: MarketData) -> pd.DataFrame:
    """Weights held *during* each bar: the latest record decided at or before the prior bar."""
    bar = timeframe_to_offset(data.timeframe)
    stamps = [pd.Timestamp(r.timestamp) for r in records]
    columns = sorted({s for r in records for s in r.weights} | set(data.symbols))
    decided = pd.DataFrame(
        [{**dict.fromkeys(columns, 0.0), **r.weights} for r in records],
        index=pd.DatetimeIndex(stamps),
        columns=columns,
    )
    decided = decided[~decided.index.duplicated(keep="last")].sort_index()
    # A position decided at the close of bar t is held during bar t + 1.
    effective_from = decided.index + bar
    held = decided.set_axis(effective_from, axis=0)
    union = held.index.union(data.index)
    return held.reindex(union).ffill().fillna(0.0).reindex(data.index)


def realized_returns(
    records: Sequence[SignalRecord],
    data: MarketData,
    costs: CostModel | None = None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-bar paper P&L of logged weights: gross, costs, funding, net.

    Bars are labelled like :class:`MarketData` (open time). ``start``/``end`` select bars
    ``start < bar <= end``. The first selected bar's trading cost is measured against the weights
    held in the bar before it, so ``data`` must include that bar.
    """
    costs = costs or CostModel()
    held = _held_weights(records, data)
    prices = data.close.reindex(columns=held.columns)
    returns = (prices / prices.shift(1) - 1.0).fillna(0.0)
    funding = data.funding_per_bar().reindex(columns=held.columns).fillna(0.0)

    traded = held.diff().abs().sum(axis=1)
    traded.iloc[0] = np.nan  # unknown without the bar before the window
    out = pd.DataFrame(
        {
            "gross": (held * returns).sum(axis=1),
            "costs": traded * costs.total_bps / 1e4,
            "funding": -(held * funding).sum(axis=1) if costs.include_funding else 0.0,
        }
    )
    out["net"] = out["gross"] - out["costs"] + out["funding"]
    if start is not None:
        out = out.loc[out.index > pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    if out["costs"].isna().any():
        raise ValueError("data must start at least one bar before the first selected bar")
    return out


def realized_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    """Totals of a :func:`realized_returns` frame, as stored in log metadata."""
    totals = frame[["gross", "costs", "funding", "net"]].sum()
    return {**{k: float(v) for k, v in totals.items()}, "bars": len(frame)}


def track_record(records: Sequence[SignalRecord], strategy: str) -> pd.DataFrame:
    """Realised P&L per record for one strategy, from the log metadata."""
    rows = []
    for r in records:
        if r.strategy != strategy:
            continue
        realized = r.metadata.get("realized")
        if realized:
            rows.append({"timestamp": pd.Timestamp(r.timestamp), **realized})
    frame = pd.DataFrame(rows, columns=["timestamp", "gross", "costs", "funding", "net", "bars"])
    frame = frame.set_index("timestamp")
    frame["equity"] = (1.0 + frame["net"]).cumprod()
    return frame


MIN_PERIODS_FOR_RATIOS = 30


def track_summary(record: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Headline statistics of a track record (returns per record ~ per rebalance).

    Annualised return, volatility and Sharpe are NaN until there are
    ``MIN_PERIODS_FOR_RATIOS`` periods: a Sharpe from a handful of days is noise.
    """
    if record.empty:
        return {"periods": 0}
    net = record["net"]
    bars = float(record["bars"].mean()) or 1.0
    ppy = periods_per_year(timeframe) / bars
    enough = len(net) >= MIN_PERIODS_FOR_RATIOS
    std = float(net.std()) if enough else float("nan")
    mean = float(net.mean()) if enough else float("nan")
    return {
        "periods": len(net),
        "start": str(record.index[0]),
        "end": str(record.index[-1]),
        "total_return": float(record["equity"].iloc[-1] - 1.0),
        "ann_return": mean * ppy,
        "ann_vol": float(std * np.sqrt(ppy)) if np.isfinite(std) else float("nan"),
        "sharpe": mean / std * float(np.sqrt(ppy)) if std and np.isfinite(std) else float("nan"),
        "max_drawdown": max_drawdown(net),
        "total_costs": float(record["costs"].sum()),
        "total_funding": float(record["funding"].sum()),
    }
