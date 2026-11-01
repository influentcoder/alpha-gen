"""Vectorised perp backtest with fees, slippage and funding."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.evaluation.metrics import performance_summary, turnover
from alpha_gen.timeframes import periods_per_year


@dataclass(frozen=True)
class CostModel:
    """Linear trading costs in basis points of traded notional, plus optional funding."""

    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    include_funding: bool = True

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass(frozen=True)
class BacktestResult:
    """Per-bar outputs of :func:`run_backtest`. All return series are fractions of capital."""

    positions: pd.DataFrame
    gross_returns: pd.Series
    trading_costs: pd.Series
    funding: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    timeframe: str

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_returns).cumprod().rename("equity")

    def summary(self) -> dict[str, float]:
        ppy = periods_per_year(self.timeframe)
        out = performance_summary(self.net_returns, ppy)
        out["gross_sharpe"] = performance_summary(self.gross_returns, ppy).get(
            "sharpe", float("nan")
        )
        out["ann_turnover"] = float(self.turnover.mean() * ppy)
        out["ann_cost"] = float(self.trading_costs.mean() * ppy)
        out["ann_funding"] = float(self.funding.mean() * ppy)
        return out


def run_backtest(
    weights: pd.DataFrame,
    data: MarketData,
    costs: CostModel | None = None,
    *,
    lag: int = 1,
) -> BacktestResult:
    """Backtest target weights against perp returns.

    ``weights[t]`` is the target decided at the close of bar ``t``. It is held from ``t + lag - 1``
    to ``t + lag``, i.e. ``lag=1`` trades at the decision bar's close (the minimum without
    look-ahead) and ``lag=2`` adds one bar of execution delay.

    Funding: a long position *pays* positive funding, a short receives it.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1; lag=0 would trade on information from the same bar")
    costs = costs or CostModel()
    target = weights.reindex(index=data.index, columns=data.symbols).fillna(0.0)
    held = target.shift(lag).fillna(0.0)
    returns = data.returns().fillna(0.0)

    traded = turnover(held)
    gross = (held * returns).sum(axis=1).rename("gross")
    trading_costs = (traded * costs.total_bps / 1e4).rename("costs")
    if costs.include_funding:
        funding = (-(held * data.funding_per_bar()).sum(axis=1)).rename("funding")
    else:
        funding = pd.Series(0.0, index=data.index, name="funding")
    net = (gross - trading_costs + funding).rename("net")
    return BacktestResult(
        positions=held,
        gross_returns=gross,
        trading_costs=trading_costs,
        funding=funding,
        net_returns=net,
        turnover=traded,
        timeframe=data.timeframe,
    )
