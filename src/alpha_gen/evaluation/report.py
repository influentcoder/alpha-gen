"""One-call factor tear sheet: IC, decay, turnover and a long/short backtest."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.data.universe import apply_universe
from alpha_gen.evaluation.backtest import BacktestResult, CostModel, run_backtest
from alpha_gen.evaluation.metrics import ICSummary, ic_decay, ic_summary, information_coefficient
from alpha_gen.factors.base import Factor
from alpha_gen.signals.portfolio import rebalance, to_weights
from alpha_gen.signals.transforms import cross_sectional_rank


@dataclass(frozen=True)
class FactorReport:
    name: str
    horizon: int
    ic: ICSummary
    decay: pd.Series
    backtest: BacktestResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "ic": self.ic.to_dict(),
            "decay": {int(k): float(v) for k, v in self.decay.items()},
            "backtest": self.backtest.summary(),
        }

    def __str__(self) -> str:
        bt = self.backtest.summary()
        ic = self.ic
        decay = "  ".join(f"{int(h)}:{v:+.4f}" for h, v in self.decay.items())
        return "\n".join(
            [
                f"== {self.name} ==",
                f"IC@{self.horizon}: mean {ic.mean:+.4f}  IR {ic.ir:+.3f}  t {ic.t_stat:+.2f}  "
                f"hit {ic.hit_rate:.1%}  n={ic.n}",
                f"IC decay (horizon:mean IC): {decay}",
                f"L/S backtest (net): sharpe {bt.get('sharpe', float('nan')):+.2f}  "
                f"ann ret {bt.get('ann_return', float('nan')):+.1%}  "
                f"max DD {bt.get('max_drawdown', float('nan')):.1%}  "
                f"turnover/yr {bt.get('ann_turnover', float('nan')):.0f}x  "
                f"gross sharpe {bt.get('gross_sharpe', float('nan')):+.2f}",
            ]
        )


def evaluate_factor(
    factor: Factor | pd.DataFrame,
    data: MarketData,
    *,
    name: str | None = None,
    horizon: int = 1,
    decay_horizons: Sequence[int] = (1, 2, 4, 8, 24, 72),
    costs: CostModel | None = None,
    universe: pd.DataFrame | None = None,
    rebalance_every: str | None = None,
) -> FactorReport:
    """Evaluate a factor (or a precomputed score panel) on ``data``.

    The backtest trades a dollar-neutral rank portfolio. By default it rebalances every bar,
    which is the *worst case* for costs; pass ``rebalance_every="8h"`` (or ``"1D"``) to trade on
    a schedule instead.

    Args:
        universe: Optional period membership (see :func:`alpha_gen.data.universe.top_n_membership`);
            scores outside the universe are ignored for both IC and the backtest.
    """
    scores = factor(data) if isinstance(factor, Factor) else factor
    if universe is not None:
        scores = apply_universe(scores, universe)
    label = name or (factor.label if isinstance(factor, Factor) else "signal")
    ic = information_coefficient(scores, data.forward_returns(horizon))
    weights = to_weights(cross_sectional_rank(scores))
    if rebalance_every is not None:
        weights = rebalance(weights, rebalance_every)
    return FactorReport(
        name=label,
        horizon=horizon,
        ic=ic_summary(ic),
        decay=ic_decay(scores, data.close, decay_horizons),
        backtest=run_backtest(weights, data, costs),
    )
