"""End-to-end walkthrough on synthetic data: factors -> signal -> backtest -> signal log.

Run with ``uv run python examples/quickstart.py``. Swap ``make_synthetic_market`` for
``CachedSource(CCXTSource()).fetch(...)`` to use real exchange data.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from alpha_gen import Strategy
from alpha_gen.data import make_synthetic_market
from alpha_gen.evaluation import CostModel, evaluate_factor, run_backtest
from alpha_gen.factors import FundingCarry, LowVolatility, Momentum
from alpha_gen.live import SignalLog

data = make_synthetic_market(n_symbols=20, n_periods=24 * 120, seed=1)
print(data, "\n")

# 1. Research individual factors.
for factor in (FundingCarry(lookback=24), Momentum(lookback=72), LowVolatility()):
    print(evaluate_factor(factor, data), "\n")

# 2. Blend them into a strategy and backtest, with execution one bar late to be conservative.
strategy = Strategy(
    name="carry_blend",
    factors={"carry": FundingCarry(lookback=24), "lowvol": LowVolatility()},
    weights={"carry": 2.0, "lowvol": 1.0},
    max_weight=0.15,
)
result = run_backtest(strategy.target_weights(data), data, CostModel(fee_bps=4), lag=2)
print("Strategy:", {k: round(v, 3) for k, v in result.summary().items()}, "\n")

# 3. Publish the latest weights to a tamper-evident log.
with TemporaryDirectory() as tmp:
    log = SignalLog(Path(tmp) / "signals.jsonl")
    weights = strategy.target_weights(data)
    for ts in weights.index[-3:]:
        log.append(ts, strategy.name, weights.loc[ts].to_dict(), published_at=ts)
    print(f"Logged {log.verify()} signals; head hash {log.last().hash}")
