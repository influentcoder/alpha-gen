# alpha-gen

A toolkit for researching and running trading signals ("alpha") on crypto perpetual futures.
You can use it as a library, a CLI, or as the engine behind a signal product.

```
data  ──▶  factors  ──▶  signals  ──▶  evaluation (research)
                                  └─▶  live runner ──▶ hash-chained signal log
```

| Layer | Module | What it does |
|---|---|---|
| Data | `alpha_gen.data` | `MarketData` holds aligned panels (perp close, quote volume, funding, open interest, spot). `BinanceVisionSource` loads years of history from Binance's free archive, including delisted coins. `CCXTSource` loads recent data from live exchanges, `CachedSource` stores it as Parquet, `top_n_membership` builds a point-in-time liquidity universe, and `make_synthetic_market` generates test data with a planted edge. |
| Factors | `alpha_gen.factors` | Registry of score functions. A higher score means a higher expected return. Built in: `funding_carry`, `basis_carry`, `oi_trend`, `momentum`, `reversal`, `low_vol`. |
| Signals | `alpha_gen.signals` | Cross-sectional z-score and rank, winsorising, static or IC-weighted blending (no look-ahead), dollar-neutral sizing, volatility targeting, and `Strategy` bundles. |
| Evaluation | `alpha_gen.evaluation` | Rank IC, IC decay, turnover, a backtest with fees, slippage and funding, walk-forward splits, and `evaluate_factor` tear sheets. |
| Live | `alpha_gen.live` | `SignalRunner` computes weights for the last complete bar. `SignalLog` is an append-only JSONL log where each record carries the previous record's hash, so the track record is tamper-evident. |

## Install

```bash
uv sync                    # core (numpy, pandas, pyarrow)
uv sync --extra exchange   # + ccxt for live exchange data
```

## Quickstart

```bash
uv run alpha-gen factors                      # list factors and their parameters
uv run alpha-gen demo                         # evaluate every factor on synthetic data (no network)
uv run alpha-gen evaluate --days 90 --factor funding_carry --param lookback=24
uv run alpha-gen verify-log signals/live.jsonl
uv run python examples/quickstart.py          # library walkthrough
```

```python
import pandas as pd
from alpha_gen import Strategy
from alpha_gen.data import CachedSource
from alpha_gen.data.ccxt_source import CCXTSource
from alpha_gen.evaluation import CostModel, evaluate_factor, run_backtest
from alpha_gen.factors import FundingCarry, Momentum

source = CachedSource(CCXTSource("binanceusdm"))
end = pd.Timestamp.now(tz="UTC").floor("h")
data = source.fetch(
    ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"], end - pd.Timedelta(days=90), end
)

print(evaluate_factor(FundingCarry(lookback=24), data))

strategy = Strategy("carry_momo", {"carry": FundingCarry(), "momo": Momentum(lookback=72)})
print(run_backtest(strategy.target_weights(data), data, CostModel(fee_bps=4), lag=2).summary())
```

## Historical data and research

`BinanceVisionSource` reads [data.binance.vision](https://data.binance.vision): monthly and daily
zipped CSVs of perp and spot klines, funding and open interest. It's free and has no rate limits.
Parsed files are cached under `.cache/binance-vision`.

```python
from alpha_gen.data import top_n_membership, candidates
from alpha_gen.data.binance_vision import BinanceVisionSource, list_perp_symbols

src = BinanceVisionSource(include_spot=True)
symbols = list_perp_symbols()  # every USDT perp ever listed, incl. delisted
volume = src.monthly_quote_volume(symbols, "2021-12-01", "2026-10-01")
universe = top_n_membership(volume, top_n=30)  # month m uses month m-1 volume
data = src.fetch(candidates(universe), "2022-01-01", "2026-10-01", "1h")
```

`research/real_data_v1.py` runs the full validation protocol: survivorship-free universe, costs,
funding, and walk-forward out-of-sample selection. It writes `research/reports/real_data_v1.md`.

## Conventions (read before adding factors)

- Panels are wide DataFrames: a UTC index of **bar open times**, with one column per ccxt perp symbol.
  A value labelled `t` is treated as known at the close of bar `t`.
- `funding_rate` is set only on settlement bars and is NaN elsewhere. Factors forward-fill it, and
  the backtester charges it only on those bars (longs pay positive funding).
- `run_backtest(weights, data, lag=1)` holds `weights[t]` over bar `t+1`. `lag=0` is rejected.
- Every registered factor is tested automatically for look-ahead: its scores must not change when
  future data is removed. Register a new factor with `@register`, and it is covered by that test.

## Adding a factor

```python
from dataclasses import dataclass
from alpha_gen.factors import Factor, register


@register
@dataclass(frozen=True)
class VolumeSurge(Factor):
    name = "volume_surge"
    requires = ("close", "volume")
    description = "Volume vs its trailing mean"
    lookback: int = 48

    def compute(self, data):
        vol = data.require("volume")
        return vol / vol.rolling(self.lookback).mean() - 1
```

## Publishing signals

`SignalLog` writes one JSON record per signal. Each record contains the previous record's hash, so
changing or deleting any past signal breaks `verify()`. To prove that signals were published
*before* the outcome, regularly anchor the latest hash somewhere you can't rewrite: a public git
commit, a social post, or OpenTimestamps.

## Development

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest -m network   # opt-in smoke test against the live Binance archive
```

## Roadmap

- [ ] Liquidations and order-book data adapters (websocket)
- [ ] Hyperliquid / OKX / Bybit sources, plus universe selection by liquidity
- [ ] Walk-forward parameter search, and adjusting t-statistics for overlapping horizons
- [ ] Scheduled runner and a public track-record page built from the signal log
- [ ] Hosted signal/factor API (paid tier)

> Not investment advice. Backtests on synthetic or historical data don't guarantee future results.
