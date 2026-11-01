# Research

| Script | Report | Question |
|---|---|---|
| `real_data_v1.py` | [`reports/real_data_v1.md`](reports/real_data_v1.md) | Does any built-in factor survive costs out of sample on a survivorship-free Binance perp universe? |

Run with `uv run python research/real_data_v1.py`. The first run downloads about 1 GB from
data.binance.vision into `.cache/binance-vision`; later runs read the cache and take about 3 minutes.
`tests/test_research_smoke.py` runs the same protocol on synthetic data with a planted edge.

## v1 findings (2022-01 to 2026-09, 1h bars, monthly top-30 universe, 291 symbols)

**No factor passes the gate.** The gate is: out-of-sample net Sharpe of at least 0.5, and positive
in at least 3 of the 4 test years (2023–2026). Nothing here is ready to sell.

- **Momentum (2–4 week lookback) comes closest.** Out of sample: net Sharpe 0.49, +16% a year, and a
  max drawdown of -37%.
  - It's positive in only 2 of 4 years: 2025 (+42%) and 2026 (+25%) made the money, and 2023–2024
    were flat to negative.
  - It's not a hidden market bet: beta to the market is about -0.1.
  - One extra bar of execution delay barely changes it (0.70 → 0.67 in-sample), so the result isn't
    an artifact of trading on the same bar as the signal.
  - It's concentrated: the 3 best coins produce 38% of its price profit.
- **Funding and basis carry fail badly** (net Sharpe -0.8 to -1.6). Collecting funding (+28% a year) is
  swamped by the long leg: coins with negative funding or a perp discount keep falling. The crowd
  paying to be short is usually *right*. Flipping the signal looks profitable in hindsight, but
  that idea came from looking at these results, so it has to be validated as a new hypothesis.
- **Short-term reversal** has positive information coefficients (IC: how well the signal ranks next
  period's returns) but loses after costs: turnover is 280–830x a year.
- **Low volatility** has the highest information coefficient (+0.10). But it pays 14% a year in
  funding on its shorts and carries a beta of -0.27, which leaves almost nothing out of sample.
- **The universe itself loses money.** An equal-weight long position in the top 30 returned -21.5% a
  year with a -95% drawdown, mostly from newly listed coins fading after their launch. That's why the
  short leg profits in nearly every variant. The long/short leg split mostly reflects this drift, so
  judge each strategy by its total and its beta, not by which leg made money.

Not covered yet: `oi_trend` (open-interest history only exists as daily files; run with
`--with-oi`), maker execution, and volatility-scaled sizing.
