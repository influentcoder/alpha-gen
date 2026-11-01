# Research

| Script | Report | Question |
|---|---|---|
| `real_data_v1.py` | [`reports/real_data_v1.md`](reports/real_data_v1.md) | Does any built-in factor survive costs out of sample on a survivorship-free Binance perp universe? |
| `v2_hypotheses.py` | [`reports/v2.md`](reports/v2.md) | Do the six specs [pre-registered](v2_preregistration.md) after v1 pass the gate? |

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

## v2 findings (pre-registered, 2023-01 to 2026-09)

**No pre-registered hypothesis passes.** All specs run through `Strategy.target_weights`, the same
code path as the paper trader.

- **H2, momentum on coins listed at least 60 days, comes closest.** Net Sharpe 0.94, positive in all
  4 years, and 0.87 with one bar of execution delay. It **fails** the concentration rule: the top 3
  coins produce 50.1% of its profit, against a 50% limit. Under the pre-registration a marginal fail
  is still a fail.
- **B0** (momentum with a 336h lookback, the v1 in-sample winner) clears every gate check, but it's
  the reference, not a hypothesis: it was picked after looking at the data. v1's walk-forward
  estimate for momentum was 0.49, and the gap from B0's 0.97 is roughly the cost of that selection.
- **Most of the profit comes from 2025.** B0 made +92% in 2025 and -8% in 2024.
- **Vol scaling (H1, H5) hurts**: it lowers Sharpe and leaves fewer profitable years.
- **H4, BTC/ETH trend** (+35% a year, positive every year), loses to a 50/50 buy-and-hold on Sharpe
  (0.74 vs 0.92).
- **H3, the flipped funding signal**, is positive (0.40) but too weak and inconsistent.
- **Maker execution adds about 0.2 Sharpe to every spec.** Fill quality is worth engineering.

**Decision:** paper-trade cross-sectional momentum (`strategies/momentum_v0.toml`, the B0 logic on
daily bars) as an **experimental** strategy, so that forward evidence starts accumulating. Don't sell
or promote it until the forward record supports it.
