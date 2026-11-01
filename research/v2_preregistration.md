# v2 pre-registration

Committed **before** any v2 backtest was run; the git history is the timestamp. v2 runs exactly
these specs: no parameter grids, no extra variants. A failing spec is reported as a fail, not
tweaked.

## Honesty note

v1 already used this data (2022-01 to 2026-09), so v2 is **not** truly out of sample: the
hypotheses below were inspired by v1's results. Pre-registering limits how many tries we get
(six specs). Real out-of-sample evidence will only come from forward paper trading, which starts
alongside v2.

## Common setup (same as v1)

- Binance USDT-M perps, 1h bars, data.binance.vision.
- Universe: monthly top 30 by previous-month quote volume, chosen from every perp ever listed.
- Costs: 5 bps taker fee + 3 bps slippage per unit traded; funding on positions; lag=1.
- Cross-sectional books: dollar-neutral, gross 1.0, rebalanced daily at 00:00 UTC.

## Specs

| ID | Spec |
|---|---|
| B0 | Reference only (v1's in-sample best, not a hypothesis): `Momentum(lookback=336)`, rank weights. |
| H1 | **Vol-scaled momentum.** As B0, but each symbol's rank score is divided by its trailing 720h volatility before neutralising (`to_weights(volatility=...)`). |
| H2 | **Seasoned universe.** As B0, but symbols listed less than 60 days ago (counted from their first bar in the archive) are excluded. |
| H3 | **Crowd-is-right funding.** Score = `+mean(funding, 168h)`, the opposite sign of `FundingCarry(168)`, with rank weights. |
| H4 | **BTC/ETH trend.** Directional, not cross-sectional: weight 0.5 × sign(720h return) on each of BTCUSDT and ETHUSDT. |
| H5 | **Pre-specified combo: H1 + H2.** Vol-scaled momentum on the seasoned universe. |

Secondary result, which doesn't count toward the gate: each spec re-costed with maker execution
(2 bps fee + 1 bp slippage).

## Gate (every condition must hold, on 2023-01-01 to 2026-09-30, taker costs)

1. Net Sharpe ≥ 0.5.
2. Positive net return in ≥ 3 of the 4 calendar years.
3. Net Sharpe with one extra bar of execution delay (lag=2) ≥ 0.4.
4. Cross-sectional specs only: the top 3 symbols contribute < 50% of price PnL.
5. H4 only: also beats 50/50 BTC/ETH buy-and-hold on Sharpe.

2022 is reported for context and doesn't count toward the gate. Any spec that passes is added as a
second paper-trading strategy. It isn't considered validated until forward results agree.
