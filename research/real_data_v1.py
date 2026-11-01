"""Real-data factor validation (MVP step 1).

Question: does any built-in factor survive realistic costs, out of sample, on a
survivorship-bias-free universe of Binance USDT perpetuals?

Protocol
--------
* Universe: each month, the top ``TOP_N`` USDT perps by the *previous* month's quote volume,
  chosen from every perp the archive has ever listed (delisted ones included).
* Portfolio: dollar-neutral cross-sectional rank weights, gross 1.0, rebalanced every 8h or daily,
  traded at the decision bar's close (lag=1); taker fee + slippage charged on turnover; funding
  paid/received on held positions.
* Out-of-sample: walk-forward with a 1-year training window and 3-month test windows. Within
  each factor family, the variant (lookback x rebalance frequency) with the best training Sharpe
  is traded in the next test window. Stitched test returns are the OOS estimate.
* Gate (per family): OOS net Sharpe >= 0.5 AND positive OOS net return in >= 3 of the 4
  calendar years tested (2023, 2024, 2025, 2026 YTD).

Run: ``uv run python research/real_data_v1.py`` (first run downloads ~1 GB, then cached).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_gen.data import apply_universe, candidates, top_n_membership
from alpha_gen.data.binance_vision import BinanceVisionSource, list_perp_symbols
from alpha_gen.evaluation import (
    CostModel,
    annualized_sharpe,
    ic_summary,
    information_coefficient,
    performance_summary,
    run_backtest,
    walk_forward_select,
    walk_forward_splits,
)
from alpha_gen.factors import (
    BasisCarry,
    Factor,
    FundingCarry,
    LowVolatility,
    Momentum,
    OpenInterestTrend,
    ShortTermReversal,
)
from alpha_gen.signals import combine, cross_sectional_rank, rebalance, to_weights
from alpha_gen.timeframes import periods_per_year

START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")  # last complete month with funding data
TOP_N = 30
TIMEFRAME = "1h"
PPY = periods_per_year(TIMEFRAME)
COSTS = CostModel(fee_bps=5.0, slippage_bps=3.0)  # Binance VIP0 taker + conservative slippage
REBALANCE = ("8h", "1D")
TRAIN_BARS, TEST_BARS, GAP_BARS = 365 * 24, 91 * 24, 24
GATE_SHARPE, GATE_YEARS = 0.5, 3

FAMILIES: dict[str, list[Factor]] = {
    "funding_carry": [FundingCarry(lookback=n) for n in (24, 72, 168)],
    "basis_carry": [BasisCarry(lookback=n) for n in (24, 72, 168)],
    "momentum": [Momentum(lookback=n) for n in (72, 168, 336, 720)]
    + [Momentum(lookback=720, skip=24)],
    "reversal": [ShortTermReversal(lookback=n) for n in (24, 72)],
    "low_vol": [LowVolatility(lookback=n) for n in (168, 720)],
}
OI_FAMILY = {"oi_trend": [OpenInterestTrend(lookback=n) for n in (24, 72)]}

log = logging.getLogger("research")


@dataclass
class Variant:
    family: str
    label: str
    rebalance: str
    net: pd.Series
    gross: pd.Series
    summary: dict[str, float]
    ic: dict[str, float]


def load(cache_dir: Path, with_oi: bool) -> tuple[pd.DataFrame, object, object | None]:
    src = BinanceVisionSource(cache_dir, include_spot=True, max_workers=48)
    symbols = list_perp_symbols()
    log.info("%d USDT perps in archive (incl. delisted)", len(symbols))
    volume = src.monthly_quote_volume(symbols, START - pd.DateOffset(months=1), END)
    membership = top_n_membership(volume, TOP_N).loc[START:]
    names = candidates(membership)
    membership = membership[names]
    log.info("%d symbols are in the top %d at least once", len(names), TOP_N)
    data = src.fetch(names, START, END, TIMEFRAME)
    oi_data = None
    if with_oi:
        oi_start = END - pd.DateOffset(years=1)
        oi_src = BinanceVisionSource(cache_dir, include_open_interest=True, max_workers=48)
        oi_data = oi_src.fetch(names, oi_start, END, TIMEFRAME)
    return membership, data, oi_data


def run_variant(family: str, factor: Factor, data, membership, freq: str) -> Variant:
    scores = apply_universe(factor(data), membership)
    weights = rebalance(to_weights(cross_sectional_rank(scores)), freq)
    bt = run_backtest(weights, data, COSTS)
    horizon = 8 if freq == "8h" else 24
    ic = ic_summary(information_coefficient(scores, data.forward_returns(horizon)))
    return Variant(
        family=family,
        label=f"{factor.label}@{freq}",
        rebalance=freq,
        net=bt.net_returns,
        gross=bt.gross_returns,
        summary=bt.summary(),
        ic={"horizon": horizon, "mean": ic.mean, "ir": ic.ir, "hit_rate": ic.hit_rate},
    )


def yearly_returns(returns: pd.Series) -> dict[int, dict[str, float]]:
    out = {}
    for year, r in returns.groupby(returns.index.year):
        out[int(year)] = {"return": float(r.sum()), "sharpe": annualized_sharpe(r, PPY)}
    return out


def fmt(x: float, pct: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.1%}" if pct else f"{x:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/binance-vision"))
    parser.add_argument("--out", type=Path, default=Path("research/reports"))
    parser.add_argument("--with-oi", action="store_true", help="also test oi_trend (last 12m)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("alpha_gen").setLevel(logging.ERROR)
    t0 = time.time()

    membership, data, oi_data = load(args.cache_dir, args.with_oi)
    log.info("Loaded %s in %.0fs", data, time.time() - t0)
    coverage = {
        "funding": float(
            apply_universe(data.funding_rate.ffill(), membership).notna().mean().mean()
        ),
        "spot": float(apply_universe(data.spot_close, membership).notna().mean().mean()),
    }

    variants: list[Variant] = []
    for family, factors in FAMILIES.items():
        for factor in factors:
            for freq in REBALANCE:
                variants.append(run_variant(family, factor, data, membership, freq))
                log.info(
                    "%-40s net sharpe %s", variants[-1].label, fmt(variants[-1].summary["sharpe"])
                )

    # Equal-weight blend of every family's default factor: no selection involved.
    blend_scores = combine(
        {name: apply_universe(fs[0](data), membership) for name, fs in FAMILIES.items()}
    )
    for freq in REBALANCE:
        weights = rebalance(to_weights(cross_sectional_rank(blend_scores)), freq)
        bt = run_backtest(weights, data, COSTS)
        variants.append(
            Variant(
                "blend_equal",
                f"blend_equal@{freq}",
                freq,
                bt.net_returns,
                bt.gross_returns,
                bt.summary(),
                {},
            )
        )

    # Benchmarks (context only): equal-weight long universe and BTC buy-and-hold.
    in_universe = apply_universe(data.close * 0 + 1.0, membership).fillna(0.0)
    long_w = rebalance(
        in_universe.div(in_universe.sum(axis=1).replace(0, np.nan), axis=0).fillna(0), "1D"
    )
    long_bt = run_backtest(long_w, data, COSTS)
    btc = data.returns()["BTCUSDT"].fillna(0.0) if "BTCUSDT" in data.symbols else None

    # Walk-forward selection within each family.
    splits = list(walk_forward_splits(data.index, TRAIN_BARS, TEST_BARS, gap=GAP_BARS))
    families = sorted({v.family for v in variants})
    oos: dict[str, dict] = {}
    for family in families:
        members = {v.label: v.net for v in variants if v.family == family}
        wf = walk_forward_select(members, splits, periods_per_year=PPY)
        perf = performance_summary(wf.returns, PPY)
        years = yearly_returns(wf.returns)
        positive_years = sum(1 for y in years.values() if y["return"] > 0)
        passed = perf.get("sharpe", -np.inf) >= GATE_SHARPE and positive_years >= GATE_YEARS
        oos[family] = {
            "performance": perf,
            "years": years,
            "positive_years": positive_years,
            "passed": bool(passed),
            "choices": wf.choices.assign(test_start=wf.choices["test_start"].astype(str))[
                ["test_start", "chosen", "train_score", "test_score"]
            ].to_dict("records"),
        }
    # Selection across *all* variants: the honest number if you "pick the best thing" each quarter.
    wf_all = walk_forward_select({v.label: v.net for v in variants}, splits, periods_per_year=PPY)
    oos_all = {
        "performance": performance_summary(wf_all.returns, PPY),
        "years": yearly_returns(wf_all.returns),
        "chosen": wf_all.choices["chosen"].tolist(),
    }

    oi_results = []
    if oi_data is not None:
        for factor in OI_FAMILY["oi_trend"]:
            for freq in REBALANCE:
                v = run_variant("oi_trend", factor, oi_data, membership, freq)
                oi_results.append({"label": v.label, "summary": v.summary, "ic": v.ic})

    write_report(
        args.out,
        data,
        membership,
        coverage,
        variants,
        oos,
        oos_all,
        long_bt,
        btc,
        oi_results,
        splits,
        time.time() - t0,
    )


def write_report(
    out,
    data,
    membership,
    coverage,
    variants,
    oos,
    oos_all,
    long_bt,
    btc,
    oi_results,
    splits,
    elapsed,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    first_test, last_test = splits[0][1][0], splits[-1][1][-1]
    lines = [
        "# Real-data factor validation v1",
        "",
        f"Generated by `research/real_data_v1.py` in {elapsed:.0f}s.",
        "",
        "## Setup",
        "",
        f"- Data: Binance USDT-M perps, {TIMEFRAME} bars, {START.date()} to {END.date()} "
        f"(data.binance.vision).",
        f"- Universe: monthly top {TOP_N} by previous-month quote volume, from every perp ever "
        f"listed; {membership.shape[1]} distinct symbols over the period.",
        f"- Coverage inside the universe: funding {coverage['funding']:.0%}, "
        f"spot price {coverage['spot']:.0%}.",
        "- Portfolio: dollar-neutral rank weights, gross 1.0, rebalanced "
        f"{' or '.join(REBALANCE)}.",
        f"- Costs: {COSTS.fee_bps:g} bps fee + {COSTS.slippage_bps:g} bps slippage per unit "
        "traded; funding charged on positions.",
        f"- Walk-forward: train {TRAIN_BARS // 24} days, test {TEST_BARS // 24} days, gap "
        f"{GAP_BARS} bars; OOS period {first_test.date()} to {last_test.date()} "
        f"({len(splits)} test windows).",
        f"- Gate: OOS net Sharpe >= {GATE_SHARPE} and positive in >= {GATE_YEARS} calendar years.",
        "",
        "## Verdict (out of sample, per family)",
        "",
        "| Family | OOS net Sharpe | OOS ann. return | Max DD | Positive years | Gate |",
        "|---|---|---|---|---|---|",
    ]
    for family, r in sorted(oos.items(), key=lambda kv: -kv[1]["performance"].get("sharpe", -9)):
        p = r["performance"]
        lines.append(
            f"| {family} | {fmt(p.get('sharpe'))} | {fmt(p.get('ann_return'), True)} | "
            f"{fmt(p.get('max_drawdown'), True)} | {r['positive_years']}/{len(r['years'])} | "
            f"{'**PASS**' if r['passed'] else 'fail'} |"
        )
    pa = oos_all["performance"]
    lines += [
        "",
        f"Picking the best of *all* {len(variants)} variants each quarter (training data only): "
        f"OOS net Sharpe {fmt(pa.get('sharpe'))}, ann. return {fmt(pa.get('ann_return'), True)}, "
        f"max DD {fmt(pa.get('max_drawdown'), True)}.",
        "",
        "### OOS return by calendar year",
        "",
        "| Family | "
        + " | ".join(str(y) for y in sorted(next(iter(oos.values()))["years"]))
        + " |",
        "|---|" + "---|" * len(next(iter(oos.values()))["years"]),
    ]
    for family, r in sorted(oos.items()):
        cells = [fmt(r["years"][y]["return"], True) for y in sorted(r["years"])]
        lines.append(f"| {family} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## All variants, full period (in-sample: for intuition, not for decisions)",
        "",
        "| Variant | Net Sharpe | Gross Sharpe | Ann. return | Max DD | Turnover/yr | Cost/yr | "
        "Funding/yr | IC mean | IC IR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in sorted(variants, key=lambda v: -np.nan_to_num(v.summary.get("sharpe", -9), nan=-9)):
        s = v.summary
        lines.append(
            f"| {v.label} | {fmt(s.get('sharpe'))} | {fmt(s.get('gross_sharpe'))} | "
            f"{fmt(s.get('ann_return'), True)} | {fmt(s.get('max_drawdown'), True)} | "
            f"{s.get('ann_turnover', float('nan')):.0f}x | {fmt(s.get('ann_cost'), True)} | "
            f"{fmt(s.get('ann_funding'), True)} | {v.ic.get('mean', float('nan')):+.4f} | "
            f"{fmt(v.ic.get('ir', float('nan')))} |"
        )
    lb = long_bt.summary()
    lines += [
        "",
        "## Benchmarks (long-only, context)",
        "",
        f"- Equal-weight long universe: Sharpe {fmt(lb.get('sharpe'))}, ann. return "
        f"{fmt(lb.get('ann_return'), True)}, max DD {fmt(lb.get('max_drawdown'), True)}.",
    ]
    if btc is not None:
        bp = performance_summary(btc, PPY)
        lines.append(
            f"- BTC buy and hold: Sharpe {fmt(bp.get('sharpe'))}, ann. return "
            f"{fmt(bp.get('ann_return'), True)}, max DD {fmt(bp.get('max_drawdown'), True)}."
        )
    if oi_results:
        lines += [
            "",
            "## oi_trend (last 12 months only: open interest history is daily files)",
            "",
            "| Variant | Net Sharpe | Gross Sharpe | IC mean |",
            "|---|---|---|---|",
        ]
        for r in oi_results:
            lines.append(
                f"| {r['label']} | {fmt(r['summary'].get('sharpe'))} | "
                f"{fmt(r['summary'].get('gross_sharpe'))} | {r['ic'].get('mean'):+.4f} |"
            )
    lines += [
        "",
        "## Caveats",
        "",
        "- Returns are arithmetic sums of per-bar returns at gross 1.0; no leverage or vol "
        "targeting.",
        "- Trades fill at the bar close used for the decision (lag=1); add a bar of delay to "
        "stress it.",
        "- Delisted coins keep their last price after delisting (positions exit at the next "
        "rebalance), so delisting losses after the final traded price are not modelled.",
        "- Walk-forward still selects among a grid we chose after seeing crypto literature; treat "
        "a marginal pass as 'needs more evidence', not as proof.",
    ]
    (out / "real_data_v1.md").write_text("\n".join(lines) + "\n")

    payload = {
        "config": {"start": str(START), "end": str(END), "top_n": TOP_N, "costs": vars(COSTS)},
        "coverage": coverage,
        "oos": oos,
        "oos_all": oos_all,
        "variants": [
            {
                "label": v.label,
                "family": v.family,
                "summary": v.summary,
                "ic": v.ic,
                "years": yearly_returns(v.net),
            }
            for v in variants
        ],
        "oi_trend": oi_results,
    }
    (out / "real_data_v1.json").write_text(json.dumps(payload, indent=2, default=str))
    log.info("Report written to %s", out / "real_data_v1.md")


if __name__ == "__main__":
    main()
