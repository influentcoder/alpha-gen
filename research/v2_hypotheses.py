"""v2: run exactly the specs pre-registered in ``research/v2_preregistration.md``.

Cross-sectional specs run through :class:`alpha_gen.Strategy`, the same code path as the live
paper trader, so a spec that passes here can be deployed without re-implementation.

Run: ``uv run python research/v2_hypotheses.py`` (reuses the v1 data cache).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from real_data_v1 import PPY, START, beta, fmt, load

from alpha_gen import Strategy
from alpha_gen.data import MarketData, expand_membership
from alpha_gen.evaluation import CostModel, annualized_sharpe, performance_summary, run_backtest
from alpha_gen.factors import FundingCarry, Momentum
from alpha_gen.signals import rebalance, time_series_trend

TAKER = CostModel(fee_bps=5.0, slippage_bps=3.0)
MAKER = CostModel(fee_bps=2.0, slippage_bps=1.0)
GATE_START, GATE_END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
MAJORS = ["BTCUSDT", "ETHUSDT"]

log = logging.getLogger("v2")


def momentum(**kw) -> Strategy:
    return Strategy(
        name="v2",
        factors={"momo": Momentum(lookback=336)},
        universe_top_n=30,
        rebalance_every="1D",
        **kw,
    )


SPECS: dict[str, tuple[str, Strategy | None]] = {
    "B0": ("Reference: Momentum(336), rank weights", momentum()),
    "H1": ("Vol-scaled momentum (720h vol)", momentum(vol_lookback=720)),
    "H2": ("Momentum, seasoned universe (>= 60 days listed)", momentum(min_listing_age_days=60)),
    "H3": (
        "Crowd-is-right funding: +mean(funding, 168h)",
        Strategy(
            name="v2",
            factors={"funding": FundingCarry(lookback=168)},
            weights={"funding": -1.0},
            universe_top_n=30,
            rebalance_every="1D",
        ),
    ),
    "H4": ("BTC/ETH trend: 0.5 x sign(720h return) each", None),
    "H5": (
        "H1 + H2: vol-scaled momentum, seasoned universe",
        momentum(vol_lookback=720, min_listing_age_days=60),
    ),
}


@dataclass
class Outcome:
    spec: str
    description: str
    gate: dict[str, float | bool | str]
    years: dict[int, float]
    context_2022: float
    maker_sharpe: float
    beta_ew: float


def weights_for(strategy: Strategy | None, data: MarketData) -> pd.DataFrame:
    """Target weights via the live code path (``Strategy.target_weights``)."""
    if strategy is None:
        return rebalance(time_series_trend(data.close, 720, weight=0.5, symbols=MAJORS), "1D")
    return strategy.target_weights(data)


def universe_agreement(membership: pd.DataFrame, data: MarketData) -> float:
    """Share of universe slots where Strategy's own volume ranking matches the archive-wide one."""
    archive = expand_membership(membership, data.index).reindex(
        columns=data.symbols, fill_value=False
    )
    own = momentum().tradable(data)
    assert own is not None
    rows = archive.any(axis=1)
    return float((own[rows] & archive[rows]).to_numpy().sum() / archive[rows].to_numpy().sum())


def evaluate(spec, description, weights, data, market) -> Outcome:
    window = slice(GATE_START, GATE_END - pd.Timedelta(1))
    taker = run_backtest(weights, data, TAKER)
    net = taker.net_returns.loc[window]
    perf = performance_summary(net, PPY)
    years = {int(y): float(r.sum()) for y, r in net.groupby(net.index.year)}
    lag2 = annualized_sharpe(run_backtest(weights, data, TAKER, lag=2).net_returns.loc[window], PPY)
    maker = annualized_sharpe(run_backtest(weights, data, MAKER).net_returns.loc[window], PPY)
    price_pnl = (taker.positions * data.returns().fillna(0.0)).loc[window].sum().sort_values()
    total = float(price_pnl.sum())
    top3 = float(price_pnl.tail(3).sum() / total) if total > 0 else float("nan")

    sharpe = perf.get("sharpe", float("nan"))
    positive = sum(1 for v in years.values() if v > 0)
    checks = {
        "sharpe>=0.5": bool(sharpe >= 0.5),
        "positive_years>=3": positive >= 3,
        "lag2>=0.4": bool(lag2 >= 0.4),
    }
    if spec == "H4":
        hold = data.returns()[MAJORS].fillna(0.0).mean(axis=1).loc[window]
        hold_sharpe = annualized_sharpe(hold, PPY)
        checks["beats_buy_and_hold"] = bool(sharpe > hold_sharpe)
    else:
        hold_sharpe = float("nan")
        checks["top3<50%"] = bool(top3 < 0.5)
    context = taker.net_returns.loc[START : GATE_START - pd.Timedelta(1)].sum()
    return Outcome(
        spec=spec,
        description=description,
        gate={
            "sharpe": sharpe,
            "ann_return": perf.get("ann_return", float("nan")),
            "max_drawdown": perf.get("max_drawdown", float("nan")),
            "positive_years": positive,
            "lag2_sharpe": lag2,
            "top3_share": top3,
            "buy_hold_sharpe": hold_sharpe,
            **checks,
            "passed": all(checks.values()),
        },
        years=years,
        context_2022=float(context),
        maker_sharpe=maker,
        beta_ew=beta(net, market.loc[window]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/binance-vision"))
    parser.add_argument("--out", type=Path, default=Path("research/reports"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("alpha_gen").setLevel(logging.ERROR)

    membership, data, _ = load(args.cache_dir, False)
    mask = expand_membership(membership, data.index).reindex(columns=data.symbols, fill_value=False)
    ew = mask.astype(float).div(mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    market = run_backtest(rebalance(ew, "1D"), data, TAKER).gross_returns

    agreement = universe_agreement(membership, data)
    log.info("Strategy universe matches archive-wide top 30 on %.2f%% of slots", 100 * agreement)
    outcomes = []
    for spec, (description, strategy) in SPECS.items():
        weights = weights_for(strategy, data)
        outcomes.append(evaluate(spec, description, weights, data, market))
        g = outcomes[-1].gate
        log.info("%s sharpe %s passed=%s", spec, fmt(g["sharpe"]), g["passed"])
    write(args.out, outcomes, agreement)


def write(out: Path, outcomes: list[Outcome], agreement: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    years = sorted({y for o in outcomes for y in o.years})
    lines = [
        "# v2 pre-registered hypotheses: results",
        "",
        "Specs and gate: [`research/v2_preregistration.md`](../v2_preregistration.md), committed "
        "before this run. Gate window: 2023-01-01 to 2026-09-30, taker costs (5 + 3 bps), "
        "funding included. B0 is a reference, not a hypothesis.",
        "",
        "All specs run through `Strategy.target_weights`, the live paper-trading code path. Its "
        "own monthly volume ranking matches the archive-wide top-30 universe on "
        f"{agreement:.1%} of universe slots.",
        "",
        "| Spec | Description | Net Sharpe | Ann. return | Max DD | Positive years | Lag-2 Sharpe "
        "| Top-3 share | Beta EW | Gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        g = o.gate
        verdict = (
            "**PASS**"
            if g["passed"]
            else "fail ("
            + ", ".join(k for k, v in g.items() if isinstance(v, bool) and k != "passed" and not v)
            + ")"
        )
        lines.append(
            f"| {o.spec} | {o.description} | {fmt(g['sharpe'])} | {fmt(g['ann_return'], True)} | "
            f"{fmt(g['max_drawdown'], True)} | {g['positive_years']}/{len(o.years)} | "
            f"{fmt(g['lag2_sharpe'])} | {fmt(g['top3_share'], True)} | {fmt(o.beta_ew)} | "
            f"{verdict} |"
        )
    lines += [
        "",
        "## Net return by calendar year",
        "",
        "| Spec | 2022 (context) | " + " | ".join(str(y) for y in years) + " |",
        "|---|---|" + "---|" * len(years),
    ]
    for o in outcomes:
        cells = [fmt(o.years.get(y, float("nan")), True) for y in years]
        lines.append(f"| {o.spec} | {fmt(o.context_2022, True)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Secondary: maker execution (2 + 1 bps), not part of the gate",
        "",
        "| Spec | Net Sharpe (taker) | Net Sharpe (maker) |",
        "|---|---|---|",
    ]
    for o in outcomes:
        lines.append(f"| {o.spec} | {fmt(o.gate['sharpe'])} | {fmt(o.maker_sharpe)} |")
    h4 = next(o for o in outcomes if o.spec == "H4")
    lines += [
        "",
        f"H4 buy-and-hold comparison: 50/50 BTC/ETH Sharpe {fmt(h4.gate['buy_hold_sharpe'])}.",
    ]
    (out / "v2.md").write_text("\n".join(lines) + "\n")
    (out / "v2.json").write_text(json.dumps([o.__dict__ for o in outcomes], indent=2, default=str))
    log.info("Report written to %s", out / "v2.md")


if __name__ == "__main__":
    main()
