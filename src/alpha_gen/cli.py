"""Command-line interface: ``alpha-gen --help``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.evaluation.backtest import CostModel
from alpha_gen.evaluation.report import evaluate_factor
from alpha_gen.factors import available_factors, get_factor
from alpha_gen.live.signal_log import LogIntegrityError, SignalLog

DEFAULT_SYMBOLS = "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT,XRP/USDT:USDT"


def _parse_params(items: Sequence[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        params[key] = json.loads(value)
    return params


def _evaluate(data: MarketData, names: Sequence[str], params: dict[str, Any], args: Any) -> None:
    costs = CostModel(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    for name in names:
        factor = get_factor(name, **(params if len(names) == 1 else {}))
        missing = [f for f in factor.requires if not data.has(f)]
        if missing:
            print(f"== {factor.label} ==\nskipped: missing {missing}\n")
            continue
        report = evaluate_factor(factor, data, horizon=args.horizon, costs=costs)
        print(json.dumps(report.to_dict(), indent=2) if args.json else f"{report}\n")


def cmd_factors(args: argparse.Namespace) -> None:
    for name, cls in available_factors().items():
        defaults = cls().params
        print(f"{name:15s} {cls.description}  defaults={defaults}  requires={cls.requires}")


def cmd_demo(args: argparse.Namespace) -> None:
    from alpha_gen.data.synthetic import make_synthetic_market

    data = make_synthetic_market(n_symbols=args.symbols, n_periods=args.periods, seed=args.seed)
    print(f"{data}\n(planted alpha: high funding -> lower returns)\n")
    _evaluate(data, args.factor or list(available_factors()), {}, args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from alpha_gen.data.cache import CachedSource
    from alpha_gen.data.ccxt_source import CCXTSource

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=args.days)
    source = CachedSource(
        CCXTSource(args.exchange, include_spot=args.spot, spot_exchange_id=args.spot_exchange),
        root=args.cache_dir,
    )
    data = source.fetch(args.symbols.split(","), start, end, args.timeframe)
    print(f"{data}\n")
    _evaluate(data, args.factor or list(available_factors()), _parse_params(args.param), args)


def cmd_verify_log(args: argparse.Namespace) -> None:
    log = SignalLog(args.path)
    try:
        count = log.verify()
    except LogIntegrityError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    last = log.last()
    print(f"OK: {count} records" + (f", head {last.hash}" if last else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-gen", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("factors", help="list registered factors").set_defaults(func=cmd_factors)

    def eval_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--factor", action="append", help="factor name (repeatable; default all)")
        p.add_argument("--horizon", type=int, default=1, help="IC horizon in bars")
        p.add_argument("--fee-bps", type=float, default=5.0)
        p.add_argument("--slippage-bps", type=float, default=2.0)
        p.add_argument("--json", action="store_true", help="print JSON reports")

    demo = sub.add_parser("demo", help="evaluate factors on synthetic data (no network)")
    demo.add_argument("--symbols", type=int, default=20)
    demo.add_argument("--periods", type=int, default=24 * 120)
    demo.add_argument("--seed", type=int, default=0)
    eval_options(demo)
    demo.set_defaults(func=cmd_demo)

    ev = sub.add_parser("evaluate", help="evaluate factors on exchange data (needs ccxt)")
    ev.add_argument("--exchange", default="binanceusdm")
    ev.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="comma-separated ccxt perp symbols")
    ev.add_argument("--timeframe", default="1h")
    ev.add_argument("--days", type=int, default=60)
    ev.add_argument("--end", help="end date (UTC), default now")
    ev.add_argument("--spot", action="store_true", help="also load spot closes for basis")
    ev.add_argument("--spot-exchange", default="binance")
    ev.add_argument("--cache-dir", default=".cache/alpha-gen")
    ev.add_argument("--param", action="append", default=[], help="factor param key=value")
    eval_options(ev)
    ev.set_defaults(func=cmd_evaluate)

    verify = sub.add_parser("verify-log", help="verify a hash-chained signal log")
    verify.add_argument("path")
    verify.set_defaults(func=cmd_verify_log)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
