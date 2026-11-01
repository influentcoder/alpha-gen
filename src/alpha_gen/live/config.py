"""Versioned strategy configs (TOML) for the live runner.

Example ``strategies/momentum_v0.toml``::

    name = "momentum_v0"
    description = "Cross-sectional 14-day momentum, monthly top-30 universe, daily rebalance"
    timeframe = "1d"
    history_bars = 100

    [source]
    exchange = "okx"
    quote = "USDT"

    [universe]
    top_n = 30
    min_listing_age_days = 0

    [[factors]]
    label = "momo"
    name = "momentum"
    weight = 1.0
    params = { lookback = 14 }

    [portfolio]
    rebalance_every = "1D"

    [costs]
    fee_bps = 5.0
    slippage_bps = 3.0

The file's SHA-256 is stored in every log record, so any edit to a live config is visible in the
track record. Edit the logic -> bump the ``name``.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha_gen.data.universe import DEFAULT_EXCLUDE
from alpha_gen.evaluation.backtest import CostModel
from alpha_gen.factors import get_factor
from alpha_gen.signals.strategy import Strategy
from alpha_gen.timeframes import timeframe_to_offset


@dataclass(frozen=True)
class LiveConfig:
    strategy: Strategy
    timeframe: str
    history_bars: int
    exchange: str
    quote: str
    costs: CostModel
    symbols: list[str] | None = None
    source_options: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""


def load_config(path: str | Path) -> LiveConfig:
    raw_bytes = Path(path).read_bytes()
    raw = tomllib.loads(raw_bytes.decode())
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        name = raw["name"]
        factor_specs = raw["factors"]
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from None

    factors, weights = {}, {}
    for spec in factor_specs:
        label = spec.get("label", spec["name"])
        if label in factors:
            raise ValueError(f"{path}: duplicate factor label {label!r}")
        factors[label] = get_factor(spec["name"], **spec.get("params", {}))
        weights[label] = float(spec.get("weight", 1.0))

    universe = raw.get("universe", {})
    portfolio = raw.get("portfolio", {})
    source = raw.get("source", {})
    timeframe = raw.get("timeframe", "1h")
    strategy = Strategy(
        name=name,
        factors=factors,
        weights=weights,
        normalize=portfolio.get("normalize", "rank"),
        gross=float(portfolio.get("gross", 1.0)),
        market_neutral=bool(portfolio.get("market_neutral", True)),
        max_weight=portfolio.get("max_weight"),
        universe_top_n=universe.get("top_n"),
        universe_exclude=frozenset(universe.get("exclude", DEFAULT_EXCLUDE)),
        min_listing_age_days=universe.get("min_listing_age_days") or None,
        vol_lookback=portfolio.get("vol_lookback"),
        rebalance_every=portfolio.get("rebalance_every"),
        metadata={"description": raw.get("description", ""), "config_sha256": sha},
    )
    history_bars = int(raw.get("history_bars", 100))
    _check_history(strategy, timeframe, history_bars, path)
    costs = raw.get("costs", {})
    return LiveConfig(
        strategy=strategy,
        timeframe=timeframe,
        history_bars=history_bars,
        exchange=source.get("exchange", "okx"),
        quote=source.get("quote", "USDT"),
        symbols=universe.get("symbols"),
        source_options=source.get("options", {}),
        costs=CostModel(
            fee_bps=float(costs.get("fee_bps", 5.0)),
            slippage_bps=float(costs.get("slippage_bps", 3.0)),
        ),
        sha256=sha,
    )


def _check_history(strategy: Strategy, timeframe: str, history_bars: int, path: Any) -> None:
    span = timeframe_to_offset(timeframe) * history_bars
    days = span.total_seconds() / 86_400
    if strategy.universe_top_n is not None and days < 62:
        raise ValueError(f"{path}: a monthly volume universe needs history >= 62 days")
    if strategy.min_listing_age_days and days < strategy.min_listing_age_days + 30:
        raise ValueError(f"{path}: history must exceed min_listing_age_days by 30+ days")
    for factor in strategy.factors.values():
        lookback = max((v for v in factor.params.values() if isinstance(v, int)), default=0)
        if lookback >= history_bars:
            raise ValueError(f"{path}: {factor.label} needs more than {history_bars} bars")
