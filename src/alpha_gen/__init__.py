"""alpha-gen: research and live-signal toolkit for crypto alpha generation.

The package is organised as a pipeline::

    data  ->  factors  ->  signals  ->  evaluation / live

* :mod:`alpha_gen.data` loads market panels (perp prices, funding, open interest, spot).
* :mod:`alpha_gen.factors` turns panels into raw factor scores (higher = more bullish).
* :mod:`alpha_gen.signals` normalises, combines and sizes factors into target weights.
* :mod:`alpha_gen.evaluation` measures predictive power and backtests with realistic costs.
* :mod:`alpha_gen.live` produces signals on a schedule and records a tamper-evident log.
"""

from alpha_gen.data.market import MarketData, MissingDataError
from alpha_gen.factors import Factor, available_factors, get_factor
from alpha_gen.signals.strategy import Strategy

__all__ = [
    "Factor",
    "MarketData",
    "MissingDataError",
    "Strategy",
    "available_factors",
    "get_factor",
]

__version__ = "0.1.0"
