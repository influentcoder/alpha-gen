"""Factor library. Importing this package registers every built-in factor."""

from alpha_gen.factors import carry, positioning, price  # noqa: F401  (registration side effect)
from alpha_gen.factors.base import Factor, available_factors, get_factor, register
from alpha_gen.factors.carry import BasisCarry, FundingCarry
from alpha_gen.factors.positioning import OpenInterestTrend
from alpha_gen.factors.price import LowVolatility, Momentum, ShortTermReversal

__all__ = [
    "BasisCarry",
    "Factor",
    "FundingCarry",
    "LowVolatility",
    "Momentum",
    "OpenInterestTrend",
    "ShortTermReversal",
    "available_factors",
    "get_factor",
    "register",
]
