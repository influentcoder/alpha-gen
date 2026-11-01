"""Helpers for converting exchange timeframe strings (``"1h"``, ``"1d"``) to pandas/time units."""

from __future__ import annotations

import pandas as pd

_UNIT_SECONDS = {"m": 60, "h": 3_600, "d": 86_400, "w": 604_800}
_UNIT_PANDAS = {"m": "min", "h": "h", "d": "D", "w": "W"}

SECONDS_PER_YEAR = 365 * 86_400  # crypto trades 24/7


def _parse(timeframe: str) -> tuple[int, str]:
    if len(timeframe) < 2 or timeframe[-1] not in _UNIT_SECONDS or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected e.g. '5m', '1h', '1d'")
    return int(timeframe[:-1]), timeframe[-1]


def timeframe_seconds(timeframe: str) -> int:
    """Length of one bar in seconds."""
    count, unit = _parse(timeframe)
    return count * _UNIT_SECONDS[unit]


def timeframe_to_offset(timeframe: str) -> pd.Timedelta:
    """Length of one bar as a :class:`pandas.Timedelta`."""
    return pd.Timedelta(seconds=timeframe_seconds(timeframe))


def timeframe_to_freq(timeframe: str) -> str:
    """Pandas frequency alias for a timeframe (``"1h"`` -> ``"1h"``, ``"1d"`` -> ``"1D"``)."""
    count, unit = _parse(timeframe)
    return f"{count}{_UNIT_PANDAS[unit]}"


def periods_per_year(timeframe: str) -> float:
    """Number of bars per year, used to annualise returns and volatility."""
    return SECONDS_PER_YEAR / timeframe_seconds(timeframe)
