"""Point-in-time universe selection.

Pick symbols by *past* liquidity so a backtest only trades what a live strategy could have chosen
at the time. Combined with an archive that keeps delisted symbols, this avoids survivorship bias.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# Index/stable/basket perps that are not single-asset bets.
DEFAULT_EXCLUDE = frozenset(
    {
        "USDCUSDT",
        "TUSDUSDT",
        "BUSDUSDT",
        "FDUSDUSDT",
        "USDPUSDT",
        "BTCDOMUSDT",
        "DEFIUSDT",
        "FOOTBALLUSDT",
        "BLUEBIRDUSDT",
    }
)


def top_n_membership(
    volume: pd.DataFrame,
    top_n: int,
    *,
    lag: int = 1,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> pd.DataFrame:
    """Boolean membership: the ``top_n`` symbols by volume of the period ``lag`` rows earlier.

    ``volume`` is typically monthly quote volume (one row per period start). The row for period
    ``p`` uses only volume from period ``p - lag``, so membership is known when ``p`` begins.
    """
    if top_n < 1 or lag < 1:
        raise ValueError("top_n and lag must be >= 1")
    excluded = set(exclude)
    eligible = volume[[c for c in volume.columns if c not in excluded]]
    ranks = eligible.rank(axis=1, ascending=False, method="first")
    member = (ranks <= top_n).shift(lag, fill_value=False).astype(bool)
    return member.reindex(columns=volume.columns, fill_value=False)


def candidates(membership: pd.DataFrame) -> list[str]:
    """Symbols that are members at least once (the set worth downloading in detail)."""
    return [str(c) for c in membership.columns[membership.any(axis=0)]]


def expand_membership(membership: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill period membership onto a bar index (each period applies until the next)."""
    member = membership.copy()
    if member.index.tz is None:
        member.index = member.index.tz_localize("UTC")
    combined = member.index.union(index)
    expanded = member.reindex(combined).ffill().reindex(index)
    return expanded.fillna(False).astype(bool)


def apply_universe(scores: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Blank out scores of symbols that are not in the universe at that bar."""
    assert isinstance(scores.index, pd.DatetimeIndex)
    member = expand_membership(membership, scores.index)
    member = member.reindex(columns=scores.columns, fill_value=False)
    return scores.where(member)
