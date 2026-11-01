"""Parquet persistence for :class:`MarketData` and a caching wrapper around any source."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from alpha_gen.data.base import DataSource
from alpha_gen.data.market import PANEL_FIELDS, MarketData

_META = "meta.json"


def save_market_data(data: MarketData, directory: str | Path) -> Path:
    """Write each loaded panel to ``<directory>/<panel>.parquet`` plus a small metadata file."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    panels = [name for name in PANEL_FIELDS if data.has(name)]
    for name in panels:
        data.require(name).to_parquet(path / f"{name}.parquet")
    (path / _META).write_text(json.dumps({"timeframe": data.timeframe, "panels": panels}))
    return path


def load_market_data(directory: str | Path) -> MarketData:
    """Inverse of :func:`save_market_data`."""
    path = Path(directory)
    meta = json.loads((path / _META).read_text())
    panels = {name: pd.read_parquet(path / f"{name}.parquet") for name in meta["panels"]}
    return MarketData(**panels, timeframe=meta["timeframe"])


class CachedSource:
    """Wrap a :class:`DataSource` and memoise each distinct request on disk.

    The cache key covers the source, symbols, range and timeframe, so changing any of them
    triggers a fresh download. Delete the directory to invalidate.
    """

    def __init__(self, source: DataSource, root: str | Path = ".cache/alpha-gen") -> None:
        self.source = source
        self.root = Path(root)

    @property
    def cache_key(self) -> str:
        return self.source.cache_key

    def _key(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, tf: str) -> str:
        payload = json.dumps(
            [self.source.cache_key, list(symbols), str(start), str(end), tf], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str = "1h",
    ) -> MarketData:
        directory = self.root / self._key(symbols, start, end, timeframe)
        if (directory / _META).exists():
            return load_market_data(directory)
        data = self.source.fetch(symbols, start, end, timeframe)
        save_market_data(data, directory)
        return data
