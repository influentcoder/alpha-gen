"""Append-only, hash-chained JSONL log of published signals.

Each record commits to the previous record's hash, so editing or deleting any past signal
breaks verification of everything after it. That makes the log a credible public track record,
provided the latest ``hash`` is periodically anchored somewhere you cannot rewrite (a public git
commit, a social post, OpenTimestamps, a blockchain transaction).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

GENESIS_HASH = "0" * 64
WEIGHT_DECIMALS = 8


class LogIntegrityError(ValueError):
    """Raised when a signal log fails hash-chain verification."""


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _iso(ts: Any) -> str:
    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
    return stamp.isoformat()


@dataclass(frozen=True)
class SignalRecord:
    timestamp: str
    strategy: str
    weights: dict[str, float]
    published_at: str
    prev_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    hash: str = ""

    def body(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("hash")
        return data

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self.body()).encode()).hexdigest()

    def to_json(self) -> str:
        return _canonical(asdict(self))


class SignalLog:
    """File-backed signal log. One JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[SignalRecord]:
        if not self.path.exists():
            return []
        with self.path.open() as fh:
            return [SignalRecord(**json.loads(line)) for line in fh if line.strip()]

    def last(self) -> SignalRecord | None:
        records = self.records()
        return records[-1] if records else None

    def append(
        self,
        timestamp: Any,
        strategy: str,
        weights: Mapping[str, float],
        metadata: Mapping[str, Any] | None = None,
        *,
        published_at: Any = None,
    ) -> SignalRecord:
        """Append a signal. Timestamps must strictly increase per strategy."""
        ts = _iso(timestamp)
        records = self.records()
        previous = [r for r in records if r.strategy == strategy]
        if previous and pd.Timestamp(previous[-1].timestamp) >= pd.Timestamp(ts):
            last_ts = previous[-1].timestamp
            raise ValueError(
                f"Signal for {strategy!r} at {ts} is not after the last one ({last_ts})"
            )
        record = SignalRecord(
            timestamp=ts,
            strategy=strategy,
            weights={k: round(float(v), WEIGHT_DECIMALS) for k, v in sorted(weights.items())},
            published_at=_iso(published_at or pd.Timestamp.now(tz="UTC")),
            prev_hash=records[-1].hash if records else GENESIS_HASH,
            metadata=dict(metadata or {}),
        )
        record = SignalRecord(**{**asdict(record), "hash": record.compute_hash()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(record.to_json() + "\n")
        return record

    def verify(self) -> int:
        """Check the chain; return the record count or raise :class:`LogIntegrityError`."""
        prev = GENESIS_HASH
        records = self.records()
        for i, record in enumerate(records):
            if record.prev_hash != prev:
                raise LogIntegrityError(f"Record {i} does not link to the previous record")
            if record.compute_hash() != record.hash:
                raise LogIntegrityError(f"Record {i} content does not match its hash")
            prev = record.hash
        return len(records)
