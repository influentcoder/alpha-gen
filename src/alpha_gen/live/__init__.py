"""Live signal production, paper-trading P&L and tamper-evident publishing."""

from alpha_gen.live.config import LiveConfig, load_config
from alpha_gen.live.runner import RunResult, SignalRunner, StaleDataError
from alpha_gen.live.signal_log import GENESIS_HASH, LogIntegrityError, SignalLog, SignalRecord
from alpha_gen.live.track import realized_returns, track_record, track_summary

__all__ = [
    "GENESIS_HASH",
    "LiveConfig",
    "LogIntegrityError",
    "RunResult",
    "SignalLog",
    "SignalRecord",
    "SignalRunner",
    "StaleDataError",
    "load_config",
    "realized_returns",
    "track_record",
    "track_summary",
]
