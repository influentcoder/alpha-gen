"""Historical data from Binance's public archive, https://data.binance.vision.

The archive serves free, rate-limit-free zipped CSVs of USDT-margined perpetual klines, funding
rates and open-interest "metrics", plus spot klines. It keeps files for **delisted** symbols, which
makes survivorship-bias-free research possible (see :func:`list_perp_symbols`).

Layout handled here:

* Klines (perp and spot) and funding: one file per symbol per **month**. Recent months that have
  no monthly file yet are loaded from **daily** files (klines only: funding has no daily files).
* Metrics (open interest, 5-minute samples): one file per symbol per **day**.
* CSVs may or may not have a header row; spot timestamps switched from milliseconds to
  microseconds in 2025. Both are detected automatically.

Every downloaded file is parsed once and cached as Parquet. Files that do not exist (symbol not
listed yet / delisted) are remembered with a ``.missing`` marker once the period is old enough
that the file can no longer appear.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_gen.data.market import MarketData
from alpha_gen.timeframes import timeframe_to_freq

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data"
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "funding_rate"]
METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

Fetcher = Callable[[str], bytes | None]
"""Downloads a URL; returns ``None`` when the file does not exist (HTTP 404)."""


class ArchiveError(RuntimeError):
    """Raised when the archive cannot be reached after retries."""


# ---------------------------------------------------------------------------------------- HTTP
def http_fetch(url: str, *, timeout: float = 30.0, retries: int = 4) -> bytes | None:
    """GET ``url`` with exponential backoff on network errors, 429 and 5xx. 404 -> ``None``."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code != 429 and exc.code < 500:
                raise ArchiveError(f"HTTP {exc.code} for {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(2**attempt)
    raise ArchiveError(f"Failed to fetch {url} after {retries} attempts") from last


# ---------------------------------------------------------------------------------------- symbols
def to_binance_id(symbol: str) -> str:
    """``"BTC/USDT:USDT"`` (ccxt unified) or ``"BTCUSDT"`` -> ``"BTCUSDT"``."""
    return symbol.split(":", 1)[0].replace("/", "").upper()


_MULTIPLIER_PREFIXES = (("1000000", 1_000_000.0), ("1000", 1_000.0))


def spot_candidates(perp_id: str) -> list[tuple[str, float]]:
    """Spot symbols (and price multipliers) that may match a perp.

    ``1000PEPEUSDT`` perp quotes 1000 PEPE, so its spot counterpart is ``PEPEUSDT`` x 1000; but
    ``1000SATSUSDT`` exists on spot under the same name. Candidates are tried in order.
    """
    out = []
    for prefix, multiplier in _MULTIPLIER_PREFIXES:
        if perp_id.startswith(prefix) and len(perp_id) > len(prefix) + 4:
            out.append((perp_id[len(prefix) :], multiplier))
            break
    out.append((perp_id, 1.0))
    return out


def list_prefixes(
    prefix: str, fetcher: Fetcher = http_fetch, list_url: str = LIST_URL
) -> list[str]:
    """List "sub-directories" of an archive prefix via the public S3 bucket listing."""
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out: list[str] = []
    marker = ""
    while True:
        query = urllib.parse.urlencode({"delimiter": "/", "prefix": prefix, "marker": marker})
        body = fetcher(f"{list_url}?{query}")
        if body is None:
            raise ArchiveError(f"Listing not found for prefix {prefix!r}")
        root = ET.fromstring(body)
        page = [el.text or "" for el in root.findall("s3:CommonPrefixes/s3:Prefix", ns)]
        out.extend(page)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns)) == "true"
        next_marker = root.findtext("s3:NextMarker", default="", namespaces=ns) or (
            page[-1] if page else ""
        )
        if not truncated or not next_marker or next_marker == marker:
            return out
        marker = next_marker


def list_perp_symbols(
    quote: str = "USDT", fetcher: Fetcher = http_fetch, list_url: str = LIST_URL
) -> list[str]:
    """Every USDT-margined perp the archive has monthly klines for, **including delisted ones**.

    Dated delivery contracts (``BTCUSDT_240329``) are excluded.
    """
    prefix = "data/futures/um/monthly/klines/"
    names = [p[len(prefix) :].strip("/") for p in list_prefixes(prefix, fetcher, list_url)]
    return sorted(n for n in names if n.endswith(quote) and "_" not in n)


# ---------------------------------------------------------------------------------------- parsing
def _to_utc(values: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        numeric = values.astype("int64")
        unit = "us" if len(numeric) and numeric.max() > 10**14 else "ms"
        return pd.DatetimeIndex(pd.to_datetime(numeric, unit=unit, utc=True))
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _read_zipped_csv(blob: bytes, columns: Sequence[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        raw = archive.read(name)
    has_header = not raw[:1].isdigit()
    frame = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        skiprows=1 if has_header else 0,
        names=list(columns),
        usecols=range(len(columns)),
    )
    return frame


def parse_klines(blob: bytes) -> pd.DataFrame:
    """Kline CSV -> float columns indexed by bar open time (UTC)."""
    frame = _read_zipped_csv(blob, KLINE_COLUMNS)
    index = _to_utc(frame["open_time"])
    keep = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
    keep += ["taker_buy_volume", "taker_buy_quote_volume"]
    out = frame[keep].astype(float).set_axis(index, axis=0)
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_funding(blob: bytes) -> pd.DataFrame:
    """Funding CSV -> ``funding_rate`` indexed by settlement time (UTC)."""
    frame = _read_zipped_csv(blob, FUNDING_COLUMNS)
    out = frame[["funding_rate"]].astype(float).set_axis(_to_utc(frame["calc_time"]), axis=0)
    return out.sort_index()


def parse_metrics(blob: bytes) -> pd.DataFrame:
    """Metrics CSV -> open interest (contracts and quote value) indexed by sample time (UTC)."""
    frame = _read_zipped_csv(blob, METRICS_COLUMNS)
    cols = {"sum_open_interest": "open_interest", "sum_open_interest_value": "open_interest_value"}
    out = frame[list(cols)].rename(columns=cols)
    out = out.apply(pd.to_numeric, errors="coerce").set_axis(_to_utc(frame["create_time"]), axis=0)
    return out.sort_index()


# ---------------------------------------------------------------------------------------- datasets
@dataclass(frozen=True)
class Dataset:
    """Where one kind of archive file lives and how to parse it."""

    key: str
    path: str  # template with {freq} {symbol} {interval} {period}
    parse: Callable[[bytes], pd.DataFrame]
    monthly: bool = True
    daily: bool = True

    def url(self, base: str, freq: str, symbol: str, period: str, interval: str) -> str:
        rel = self.path.format(freq=freq, symbol=symbol, interval=interval, period=period)
        return f"{base}/{urllib.parse.quote(rel)}"  # some symbols are not ASCII


PERP_KLINES = Dataset(
    "um_klines",
    "futures/um/{freq}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip",
    parse_klines,
)
SPOT_KLINES = Dataset(
    "spot_klines",
    "spot/{freq}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip",
    parse_klines,
)
FUNDING = Dataset(
    "um_funding",
    "futures/um/{freq}/fundingRate/{symbol}/{symbol}-fundingRate-{period}.zip",
    parse_funding,
    daily=False,
)
METRICS = Dataset(
    "um_metrics",
    "futures/um/{freq}/metrics/{symbol}/{symbol}-metrics-{period}.zip",
    parse_metrics,
    monthly=False,
)


@dataclass(frozen=True)
class Job:
    dataset: Dataset
    symbol: str
    freq: str  # "monthly" | "daily"
    period: str  # "2024-01" | "2024-01-05"
    interval: str  # kline interval, "" for other datasets
    end: pd.Timestamp  # first instant after the period


# ---------------------------------------------------------------------------------------- source
class BinanceVisionSource:
    """:class:`~alpha_gen.data.base.DataSource` backed by the Binance public archive.

    Args:
        cache_dir: Where parsed files are cached (Parquet).
        include_funding / include_spot / include_open_interest: Which optional panels to load.
            Open interest comes from daily files (one request per symbol-day), so it is off by
            default; enable it for shorter windows.
        max_workers: Parallel downloads.
        recent_days: Months ending less than this many days ago are loaded from daily files,
            because Binance publishes the monthly file only after the month closes.
        fetcher / base_url / today: Injection points for tests.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/binance-vision",
        *,
        include_funding: bool = True,
        include_spot: bool = False,
        include_open_interest: bool = False,
        max_workers: int = 32,
        recent_days: int = 10,
        fetcher: Fetcher = http_fetch,
        base_url: str = BASE_URL,
        today: pd.Timestamp | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.include_funding = include_funding
        self.include_spot = include_spot
        self.include_open_interest = include_open_interest
        self.max_workers = max_workers
        self.recent_days = recent_days
        self.fetcher = fetcher
        self.base_url = base_url
        self._today = today

    @property
    def cache_key(self) -> str:
        flags = f"f{int(self.include_funding)}s{int(self.include_spot)}"
        return f"binance-vision_{flags}oi{int(self.include_open_interest)}"

    @property
    def today(self) -> pd.Timestamp:
        today = self._today if self._today is not None else pd.Timestamp.now(tz="UTC")
        today = today.tz_localize("UTC") if today.tz is None else today.tz_convert("UTC")
        return today.floor("D")

    # ------------------------------------------------------------------ planning
    def plan(
        self, dataset: Dataset, symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str
    ) -> list[Job]:
        """Files needed to cover ``[start, end)`` for one symbol."""
        start, end = _utc(start), _utc(end)
        last_day = min(end, self.today)  # today's daily file never exists yet
        jobs: list[Job] = []
        if end <= start:
            return jobs
        months = pd.period_range(
            start.tz_localize(None), (end - pd.Timedelta(1)).tz_localize(None), freq="M"
        )
        for month in months:
            month_start = month.start_time.tz_localize("UTC")
            month_end = (month + 1).start_time.tz_localize("UTC")
            recent = month_end > self.today - pd.Timedelta(days=self.recent_days)
            if dataset.monthly and not (recent and dataset.daily):
                if not (recent and month_end > self.today):  # current month: no monthly file
                    jobs.append(Job(dataset, symbol, "monthly", str(month), interval, month_end))
                continue
            if not dataset.daily:
                continue
            first = max(month_start, start.floor("D"))
            for day in pd.date_range(first, min(month_end, last_day), freq="D", inclusive="left"):
                jobs.append(
                    Job(
                        dataset,
                        symbol,
                        "daily",
                        day.strftime("%Y-%m-%d"),
                        interval,
                        day + pd.Timedelta(days=1),
                    )
                )
        return jobs

    # ------------------------------------------------------------------ loading
    def _paths(self, job: Job) -> tuple[Path, Path]:
        folder = self.cache_dir / (
            f"{job.dataset.key}_{job.interval}" if job.interval else job.dataset.key
        )
        stem = folder / job.symbol / job.period
        return stem.with_suffix(".parquet"), stem.with_suffix(".missing")

    def _is_final(self, job: Job) -> bool:
        """A missing file is permanent once its period ended a few days ago."""
        return job.end < self.today - pd.Timedelta(days=3)

    def _load(self, job: Job) -> pd.DataFrame | None:
        data_path, missing_path = self._paths(job)
        if data_path.exists():
            return pd.read_parquet(data_path)
        if missing_path.exists():
            return None
        url = job.dataset.url(self.base_url, job.freq, job.symbol, job.period, job.interval)
        blob = self.fetcher(url)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if blob is None:
            if self._is_final(job):
                missing_path.touch()
            return None
        frame = job.dataset.parse(blob)
        tmp = data_path.with_suffix(".tmp")
        frame.to_parquet(tmp)
        tmp.replace(data_path)
        return frame

    def load_jobs(self, jobs: Iterable[Job]) -> dict[Job, pd.DataFrame | None]:
        """Load (download or read from cache) many files in parallel."""
        unique = list(dict.fromkeys(jobs))
        with ThreadPoolExecutor(self.max_workers) as pool:
            return dict(zip(unique, pool.map(self._load, unique), strict=True))

    @staticmethod
    def _concat(frames: Iterable[pd.DataFrame | None]) -> pd.DataFrame | None:
        present = [f for f in frames if f is not None and len(f)]
        if not present:
            return None
        out = pd.concat(present).sort_index()
        return out[~out.index.duplicated(keep="last")]

    def _collect(
        self, loaded: dict[Job, pd.DataFrame | None], dataset: Dataset, symbol: str
    ) -> pd.DataFrame | None:
        return self._concat(
            f for j, f in loaded.items() if j.dataset is dataset and j.symbol == symbol
        )

    # ------------------------------------------------------------------ public API
    def klines(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1h",
        *,
        field: str = "close",
        spot: bool = False,
    ) -> pd.DataFrame:
        """One kline field (e.g. ``"quote_volume"``) for many symbols as a wide panel."""
        if interval not in INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}; expected one of {INTERVALS}")
        dataset = SPOT_KLINES if spot else PERP_KLINES
        ids = {s: to_binance_id(s) for s in symbols}
        jobs = [j for b in ids.values() for j in self.plan(dataset, b, start, end, interval)]
        loaded = self.load_jobs(jobs)
        columns = {}
        for symbol, binance_id in ids.items():
            frame = self._collect(loaded, dataset, binance_id)
            if frame is not None:
                columns[symbol] = frame[field]
        panel = pd.DataFrame(columns, columns=list(symbols))
        return panel.loc[(panel.index >= _utc(start)) & (panel.index < _utc(end))]

    def monthly_quote_volume(
        self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Total quote (USDT) volume per calendar month, from daily klines. Index = month start."""
        daily = self.klines(symbols, start, end, "1d", field="quote_volume")
        return daily.resample("MS").sum(min_count=1)

    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str = "1h",
    ) -> MarketData:
        """Load bars with open time in ``[start, end)``. ``volume`` is quote (USDT) volume."""
        if timeframe not in INTERVALS:
            raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of {INTERVALS}")
        start, end = _utc(start), _utc(end)
        freq = timeframe_to_freq(timeframe)
        ids = {s: to_binance_id(s) for s in symbols}

        jobs: list[Job] = []
        for binance_id in ids.values():
            jobs += self.plan(PERP_KLINES, binance_id, start, end, timeframe)
            if self.include_funding:
                jobs += self.plan(FUNDING, binance_id, start, end, "")
            if self.include_open_interest:
                jobs += self.plan(METRICS, binance_id, start, end, "")
            if self.include_spot:
                spot_id = spot_candidates(binance_id)[0][0]
                jobs += self.plan(SPOT_KLINES, spot_id, start, end, timeframe)
        loaded = self.load_jobs(jobs)

        close: dict[str, pd.Series] = {}
        volume: dict[str, pd.Series] = {}
        funding: dict[str, pd.Series] = {}
        oi: dict[str, pd.Series] = {}
        spot: dict[str, pd.Series] = {}
        for symbol, binance_id in ids.items():
            bars = self._collect(loaded, PERP_KLINES, binance_id)
            if bars is None:
                log.warning("No klines for %s in [%s, %s)", symbol, start, end)
                continue
            close[symbol], volume[symbol] = bars["close"], bars["quote_volume"]
            if self.include_funding:
                rates = self._collect(loaded, FUNDING, binance_id)
                if rates is not None:
                    funding[symbol] = _floor_last(rates["funding_rate"], freq)
            if self.include_open_interest:
                metrics = self._collect(loaded, METRICS, binance_id)
                if metrics is not None:
                    oi[symbol] = _floor_last(metrics["open_interest_value"], freq)
            if self.include_spot:
                series = self._spot_close(binance_id, bars["close"], loaded, start, end, timeframe)
                if series is not None:
                    spot[symbol] = series

        if not close:
            raise ValueError(f"No klines found for any of {list(symbols)} in [{start}, {end})")

        def window(d: dict[str, pd.Series]) -> pd.DataFrame:
            frame = pd.DataFrame(d)
            return frame.loc[(frame.index >= start) & (frame.index < end)]

        def panel(d: dict[str, pd.Series]) -> pd.DataFrame | None:
            return window(d) if d else None

        return MarketData(
            close=window(close),
            volume=panel(volume),
            funding_rate=panel(funding),
            open_interest=panel(oi),
            spot_close=panel(spot),
            timeframe=timeframe,
        )

    def _spot_close(
        self,
        perp_id: str,
        perp_close: pd.Series,
        loaded: dict[Job, pd.DataFrame | None],
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str,
    ) -> pd.Series | None:
        for i, (spot_id, multiplier) in enumerate(spot_candidates(perp_id)):
            if i > 0:  # fallback candidates were not pre-planned
                loaded = {
                    **loaded,
                    **self.load_jobs(self.plan(SPOT_KLINES, spot_id, start, end, timeframe)),
                }
            frame = self._collect(loaded, SPOT_KLINES, spot_id)
            if frame is None:
                continue
            series = frame["close"] * multiplier
            ratio = (perp_close / series).dropna()
            if len(ratio) and float(np.abs(ratio.median() - 1.0)) < 0.2:
                return series
            log.warning("Spot %s does not match perp %s in price; skipped", spot_id, perp_id)
        return None


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


def _floor_last(series: pd.Series, freq: str) -> pd.Series:
    """Snap timestamps to the bar they fall in, keeping the last value per bar."""
    floored = series.set_axis(pd.DatetimeIndex(series.index).floor(freq), axis=0)
    return floored.groupby(level=0).last()
