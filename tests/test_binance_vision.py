"""Binance archive loader: golden tests on real files, behaviour tests on a fake archive."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from alpha_gen.data.binance_vision import (
    FUNDING,
    METRICS,
    PERP_KLINES,
    BinanceVisionSource,
    list_perp_symbols,
    parse_funding,
    parse_klines,
    parse_metrics,
    spot_candidates,
    to_binance_id,
)

FIXTURES = Path(__file__).parent / "fixtures" / "binance_vision"
BASE = "https://archive.test/data"
UTC = "UTC"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# ---------------------------------------------------------------- golden tests on real files
def test_parse_real_perp_klines_with_header():
    bars = parse_klines(fixture("BTCUSDT-1h-2024-01.zip"))
    assert len(bars) == 744  # 31 days x 24h
    first = bars.iloc[0]
    assert bars.index[0] == pd.Timestamp("2024-01-01 00:00", tz=UTC)
    assert (first["open"], first["close"]) == (42314.0, 42503.5)
    assert first["quote_volume"] == pytest.approx(359196345.08716)
    assert bars.index[-1] == pd.Timestamp("2024-01-31 23:00", tz=UTC)


def test_parse_real_perp_klines_without_header():
    bars = parse_klines(fixture("BTCUSDT-1h-2021-01.zip"))
    assert bars.index[0] == pd.Timestamp("2021-01-01 00:00", tz=UTC)
    assert bars["close"].iloc[0] == 29015.0


def test_parse_real_spot_klines_microseconds():
    bars = parse_klines(fixture("spot-BTCUSDT-1h-2025-03.zip"))
    assert bars.index[0] == pd.Timestamp("2025-03-01 00:00", tz=UTC)
    assert bars.index[-1] == pd.Timestamp("2025-03-31 23:00", tz=UTC)
    assert bars["close"].iloc[0] == pytest.approx(83857.92)


def test_parse_real_funding():
    rates = parse_funding(fixture("BTCUSDT-fundingRate-2024-01.zip"))
    assert len(rates) == 93  # 3 settlements x 31 days
    assert rates.index[0] == pd.Timestamp("2024-01-01 00:00", tz=UTC)
    assert rates["funding_rate"].iloc[0] == pytest.approx(0.00037409)


def test_parse_real_metrics():
    oi = parse_metrics(fixture("BTCUSDT-metrics-2024-01-01.zip"))
    assert len(oi) == 288  # 5-minute samples
    assert oi.index[0] == pd.Timestamp("2024-01-01 00:00", tz=UTC)
    assert oi["open_interest_value"].iloc[0] == pytest.approx(3131493738.8974)


# ---------------------------------------------------------------- fake archive helpers
def _zip(csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", csv)
    return buf.getvalue()


def kline_zip(start: str, hours: int, price: float = 100.0, *, micro: bool = False) -> bytes:
    unit = 1000 if micro else 1
    t0 = int(pd.Timestamp(start, tz=UTC).value // 1_000_000)
    rows = [
        f"{(t0 + h * 3_600_000) * unit},{price},{price},{price},{price * (1 + h / 1000)},"
        f"1,0,{price * (1 + h / 1000) * 2},1,0,0,0"
        for h in range(hours)
    ]
    return _zip("\n".join(rows) + "\n")


def funding_zip(times: list[str], rate: float) -> bytes:
    rows = [f"{int(pd.Timestamp(t, tz=UTC).value // 1_000_000) + 3},8,{rate}" for t in times]
    return _zip("calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows))


class FakeArchive:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[str] = []

    def __call__(self, url: str) -> bytes | None:
        self.requests.append(url)
        return self.files.get(url)


def source(tmp_path, archive, today="2024-04-15", **kw) -> BinanceVisionSource:
    return BinanceVisionSource(
        tmp_path, fetcher=archive, base_url=BASE, today=pd.Timestamp(today, tz=UTC), **kw
    )


# ---------------------------------------------------------------- planning
def test_plan_uses_monthly_for_old_and_daily_for_recent_months(tmp_path):
    src = source(tmp_path, FakeArchive({}), today="2024-04-05")
    jobs = src.plan(
        PERP_KLINES, "BTCUSDT", pd.Timestamp("2024-02-10"), pd.Timestamp("2024-04-10"), "1h"
    )
    monthly = [j.period for j in jobs if j.freq == "monthly"]
    daily = [j.period for j in jobs if j.freq == "daily"]
    assert monthly == ["2024-02"]  # March ended < 10 days ago -> daily files instead
    assert daily[0] == "2024-03-01" and daily[-1] == "2024-04-04"  # today's file is not out yet


def test_plan_funding_has_no_daily_files(tmp_path):
    src = source(tmp_path, FakeArchive({}), today="2024-04-05")
    jobs = src.plan(FUNDING, "BTCUSDT", pd.Timestamp("2024-02-01"), pd.Timestamp("2024-04-05"), "")
    assert [(j.freq, j.period) for j in jobs] == [("monthly", "2024-02"), ("monthly", "2024-03")]


def test_plan_metrics_is_daily_only(tmp_path):
    src = source(tmp_path, FakeArchive({}))
    jobs = src.plan(METRICS, "BTCUSDT", pd.Timestamp("2024-01-30"), pd.Timestamp("2024-02-02"), "")
    assert [j.period for j in jobs] == ["2024-01-30", "2024-01-31", "2024-02-01"]


# ---------------------------------------------------------------- fetch
def btc_archive() -> FakeArchive:
    k = f"{BASE}/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h"
    f = f"{BASE}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate"
    return FakeArchive(
        {
            f"{k}-2024-01.zip": kline_zip("2024-01-01", 31 * 24),
            f"{k}-2024-02.zip": kline_zip("2024-02-01", 29 * 24, price=200.0),
            f"{f}-2024-01.zip": funding_zip(["2024-01-31 08:00", "2024-01-31 16:00"], 0.0001),
            f"{f}-2024-02.zip": funding_zip(["2024-02-01 00:00"], 0.0003),
        }
    )


def test_fetch_stitches_months_and_floors_funding(tmp_path):
    src = source(tmp_path, btc_archive())
    data = src.fetch(
        ["BTC/USDT:USDT"], pd.Timestamp("2024-01-31 12:00"), pd.Timestamp("2024-02-01 06:00")
    )
    col = "BTC/USDT:USDT"
    assert data.symbols == [col]
    assert data.index[0] == pd.Timestamp("2024-01-31 12:00", tz=UTC)
    assert data.index[-1] == pd.Timestamp("2024-02-01 05:00", tz=UTC)  # end is exclusive
    assert data.close[col].loc["2024-02-01 00:00"] == 200.0
    assert data.volume[col].loc["2024-02-01 00:00"] == 400.0  # quote volume
    rates = data.funding_rate[col].dropna()
    assert rates.to_dict() == {
        pd.Timestamp("2024-01-31 16:00", tz=UTC): 0.0001,
        pd.Timestamp("2024-02-01 00:00", tz=UTC): 0.0003,
    }


def test_fetch_uses_cache_on_second_call(tmp_path):
    archive = btc_archive()
    src = source(tmp_path, archive)
    args = (["BTCUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"))
    first = src.fetch(*args)
    n_requests = len(archive.requests)
    second = source(tmp_path, archive).fetch(*args)
    assert len(archive.requests) == n_requests
    pd.testing.assert_frame_equal(first.close, second.close)


def test_missing_files_are_remembered_only_when_final(tmp_path):
    archive = btc_archive()
    src = source(tmp_path, archive, today="2024-04-15")
    src.fetch(["BTCUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-15"))
    missing = {p.name for p in tmp_path.rglob("*.missing")}
    assert "2024-03.missing" in missing  # month long over: will never appear
    assert "2024-04-10.missing" in missing
    recent = {f"2024-04-{d}.missing" for d in range(11, 15)}
    assert not recent & missing  # the last few days may still be published
    before = len(archive.requests)
    src.fetch(["BTCUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-15"))
    retried = archive.requests[before:]
    assert retried and all("2024-04-1" in url for url in retried)


def test_fetch_recent_month_from_daily_files(tmp_path):
    d = f"{BASE}/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h"
    archive = FakeArchive(
        {
            f"{d}-2024-04-01.zip": kline_zip("2024-04-01", 24),
            f"{d}-2024-04-02.zip": kline_zip("2024-04-02", 24, price=300.0),
        }
    )
    src = source(tmp_path, archive, today="2024-04-03", include_funding=False)
    data = src.fetch(["BTCUSDT"], pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-03"))
    assert len(data) == 48 and data.close["BTCUSDT"].iloc[24] == 300.0


def test_spot_multiplier_mapping_and_sanity_check(tmp_path):
    perp = f"{BASE}/futures/um/monthly/klines"
    spot = f"{BASE}/spot/monthly/klines"
    archive = FakeArchive(
        {
            f"{perp}/1000PEPEUSDT/1h/1000PEPEUSDT-1h-2024-01.zip": kline_zip("2024-01-01", 24, 1.0),
            f"{spot}/PEPEUSDT/1h/PEPEUSDT-1h-2024-01.zip": kline_zip("2024-01-01", 24, 0.001),
            f"{perp}/1000SATSUSDT/1h/1000SATSUSDT-1h-2024-01.zip": kline_zip("2024-01-01", 24, 1.0),
            # SATSUSDT x1000 would be 1000x off; the same-name spot pair is the right one
            f"{spot}/SATSUSDT/1h/SATSUSDT-1h-2024-01.zip": kline_zip("2024-01-01", 24, 1.0),
            f"{spot}/1000SATSUSDT/1h/1000SATSUSDT-1h-2024-01.zip": kline_zip("2024-01-01", 24, 1.0),
        }
    )
    src = source(tmp_path, archive, include_funding=False, include_spot=True)
    data = src.fetch(
        ["1000PEPEUSDT", "1000SATSUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    )
    assert data.spot_close["1000PEPEUSDT"].iloc[0] == pytest.approx(1.0)
    assert data.spot_close["1000SATSUSDT"].iloc[0] == pytest.approx(1.0)


def test_fetch_skips_unknown_symbols_and_raises_if_none(tmp_path):
    src = source(tmp_path, btc_archive())
    data = src.fetch(
        ["BTCUSDT", "NOPEUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    )
    assert data.symbols == ["BTCUSDT"]
    with pytest.raises(ValueError):
        src.fetch(["NOPEUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))


def test_monthly_quote_volume(tmp_path):
    k = f"{BASE}/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d"
    archive = FakeArchive(
        {
            f"{k}-2024-01.zip": _zip(
                "\n".join(
                    f"{int(pd.Timestamp(f'2024-01-{d:02d}', tz=UTC).value // 1_000_000)},"
                    "1,1,1,1,1,0,10,1,0,0,0"
                    for d in range(1, 32)
                )
            )
        }
    )
    vol = source(tmp_path, archive).monthly_quote_volume(
        ["BTCUSDT", "ETHUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")
    )
    assert vol.loc[pd.Timestamp("2024-01-01", tz=UTC), "BTCUSDT"] == 310
    assert vol["ETHUSDT"].isna().all()


# ---------------------------------------------------------------- symbols and listing
def test_symbol_helpers():
    assert to_binance_id("BTC/USDT:USDT") == "BTCUSDT"
    assert to_binance_id("1000PEPE/USDT:USDT") == "1000PEPEUSDT"
    assert spot_candidates("BTCUSDT") == [("BTCUSDT", 1.0)]
    assert spot_candidates("1000PEPEUSDT") == [("PEPEUSDT", 1000.0), ("1000PEPEUSDT", 1.0)]
    assert spot_candidates("1000000MOGUSDT")[0] == ("MOGUSDT", 1_000_000.0)
    assert spot_candidates("1INCHUSDT") == [("1INCHUSDT", 1.0)]


def test_non_ascii_symbols_are_url_encoded():
    url = PERP_KLINES.url(BASE, "monthly", "我踏马来了USDT", "2025-10", "1d")
    assert url.isascii()
    assert url.startswith(f"{BASE}/futures/um/monthly/klines/%E6%88%91")
    assert url.endswith("USDT-1d-2025-10.zip")


def _listing(prefixes: list[str], truncated: bool, next_marker: str = "") -> bytes:
    items = "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    marker = f"<NextMarker>{next_marker}</NextMarker>" if next_marker else ""
    return (
        '<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{marker}{items}</ListBucketResult>"
    ).encode()


def test_list_perp_symbols_paginates_and_filters():
    root = "data/futures/um/monthly/klines/"
    pages = {
        "": _listing([root + "BTCUSDT/", root + "BTCUSDT_240329/"], True, root + "BTCUSDT_240329/"),
        root + "BTCUSDT_240329/": _listing([root + "ETHBUSD/", root + "FTTUSDT/"], False),
    }

    def fetch(url: str) -> bytes:
        marker = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))["marker"]
        return pages[marker.replace("%2F", "/")]

    assert list_perp_symbols(fetcher=fetch, list_url="https://list.test") == ["BTCUSDT", "FTTUSDT"]


# ---------------------------------------------------------------- live archive (opt-in)
@pytest.mark.network
def test_live_archive_smoke(tmp_path):
    src = BinanceVisionSource(tmp_path, include_open_interest=True)
    data = src.fetch(["BTCUSDT"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))
    assert len(data) == 48
    assert data.funding_rate["BTCUSDT"].notna().sum() == 6
    assert data.open_interest["BTCUSDT"].notna().all()
    assert "BTCUSDT" in list_perp_symbols()
