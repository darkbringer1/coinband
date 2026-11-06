from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from coinband import cache, yahoo_data
from coinband.binance_data import Klines
from coinband.sessions import future_times

NY = ZoneInfo("America/New_York")
HOUR = 3_600_000


def ny(*args) -> int:
    return int(datetime(*args, tzinfo=NY).timestamp() * 1000)


def chart(ts, closes, session_end, volume=None):
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "exchangeTimezoneName": "America/New_York",
                        "regularMarketPrice": 101.5,
                        "regularMarketTime": ts[-1],
                        "longName": "Test Corp",
                        "currentTradingPeriod": {"regular": {"start": ts[0], "end": session_end}},
                    },
                    "timestamp": ts,
                    "indicators": {
                        "quote": [
                            {
                                "open": closes,
                                "high": closes,
                                "low": closes,
                                "close": closes,
                                "volume": volume or [1000] * len(ts),
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def test_yahoo_drops_nulls_and_the_forming_bar():
    t0 = ny(2026, 11, 6, 9, 30) // 1000
    # 9:30 (null), 10:30, 11:30 bars, the 12:30 bar forming, and a 12:35 price snapshot
    ts = [t0, t0 + 3600, t0 + 7200, t0 + 3 * 3600, t0 + 3 * 3600 + 300]
    closes = [None, 100.0, 102.0, 103.0, 103.5]
    now = (t0 + 3 * 3600 + 320) * 1000
    k = yahoo_data.fetch_klines("test", "1h", 100, fetch=lambda s, p: chart(ts, closes, t0 + 6.5 * 3600), now_ms=now)
    assert k.close == [100.0, 102.0]
    assert k.info["continuous"] is False and k.info["timezone"] == "America/New_York"
    assert k.info["price"] == 101.5 and k.info["name"] == "Test Corp"


def test_yahoo_keeps_the_last_bar_once_the_session_ended():
    t0 = ny(2026, 11, 6, 9, 30) // 1000
    ts = [t0, t0 + 3600, t0 + 6 * 3600]  # 15:30 bar, session ends 16:00
    end = t0 + int(6.5 * 3600)
    k = yahoo_data.fetch_klines("T", "1h", 100, fetch=lambda s, p: chart(ts, [1.0, 2.0, 3.0], end), now_ms=(end + 60) * 1000)
    assert k.close == [1.0, 2.0, 3.0]


def test_yahoo_rejects_unknown_interval():
    with pytest.raises(ValueError):
        yahoo_data.fetch_klines("AAPL", "4h", 10, fetch=lambda s, p: {})


def test_future_times_skip_nights_and_weekends():
    # hourly bars 9:30..15:30 on Thu and Fri; the next bars are Monday's
    days = [(2026, 11, 5), (2026, 11, 6)]
    times = [ny(*d, h, 30) for d in days for h in range(9, 16)]
    out = future_times(times, HOUR, 9, "America/New_York", continuous=False)
    got = [datetime.fromtimestamp(t / 1000, NY).strftime("%a %H:%M") for t in out]
    assert got[:7] == [f"Mon {h:02d}:30" for h in range(9, 16)]
    assert got[7:] == ["Tue 09:30", "Tue 10:30"]


def test_future_times_continue_the_day():
    times = [ny(2026, 11, 5, h, 30) for h in range(9, 16)] + [ny(2026, 11, 6, h, 30) for h in range(9, 12)]
    out = future_times(times, HOUR, 2, "America/New_York", continuous=False)
    assert [datetime.fromtimestamp(t / 1000, NY).hour for t in out] == [12, 13]


def test_future_times_daily_and_continuous():
    fri = ny(2026, 11, 6, 9, 30)
    out = future_times([fri - 86_400_000, fri], 86_400_000, 2, "America/New_York", continuous=False)
    assert [datetime.fromtimestamp(t / 1000, NY).strftime("%a") for t in out] == ["Mon", "Tue"]
    assert future_times([0, HOUR], HOUR, 3, "UTC", continuous=True) == [2 * HOUR, 3 * HOUR, 4 * HOUR]


def make_klines(times):
    return Klines("X", "1h", list(times), [1.0] * len(times), [1.0] * len(times), [1.0] * len(times),
                  [float(t) for t in times], [1.0] * len(times), {"continuous": True})


def test_cache_serves_fresh_data_and_fetches_only_the_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("COINBAND_CACHE", str(tmp_path))
    calls = []
    series = {"now": 100}

    def fetch(symbol, interval, n, progress=None):
        calls.append(n)
        end = series["now"]
        return make_klines(range(max(0, end - n) * HOUR, end * HOUR, HOUR))

    k = cache.get_klines("binance", "X", "1h", 50, fetch, HOUR, now=100 * 3600)
    assert len(k) == 50 and calls == [50]
    k = cache.get_klines("binance", "X", "1h", 50, fetch, HOUR, now=100 * 3600 + 10)
    assert calls == [50]  # fresh: no fetch
    series["now"] = 103
    k = cache.get_klines("binance", "X", "1h", 50, fetch, HOUR, now=103 * 3600 + 120)
    assert calls[-1] == 5 + cache.TAIL_PAD  # only the missing bars (and a small overlap)
    assert k.open_time[-1] == 102 * HOUR and len(k) == 50
    assert k.close[-1] == float(102 * HOUR)
    # asking for more history than cached fetches the full range again
    cache.get_klines("binance", "X", "1h", 80, fetch, HOUR, now=103 * 3600 + 200)
    assert calls[-1] == 80
