import pytest

from coinband import binance_data
from coinband.binance_data import fetch_klines

HOUR = 3_600_000


def make_exchange(n_bars, now_ms):
    """A fake /api/v3/klines with `n_bars` hourly candles, the last one still open at `now_ms`."""
    first_open = now_ms - (n_bars - 1) * HOUR - HOUR // 2
    all_rows = [
        [first_open + i * HOUR, "1", "2", "0.5", str(100 + i), "10", first_open + (i + 1) * HOUR - 1]
        for i in range(n_bars)
    ]
    calls = []

    def fetch(path, params, timeout=20.0):
        assert path == "/api/v3/klines"
        calls.append(dict(params))
        rows = all_rows
        if "endTime" in params:
            rows = [r for r in rows if r[0] <= params["endTime"]]
        return rows[-params["limit"] :]

    return fetch, calls, all_rows


def test_pages_backward_and_drops_open_candle():
    now = 10_000 * HOUR
    fetch, calls, rows = make_exchange(2500, now)
    k = fetch_klines("btcusdt", "1h", 2200, fetch=fetch, now_ms=now)
    assert len(k) == 2200
    assert k.symbol == "BTCUSDT"
    # The newest (open) candle is gone; the result ends on the last closed one.
    assert k.open_time[-1] == rows[-2][0]
    assert k.close[-1] == float(rows[-2][4])
    assert all(b - a == HOUR for a, b in zip(k.open_time, k.open_time[1:]))
    assert len(calls) == 3
    assert all(c["limit"] <= 1000 for c in calls)
    assert calls[1]["endTime"] < calls[0].get("endTime", float("inf"))


def test_young_pair_returns_what_exists():
    now = 10_000 * HOUR
    fetch, calls, rows = make_exchange(300, now)
    k = fetch_klines("NEWUSDT", "1h", 5000, fetch=fetch, now_ms=now)
    assert len(k) == 299
    assert len(calls) == 1


def test_unknown_interval():
    with pytest.raises(ValueError):
        fetch_klines("BTCUSDT", "7m", 10, fetch=lambda *a, **k: [])


def test_symbols_filtered_and_cached(monkeypatch):
    monkeypatch.setattr(binance_data, "_symbols_cache", None)
    calls = []

    def fetch(path, params, timeout=20.0):
        calls.append(path)
        return {
            "symbols": [
                {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT", "status": "TRADING"},
                {"symbol": "OLDBTC", "baseAsset": "OLD", "quoteAsset": "BTC", "status": "BREAK"},
                {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING"},
            ]
        }

    first = binance_data.list_symbols(fetch=fetch)
    assert [s["symbol"] for s in first] == ["BTCUSDT", "ETHUSDT"]
    assert binance_data.list_symbols(fetch=fetch) is first
    assert len(calls) == 1
