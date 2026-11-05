"""Binance spot market data: the symbol list and closed klines.

Uses the market-data-only mirror `data-api.binance.vision` (no key, reachable where `api.binance.com`
is geo-blocked) and falls back to `api.binance.com`.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
PAGE_LIMIT = 1000
USER_AGENT = "coinband/0.1 (+https://github.com/darkbringer1/coinband)"

# Binance interval code -> length in milliseconds (1M is approximated as 30 days; only used for spacing).
INTERVALS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}

SYMBOLS_TTL_S = 6 * 3600


class BinanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Klines:
    """Closed candles, oldest first. Times are open times in ms since the epoch."""

    symbol: str
    interval: str
    open_time: list[int]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]

    def __len__(self) -> int:
        return len(self.close)


def _get_json(path: str, params: dict[str, object], timeout: float = 20.0) -> object:
    """GET `path` from the first base URL that answers; 4xx errors from Binance are not retried elsewhere."""
    query = urllib.parse.urlencode(params)
    errors: list[str] = []
    for base in BASE_URLS:
        url = f"{base}{path}?{query}" if query else f"{base}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            # 400 means a bad request (unknown symbol...): the fallback would say the same.
            if e.code == 400:
                try:
                    msg = json.loads(body).get("msg", body)
                except ValueError:
                    msg = body
                raise BinanceError(msg) from e
            errors.append(f"{base}: HTTP {e.code} {body}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append(f"{base}: {e}")
    raise BinanceError("Binance unreachable: " + "; ".join(errors))


_symbols_lock = threading.Lock()
_symbols_cache: tuple[float, list[dict[str, str]]] | None = None


def list_symbols(fetch=_get_json) -> list[dict[str, str]]:
    """Spot pairs currently trading, as `{"symbol", "base", "quote"}`, cached for a few hours."""
    global _symbols_cache
    with _symbols_lock:
        if _symbols_cache is not None and time.time() - _symbols_cache[0] < SYMBOLS_TTL_S:
            return _symbols_cache[1]
        info = fetch("/api/v3/exchangeInfo", {"permissions": "SPOT", "symbolStatus": "TRADING"}, timeout=30.0)
        symbols = [
            {"symbol": s["symbol"], "base": s["baseAsset"], "quote": s["quoteAsset"]}
            for s in info.get("symbols", [])
            if s.get("status") == "TRADING"
        ]
        symbols.sort(key=lambda s: s["symbol"])
        _symbols_cache = (time.time(), symbols)
        return symbols


def fetch_klines(
    symbol: str,
    interval: str,
    bars: int,
    fetch=_get_json,
    now_ms: int | None = None,
    progress=None,
) -> Klines:
    """The last `bars` closed candles of `symbol`, paged backward `PAGE_LIMIT` at a time.

    The still-open last candle is dropped. Fewer bars come back when the pair is younger.
    `progress(done, total)` is called after each page.
    """
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}")
    if bars < 1:
        raise ValueError("bars must be >= 1")
    symbol = symbol.upper()
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms

    rows: list[list] = []
    end_time: int | None = None
    # One extra row in case the newest one is still open.
    want = bars + 1
    while len(rows) < want:
        params: dict[str, object] = {"symbol": symbol, "interval": interval, "limit": min(PAGE_LIMIT, want - len(rows))}
        if end_time is not None:
            params["endTime"] = end_time
        page = fetch("/api/v3/klines", params)
        if not isinstance(page, list):
            raise BinanceError(f"unexpected klines response: {str(page)[:200]}")
        if not page:
            break
        rows = page + rows
        if progress is not None:
            progress(min(len(rows), bars), bars)
        if len(page) < params["limit"]:
            break  # reached the pair's first candle
        end_time = int(page[0][0]) - 1

    # Close time is index 6; a candle is closed once its close time is in the past.
    rows = [r for r in rows if int(r[6]) < now_ms]
    # Paging can overlap if Binance returns inclusive bounds; dedupe on open time.
    seen: set[int] = set()
    unique = []
    for r in rows:
        t = int(r[0])
        if t not in seen:
            seen.add(t)
            unique.append(r)
    unique.sort(key=lambda r: int(r[0]))
    unique = unique[-bars:]
    if not unique:
        raise BinanceError(f"no closed candles for {symbol} {interval}")
    return Klines(
        symbol=symbol,
        interval=interval,
        open_time=[int(r[0]) for r in unique],
        open=[float(r[1]) for r in unique],
        high=[float(r[2]) for r in unique],
        low=[float(r[3]) for r in unique],
        close=[float(r[4]) for r in unique],
        volume=[float(r[5]) for r in unique],
    )
