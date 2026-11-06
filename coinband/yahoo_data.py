"""Stocks, indices, FX and futures from Yahoo Finance's public chart endpoint (no key).

The endpoint is unofficial and Yahoo's terms restrict redistributing its data: fine for a personal
backtest, but coinband does not ship or cache Yahoo data in the repository.

Bars only exist while a market trades, so a series of hourly stock bars skips nights, weekends and holidays.
Timestral sees consecutive bars either way. The still-open bar is dropped while its session runs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from coinband.binance_data import BinanceError, Klines

BASE_URLS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
USER_AGENT = "Mozilla/5.0 (compatible; coinband/0.1)"

# interval -> (Yahoo interval code, bar length in ms, maximum lookback in days or None for unlimited)
INTERVALS: dict[str, tuple[str, int, int | None]] = {
    "15m": ("15m", 900_000, 59),
    "30m": ("30m", 1_800_000, 59),
    "1h": ("60m", 3_600_000, 729),
    "1d": ("1d", 86_400_000, None),
    "1w": ("1wk", 604_800_000, None),
}

# Suggestions for the symbol picker. Index memberships change; any Yahoo symbol can be typed in.
PRESETS: dict[str, list[tuple[str, str]]] = {
    "Nasdaq": [
        ("^NDX", "Nasdaq-100 index"), ("^IXIC", "Nasdaq Composite"), ("QQQ", "Invesco QQQ ETF"),
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"), ("AMZN", "Amazon"), ("GOOGL", "Alphabet"),
        ("META", "Meta"), ("TSLA", "Tesla"), ("AVGO", "Broadcom"), ("COST", "Costco"), ("NFLX", "Netflix"),
        ("AMD", "AMD"), ("PEP", "PepsiCo"), ("ADBE", "Adobe"), ("CSCO", "Cisco"), ("INTC", "Intel"),
        ("QCOM", "Qualcomm"), ("PYPL", "PayPal"),
    ],
    "BIST": [
        ("XU100.IS", "BIST 100 index"), ("XU030.IS", "BIST 30 index"),
        ("THYAO.IS", "Turkish Airlines"), ("ASELS.IS", "Aselsan"), ("GARAN.IS", "Garanti BBVA"),
        ("AKBNK.IS", "Akbank"), ("ISCTR.IS", "Isbank"), ("YKBNK.IS", "Yapi Kredi"), ("KCHOL.IS", "Koc Holding"),
        ("SAHOL.IS", "Sabanci Holding"), ("BIMAS.IS", "BIM"), ("EREGL.IS", "Eregli Demir Celik"),
        ("TUPRS.IS", "Tupras"), ("SISE.IS", "Sisecam"), ("FROTO.IS", "Ford Otosan"), ("TCELL.IS", "Turkcell"),
        ("PGSUS.IS", "Pegasus"), ("TOASO.IS", "Tofas"), ("ARCLK.IS", "Arcelik"), ("KOZAL.IS", "Koza Altin"),
    ],
    "Global": [
        ("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^GDAXI", "DAX"), ("^FTSE", "FTSE 100"),
        ("^N225", "Nikkei 225"), ("GC=F", "Gold futures"), ("CL=F", "Crude oil futures"),
        ("EURUSD=X", "EUR/USD"), ("TRY=X", "USD/TRY"),
    ],
}


class YahooError(BinanceError):
    """Same handling as Binance errors in the server: shown to the user as is."""


def _get_json(symbol: str, params: dict[str, object], timeout: float = 20.0) -> dict:
    query = urllib.parse.urlencode(params)
    errors = []
    for base in BASE_URLS:
        url = f"{base}/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            try:
                msg = json.loads(body)["chart"]["error"]["description"]
            except (ValueError, KeyError, TypeError):
                msg = body
            if e.code in (400, 404, 422):
                raise YahooError(f"{symbol}: {msg}") from e
            errors.append(f"{base}: HTTP {e.code} {msg}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append(f"{base}: {e}")
    raise YahooError("Yahoo unreachable: " + "; ".join(errors))


def _chart(symbol: str, interval: str, fetch=_get_json, now: float | None = None) -> dict:
    code, _, max_days = INTERVALS[interval]
    now = time.time() if now is None else now
    start = 0 if max_days is None else int(now - max_days * 86_400)
    data = fetch(symbol, {"interval": code, "period1": start, "period2": int(now), "includePrePost": "false"})
    chart = data.get("chart") or {}
    if chart.get("error"):
        raise YahooError(f"{symbol}: {chart['error'].get('description', chart['error'])}")
    results = chart.get("result") or []
    if not results:
        raise YahooError(f"{symbol}: no data")
    return results[0]


def fetch_klines(symbol: str, interval: str, bars: int, fetch=_get_json, now_ms: int | None = None, progress=None) -> Klines:
    """The last `bars` closed bars of `symbol` (as many as Yahoo has for the interval), oldest first."""
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {', '.join(INTERVALS)} for stocks and indices")
    symbol = symbol.strip().upper()
    now = (now_ms / 1000) if now_ms is not None else time.time()
    r = _chart(symbol, interval, fetch, now)
    meta = r.get("meta", {})
    ts = r.get("timestamp") or []
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    cols = {k: quote.get(k) or [None] * len(ts) for k in ("open", "high", "low", "close", "volume")}
    bar_s = INTERVALS[interval][1] / 1000
    session_end = (meta.get("currentTradingPeriod") or {}).get("regular", {}).get("end")
    session_open = session_end is not None and now < session_end
    rows = []
    for i, t in enumerate(ts):
        c = cols["close"][i]
        if c is None:
            continue
        # a bar is still forming while its session runs and its period has not ended
        if session_open and t + bar_s > now:
            continue
        o, h, lo = cols["open"][i], cols["high"][i], cols["low"][i]
        v = cols["volume"][i]
        rows.append((int(t) * 1000, o if o is not None else c, h if h is not None else c, lo if lo is not None else c, c, v or 0.0))
    rows = rows[-bars:]
    if not rows:
        raise YahooError(f"no closed bars for {symbol} {interval}")
    if progress is not None:
        progress(len(rows), len(rows))
    return Klines(
        symbol=symbol,
        interval=interval,
        open_time=[r[0] for r in rows],
        open=[float(r[1]) for r in rows],
        high=[float(r[2]) for r in rows],
        low=[float(r[3]) for r in rows],
        close=[float(r[4]) for r in rows],
        volume=[float(r[5]) for r in rows],
        info={
            "name": meta.get("longName") or meta.get("shortName") or symbol,
            "currency": meta.get("currency"),
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
            "timezone": meta.get("exchangeTimezoneName"),
            "continuous": False,
            "price": meta.get("regularMarketPrice"),
            "price_time": int(meta["regularMarketTime"]) * 1000 if meta.get("regularMarketTime") else None,
        },
    )


def current_price(symbol: str, fetch=_get_json) -> dict:
    r = _chart(symbol.strip().upper(), "1d", fetch)
    meta = r.get("meta", {})
    return {"price": meta.get("regularMarketPrice"), "time": int(meta.get("regularMarketTime", 0)) * 1000}


def symbols() -> list[dict[str, str]]:
    return [{"symbol": s, "name": n, "group": g} for g, items in PRESETS.items() for s, n in items]
