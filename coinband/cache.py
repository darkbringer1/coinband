"""Local cache of fetched bars, updated incrementally.

One JSON file per (source, symbol, interval) under `~/.cache/coinband` (or `$COINBAND_CACHE`), never in
the repository. A request within `TTL_S` of the last fetch is served from disk. Otherwise only the bars
since the newest cached one are fetched and merged in, unless more history is asked for than is cached.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from coinband.binance_data import Klines

TTL_S = 60
TAIL_PAD = 5  # extra bars fetched on an update, for overlap
_FIELDS = ("open_time", "open", "high", "low", "close", "volume")
_lock = threading.Lock()


def cache_dir() -> Path:
    return Path(os.environ.get("COINBAND_CACHE", Path.home() / ".cache" / "coinband"))


def _path(source: str, symbol: str, interval: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
    return cache_dir() / source / f"{safe}-{interval}.json"


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)


def _to_klines(d: dict, symbol: str, interval: str, bars: int) -> Klines:
    return Klines(symbol=symbol, interval=interval, info=d.get("info", {}), **{f: d[f][-bars:] for f in _FIELDS})


def _merge(old: dict | None, new: Klines) -> dict:
    rows = {}
    if old:
        for i, t in enumerate(old["open_time"]):
            rows[t] = tuple(old[f][i] for f in _FIELDS)
    for i, t in enumerate(new.open_time):
        rows[t] = tuple(getattr(new, f)[i] for f in _FIELDS)
    ordered = [rows[t] for t in sorted(rows)]
    out = {f: [r[k] for r in ordered] for k, f in enumerate(_FIELDS)}
    out["info"] = {**(old or {}).get("info", {}), **new.info}
    return out


def get_klines(
    source: str,
    symbol: str,
    interval: str,
    bars: int,
    fetch: Callable[..., Klines],
    interval_ms: int,
    progress=None,
    now: float | None = None,
) -> Klines:
    """`bars` closed bars, from the cache when fresh, fetching only what is missing otherwise.

    `fetch(symbol, interval, n, progress=...)` returns the last `n` closed bars from the source.
    """
    now = time.time() if now is None else now
    symbol = symbol.upper()
    path = _path(source, symbol, interval)
    with _lock:
        cached = _load(path)
    if cached and cached.get("open_time"):
        have = len(cached["open_time"])
        enough_history = have >= bars or cached.get("complete", False)
        fresh = now - cached.get("fetched_at", 0) < TTL_S
        if enough_history and fresh:
            if progress is not None:
                progress(min(have, bars), bars)
            return _to_klines(cached, symbol, interval, bars)
        if enough_history:
            missing = math.ceil((now * 1000 - cached["open_time"][-1]) / interval_ms)
            new = fetch(symbol, interval, max(1, missing) + TAIL_PAD, progress=progress)
            merged = _merge(cached, new)
            merged["complete"] = cached.get("complete", False)
            merged["fetched_at"] = now
            with _lock:
                _save(path, merged)
            return _to_klines(merged, symbol, interval, bars)
    new = fetch(symbol, interval, bars, progress=progress)
    merged = _merge(cached, new)
    # fewer bars than asked for means the series starts there: no older history to fetch later
    merged["complete"] = len(new) < bars or (cached or {}).get("complete", False)
    merged["fetched_at"] = now
    with _lock:
        _save(path, merged)
    return _to_klines(merged, symbol, interval, bars)
