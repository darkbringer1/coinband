"""Open times of the bars after the data end, for labelling the live forecast.

Markets that trade around the clock step by the interval. Session markets (stocks, indices) repeat the
bar times of day seen in recent sessions on the following weekdays. Holidays are not known, so a forecast
that crosses one is labelled a day early.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DAY_MS = 86_400_000


def future_times(open_times: list[int], interval_ms: int, horizon: int, timezone: str | None, continuous: bool) -> list[int]:
    if not open_times:
        return []
    last = open_times[-1]
    if continuous or interval_ms >= 7 * DAY_MS:
        return [last + k * interval_ms for k in range(1, horizon + 1)]
    try:
        tz = ZoneInfo(timezone or "UTC")
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    local = [datetime.fromtimestamp(t / 1000, tz) for t in open_times[-400:]]
    if interval_ms >= DAY_MS:
        slots = [local[-1].time()]
    else:
        # bar times of day over the last sessions; with enough days, drop times seen only once (odd prints)
        seen: dict = {}
        for d in local:
            seen.setdefault(d.time(), set()).add(d.date())
        n_days = len({d.date() for d in local})
        slots = sorted(t for t, days in seen.items() if n_days < 5 or len(days) >= 2) or sorted(seen)
    out: list[int] = []
    day = local[-1].date()
    after = local[-1].time()
    while len(out) < horizon:
        for slot in slots:
            if after is not None and slot <= after:
                continue
            dt = datetime.combine(day, slot, tz)
            out.append(int(dt.timestamp() * 1000))
            if len(out) == horizon:
                break
        after = None
        day += timedelta(days=1)
        while day.weekday() >= 5:  # Saturday, Sunday
            day += timedelta(days=1)
    return out
