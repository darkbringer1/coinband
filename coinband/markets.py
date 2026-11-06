"""Data sources the app can backtest: Binance (crypto) and Yahoo (stocks, indices, FX, futures)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from coinband import binance_data, yahoo_data


@dataclass
class Source:
    name: str
    label: str
    intervals: dict[str, int]  # interval -> bar length in ms
    fetch: Callable  # (symbol, interval, bars, progress=None) -> Klines
    current_price: Callable  # (symbol) -> {"price", "time"}
    symbols: Callable  # () -> [{"symbol", ...}]
    symbol_pattern: str
    default_symbol: str
    default_interval: str
    leaderboard: list[str] = field(default_factory=list)

    def parse_symbol(self, raw) -> str:
        symbol = str(raw).strip().upper()
        if not re.fullmatch(self.symbol_pattern, symbol):
            raise ValueError(f"{symbol or 'symbol'}: not a valid {self.label} symbol")
        return symbol


def default_sources() -> dict[str, Source]:
    return {
        "binance": Source(
            name="binance",
            label="Binance",
            intervals=dict(binance_data.INTERVALS),
            fetch=binance_data.fetch_klines,
            current_price=binance_data.current_price,
            symbols=binance_data.list_symbols,
            symbol_pattern=r"[A-Z0-9]{2,20}",
            default_symbol="BTCUSDT",
            default_interval="1h",
            leaderboard=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"],
        ),
        "yahoo": Source(
            name="yahoo",
            label="Yahoo",
            intervals={k: v[1] for k, v in yahoo_data.INTERVALS.items()},
            fetch=yahoo_data.fetch_klines,
            current_price=yahoo_data.current_price,
            symbols=yahoo_data.symbols,
            symbol_pattern=r"[A-Z0-9^][A-Z0-9.^=\-]{0,19}",
            default_symbol="^NDX",
            default_interval="1h",
            leaderboard=["^NDX", "AAPL", "MSFT", "NVDA", "XU100.IS", "THYAO.IS", "GARAN.IS", "ASELS.IS"],
        ),
    }
