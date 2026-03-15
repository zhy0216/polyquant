"""Binance OHLCV data fetching via ccxt."""

import ccxt
import pandas as pd


class BinanceFetcher:
    """Fetches candlestick data from Binance."""

    def __init__(self, exchange: ccxt.Exchange | None = None) -> None:
        self.exchange = exchange or ccxt.binance()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return as DataFrame."""
        raw = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit,
        )
        if not raw:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
