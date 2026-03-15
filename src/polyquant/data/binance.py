"""Binance OHLCV data fetching via ccxt."""

import ccxt
import pandas as pd

# CCXT single-request limit for Binance
_MAX_PER_REQUEST = 1000


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
        """Fetch OHLCV candles with automatic pagination.

        When *limit* exceeds the per-request cap (1000), multiple sequential
        requests are issued, each starting after the last returned timestamp.
        """
        all_rows: list[list] = []
        remaining = limit

        while remaining > 0:
            batch_size = min(remaining, _MAX_PER_REQUEST)
            raw = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=batch_size,
            )
            if not raw:
                break

            all_rows.extend(raw)
            remaining -= len(raw)

            # If we got fewer than requested, there's no more data
            if len(raw) < batch_size:
                break

            # Next page starts after the last candle's timestamp
            since = raw[-1][0] + 1

        if not all_rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        return df
