"""Binance OHLCV data fetching via ccxt."""

import logging
import time

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# CCXT single-request limit for Binance
_MAX_PER_REQUEST = 1000
_BATCH_MAX_RETRIES = 3
_BATCH_BASE_DELAY = 1.0


class BinanceFetcher:
    """Fetches candlestick data from Binance."""

    def __init__(self, exchange: ccxt.Exchange | None = None) -> None:
        self.exchange = exchange or ccxt.binance()
        self.exchange.timeout = 30000

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
        logger.info("Fetching OHLCV for %s (%s), limit=%d", symbol, timeframe, limit)
        all_rows: list[list] = []
        remaining = limit

        while remaining > 0:
            batch_size = min(remaining, _MAX_PER_REQUEST)
            raw = None
            for attempt in range(_BATCH_MAX_RETRIES + 1):
                try:
                    raw = self.exchange.fetch_ohlcv(
                        symbol, timeframe=timeframe, since=since, limit=batch_size,
                    )
                    break
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                    if attempt == _BATCH_MAX_RETRIES:
                        logger.error("All %d retries exhausted for fetch_ohlcv: %s", _BATCH_MAX_RETRIES, e)
                        raise
                    delay = _BATCH_BASE_DELAY * (2 ** attempt)
                    logger.warning("Retry %d/%d for fetch_ohlcv after %.1fs: %s", attempt + 1, _BATCH_MAX_RETRIES, delay, e)
                    time.sleep(delay)
                except ccxt.BaseError:
                    logger.error("Unrecoverable ccxt error fetching %s", symbol)
                    raise
            if not raw:
                break

            all_rows.extend(raw)
            remaining -= len(raw)
            logger.debug("Fetched %d candles, %d remaining", len(raw), max(remaining, 0))

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
        logger.info("Fetched %d candles for %s", len(df), symbol)
        return df
