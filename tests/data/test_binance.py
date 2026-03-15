from unittest.mock import MagicMock, patch
import pandas as pd
from polyquant.data.binance import BinanceFetcher


def _make_mock_exchange(ohlcv_data):
    """Create a mock ccxt exchange returning given OHLCV data."""
    exchange = MagicMock()
    exchange.fetch_ohlcv.return_value = ohlcv_data
    return exchange


def test_fetch_ohlcv_returns_dataframe():
    raw = [
        [1704067200000, 100.0, 102.0, 99.0, 101.0, 1000.0],
        [1704070800000, 101.0, 103.0, 100.0, 102.0, 1100.0],
    ]
    exchange = _make_mock_exchange(raw)
    fetcher = BinanceFetcher(exchange=exchange)
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 101.0


def test_fetch_ohlcv_empty():
    exchange = _make_mock_exchange([])
    fetcher = BinanceFetcher(exchange=exchange)
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
