from unittest.mock import MagicMock, patch
import ccxt
import pandas as pd
import pytest
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


def test_fetch_ohlcv_pagination():
    """When limit > 1000, fetch_ohlcv paginates across multiple requests."""
    ts_base = 1704067200000
    hour_ms = 3_600_000

    batch1 = [[ts_base + i * hour_ms, 100.0, 102.0, 99.0, 101.0, 1000.0]
               for i in range(1000)]
    batch2 = [[ts_base + (1000 + i) * hour_ms, 100.0, 102.0, 99.0, 101.0, 1000.0]
               for i in range(500)]

    exchange = MagicMock()
    exchange.fetch_ohlcv.side_effect = [batch1, batch2]

    fetcher = BinanceFetcher(exchange=exchange)
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1500)

    assert len(df) == 1500
    assert exchange.fetch_ohlcv.call_count == 2
    # Second call should start after the last timestamp of batch1
    second_call = exchange.fetch_ohlcv.call_args_list[1]
    assert second_call.kwargs["since"] == batch1[-1][0] + 1


def test_fetch_ohlcv_pagination_stops_early():
    """Pagination stops when a batch returns fewer rows than requested."""
    ts_base = 1704067200000
    hour_ms = 3_600_000

    batch1 = [[ts_base + i * hour_ms, 100.0, 102.0, 99.0, 101.0, 1000.0]
               for i in range(800)]  # asked for 1000, got 800 → no more data

    exchange = MagicMock()
    exchange.fetch_ohlcv.side_effect = [batch1]

    fetcher = BinanceFetcher(exchange=exchange)
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=2000)

    assert len(df) == 800
    assert exchange.fetch_ohlcv.call_count == 1


@patch("polyquant.data.binance.time.sleep")
def test_fetch_ohlcv_retries_on_network_error(mock_sleep):
    raw = [
        [1704067200000, 100.0, 102.0, 99.0, 101.0, 1000.0],
    ]
    exchange = MagicMock()
    exchange.timeout = 30000
    exchange.fetch_ohlcv.side_effect = [
        ccxt.NetworkError("connection reset"),
        raw,
    ]
    fetcher = BinanceFetcher(exchange=exchange)
    df = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1)
    assert len(df) == 1
    assert exchange.fetch_ohlcv.call_count == 2
    mock_sleep.assert_called_once()


@patch("polyquant.data.binance.time.sleep")
def test_fetch_ohlcv_raises_after_max_retries(mock_sleep):
    exchange = MagicMock()
    exchange.timeout = 30000
    exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("down")
    fetcher = BinanceFetcher(exchange=exchange)
    with pytest.raises(ccxt.NetworkError):
        fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1)
    assert exchange.fetch_ohlcv.call_count == 4  # 1 + 3 retries
