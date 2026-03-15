import pandas as pd
import pytest
from polyquant.data.store import DataStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return DataStore(db_path)


def test_save_and_load_ohlcv(store):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000.0, 1100.0],
    })
    store.save_ohlcv("BTC/USDT", "1h", df)
    loaded = store.load_ohlcv("BTC/USDT", "1h")
    assert len(loaded) == 2
    assert loaded["close"].iloc[0] == 101.0


def test_ohlcv_upsert_no_duplicates(store):
    df1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 00:00"]),
        "open": [100.0], "high": [102.0], "low": [99.0],
        "close": [101.0], "volume": [1000.0],
    })
    df2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
        "open": [100.0, 105.0], "high": [102.0, 107.0], "low": [99.0, 104.0],
        "close": [101.0, 106.0], "volume": [1000.0, 1200.0],
    })
    store.save_ohlcv("BTC/USDT", "1h", df1)
    store.save_ohlcv("BTC/USDT", "1h", df2)
    loaded = store.load_ohlcv("BTC/USDT", "1h")
    assert len(loaded) == 2


def test_save_and_load_polymarket_prices(store):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 12:00"]),
        "market_slug": ["will-btc-above-70000"],
        "token_id": ["0xabc"],
        "yes_price": [0.55],
        "no_price": [0.45],
    })
    store.save_polymarket_prices(df)
    loaded = store.load_polymarket_prices("will-btc-above-70000")
    assert len(loaded) == 1
    assert loaded["yes_price"].iloc[0] == 0.55
