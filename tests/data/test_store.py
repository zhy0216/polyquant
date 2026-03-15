import pandas as pd
import pytest
import sqlite3
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


def test_polymarket_prices_different_tokens_same_timestamp(store):
    """Two snapshots with same timestamp+market_slug but different token_id should both be preserved."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 12:00"]),
        "market_slug": ["will-btc-above-70000", "will-btc-above-70000"],
        "token_id": ["0xyes_token", "0xno_token"],
        "yes_price": [0.55, 0.45],
        "no_price": [0.45, 0.55],
    })
    store.save_polymarket_prices(df)
    loaded = store.load_polymarket_prices("will-btc-above-70000")
    assert len(loaded) == 2
    token_ids = set(loaded["token_id"])
    assert token_ids == {"0xyes_token", "0xno_token"}


def test_polymarket_prices_migration(tmp_path):
    """Migration should convert old schema (PK without token_id) to new schema."""
    db_path = str(tmp_path / "migrate.db")
    # Create old-schema table manually
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE polymarket_prices (
            timestamp TEXT NOT NULL,
            market_slug TEXT NOT NULL,
            token_id TEXT NOT NULL,
            yes_price REAL NOT NULL,
            no_price REAL NOT NULL,
            PRIMARY KEY (timestamp, market_slug)
        )
    """)
    conn.execute(
        "INSERT INTO polymarket_prices VALUES (?, ?, ?, ?, ?)",
        ("2025-01-01 12:00:00", "will-btc-above-70000", "0xabc", 0.55, 0.45),
    )
    conn.commit()
    conn.close()

    # Opening DataStore should trigger migration
    store = DataStore(db_path)

    # Verify migrated data survived
    loaded = store.load_polymarket_prices("will-btc-above-70000")
    assert len(loaded) == 1
    assert loaded["token_id"].iloc[0] == "0xabc"

    # Verify we can now insert two rows with same timestamp+market_slug but different token_id
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-06-01 12:00", "2025-06-01 12:00"]),
        "market_slug": ["will-btc-above-70000", "will-btc-above-70000"],
        "token_id": ["0xyes", "0xno"],
        "yes_price": [0.55, 0.45],
        "no_price": [0.45, 0.55],
    })
    store.save_polymarket_prices(df)
    loaded = store.load_polymarket_prices("will-btc-above-70000")
    assert len(loaded) == 3  # 1 migrated + 2 new
