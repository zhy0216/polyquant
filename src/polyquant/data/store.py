"""SQLite data storage for OHLCV and Polymarket price snapshots."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataStore:
    """Local SQLite store for market data."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        self._migrate_tables()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self) -> None:
        logger.info("Initializing database tables at %s", self.db_path)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS polymarket_prices (
                    timestamp TEXT NOT NULL,
                    market_slug TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    yes_price REAL NOT NULL,
                    no_price REAL NOT NULL,
                    PRIMARY KEY (timestamp, market_slug, token_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_slug TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL
                )
            """)

    def _migrate_tables(self) -> None:
        """Migrate existing tables to updated schemas if needed."""
        with self._get_conn() as conn:
            # Check if polymarket_prices needs migration
            info = conn.execute("PRAGMA table_info(polymarket_prices)").fetchall()
            if not info:
                return  # Table doesn't exist yet
            # Check primary key columns
            pk_cols = [row[1] for row in info if row[5] > 0]  # col[5] is pk flag
            if "token_id" not in pk_cols:
                logger.info("Migrating polymarket_prices table to include token_id in primary key")
                conn.execute("ALTER TABLE polymarket_prices RENAME TO _polymarket_prices_old")
                conn.execute("""
                    CREATE TABLE polymarket_prices (
                        timestamp TEXT NOT NULL,
                        market_slug TEXT NOT NULL,
                        token_id TEXT NOT NULL,
                        yes_price REAL NOT NULL,
                        no_price REAL NOT NULL,
                        PRIMARY KEY (timestamp, market_slug, token_id)
                    )
                """)
                conn.execute("""
                    INSERT INTO polymarket_prices
                    SELECT * FROM _polymarket_prices_old
                """)
                conn.execute("DROP TABLE _polymarket_prices_old")

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save OHLCV data, upserting on (symbol, timeframe, timestamp)."""
        logger.info("Saving %d OHLCV rows for %s/%s", len(df), symbol, timeframe)
        with self._get_conn() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO ohlcv
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, timeframe, str(row["timestamp"]),
                     row["open"], row["high"], row["low"], row["close"], row["volume"]),
                )

    def load_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load OHLCV data as DataFrame, sorted by timestamp."""
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
                conn, params=(symbol, timeframe),
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        logger.info("Loaded %d OHLCV rows for %s/%s", len(df), symbol, timeframe)
        return df

    def save_polymarket_prices(self, df: pd.DataFrame) -> None:
        """Save Polymarket price snapshots."""
        logger.info("Saving %d Polymarket price snapshots", len(df))
        with self._get_conn() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO polymarket_prices
                    (timestamp, market_slug, token_id, yes_price, no_price)
                    VALUES (?, ?, ?, ?, ?)""",
                    (str(row["timestamp"]), row["market_slug"], row["token_id"],
                     row["yes_price"], row["no_price"]),
                )

    def load_polymarket_prices(self, market_slug: str) -> pd.DataFrame:
        """Load Polymarket price snapshots for a market."""
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM polymarket_prices WHERE market_slug = ? ORDER BY timestamp",
                conn, params=(market_slug,),
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        return df
