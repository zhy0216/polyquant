# PolyQuant Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python trading pipeline that uses LightGBM to predict BTC/ETH prices, compares predictions against Polymarket market prices, and trades when mispricings are detected.

**Architecture:** Linear pipeline — data collection (Binance OHLCV + Polymarket prices) → feature engineering (technical indicators) → LightGBM probability model → signal generation (model vs market comparison) → execution (backtest → paper → live). SQLite for local storage, asyncio for scheduling.

**Tech Stack:** Python 3.11+, ccxt, py-clob-client, LightGBM, pandas, ta, pydantic, pytest, SQLite

**Spec:** `docs/superpowers/specs/2026-03-15-polyquant-design.md`

---

## File Structure

```
polyquant/
├── pyproject.toml                          # Project config, dependencies
├── .env.example                            # Template for env vars
├── .gitignore                              # Ignore .env, data/, *.db, models/
├── src/
│   └── polyquant/
│       ├── __init__.py
│       ├── config.py                       # Pydantic settings + strategy params
│       ├── data/
│       │   ├── __init__.py
│       │   ├── store.py                    # SQLite schema + query interface
│       │   ├── binance.py                  # Binance OHLCV fetcher via ccxt
│       │   └── polymarket.py               # Polymarket market discovery + price snapshots
│       ├── model/
│       │   ├── __init__.py
│       │   ├── features.py                 # Technical indicator feature engineering
│       │   └── predictor.py                # LightGBM train/predict/save/load
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── signal.py                   # Signal generation (model prob vs market price)
│       │   └── sizing.py                   # Kelly criterion position sizing
│       └── execution/
│           ├── __init__.py
│           ├── backtest.py                 # Backtesting engine
│           ├── paper.py                    # Paper trading (simulated)
│           └── live.py                     # Live trading via Polymarket CLOB
├── tests/
│   ├── conftest.py                         # Shared fixtures
│   ├── test_config.py
│   ├── data/
│   │   ├── test_store.py
│   │   ├── test_binance.py
│   │   └── test_polymarket.py
│   ├── model/
│   │   ├── test_features.py
│   │   └── test_predictor.py
│   ├── strategy/
│   │   ├── test_signal.py
│   │   └── test_sizing.py
│   └── execution/
│       ├── test_backtest.py
│       └── test_paper.py
├── notebooks/
│   └── exploration.ipynb                   # Analysis & visualization
└── data/                                   # Local data dir (gitignored)
```

---

## Chunk 1: Project Scaffolding + Data Layer

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/polyquant/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "polyquant"
version = "0.1.0"
description = "Polymarket BTC/ETH automated trading system"
requires-python = ">=3.11"
dependencies = [
    "ccxt>=4.0",
    "py-clob-client>=0.1",
    "lightgbm>=4.0",
    "pandas>=2.0",
    "ta>=0.11",
    "joblib>=1.3",
    "python-dotenv>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "requests>=2.28",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create .gitignore**

```
.env
data/*.db
data/*.sqlite
models/
__pycache__/
*.pyc
.pytest_cache/
dist/
*.egg-info/
.venv/
```

- [ ] **Step 3: Create .env.example**

```
# Binance (read-only, no key needed for public data)
# BINANCE_API_KEY=
# BINANCE_API_SECRET=

# Polymarket
POLYMARKET_PRIVATE_KEY=0x_your_private_key_here
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=
```

- [ ] **Step 4: Create src/polyquant/__init__.py**

```python
"""PolyQuant: Polymarket BTC/ETH automated trading system."""
```

- [ ] **Step 5: Create empty __init__.py files for subpackages**

Create empty `__init__.py` in: `src/polyquant/data/`, `src/polyquant/model/`, `src/polyquant/strategy/`, `src/polyquant/execution/`

- [ ] **Step 6: Install project in dev mode**

Run: `cd /Users/yang/workspace/polyquant && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: Successful install with all dependencies

- [ ] **Step 7: Verify pytest runs**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest --co`
Expected: "no tests ran" (no test files yet)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/
git commit -m "feat: project scaffolding with dependencies"
```

---

### Task 2: Configuration

**Files:**
- Create: `src/polyquant/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config**

```python
# tests/test_config.py
from polyquant.config import Settings


def test_default_settings():
    settings = Settings(
        polymarket_private_key="0xabc123",
    )
    assert settings.trading_pairs == ["BTC/USDT", "ETH/USDT"]
    assert settings.signal_threshold == 0.10
    assert settings.kelly_fraction == 0.5
    assert settings.max_position_pct == 0.05
    assert settings.max_exposure_pct == 0.30
    assert settings.prediction_horizon_hours == 24
    assert settings.data_fetch_interval_minutes == 15
    assert settings.ohlcv_timeframe == "1h"


def test_settings_override():
    settings = Settings(
        polymarket_private_key="0xabc123",
        signal_threshold=0.15,
        kelly_fraction=0.25,
    )
    assert settings.signal_threshold == 0.15
    assert settings.kelly_fraction == 0.25


def test_settings_default_empty_key():
    settings = Settings()
    assert settings.polymarket_private_key == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/test_config.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement config**

```python
# src/polyquant/config.py
"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    # Polymarket credentials
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""

    # Trading pairs (Binance symbols)
    trading_pairs: list[str] = Field(default=["BTC/USDT", "ETH/USDT"])

    # Strategy parameters
    signal_threshold: float = 0.10
    kelly_fraction: float = 0.5
    max_position_pct: float = 0.05
    max_exposure_pct: float = 0.30

    # Model parameters
    prediction_horizon_hours: int = 24
    ohlcv_timeframe: str = "1h"

    # Scheduling
    data_fetch_interval_minutes: int = 15

    # Paths
    db_path: str = "data/polyquant.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/config.py tests/test_config.py
git commit -m "feat: add pydantic settings configuration"
```

---

### Task 3: Data Store (SQLite)

**Files:**
- Create: `src/polyquant/data/store.py`
- Create: `tests/data/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/data/test_store.py`

- [ ] **Step 1: Write failing tests for DataStore**

```python
# tests/data/test_store.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/data/test_store.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement DataStore**

```python
# src/polyquant/data/store.py
"""SQLite data storage for OHLCV and Polymarket price snapshots."""

import sqlite3
from pathlib import Path

import pandas as pd


class DataStore:
    """Local SQLite store for market data."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self) -> None:
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
                    PRIMARY KEY (timestamp, market_slug)
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

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save OHLCV data, upserting on (symbol, timeframe, timestamp)."""
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
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def save_polymarket_prices(self, df: pd.DataFrame) -> None:
        """Save Polymarket price snapshots."""
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
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
```

- [ ] **Step 4: Create tests/__init__.py, tests/conftest.py and tests/data/__init__.py**

`tests/conftest.py`:
```python
# tests/conftest.py
"""Shared test fixtures."""
```

Empty `__init__.py` files for `tests/` and `tests/data/`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/data/test_store.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/polyquant/data/store.py tests/
git commit -m "feat: SQLite data store with OHLCV and Polymarket tables"
```

---

### Task 4: Binance Data Fetcher

**Files:**
- Create: `src/polyquant/data/binance.py`
- Create: `tests/data/test_binance.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_binance.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/data/test_binance.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement BinanceFetcher**

```python
# src/polyquant/data/binance.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/data/test_binance.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/data/binance.py tests/data/test_binance.py
git commit -m "feat: Binance OHLCV fetcher with ccxt"
```

---

### Task 5: Polymarket Data Fetcher

**Files:**
- Create: `src/polyquant/data/polymarket.py`
- Create: `tests/data/test_polymarket.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_polymarket.py
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from polyquant.data.polymarket import PolymarketFetcher


def test_parse_markets_filters_crypto():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    raw_markets = [
        {
            "condition_id": "0x1",
            "question": "Will BTC be above $70,000 on March 20?",
            "market_slug": "will-btc-above-70000-march-20",
            "tokens": [
                {"token_id": "0xyes1", "outcome": "Yes"},
                {"token_id": "0xno1", "outcome": "No"},
            ],
            "end_date_iso": "2026-03-20T00:00:00Z",
            "active": True,
        },
        {
            "condition_id": "0x2",
            "question": "Will the election result in X?",
            "market_slug": "will-election-result",
            "tokens": [
                {"token_id": "0xyes2", "outcome": "Yes"},
                {"token_id": "0xno2", "outcome": "No"},
            ],
            "end_date_iso": "2026-11-05T00:00:00Z",
            "active": True,
        },
    ]
    crypto_markets = fetcher._filter_crypto_markets(raw_markets)
    assert len(crypto_markets) == 1
    assert crypto_markets[0]["market_slug"] == "will-btc-above-70000-march-20"


def test_extract_threshold_from_question():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    assert fetcher._extract_threshold("Will BTC be above $70,000 on March 20?") == 70000.0
    assert fetcher._extract_threshold("Will ETH be above $4,500 on April 1?") == 4500.0
    assert fetcher._extract_threshold("No price here") is None


def test_build_price_snapshot():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    market = {
        "market_slug": "will-btc-above-70000",
        "tokens": [
            {"token_id": "0xyes1", "outcome": "Yes"},
            {"token_id": "0xno1", "outcome": "No"},
        ],
    }
    prices = {"0xyes1": 0.55, "0xno1": 0.45}
    row = fetcher._build_price_row(market, prices)
    assert row["yes_price"] == 0.55
    assert row["no_price"] == 0.45
    assert row["token_id"] == "0xyes1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/data/test_polymarket.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement PolymarketFetcher**

```python
# src/polyquant/data/polymarket.py
"""Polymarket market discovery and price snapshot fetching."""

import re
from datetime import datetime, timezone

import requests


GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

CRYPTO_KEYWORDS = ["btc", "bitcoin", "eth", "ethereum"]


class PolymarketFetcher:
    """Discovers crypto markets and fetches price snapshots from Polymarket."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def fetch_active_markets(self) -> list[dict]:
        """Fetch all active markets from Gamma API."""
        resp = self.session.get(
            f"{GAMMA_API_URL}/markets",
            params={"active": "true", "closed": "false"},
        )
        resp.raise_for_status()
        return resp.json()

    def get_crypto_markets(self) -> list[dict]:
        """Fetch and filter to crypto price threshold markets."""
        all_markets = self.fetch_active_markets()
        return self._filter_crypto_markets(all_markets)

    def _filter_crypto_markets(self, markets: list[dict]) -> list[dict]:
        """Filter markets to BTC/ETH price threshold markets."""
        result = []
        for m in markets:
            question = m.get("question", "").lower()
            if any(kw in question for kw in CRYPTO_KEYWORDS):
                if self._extract_threshold(m.get("question", "")) is not None:
                    result.append(m)
        return result

    def _extract_threshold(self, question: str) -> float | None:
        """Extract dollar threshold from market question like 'above $70,000'."""
        match = re.search(r"\$[\d,]+", question)
        if not match:
            return None
        price_str = match.group().replace("$", "").replace(",", "")
        try:
            return float(price_str)
        except ValueError:
            return None

    def fetch_price(self, token_id: str) -> float | None:
        """Fetch current midpoint price for a token from CLOB API."""
        try:
            resp = self.session.get(f"{CLOB_API_URL}/midpoint", params={"token_id": token_id})
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except Exception:
            return None

    def _build_price_row(self, market: dict, prices: dict) -> dict:
        """Build a price snapshot row from market info and fetched prices."""
        tokens = market["tokens"]
        yes_token = next(t for t in tokens if t["outcome"] == "Yes")
        no_token = next(t for t in tokens if t["outcome"] == "No")
        return {
            "timestamp": datetime.now(timezone.utc),
            "market_slug": market["market_slug"],
            "token_id": yes_token["token_id"],
            "yes_price": prices.get(yes_token["token_id"], 0.0),
            "no_price": prices.get(no_token["token_id"], 0.0),
        }

    def snapshot_prices(self, markets: list[dict]) -> list[dict]:
        """Fetch current prices for a list of markets and return snapshot rows."""
        rows = []
        for market in markets:
            prices = {}
            for token in market["tokens"]:
                tid = token["token_id"]
                price = self.fetch_price(tid)
                if price is not None:
                    prices[tid] = price
            if prices:
                rows.append(self._build_price_row(market, prices))
        return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/data/test_polymarket.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/data/polymarket.py tests/data/test_polymarket.py
git commit -m "feat: Polymarket market discovery and price fetcher"
```

---

## Chunk 2: Model Layer

### Task 6: Feature Engineering

**Files:**
- Create: `src/polyquant/model/features.py`
- Create: `tests/model/__init__.py`
- Create: `tests/model/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/model/test_features.py
import pandas as pd
import numpy as np
import pytest
from polyquant.model.features import compute_features


@pytest.fixture
def sample_ohlcv():
    """Generate 200 rows of synthetic OHLCV data for feature computation."""
    np.random.seed(42)
    n = 200
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)),
        "low": prices - np.abs(np.random.randn(n)),
        "close": prices + np.random.randn(n) * 0.3,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


def test_compute_features_returns_dataframe(sample_ohlcv):
    features = compute_features(sample_ohlcv)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0


def test_compute_features_has_expected_columns(sample_ohlcv):
    features = compute_features(sample_ohlcv)
    expected = ["rsi_14", "sma_7", "sma_25", "macd", "bb_upper", "bb_lower", "atr_14", "return_1h"]
    for col in expected:
        assert col in features.columns, f"Missing column: {col}"


def test_compute_features_no_nans_in_output(sample_ohlcv):
    features = compute_features(sample_ohlcv)
    # After dropping warmup rows, no NaNs should remain in feature columns
    feature_cols = [c for c in features.columns if c != "timestamp"]
    assert not features[feature_cols].isna().any().any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/model/test_features.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement feature engineering**

```python
# src/polyquant/model/features.py
"""Technical indicator feature engineering from OHLCV data."""

import numpy as np
import pandas as pd
import ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].

    Returns:
        DataFrame with feature columns, NaN warmup rows dropped.
    """
    out = df.copy()

    # Trend
    out["sma_7"] = ta.trend.sma_indicator(out["close"], window=7)
    out["sma_25"] = ta.trend.sma_indicator(out["close"], window=25)
    out["sma_99"] = ta.trend.sma_indicator(out["close"], window=99)
    out["ema_12"] = ta.trend.ema_indicator(out["close"], window=12)
    out["ema_26"] = ta.trend.ema_indicator(out["close"], window=26)
    macd_indicator = ta.trend.MACD(out["close"])
    out["macd"] = macd_indicator.macd()
    out["macd_signal"] = macd_indicator.macd_signal()

    # Momentum
    out["rsi_14"] = ta.momentum.rsi(out["close"], window=14)
    stoch = ta.momentum.StochRSIIndicator(out["close"])
    out["stoch_rsi"] = stoch.stochrsi()

    # Volatility
    bb = ta.volatility.BollingerBands(out["close"])
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_width"] = bb.bollinger_wband()
    out["atr_14"] = ta.volatility.average_true_range(out["high"], out["low"], out["close"], window=14)

    # Volume
    out["volume_sma_20"] = ta.trend.sma_indicator(out["volume"], window=20)
    out["volume_ratio"] = out["volume"] / out["volume_sma_20"]
    out["obv"] = ta.volume.on_balance_volume(out["close"], out["volume"])

    # Price derivatives
    out["return_1h"] = out["close"].pct_change(1)
    out["return_4h"] = out["close"].pct_change(4)
    out["return_24h"] = out["close"].pct_change(24)
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    # Historical volatility (rolling std of returns)
    out["hist_vol_24"] = out["return_1h"].rolling(24).std()

    # Drop warmup rows (99 needed for SMA_99)
    out = out.dropna().reset_index(drop=True)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/model/test_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/model/ tests/model/
git commit -m "feat: technical indicator feature engineering"
```

---

### Task 7: LightGBM Predictor

**Files:**
- Create: `src/polyquant/model/predictor.py`
- Create: `tests/model/test_predictor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/model/test_predictor.py
import numpy as np
import pandas as pd
import pytest
from polyquant.model.predictor import Predictor


@pytest.fixture
def training_data():
    """Synthetic training data with features and binary target."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "feature_c": np.random.randn(n),
    })
    # Target correlated with feature_a
    y = (X["feature_a"] + np.random.randn(n) * 0.5 > 0).astype(int)
    return X, y


def test_train_and_predict(training_data):
    X, y = training_data
    predictor = Predictor()
    predictor.train(X, y)
    probs = predictor.predict_proba(X.iloc[:10])
    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_save_and_load(training_data, tmp_path):
    X, y = training_data
    predictor = Predictor()
    predictor.train(X, y)

    model_path = str(tmp_path / "model.joblib")
    predictor.save(model_path)

    loaded = Predictor.load(model_path)
    probs_orig = predictor.predict_proba(X.iloc[:5])
    probs_loaded = loaded.predict_proba(X.iloc[:5])
    np.testing.assert_array_almost_equal(probs_orig, probs_loaded)


def test_feature_importance(training_data):
    X, y = training_data
    predictor = Predictor()
    predictor.train(X, y)
    importance = predictor.feature_importance()
    assert len(importance) == 3
    assert all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in importance.items())


def test_create_labels():
    close = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0])
    threshold = 100.0
    labels = Predictor.create_threshold_labels(close, threshold, horizon=1)
    # label at index 0: is close[1] > 100? yes(101>100) => 1
    # label at index 1: is close[2] > 100? no(99<100) => 0
    # label at index 2: is close[3] > 100? yes(102>100) => 1
    # label at index 3: is close[4] > 100? no(98<100) => 0
    # label at index 4: NaN (no future data)
    assert list(labels[:4]) == [1, 0, 1, 0]
    assert pd.isna(labels.iloc[4])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/model/test_predictor.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Predictor**

```python
# src/polyquant/model/predictor.py
"""LightGBM probability predictor for price threshold prediction."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


class Predictor:
    """Binary classifier predicting P(price > threshold) using LightGBM."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
        }
        self.model: lgb.LGBMClassifier | None = None

    def train(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """Train the model on features X and binary labels y."""
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class (price above threshold)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        """Return feature name -> importance mapping."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        names = self.model.feature_name_
        importances = self.model.feature_importances_
        return dict(zip(names, importances))

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "Predictor":
        """Load a trained model from disk."""
        predictor = cls()
        predictor.model = joblib.load(path)
        return predictor

    @staticmethod
    def create_threshold_labels(
        close: pd.Series, threshold: float, horizon: int = 24,
    ) -> pd.Series:
        """Create binary labels: 1 if future close > threshold, 0 otherwise.

        Args:
            close: Close price series.
            threshold: Price threshold to predict against.
            horizon: Number of periods to look ahead.

        Returns:
            Series of 0/1 labels with NaN for rows without sufficient future data.
        """
        future_close = close.shift(-horizon)
        labels = (future_close > threshold).astype(float)
        labels[future_close.isna()] = np.nan
        return labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/model/test_predictor.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/model/predictor.py tests/model/test_predictor.py
git commit -m "feat: LightGBM predictor with threshold label creation"
```

---

## Chunk 3: Strategy Layer

### Task 8: Signal Generation

**Files:**
- Create: `src/polyquant/strategy/signal.py`
- Create: `tests/strategy/__init__.py`
- Create: `tests/strategy/test_signal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/test_signal.py
from polyquant.strategy.signal import generate_signal, Signal


def test_buy_yes_when_model_higher():
    # Model says 70% chance, market prices at 55% → edge = +15% > threshold(10%)
    signal = generate_signal(model_prob=0.70, market_price=0.55, threshold=0.10)
    assert signal == Signal.BUY_YES


def test_buy_no_when_model_lower():
    # Model says 30% chance, market prices at 55% → edge = -25% > threshold
    signal = generate_signal(model_prob=0.30, market_price=0.55, threshold=0.10)
    assert signal == Signal.BUY_NO


def test_no_signal_within_threshold():
    # Model says 60%, market at 55% → edge = 5% < threshold(10%)
    signal = generate_signal(model_prob=0.60, market_price=0.55, threshold=0.10)
    assert signal == Signal.NONE


def test_edge_exactly_at_threshold():
    # Edge exactly at threshold → no trade (strict inequality)
    signal = generate_signal(model_prob=0.65, market_price=0.55, threshold=0.10)
    assert signal == Signal.NONE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/strategy/test_signal.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement signal generation**

```python
# src/polyquant/strategy/signal.py
"""Signal generation: compare model probability vs Polymarket price."""

from enum import Enum


class Signal(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    NONE = "none"


def generate_signal(
    model_prob: float,
    market_price: float,
    threshold: float = 0.10,
) -> Signal:
    """Generate a trading signal by comparing model probability to market price.

    Args:
        model_prob: Model's predicted probability of YES outcome (0-1).
        market_price: Current YES token price on Polymarket (0-1).
        threshold: Minimum edge required to trade.

    Returns:
        Signal indicating BUY_YES, BUY_NO, or NONE.
    """
    edge = model_prob - market_price
    if edge > threshold:
        return Signal.BUY_YES
    elif edge < -threshold:
        return Signal.BUY_NO
    return Signal.NONE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/strategy/test_signal.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/strategy/ tests/strategy/
git commit -m "feat: signal generation comparing model prob vs market price"
```

---

### Task 9: Kelly Position Sizing

**Files:**
- Create: `src/polyquant/strategy/sizing.py`
- Create: `tests/strategy/test_sizing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/test_sizing.py
from polyquant.strategy.sizing import kelly_size


def test_kelly_positive_edge():
    # prob=0.7, market=0.5 → odds = 0.5/0.5 = 1.0
    # kelly = (0.7*1 - 0.3)/1 = 0.4, half_kelly = 0.2
    size = kelly_size(prob=0.7, market_price=0.5, capital=1000.0, kelly_fraction=0.5)
    assert abs(size - 200.0) < 1.0  # 0.2 * 1000


def test_kelly_caps_at_max_position():
    # Large edge, but capped at max_position_pct
    size = kelly_size(
        prob=0.95, market_price=0.1, capital=10000.0,
        kelly_fraction=0.5, max_position_pct=0.05,
    )
    assert size == 500.0  # 5% of 10000


def test_kelly_zero_for_no_edge():
    size = kelly_size(prob=0.5, market_price=0.5, capital=1000.0)
    assert size == 0.0


def test_kelly_zero_for_negative_edge():
    # prob < market → negative edge → no position
    size = kelly_size(prob=0.3, market_price=0.5, capital=1000.0)
    assert size == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/strategy/test_sizing.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Kelly sizing**

```python
# src/polyquant/strategy/sizing.py
"""Kelly criterion position sizing."""


def kelly_size(
    prob: float,
    market_price: float,
    capital: float,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.05,
) -> float:
    """Calculate position size using fractional Kelly criterion.

    Args:
        prob: Model's estimated probability of YES outcome.
        market_price: Current YES token price (also represents the cost).
        capital: Total available capital.
        kelly_fraction: Fraction of full Kelly to use (0.5 = half Kelly).
        max_position_pct: Maximum position as fraction of capital.

    Returns:
        Dollar amount to allocate to this position.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    # Odds: payout/cost. Buying YES at price p pays (1-p)/p if correct
    odds = (1 - market_price) / market_price
    q = 1 - prob

    # Full Kelly: f = (p*b - q) / b
    full_kelly = (prob * odds - q) / odds

    if full_kelly <= 0:
        return 0.0

    fraction = full_kelly * kelly_fraction
    max_fraction = max_position_pct
    fraction = min(fraction, max_fraction)

    return round(fraction * capital, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/strategy/test_sizing.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/strategy/sizing.py tests/strategy/test_sizing.py
git commit -m "feat: Kelly criterion position sizing"
```

---

## Chunk 4: Execution Layer

### Task 10: Backtesting Engine

**Files:**
- Create: `src/polyquant/execution/backtest.py`
- Create: `tests/execution/__init__.py`
- Create: `tests/execution/test_backtest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/execution/test_backtest.py
import numpy as np
import pandas as pd
import pytest
from polyquant.execution.backtest import run_model_backtest, BacktestResult


@pytest.fixture
def backtest_data():
    """Synthetic OHLCV data for backtesting (500 hourly rows)."""
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)),
        "low": prices - np.abs(np.random.randn(n)),
        "close": prices + np.random.randn(n) * 0.3,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


def test_backtest_returns_result(backtest_data):
    result = run_model_backtest(
        ohlcv=backtest_data,
        threshold=100.0,
        train_window=200,
        prediction_horizon=24,
    )
    assert isinstance(result, BacktestResult)
    assert isinstance(result.accuracy, float)
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.predictions) > 0


def test_backtest_predictions_have_expected_fields(backtest_data):
    result = run_model_backtest(
        ohlcv=backtest_data,
        threshold=100.0,
        train_window=200,
        prediction_horizon=24,
    )
    pred = result.predictions
    assert "predicted_prob" in pred.columns
    assert "actual_label" in pred.columns
    assert "timestamp" in pred.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/execution/test_backtest.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement backtesting engine**

```python
# src/polyquant/execution/backtest.py
"""Backtesting engine with rolling window training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polyquant.model.features import compute_features
from polyquant.model.predictor import Predictor


@dataclass
class BacktestResult:
    """Results from a model backtest run."""
    predictions: pd.DataFrame  # timestamp, predicted_prob, actual_label
    accuracy: float
    auc_roc: float | None
    brier_score: float


def run_model_backtest(
    ohlcv: pd.DataFrame,
    threshold: float,
    train_window: int = 200,
    prediction_horizon: int = 24,
    step_size: int = 24,
) -> BacktestResult:
    """Run rolling-window model backtest on OHLCV data.

    For each step:
    1. Train on [i - train_window : i]
    2. Predict at i
    3. Compare against actual outcome at i + prediction_horizon

    Args:
        ohlcv: OHLCV DataFrame.
        threshold: Price threshold for binary label.
        train_window: Number of rows for training window.
        prediction_horizon: Periods ahead to predict.
        step_size: How many rows to advance each step.

    Returns:
        BacktestResult with predictions and metrics.
    """
    features_df = compute_features(ohlcv)

    # Create labels
    labels = Predictor.create_threshold_labels(
        features_df["close"], threshold, horizon=prediction_horizon,
    )

    feature_cols = [c for c in features_df.columns
                    if c not in ("timestamp", "open", "high", "low", "close", "volume")]

    results = []
    start = train_window

    while start + prediction_horizon < len(features_df):
        train_X = features_df[feature_cols].iloc[start - train_window:start]
        train_y = labels.iloc[start - train_window:start]

        # Skip if train set has NaN labels
        valid_mask = ~train_y.isna()
        if valid_mask.sum() < 50:
            start += step_size
            continue

        predictor = Predictor()
        predictor.train(train_X[valid_mask], train_y[valid_mask].astype(int))

        # Predict current step
        pred_X = features_df[feature_cols].iloc[start:start + 1]
        actual = labels.iloc[start]

        if pd.isna(actual):
            start += step_size
            continue

        prob = predictor.predict_proba(pred_X)[0]
        results.append({
            "timestamp": features_df["timestamp"].iloc[start],
            "predicted_prob": prob,
            "actual_label": int(actual),
        })

        start += step_size

    pred_df = pd.DataFrame(results)

    if pred_df.empty:
        return BacktestResult(
            predictions=pred_df, accuracy=0.0, auc_roc=None, brier_score=1.0,
        )

    # Metrics
    predicted_labels = (pred_df["predicted_prob"] >= 0.5).astype(int)
    accuracy = (predicted_labels == pred_df["actual_label"]).mean()

    brier = ((pred_df["predicted_prob"] - pred_df["actual_label"]) ** 2).mean()

    auc = None
    from sklearn.metrics import roc_auc_score
    if pred_df["actual_label"].nunique() > 1:
        auc = roc_auc_score(pred_df["actual_label"], pred_df["predicted_prob"])

    return BacktestResult(
        predictions=pred_df,
        accuracy=float(accuracy),
        auc_roc=auc,
        brier_score=float(brier),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/execution/test_backtest.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/execution/ tests/execution/
git commit -m "feat: rolling window backtesting engine"
```

---

### Task 11: Paper Trading

**Files:**
- Create: `src/polyquant/execution/paper.py`
- Create: `tests/execution/test_paper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/execution/test_paper.py
from polyquant.execution.paper import PaperTrader
from polyquant.strategy.signal import Signal


def test_paper_trader_opens_position():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=200.0,
    )
    assert len(trader.positions) == 1
    assert trader.positions[0]["side"] == "yes"
    assert trader.positions[0]["size"] == 200.0


def test_paper_trader_tracks_capital():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=200.0,
    )
    assert trader.available_capital == 9800.0


def test_paper_trader_no_trade_on_none_signal():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.NONE,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=200.0,
    )
    assert len(trader.positions) == 0
    assert trader.available_capital == 10000.0


def test_paper_trader_resolve_winning():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=100.0,
    )
    # Resolve: YES wins → payout = size / entry_price (shares * $1)
    pnl = trader.resolve("btc-above-70000", outcome_yes=True)
    # Bought $100 worth at $0.55/share = 181.8 shares. Payout = 181.8 * $1 = $181.8. PnL = +$81.8
    assert pnl > 0
    assert trader.positions[0]["status"] == "resolved"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/execution/test_paper.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement PaperTrader**

```python
# src/polyquant/execution/paper.py
"""Paper trading: simulated order execution without real money."""

from datetime import datetime, timezone

from polyquant.strategy.signal import Signal


class PaperTrader:
    """Simulates trading by tracking virtual positions and P&L."""

    def __init__(self, capital: float) -> None:
        self.initial_capital = capital
        self.available_capital = capital
        self.positions: list[dict] = []
        self.trade_log: list[dict] = []

    def execute(
        self,
        signal: Signal,
        market_slug: str,
        token_id: str,
        market_price: float,
        size: float,
    ) -> None:
        """Execute a paper trade based on the signal.

        Args:
            signal: BUY_YES, BUY_NO, or NONE.
            market_slug: Market identifier.
            token_id: YES token ID.
            market_price: Current YES token price.
            size: Dollar amount to invest.
        """
        if signal == Signal.NONE:
            return

        if size > self.available_capital:
            size = self.available_capital

        if size <= 0:
            return

        side = "yes" if signal == Signal.BUY_YES else "no"
        entry_price = market_price if side == "yes" else (1 - market_price)

        position = {
            "market_slug": market_slug,
            "token_id": token_id,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "shares": size / entry_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "pnl": None,
        }
        self.positions.append(position)
        self.available_capital -= size

        self.trade_log.append({
            "action": "open",
            "time": position["entry_time"],
            **position,
        })

    def resolve(self, market_slug: str, outcome_yes: bool) -> float:
        """Resolve all open positions for a market.

        Args:
            market_slug: Market to resolve.
            outcome_yes: True if YES outcome won.

        Returns:
            Total P&L for resolved positions.
        """
        total_pnl = 0.0
        for pos in self.positions:
            if pos["market_slug"] == market_slug and pos["status"] == "open":
                won = (pos["side"] == "yes" and outcome_yes) or \
                      (pos["side"] == "no" and not outcome_yes)
                if won:
                    payout = pos["shares"] * 1.0  # Each winning share pays $1
                else:
                    payout = 0.0
                pos["pnl"] = payout - pos["size"]
                pos["status"] = "resolved"
                self.available_capital += payout
                total_pnl += pos["pnl"]
        return total_pnl

    @property
    def total_pnl(self) -> float:
        """Total realized P&L across all resolved positions."""
        return sum(p["pnl"] for p in self.positions if p["pnl"] is not None)

    @property
    def open_exposure(self) -> float:
        """Total dollar amount in open positions."""
        return sum(p["size"] for p in self.positions if p["status"] == "open")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/execution/test_paper.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/polyquant/execution/paper.py tests/execution/test_paper.py
git commit -m "feat: paper trading with position tracking and resolution"
```

---

### Task 12: Live Trading (Stub)

**Files:**
- Create: `src/polyquant/execution/live.py`

This is Phase 3 — implement as a stub with the interface defined but raises NotImplementedError. Will be filled in once backtesting and paper trading are validated.

- [ ] **Step 1: Create live trading stub**

```python
# src/polyquant/execution/live.py
"""Live trading via Polymarket CLOB API.

This module is Phase 3 — implement after backtesting and paper trading
are validated. Currently a stub defining the interface.
"""

from polyquant.strategy.signal import Signal


class LiveTrader:
    """Places real orders on Polymarket via CLOB API."""

    def __init__(self, private_key: str, api_key: str, api_secret: str, api_passphrase: str):
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        raise NotImplementedError(
            "LiveTrader is not yet implemented. Complete backtesting and paper "
            "trading validation first. See spec Phase 3."
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/polyquant/execution/live.py
git commit -m "feat: live trader stub (Phase 3 placeholder)"
```

---

## Chunk 5: Integration & CLI Entry Point

### Task 13: Data Collection Script

**Files:**
- Create: `src/polyquant/cli.py`

- [ ] **Step 1: Write the CLI entry point**

```python
# src/polyquant/cli.py
"""CLI entry points for data collection, backtesting, and paper trading."""

import argparse
import sys

import pandas as pd

from polyquant.config import Settings
from polyquant.data.binance import BinanceFetcher
from polyquant.data.polymarket import PolymarketFetcher
from polyquant.data.store import DataStore
from polyquant.execution.backtest import run_model_backtest
from polyquant.model.features import compute_features
from polyquant.model.predictor import Predictor
from polyquant.strategy.signal import generate_signal
from polyquant.strategy.sizing import kelly_size
from polyquant.execution.paper import PaperTrader


def collect_data(settings: Settings) -> None:
    """Fetch latest OHLCV and Polymarket price data, save to SQLite."""
    store = DataStore(settings.db_path)
    binance = BinanceFetcher()

    print("Fetching Binance OHLCV data...")
    for pair in settings.trading_pairs:
        df = binance.fetch_ohlcv(pair, settings.ohlcv_timeframe, limit=1000)
        store.save_ohlcv(pair, settings.ohlcv_timeframe, df)
        print(f"  {pair}: {len(df)} candles saved")

    print("Fetching Polymarket markets...")
    pm = PolymarketFetcher()
    markets = pm.get_crypto_markets()
    print(f"  Found {len(markets)} crypto markets")

    if markets:
        snapshots = pm.snapshot_prices(markets)
        if snapshots:
            store.save_polymarket_prices(pd.DataFrame(snapshots))
            print(f"  Saved {len(snapshots)} price snapshots")


def backtest(settings: Settings) -> None:
    """Run model backtest on stored OHLCV data."""
    store = DataStore(settings.db_path)

    for pair in settings.trading_pairs:
        ohlcv = store.load_ohlcv(pair, settings.ohlcv_timeframe)
        if ohlcv.empty:
            print(f"No data for {pair}. Run 'collect' first.")
            continue

        symbol = pair.split("/")[0]  # BTC or ETH
        current_price = ohlcv["close"].iloc[-1]
        threshold = current_price  # Predict: will price stay above current?

        print(f"\nBacktesting {pair} (threshold=${threshold:.0f})...")
        result = run_model_backtest(
            ohlcv=ohlcv,
            threshold=threshold,
            train_window=200,
            prediction_horizon=settings.prediction_horizon_hours,
        )
        print(f"  Predictions: {len(result.predictions)}")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Brier Score: {result.brier_score:.4f}")
        if result.auc_roc is not None:
            print(f"  AUC-ROC: {result.auc_roc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyQuant trading system")
    parser.add_argument("command", choices=["collect", "backtest", "paper"],
                        help="Command to run")
    args = parser.parse_args()

    settings = Settings()

    if args.command == "collect":
        collect_data(settings)
    elif args.command == "backtest":
        backtest(settings)
    elif args.command == "paper":
        paper_trade(settings)


def paper_trade(settings: Settings) -> None:
    """Run one cycle of paper trading: fetch data, predict, generate signals."""
    store = DataStore(settings.db_path)
    binance = BinanceFetcher()
    pm = PolymarketFetcher()
    trader = PaperTrader(capital=1000.0)

    # Fetch latest data
    for pair in settings.trading_pairs:
        df = binance.fetch_ohlcv(pair, settings.ohlcv_timeframe, limit=500)
        store.save_ohlcv(pair, settings.ohlcv_timeframe, df)

    # Get crypto markets
    markets = pm.get_crypto_markets()
    if not markets:
        print("No active crypto markets found.")
        return

    for market in markets:
        question = market.get("question", "")
        threshold = pm._extract_threshold(question)
        if threshold is None:
            continue

        # Determine which pair
        q_lower = question.lower()
        pair = "BTC/USDT" if "btc" in q_lower or "bitcoin" in q_lower else "ETH/USDT"
        ohlcv = store.load_ohlcv(pair, settings.ohlcv_timeframe)
        if len(ohlcv) < 200:
            continue

        # Compute features and predict
        features = compute_features(ohlcv)
        if features.empty:
            continue

        feature_cols = [c for c in features.columns
                        if c not in ("timestamp", "open", "high", "low", "close", "volume")]

        labels = Predictor.create_threshold_labels(
            features["close"], threshold, horizon=settings.prediction_horizon_hours,
        )
        valid = ~labels.isna()
        if valid.sum() < 100:
            continue

        predictor = Predictor()
        predictor.train(features[feature_cols][valid], labels[valid].astype(int))

        latest = features[feature_cols].iloc[-1:]
        prob = predictor.predict_proba(latest)[0]

        # Get market price
        tokens = market.get("tokens", [])
        yes_token = next((t for t in tokens if t["outcome"] == "Yes"), None)
        if not yes_token:
            continue
        market_price = pm.fetch_price(yes_token["token_id"])
        if market_price is None:
            continue

        signal = generate_signal(prob, market_price, settings.signal_threshold)
        size = kelly_size(prob, market_price, trader.available_capital,
                          settings.kelly_fraction, settings.max_position_pct)

        slug = market.get("market_slug", "unknown")
        print(f"\n{slug}: model={prob:.2%} market={market_price:.2%} signal={signal.value} size=${size:.2f}")

        trader.execute(signal, slug, yes_token["token_id"], market_price, size)

    print(f"\nOpen positions: {len([p for p in trader.positions if p['status'] == 'open'])}")
    print(f"Available capital: ${trader.available_capital:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add console_scripts entry point to pyproject.toml**

Add to pyproject.toml under `[project]`:
```toml
[project.scripts]
polyquant = "polyquant.cli:main"
```

- [ ] **Step 3: Reinstall package**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: Successful reinstall

- [ ] **Step 4: Commit**

```bash
git add src/polyquant/cli.py pyproject.toml
git commit -m "feat: CLI entry point for data collection and backtesting"
```

---

### Task 14: Smoke Test End-to-End

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify CLI help works**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && polyquant --help`
Expected: Shows usage with collect/backtest commands

- [ ] **Step 3: Test data collection with real APIs**

Run: `cd /Users/yang/workspace/polyquant && source .venv/bin/activate && polyquant collect`
Note: This will need a `.env` file with at least `POLYMARKET_PRIVATE_KEY` set (can be a dummy value for Binance-only collection; Polymarket collection will fail gracefully without valid credentials)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: PolyQuant MVP complete - data, model, strategy, backtest"
```
