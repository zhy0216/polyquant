# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/strategy/test_signal.py -v

# Run a specific test
pytest tests/strategy/test_signal.py::test_buy_yes_signal -v
```

Pytest is configured in `pyproject.toml`: testpaths=`tests/`, pythonpath=`src/`, asyncio_mode=`auto`.

**CLI entry point** (`polyquant = polyquant.cli:main`):
```bash
polyquant collect    # Fetch Binance OHLCV + Polymarket prices → SQLite
polyquant backtest   # Rolling-window model backtest with metrics
polyquant paper      # One paper trading cycle: fetch → train → predict → trade
```

## Architecture

Four-layer pipeline for trading Polymarket BTC/ETH price threshold markets:

```
Data Layer → Model Layer → Strategy Layer → Execution Layer
```

**Data** (`src/polyquant/data/`): `BinanceFetcher` (CCXT OHLCV), `PolymarketFetcher` (Gamma API market discovery + CLOB prices), `DataStore` (SQLite persistence with three tables: ohlcv, polymarket_prices, positions).

**Model** (`src/polyquant/model/`): `compute_features()` generates ~20 technical indicators (SMA, RSI, MACD, Bollinger, ATR, OBV, returns) from OHLCV using the `ta` library — drops first 99 rows for warmup. `Predictor` wraps LightGBM binary classifier predicting P(price > threshold in 24h).

**Strategy** (`src/polyquant/strategy/`): `generate_signal()` compares model probability vs Polymarket price, requires 10% edge. `kelly_size()` implements half-Kelly position sizing capped at 5% of capital.

**Execution** (`src/polyquant/execution/`): `run_model_backtest()` uses rolling window (train 200 candles, step 24). `PaperTrader` tracks virtual positions with P&L. `LiveTrader` is a Phase 3 stub (raises NotImplementedError).

**Config** (`src/polyquant/config.py`): Pydantic `Settings` class loading from `.env` / environment variables. All parameters have defaults — no required env vars for backtesting.

## Key Conventions

- Python >=3.11, build system is Hatchling
- Source layout: `src/polyquant/` (not flat)
- Tests mirror source structure under `tests/`
- Data stored in `data/polyquant.db` (SQLite, gitignored)
- Models saved as `.joblib` files in `models/` (gitignored)
- Polymarket credentials go in `.env` (see `.env.example`)
