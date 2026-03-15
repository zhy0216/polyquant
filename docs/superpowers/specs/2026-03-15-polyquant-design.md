# PolyQuant: Polymarket BTC/ETH Automated Trading System

## Overview

A Python-based quantitative trading system that uses probability models to predict BTC/ETH price movements, compares predictions against Polymarket market prices, and automatically trades when significant mispricings are detected.

## Goals

- Build a model-based trading pipeline for BTC/ETH price prediction markets on Polymarket
- Start with backtesting, progress to paper trading, then live trading
- MVP-first approach: minimal data sources, simple model, get the pipeline working end-to-end

## Non-Goals

- Multi-strategy framework (single pipeline for now)
- Multiple data source fusion (news, sentiment, on-chain data)
- High-frequency trading or market making
- Support for non-crypto Polymarket markets

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐
│  Data Layer  │───▶│ Model Layer  │───▶│ Signal Layer  │───▶│ Execution  │
│              │    │              │    │               │    │   Layer    │
│ - Binance    │    │ - Feature    │    │ - Compare     │    │ - Backtest │
│   OHLCV      │    │   Engineering│    │   model prob  │    │   Engine   │
│ - Polymarket │    │ - LightGBM   │    │   vs market   │    │ - Paper    │
│   prices     │    │   Predictor  │    │ - Kelly       │    │   Trading  │
│              │    │              │    │   sizing      │    │ - Live     │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────┘
```

## Project Structure

```
polyquant/
├── pyproject.toml
├── src/
│   └── polyquant/
│       ├── data/
│       │   ├── binance.py       # Binance OHLCV data fetching via ccxt
│       │   ├── polymarket.py    # Polymarket market data via py-clob-client
│       │   └── store.py         # SQLite local storage
│       ├── model/
│       │   ├── features.py      # Technical indicator feature engineering
│       │   └── predictor.py     # LightGBM probability model
│       ├── strategy/
│       │   ├── signal.py        # Signal generation (model vs market)
│       │   └── sizing.py        # Kelly criterion position sizing
│       ├── execution/
│       │   ├── backtest.py      # Backtesting engine
│       │   ├── paper.py         # Paper trading (simulated)
│       │   └── live.py          # Live trading via Polymarket CLOB API
│       └── config.py            # Configuration management
├── tests/
├── notebooks/                   # Analysis notebooks
└── data/                        # Local data directory
```

## Component Details

### Data Layer

**Binance Data** (`data/binance.py`):
- Uses `ccxt` library to fetch BTC/USDT and ETH/USDT candlestick data
- Supports multiple timeframes: 1h, 4h, 1d
- Stores to local SQLite to avoid redundant API calls
- Fields: open, high, low, close, volume, timestamp

**Polymarket Data** (`data/polymarket.py`):
- Uses `py-clob-client` to fetch current prices for BTC/ETH prediction markets
- Periodically snapshots market prices (for backtesting)
- Discovers relevant market `token_id`s via the Gamma API

**Data Store** (`data/store.py`):
- SQLite as local storage
- Two main tables: `ohlcv` (candlesticks) and `polymarket_prices` (market snapshots)
- Returns pandas DataFrames via simple query interface

### Model Layer

**Feature Engineering** (`model/features.py`):
- Trend indicators: SMA(7, 25, 99), EMA(12, 26), MACD
- Momentum indicators: RSI(14), Stochastic RSI
- Volatility: Bollinger Bands, ATR(14), historical volatility
- Volume: Volume MA ratio, OBV
- Price derivatives: returns, log returns, multi-period returns (1h, 4h, 24h)
- Uses `ta` library for calculation

**Prediction Model** (`model/predictor.py`):
- Algorithm: LightGBM (gradient boosting)
- Target: binary classification — price up (1) vs down (0) over next N hours
- Output: probability value (0.0 ~ 1.0)
- Training: rolling window to prevent look-ahead bias
- Evaluation metrics: AUC-ROC, Brier Score, log loss
- Model persistence via `joblib`

### Signal Layer

**Signal Generation** (`strategy/signal.py`):
- Compares model predicted probability against Polymarket market price
- Trade condition: `|model_prob - market_price| > threshold` (default: 0.10)
- model_prob > market_price + threshold → BUY YES token
- model_prob < market_price - threshold → BUY NO token
- Threshold is configurable, starts conservatively high

**Position Sizing** (`strategy/sizing.py`):
- Kelly Criterion: `f = (p * b - q) / b` where p = model prob, q = 1-p, b = odds
- Uses half-Kelly (Kelly / 2) for conservatism
- Single position cap: 5% of total capital
- Total exposure cap: 30% of total capital

### Execution Layer

**Backtesting** (`execution/backtest.py`):
- Input: historical features + historical Polymarket price snapshots
- Simulates rolling window: train on past data, predict current, generate signals
- Metrics: total return, Sharpe ratio, max drawdown, win rate
- Outputs report for notebook visualization

**Paper Trading** (`execution/paper.py`):
- Connects to live data feeds, runs model and strategy in real-time
- Records virtual positions and P&L without placing real orders
- Validates strategy behavior in live environment

**Live Trading** (`execution/live.py`):
- Places orders via `py-clob-client` using Polymarket CLOB API
- Uses limit orders (recommended over market orders)
- Risk controls: daily loss limit, abnormal price protection, network error retry
- Logs every operation

### Configuration

**Config Management** (`config.py`):
- `.env` file for sensitive data (API keys, private key)
- Python dataclass or pydantic for strategy parameters
- Parameters: threshold, Kelly fraction, max position size, trading pairs, timeframes

## Dependencies

| Package | Purpose |
|---------|---------|
| `ccxt` | Unified crypto exchange API (Binance data) |
| `py-clob-client` | Polymarket official Python SDK |
| `lightgbm` | Gradient boosting model |
| `pandas` | Data manipulation |
| `ta` | Technical analysis indicators |
| `joblib` | Model serialization |
| `python-dotenv` | Environment variable management |
| `pydantic` | Configuration validation |
| `pytest` | Testing |

## Deployment Approach

1. **Phase 1 (MVP)**: Data collection + feature engineering + model training + backtesting
2. **Phase 2**: Paper trading with live data
3. **Phase 3**: Live trading with conservative parameters

## Risk Management

- Start with half-Kelly sizing
- Single position capped at 5% of capital
- Total exposure capped at 30%
- Daily loss limit triggers shutdown
- All trades logged for post-mortem analysis
