"""CLI entry points for data collection, backtesting, and paper trading."""

import argparse
import logging
import sys

import pandas as pd

from polyquant.config import Settings
from polyquant.data.binance import BinanceFetcher
from polyquant.data.polymarket import PolymarketFetcher
from polyquant.data.store import DataStore
from polyquant.execution.backtest import run_model_backtest
from polyquant.logging_config import setup_logging
from polyquant.model.features import compute_features
from polyquant.model.predictor import Predictor
from polyquant.strategy.signal import generate_signal
from polyquant.strategy.sizing import kelly_size
from polyquant.execution.paper import PaperTrader

logger = logging.getLogger(__name__)


def collect_data(settings: Settings) -> None:
    """Fetch latest OHLCV and Polymarket price data, save to SQLite."""
    store = DataStore(settings.db_path)
    binance = BinanceFetcher()

    print("Fetching Binance OHLCV data...")
    for pair in settings.trading_pairs:
        df = binance.fetch_ohlcv(pair, settings.ohlcv_timeframe, limit=settings.ohlcv_limit)
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

        symbol = pair.split("/")[0]
        current_price = ohlcv["close"].iloc[-1]
        threshold = current_price

        print(f"\nBacktesting {pair} (threshold=${threshold:.0f})...")
        result = run_model_backtest(
            ohlcv=ohlcv,
            threshold=threshold,
            train_window=settings.train_window,
            prediction_horizon=settings.prediction_horizon_hours,
        )
        print(f"  Predictions: {len(result.predictions)}")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Brier Score: {result.brier_score:.4f}")
        if result.auc_roc is not None:
            print(f"  AUC-ROC: {result.auc_roc:.4f}")


def paper_trade(settings: Settings) -> None:
    """Run one cycle of paper trading: fetch data, predict, generate signals."""
    store = DataStore(settings.db_path)
    binance = BinanceFetcher()
    pm = PolymarketFetcher()
    trader = PaperTrader(capital=1000.0)

    for pair in settings.trading_pairs:
        df = binance.fetch_ohlcv(pair, settings.ohlcv_timeframe, limit=settings.ohlcv_limit)
        store.save_ohlcv(pair, settings.ohlcv_timeframe, df)

    markets = pm.get_crypto_markets()
    if not markets:
        print("No active crypto markets found.")
        return

    for market in markets:
        question = market.get("question", "")
        threshold = pm._extract_threshold(question)
        if threshold is None:
            continue

        q_lower = question.lower()
        pair = "BTC/USDT" if "btc" in q_lower or "bitcoin" in q_lower else "ETH/USDT"
        ohlcv = store.load_ohlcv(pair, settings.ohlcv_timeframe)
        if len(ohlcv) < settings.train_window:
            continue

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


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyQuant trading system")
    parser.add_argument("command", choices=["collect", "backtest", "paper"],
                        help="Command to run")
    args = parser.parse_args()

    setup_logging()

    settings = Settings()
    logger.info("Starting command: %s", args.command)

    if args.command == "collect":
        collect_data(settings)
    elif args.command == "backtest":
        backtest(settings)
    elif args.command == "paper":
        paper_trade(settings)

    logger.info("Command %s finished", args.command)


if __name__ == "__main__":
    main()
