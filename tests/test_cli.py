"""CLI integration tests — all network calls are mocked."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from polyquant.cli import collect_data, backtest, paper_trade, main
from polyquant.config import Settings
from polyquant.data.store import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)),
        "low": prices - np.abs(np.random.randn(n)),
        "close": prices + np.random.randn(n) * 0.3,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


def _settings(tmp_path) -> Settings:
    return Settings(db_path=str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("polyquant.cli.PolymarketFetcher")
@patch("polyquant.cli.BinanceFetcher")
def test_collect_data_success(mock_binance_cls, mock_pm_cls, tmp_path, capsys):
    """collect_data() runs without error when fetchers return valid data."""
    mock_binance = mock_binance_cls.return_value
    mock_binance.fetch_ohlcv.return_value = _make_ohlcv(10)

    mock_pm = mock_pm_cls.return_value
    mock_pm.get_crypto_markets.return_value = [{"question": "BTC above 100k?"}]
    mock_pm.snapshot_prices.return_value = [
        {
            "timestamp": pd.Timestamp.now(),
            "market_slug": "btc-100k",
            "token_id": "0xabc",
            "yes_price": 0.55,
            "no_price": 0.45,
        }
    ]

    settings = _settings(tmp_path)
    collect_data(settings)

    captured = capsys.readouterr().out
    assert "Fetching Binance OHLCV data" in captured
    assert "10 candles saved" in captured
    assert "Fetching Polymarket markets" in captured
    assert "1 crypto markets" in captured
    assert "1 price snapshots" in captured


def test_backtest_with_data(tmp_path, capsys):
    """backtest() prints results when the DB has enough OHLCV rows."""
    settings = Settings(db_path=str(tmp_path / "test.db"), train_window=200)
    store = DataStore(settings.db_path)
    ohlcv = _make_ohlcv(1200)
    store.save_ohlcv("BTC/USDT", "1h", ohlcv)
    store.save_ohlcv("ETH/USDT", "1h", ohlcv)

    backtest(settings)

    captured = capsys.readouterr().out
    assert "Backtesting" in captured
    assert "Predictions:" in captured
    assert "Accuracy:" in captured
    assert "Brier Score:" in captured


def test_backtest_no_data(tmp_path, capsys):
    """backtest() prints 'No data' when the DB is empty."""
    settings = _settings(tmp_path)
    # Create the store (empty tables) but don't insert any data
    DataStore(settings.db_path)

    backtest(settings)

    captured = capsys.readouterr().out
    assert "No data" in captured


@patch("polyquant.cli.BinanceFetcher")
@patch("polyquant.cli.PolymarketFetcher")
def test_paper_trade_no_markets(mock_pm_cls, mock_binance_cls, tmp_path, capsys):
    """paper_trade() prints message when no crypto markets are found."""
    mock_binance = mock_binance_cls.return_value
    mock_binance.fetch_ohlcv.return_value = _make_ohlcv(10)

    mock_pm = mock_pm_cls.return_value
    mock_pm.get_crypto_markets.return_value = []

    settings = _settings(tmp_path)
    paper_trade(settings)

    captured = capsys.readouterr().out
    assert "No active crypto markets found." in captured


@patch("polyquant.cli.PolymarketFetcher")
@patch("polyquant.cli.BinanceFetcher")
def test_collect_data_network_error(mock_binance_cls, mock_pm_cls, tmp_path, capsys):
    """collect_data() catches network errors via Phase 3 error handling."""
    import ccxt

    mock_binance = mock_binance_cls.return_value
    mock_binance.fetch_ohlcv.side_effect = ccxt.NetworkError("connection reset")

    settings = _settings(tmp_path)
    with pytest.raises(ccxt.NetworkError):
        collect_data(settings)

    captured = capsys.readouterr().out
    assert "Error: data collection failed" in captured


@patch("polyquant.cli.setup_logging")
def test_main_invalid_command(mock_logging):
    """main() exits with SystemExit for an invalid command."""
    with patch("sys.argv", ["polyquant", "invalid"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2  # argparse exits with code 2


@patch("polyquant.cli.setup_logging")
@patch("polyquant.cli.collect_data", side_effect=KeyboardInterrupt)
def test_main_keyboard_interrupt(mock_collect, mock_logging):
    """main() exits with code 130 on KeyboardInterrupt."""
    with patch("sys.argv", ["polyquant", "collect"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 130


@patch("polyquant.cli.setup_logging")
@patch("polyquant.cli.collect_data", side_effect=RuntimeError("boom"))
def test_main_exception_handling(mock_collect, mock_logging):
    """main() exits with code 1 on unhandled exception."""
    with patch("sys.argv", ["polyquant", "collect"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
