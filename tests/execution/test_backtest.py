import numpy as np
import pandas as pd
import pytest
from polyquant.execution.backtest import run_model_backtest, BacktestResult


@pytest.fixture
def backtest_data():
    """Synthetic OHLCV data for backtesting (1200 hourly rows)."""
    np.random.seed(42)
    n = 1200
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


def test_backtest_invalid_train_window(backtest_data):
    with pytest.raises(ValueError, match="train_window must be >= 100"):
        run_model_backtest(ohlcv=backtest_data, threshold=100.0, train_window=50)


def test_backtest_invalid_step_size(backtest_data):
    with pytest.raises(ValueError, match="step_size must be positive"):
        run_model_backtest(ohlcv=backtest_data, threshold=100.0, step_size=0)
    with pytest.raises(ValueError, match="step_size must be positive"):
        run_model_backtest(ohlcv=backtest_data, threshold=100.0, step_size=-1)


def test_backtest_invalid_prediction_horizon(backtest_data):
    with pytest.raises(ValueError, match="prediction_horizon must be positive"):
        run_model_backtest(ohlcv=backtest_data, threshold=100.0, prediction_horizon=0)
    with pytest.raises(ValueError, match="prediction_horizon must be positive"):
        run_model_backtest(ohlcv=backtest_data, threshold=100.0, prediction_horizon=-1)


def test_backtest_insufficient_ohlcv_rows():
    """ohlcv must have at least train_window + 100 rows."""
    np.random.seed(42)
    n = 250  # less than 200 + 100 = 300
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    small_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)),
        "low": prices - np.abs(np.random.randn(n)),
        "close": prices + np.random.randn(n) * 0.3,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })
    with pytest.raises(ValueError, match="ohlcv must have at least"):
        run_model_backtest(ohlcv=small_df, threshold=100.0, train_window=200)


def test_backtest_boundary_train_window(backtest_data):
    """train_window=100 (boundary) should be accepted."""
    result = run_model_backtest(
        ohlcv=backtest_data,
        threshold=100.0,
        train_window=100,
        prediction_horizon=24,
    )
    assert isinstance(result, BacktestResult)


def test_backtest_result_has_strategy_metrics(backtest_data):
    """BacktestResult should include log_loss and strategy P&L with costs."""
    result = run_model_backtest(
        ohlcv=backtest_data,
        threshold=100.0,
        train_window=200,
        prediction_horizon=24,
        fee_rate=0.02,
        slippage_rate=0.01,
    )
    assert isinstance(result, BacktestResult)
    assert hasattr(result, "log_loss")
    assert hasattr(result, "net_pnl")
    assert hasattr(result, "gross_pnl")
    assert hasattr(result, "total_fees")
    assert result.total_fees >= 0
    assert result.net_pnl <= result.gross_pnl


def test_backtest_with_early_stopping(backtest_data):
    result = run_model_backtest(
        ohlcv=backtest_data,
        threshold=100.0,
        train_window=200,
        prediction_horizon=24,
        early_stopping=True,
    )
    assert isinstance(result, BacktestResult)
    assert len(result.predictions) > 0


def test_backtest_zero_fees_equal_gross(backtest_data):
    """With zero fees, net_pnl equals gross_pnl."""
    result = run_model_backtest(
        ohlcv=backtest_data,
        threshold=100.0,
        train_window=200,
        prediction_horizon=24,
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    assert abs(result.net_pnl - result.gross_pnl) < 1e-6
