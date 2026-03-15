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
