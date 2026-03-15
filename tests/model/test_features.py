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


def test_compute_features_too_few_rows():
    np.random.seed(42)
    n = 50
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)),
        "low": prices - np.abs(np.random.randn(n)),
        "close": prices + np.random.randn(n) * 0.3,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })
    with pytest.raises(ValueError, match="Need at least 100 rows"):
        compute_features(df)
