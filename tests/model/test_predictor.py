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
    assert list(labels[:4]) == [1, 0, 1, 0]
    assert pd.isna(labels.iloc[4])


def test_create_labels_invalid_horizon():
    close = pd.Series([100.0, 101.0, 99.0])
    with pytest.raises(ValueError, match="horizon must be positive"):
        Predictor.create_threshold_labels(close, threshold=100.0, horizon=0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        Predictor.create_threshold_labels(close, threshold=100.0, horizon=-1)


def test_create_labels_invalid_threshold():
    close = pd.Series([100.0, 101.0, 99.0])
    with pytest.raises(ValueError, match="threshold must be positive"):
        Predictor.create_threshold_labels(close, threshold=0, horizon=1)
    with pytest.raises(ValueError, match="threshold must be positive"):
        Predictor.create_threshold_labels(close, threshold=-10, horizon=1)
