"""LightGBM probability predictor for price threshold prediction."""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        logger.info("Training model on %d samples with %d features", len(X), X.shape[1])
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        logger.info("Training complete")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class (price above threshold)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        probs = self.model.predict_proba(X)[:, 1]
        logger.debug("Generated %d predictions", len(probs))
        return probs

    def feature_importance(self) -> dict[str, float]:
        """Return feature name -> importance mapping."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        names = self.model.feature_name_
        importances = self.model.feature_importances_
        return {k: float(v) for k, v in zip(names, importances)}

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
        """Create binary labels: 1 if future close > threshold, 0 otherwise."""
        future_close = close.shift(-horizon)
        labels = (future_close > threshold).astype(float)
        labels[future_close.isna()] = np.nan
        return labels
