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

    def train(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, early_stopping: bool = False,
    ) -> None:
        """Train the model on features X and binary labels y.

        Args:
            X: Feature matrix.
            y: Binary labels.
            early_stopping: When True and len(X) > 100, hold out the last 20%
                as a chronological validation set and stop training when
                validation loss stops improving for 20 rounds.
        """
        logger.info("Training model on %d samples with %d features", len(X), X.shape[1])
        self.model = lgb.LGBMClassifier(**self.params)

        if early_stopping and len(X) > 100:
            split = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[:split], y.iloc[split:]
            else:
                y_train, y_val = y[:split], y[split:]

            # Both splits must contain both classes for LightGBM's label encoder
            y_train_arr = np.asarray(y_train)
            y_val_arr = np.asarray(y_val)
            if len(np.unique(y_train_arr)) < 2 or len(np.unique(y_val_arr)) < 2:
                logger.info("Early stopping skipped: insufficient class diversity in split")
                self.model.fit(X, y)
            else:
                logger.info("Early stopping enabled: train=%d, val=%d", len(X_train), len(X_val))
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                )
        else:
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
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        future_close = close.shift(-horizon)
        labels = (future_close > threshold).astype(float)
        labels[future_close.isna()] = np.nan
        return labels
