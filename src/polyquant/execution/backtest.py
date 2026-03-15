"""Backtesting engine with rolling window training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from polyquant.model.features import compute_features
from polyquant.model.predictor import Predictor


@dataclass
class BacktestResult:
    """Results from a model backtest run."""
    predictions: pd.DataFrame
    accuracy: float
    auc_roc: float | None
    brier_score: float


def run_model_backtest(
    ohlcv: pd.DataFrame,
    threshold: float,
    train_window: int = 720,
    prediction_horizon: int = 24,
    step_size: int = 24,
) -> BacktestResult:
    """Run rolling-window model backtest on OHLCV data."""
    features_df = compute_features(ohlcv)

    labels = Predictor.create_threshold_labels(
        features_df["close"], threshold, horizon=prediction_horizon,
    )

    feature_cols = [c for c in features_df.columns
                    if c not in ("timestamp", "open", "high", "low", "close", "volume")]

    results = []
    start = train_window

    while start + prediction_horizon < len(features_df):
        train_X = features_df[feature_cols].iloc[start - train_window:start]
        train_y = labels.iloc[start - train_window:start]

        valid_mask = ~train_y.isna()
        if valid_mask.sum() < 50:
            start += step_size
            continue

        predictor = Predictor()
        predictor.train(train_X[valid_mask], train_y[valid_mask].astype(int))

        pred_X = features_df[feature_cols].iloc[start:start + 1]
        actual = labels.iloc[start]

        if pd.isna(actual):
            start += step_size
            continue

        prob = predictor.predict_proba(pred_X)[0]
        results.append({
            "timestamp": features_df["timestamp"].iloc[start],
            "predicted_prob": prob,
            "actual_label": int(actual),
        })

        start += step_size

    pred_df = pd.DataFrame(results)

    if pred_df.empty:
        return BacktestResult(
            predictions=pred_df, accuracy=0.0, auc_roc=None, brier_score=1.0,
        )

    predicted_labels = (pred_df["predicted_prob"] >= 0.5).astype(int)
    accuracy = (predicted_labels == pred_df["actual_label"]).mean()
    brier = ((pred_df["predicted_prob"] - pred_df["actual_label"]) ** 2).mean()

    auc = None
    if pred_df["actual_label"].nunique() > 1:
        auc = roc_auc_score(pred_df["actual_label"], pred_df["predicted_prob"])

    return BacktestResult(
        predictions=pred_df,
        accuracy=float(accuracy),
        auc_roc=auc,
        brier_score=float(brier),
    )
