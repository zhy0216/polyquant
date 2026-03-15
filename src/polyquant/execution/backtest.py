"""Backtesting engine with rolling window training."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score

from polyquant.model.features import compute_features, get_feature_columns
from polyquant.model.predictor import Predictor

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a model backtest run."""
    predictions: pd.DataFrame
    accuracy: float
    auc_roc: float | None
    brier_score: float
    log_loss: float
    gross_pnl: float
    net_pnl: float
    total_fees: float


def run_model_backtest(
    ohlcv: pd.DataFrame,
    threshold: float,
    train_window: int = 720,
    prediction_horizon: int = 24,
    step_size: int = 24,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> BacktestResult:
    """Run rolling-window model backtest on OHLCV data."""
    if train_window < 100:
        raise ValueError("train_window must be >= 100")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    min_rows = train_window + 100
    if len(ohlcv) < min_rows:
        raise ValueError(
            f"ohlcv must have at least {min_rows} rows "
            f"(train_window={train_window} + 100 warmup), got {len(ohlcv)}"
        )
    logger.info("Starting backtest: threshold=%.2f, train_window=%d, horizon=%d, step=%d",
                threshold, train_window, prediction_horizon, step_size)
    features_df = compute_features(ohlcv)

    labels = Predictor.create_threshold_labels(
        features_df["close"], threshold, horizon=prediction_horizon,
    )

    feature_cols = get_feature_columns(features_df)

    results = []
    start = train_window
    iteration = 0

    while start + prediction_horizon < len(features_df):
        iteration += 1
        if iteration % 10 == 0:
            logger.info("Backtest iteration %d (row %d/%d)", iteration, start, len(features_df))
        train_X = features_df[feature_cols].iloc[start - train_window:start]
        train_y = labels.iloc[start - train_window:start]

        train_end = start - train_window + len(train_X)  # actual end of training slice
        if train_end > start:
            raise RuntimeError(
                f"Look-ahead bias: training data extends to row {train_end}, "
                f"but prediction point is row {start}"
            )

        valid_mask = ~train_y.isna()
        if valid_mask.sum() < 50:
            start += step_size
            continue

        predictor = Predictor()
        predictor.train(train_X[valid_mask], train_y[valid_mask].astype(int))

        pred_X = features_df[feature_cols].iloc[start:start + 1]
        if len(pred_X) != 1:
            raise RuntimeError(
                f"Expected exactly 1 prediction row, got {len(pred_X)} at start={start}"
            )
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
        logger.warning("Backtest produced no predictions")
        return BacktestResult(
            predictions=pred_df, accuracy=0.0, auc_roc=None, brier_score=1.0,
            log_loss=10.0, gross_pnl=0.0, net_pnl=0.0, total_fees=0.0,
        )

    predicted_labels = (pred_df["predicted_prob"] >= 0.5).astype(int)
    accuracy = (predicted_labels == pred_df["actual_label"]).mean()
    brier = ((pred_df["predicted_prob"] - pred_df["actual_label"]) ** 2).mean()

    auc = None
    if pred_df["actual_label"].nunique() > 1:
        auc = roc_auc_score(pred_df["actual_label"], pred_df["predicted_prob"])

    # Compute log_loss
    if pred_df["actual_label"].nunique() > 1:
        ll = float(sklearn_log_loss(pred_df["actual_label"], pred_df["predicted_prob"]))
    else:
        ll = 10.0

    # Simulate strategy P&L with costs
    bet_size = 100.0
    cost_per_trade = bet_size * (fee_rate + slippage_rate)
    gross_pnl = 0.0
    total_fees = 0.0

    for _, row in pred_df.iterrows():
        prob = row["predicted_prob"]
        actual = int(row["actual_label"])

        if prob >= 0.6:
            # Bet YES at entry_price = prob
            entry_price = prob
            if actual == 1:
                payout = bet_size / entry_price - bet_size
            else:
                payout = -bet_size
            gross_pnl += payout
            total_fees += cost_per_trade
        elif prob <= 0.4:
            # Bet NO at entry_price = prob (NO price = 1 - prob)
            entry_price = prob
            if actual == 0:
                payout = bet_size / (1 - entry_price) - bet_size
            else:
                payout = -bet_size
            gross_pnl += payout
            total_fees += cost_per_trade
        # else: no trade

    net_pnl = gross_pnl - total_fees

    logger.info("Backtest complete: %d predictions, accuracy=%.2f%%", len(pred_df), accuracy * 100)

    return BacktestResult(
        predictions=pred_df,
        accuracy=float(accuracy),
        auc_roc=auc,
        brier_score=float(brier),
        log_loss=ll,
        gross_pnl=float(gross_pnl),
        net_pnl=float(net_pnl),
        total_fees=float(total_fees),
    )
