"""Technical indicator feature engineering from OHLCV data."""

import logging

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].

    Returns:
        DataFrame with feature columns, NaN warmup rows dropped.
    """
    out = df.copy()
    if len(df) < 100:
        raise ValueError(f"Need at least 100 rows for feature computation, got {len(df)}")
    logger.info("Computing features from %d input rows", len(df))

    # Trend
    out["sma_7"] = ta.trend.sma_indicator(out["close"], window=7)
    out["sma_25"] = ta.trend.sma_indicator(out["close"], window=25)
    out["sma_99"] = ta.trend.sma_indicator(out["close"], window=99)
    out["ema_12"] = ta.trend.ema_indicator(out["close"], window=12)
    out["ema_26"] = ta.trend.ema_indicator(out["close"], window=26)
    macd_indicator = ta.trend.MACD(out["close"])
    out["macd"] = macd_indicator.macd()
    out["macd_signal"] = macd_indicator.macd_signal()

    # Momentum
    out["rsi_14"] = ta.momentum.rsi(out["close"], window=14)
    stoch = ta.momentum.StochRSIIndicator(out["close"])
    out["stoch_rsi"] = stoch.stochrsi()

    # Volatility
    bb = ta.volatility.BollingerBands(out["close"])
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_width"] = bb.bollinger_wband()
    out["atr_14"] = ta.volatility.average_true_range(out["high"], out["low"], out["close"], window=14)

    # Volume
    out["volume_sma_20"] = ta.trend.sma_indicator(out["volume"], window=20)
    out["volume_ratio"] = out["volume"] / out["volume_sma_20"]
    out["obv"] = ta.volume.on_balance_volume(out["close"], out["volume"])

    # Price derivatives
    out["return_1h"] = out["close"].pct_change(1)
    out["return_4h"] = out["close"].pct_change(4)
    out["return_24h"] = out["close"].pct_change(24)
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    # Historical volatility (rolling std of returns)
    out["hist_vol_24"] = out["return_1h"].rolling(24).std()

    # Drop warmup rows (99 needed for SMA_99)
    rows_before = len(out)
    out = out.dropna().reset_index(drop=True)
    logger.info("Dropped %d warmup rows, %d rows remaining", rows_before - len(out), len(out))

    return out


_OHLCV_COLS = frozenset(("timestamp", "open", "high", "low", "close", "volume"))


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes raw OHLCV columns)."""
    return [c for c in df.columns if c not in _OHLCV_COLS]
