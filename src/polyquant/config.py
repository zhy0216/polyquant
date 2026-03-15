"""Configuration management using pydantic-settings."""

import logging

from pydantic_settings import BaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    # Polymarket credentials
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""

    # Trading pairs (Binance symbols)
    trading_pairs: list[str] = Field(default=["BTC/USDT", "ETH/USDT"])

    # Strategy parameters
    signal_threshold: float = Field(default=0.10, gt=0, le=1)
    kelly_fraction: float = Field(default=0.5, gt=0, le=1)
    max_position_pct: float = Field(default=0.05, gt=0, le=1)
    max_exposure_pct: float = Field(default=0.30, gt=0, le=1)

    # Model parameters
    prediction_horizon_hours: int = Field(default=24, gt=0)
    ohlcv_timeframe: str = "1h"
    train_window: int = Field(default=720, ge=100)
    ohlcv_limit: int = Field(default=8760, ge=200)

    # Scheduling
    data_fetch_interval_minutes: int = 15

    # Paths
    db_path: str = "data/polyquant.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
