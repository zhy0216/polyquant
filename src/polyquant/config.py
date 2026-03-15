"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


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
    signal_threshold: float = 0.10
    kelly_fraction: float = 0.5
    max_position_pct: float = 0.05
    max_exposure_pct: float = 0.30

    # Model parameters
    prediction_horizon_hours: int = 24
    ohlcv_timeframe: str = "1h"

    # Scheduling
    data_fetch_interval_minutes: int = 15

    # Paths
    db_path: str = "data/polyquant.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
