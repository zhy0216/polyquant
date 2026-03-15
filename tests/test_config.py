from polyquant.config import Settings

import pytest
from pydantic import ValidationError


def test_default_settings():
    settings = Settings(
        polymarket_private_key="0xabc123",
    )
    assert settings.trading_pairs == ["BTC/USDT", "ETH/USDT"]
    assert settings.signal_threshold == 0.10
    assert settings.kelly_fraction == 0.5
    assert settings.max_position_pct == 0.05
    assert settings.max_exposure_pct == 0.30
    assert settings.prediction_horizon_hours == 24
    assert settings.ohlcv_timeframe == "1h"
    assert settings.train_window == 720
    assert settings.ohlcv_limit == 8760


def test_settings_override():
    settings = Settings(
        polymarket_private_key="0xabc123",
        signal_threshold=0.15,
        kelly_fraction=0.25,
    )
    assert settings.signal_threshold == 0.15
    assert settings.kelly_fraction == 0.25


def test_settings_default_empty_key():
    settings = Settings()
    assert settings.polymarket_private_key == ""


def test_signal_threshold_invalid():
    with pytest.raises(ValidationError):
        Settings(signal_threshold=0)
    with pytest.raises(ValidationError):
        Settings(signal_threshold=-0.1)
    with pytest.raises(ValidationError):
        Settings(signal_threshold=1.5)


def test_kelly_fraction_invalid():
    with pytest.raises(ValidationError):
        Settings(kelly_fraction=0)
    with pytest.raises(ValidationError):
        Settings(kelly_fraction=1.1)


def test_max_position_pct_invalid():
    with pytest.raises(ValidationError):
        Settings(max_position_pct=0)
    with pytest.raises(ValidationError):
        Settings(max_position_pct=1.5)


def test_max_exposure_pct_invalid():
    with pytest.raises(ValidationError):
        Settings(max_exposure_pct=0)
    with pytest.raises(ValidationError):
        Settings(max_exposure_pct=1.5)


def test_prediction_horizon_hours_invalid():
    with pytest.raises(ValidationError):
        Settings(prediction_horizon_hours=0)
    with pytest.raises(ValidationError):
        Settings(prediction_horizon_hours=-1)


def test_train_window_invalid():
    with pytest.raises(ValidationError):
        Settings(train_window=99)


def test_ohlcv_limit_invalid():
    with pytest.raises(ValidationError):
        Settings(ohlcv_limit=199)
