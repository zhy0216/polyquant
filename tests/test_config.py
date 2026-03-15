from polyquant.config import Settings


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
    assert settings.data_fetch_interval_minutes == 15
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
