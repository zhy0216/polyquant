from polyquant.strategy.signal import generate_signal, Signal

import pytest


def test_buy_yes_when_model_higher():
    signal = generate_signal(model_prob=0.70, market_price=0.55, threshold=0.10)
    assert signal == Signal.BUY_YES


def test_buy_no_when_model_lower():
    signal = generate_signal(model_prob=0.30, market_price=0.55, threshold=0.10)
    assert signal == Signal.BUY_NO


def test_no_signal_within_threshold():
    signal = generate_signal(model_prob=0.60, market_price=0.55, threshold=0.10)
    assert signal == Signal.NONE


def test_edge_exactly_at_threshold():
    signal = generate_signal(model_prob=0.65, market_price=0.55, threshold=0.10)
    assert signal == Signal.NONE


def test_invalid_model_prob():
    with pytest.raises(ValueError, match="model_prob"):
        generate_signal(model_prob=-0.1, market_price=0.5)
    with pytest.raises(ValueError, match="model_prob"):
        generate_signal(model_prob=1.1, market_price=0.5)


def test_invalid_market_price():
    with pytest.raises(ValueError, match="market_price"):
        generate_signal(model_prob=0.5, market_price=-0.1)
    with pytest.raises(ValueError, match="market_price"):
        generate_signal(model_prob=0.5, market_price=1.1)


def test_boundary_model_prob():
    # Boundary values 0 and 1 should be accepted
    assert generate_signal(model_prob=0.0, market_price=0.5) is not None
    assert generate_signal(model_prob=1.0, market_price=0.5) is not None


def test_boundary_market_price():
    assert generate_signal(model_prob=0.5, market_price=0.0) is not None
    assert generate_signal(model_prob=0.5, market_price=1.0) is not None
