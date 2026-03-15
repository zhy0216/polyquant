from polyquant.strategy.signal import generate_signal, Signal


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
