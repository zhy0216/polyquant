from polyquant.strategy.sizing import kelly_size


def test_kelly_positive_edge():
    size = kelly_size(prob=0.7, market_price=0.5, capital=1000.0, kelly_fraction=0.5)
    assert abs(size - 200.0) < 1.0


def test_kelly_caps_at_max_position():
    size = kelly_size(
        prob=0.95, market_price=0.1, capital=10000.0,
        kelly_fraction=0.5, max_position_pct=0.05,
    )
    assert size == 500.0


def test_kelly_zero_for_no_edge():
    size = kelly_size(prob=0.5, market_price=0.5, capital=1000.0)
    assert size == 0.0


def test_kelly_zero_for_negative_edge():
    size = kelly_size(prob=0.3, market_price=0.5, capital=1000.0)
    assert size == 0.0
