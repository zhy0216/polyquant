from polyquant.execution.paper import PaperTrader
from polyquant.strategy.signal import Signal


def test_paper_trader_opens_position():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=200.0,
    )
    assert len(trader.positions) == 1
    assert trader.positions[0]["side"] == "yes"
    assert trader.positions[0]["size"] == 200.0


def test_paper_trader_tracks_capital():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=200.0,
    )
    assert trader.available_capital == 9800.0


def test_paper_trader_no_trade_on_none_signal():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.NONE,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=200.0,
    )
    assert len(trader.positions) == 0
    assert trader.available_capital == 10000.0


def test_paper_trader_resolve_winning():
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=100.0,
    )
    pnl = trader.resolve("btc-above-70000", outcome_yes=True)
    assert pnl > 0
    assert trader.positions[0]["status"] == "resolved"


def test_paper_trader_skips_zero_entry_price_buy_yes():
    """BUY_YES with market_price=0 gives entry_price=0, should skip."""
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.0,
        size=200.0,
    )
    assert len(trader.positions) == 0
    assert trader.available_capital == 10000.0


def test_paper_trader_skips_zero_entry_price_buy_no():
    """BUY_NO with market_price=1.0 gives entry_price=0, should skip."""
    trader = PaperTrader(capital=10000.0)
    trader.execute(
        signal=Signal.BUY_NO,
        market_slug="btc-above-70000",
        token_id="0xno",
        market_price=1.0,
        size=200.0,
    )
    assert len(trader.positions) == 0
    assert trader.available_capital == 10000.0


def test_paper_trader_rejects_trade_exceeding_max_exposure():
    """Trades that would push total exposure above max_exposure_pct are rejected."""
    trader = PaperTrader(capital=10000.0, max_exposure_pct=0.10)  # 10% = $1000 max
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=800.0,
    )
    assert len(trader.positions) == 1  # First trade fits

    # Second trade would push exposure to $1600 > $1000 limit
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-80000",
        token_id="0xyes2",
        market_price=0.60,
        size=800.0,
    )
    # Should be capped: only $200 more allowed ($1000 - $800 = $200)
    assert len(trader.positions) == 2
    assert trader.positions[1]["size"] == 200.0


def test_paper_trader_blocks_trade_at_max_exposure():
    """Trades are fully blocked when already at max exposure."""
    trader = PaperTrader(capital=10000.0, max_exposure_pct=0.05)  # 5% = $500 max
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-70000",
        token_id="0xyes",
        market_price=0.55,
        size=500.0,
    )
    assert len(trader.positions) == 1

    # Already at limit, should reject entirely
    trader.execute(
        signal=Signal.BUY_YES,
        market_slug="btc-above-80000",
        token_id="0xyes2",
        market_price=0.60,
        size=100.0,
    )
    assert len(trader.positions) == 1  # No new position


def test_paper_trader_default_max_exposure_is_one():
    """Default max_exposure_pct=1.0 (no effective limit)."""
    trader = PaperTrader(capital=1000.0)
    assert trader.max_exposure_pct == 1.0
