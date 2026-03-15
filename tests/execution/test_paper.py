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
