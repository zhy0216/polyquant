"""Paper trading: simulated order execution without real money."""

import logging
from datetime import datetime, timezone

from polyquant.strategy.signal import Signal

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates trading by tracking virtual positions and P&L."""

    def __init__(self, capital: float, max_exposure_pct: float = 1.0) -> None:
        if not 0.0 < max_exposure_pct <= 1.0:
            raise ValueError(f"max_exposure_pct must be in (0, 1.0], got {max_exposure_pct}")
        self.initial_capital = capital
        self.available_capital = capital
        self.max_exposure_pct = max_exposure_pct
        self.positions: list[dict] = []
        self.trade_log: list[dict] = []

    def execute(
        self,
        signal: Signal,
        market_slug: str,
        token_id: str,
        market_price: float,
        size: float,
    ) -> None:
        if signal == Signal.NONE:
            return

        max_exposure = self.initial_capital * self.max_exposure_pct
        remaining_exposure = max_exposure - self.open_exposure
        if remaining_exposure <= 0:
            logger.warning("Max exposure (%.0f%%) reached, rejecting trade on %s",
                           self.max_exposure_pct * 100, market_slug)
            return
        size = min(size, remaining_exposure)

        if size > self.available_capital:
            size = self.available_capital
        if size <= 0:
            return

        side = "yes" if signal == Signal.BUY_YES else "no"
        entry_price = market_price if side == "yes" else (1 - market_price)

        if entry_price <= 0:
            logger.warning("Invalid entry price %.4f for %s, skipping trade", entry_price, market_slug)
            return

        position = {
            "market_slug": market_slug,
            "token_id": token_id,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "shares": size / entry_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "pnl": None,
        }
        self.positions.append(position)
        self.available_capital -= size
        logger.info("Opened %s position on %s: size=$%.2f, entry_price=%.4f",
                     side, market_slug, size, entry_price)

        self.trade_log.append({
            "action": "open",
            "time": position["entry_time"],
            **position,
        })

    def resolve(self, market_slug: str, outcome_yes: bool) -> float:
        total_pnl = 0.0
        for pos in self.positions:
            if pos["market_slug"] == market_slug and pos["status"] == "open":
                won = (pos["side"] == "yes" and outcome_yes) or \
                      (pos["side"] == "no" and not outcome_yes)
                if won:
                    payout = pos["shares"] * 1.0
                else:
                    payout = 0.0
                pos["pnl"] = payout - pos["size"]
                pos["status"] = "resolved"
                self.available_capital += payout
                total_pnl += pos["pnl"]
        logger.info("Resolved market %s: outcome_yes=%s, pnl=$%.2f",
                     market_slug, outcome_yes, total_pnl)
        return total_pnl

    @property
    def total_pnl(self) -> float:
        return sum(p["pnl"] for p in self.positions if p["pnl"] is not None)

    @property
    def open_exposure(self) -> float:
        return sum(p["size"] for p in self.positions if p["status"] == "open")
