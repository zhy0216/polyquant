"""Live trading via Polymarket CLOB API.

This module is Phase 3 — implement after backtesting and paper trading
are validated. Currently a stub defining the interface.
"""

import logging

from polyquant.strategy.signal import Signal

logger = logging.getLogger(__name__)


class LiveTrader:
    """Places real orders on Polymarket via CLOB API."""

    def __init__(self, private_key: str, api_key: str, api_secret: str, api_passphrase: str):
        self.private_key = private_key
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        raise NotImplementedError(
            "LiveTrader is not yet implemented. Complete backtesting and paper "
            "trading validation first. See spec Phase 3."
        )
