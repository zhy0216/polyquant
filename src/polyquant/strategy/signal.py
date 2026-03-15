"""Signal generation: compare model probability vs Polymarket price."""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    NONE = "none"


def generate_signal(
    model_prob: float,
    market_price: float,
    threshold: float = 0.10,
) -> Signal:
    if model_prob < 0 or model_prob > 1:
        raise ValueError("model_prob must be in [0, 1]")
    if market_price < 0 or market_price > 1:
        raise ValueError("market_price must be in [0, 1]")

    edge = model_prob - market_price
    if edge > threshold:
        return Signal.BUY_YES
    elif edge < -threshold:
        return Signal.BUY_NO
    return Signal.NONE
