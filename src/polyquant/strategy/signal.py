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
    edge = model_prob - market_price
    if edge > threshold:
        return Signal.BUY_YES
    elif edge < -threshold:
        return Signal.BUY_NO
    return Signal.NONE
