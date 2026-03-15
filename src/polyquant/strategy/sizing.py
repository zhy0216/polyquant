"""Kelly criterion position sizing."""

import logging

logger = logging.getLogger(__name__)


def kelly_size(
    prob: float,
    market_price: float,
    capital: float,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 1.0,
) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0

    odds = (1 - market_price) / market_price
    q = 1 - prob

    full_kelly = (prob * odds - q) / odds

    if full_kelly <= 0:
        return 0.0

    fraction = full_kelly * kelly_fraction
    max_fraction = max_position_pct
    fraction = min(fraction, max_fraction)

    return round(fraction * capital, 2)
