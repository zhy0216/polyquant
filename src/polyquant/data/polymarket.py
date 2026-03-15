"""Polymarket market discovery and price snapshot fetching."""

import logging
import re
from datetime import datetime, timezone

import requests

from polyquant.utils import retry_with_backoff

logger = logging.getLogger(__name__)


GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

CRYPTO_KEYWORDS = ["btc", "bitcoin", "eth", "ethereum"]


class PolymarketFetcher:
    """Discovers crypto markets and fetches price snapshots from Polymarket."""

    def __init__(self) -> None:
        self.session = requests.Session()

    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(requests.RequestException,))
    def fetch_active_markets(self) -> list[dict]:
        """Fetch all active markets from Gamma API."""
        resp = self.session.get(
            f"{GAMMA_API_URL}/markets",
            params={"active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def get_crypto_markets(self) -> list[dict]:
        """Fetch and filter to crypto price threshold markets."""
        all_markets = self.fetch_active_markets()
        filtered = self._filter_crypto_markets(all_markets)
        logger.info("Found %d crypto markets out of %d total", len(filtered), len(all_markets))
        return filtered

    def _filter_crypto_markets(self, markets: list[dict]) -> list[dict]:
        """Filter markets to BTC/ETH price threshold markets."""
        result = []
        for m in markets:
            question = m.get("question", "").lower()
            if any(kw in question for kw in CRYPTO_KEYWORDS):
                if self._extract_threshold(m.get("question", "")) is not None:
                    result.append(m)
        return result

    def _extract_threshold(self, question: str) -> float | None:
        """Extract dollar threshold from market question like 'above $70,000'."""
        match = re.search(r"\$[\d,]+", question)
        if not match:
            return None
        price_str = match.group().replace("$", "").replace(",", "")
        try:
            return float(price_str)
        except ValueError:
            return None

    def fetch_price(self, token_id: str) -> float | None:
        """Fetch current midpoint price for a token from CLOB API."""
        try:
            resp = self.session.get(f"{CLOB_API_URL}/midpoint", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except (requests.RequestException, ValueError, KeyError):
            logger.warning("Failed to fetch price for token %s", token_id)
            return None

    def _build_price_row(self, market: dict, prices: dict) -> dict | None:
        """Build a price snapshot row from market info and fetched prices."""
        tokens = market["tokens"]
        try:
            yes_token = next(t for t in tokens if t["outcome"] == "Yes")
            no_token = next(t for t in tokens if t["outcome"] == "No")
        except StopIteration:
            logger.warning("Missing Yes/No tokens for market %s", market.get("market_slug", "unknown"))
            return None
        return {
            "timestamp": datetime.now(timezone.utc),
            "market_slug": market["market_slug"],
            "token_id": yes_token["token_id"],
            "yes_price": prices.get(yes_token["token_id"], 0.0),
            "no_price": prices.get(no_token["token_id"], 0.0),
        }

    def snapshot_prices(self, markets: list[dict]) -> list[dict]:
        """Fetch current prices for a list of markets and return snapshot rows.

        Only includes a market when BOTH YES and NO token prices are available.
        """
        rows = []
        for market in markets:
            tokens = market.get("tokens", [])
            try:
                yes_token = next(t for t in tokens if t["outcome"] == "Yes")
                no_token = next(t for t in tokens if t["outcome"] == "No")
            except StopIteration:
                logger.warning("Missing Yes/No tokens for market %s", market.get("market_slug", "unknown"))
                continue

            yes_price = self.fetch_price(yes_token["token_id"])
            no_price = self.fetch_price(no_token["token_id"])

            if yes_price is None or no_price is None:
                logger.warning(
                    "Partial price failure for market %s (yes=%s, no=%s), skipping",
                    market.get("market_slug", "unknown"), yes_price, no_price,
                )
                continue

            row = {
                "timestamp": datetime.now(timezone.utc),
                "market_slug": market["market_slug"],
                "token_id": yes_token["token_id"],
                "yes_price": yes_price,
                "no_price": no_price,
            }
            rows.append(row)
        return rows
