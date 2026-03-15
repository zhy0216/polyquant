from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests
from polyquant.data.polymarket import PolymarketFetcher


def test_parse_markets_filters_crypto():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    raw_markets = [
        {
            "condition_id": "0x1",
            "question": "Will BTC be above $70,000 on March 20?",
            "market_slug": "will-btc-above-70000-march-20",
            "tokens": [
                {"token_id": "0xyes1", "outcome": "Yes"},
                {"token_id": "0xno1", "outcome": "No"},
            ],
            "end_date_iso": "2026-03-20T00:00:00Z",
            "active": True,
        },
        {
            "condition_id": "0x2",
            "question": "Will the election result in X?",
            "market_slug": "will-election-result",
            "tokens": [
                {"token_id": "0xyes2", "outcome": "Yes"},
                {"token_id": "0xno2", "outcome": "No"},
            ],
            "end_date_iso": "2026-11-05T00:00:00Z",
            "active": True,
        },
    ]
    crypto_markets = fetcher._filter_crypto_markets(raw_markets)
    assert len(crypto_markets) == 1
    assert crypto_markets[0]["market_slug"] == "will-btc-above-70000-march-20"


def test_extract_threshold_from_question():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    assert fetcher._extract_threshold("Will BTC be above $70,000 on March 20?") == 70000.0
    assert fetcher._extract_threshold("Will ETH be above $4,500 on April 1?") == 4500.0
    assert fetcher._extract_threshold("No price here") is None


def test_build_price_snapshot():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    market = {
        "market_slug": "will-btc-above-70000",
        "tokens": [
            {"token_id": "0xyes1", "outcome": "Yes"},
            {"token_id": "0xno1", "outcome": "No"},
        ],
    }
    prices = {"0xyes1": 0.55, "0xno1": 0.45}
    row = fetcher._build_price_row(market, prices)
    assert row["yes_price"] == 0.55
    assert row["no_price"] == 0.45
    assert row["token_id"] == "0xyes1"


def test_fetch_price_returns_none_on_request_exception():
    fetcher = PolymarketFetcher()
    fetcher.session = MagicMock()
    fetcher.session.get.side_effect = requests.RequestException("timeout")
    result = fetcher.fetch_price("0xtoken")
    assert result is None


def test_build_price_row_returns_none_on_missing_tokens():
    fetcher = PolymarketFetcher.__new__(PolymarketFetcher)
    market = {
        "market_slug": "test-market",
        "tokens": [{"token_id": "0x1", "outcome": "Maybe"}],
    }
    result = fetcher._build_price_row(market, {"0x1": 0.5})
    assert result is None
