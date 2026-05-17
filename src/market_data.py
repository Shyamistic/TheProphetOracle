"""Kalshi Market Price Fetcher.

Fetches current market prices from Kalshi's public API (no authentication required).
Used when the hackathon input format does not include market_stats — we fetch
prices ourselves to enable market-anchored predictions.

Key insight from hackathon organizer: "Only make prediction when you are confident
enough, otherwise just use the market probability as your prediction."
"""

import httpx
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


async def fetch_kalshi_market_price(market_ticker: str) -> Optional[Dict[str, float]]:
    """Fetch current market prices from Kalshi's public API.

    No authentication required. Returns outcome -> probability mapping.

    Args:
        market_ticker: The Kalshi market ticker (e.g., "KXHIGHNY-24JAN01-T60")

    Returns:
        Dict mapping outcome labels to market prices (0-1 scale), or None if fetch fails.
    """
    if not market_ticker or market_ticker == "UNKNOWN":
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{KALSHI_BASE_URL}/markets/{market_ticker}"
            )
            if response.status_code == 200:
                data = response.json()
                market = data.get("market", {})

                # Kalshi returns prices in cents (0-100 scale)
                yes_price = market.get("last_price")
                if yes_price is not None:
                    yes_prob = yes_price / 100.0
                    return {"Yes": yes_prob, "No": 1.0 - yes_prob}

                # Try yes_ask/no_ask as fallback
                yes_ask = market.get("yes_ask")
                if yes_ask is not None:
                    yes_prob = yes_ask / 100.0
                    return {"Yes": yes_prob, "No": 1.0 - yes_prob}

            logger.debug(
                f"Kalshi API returned status {response.status_code} for {market_ticker}"
            )
            return None

    except httpx.TimeoutException:
        logger.debug(f"Timeout fetching Kalshi price for {market_ticker}")
        return None
    except Exception as e:
        logger.debug(f"Could not fetch Kalshi price for {market_ticker}: {e}")
        return None


async def fetch_kalshi_event_markets(event_ticker: str) -> Optional[Dict[str, float]]:
    """Fetch market prices for all markets in a Kalshi event.

    Useful when the event has multiple outcomes (not just Yes/No).
    Uses the /events/{event_ticker} endpoint which returns nested markets.

    Args:
        event_ticker: The Kalshi event ticker (e.g., "KXHIGHNY")

    Returns:
        Dict mapping outcome labels to market prices (0-1 scale), or None if fetch fails.
    """
    if not event_ticker or event_ticker == "UNKNOWN":
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{KALSHI_BASE_URL}/events/{event_ticker}",
                params={"with_nested_markets": True},
            )
            if response.status_code == 200:
                data = response.json()
                event = data.get("event", {})
                markets = event.get("markets", [])

                if not markets:
                    return None

                prices = {}
                for mkt in markets:
                    # Use subtitle or title as outcome label
                    label = mkt.get("subtitle") or mkt.get("title", "")
                    last_price = mkt.get("last_price")
                    if label and last_price is not None:
                        prices[label] = last_price / 100.0

                return prices if prices else None

            return None

    except httpx.TimeoutException:
        logger.debug(f"Timeout fetching Kalshi event markets for {event_ticker}")
        return None
    except Exception as e:
        logger.debug(f"Could not fetch Kalshi event markets for {event_ticker}: {e}")
        return None


async def get_market_prices(
    market_ticker: str,
    event_ticker: str,
    outcomes: List[str],
) -> Optional[Dict[str, float]]:
    """Try multiple strategies to get market prices from Kalshi.

    Strategy order:
    1. Direct market ticker lookup (for Yes/No markets)
    2. Event ticker lookup with nested markets (for multi-outcome)
    3. Return None if all fail

    Args:
        market_ticker: The Kalshi market ticker.
        event_ticker: The Kalshi event ticker.
        outcomes: List of outcome labels for the event.

    Returns:
        Dict mapping outcome labels to market prices, or None.
    """
    # Strategy 1: Direct market ticker (works for binary Yes/No markets)
    prices = await fetch_kalshi_market_price(market_ticker)
    if prices:
        # Check if the outcomes match what we got
        if _outcomes_match(prices, outcomes):
            return prices
        # If outcomes don't match but we have Yes/No and event has 2 outcomes,
        # map them to the actual outcome labels
        if len(outcomes) == 2 and "Yes" in prices:
            return {outcomes[0]: prices["Yes"], outcomes[1]: prices["No"]}

    # Strategy 2: Event-level lookup for multi-outcome markets
    event_prices = await fetch_kalshi_event_markets(event_ticker)
    if event_prices:
        # Try to match event market labels to our outcomes
        matched = _match_outcomes(event_prices, outcomes)
        if matched:
            return matched

    return None


def _outcomes_match(prices: Dict[str, float], outcomes: List[str]) -> bool:
    """Check if price dict keys match the expected outcomes."""
    return set(prices.keys()) == set(outcomes)


def _match_outcomes(
    prices: Dict[str, float], outcomes: List[str]
) -> Optional[Dict[str, float]]:
    """Try to match fetched market labels to expected outcome labels.

    Uses case-insensitive substring matching as a fallback.
    """
    # Exact match
    if set(prices.keys()) == set(outcomes):
        return prices

    # Case-insensitive exact match
    prices_lower = {k.lower(): v for k, v in prices.items()}
    matched = {}
    for outcome in outcomes:
        if outcome.lower() in prices_lower:
            matched[outcome] = prices_lower[outcome.lower()]

    if len(matched) == len(outcomes):
        return matched

    # Substring match (e.g., "Pittsburgh Steelers" matches "Pittsburgh")
    matched = {}
    for outcome in outcomes:
        for label, price in prices.items():
            if (
                outcome.lower() in label.lower()
                or label.lower() in outcome.lower()
            ):
                matched[outcome] = price
                break

    if len(matched) == len(outcomes):
        return matched

    return None
