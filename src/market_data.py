"""Market Price Fetcher (Kalshi + Polymarket).

Fetches current market prices from Kalshi and Polymarket public APIs (no auth required).
Used when the hackathon input format does not include market_stats — we fetch
prices ourselves to enable market-anchored predictions.

Cross-references both sources when available and averages for a stronger prior.

Key insight from hackathon organizer: "Only make prediction when you are confident
enough, otherwise just use the market probability as your prediction."
"""

import re
import httpx
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"


async def fetch_kalshi_market_price(market_ticker: str) -> Optional[Dict[str, float]]:
    """Fetch current market prices from Kalshi's public API.

    No authentication required. Returns outcome -> probability mapping.
    Handles both dollar-denominated and cent-denominated price fields.

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

                # Try dollar-denominated price first (string like "0.7200")
                last_price_dollars = market.get("last_price_dollars")
                if last_price_dollars and last_price_dollars != "0.0000":
                    try:
                        yes_prob = float(last_price_dollars)
                        if 0.0 < yes_prob < 1.0:
                            return {"Yes": yes_prob, "No": 1.0 - yes_prob}
                    except (ValueError, TypeError):
                        pass

                # Fallback to cent-denominated (integer 0-100)
                yes_price = market.get("last_price")
                if yes_price is not None and yes_price > 0:
                    yes_prob = yes_price / 100.0
                    return {"Yes": yes_prob, "No": 1.0 - yes_prob}

                # Try yes_ask as fallback
                yes_ask = market.get("yes_ask_dollars")
                if yes_ask and yes_ask != "0.0000":
                    try:
                        yes_prob = float(yes_ask)
                        if 0.0 < yes_prob < 1.0:
                            return {"Yes": yes_prob, "No": 1.0 - yes_prob}
                    except (ValueError, TypeError):
                        pass

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

    Uses the /events/{event_ticker} endpoint with nested markets.
    Handles both dollar-denominated prices (last_price_dollars) and
    cent-denominated prices (last_price).
    
    For threshold events (gas prices, TSA, etc.), uses yes_sub_title
    as the outcome label.

    Args:
        event_ticker: The Kalshi event ticker (e.g., "KXAAAGASD-26MAY21")

    Returns:
        Dict mapping outcome labels to market prices (0-1 scale), or None if fetch fails.
    """
    if not event_ticker or event_ticker == "UNKNOWN":
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{KALSHI_BASE_URL}/events/{event_ticker}",
                params={"with_nested_markets": "true"},
            )
            if response.status_code == 200:
                data = response.json()
                event = data.get("event", {})
                markets = event.get("markets", [])

                if not markets:
                    return None

                prices = {}
                for mkt in markets:
                    # Skip settled/closed markets with no useful price
                    status = mkt.get("status", "")
                    
                    # Get the label — try yes_sub_title first (most reliable for threshold events)
                    label = (
                        mkt.get("yes_sub_title") or 
                        mkt.get("subtitle") or 
                        mkt.get("title") or 
                        ""
                    )
                    
                    # If no label, extract from ticker (e.g., KXAAAGASD-26MAY21-4.560 → "Above 4.560")
                    if not label:
                        ticker = mkt.get("ticker", "")
                        parts = ticker.split("-")
                        if len(parts) >= 3:
                            threshold_part = parts[-1]
                            try:
                                float(threshold_part)
                                label = f"Above {threshold_part}"
                            except ValueError:
                                pass
                    
                    if not label:
                        continue
                    
                    # Get price — try multiple fields
                    # Priority: last_price_dollars > last_price (cents) > yes_bid_dollars
                    price = None
                    
                    # Dollar-denominated (string like "0.9800")
                    last_price_dollars = mkt.get("last_price_dollars")
                    if last_price_dollars and last_price_dollars != "0.0000":
                        try:
                            price = float(last_price_dollars)
                        except (ValueError, TypeError):
                            pass
                    
                    # Cent-denominated (integer like 98)
                    if price is None:
                        last_price_cents = mkt.get("last_price")
                        if last_price_cents is not None and last_price_cents > 0:
                            try:
                                price = int(last_price_cents) / 100.0
                            except (ValueError, TypeError):
                                pass
                    
                    # Fallback to yes_bid_dollars
                    if price is None:
                        yes_bid = mkt.get("yes_bid_dollars")
                        if yes_bid and yes_bid != "0.0000":
                            try:
                                price = float(yes_bid)
                            except (ValueError, TypeError):
                                pass
                    
                    # For closed/settled markets, check result
                    if price is None and status == "closed":
                        result = mkt.get("result", "")
                        if result == "yes":
                            price = 0.99
                        elif result == "no":
                            price = 0.01
                    
                    if label and price is not None and 0.0 <= price <= 1.0:
                        prices[label] = price

                if prices:
                    logger.info(
                        f"Kalshi event {event_ticker}: fetched {len(prices)} market prices "
                        f"(sample: {list(prices.items())[:3]})"
                    )
                return prices if prices else None

            elif response.status_code == 404:
                logger.debug(f"Kalshi event {event_ticker} not found (404)")
            else:
                logger.debug(f"Kalshi API returned {response.status_code} for event {event_ticker}")
            return None

    except httpx.TimeoutException:
        logger.debug(f"Timeout fetching Kalshi event markets for {event_ticker}")
        return None
    except Exception as e:
        logger.debug(f"Could not fetch Kalshi event markets for {event_ticker}: {e}")
        return None


async def fetch_polymarket_price(title: str) -> Optional[Dict[str, float]]:
    """Fetch market prices from Polymarket's public Gamma API by searching for a matching market.

    Searches Polymarket for a market matching the event title and returns
    outcome -> probability mapping if found.

    Args:
        title: The event title to search for on Polymarket.

    Returns:
        Dict mapping outcome labels to probabilities (0-1 scale), or None if not found.
    """
    if not title or not title.strip():
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Search Polymarket gamma API with the event title
            response = await client.get(
                POLYMARKET_GAMMA_URL,
                params={
                    "closed": "false",
                    "limit": 5,
                    "active": "true",
                    "ascending": "false",
                    "order": "liquidity",
                    "tag": "",
                    "slug": "",
                },
            )
            if response.status_code != 200:
                logger.debug(f"Polymarket API returned status {response.status_code}")
                return None

            markets = response.json()
            if not isinstance(markets, list) or not markets:
                return None

            # Find the best matching market by title similarity
            title_lower = title.lower()
            best_match = None
            best_score = 0.0

            for market in markets:
                market_question = (market.get("question") or market.get("title") or "").lower()
                if not market_question:
                    continue

                # Simple word overlap scoring
                title_words = set(title_lower.split())
                market_words = set(market_question.split())
                if not title_words:
                    continue

                overlap = len(title_words & market_words)
                score = overlap / max(len(title_words), 1)

                if score > best_score and score >= 0.4:  # At least 40% word overlap
                    best_score = score
                    best_match = market

            if not best_match:
                logger.debug(f"No Polymarket match found for: {title[:60]}")
                return None

            # Extract probabilities from the matched market
            # Polymarket stores outcome prices in different formats
            outcomes_data = best_match.get("outcomePrices")
            if outcomes_data:
                # outcomePrices is typically a JSON string like "[\"0.72\", \"0.28\"]"
                if isinstance(outcomes_data, str):
                    import json
                    try:
                        prices_list = json.loads(outcomes_data)
                    except (json.JSONDecodeError, TypeError):
                        prices_list = None
                elif isinstance(outcomes_data, list):
                    prices_list = outcomes_data
                else:
                    prices_list = None

                if prices_list and len(prices_list) >= 2:
                    try:
                        yes_prob = float(prices_list[0])
                        no_prob = float(prices_list[1])
                        logger.debug(
                            f"Polymarket match for '{title[:40]}': "
                            f"Yes={yes_prob:.3f}, No={no_prob:.3f} "
                            f"(match score: {best_score:.2f})"
                        )
                        return {"Yes": yes_prob, "No": no_prob}
                    except (TypeError, ValueError, IndexError):
                        pass

            # Fallback: try bestBid/bestAsk or other price fields
            best_bid = best_match.get("bestBid")
            if best_bid is not None:
                try:
                    yes_prob = float(best_bid)
                    if 0.0 < yes_prob < 1.0:
                        return {"Yes": yes_prob, "No": 1.0 - yes_prob}
                except (TypeError, ValueError):
                    pass

            return None

    except httpx.TimeoutException:
        logger.debug(f"Timeout fetching Polymarket price for: {title[:40]}")
        return None
    except Exception as e:
        logger.debug(f"Could not fetch Polymarket price for '{title[:40]}': {e}")
        return None


async def get_market_prices(
    market_ticker: str,
    event_ticker: str,
    outcomes: List[str],
    title: str = "",
) -> Optional[Dict[str, float]]:
    """Try multiple strategies to get market prices from Kalshi AND Polymarket.

    Cross-references both sources when available:
    - If both return prices, averages them for a stronger prior
    - If only one returns, uses that one
    - Logs which sources were used

    Strategy order:
    1. Direct market ticker lookup on Kalshi (for Yes/No markets)
    2. Event ticker lookup with nested markets on Kalshi (for multi-outcome)
    3. Polymarket title search (for binary markets)
    4. Average if both sources available

    Args:
        market_ticker: The Kalshi market ticker.
        event_ticker: The Kalshi event ticker.
        outcomes: List of outcome labels for the event.
        title: Event title for Polymarket search.

    Returns:
        Dict mapping outcome labels to market prices, or None.
    """
    sources_used = []

    # --- Kalshi ---
    kalshi_prices = None

    # Strategy 1: Direct market ticker (works for binary Yes/No markets)
    prices = await fetch_kalshi_market_price(market_ticker)
    if prices:
        # Check if the outcomes match what we got
        if _outcomes_match(prices, outcomes):
            kalshi_prices = prices
        # If outcomes don't match but we have Yes/No and event has 2 outcomes,
        # map them to the actual outcome labels
        elif len(outcomes) == 2 and "Yes" in prices:
            kalshi_prices = {outcomes[0]: prices["Yes"], outcomes[1]: prices["No"]}

    # Strategy 2: Event-level lookup for multi-outcome markets
    if not kalshi_prices:
        event_prices = await fetch_kalshi_event_markets(event_ticker)
        if event_prices:
            # Try to match event market labels to our outcomes
            matched = _match_outcomes(event_prices, outcomes)
            if matched:
                kalshi_prices = matched

    if kalshi_prices:
        sources_used.append("Kalshi")

    # --- Polymarket (only for binary markets with a title) ---
    polymarket_prices = None
    if title and len(outcomes) == 2:
        poly_raw = await fetch_polymarket_price(title)
        if poly_raw and "Yes" in poly_raw:
            # Map Yes/No to actual outcome labels
            polymarket_prices = {outcomes[0]: poly_raw["Yes"], outcomes[1]: poly_raw["No"]}
            sources_used.append("Polymarket")

    # --- Cross-reference: average if both available ---
    if kalshi_prices and polymarket_prices:
        averaged = {}
        for outcome in outcomes:
            k_price = kalshi_prices.get(outcome, 0.5)
            p_price = polymarket_prices.get(outcome, 0.5)
            averaged[outcome] = (k_price + p_price) / 2.0
        logger.info(
            f"Market prices from {' + '.join(sources_used)} (averaged): "
            f"{{{', '.join(f'{k}: {v:.3f}' for k, v in averaged.items())}}}"
        )
        return averaged
    elif kalshi_prices:
        logger.info(f"Market prices from Kalshi only: {kalshi_prices}")
        return kalshi_prices
    elif polymarket_prices:
        logger.info(f"Market prices from Polymarket only: {polymarket_prices}")
        return polymarket_prices

    return None


def _outcomes_match(prices: Dict[str, float], outcomes: List[str]) -> bool:
    """Check if price dict keys match the expected outcomes."""
    return set(prices.keys()) == set(outcomes)


def _match_outcomes(
    prices: Dict[str, float], outcomes: List[str]
) -> Optional[Dict[str, float]]:
    """Try to match fetched market labels to expected outcome labels.
    
    Handles multiple matching strategies:
    1. Exact match
    2. Case-insensitive match
    3. Numeric threshold extraction and matching
    4. Substring matching
    """
    # Exact match
    if set(prices.keys()) == set(outcomes):
        return prices

    # Case-insensitive exact match
    prices_lower = {k.lower(): (k, v) for k, v in prices.items()}
    matched = {}
    for outcome in outcomes:
        if outcome.lower() in prices_lower:
            matched[outcome] = prices_lower[outcome.lower()][1]
    if len(matched) == len(outcomes):
        return matched

    # Numeric threshold extraction matching
    # Extract numbers from both price labels and outcomes, match by number
    def extract_number(s):
        """Extract the primary number from a string."""
        # Remove currency symbols and commas
        cleaned = s.replace("$", "").replace(",", "")
        numbers = re.findall(r'[\d.]+', cleaned)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        return None
    
    # Build a map of number -> price from Kalshi data
    number_to_price = {}
    for label, price in prices.items():
        num = extract_number(label)
        if num is not None:
            number_to_price[num] = price
    
    # Try to match outcomes by their numeric value
    if number_to_price:
        matched = {}
        for outcome in outcomes:
            num = extract_number(outcome)
            if num is not None and num in number_to_price:
                matched[outcome] = number_to_price[num]
            elif num is not None:
                # Try close match (within 0.001 for floating point)
                for k_num, k_price in number_to_price.items():
                    if abs(k_num - num) < 0.001:
                        matched[outcome] = k_price
                        break
        
        if len(matched) >= len(outcomes) * 0.6:  # At least 60% matched
            # Fill unmatched with interpolation
            matched_nums = sorted([(extract_number(o), matched[o]) for o in matched], key=lambda x: x[0])
            for outcome in outcomes:
                if outcome not in matched:
                    num = extract_number(outcome)
                    if num is not None and matched_nums:
                        # Linear interpolation from nearest matched values
                        below = [(n, p) for n, p in matched_nums if n <= num]
                        above = [(n, p) for n, p in matched_nums if n >= num]
                        if below and above:
                            n1, p1 = below[-1]
                            n2, p2 = above[0]
                            if n2 != n1:
                                matched[outcome] = p1 + (p2 - p1) * (num - n1) / (n2 - n1)
                            else:
                                matched[outcome] = p1
                        elif below:
                            matched[outcome] = below[-1][1]
                        elif above:
                            matched[outcome] = above[0][1]
            
            if len(matched) == len(outcomes):
                return matched

    # Substring match (fallback)
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
