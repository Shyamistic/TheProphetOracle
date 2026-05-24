"""FastAPI application for the Prophet Forecasting Agent.

Implements the main prediction orchestration pipeline with endpoints for
single event prediction, batch prediction, health checks, and cost monitoring.
Supports both the internal format and Prophet Arena's input format.
"""

import asyncio
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from tavily import TavilyClient

from src.aggregator import aggregate_predictions
from src.cache import PredictionCache
from src.calibrator import CalibrationModule
from src.calibration_refit import (
    apply_calibration,
    check_resolutions,
    store_prediction,
    get_calibration_status,
)
from src.config import AgentConfig, load_config
from src.cost_tracker import CostTracker
from src.ensemble_reasoner import EnsembleReasoner
from src.market_data import get_market_prices
from src.models import (
    ComplexityTier,
    ErrorResponse,
    EvidenceItem,
    EventCategory,
    EventRequest,
    PredictionResponse,
    ProbabilityEntry,
    ResearchResult,
    RoutingConfig,
)
from src.reasoner import ReasoningEngine
from src.research import run_parallel_research
from src.router import classify_event, detect_category
from src.search_client import SearchClient
from src.dashboard import get_dashboard_html
from src.supervisor import SupervisorAgent
from src.validator import ResponseValidator

logger = logging.getLogger(__name__)

# Global prediction log (in-memory, last N predictions for dashboard)
prediction_log: List[Dict] = []


# --- Upgrade 2: Time-to-Resolution Adaptive Strategy ---


def get_adaptive_anchor_weight(close_time_str: str, base_weight: float = 0.3) -> float:
    """Adjust market anchor weight based on time to resolution.

    STRATEGY SHIFT: Top teams (Dr Strange +0.0025) barely deviate from market.
    We were deviating too much and it hurt us (+0.1495 = worse than market).
    Now anchor MUCH harder to market across all time horizons.

    Imminent (≤1 day): 97% market
    Near-term (1-2 days): 95% market
    Short-term (2-4 days): 90% market
    Medium-term (4-7 days): 80% market
    Long-term (8-14 days): 65% market

    Args:
        close_time_str: ISO format close time string.
        base_weight: Default weight if parsing fails.

    Returns:
        Adaptive anchor weight between 0.0 and 1.0.
    """
    try:
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_remaining = (close_time - now).total_seconds() / 86400

        if days_remaining <= 1:
            return 0.97  # Imminent: essentially return market
        elif days_remaining <= 2:
            return 0.95  # Near-term: barely deviate
        elif days_remaining <= 4:
            return 0.90  # Short-term: very small deviation allowed
        elif days_remaining <= 7:
            return 0.80  # Medium: modest deviation
        else:
            return 0.65  # Long-term: more room but still market-heavy
    except Exception:
        return base_weight


# --- Upgrade 3: Category-Specific Confidence Tuning ---


CATEGORY_ANCHOR_MULTIPLIER = {
    "sports": 2.0,        # Anchor VERY heavily to market (sports bettors are sharp, we can't beat them)
    "entertainment": 1.5,  # Anchor heavily (we have no edge on reality TV/awards)
    "economics": 0.8,      # Trust our research more
    "geopolitics": 0.7,    # Trust our research more (markets often slow on geopolitics)
    "technology": 0.9,     # Balanced
    "science": 0.9,        # Balanced
    "general": 1.0,        # Default
}


# --- Threshold Correction (Post-Calibration) ---


def detect_threshold_outcomes(outcomes: List[str]) -> Optional[dict]:
    """Detect if outcomes follow a threshold pattern (Above X, At least X, etc.)
    
    Returns dict with:
        - direction: "above" or "below"
        - thresholds: list of (outcome_label, numeric_value) sorted by value
    Or None if not a threshold pattern.
    """
    # Patterns to match
    patterns = [
        (r'^[Aa]bove\s+([\d,.]+)', "above"),
        (r'^[Aa]t\s+least\s+([\d,.]+)', "above"),
        (r'^[Gg]reater\s+than\s+([\d,.]+)', "above"),
        (r'^>\s*([\d,.]+)', "above"),
        (r'^>=\s*([\d,.]+)', "above"),
        (r'^[Bb]elow\s+([\d,.]+)', "below"),
        (r'^[Ll]ess\s+than\s+([\d,.]+)', "below"),
        (r'^<\s*([\d,.]+)', "below"),
        (r'^<=\s*([\d,.]+)', "below"),
    ]
    
    matched = []
    direction = None
    
    for outcome in outcomes:
        found = False
        for pattern, dir_type in patterns:
            m = re.match(pattern, outcome.strip())
            if m:
                try:
                    value = float(m.group(1).replace(",", ""))
                    matched.append((outcome, value))
                    if direction is None:
                        direction = dir_type
                    elif direction != dir_type:
                        return None  # Mixed directions, not a threshold pattern
                    found = True
                    break
                except ValueError:
                    continue
        if not found:
            return None  # Not all outcomes match threshold pattern
    
    if len(matched) < 3:  # Need at least 3 thresholds to be meaningful
        return None
    
    # Sort by threshold value
    matched.sort(key=lambda x: x[1])
    
    return {"direction": direction, "thresholds": matched}


def apply_threshold_correction(
    probabilities: Dict[str, float],
    outcomes: List[str],
    event_title: str,
) -> Dict[str, float]:
    """Apply monotonic correction for threshold-type events.
    
    Detects "Above X" / "At least X" patterns and ensures probabilities
    are monotonically decreasing (higher threshold = lower probability).
    
    Uses the ensemble's raw probabilities to estimate the "center" value,
    then builds a logistic cumulative distribution around it.
    """
    threshold_info = detect_threshold_outcomes(outcomes)
    if threshold_info is None:
        return probabilities  # Not a threshold event, return unchanged
    
    direction = threshold_info["direction"]
    thresholds = threshold_info["thresholds"]  # sorted by value
    
    # Find the "center" — estimate where the actual value likely is
    # Strategy: use the median threshold as default center, but adjust if
    # the ensemble gives a clear signal (one outcome much higher than others)
    n = len(thresholds)
    
    # Default: use median threshold as center
    median_idx = n // 2
    center_value = thresholds[median_idx][1]
    
    # Check if ensemble gives a signal — find the outcome with highest probability
    max_prob = 0.0
    max_prob_idx = median_idx
    for i, (label, value) in enumerate(thresholds):
        p = probabilities.get(label, 0.0)
        if p > max_prob:
            max_prob = p
            max_prob_idx = i
    
    # If the highest-probability outcome is significantly above uniform,
    # use it as a hint for the center (the value is likely near this threshold)
    uniform_p = 1.0 / n
    if max_prob > uniform_p * 1.5:
        # Blend: 70% median, 30% ensemble hint
        hint_value = thresholds[max_prob_idx][1]
        center_value = 0.5 * center_value + 0.5 * hint_value
    
    # Calculate the spread (standard deviation estimate)
    # Use the range of thresholds to estimate volatility
    min_val = thresholds[0][1]
    max_val = thresholds[-1][1]
    value_range = max_val - min_val
    
    if value_range == 0:
        return probabilities  # All same value, can't correct
    
    # Estimate sigma as ~1/4 of the range (covers ~95% of distribution)
    sigma = value_range / 4.0
    if sigma == 0:
        sigma = 1.0
    
    # Build logistic CDF probabilities
    corrected = {}
    for label, value in thresholds:
        # Logistic CDF: P(X > threshold) = 1 / (1 + exp((threshold - center) / scale))
        # scale ≈ sigma * sqrt(3) / pi for logistic approximation of normal
        scale = sigma * 0.55  # Approximation
        if scale == 0:
            scale = 1.0
        
        z = (value - center_value) / scale
        
        if direction == "above":
            # P(Above threshold) decreases as threshold increases
            p = 1.0 / (1.0 + math.exp(z))
        else:
            # P(Below threshold) increases as threshold increases
            p = 1.0 / (1.0 + math.exp(-z))
        
        # Clamp to [0.02, 0.98]
        p = max(0.02, min(0.98, p))
        corrected[label] = p
    
    logger.info(
        f"Threshold correction applied: direction={direction}, "
        f"center={center_value:.3f}, sigma={sigma:.3f}, "
        f"range=[{min_val:.3f}, {max_val:.3f}]"
    )
    
    return corrected


# --- Range-Bucket Correction ---


def detect_range_buckets(outcomes: List[str]) -> Optional[dict]:
    """Detect if outcomes are numeric range buckets (e.g., "120-139", "140-159", "<80", ">220").
    
    Returns dict with:
        - buckets: list of (outcome_label, midpoint_value) sorted by midpoint
    Or None if not a range-bucket pattern.
    """
    # Patterns for range buckets
    range_pattern = r'^(\d[\d,.]*)\s*[-–to]+\s*(\d[\d,.]*)'  # "120-139" or "120 to 139"
    below_pattern = r'^[<≤]?\s*(\d[\d,.]*)'  # "<80" or "Below 80"
    above_pattern = r'^[>≥]?\s*(\d[\d,.]*)'  # ">220" or "Above 220"
    below_word = r'^[Bb]elow\s+([\d,.]+)'
    above_word = r'^[Aa]bove\s+([\d,.]+)'
    
    buckets = []
    
    for outcome in outcomes:
        o = outcome.strip()
        
        # Try range pattern first: "120-139"
        m = re.match(range_pattern, o)
        if m:
            try:
                low = float(m.group(1).replace(",", ""))
                high = float(m.group(2).replace(",", ""))
                midpoint = (low + high) / 2.0
                buckets.append((outcome, midpoint))
                continue
            except ValueError:
                pass
        
        # Try "<80" or "Below X" (use value - half_step as midpoint)
        m = re.match(below_word, o) or re.match(r'^<\s*([\d,.]+)', o)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                buckets.append((outcome, val - 10))  # Approximate midpoint below
                continue
            except ValueError:
                pass
        
        # Try ">220" or "Above X" (use value + half_step as midpoint)
        m = re.match(above_word, o) or re.match(r'^>\s*([\d,.]+)', o)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                buckets.append((outcome, val + 10))  # Approximate midpoint above
                continue
            except ValueError:
                pass
        
        # Doesn't match any pattern
        return None
    
    if len(buckets) < 3:
        return None
    
    # Sort by midpoint
    buckets.sort(key=lambda x: x[1])
    return {"buckets": buckets}


def apply_range_bucket_correction(
    probabilities: Dict[str, float],
    outcomes: List[str],
) -> Dict[str, float]:
    """Apply bell-curve correction for range-bucket events.
    
    For events like Trump posts (120-139, 140-159, ...) or approval ratings,
    applies a normal distribution centered on the bucket with highest ensemble probability.
    """
    bucket_info = detect_range_buckets(outcomes)
    if bucket_info is None:
        return probabilities  # Not a range-bucket event
    
    buckets = bucket_info["buckets"]
    n = len(buckets)
    
    # Find center: use the bucket with highest probability from ensemble
    max_prob = 0.0
    center_idx = n // 2
    for i, (label, midpoint) in enumerate(buckets):
        p = probabilities.get(label, 0.0)
        if p > max_prob:
            max_prob = p
            center_idx = i
    
    center_midpoint = buckets[center_idx][1]
    
    # Calculate spread from bucket range
    min_mid = buckets[0][1]
    max_mid = buckets[-1][1]
    total_range = max_mid - min_mid
    if total_range == 0:
        return probabilities
    
    sigma = total_range / 3.0  # ~99% within range
    if sigma == 0:
        sigma = 1.0
    
    # Apply normal distribution
    corrected = {}
    total = 0.0
    for label, midpoint in buckets:
        z = (midpoint - center_midpoint) / sigma
        p = math.exp(-0.5 * z * z)  # Unnormalized Gaussian
        corrected[label] = p
        total += p
    
    # Normalize to sum to 1.0
    if total > 0:
        corrected = {k: max(0.02, v / total) for k, v in corrected.items()}
        # Soft cap on tail outcomes: tails should never dominate without hard evidence
        # For 7+ bucket events, cap the extreme tails at 0.20
        if n >= 7:
            tail_cap = 0.20
            # First and last buckets are tails
            tail_labels = [buckets[0][0], buckets[-1][0]]
            for label in tail_labels:
                if label in corrected and corrected[label] > tail_cap:
                    corrected[label] = tail_cap
        # Re-normalize after clamping
        total2 = sum(corrected.values())
        corrected = {k: v / total2 for k, v in corrected.items()}
    
    logger.info(
        f"Range-bucket correction applied: center={center_midpoint:.1f}, "
        f"sigma={sigma:.1f}, peak={max(corrected.values()):.3f}"
    )
    
    return corrected


# --- Non-Mutually-Exclusive Detection ---

# Keywords that suggest a top-K / non-mutually-exclusive event
_TOP_K_KEYWORDS = [
    "top 5", "top 10", "top 3", "top 4", "top 8", "top 15", "top 20",
    "qualify", "advance", "make it to", "finish in the top",
    "which of these", "which of the following",
    "semifinal", "semi-final", "quarterfinal", "quarter-final",
    "will qualify", "will advance", "will make",
]


def is_non_mutually_exclusive(event_title: str, event_description: str, outcomes: List[str]) -> bool:
    """Detect if an event has non-mutually-exclusive outcomes (top-K style).
    
    Heuristics:
    1. Title/description contains top-K keywords
    2. Many outcomes (>4) with no "Yes"/"No" pattern
    3. Outcomes look like entity names (countries, teams, people) rather than
       mutually exclusive choices
    
    Args:
        event_title: The event title.
        event_description: The event description/context.
        outcomes: List of outcome labels.
    
    Returns:
        True if the event appears to be non-mutually-exclusive.
    """
    combined_text = (event_title + " " + event_description).lower()
    
    # Check for top-K keywords
    for keyword in _TOP_K_KEYWORDS:
        if keyword in combined_text:
            return True
    
    # Binary events (Yes/No) are always mutually exclusive
    if len(outcomes) == 2:
        outcome_set = {o.lower() for o in outcomes}
        if outcome_set == {"yes", "no"} or outcome_set == {"true", "false"}:
            return False
    
    # If there are many outcomes (>4) and they look like entity names
    # (not "Option A", "Option B" style), likely non-mutually-exclusive
    if len(outcomes) > 4:
        # Check if outcomes look like entity names (countries, teams, people)
        # Simple heuristic: if none of the outcomes contain common choice words
        choice_words = {"yes", "no", "true", "false", "option", "choice", "over", "under"}
        outcomes_lower = [o.lower() for o in outcomes]
        if not any(word in o for o in outcomes_lower for word in choice_words):
            # Many entity-like outcomes — could be top-K
            # But only if the title suggests selection/qualification
            selection_words = ["which", "who", "what", "top", "best", "finish", "win"]
            if any(word in combined_text for word in selection_words):
                return True
    
    return False

# --- Startup validation ---
# load_config() validates required env vars and calls sys.exit(1) if missing
try:
    config: AgentConfig = load_config()
except SystemExit:
    raise
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    sys.exit(1)

# --- Initialize shared components ---
app = FastAPI(title="Prophet Forecasting Agent")

# Search client with Tavily primary + Serper secondary + DuckDuckGo tertiary
search_client = SearchClient(
    tavily_api_key=config.tavily_api_key,
    serper_api_key=config.serper_api_key,
)

# Alias for backward compatibility in tests
tavily_client = search_client

# Async OpenAI client for research pipeline (uses OpenRouter)
async_llm_client = AsyncOpenAI(
    api_key=config.anthropic_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Reasoning engine (OpenAI-compatible client, wraps in asyncio.to_thread)
reasoning_engine = ReasoningEngine.from_config(config)

# Ensemble reasoner (multi-model with structured FutureSearch-style prompting)
ensemble_reasoner = EnsembleReasoner.from_config(config)

# Supervisor agent for final reconciliation
supervisor_agent = SupervisorAgent.from_config(config)

# Calibration module
calibrator = CalibrationModule(
    shrinkage_factor=config.shrinkage_factor,
    platt_coefficient=config.platt_coefficient,
    market_anchor_weight=config.market_anchor_weight,
)

# Response validator
validator = ResponseValidator()

# Prediction cache
cache = PredictionCache(ttl_hours=config.cache_ttl_hours)

# Cost tracker
cost_tracker = CostTracker(
    budget_usd=config.total_budget_usd,
    alert_threshold=config.budget_alert_threshold,
)

# Concurrency semaphore for batch processing
_semaphore = asyncio.Semaphore(config.max_concurrency)


# --- Prophet Arena Format Compatibility ---


def parse_prophet_arena_request(data: dict) -> dict:
    """Convert Prophet Arena format to our internal EventRequest format.

    Prophet Arena sends:
    {
        "event_id": "EVT_1023",
        "title": "...",
        "markets": ["Yes", "No"],
        "rules": "...",
        "market_stats": {"Yes": {"last_price": 0.72, ...}, ...}
    }

    We need:
    {
        "event_ticker": "...",
        "market_ticker": "...",
        "title": "...",
        "description": "...",
        "category": "...",
        "rules": "...",
        "close_time": "...",
        "outcomes": [...],
        "market_stats": {...}
    }

    Args:
        data: Raw request body dict.

    Returns:
        Normalized dict compatible with EventRequest model.
    """
    # If it already has event_ticker, it's our format — just pass through
    if "event_ticker" in data:
        return data

    # Prophet Arena dataset format: has "task_id" field
    if "task_id" in data:
        context = data.get("context", "") or data.get("description", "") or data.get("title", "")
        category = "general"
        if data.get("metadata") and isinstance(data["metadata"], dict):
            category = data["metadata"].get("category", "general")

        normalized = {
            "event_ticker": data.get("task_id", "UNKNOWN"),
            "market_ticker": data.get("source", data.get("task_id", "UNKNOWN")),
            "title": data.get("title", "Unknown event"),
            "description": context if context else data.get("title", "No description"),
            "category": category,
            "rules": context if context else "Standard resolution rules apply.",
            "close_time": data.get("predict_by", data.get("close_time", "2030-01-01T00:00:00Z")),
            "outcomes": data.get("outcomes", []),
        }

        # Preserve market_stats if present
        if "market_stats" in data:
            normalized["market_stats"] = data["market_stats"]

        # Handle resolved_outcome — Prophet Arena uses {"value": [...]} format
        if "resolved_outcome" in data:
            ro = data["resolved_outcome"]
            if isinstance(ro, dict) and "value" in ro:
                # Convert to string (first resolved value)
                values = ro["value"]
                normalized["resolved_outcome"] = values[0] if values else None
            elif isinstance(ro, str):
                normalized["resolved_outcome"] = ro
            else:
                normalized["resolved_outcome"] = None

        return normalized

    # Prophet Arena format detection: has "event_id" or "markets" but no "event_ticker"
    if "event_id" in data or "markets" in data:
        normalized = {
            "event_ticker": data.get("event_id", data.get("event_ticker", "UNKNOWN")),
            "market_ticker": data.get("market_ticker", data.get("event_id", "UNKNOWN")),
            "title": data.get("title", ""),
            "description": data.get("description", data.get("rules", "")),
            "category": data.get("category", "general"),
            "rules": data.get("rules", ""),
            "close_time": data.get("close_time", data.get("end_date", "2030-01-01T00:00:00Z")),
            "outcomes": data.get("markets", data.get("outcomes", [])),
        }

        # Preserve market_stats if present
        if "market_stats" in data:
            normalized["market_stats"] = data["market_stats"]

        # Preserve resolved_outcome if present
        if "resolved_outcome" in data:
            normalized["resolved_outcome"] = data["resolved_outcome"]

        return normalized

    # Unknown format — return as-is and let validation catch issues
    return data


def extract_market_prices(market_stats: Optional[Dict]) -> Optional[Dict[str, float]]:
    """Extract simple outcome -> price mapping from market_stats.

    Args:
        market_stats: Raw market_stats dict from request.

    Returns:
        Dict mapping outcome label to last_price (0-1), or None.
    """
    if not market_stats:
        return None

    prices = {}
    for outcome, stats in market_stats.items():
        if isinstance(stats, dict):
            price = stats.get("last_price")
            if price is not None:
                try:
                    prices[outcome] = float(price)
                except (TypeError, ValueError):
                    pass

    return prices if prices else None


# --- Request validation ---


def validate_event_request(data: dict) -> List[str]:
    """Validate that all required fields are present and well-formed.

    Returns a list of error messages. Empty list means valid.
    """
    errors: List[str] = []
    required_fields = [
        "event_ticker",
        "market_ticker",
        "title",
        "description",
        "category",
        "rules",
        "close_time",
        "outcomes",
    ]

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Field '{field}' cannot be null")

    # Check outcomes is a non-empty list of strings
    if "outcomes" in data and data["outcomes"] is not None:
        if not isinstance(data["outcomes"], list):
            errors.append("Field 'outcomes' must be a list")
        elif len(data["outcomes"]) < 2:
            errors.append("Field 'outcomes' must contain at least 2 items")
        else:
            for i, outcome in enumerate(data["outcomes"]):
                if not isinstance(outcome, str) or not outcome.strip():
                    errors.append(f"outcomes[{i}] must be a non-empty string")

    # Check string fields are actually strings
    string_fields = [
        "event_ticker",
        "market_ticker",
        "title",
        "description",
        "category",
        "rules",
        "close_time",
    ]
    for field in string_fields:
        if field in data and data[field] is not None:
            if not isinstance(data[field], str):
                errors.append(f"Field '{field}' must be a string")
            elif not data[field].strip():
                errors.append(f"Field '{field}' cannot be empty")

    return errors


# --- Prediction orchestration ---


async def _extract_threshold_center_value(
    event, search_text: str
) -> Optional[float]:
    """Use a cheap, fast LLM call to extract the current/expected numeric value.
    
    Uses gpt-4o-mini for cost efficiency (instead of full ensemble models).
    
    Returns the extracted center value, or None if extraction fails.
    """
    if not search_text.strip():
        return None
    
    try:
        extract_response = await asyncio.wait_for(
            asyncio.to_thread(
                ensemble_reasoner.openrouter_client.chat.completions.create,
                model="openai/gpt-4o-mini",  # Cheap, fast model for extraction
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Based on this info, what is the current/expected numeric value for: "
                        f"{event.title}?\n\nInfo: {search_text[:500]}\n\n"
                        f"Return ONLY a single number (e.g., 4.55 or 2600000). If unsure, return 'unknown'."
                    ),
                }],
            ),
            timeout=15,
        )
        value_text = extract_response.choices[0].message.content.strip() if extract_response.choices else ""
        # Try to parse the number
        value_text = value_text.replace(',', '').replace('$', '').strip()
        if value_text.lower() != 'unknown':
            return float(value_text)
    except (ValueError, TypeError):
        pass
    except Exception as e:
        logger.debug(f"Center value extraction failed: {e}")
    
    return None


async def process_single_event(event: EventRequest) -> Dict[str, float]:
    """Full prediction orchestration for a single event.

    Pipeline: validate → route → cache check → research → reason →
              aggregate → supervisor → calibrate → confidence check → validate response

    Returns:
        Dict mapping outcome label to probability.
    """
    start_time = time.time()

    # Step 1: Route (classify category and complexity)
    routing_config = classify_event(event)

    # If category is GENERAL, try to detect from title keywords
    if routing_config.category == EventCategory.GENERAL:
        title_category = detect_category(event.title)
        if title_category != EventCategory.GENERAL:
            logger.info(
                f"Title-based category override for {event.event_ticker}: "
                f"GENERAL -> {title_category.value}"
            )
            routing_config = RoutingConfig(
                category=title_category,
                complexity=routing_config.complexity,
                num_agents=routing_config.num_agents,
                max_searches=routing_config.max_searches,
                max_llm_calls=routing_config.max_llm_calls,
                search_strategies=routing_config.search_strategies,
            )

    # Detect non-mutually-exclusive events (top-K style)
    mutually_exclusive = not is_non_mutually_exclusive(
        event.title, event.description, event.outcomes
    )
    if not mutually_exclusive:
        logger.info(
            f"Event {event.event_ticker} detected as NON-mutually-exclusive (top-K). "
            "Skipping normalization throughout pipeline."
        )

    # Extract market stats for downstream use
    market_stats = event.market_stats
    market_prices = extract_market_prices(market_stats)

    # If no market prices from input, fetch from Kalshi public API
    if not market_prices and config.use_kalshi_prices:
        fetched = await get_market_prices(
            market_ticker=event.market_ticker,
            event_ticker=event.event_ticker,
            outcomes=event.outcomes,
            title=event.title,
        )
        if fetched:
            market_prices = fetched
            logger.info(
                f"Fetched Kalshi prices for {event.event_ticker}: {fetched}"
            )
            # Also populate market_stats so supervisor can use it
            if not market_stats:
                market_stats = {
                    outcome: {"last_price": price}
                    for outcome, price in fetched.items()
                }

    # Budget-critical mode: force LOW complexity tier
    if cost_tracker.is_budget_critical:
        logger.warning(
            f"Budget critical for event {event.event_ticker}, "
            "forcing LOW complexity tier"
        )
        routing_config = RoutingConfig(
            category=routing_config.category,
            complexity=ComplexityTier.LOW,
            num_agents=1,
            max_searches=1,
            max_llm_calls=2,
            search_strategies=routing_config.search_strategies[:1],
        )

    # Step 1.5: KALSHI-FIRST STRATEGY
    # If Kalshi returns prices for most outcomes, use them directly.
    # The market is almost always right. Skip expensive ensemble.
    # This is what top teams (Brier Patch, Shirish) do.
    if market_prices and len(market_prices) >= len(event.outcomes) * 0.6:
        # Check if prices are "liquid" (not all at 0.5 or extremes)
        price_values = list(market_prices.values())
        has_signal = any(abs(p - 0.5) > 0.03 for p in price_values)
        
        if has_signal:
            # Use market prices directly with minimal shrink toward uniform
            shrink_alpha = 0.01  # 1% shrink (was 3% — top teams barely deviate from market)
            kalshi_direct = {}
            for outcome in event.outcomes:
                p = market_prices.get(outcome, 1.0 / len(event.outcomes))
                # Shrink toward 0.5 very slightly
                p = p * (1 - shrink_alpha) + 0.5 * shrink_alpha
                # Clamp
                p = max(0.02, min(0.98, p))
                kalshi_direct[outcome] = p
            
            # CRITICAL: For threshold events, Kalshi prices are CDF values (P(Above X))
            # NOT mutually exclusive probabilities. We need CDF differences.
            # P(outcome "Above X" wins) = P(price > X) - P(price > X_next)
            threshold_info = detect_threshold_outcomes(event.outcomes)
            if threshold_info:
                thresholds = threshold_info["thresholds"]  # sorted by value
                direction = threshold_info["direction"]
                
                # Convert CDF values to bucket probabilities
                kalshi_direct = {}
                for i in range(len(thresholds)):
                    label, value = thresholds[i]
                    cdf_val = market_prices.get(label, 0.5)
                    
                    if direction == "above":
                        # P(Above X wins) = P(>X) - P(>X_next)
                        if i < len(thresholds) - 1:
                            next_label = thresholds[i + 1][0]
                            next_cdf = market_prices.get(next_label, 0.0)
                            bucket_prob = max(0.01, cdf_val - next_cdf)
                        else:
                            # Highest threshold: probability of being above it
                            bucket_prob = max(0.01, cdf_val)
                    else:
                        # P(Below X wins) = P(<X) - P(<X_prev)
                        if i > 0:
                            prev_label = thresholds[i - 1][0]
                            prev_cdf = market_prices.get(prev_label, 0.0)
                            bucket_prob = max(0.01, cdf_val - prev_cdf)
                        else:
                            bucket_prob = max(0.01, cdf_val)
                    
                    kalshi_direct[label] = bucket_prob
                
                # Normalize to sum to 1
                total = sum(kalshi_direct.values())
                if total > 0:
                    kalshi_direct = {k: v / total for k, v in kalshi_direct.items()}
                
                logger.info(
                    f"KALSHI-FIRST (CDF-diff): threshold event {event.event_ticker}, "
                    f"converted {len(thresholds)} CDF values to bucket probs. "
                    f"Top: {sorted(kalshi_direct.items(), key=lambda x: -x[1])[:3]}"
                )
            else:
                # Non-threshold event: use prices directly with minimal shrink
                for outcome in event.outcomes:
                    p = market_prices.get(outcome, 1.0 / len(event.outcomes))
                    # Minimal shrink toward 0.5
                    p = p * (1 - shrink_alpha) + 0.5 * shrink_alpha
                    p = max(0.02, min(0.98, p))
                    kalshi_direct[outcome] = p
                
                # Normalize for mutually exclusive events
                if mutually_exclusive:
                    total = sum(kalshi_direct.values())
                    if total > 0:
                        kalshi_direct = {k: v / total for k, v in kalshi_direct.items()}
            
            logger.info(
                f"KALSHI-FIRST: Using market prices directly for {event.event_ticker} "
                f"({len(market_prices)}/{len(event.outcomes)} outcomes covered). "
                f"Skipping ensemble. Prices: {{{', '.join(f'{k}: {v:.3f}' for k, v in list(kalshi_direct.items())[:5])}}}"
            )
            
            # Cache and log
            await cache.set(event, kalshi_direct)
            
            prediction_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_ticker": event.event_ticker,
                "title": event.title[:60],
                "category": routing_config.category.value,
                "outcomes": event.outcomes[:5],
                "probabilities": {k: round(v, 4) for k, v in kalshi_direct.items()},
                "duration": round(time.time() - start_time, 1),
                "had_disagreement": False,
            })
            if len(prediction_log) > 200:
                prediction_log[:] = prediction_log[-200:]
            
            # Persist to JSONL
            import json as _json
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_ticker": event.event_ticker,
                "title": event.title,
                "category": routing_config.category.value,
                "outcomes": event.outcomes,
                "probabilities": {k: round(v, 4) for k, v in kalshi_direct.items()},
                "duration": round(time.time() - start_time, 1),
                "had_disagreement": False,
                "mutually_exclusive": mutually_exclusive,
                "kalshi_first": True,
            }
            try:
                with open("predictions_log.jsonl", "a") as f:
                    f.write(_json.dumps(log_entry) + "\n")
            except Exception:
                pass
            
            # Store for calibration refit
            try:
                store_prediction(event.event_ticker, kalshi_direct)
            except Exception:
                pass
            
            return kalshi_direct

    # Step 2: Cache check
    cached = await cache.get(event)
    if cached is not None:
        logger.info(f"Cache hit for event {event.event_ticker}")
        return cached

    # Step 3: Research
    research_results = await run_parallel_research(
        event=event,
        config=routing_config,
        search_client=tavily_client,
        llm_client=async_llm_client,
    )

    # Step 3.4: ESPN sports data injection
    # For sports events, fetch LIVE data from ESPN (scores, odds, series status)
    # This is deterministic data that beats any LLM guess
    if routing_config.category == EventCategory.SPORTS and research_results:
        try:
            from src.sports_data import get_sports_context
            sports_context = await get_sports_context(event.title, event.description)
            if sports_context:
                research_results[0].evidence.insert(0, EvidenceItem(
                    source_url="espn://live-data",
                    summary=sports_context,
                    relevance_score=0.98,
                    publication_date=None,
                    corroborated=True,
                ))
                logger.info(f"ESPN data injected for {event.event_ticker}")
        except Exception as e:
            logger.debug(f"ESPN data fetch failed: {e}")

    # Step 3.5: Category-specific number extraction
    # For events where the answer is a PUBLISHED NUMBER (charts, counts, ratings),
    # search for the actual current value. This beats any LLM guess.
    _number_search_triggers = {
        EventCategory.ENTERTAINMENT: [
            ("netflix", "Netflix top 10 US most watched show this week"),
            ("spotify", "Spotify top chart this week global"),
            ("box office", "box office opening weekend numbers"),
            ("billboard", "Billboard Hot 100 chart this week"),
            ("streaming", "streaming viewership numbers this week"),
        ],
        EventCategory.GEOPOLITICS: [
            ("truth social", "Trump Truth Social posts count this week"),
            ("approval", "presidential approval rating latest poll 538"),
            ("poll", "latest polling numbers today"),
        ],
        EventCategory.ECONOMICS: [
            ("gas price", "national average gas price today AAA"),
            ("unemployment", "unemployment rate latest BLS"),
            ("inflation", "CPI inflation rate latest"),
            ("interest rate", "federal funds rate current"),
        ],
    }
    
    if search_client and research_results:
        title_lower = event.title.lower()
        triggers = _number_search_triggers.get(routing_config.category, [])
        # Also always check geopolitics triggers for Truth Social / approval
        if routing_config.category != EventCategory.GEOPOLITICS:
            triggers += _number_search_triggers.get(EventCategory.GEOPOLITICS, [])
        
        for keyword, query_template in triggers:
            if keyword in title_lower:
                try:
                    # Use the specific query or fall back to title-based
                    search_query = query_template if keyword in title_lower else f"{event.title} current data this week"
                    chart_results = search_client.search(
                        search_query, max_results=3, topic="news", time_range="week"
                    )
                    chart_items = chart_results.get("results", [])
                    if chart_items:
                        for item in chart_items[:2]:
                            content = item.get("content", "")
                            if content:
                                research_results[0].evidence.insert(0, EvidenceItem(
                                    source_url=item.get("url", "number_extraction"),
                                    summary=f"[CURRENT DATA - {keyword.upper()}] {content[:300]}",
                                    relevance_score=0.97,
                                    publication_date=None,
                                    corroborated=True,
                                ))
                        logger.info(
                            f"Number extraction boost ({keyword}): added data for {event.event_ticker}"
                        )
                    break  # Only do one targeted search per event
                except Exception as e:
                    logger.debug(f"Number extraction search failed ({keyword}): {e}")
                break
        else:
            # Fallback: generic entertainment/chart search if no specific trigger matched
            if routing_config.category == EventCategory.ENTERTAINMENT:
                try:
                    chart_results = search_client.search(
                        f"{event.title} this week current ranking",
                        max_results=3, topic="news", time_range="week"
                    )
                    chart_items = chart_results.get("results", [])
                    if chart_items:
                        for item in chart_items[:2]:
                            content = item.get("content", "")
                            if content:
                                research_results[0].evidence.insert(0, EvidenceItem(
                                    source_url=item.get("url", "chart_search"),
                                    summary=f"[CURRENT CHART DATA] {content[:300]}",
                                    relevance_score=0.95,
                                    publication_date=None,
                                    corroborated=True,
                                ))
                        logger.info(f"Entertainment boost: added chart data for {event.event_ticker}")
                except Exception as e:
                    logger.debug(f"Entertainment chart search failed: {e}")

    # Step 4: Reason (run ensemble reasoner for each research result)
    prediction_results = []
    for research_result in research_results:
        prediction = await ensemble_reasoner.predict(
            event, research_result, market_stats=market_stats,
            mutually_exclusive=mutually_exclusive,
        )
        prediction_results.append(prediction)

    # Step 5: Aggregate predictions
    aggregated = aggregate_predictions(prediction_results, event.outcomes, mutually_exclusive=mutually_exclusive)

    # Step 5.5: Second research pass on disagreement
    # If ensemble had significant disagreement, do a supplementary search
    # and re-run supervisor with extra evidence for better reconciliation
    any_disagreement = any(
        getattr(pr, "had_disagreement", False) for pr in prediction_results
    )
    supplementary_evidence = ""
    if any_disagreement and search_client:
        logger.info(
            f"Ensemble disagreement detected for {event.event_ticker}, "
            "doing supplementary search"
        )
        try:
            supplementary_results = search_client.search(
                f"{event.title} latest news update",
                max_results=3,
                topic="news",
                time_range="week",
            )
            # Extract summaries from supplementary search
            supp_items = supplementary_results.get("results", [])
            if supp_items:
                supp_summaries = []
                for item in supp_items[:3]:
                    content = item.get("content") or item.get("snippet") or ""
                    if content:
                        supp_summaries.append(content[:200])
                if supp_summaries:
                    supplementary_evidence = (
                        " [SUPPLEMENTARY EVIDENCE from disagreement re-search]: "
                        + " | ".join(supp_summaries)
                    )
                    logger.info(
                        f"Got {len(supp_summaries)} supplementary evidence items "
                        f"for {event.event_ticker}"
                    )
        except Exception as e:
            logger.debug(f"Supplementary search failed for {event.event_ticker}: {e}")

    # Step 5.6: Iterative counter-evidence research (BLF-inspired)
    # If the leading prediction is strong (>70%), search for counter-evidence
    # to avoid confirmation bias
    top_outcome = max(aggregated, key=aggregated.get)
    top_prob = aggregated[top_outcome]

    if top_prob > 0.70 and search_client:
        logger.info(
            f"Strong prediction ({top_outcome}: {top_prob:.1%}) for {event.event_ticker}, "
            "searching for counter-evidence to debias"
        )
        try:
            counter_query = f"{event.title} why {top_outcome} might NOT happen unlikely"
            counter_results = search_client.search(
                counter_query,
                max_results=2,
                topic="news",
                time_range="month",
            )
            counter_items = counter_results.get("results", [])
            if counter_items:
                counter_summaries = [
                    item.get("content", "")[:200] for item in counter_items[:2] if item.get("content")
                ]
                if counter_summaries:
                    supplementary_evidence += (
                        " [COUNTER-EVIDENCE to debias strong prediction]: "
                        + " | ".join(counter_summaries)
                    )
                    logger.info(f"Found {len(counter_summaries)} counter-evidence items")
        except Exception as e:
            logger.debug(f"Counter-evidence search failed: {e}")

    # Step 5.7: Iterative research for moderate confidence (BLF-inspired)
    # If prediction is moderate (40-70% for top outcome), do a second targeted search
    # to gather more evidence before finalizing
    top_outcome = max(aggregated, key=aggregated.get)
    top_prob = aggregated[top_outcome]

    if 0.40 <= top_prob <= 0.70 and search_client:
        elapsed = time.time() - start_time
        if elapsed < 120:  # Only if we have time (< 2 min elapsed)
            logger.info(
                f"Moderate confidence ({top_outcome}: {top_prob:.1%}) for {event.event_ticker}, "
                "doing iterative research pass"
            )
            try:
                # Search with more specific query based on what we know
                iterative_query = f"{event.title} latest update prediction odds forecast"
                iterative_results = search_client.search(
                    iterative_query,
                    max_results=3,
                    topic="news",
                    time_range="week",
                )
                iter_items = iterative_results.get("results", [])
                if iter_items:
                    iter_summaries = [
                        item.get("content", "")[:200] for item in iter_items[:3] if item.get("content")
                    ]
                    if iter_summaries:
                        supplementary_evidence += (
                            " [ITERATIVE RESEARCH for moderate confidence]: "
                            + " | ".join(iter_summaries)
                        )
                        logger.info(f"Iterative research found {len(iter_summaries)} items")
            except Exception as e:
                logger.debug(f"Iterative research failed: {e}")

    # Step 5.8: Second ensemble pass for highly uncertain events
    # If top probability is in the "coin flip" zone (35-65%) AND we have supplementary evidence,
    # re-run the ensemble with enriched context for a more informed prediction
    top_outcome_2 = max(aggregated, key=aggregated.get)
    top_prob_2 = aggregated[top_outcome_2]
    
    if 0.35 <= top_prob_2 <= 0.65 and supplementary_evidence and search_client:
        elapsed = time.time() - start_time
        if elapsed < 300:  # Only if we have time (< 5 min elapsed)
            logger.info(
                f"Highly uncertain ({top_outcome_2}: {top_prob_2:.1%}) for {event.event_ticker}, "
                "running second ensemble pass with enriched evidence"
            )
            # Create an enriched research result with supplementary evidence
            enriched_evidence = []
            if research_results:
                enriched_evidence = list(research_results[0].evidence)
            # Add a synthetic evidence item with supplementary findings
            if supplementary_evidence:
                enriched_evidence.append(EvidenceItem(
                    source_url="internal://supplementary-research",
                    summary=supplementary_evidence[:500],
                    relevance_score=0.9,
                    publication_date=None,
                    corroborated=True,
                ))
            enriched_research = ResearchResult(
                event_ticker=event.event_ticker,
                evidence=enriched_evidence,
                search_queries_used=[event.title],
                failed_sources=[],
                duration_seconds=0.0,
            )
            
            # Run second ensemble pass
            try:
                second_prediction = await ensemble_reasoner.predict(
                    event, enriched_research, market_stats=market_stats,
                    mutually_exclusive=mutually_exclusive,
                )
                # Blend: 60% second pass (more informed), 40% first pass
                for outcome in event.outcomes:
                    first_p = aggregated.get(outcome, 0.5)
                    second_p = second_prediction.probabilities.get(outcome, 0.5)
                    aggregated[outcome] = 0.4 * first_p + 0.6 * second_p
                
                logger.info(
                    f"Second ensemble pass complete for {event.event_ticker}, "
                    f"blended prediction: {{{', '.join(f'{k}: {v:.3f}' for k, v in aggregated.items())}}}"
                )
            except Exception as e:
                logger.warning(f"Second ensemble pass failed for {event.event_ticker}: {e}")

    # Step 6: Supervisor reconciliation (if market stats available)
    if market_stats:
        evidence_summary = _build_evidence_summary(research_results)
        # Append supplementary evidence from disagreement re-search
        if supplementary_evidence:
            evidence_summary += supplementary_evidence
        aggregated = await supervisor_agent.reconcile(
            predictions=aggregated,
            market_stats=market_stats,
            evidence_summary=evidence_summary,
            event_title=event.title,
            outcomes=event.outcomes,
            mutually_exclusive=mutually_exclusive,
        )

    # Step 7: Calibrate (with market anchoring if available)
    # Use adaptive anchor weight based on time-to-resolution and category
    adaptive_weight = get_adaptive_anchor_weight(
        event.close_time, base_weight=config.market_anchor_weight
    )
    category_mult = CATEGORY_ANCHOR_MULTIPLIER.get(routing_config.category.value, 1.0)
    final_anchor_weight = min(0.90, adaptive_weight * category_mult)

    logger.info(
        f"Anchor weight for {event.event_ticker}: adaptive={adaptive_weight:.2f}, "
        f"category_mult={category_mult:.1f} ({routing_config.category.value}), "
        f"final={final_anchor_weight:.2f}"
    )

    if market_prices:
        calibrated = calibrator.calibrate_with_market(
            aggregated, market_prices, anchor_weight=final_anchor_weight,
            normalize=mutually_exclusive,
        )
    else:
        calibrated = calibrator.calibrate(aggregated, normalize=mutually_exclusive)

    # Step 7.1: Threshold correction for "Above X" / "At least X" events
    # These need monotonically decreasing probabilities, not uniform
    # Skip if we already have market prices (they're better than our CDF estimate)
    if not market_prices:
        calibrated = apply_threshold_correction(calibrated, event.outcomes, event.title)
        # Also try range-bucket correction (for "120-139", "140-159" style events)
        calibrated = apply_range_bucket_correction(calibrated, event.outcomes)

    # Step 7.2: Market sanity guardrail
    # If our prediction deviates >0.15 from Kalshi on any outcome, anchor back toward market
    # Top teams barely deviate from market — we were deviating too much
    if market_prices:
        max_deviation = max(
            abs(calibrated.get(o, 0.5) - market_prices.get(o, 0.5))
            for o in event.outcomes if o in market_prices
        )
        if max_deviation > 0.15:
            # Anchor 70% toward market, 30% our prediction (was 60/40)
            logger.info(
                f"Market sanity guardrail triggered for {event.event_ticker}: "
                f"max deviation {max_deviation:.3f} > 0.15, anchoring toward market"
            )
            for outcome in event.outcomes:
                if outcome in market_prices:
                    our_p = calibrated.get(outcome, 0.5)
                    market_p = market_prices[outcome]
                    calibrated[outcome] = 0.30 * our_p + 0.70 * market_p

    # Step 7.5: Confidence check — use market as default when we have no edge
    # Key hackathon insight: "Only make prediction when you are confident enough,
    # otherwise just use the market probability as your prediction"
    if market_prices and config.confidence_threshold > 0:
        max_deviation = max(
            abs(calibrated.get(o, 0.5) - market_prices.get(o, 0.5))
            for o in event.outcomes
        )
        if max_deviation < config.confidence_threshold:
            # We have no edge — use market prices directly
            logger.info(
                f"No edge detected for {event.event_ticker} "
                f"(max deviation {max_deviation:.3f} < threshold {config.confidence_threshold}), "
                f"using market prices"
            )
            calibrated = {o: market_prices.get(o, 1.0 / len(event.outcomes)) for o in event.outcomes}
            # Normalize market prices only for mutually exclusive events
            if mutually_exclusive:
                total = sum(calibrated.values())
                if total > 0:
                    calibrated = {k: v / total for k, v in calibrated.items()}

    # Step 7.9: Apply calibration refit (Platt scaling from resolved outcomes)
    # This is Brier Patch's key insight — continuously recalibrate based on results
    if mutually_exclusive:
        calibrated = apply_calibration(calibrated)

    # Step 8: Validate response
    is_valid, violations = validator.validate(calibrated, event.outcomes, mutually_exclusive=mutually_exclusive)
    if not is_valid:
        logger.warning(
            f"Validation failed for {event.event_ticker}: {violations}. "
            "Attempting correction."
        )
        calibrated = validator.correct(calibrated, event.outcomes, mutually_exclusive=mutually_exclusive)
        # Re-validate after correction
        is_valid, violations = validator.validate(calibrated, event.outcomes, mutually_exclusive=mutually_exclusive)
        if not is_valid:
            logger.error(
                f"Correction failed for {event.event_ticker}: {violations}. "
                "Falling back to uniform."
            )
            calibrated = validator.fallback_uniform(event.outcomes)

    # Cache the result
    await cache.set(event, calibrated)

    # Log prediction for dashboard
    prediction_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_ticker": event.event_ticker,
        "title": event.title[:60],
        "category": routing_config.category.value,
        "outcomes": event.outcomes[:5],
        "probabilities": {k: round(v, 4) for k, v in calibrated.items()},
        "duration": round(time.time() - start_time, 1) if 'start_time' in locals() else 0,
        "had_disagreement": any(
            getattr(pr, "had_disagreement", False) for pr in prediction_results
        ) if prediction_results else False,
    })
    # Keep log bounded to last 200 entries
    if len(prediction_log) > 200:
        prediction_log[:] = prediction_log[-200:]

    # Persist to JSONL file for post-hoc analysis
    import json as _json
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_ticker": event.event_ticker,
        "title": event.title,
        "category": routing_config.category.value,
        "outcomes": event.outcomes,
        "probabilities": {k: round(v, 4) for k, v in calibrated.items()},
        "duration": round(time.time() - start_time, 1),
        "had_disagreement": any(
            getattr(pr, "had_disagreement", False) for pr in prediction_results
        ) if prediction_results else False,
        "mutually_exclusive": mutually_exclusive,
    }
    try:
        with open("predictions_log.jsonl", "a") as f:
            f.write(_json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Don't let logging failures break predictions

    # Store prediction for calibration refit (non-blocking)
    try:
        store_prediction(event.event_ticker, calibrated)
    except Exception:
        pass

    # Periodically check for resolved markets (every ~10 predictions)
    # This is lightweight and non-blocking
    if len(prediction_log) % 10 == 0:
        try:
            asyncio.create_task(check_resolutions())
        except Exception:
            pass

    return calibrated


def _build_evidence_summary(research_results: list) -> str:
    """Build a brief evidence summary from research results for the supervisor.

    Args:
        research_results: List of ResearchResult objects.

    Returns:
        Brief text summary of key evidence.
    """
    summaries = []
    for result in research_results:
        for item in result.evidence[:3]:  # Top 3 evidence items per agent
            if item.summary:
                summaries.append(item.summary[:150])

    if not summaries:
        return "No specific evidence found."

    return " | ".join(summaries[:5])  # Cap at 5 items total


async def process_event_with_timeout(
    event: EventRequest, timeout_seconds: float
) -> Dict[str, float]:
    """Process a single event with timeout and error isolation.

    On timeout or error, falls back to uniform distribution.
    """
    try:
        async with _semaphore:
            result = await asyncio.wait_for(
                process_single_event(event),
                timeout=timeout_seconds,
            )
            return result
    except asyncio.TimeoutError:
        logger.warning(
            f"Event {event.event_ticker} timed out after {timeout_seconds}s, "
            "falling back to uniform distribution"
        )
        return validator.fallback_uniform(event.outcomes)
    except Exception as e:
        logger.error(
            f"Error processing event {event.event_ticker}: {e}",
            exc_info=True,
        )
        return validator.fallback_uniform(event.outcomes)


def build_prediction_response(
    probabilities: Dict[str, float],
) -> PredictionResponse:
    """Convert a probabilities dict to a PredictionResponse.
    
    Note: We do NOT force probabilities to sum to 1 here.
    Prophet Arena scores as-is and outcomes may not be mutually exclusive.
    """
    entries = [
        ProbabilityEntry(market=outcome, probability=round(prob, 4))
        for outcome, prob in probabilities.items()
    ]
    return PredictionResponse(probabilities=entries)


# --- Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request) -> PredictionResponse:
    """Main prediction endpoint. Processes a single event.

    Supports both our internal format and Prophet Arena's format.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid JSON in request body"},
        )

    # Convert Prophet Arena format if needed
    body = parse_prophet_arena_request(body)

    # Validate request fields
    errors = validate_event_request(body)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"error": "; ".join(errors)},
        )

    # Parse into EventRequest model
    try:
        event = EventRequest(**body)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Request validation failed: {str(e)}"},
        )

    # Process with per-event timeout
    probabilities = await process_event_with_timeout(
        event, config.per_event_timeout_seconds
    )

    return build_prediction_response(probabilities)


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: Request) -> List[PredictionResponse]:
    """Batch prediction endpoint. Processes multiple events concurrently."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid JSON in request body"},
        )

    if not isinstance(body, list):
        raise HTTPException(
            status_code=400,
            detail={"error": "Request body must be a JSON array of events"},
        )

    if len(body) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "Request body must contain at least one event"},
        )

    # Validate each event
    events: List[EventRequest] = []
    for i, event_data in enumerate(body):
        if not isinstance(event_data, dict):
            raise HTTPException(
                status_code=400,
                detail={"error": f"Event at index {i} must be a JSON object"},
            )
        errors = validate_event_request(event_data)
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Event at index {i}: {'; '.join(errors)}"},
            )
        try:
            events.append(EventRequest(**event_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Event at index {i} validation failed: {str(e)}"
                },
            )

    # Batch timeout: 9.5 minutes with 30-second safety margin for response assembly
    batch_start = time.time()
    batch_timeout = config.response_timeout_seconds  # 570 seconds (9.5 min)
    per_event_timeout = float(config.per_event_timeout_seconds)  # 480 seconds (8 min)

    # Process all events concurrently with error isolation
    tasks = [
        asyncio.create_task(
            process_event_with_timeout(event, per_event_timeout)
        )
        for event in events
    ]

    # Wait for all tasks with batch-level timeout (minus safety margin)
    safety_margin = 30.0
    remaining_time = batch_timeout - (time.time() - batch_start) - safety_margin

    results: List[Dict[str, float]] = []
    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=max(remaining_time, 1.0),
            return_when=asyncio.ALL_COMPLETED,
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()

        # Collect results in order
        for i, task in enumerate(tasks):
            if task in done and not task.cancelled():
                try:
                    results.append(task.result())
                except Exception:
                    # Fallback for failed tasks
                    results.append(
                        validator.fallback_uniform(events[i].outcomes)
                    )
            else:
                # Timed out or cancelled - use uniform fallback
                results.append(
                    validator.fallback_uniform(events[i].outcomes)
                )

    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        # Return uniform for all events on catastrophic failure
        results = [
            validator.fallback_uniform(event.outcomes) for event in events
        ]

    # Build responses
    responses = [build_prediction_response(probs) for probs in results]
    return responses


@app.get("/health")
async def health_check() -> dict:
    """Health check verifying API key validity and network connectivity."""
    status = {"status": "healthy", "checks": {}}

    # Check Anthropic/OpenRouter API connectivity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "Authorization": f"Bearer {config.anthropic_api_key}"
                },
            )
            if response.status_code == 200:
                status["checks"]["llm_api"] = "ok"
            else:
                status["checks"]["llm_api"] = (
                    f"degraded (status {response.status_code})"
                )
                status["status"] = "degraded"
    except Exception as e:
        status["checks"]["llm_api"] = f"error: {str(e)}"
        status["status"] = "unhealthy"

    # Check Tavily API connectivity (just verify key is configured, don't make a real search)
    if config.tavily_api_key and len(config.tavily_api_key) > 10:
        status["checks"]["search_api"] = "ok"
    else:
        status["checks"]["search_api"] = "error: no API key configured"
        status["status"] = "unhealthy"

    # Add cost info
    status["budget"] = {
        "total_spend_usd": round(cost_tracker.total_spend, 4),
        "budget_remaining_usd": round(cost_tracker.budget_remaining, 4),
        "is_budget_critical": cost_tracker.is_budget_critical,
    }

    return status


@app.get("/costs")
async def get_costs() -> dict:
    """Returns cumulative cost summary."""
    return cost_tracker.get_summary()


@app.get("/calibration")
async def get_calibration() -> dict:
    """Returns current calibration refit status."""
    return get_calibration_status()


@app.get("/api-status")
async def get_api_status() -> dict:
    """Fetch real-time API balances from OpenRouter and Tavily.
    
    Returns actual remaining credits/balance from each service.
    """
    result = {
        "openrouter": {"status": "unknown", "balance": None, "usage": None},
        "tavily": {"status": "unknown", "credits_remaining": None},
        "featherless": {"status": "unknown"},
    }
    
    # Check OpenRouter balance via /api/v1/key
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {config.anthropic_api_key}"},
            )
            if response.status_code == 200:
                data = response.json()
                key_data = data.get("data", {})
                result["openrouter"] = {
                    "status": "ok",
                    "label": key_data.get("label", ""),
                    "limit": key_data.get("limit"),
                    "usage": key_data.get("usage"),
                    "remaining": (
                        round(key_data["limit"] - key_data["usage"], 4)
                        if key_data.get("limit") is not None and key_data.get("usage") is not None
                        else None
                    ),
                    "limit_remaining": key_data.get("limit_remaining"),
                    "is_free_tier": key_data.get("is_free_tier", False),
                    "rate_limit": key_data.get("rate_limit", {}),
                }
            else:
                result["openrouter"]["status"] = f"error ({response.status_code})"
    except Exception as e:
        result["openrouter"]["status"] = f"error: {str(e)[:100]}"
    
    # Check Featherless connectivity
    if config.featherless_api_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.featherless.ai/v1/models",
                    headers={"Authorization": f"Bearer {config.featherless_api_key}"},
                )
                if response.status_code == 200:
                    result["featherless"]["status"] = "ok"
                else:
                    result["featherless"]["status"] = f"error ({response.status_code})"
        except Exception as e:
            result["featherless"]["status"] = f"error: {str(e)[:100]}"
    
    # Add internal cost tracking info
    result["internal_tracking"] = {
        "total_predictions": len(prediction_log),
        "estimated_spend_usd": round(cost_tracker.total_spend, 4),
        "budget_usd": cost_tracker.budget_usd,
    }
    
    return result


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard HTML page."""
    return get_dashboard_html()


@app.get("/logs")
async def get_logs():
    """Return the last 50 predictions from the in-memory log."""
    return {"predictions": prediction_log[-50:], "total": len(prediction_log)}


# --- Error handlers ---


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler to ensure error responses match the ErrorResponse schema."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        error_msg = detail["error"]
    else:
        error_msg = str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_msg},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )
