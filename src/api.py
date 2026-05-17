"""FastAPI application for the Prophet Forecasting Agent.

Implements the main prediction orchestration pipeline with endpoints for
single event prediction, batch prediction, health checks, and cost monitoring.
Supports both the internal format and Prophet Arena's input format.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from tavily import TavilyClient

from src.aggregator import aggregate_predictions
from src.cache import PredictionCache
from src.calibrator import CalibrationModule
from src.config import AgentConfig, load_config
from src.cost_tracker import CostTracker
from src.ensemble_reasoner import EnsembleReasoner
from src.market_data import get_market_prices
from src.models import (
    ComplexityTier,
    ErrorResponse,
    EventRequest,
    PredictionResponse,
    ProbabilityEntry,
    RoutingConfig,
)
from src.reasoner import ReasoningEngine
from src.research import run_parallel_research
from src.router import classify_event
from src.search_client import SearchClient
from src.supervisor import SupervisorAgent
from src.validator import ResponseValidator

logger = logging.getLogger(__name__)

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


async def process_single_event(event: EventRequest) -> Dict[str, float]:
    """Full prediction orchestration for a single event.

    Pipeline: validate → route → cache check → research → reason →
              aggregate → supervisor → calibrate → confidence check → validate response

    Returns:
        Dict mapping outcome label to probability.
    """
    # Step 1: Route (classify category and complexity)
    routing_config = classify_event(event)

    # Extract market stats for downstream use
    market_stats = event.market_stats
    market_prices = extract_market_prices(market_stats)

    # If no market prices from input, fetch from Kalshi public API
    if not market_prices and config.use_kalshi_prices:
        fetched = await get_market_prices(
            market_ticker=event.market_ticker,
            event_ticker=event.event_ticker,
            outcomes=event.outcomes,
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

    # Step 4: Reason (run ensemble reasoner for each research result)
    prediction_results = []
    for research_result in research_results:
        prediction = await ensemble_reasoner.predict(
            event, research_result, market_stats=market_stats
        )
        prediction_results.append(prediction)

    # Step 5: Aggregate predictions
    aggregated = aggregate_predictions(prediction_results, event.outcomes)

    # Step 6: Supervisor reconciliation (if market stats available)
    if market_stats:
        evidence_summary = _build_evidence_summary(research_results)
        aggregated = await supervisor_agent.reconcile(
            predictions=aggregated,
            market_stats=market_stats,
            evidence_summary=evidence_summary,
            event_title=event.title,
            outcomes=event.outcomes,
        )

    # Step 7: Calibrate (with market anchoring if available)
    if market_prices:
        calibrated = calibrator.calibrate_with_market(aggregated, market_prices)
    else:
        calibrated = calibrator.calibrate(aggregated)

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
            # Normalize market prices (they may not sum to 1)
            total = sum(calibrated.values())
            if total > 0:
                calibrated = {k: v / total for k, v in calibrated.items()}

    # Step 8: Validate response
    is_valid, violations = validator.validate(calibrated, event.outcomes)
    if not is_valid:
        logger.warning(
            f"Validation failed for {event.event_ticker}: {violations}. "
            "Attempting correction."
        )
        calibrated = validator.correct(calibrated, event.outcomes)
        # Re-validate after correction
        is_valid, violations = validator.validate(calibrated, event.outcomes)
        if not is_valid:
            logger.error(
                f"Correction failed for {event.event_ticker}: {violations}. "
                "Falling back to uniform."
            )
            calibrated = validator.fallback_uniform(event.outcomes)

    # Cache the result
    await cache.set(event, calibrated)

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
    """Convert a probabilities dict to a PredictionResponse."""
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

    # Check Tavily API connectivity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Tavily doesn't have a simple health endpoint, so we check
            # that we can reach their API
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": config.tavily_api_key,
                    "query": "test",
                    "max_results": 1,
                },
                timeout=10.0,
            )
            if response.status_code == 200:
                status["checks"]["search_api"] = "ok"
            elif response.status_code == 401:
                status["checks"]["search_api"] = "error: invalid API key"
                status["status"] = "unhealthy"
            else:
                status["checks"]["search_api"] = (
                    f"degraded (status {response.status_code})"
                )
                status["status"] = "degraded"
    except Exception as e:
        status["checks"]["search_api"] = f"error: {str(e)}"
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
