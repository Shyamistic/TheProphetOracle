"""Shared Pydantic models, data classes, and enums for the Prophet Forecasting Agent."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# === Enums ===


class EventCategory(str, Enum):
    """Event category classification for routing to category-specific strategies."""

    SPORTS = "sports"
    ECONOMICS = "economics"
    GEOPOLITICS = "geopolitics"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    GENERAL = "general"


class ComplexityTier(str, Enum):
    """Complexity tier for tiered reasoning strategies.

    LOW: 2 outcomes AND description < 200 chars → 1 agent, 1 search
    MEDIUM: 2-3 outcomes AND description < 500 chars → 2 agents, 2 searches
    HIGH: 4+ outcomes OR description >= 500 chars → 3 agents, 3 searches
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# === API Request/Response Models ===


class MarketStats(BaseModel):
    """Live market statistics from Kalshi."""

    last_price: Optional[float] = None
    yes_ask: Optional[float] = None
    no_ask: Optional[float] = None


class EventRequest(BaseModel):
    """Incoming event request conforming to the Prophet Hacks event schema."""

    event_ticker: str
    market_ticker: str
    title: str
    description: str
    category: str
    rules: str
    close_time: str  # ISO 8601 datetime string
    outcomes: List[str]
    resolved_outcome: Optional[str] = None
    market_stats: Optional[Dict[str, Any]] = None  # outcome -> MarketStats or raw dict


class ProbabilityEntry(BaseModel):
    """A single outcome probability entry in the prediction response."""

    market: str
    probability: float = Field(ge=0.01, le=0.99)


class PredictionResponse(BaseModel):
    """Prediction response containing probabilities for all outcomes."""

    probabilities: List[ProbabilityEntry]


class ErrorResponse(BaseModel):
    """Error response returned on validation or processing failures."""

    error: str


# === Internal Data Classes ===


@dataclass
class EvidenceItem:
    """A single piece of evidence gathered during research."""

    source_url: str
    publication_date: Optional[datetime]
    summary: str
    relevance_score: float  # 0.0-1.0
    corroborated: bool  # True if confirmed by another source


@dataclass
class ResearchResult:
    """Result of a research agent's investigation of an event."""

    event_ticker: str
    evidence: List[EvidenceItem]
    search_queries_used: List[str]
    failed_sources: List[dict]  # [{source, reason}]
    duration_seconds: float


@dataclass
class SearchQuery:
    """A search query to be executed by the research pipeline."""

    query_text: str
    source_type: str  # e.g., "team_statistics", "market_indicators"
    max_results: int = 5


@dataclass
class ReasoningTrace:
    """Trace of the reasoning process for a prediction."""

    evidence_considered: List[str]
    base_rate: float
    supporting_factors: List[str]  # At least 2
    conflicting_evidence: List[str]
    conflict_resolution: str
    confidence_level: str  # "low", "medium", "high"


@dataclass
class PredictionResult:
    """Result of the reasoning engine for a single event."""

    event_ticker: str
    probabilities: Dict[str, float]  # outcome -> probability
    reasoning_trace: ReasoningTrace
    duration_seconds: float
    had_disagreement: bool = False  # True if ensemble models disagreed >15%


@dataclass
class APICallRecord:
    """Record of a single API call for cost tracking."""

    timestamp: datetime
    service: str  # "anthropic", "openai", "tavily"
    model: str  # "claude-sonnet-4", "gpt-4o-mini", etc.
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    event_ticker: str
    category: str


@dataclass
class RoutingConfig:
    """Configuration for how an event should be processed."""

    category: EventCategory
    complexity: ComplexityTier
    num_agents: int  # 1-3 based on complexity
    max_searches: int  # 1-3 based on complexity
    max_llm_calls: int  # 2-5 based on complexity
    search_strategies: List[str]  # Category-specific query templates
