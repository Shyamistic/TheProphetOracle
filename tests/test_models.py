"""Unit tests for shared Pydantic models, data classes, and enums."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models import (
    APICallRecord,
    ComplexityTier,
    ErrorResponse,
    EventCategory,
    EventRequest,
    EvidenceItem,
    PredictionResponse,
    PredictionResult,
    ProbabilityEntry,
    ReasoningTrace,
    ResearchResult,
    RoutingConfig,
    SearchQuery,
)


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_all_values_present(self):
        values = {e.value for e in EventCategory}
        assert values == {"sports", "economics", "geopolitics", "technology", "science", "general"}

    def test_string_enum_comparison(self):
        assert EventCategory.SPORTS == "sports"
        assert EventCategory.GENERAL == "general"


class TestComplexityTier:
    """Tests for ComplexityTier enum."""

    def test_all_values_present(self):
        values = {e.value for e in ComplexityTier}
        assert values == {"low", "medium", "high"}

    def test_string_enum_comparison(self):
        assert ComplexityTier.LOW == "low"
        assert ComplexityTier.HIGH == "high"


class TestEventRequest:
    """Tests for EventRequest Pydantic model."""

    def test_valid_event_request(self):
        event = EventRequest(
            event_ticker="EVT-001",
            market_ticker="MKT-001",
            title="Will X happen?",
            description="Detailed description",
            category="Sports",
            rules="Standard rules",
            close_time="2026-05-31T00:00:00Z",
            outcomes=["Yes", "No"],
            resolved_outcome=None,
        )
        assert event.event_ticker == "EVT-001"
        assert event.outcomes == ["Yes", "No"]
        assert event.resolved_outcome is None

    def test_resolved_outcome_optional(self):
        event = EventRequest(
            event_ticker="EVT-002",
            market_ticker="MKT-002",
            title="Test",
            description="Test",
            category="Economics",
            rules="Rules",
            close_time="2026-06-01T00:00:00Z",
            outcomes=["Up", "Down", "Flat"],
        )
        assert event.resolved_outcome is None

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            EventRequest(
                event_ticker="EVT-001",
                # missing market_ticker and other required fields
            )


class TestProbabilityEntry:
    """Tests for ProbabilityEntry Pydantic model."""

    def test_valid_probability(self):
        entry = ProbabilityEntry(market="Yes", probability=0.65)
        assert entry.market == "Yes"
        assert entry.probability == 0.65

    def test_minimum_probability(self):
        entry = ProbabilityEntry(market="Yes", probability=0.01)
        assert entry.probability == 0.01

    def test_maximum_probability(self):
        entry = ProbabilityEntry(market="Yes", probability=0.99)
        assert entry.probability == 0.99

    def test_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            ProbabilityEntry(market="Yes", probability=0.009)

    def test_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            ProbabilityEntry(market="Yes", probability=0.991)

    def test_zero_probability_raises(self):
        with pytest.raises(ValidationError):
            ProbabilityEntry(market="Yes", probability=0.0)

    def test_one_probability_raises(self):
        with pytest.raises(ValidationError):
            ProbabilityEntry(market="Yes", probability=1.0)


class TestPredictionResponse:
    """Tests for PredictionResponse Pydantic model."""

    def test_valid_response(self):
        resp = PredictionResponse(
            probabilities=[
                ProbabilityEntry(market="Yes", probability=0.6),
                ProbabilityEntry(market="No", probability=0.4),
            ]
        )
        assert len(resp.probabilities) == 2

    def test_empty_probabilities(self):
        resp = PredictionResponse(probabilities=[])
        assert resp.probabilities == []


class TestErrorResponse:
    """Tests for ErrorResponse Pydantic model."""

    def test_valid_error(self):
        err = ErrorResponse(error="Something went wrong")
        assert err.error == "Something went wrong"

    def test_missing_error_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse()


class TestEvidenceItem:
    """Tests for EvidenceItem dataclass."""

    def test_creation(self):
        item = EvidenceItem(
            source_url="https://example.com",
            publication_date=datetime(2026, 5, 1),
            summary="Key finding",
            relevance_score=0.85,
            corroborated=True,
        )
        assert item.source_url == "https://example.com"
        assert item.relevance_score == 0.85
        assert item.corroborated is True

    def test_optional_publication_date(self):
        item = EvidenceItem(
            source_url="https://example.com",
            publication_date=None,
            summary="No date available",
            relevance_score=0.5,
            corroborated=False,
        )
        assert item.publication_date is None


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""

    def test_default_max_results(self):
        query = SearchQuery(query_text="test query", source_type="general_news")
        assert query.max_results == 5

    def test_custom_max_results(self):
        query = SearchQuery(query_text="test", source_type="team_statistics", max_results=10)
        assert query.max_results == 10


class TestReasoningTrace:
    """Tests for ReasoningTrace dataclass."""

    def test_creation(self):
        trace = ReasoningTrace(
            evidence_considered=["source1", "source2"],
            base_rate=0.5,
            supporting_factors=["factor1", "factor2"],
            conflicting_evidence=["conflict1"],
            conflict_resolution="Weighted by recency",
            confidence_level="medium",
        )
        assert len(trace.supporting_factors) >= 2
        assert trace.confidence_level in ("low", "medium", "high")


class TestRoutingConfig:
    """Tests for RoutingConfig dataclass."""

    def test_creation(self):
        config = RoutingConfig(
            category=EventCategory.SPORTS,
            complexity=ComplexityTier.HIGH,
            num_agents=3,
            max_searches=3,
            max_llm_calls=5,
            search_strategies=["template1", "template2"],
        )
        assert config.category == EventCategory.SPORTS
        assert config.complexity == ComplexityTier.HIGH
        assert config.num_agents == 3
