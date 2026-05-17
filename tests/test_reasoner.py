"""Unit tests for the ReasoningEngine in src/reasoner.py.

Tests cover prompt building, response parsing, normalization,
fallback behavior, and timeout handling.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    EventRequest,
    EvidenceItem,
    PredictionResult,
    ReasoningTrace,
    ResearchResult,
)
from src.reasoner import ReasoningEngine


# === Fixtures ===


@pytest.fixture
def engine():
    """Create a ReasoningEngine with a dummy API key (no real calls)."""
    return ReasoningEngine(
        api_key="test-key-123",
        model="claude-sonnet-4-20250514",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=10,
    )


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return EventRequest(
        event_ticker="EVT-001",
        market_ticker="MKT-001",
        title="Will Team A beat Team B in the championship?",
        description="Team A faces Team B in the finals on June 15.",
        category="Sports",
        rules="Resolves YES if Team A wins, NO otherwise.",
        close_time="2026-06-15T23:59:00Z",
        outcomes=["Yes", "No"],
    )


@pytest.fixture
def sample_research():
    """Create sample research results for testing."""
    return ResearchResult(
        event_ticker="EVT-001",
        evidence=[
            EvidenceItem(
                source_url="https://espn.com/article1",
                publication_date=datetime(2026, 6, 1),
                summary="Team A has won 8 of their last 10 games.",
                relevance_score=0.9,
                corroborated=True,
            ),
            EvidenceItem(
                source_url="https://sports-ref.com/stats",
                publication_date=datetime(2026, 5, 28),
                summary="Team B has a strong defensive record this season.",
                relevance_score=0.8,
                corroborated=True,
            ),
        ],
        search_queries_used=["Team A vs Team B 2026", "championship odds"],
        failed_sources=[],
        duration_seconds=5.0,
    )


@pytest.fixture
def empty_research():
    """Create research results with no evidence."""
    return ResearchResult(
        event_ticker="EVT-002",
        evidence=[],
        search_queries_used=["query1"],
        failed_sources=[{"source": "example.com", "reason": "timeout"}],
        duration_seconds=2.0,
    )


@pytest.fixture
def multi_outcome_event():
    """Create an event with multiple outcomes."""
    return EventRequest(
        event_ticker="EVT-003",
        market_ticker="MKT-003",
        title="Which party will win the election?",
        description="The general election is scheduled for November.",
        category="Geopolitics",
        rules="Resolves to the winning party.",
        close_time="2026-11-05T23:59:00Z",
        outcomes=["Party A", "Party B", "Party C"],
    )


# === Tests for build_prompt ===


class TestBuildPrompt:
    def test_includes_event_title(self, engine, sample_event, sample_research):
        prompt = engine.build_prompt(sample_event, sample_research)
        assert sample_event.title in prompt

    def test_includes_event_description(self, engine, sample_event, sample_research):
        prompt = engine.build_prompt(sample_event, sample_research)
        assert sample_event.description in prompt

    def test_includes_outcomes(self, engine, sample_event, sample_research):
        prompt = engine.build_prompt(sample_event, sample_research)
        for outcome in sample_event.outcomes:
            assert outcome in prompt

    def test_includes_evidence_summaries(self, engine, sample_event, sample_research):
        prompt = engine.build_prompt(sample_event, sample_research)
        for item in sample_research.evidence:
            assert item.summary in prompt

    def test_includes_evidence_source_urls(self, engine, sample_event, sample_research):
        prompt = engine.build_prompt(sample_event, sample_research)
        for item in sample_research.evidence:
            assert item.source_url in prompt

    def test_no_evidence_message(self, engine, sample_event, empty_research):
        prompt = engine.build_prompt(sample_event, empty_research)
        assert "No research evidence" in prompt
        assert "base rates" in prompt.lower()

    def test_includes_json_format_instructions(
        self, engine, sample_event, sample_research
    ):
        prompt = engine.build_prompt(sample_event, sample_research)
        assert "probabilities" in prompt
        assert "reasoning_trace" in prompt
        assert "evidence_considered" in prompt
        assert "base_rate" in prompt
        assert "supporting_factors" in prompt
        assert "confidence_level" in prompt

    def test_includes_probability_constraints(
        self, engine, sample_event, sample_research
    ):
        prompt = engine.build_prompt(sample_event, sample_research)
        assert "0.01" in prompt
        assert "0.99" in prompt
        assert "sum to" in prompt.lower() or "sum to 1.0" in prompt


# === Tests for parse_prediction ===


class TestParsePrediction:
    def test_valid_two_outcome_response(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Yes": 0.7, "No": 0.3},
                "reasoning_trace": {
                    "evidence_considered": ["Team A strong record"],
                    "base_rate": 0.5,
                    "supporting_factors": ["Recent wins", "Home advantage"],
                    "conflicting_evidence": ["Team B defense"],
                    "conflict_resolution": "Weighted by recency",
                    "confidence_level": "medium",
                },
            }
        )
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert abs(result.probabilities["Yes"] - 0.7) < 0.01
        assert abs(result.probabilities["No"] - 0.3) < 0.01
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.001
        assert result.reasoning_trace.confidence_level == "medium"
        assert len(result.reasoning_trace.supporting_factors) >= 2

    def test_valid_three_outcome_response(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Party A": 0.5, "Party B": 0.35, "Party C": 0.15},
                "reasoning_trace": {
                    "evidence_considered": ["Polls show Party A leading"],
                    "base_rate": 0.33,
                    "supporting_factors": ["Poll data", "Historical trends"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "No conflicts",
                    "confidence_level": "high",
                },
            }
        )
        result = engine.parse_prediction(response, ["Party A", "Party B", "Party C"])

        assert abs(sum(result.probabilities.values()) - 1.0) < 0.001
        assert all(0.01 <= p <= 0.99 for p in result.probabilities.values())

    def test_handles_markdown_code_fences(self, engine):
        inner = json.dumps(
            {
                "probabilities": {"Yes": 0.6, "No": 0.4},
                "reasoning_trace": {
                    "evidence_considered": [],
                    "base_rate": 0.5,
                    "supporting_factors": ["Factor 1", "Factor 2"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "",
                    "confidence_level": "low",
                },
            }
        )
        response = f"```json\n{inner}\n```"
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert abs(result.probabilities["Yes"] - 0.6) < 0.01
        assert abs(result.probabilities["No"] - 0.4) < 0.01

    def test_clamps_extreme_probabilities(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Yes": 0.999, "No": 0.001},
                "reasoning_trace": {
                    "evidence_considered": [],
                    "base_rate": 0.5,
                    "supporting_factors": ["A", "B"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "",
                    "confidence_level": "high",
                },
            }
        )
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert result.probabilities["Yes"] <= 0.99
        assert result.probabilities["No"] >= 0.01

    def test_handles_missing_outcome_in_response(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Yes": 0.8},
                "reasoning_trace": {
                    "evidence_considered": [],
                    "base_rate": 0.5,
                    "supporting_factors": ["A", "B"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "",
                    "confidence_level": "low",
                },
            }
        )
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert "Yes" in result.probabilities
        assert "No" in result.probabilities
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.001

    def test_normalizes_non_summing_probabilities(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Yes": 0.6, "No": 0.6},
                "reasoning_trace": {
                    "evidence_considered": [],
                    "base_rate": 0.5,
                    "supporting_factors": ["A", "B"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "",
                    "confidence_level": "low",
                },
            }
        )
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert abs(sum(result.probabilities.values()) - 1.0) < 0.001

    def test_raises_on_invalid_json(self, engine):
        with pytest.raises(ValueError, match="Failed to parse"):
            engine.parse_prediction("not valid json at all", ["Yes", "No"])

    def test_raises_on_empty_probabilities(self, engine):
        response = json.dumps(
            {
                "probabilities": {},
                "reasoning_trace": {},
            }
        )
        with pytest.raises(ValueError, match="No probabilities"):
            engine.parse_prediction(response, ["Yes", "No"])

    def test_ensures_minimum_supporting_factors(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Yes": 0.5, "No": 0.5},
                "reasoning_trace": {
                    "evidence_considered": [],
                    "base_rate": 0.5,
                    "supporting_factors": ["Only one"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "",
                    "confidence_level": "low",
                },
            }
        )
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert len(result.reasoning_trace.supporting_factors) >= 2

    def test_defaults_confidence_to_low_on_invalid(self, engine):
        response = json.dumps(
            {
                "probabilities": {"Yes": 0.5, "No": 0.5},
                "reasoning_trace": {
                    "evidence_considered": [],
                    "base_rate": 0.5,
                    "supporting_factors": ["A", "B"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "",
                    "confidence_level": "invalid_value",
                },
            }
        )
        result = engine.parse_prediction(response, ["Yes", "No"])

        assert result.reasoning_trace.confidence_level == "low"


# === Tests for predict (async with mocking) ===


class TestPredict:
    @pytest.mark.asyncio
    async def test_successful_prediction(self, engine, sample_event, sample_research):
        """Test that predict returns valid results when LLM responds correctly."""
        mock_response_data = json.dumps(
            {
                "probabilities": {"Yes": 0.65, "No": 0.35},
                "reasoning_trace": {
                    "evidence_considered": ["Team A record"],
                    "base_rate": 0.5,
                    "supporting_factors": ["Recent form", "Home advantage"],
                    "conflicting_evidence": ["Team B defense"],
                    "conflict_resolution": "Recency weighted",
                    "confidence_level": "medium",
                },
            }
        )

        mock_message = MagicMock()
        mock_message.content = mock_response_data

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(engine.client.chat.completions, "create", return_value=mock_response):
            result = await engine.predict(sample_event, sample_research)

        assert result.event_ticker == "EVT-001"
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.001
        assert all(0.01 <= p <= 0.99 for p in result.probabilities.values())
        assert result.duration_seconds > 0
        assert result.reasoning_trace.confidence_level == "medium"

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback(self, sample_event, sample_research):
        """Test that timeout produces uniform fallback prediction."""
        engine = ReasoningEngine(
            api_key="test-key",
            model="test-model",
            timeout_seconds=0.001,  # Very short timeout to trigger
        )

        def slow_call(*args, **kwargs):
            import time
            time.sleep(10)

        with patch.object(engine.client.chat.completions, "create", side_effect=slow_call):
            result = await engine.predict(sample_event, sample_research)

        # Should get uniform distribution
        expected_prob = 1.0 / len(sample_event.outcomes)
        for outcome in sample_event.outcomes:
            assert abs(result.probabilities[outcome] - expected_prob) < 0.001

        assert result.reasoning_trace.confidence_level == "low"
        assert result.event_ticker == "EVT-001"

    @pytest.mark.asyncio
    async def test_api_error_returns_fallback(
        self, engine, sample_event, sample_research
    ):
        """Test that API errors produce uniform fallback prediction."""
        with patch.object(
            engine.client.chat.completions,
            "create",
            side_effect=Exception("API connection failed"),
        ):
            result = await engine.predict(sample_event, sample_research)

        expected_prob = 1.0 / len(sample_event.outcomes)
        for outcome in sample_event.outcomes:
            assert abs(result.probabilities[outcome] - expected_prob) < 0.001

        assert result.reasoning_trace.confidence_level == "low"

    @pytest.mark.asyncio
    async def test_multi_outcome_prediction(
        self, engine, multi_outcome_event, empty_research
    ):
        """Test prediction with 3 outcomes."""
        mock_response_data = json.dumps(
            {
                "probabilities": {
                    "Party A": 0.5,
                    "Party B": 0.3,
                    "Party C": 0.2,
                },
                "reasoning_trace": {
                    "evidence_considered": ["No evidence available"],
                    "base_rate": 0.33,
                    "supporting_factors": ["Prior knowledge", "Base rates"],
                    "conflicting_evidence": [],
                    "conflict_resolution": "N/A",
                    "confidence_level": "low",
                },
            }
        )

        mock_message = MagicMock()
        mock_message.content = mock_response_data

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(engine.client.chat.completions, "create", return_value=mock_response):
            result = await engine.predict(multi_outcome_event, empty_research)

        assert len(result.probabilities) == 3
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.001
        assert all(0.01 <= p <= 0.99 for p in result.probabilities.values())


# === Tests for _fallback_prediction ===


class TestFallbackPrediction:
    def test_uniform_distribution_two_outcomes(self, engine, sample_event):
        result = engine._fallback_prediction(sample_event, 5.0)

        assert len(result.probabilities) == 2
        assert abs(result.probabilities["Yes"] - 0.5) < 0.001
        assert abs(result.probabilities["No"] - 0.5) < 0.001
        assert result.reasoning_trace.confidence_level == "low"
        assert result.duration_seconds == 5.0

    def test_uniform_distribution_three_outcomes(self, engine, multi_outcome_event):
        result = engine._fallback_prediction(multi_outcome_event, 3.0)

        assert len(result.probabilities) == 3
        expected = 1.0 / 3
        for p in result.probabilities.values():
            assert abs(p - expected) < 0.001

    def test_fallback_has_valid_trace(self, engine, sample_event):
        result = engine._fallback_prediction(sample_event, 1.0)

        assert len(result.reasoning_trace.supporting_factors) >= 2
        assert result.reasoning_trace.confidence_level == "low"
        assert result.reasoning_trace.base_rate > 0


# === Tests for _normalize_probabilities ===


class TestNormalizeProbabilities:
    def test_already_normalized(self, engine):
        raw = {"Yes": 0.7, "No": 0.3}
        result = engine._normalize_probabilities(raw, ["Yes", "No"])
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_normalizes_oversized(self, engine):
        raw = {"Yes": 0.8, "No": 0.8}
        result = engine._normalize_probabilities(raw, ["Yes", "No"])
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_clamps_to_valid_range(self, engine):
        raw = {"Yes": 1.5, "No": -0.5}
        result = engine._normalize_probabilities(raw, ["Yes", "No"])
        assert all(0.01 <= p <= 0.99 for p in result.values())
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_handles_missing_outcomes(self, engine):
        raw = {"Yes": 0.9}
        result = engine._normalize_probabilities(raw, ["Yes", "No"])
        assert "Yes" in result
        assert "No" in result
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_handles_all_zero(self, engine):
        raw = {"Yes": 0.0, "No": 0.0}
        result = engine._normalize_probabilities(raw, ["Yes", "No"])
        # Should fallback to uniform
        assert abs(result["Yes"] - 0.5) < 0.01
        assert abs(result["No"] - 0.5) < 0.01
