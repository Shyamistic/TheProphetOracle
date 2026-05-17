"""Property-based tests for the full prediction pipeline (src/api.py).

Property 1: Pipeline Output Validity Invariant
For any valid event with N outcomes processed through the full prediction pipeline
(research → reasoning → calibration → validation), the output probabilities array
SHALL contain exactly N entries, each with a probability value in [0.01, 0.99],
market fields matching the event outcomes, and all probabilities summing to 1.0
within tolerance of 0.001.

**Validates: Requirements 1.2, 1.3, 3.3, 3.4, 4.3, 4.4, 9.1, 9.2, 9.3, 9.4, 9.7**
"""

import asyncio
import os
import random
import string
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Set required env vars before importing the app
os.environ.setdefault("PROPHET_ANTHROPIC_API_KEY", "test-key-123")
os.environ.setdefault("PROPHET_TAVILY_API_KEY", "test-tavily-key-456")

from src.models import (
    EventRequest,
    PredictionResult,
    ReasoningTrace,
    ResearchResult,
)


# --- Strategies ---

# Generate outcome labels: unique non-empty strings
outcome_label_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        min_codepoint=65,
        max_codepoint=122,
    ),
    min_size=1,
    max_size=15,
)

# Generate lists of 2-10 unique outcome labels
outcomes_strategy = st.lists(
    outcome_label_strategy,
    min_size=2,
    max_size=10,
    unique=True,
)

# Generate valid event category strings
category_strategy = st.sampled_from(
    ["Sports", "Economics", "Geopolitics", "Technology", "Science", "General",
     "sports", "SPORTS", "economics", "TECHNOLOGY", "unknown_category"]
)

# Generate non-empty description strings of varying lengths
description_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), min_codepoint=32, max_codepoint=122),
    min_size=10,
    max_size=600,
)


@st.composite
def event_strategy(draw):
    """Generate a valid EventRequest with random outcomes (2-10)."""
    outcomes = draw(outcomes_strategy)
    assume(len(outcomes) >= 2)
    # Ensure all outcomes are non-empty after strip
    assume(all(o.strip() for o in outcomes))

    category = draw(category_strategy)
    description = draw(description_strategy)
    assume(description.strip())

    title = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Z"), min_codepoint=32, max_codepoint=122),
        min_size=5,
        max_size=100,
    ))
    assume(title.strip())

    return EventRequest(
        event_ticker=f"EVT-{draw(st.integers(min_value=1, max_value=99999))}",
        market_ticker=f"MKT-{draw(st.integers(min_value=1, max_value=99999))}",
        title=title,
        description=description,
        category=category,
        rules="Standard resolution rules apply.",
        close_time="2026-06-15T00:00:00Z",
        outcomes=outcomes,
    )


def make_mock_research_result(event: EventRequest) -> ResearchResult:
    """Create a mock ResearchResult for a given event."""
    return ResearchResult(
        event_ticker=event.event_ticker,
        evidence=[],
        search_queries_used=["mock query"],
        failed_sources=[],
        duration_seconds=0.1,
    )


def make_mock_prediction_result(event: EventRequest) -> PredictionResult:
    """Create a mock PredictionResult with random valid probabilities for the event outcomes."""
    n = len(event.outcomes)
    # Generate random positive values and normalize
    raw = [random.uniform(0.05, 0.95) for _ in range(n)]
    total = sum(raw)
    probs = {outcome: val / total for outcome, val in zip(event.outcomes, raw)}

    return PredictionResult(
        event_ticker=event.event_ticker,
        probabilities=probs,
        reasoning_trace=ReasoningTrace(
            evidence_considered=["Mock evidence"],
            base_rate=1.0 / n,
            supporting_factors=["Factor 1", "Factor 2"],
            conflicting_evidence=[],
            conflict_resolution="No conflicts",
            confidence_level="medium",
        ),
        duration_seconds=0.5,
    )


# --- Property 1: Pipeline Output Validity ---


class TestPipelineOutputValidityProperty:
    """Property 1: Pipeline Output Validity Invariant.

    **Validates: Requirements 1.2, 1.3, 3.3, 3.4, 4.3, 4.4, 9.1, 9.2, 9.3, 9.4, 9.7**
    """

    @given(event=event_strategy())
    @settings(max_examples=100, deadline=None)
    def test_pipeline_produces_valid_output(self, event: EventRequest):
        """For any valid event with N outcomes, the full pipeline produces exactly N entries,
        each in [0.01, 0.99], markets matching outcomes, sum to 1.0.

        **Validates: Requirements 1.2, 1.3, 3.3, 3.4, 4.3, 4.4, 9.1, 9.2, 9.3, 9.4, 9.7**
        """
        n = len(event.outcomes)

        # Mock the research pipeline to return a mock result
        mock_research = AsyncMock(
            return_value=[make_mock_research_result(event)]
        )

        # Mock the reasoning engine to return a valid prediction
        mock_predict = AsyncMock(
            return_value=make_mock_prediction_result(event)
        )

        # Mock the cache to always miss (return None) and accept sets
        mock_cache_get = AsyncMock(return_value=None)
        mock_cache_set = AsyncMock()

        with patch("src.api.run_parallel_research", mock_research), \
             patch("src.api.reasoning_engine") as mock_reasoner, \
             patch("src.api.cache") as mock_cache:

            mock_reasoner.predict = mock_predict
            mock_cache.get = mock_cache_get
            mock_cache.set = mock_cache_set

            # Run the pipeline
            from src.api import process_single_event
            result = asyncio.run(process_single_event(event))

        # Property assertions:
        # 1. Exactly N entries (one per outcome)
        assert len(result) == n, (
            f"Expected {n} entries, got {len(result)}.\n"
            f"Outcomes: {event.outcomes}\n"
            f"Result keys: {list(result.keys())}"
        )

        # 2. Each probability in [0.01, 0.99]
        for outcome, prob in result.items():
            assert 0.01 - 1e-9 <= prob <= 0.99 + 1e-9, (
                f"Probability for '{outcome}' is {prob}, outside [0.01, 0.99].\n"
                f"Full result: {result}"
            )

        # 3. Market fields match outcomes exactly
        assert set(result.keys()) == set(event.outcomes), (
            f"Result keys {set(result.keys())} don't match outcomes {set(event.outcomes)}.\n"
            f"Result: {result}"
        )

        # 4. Sum to 1.0 within tolerance of 0.001
        total = sum(result.values())
        assert abs(total - 1.0) <= 0.001, (
            f"Probabilities sum to {total}, expected 1.0 (tolerance 0.001).\n"
            f"Result: {result}"
        )

    @given(event=event_strategy())
    @settings(max_examples=50, deadline=None)
    def test_pipeline_valid_with_skewed_reasoning(self, event: EventRequest):
        """Pipeline produces valid output even when reasoning returns skewed probabilities.

        Tests that calibration and validation correct extreme values from the reasoner.

        **Validates: Requirements 1.2, 1.3, 3.3, 3.4, 4.3, 4.4, 9.1, 9.2, 9.3, 9.4, 9.7**
        """
        n = len(event.outcomes)

        # Create a skewed prediction: one outcome gets most probability
        probs = {}
        for i, outcome in enumerate(event.outcomes):
            if i == 0:
                probs[outcome] = 0.95
            else:
                probs[outcome] = 0.05 / (n - 1)

        skewed_prediction = PredictionResult(
            event_ticker=event.event_ticker,
            probabilities=probs,
            reasoning_trace=ReasoningTrace(
                evidence_considered=["Skewed evidence"],
                base_rate=1.0 / n,
                supporting_factors=["Strong factor 1", "Strong factor 2"],
                conflicting_evidence=[],
                conflict_resolution="No conflicts",
                confidence_level="high",
            ),
            duration_seconds=0.3,
        )

        mock_research = AsyncMock(
            return_value=[make_mock_research_result(event)]
        )
        mock_predict = AsyncMock(return_value=skewed_prediction)
        mock_cache_get = AsyncMock(return_value=None)
        mock_cache_set = AsyncMock()

        with patch("src.api.run_parallel_research", mock_research), \
             patch("src.api.reasoning_engine") as mock_reasoner, \
             patch("src.api.cache") as mock_cache:

            mock_reasoner.predict = mock_predict
            mock_cache.get = mock_cache_get
            mock_cache.set = mock_cache_set

            from src.api import process_single_event
            result = asyncio.run(process_single_event(event))

        # Same validity assertions
        assert len(result) == n
        assert set(result.keys()) == set(event.outcomes)
        for outcome, prob in result.items():
            assert 0.01 - 1e-9 <= prob <= 0.99 + 1e-9, (
                f"Probability for '{outcome}' is {prob}, outside [0.01, 0.99]."
            )
        total = sum(result.values())
        assert abs(total - 1.0) <= 0.001, (
            f"Probabilities sum to {total}, expected 1.0."
        )

    @given(event=event_strategy())
    @settings(max_examples=50, deadline=None)
    def test_pipeline_valid_with_uniform_reasoning(self, event: EventRequest):
        """Pipeline produces valid output when reasoning returns uniform probabilities.

        **Validates: Requirements 1.2, 1.3, 3.3, 3.4, 4.3, 4.4, 9.1, 9.2, 9.3, 9.4, 9.7**
        """
        n = len(event.outcomes)

        # Uniform prediction
        uniform_prob = 1.0 / n
        probs = {outcome: uniform_prob for outcome in event.outcomes}

        uniform_prediction = PredictionResult(
            event_ticker=event.event_ticker,
            probabilities=probs,
            reasoning_trace=ReasoningTrace(
                evidence_considered=["No strong evidence"],
                base_rate=uniform_prob,
                supporting_factors=["Uniform prior", "No differentiating info"],
                conflicting_evidence=[],
                conflict_resolution="N/A",
                confidence_level="low",
            ),
            duration_seconds=0.2,
        )

        mock_research = AsyncMock(
            return_value=[make_mock_research_result(event)]
        )
        mock_predict = AsyncMock(return_value=uniform_prediction)
        mock_cache_get = AsyncMock(return_value=None)
        mock_cache_set = AsyncMock()

        with patch("src.api.run_parallel_research", mock_research), \
             patch("src.api.reasoning_engine") as mock_reasoner, \
             patch("src.api.cache") as mock_cache:

            mock_reasoner.predict = mock_predict
            mock_cache.get = mock_cache_get
            mock_cache.set = mock_cache_set

            from src.api import process_single_event
            result = asyncio.run(process_single_event(event))

        # Validity assertions
        assert len(result) == n
        assert set(result.keys()) == set(event.outcomes)
        for outcome, prob in result.items():
            assert 0.01 - 1e-9 <= prob <= 0.99 + 1e-9
        total = sum(result.values())
        assert abs(total - 1.0) <= 0.001


# --- Strategies for Property 7 ---


def budget_event_request_strategy(
    min_outcomes: int = 2,
    max_outcomes: int = 10,
    min_desc_len: int = 0,
    max_desc_len: int = 1000,
):
    """Generate random EventRequest objects with varying complexity characteristics."""
    return st.builds(
        EventRequest,
        event_ticker=st.text(
            alphabet=string.ascii_lowercase + string.digits, min_size=3, max_size=10
        ),
        market_ticker=st.text(
            alphabet=string.ascii_lowercase + string.digits, min_size=3, max_size=10
        ),
        title=st.text(alphabet=string.ascii_letters + " ", min_size=5, max_size=100),
        description=st.text(
            alphabet=string.ascii_letters + " .,!?",
            min_size=min_desc_len,
            max_size=max_desc_len,
        ),
        category=st.sampled_from(
            [
                "Sports",
                "Economics",
                "Geopolitics",
                "Technology",
                "Science",
                "General",
                "sports",
                "ECONOMICS",
                "Unknown",
                "random_category",
            ]
        ),
        rules=st.text(alphabet=string.ascii_letters + " ", min_size=5, max_size=100),
        close_time=st.just("2026-05-30T00:00:00Z"),
        outcomes=st.lists(
            st.text(alphabet=string.ascii_letters, min_size=2, max_size=15),
            min_size=min_outcomes,
            max_size=max_outcomes,
        ),
        resolved_outcome=st.none(),
    )


# --- Property 7: API Call Budget Enforcement ---

from src.router import classify_event


class TestAPICallBudgetEnforcement:
    """Property 7: API Call Budget Enforcement.

    **Validates: Requirements 6.3, 6.4**

    Tests that total LLM calls ≤ 5 and search calls ≤ 3 per event
    regardless of complexity tier or category.
    """

    @given(event=budget_event_request_strategy())
    @settings(max_examples=300)
    def test_max_llm_calls_never_exceeds_5(self, event: EventRequest):
        """For any event, the routing config max_llm_calls is at most 5.

        Requirement 6.3: The Agent SHALL limit the number of LLM API calls
        per event to a configurable maximum (default: 5 calls per event).
        """
        config = classify_event(event)
        assert config.max_llm_calls <= 5, (
            f"max_llm_calls={config.max_llm_calls} exceeds limit of 5 "
            f"for event with {len(event.outcomes)} outcomes, "
            f"desc_len={len(event.description)}, category='{event.category}', "
            f"complexity={config.complexity.value}"
        )

    @given(event=budget_event_request_strategy())
    @settings(max_examples=300)
    def test_max_searches_never_exceeds_3(self, event: EventRequest):
        """For any event, the routing config max_searches is at most 3.

        Requirement 6.4: The Agent SHALL limit the number of web search API
        calls per event to a configurable maximum (default: 3 searches per event).
        """
        config = classify_event(event)
        assert config.max_searches <= 3, (
            f"max_searches={config.max_searches} exceeds limit of 3 "
            f"for event with {len(event.outcomes)} outcomes, "
            f"desc_len={len(event.description)}, category='{event.category}', "
            f"complexity={config.complexity.value}"
        )

    @given(event=budget_event_request_strategy())
    @settings(max_examples=300)
    def test_llm_calls_at_least_2(self, event: EventRequest):
        """For any event, the routing config allocates at least 2 LLM calls.

        Even the lowest complexity tier requires at least 2 LLM calls
        (entity extraction + reasoning).
        """
        config = classify_event(event)
        assert config.max_llm_calls >= 2, (
            f"max_llm_calls={config.max_llm_calls} is below minimum of 2 "
            f"for complexity={config.complexity.value}"
        )

    @given(event=budget_event_request_strategy())
    @settings(max_examples=300)
    def test_searches_at_least_1(self, event: EventRequest):
        """For any event, the routing config allocates at least 1 search call.

        Even the lowest complexity tier requires at least 1 search.
        """
        config = classify_event(event)
        assert config.max_searches >= 1, (
            f"max_searches={config.max_searches} is below minimum of 1 "
            f"for complexity={config.complexity.value}"
        )

    @given(event=budget_event_request_strategy())
    @settings(max_examples=300)
    def test_budget_limits_hold_across_all_complexity_tiers(self, event: EventRequest):
        """Combined assertion: for any event, both LLM and search limits hold simultaneously.

        This tests the conjunction of Requirements 6.3 and 6.4: regardless of
        how the event is classified (LOW, MEDIUM, HIGH), the routing config
        respects both budget constraints at the same time.
        """
        config = classify_event(event)
        assert config.max_llm_calls <= 5 and config.max_searches <= 3, (
            f"Budget limits violated: max_llm_calls={config.max_llm_calls} (limit 5), "
            f"max_searches={config.max_searches} (limit 3) "
            f"for complexity={config.complexity.value}, category={config.category.value}"
        )
        # Also verify the num_agents is bounded (agents drive API calls)
        assert config.num_agents <= 3, (
            f"num_agents={config.num_agents} exceeds maximum of 3"
        )



# === Property 12: Invalid Input Rejection ===


# Required fields for a valid event request
REQUIRED_FIELDS = [
    "event_ticker",
    "market_ticker",
    "title",
    "description",
    "category",
    "rules",
    "close_time",
    "outcomes",
]


def _make_valid_event() -> dict:
    """Create a valid event request dict as a baseline."""
    return {
        "event_ticker": "EVT-001",
        "market_ticker": "MKT-001",
        "title": "Will it rain tomorrow?",
        "description": "Prediction about weather conditions.",
        "category": "general",
        "rules": "Resolves Yes if it rains.",
        "close_time": "2026-06-01T00:00:00Z",
        "outcomes": ["Yes", "No"],
    }


# Strategy for generating a non-empty subset of required fields to remove
fields_to_remove_strategy = st.lists(
    st.sampled_from(REQUIRED_FIELDS),
    min_size=1,
    max_size=len(REQUIRED_FIELDS),
    unique=True,
)

# Strategy for invalid field values (non-string types for string fields)
invalid_string_values_strategy = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.integers(), min_size=0, max_size=3),
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=0, max_size=3),
    st.booleans(),
)

# Strategy for invalid outcomes values
invalid_outcomes_strategy = st.one_of(
    # Not a list
    st.text(min_size=1, max_size=20),
    st.integers(),
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=3),
    # List with fewer than 2 items
    st.just([]),
    st.just(["OnlyOne"]),
    # List with non-string items
    st.lists(st.integers(), min_size=2, max_size=5),
    # List with empty strings
    st.just(["Yes", ""]),
    st.just(["", ""]),
    st.just(["Valid", "   "]),
)

# Strategy for generating random malformed JSON bodies
random_json_values_strategy = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1000, max_value=1000),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
        st.text(min_size=0, max_size=20),
    ),
    lambda children: st.one_of(
        st.lists(children, min_size=0, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, min_size=0, max_size=5),
    ),
    max_leaves=20,
)


class TestInvalidInputRejection:
    """Property 12: Invalid Input Rejection.

    **Validates: Requirements 1.6**

    For any request with missing required fields or malformed event JSON,
    the API SHALL return HTTP 400 with a JSON body containing an error field
    describing which fields are missing or malformed.
    """

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Create test client."""
        from fastapi.testclient import TestClient as TC
        import importlib
        import src.api

        importlib.reload(src.api)
        self.client = TC(src.api.app)

    @given(removed_fields=fields_to_remove_strategy)
    @settings(max_examples=100, deadline=None)
    def test_missing_required_fields_returns_400(self, removed_fields):
        """Removing any subset of required fields from a valid event returns HTTP 400
        with an error description mentioning the missing fields.

        **Validates: Requirements 1.6**
        """
        event = _make_valid_event()
        for field in removed_fields:
            del event[field]

        response = self.client.post("/predict", json=event)

        # Must return 400
        assert response.status_code == 400, (
            f"Expected 400 for missing fields {removed_fields}, got {response.status_code}"
        )

        # Must contain error field
        body = response.json()
        assert "error" in body, (
            f"Response body missing 'error' field. Body: {body}"
        )

        # Error description must mention at least one of the missing fields
        error_msg = body["error"].lower()
        mentioned = any(field.lower() in error_msg for field in removed_fields)
        assert mentioned, (
            f"Error message does not mention any of the missing fields {removed_fields}. "
            f"Error: {body['error']}"
        )

    @given(
        field=st.sampled_from([f for f in REQUIRED_FIELDS if f != "outcomes"]),
        value=invalid_string_values_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_type_for_string_fields_returns_400(self, field, value):
        """Setting a string field to a non-string type returns HTTP 400 with error description.

        **Validates: Requirements 1.6**
        """
        event = _make_valid_event()
        event[field] = value

        response = self.client.post("/predict", json=event)

        assert response.status_code == 400, (
            f"Expected 400 for field '{field}' = {value!r} (type {type(value).__name__}), "
            f"got {response.status_code}"
        )

        body = response.json()
        assert "error" in body, (
            f"Response body missing 'error' field. Body: {body}"
        )

    @given(outcomes=invalid_outcomes_strategy)
    @settings(max_examples=100, deadline=None)
    def test_invalid_outcomes_returns_400(self, outcomes):
        """Invalid outcomes field (wrong type, too few items, non-string items) returns HTTP 400.

        **Validates: Requirements 1.6**
        """
        event = _make_valid_event()
        event["outcomes"] = outcomes

        response = self.client.post("/predict", json=event)

        assert response.status_code == 400, (
            f"Expected 400 for outcomes = {outcomes!r}, got {response.status_code}"
        )

        body = response.json()
        assert "error" in body, (
            f"Response body missing 'error' field. Body: {body}"
        )

    @given(
        field=st.sampled_from([f for f in REQUIRED_FIELDS if f != "outcomes"]),
    )
    @settings(max_examples=50, deadline=None)
    def test_null_required_field_returns_400(self, field):
        """Setting any required field to null returns HTTP 400 with error description.

        **Validates: Requirements 1.6**
        """
        event = _make_valid_event()
        event[field] = None

        response = self.client.post("/predict", json=event)

        assert response.status_code == 400, (
            f"Expected 400 for field '{field}' = None, got {response.status_code}"
        )

        body = response.json()
        assert "error" in body, (
            f"Response body missing 'error' field. Body: {body}"
        )

    @given(malformed_body=st.dictionaries(
        st.text(min_size=1, max_size=15),
        random_json_values_strategy,
        min_size=0,
        max_size=8,
    ))
    @settings(max_examples=100, deadline=None)
    def test_random_malformed_json_returns_400(self, malformed_body):
        """Random malformed JSON bodies (not matching event schema) return HTTP 400.

        **Validates: Requirements 1.6**
        """
        from hypothesis import assume
        from src.api import validate_event_request

        # Ensure the body is actually invalid by checking it fails validation
        errors = validate_event_request(malformed_body)
        assume(len(errors) > 0)

        response = self.client.post("/predict", json=malformed_body)

        assert response.status_code == 400, (
            f"Expected 400 for malformed body {malformed_body}, got {response.status_code}"
        )

        body = response.json()
        assert "error" in body, (
            f"Response body missing 'error' field. Body: {body}"
        )
        # Error must be a non-empty string describing the issue
        assert isinstance(body["error"], str) and len(body["error"]) > 0, (
            f"Error field must be a non-empty string. Got: {body['error']!r}"
        )
