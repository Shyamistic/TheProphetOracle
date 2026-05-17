"""Tests for the Prophet Forecasting Agent API (src/api.py).

Tests request validation, prediction orchestration, batch processing,
timeout handling, error isolation, and budget-critical mode.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models import ComplexityTier


# Set required env vars before importing the app
os.environ.setdefault("PROPHET_ANTHROPIC_API_KEY", "test-key-123")
os.environ.setdefault("PROPHET_TAVILY_API_KEY", "test-tavily-key-456")


def _make_valid_event(ticker: str = "EVT-001") -> dict:
    """Create a valid event request dict for testing."""
    return {
        "event_ticker": ticker,
        "market_ticker": "MKT-001",
        "title": "Will it rain tomorrow?",
        "description": "Prediction about weather conditions.",
        "category": "general",
        "rules": "Resolves Yes if it rains.",
        "close_time": "2026-06-01T00:00:00Z",
        "outcomes": ["Yes", "No"],
    }


# --- Request Validation Tests ---


class TestRequestValidation:
    """Tests for request validation returning HTTP 400."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Import and create test client with mocked config."""
        with patch("src.api.load_config") as mock_load:
            from src.config import AgentConfig

            mock_config = AgentConfig()
            mock_load.return_value = mock_config
            # Need to reimport to get fresh app with mocked internals
            import importlib
            import src.api

            importlib.reload(src.api)
            self.client = TestClient(src.api.app)

    def test_predict_missing_event_ticker(self):
        """Missing required field returns 400."""
        event = _make_valid_event()
        del event["event_ticker"]
        response = self.client.post("/predict", json=event)
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "event_ticker" in body["error"]

    def test_predict_missing_outcomes(self):
        """Missing outcomes field returns 400."""
        event = _make_valid_event()
        del event["outcomes"]
        response = self.client.post("/predict", json=event)
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "outcomes" in body["error"]

    def test_predict_empty_outcomes(self):
        """Outcomes with fewer than 2 items returns 400."""
        event = _make_valid_event()
        event["outcomes"] = ["Yes"]
        response = self.client.post("/predict", json=event)
        assert response.status_code == 400
        body = response.json()
        assert "error" in body

    def test_predict_invalid_json(self):
        """Invalid JSON body returns 400."""
        response = self.client.post(
            "/predict",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_predict_null_field(self):
        """Null required field returns 400."""
        event = _make_valid_event()
        event["title"] = None
        response = self.client.post("/predict", json=event)
        assert response.status_code == 400
        body = response.json()
        assert "error" in body

    def test_predict_empty_string_field(self):
        """Empty string required field returns 400."""
        event = _make_valid_event()
        event["title"] = "   "
        response = self.client.post("/predict", json=event)
        assert response.status_code == 400

    def test_batch_not_a_list(self):
        """Batch endpoint with non-list body returns 400."""
        response = self.client.post("/predict/batch", json={"event": "data"})
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "array" in body["error"].lower()

    def test_batch_empty_list(self):
        """Batch endpoint with empty list returns 400."""
        response = self.client.post("/predict/batch", json=[])
        assert response.status_code == 400

    def test_batch_invalid_event_in_list(self):
        """Batch with one invalid event returns 400."""
        events = [_make_valid_event(), {"bad": "event"}]
        response = self.client.post("/predict/batch", json=events)
        assert response.status_code == 400
        body = response.json()
        assert "index 1" in body["error"]


# --- Prediction Orchestration Tests ---


class TestPredictionOrchestration:
    """Tests for the full prediction pipeline."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up test client with mocked pipeline components."""
        import importlib
        import src.api

        importlib.reload(src.api)
        self.client = TestClient(src.api.app)

    @patch("src.api.process_single_event")
    def test_predict_returns_valid_response(self, mock_process):
        """Valid event returns 200 with correct probability format."""
        mock_process.return_value = {"Yes": 0.6, "No": 0.4}

        # We need to make it an async function
        async def async_process(event):
            return {"Yes": 0.6, "No": 0.4}

        mock_process.side_effect = async_process

        event = _make_valid_event()
        response = self.client.post("/predict", json=event)
        assert response.status_code == 200
        body = response.json()
        assert "probabilities" in body
        assert len(body["probabilities"]) == 2

        # Check probability format
        probs = {p["market"]: p["probability"] for p in body["probabilities"]}
        assert "Yes" in probs
        assert "No" in probs
        assert abs(sum(probs.values()) - 1.0) < 0.01

    @patch("src.api.process_single_event")
    def test_predict_timeout_returns_uniform(self, mock_process):
        """Timeout falls back to uniform distribution."""

        async def slow_process(event):
            await asyncio.sleep(100)  # Will be cancelled by timeout
            return {"Yes": 0.7, "No": 0.3}

        mock_process.side_effect = slow_process

        # Temporarily set a very short timeout
        import src.api

        original_timeout = src.api.config.per_event_timeout_seconds
        src.api.config.per_event_timeout_seconds = 0.01

        try:
            event = _make_valid_event()
            response = self.client.post("/predict", json=event)
            assert response.status_code == 200
            body = response.json()
            probs = {
                p["market"]: p["probability"] for p in body["probabilities"]
            }
            # Should be uniform (0.5, 0.5) on timeout
            assert abs(probs["Yes"] - 0.5) < 0.01
            assert abs(probs["No"] - 0.5) < 0.01
        finally:
            src.api.config.per_event_timeout_seconds = original_timeout

    @patch("src.api.process_single_event")
    def test_predict_error_returns_uniform(self, mock_process):
        """Processing error falls back to uniform distribution."""

        async def failing_process(event):
            raise RuntimeError("LLM API error")

        mock_process.side_effect = failing_process

        event = _make_valid_event()
        response = self.client.post("/predict", json=event)
        assert response.status_code == 200
        body = response.json()
        probs = {p["market"]: p["probability"] for p in body["probabilities"]}
        assert abs(probs["Yes"] - 0.5) < 0.01
        assert abs(probs["No"] - 0.5) < 0.01


# --- Batch Processing Tests ---


class TestBatchProcessing:
    """Tests for batch prediction endpoint."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up test client."""
        import importlib
        import src.api

        importlib.reload(src.api)
        self.client = TestClient(src.api.app)

    @patch("src.api.process_single_event")
    def test_batch_processes_multiple_events(self, mock_process):
        """Batch endpoint processes all events and returns results."""

        async def async_process(event):
            return {o: 1.0 / len(event.outcomes) for o in event.outcomes}

        mock_process.side_effect = async_process

        events = [_make_valid_event(f"EVT-{i}") for i in range(3)]
        response = self.client.post("/predict/batch", json=events)
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 3

    @patch("src.api.process_single_event")
    def test_batch_error_isolation(self, mock_process):
        """Error in one event doesn't affect others."""
        call_count = 0

        async def mixed_process(event):
            nonlocal call_count
            call_count += 1
            if event.event_ticker == "EVT-1":
                raise RuntimeError("Simulated failure")
            return {"Yes": 0.7, "No": 0.3}

        mock_process.side_effect = mixed_process

        events = [_make_valid_event(f"EVT-{i}") for i in range(3)]
        response = self.client.post("/predict/batch", json=events)
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 3

        # EVT-1 should have uniform fallback, others should have real predictions
        probs_1 = {
            p["market"]: p["probability"] for p in body[1]["probabilities"]
        }
        assert abs(probs_1["Yes"] - 0.5) < 0.01  # Uniform fallback

        probs_0 = {
            p["market"]: p["probability"] for p in body[0]["probabilities"]
        }
        assert abs(probs_0["Yes"] - 0.7) < 0.01  # Normal result


# --- Budget Critical Mode Tests ---


class TestBudgetCriticalMode:
    """Tests for budget-critical mode switching to LOW complexity."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up test client."""
        import importlib
        import src.api

        importlib.reload(src.api)
        self.client = TestClient(src.api.app)

    @patch("src.api.run_parallel_research")
    @patch("src.api.reasoning_engine")
    @patch("src.api.cache")
    def test_budget_critical_forces_low_complexity(
        self, mock_cache, mock_reasoner, mock_research
    ):
        """When budget is critical, events use LOW complexity tier."""
        import src.api
        from src.models import PredictionResult, ReasoningTrace, ResearchResult

        # Set budget critical
        original_critical = src.api.cost_tracker.is_budget_critical
        src.api.cost_tracker.budget_usd = 10.0
        # Add records to push past threshold
        from datetime import datetime
        from src.models import APICallRecord

        src.api.cost_tracker.records = []
        src.api.cost_tracker.records.append(
            APICallRecord(
                timestamp=datetime.now(),
                service="anthropic",
                model="claude-sonnet-4",
                input_tokens=1000,
                output_tokens=500,
                estimated_cost_usd=9.5,  # 95% of $10 budget
                event_ticker="test",
                category="general",
            )
        )

        assert src.api.cost_tracker.is_budget_critical

        # Mock cache miss
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        # Mock research
        mock_research.return_value = [
            ResearchResult(
                event_ticker="EVT-001",
                evidence=[],
                search_queries_used=[],
                failed_sources=[],
                duration_seconds=1.0,
            )
        ]

        # Mock reasoner
        mock_reasoner.predict = AsyncMock(
            return_value=PredictionResult(
                event_ticker="EVT-001",
                probabilities={"Yes": 0.6, "No": 0.4},
                reasoning_trace=ReasoningTrace(
                    evidence_considered=[],
                    base_rate=0.5,
                    supporting_factors=["f1", "f2"],
                    conflicting_evidence=[],
                    conflict_resolution="none",
                    confidence_level="low",
                ),
                duration_seconds=1.0,
            )
        )

        event = _make_valid_event()
        response = self.client.post("/predict", json=event)
        assert response.status_code == 200

        # Verify research was called with LOW complexity config
        call_args = mock_research.call_args
        routing_config = call_args.kwargs.get("config") or call_args[1].get(
            "config", call_args[0][1] if len(call_args[0]) > 1 else None
        )
        assert routing_config.complexity == ComplexityTier.LOW
        assert routing_config.num_agents == 1

        # Cleanup
        src.api.cost_tracker.records = []
        src.api.cost_tracker.budget_usd = 50.0


# --- Health and Costs Endpoint Tests ---


class TestHealthAndCosts:
    """Tests for /health and /costs endpoints."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up test client."""
        import importlib
        import src.api

        importlib.reload(src.api)
        self.client = TestClient(src.api.app)

    @patch("httpx.AsyncClient.get")
    @patch("httpx.AsyncClient.post")
    def test_health_check_healthy(self, mock_post, mock_get):
        """Health check returns healthy when APIs are reachable."""
        # Mock successful responses
        mock_response_get = MagicMock()
        mock_response_get.status_code = 200
        mock_get.return_value = mock_response_get

        mock_response_post = MagicMock()
        mock_response_post.status_code = 200
        mock_post.return_value = mock_response_post

        response = self.client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "checks" in body
        assert "budget" in body

    def test_costs_endpoint(self):
        """Costs endpoint returns summary."""
        response = self.client.get("/costs")
        assert response.status_code == 200
        body = response.json()
        assert "total_spend_usd" in body
        assert "budget_usd" in body
        assert "budget_remaining_usd" in body
        assert "is_budget_critical" in body
        assert "num_records" in body


# --- Validate Event Request Function Tests ---


class TestValidateEventRequest:
    """Unit tests for the validate_event_request function."""

    def test_valid_event_no_errors(self):
        """Valid event produces no validation errors."""
        from src.api import validate_event_request

        event = _make_valid_event()
        errors = validate_event_request(event)
        assert errors == []

    def test_missing_multiple_fields(self):
        """Multiple missing fields are all reported."""
        from src.api import validate_event_request

        errors = validate_event_request({})
        assert len(errors) >= 8  # All required fields missing

    def test_outcomes_not_a_list(self):
        """Non-list outcomes field is caught."""
        from src.api import validate_event_request

        event = _make_valid_event()
        event["outcomes"] = "not a list"
        errors = validate_event_request(event)
        assert any("list" in e for e in errors)

    def test_outcomes_with_empty_string(self):
        """Empty string in outcomes is caught."""
        from src.api import validate_event_request

        event = _make_valid_event()
        event["outcomes"] = ["Yes", ""]
        errors = validate_event_request(event)
        assert any("non-empty" in e for e in errors)

    def test_non_string_field(self):
        """Non-string value for string field is caught."""
        from src.api import validate_event_request

        event = _make_valid_event()
        event["title"] = 12345
        errors = validate_event_request(event)
        assert any("string" in e for e in errors)
