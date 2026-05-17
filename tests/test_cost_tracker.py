"""Tests for cost tracker module (src/cost_tracker.py)."""

import os
import tempfile
from datetime import datetime

import pytest

from src.cost_tracker import (
    AVG_LLM_CALL_COST_USD,
    AVG_SEARCH_COST_USD,
    CostTracker,
)
from src.models import APICallRecord, ComplexityTier, EventCategory, RoutingConfig


@pytest.fixture
def tmp_db_path():
    """Provide a temporary database path and clean up after test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # On Windows, SQLite WAL/journal files may hold locks briefly.
    # Attempt cleanup but don't fail the test if the file is still locked.
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


@pytest.fixture
def tracker(tmp_db_path):
    """Create a CostTracker with a temporary database."""
    return CostTracker(budget_usd=50.0, alert_threshold=0.90, db_path=tmp_db_path)


def make_record(
    cost: float = 0.01,
    service: str = "anthropic",
    model: str = "claude-sonnet-4",
    category: str = "sports",
    event_ticker: str = "EVT-001",
    input_tokens: int = 1000,
    output_tokens: int = 500,
) -> APICallRecord:
    """Helper to create an APICallRecord with sensible defaults."""
    return APICallRecord(
        timestamp=datetime.now(),
        service=service,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        event_ticker=event_ticker,
        category=category,
    )


class TestCostTrackerInit:
    """Tests for CostTracker initialization."""

    def test_default_parameters(self, tmp_db_path):
        """Uses default budget and alert threshold."""
        tracker = CostTracker(db_path=tmp_db_path)
        assert tracker.budget_usd == 50.0
        assert tracker.alert_threshold == 0.90

    def test_custom_parameters(self, tmp_db_path):
        """Accepts custom budget and alert threshold."""
        tracker = CostTracker(
            budget_usd=100.0, alert_threshold=0.80, db_path=tmp_db_path
        )
        assert tracker.budget_usd == 100.0
        assert tracker.alert_threshold == 0.80

    def test_starts_with_empty_records(self, tracker):
        """New tracker starts with no records."""
        assert len(tracker.records) == 0
        assert tracker.total_spend == 0.0

    def test_creates_database_table(self, tmp_db_path):
        """Initializing creates the cost_records table."""
        import sqlite3

        CostTracker(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cost_records'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestRecordCall:
    """Tests for record_call method."""

    def test_appends_to_records(self, tracker):
        """Recording a call adds it to the records list."""
        record = make_record(cost=0.005)
        tracker.record_call(record)
        assert len(tracker.records) == 1
        assert tracker.records[0] is record

    def test_multiple_records(self, tracker):
        """Multiple calls accumulate in records."""
        for i in range(5):
            tracker.record_call(make_record(cost=0.01))
        assert len(tracker.records) == 5

    def test_persists_to_database(self, tmp_db_path):
        """Records are persisted to SQLite and survive reload."""
        tracker = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        tracker.record_call(make_record(cost=0.015, service="tavily", category="economics"))

        # Create a new tracker pointing to the same DB
        tracker2 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        assert len(tracker2.records) == 1
        assert tracker2.records[0].service == "tavily"
        assert tracker2.records[0].category == "economics"
        assert tracker2.records[0].estimated_cost_usd == pytest.approx(0.015)


class TestTotalSpend:
    """Tests for total_spend property."""

    def test_zero_with_no_records(self, tracker):
        """Total spend is 0 with no records."""
        assert tracker.total_spend == 0.0

    def test_sum_of_costs(self, tracker):
        """Total spend equals sum of all record costs."""
        tracker.record_call(make_record(cost=0.01))
        tracker.record_call(make_record(cost=0.02))
        tracker.record_call(make_record(cost=0.03))
        assert tracker.total_spend == pytest.approx(0.06)


class TestBudgetRemaining:
    """Tests for budget_remaining property."""

    def test_full_budget_when_empty(self, tracker):
        """Budget remaining equals full budget with no spend."""
        assert tracker.budget_remaining == 50.0

    def test_decreases_with_spend(self, tracker):
        """Budget remaining decreases as costs are recorded."""
        tracker.record_call(make_record(cost=10.0))
        assert tracker.budget_remaining == pytest.approx(40.0)

    def test_can_go_negative(self, tracker):
        """Budget remaining can go negative if overspent."""
        tracker.record_call(make_record(cost=60.0))
        assert tracker.budget_remaining == pytest.approx(-10.0)


class TestIsBudgetCritical:
    """Tests for is_budget_critical property."""

    def test_false_when_under_threshold(self, tracker):
        """Not critical when spend is below threshold."""
        tracker.record_call(make_record(cost=10.0))  # 20% of $50
        assert tracker.is_budget_critical is False

    def test_true_at_threshold(self, tracker):
        """Critical when spend equals exactly threshold."""
        tracker.record_call(make_record(cost=45.0))  # 90% of $50
        assert tracker.is_budget_critical is True

    def test_true_above_threshold(self, tracker):
        """Critical when spend exceeds threshold."""
        tracker.record_call(make_record(cost=48.0))  # 96% of $50
        assert tracker.is_budget_critical is True

    def test_custom_threshold(self, tmp_db_path):
        """Works with custom alert threshold."""
        tracker = CostTracker(
            budget_usd=100.0, alert_threshold=0.50, db_path=tmp_db_path
        )
        tracker.record_call(make_record(cost=49.0))
        assert tracker.is_budget_critical is False
        tracker.record_call(make_record(cost=2.0))
        assert tracker.is_budget_critical is True


class TestSpendByCategory:
    """Tests for spend_by_category method."""

    def test_empty_when_no_records(self, tracker):
        """Returns empty dict with no records."""
        assert tracker.spend_by_category() == {}

    def test_single_category(self, tracker):
        """Correctly sums for a single category."""
        tracker.record_call(make_record(cost=0.01, category="sports"))
        tracker.record_call(make_record(cost=0.02, category="sports"))
        result = tracker.spend_by_category()
        assert result == {"sports": pytest.approx(0.03)}

    def test_multiple_categories(self, tracker):
        """Correctly breaks down across multiple categories."""
        tracker.record_call(make_record(cost=0.01, category="sports"))
        tracker.record_call(make_record(cost=0.02, category="economics"))
        tracker.record_call(make_record(cost=0.03, category="sports"))
        result = tracker.spend_by_category()
        assert result["sports"] == pytest.approx(0.04)
        assert result["economics"] == pytest.approx(0.02)

    def test_all_categories_present(self, tracker):
        """All recorded categories appear in breakdown."""
        categories = ["sports", "economics", "geopolitics", "technology", "science"]
        for i, cat in enumerate(categories):
            tracker.record_call(make_record(cost=0.01 * (i + 1), category=cat))
        result = tracker.spend_by_category()
        assert len(result) == 5
        assert all(cat in result for cat in categories)


class TestEstimateEventCost:
    """Tests for estimate_event_cost method."""

    def test_low_complexity(self, tracker):
        """Low complexity event has lowest estimated cost."""
        config = RoutingConfig(
            category=EventCategory.SPORTS,
            complexity=ComplexityTier.LOW,
            num_agents=1,
            max_searches=1,
            max_llm_calls=2,
            search_strategies=["query1"],
        )
        cost = tracker.estimate_event_cost(config)
        expected = 1 * 2 * AVG_LLM_CALL_COST_USD + 1 * AVG_SEARCH_COST_USD
        assert cost == pytest.approx(expected)

    def test_high_complexity(self, tracker):
        """High complexity event has highest estimated cost."""
        config = RoutingConfig(
            category=EventCategory.GEOPOLITICS,
            complexity=ComplexityTier.HIGH,
            num_agents=3,
            max_searches=3,
            max_llm_calls=5,
            search_strategies=["q1", "q2", "q3"],
        )
        cost = tracker.estimate_event_cost(config)
        expected = 3 * 5 * AVG_LLM_CALL_COST_USD + 3 * AVG_SEARCH_COST_USD
        assert cost == pytest.approx(expected)

    def test_cost_increases_with_agents(self, tracker):
        """More agents means higher estimated cost."""
        config_low = RoutingConfig(
            category=EventCategory.GENERAL,
            complexity=ComplexityTier.LOW,
            num_agents=1,
            max_searches=1,
            max_llm_calls=2,
            search_strategies=["q1"],
        )
        config_high = RoutingConfig(
            category=EventCategory.GENERAL,
            complexity=ComplexityTier.HIGH,
            num_agents=3,
            max_searches=1,
            max_llm_calls=2,
            search_strategies=["q1"],
        )
        assert tracker.estimate_event_cost(config_high) > tracker.estimate_event_cost(
            config_low
        )


class TestGetSummary:
    """Tests for get_summary method."""

    def test_empty_summary(self, tracker):
        """Summary with no records has zero spend."""
        summary = tracker.get_summary()
        assert summary["total_spend_usd"] == 0.0
        assert summary["budget_usd"] == 50.0
        assert summary["budget_remaining_usd"] == 50.0
        assert summary["is_budget_critical"] is False
        assert summary["alert_threshold"] == 0.90
        assert summary["num_records"] == 0
        assert summary["spend_by_category"] == {}
        assert summary["spend_by_service"] == {}

    def test_summary_with_records(self, tracker):
        """Summary reflects recorded calls."""
        tracker.record_call(
            make_record(cost=0.01, service="anthropic", category="sports")
        )
        tracker.record_call(
            make_record(cost=0.005, service="tavily", category="economics")
        )
        summary = tracker.get_summary()
        assert summary["total_spend_usd"] == pytest.approx(0.015)
        assert summary["budget_remaining_usd"] == pytest.approx(50.0 - 0.015)
        assert summary["num_records"] == 2
        assert summary["spend_by_category"]["sports"] == pytest.approx(0.01)
        assert summary["spend_by_category"]["economics"] == pytest.approx(0.005)
        assert summary["spend_by_service"]["anthropic"] == pytest.approx(0.01)
        assert summary["spend_by_service"]["tavily"] == pytest.approx(0.005)

    def test_summary_keys_present(self, tracker):
        """Summary contains all expected keys."""
        summary = tracker.get_summary()
        expected_keys = {
            "total_spend_usd",
            "budget_usd",
            "budget_remaining_usd",
            "is_budget_critical",
            "alert_threshold",
            "num_records",
            "spend_by_category",
            "spend_by_service",
        }
        assert set(summary.keys()) == expected_keys


class TestPersistence:
    """Tests for SQLite persistence across tracker instances."""

    def test_records_survive_restart(self, tmp_db_path):
        """Records persist across CostTracker instances."""
        tracker1 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        tracker1.record_call(make_record(cost=0.01, category="sports"))
        tracker1.record_call(make_record(cost=0.02, category="economics"))

        tracker2 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        assert len(tracker2.records) == 2
        assert tracker2.total_spend == pytest.approx(0.03)

    def test_total_spend_persists(self, tmp_db_path):
        """Total spend is correctly computed from persisted records."""
        tracker1 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        for i in range(10):
            tracker1.record_call(make_record(cost=0.005))

        tracker2 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        assert tracker2.total_spend == pytest.approx(0.05)

    def test_category_breakdown_persists(self, tmp_db_path):
        """Category breakdown is correct after reload."""
        tracker1 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        tracker1.record_call(make_record(cost=0.01, category="sports"))
        tracker1.record_call(make_record(cost=0.02, category="technology"))

        tracker2 = CostTracker(budget_usd=50.0, db_path=tmp_db_path)
        breakdown = tracker2.spend_by_category()
        assert breakdown["sports"] == pytest.approx(0.01)
        assert breakdown["technology"] == pytest.approx(0.02)
