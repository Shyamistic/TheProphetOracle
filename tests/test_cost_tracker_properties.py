"""Property-based tests for CostTracker (src/cost_tracker.py).

Property 8: Cost Tracking Accumulation
For any sequence of recorded API calls with known costs, the cumulative total
reported by the cost tracker SHALL equal the sum of all individual call costs,
and the per-category breakdown SHALL equal the sum of costs for calls in each category.

Property 9: Budget Threshold Tier Switch
For any budget amount B and sequence of API calls whose cumulative cost
reaches or exceeds 0.9 * B, the is_budget_critical flag SHALL become True.

Validates: Requirements 6.6, 6.7
"""

# Feature: prophet-forecasting-agent, Property 8: Cost Tracking Accumulation
# Feature: prophet-forecasting-agent, Property 9: Budget Threshold Tier Switch

import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.cost_tracker import CostTracker
from src.models import APICallRecord


# === Strategies ===

SERVICES = ["anthropic", "openai", "tavily"]
MODELS = ["claude-sonnet-4", "gpt-4o-mini", "tavily-search", "claude-haiku"]
CATEGORIES = ["sports", "economics", "geopolitics", "technology", "science", "general"]


@st.composite
def api_call_record_strategy(draw):
    """Generate a random APICallRecord with realistic values."""
    return APICallRecord(
        timestamp=datetime(2025, 5, 1) + timedelta(seconds=draw(st.integers(min_value=0, max_value=100000))),
        service=draw(st.sampled_from(SERVICES)),
        model=draw(st.sampled_from(MODELS)),
        input_tokens=draw(st.integers(min_value=1, max_value=100000)),
        output_tokens=draw(st.integers(min_value=1, max_value=100000)),
        estimated_cost_usd=draw(st.floats(min_value=0.0001, max_value=1.0, allow_nan=False, allow_infinity=False)),
        event_ticker=f"EVT-{draw(st.integers(min_value=1, max_value=999)):03d}",
        category=draw(st.sampled_from(CATEGORIES)),
    )


# Generate sequences of 1-50 API call records
api_call_sequences = st.lists(
    api_call_record_strategy(),
    min_size=1,
    max_size=50,
)


# === Property 8: Cost Tracking Accumulation ===


class TestCostTrackingAccumulationProperty:
    """Property 8: Cost Tracking Accumulation.

    For any sequence of recorded API calls with known costs, the cumulative total
    reported by the cost tracker SHALL equal the sum of all individual call costs,
    and the per-category breakdown SHALL equal the sum of costs for calls in each category.

    **Validates: Requirements 6.6**
    """

    def _make_tracker(self):
        """Create a CostTracker with a fresh temp database."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        tracker = CostTracker(budget_usd=1000.0, alert_threshold=0.99, db_path=path)
        return tracker, path

    def _cleanup(self, path):
        """Clean up temp database file."""
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            pass

    @given(records=api_call_sequences)
    @settings(max_examples=200, deadline=None)
    def test_cumulative_total_equals_sum_of_individual_costs(self, records):
        """Cumulative total reported by cost tracker equals sum of all individual call costs.

        **Validates: Requirements 6.6**
        """
        tracker, db_path = self._make_tracker()
        try:
            for record in records:
                tracker.record_call(record)

            expected_total = sum(r.estimated_cost_usd for r in records)

            assert abs(tracker.total_spend - expected_total) < 1e-9, (
                f"Total spend {tracker.total_spend} != expected {expected_total}.\n"
                f"Num records: {len(records)}"
            )
        finally:
            self._cleanup(db_path)

    @given(records=api_call_sequences)
    @settings(max_examples=200, deadline=None)
    def test_per_category_breakdown_equals_sum_per_category(self, records):
        """Per-category breakdown equals the sum of costs for calls in each category.

        **Validates: Requirements 6.6**
        """
        tracker, db_path = self._make_tracker()
        try:
            for record in records:
                tracker.record_call(record)

            expected_by_category = defaultdict(float)
            for record in records:
                expected_by_category[record.category] += record.estimated_cost_usd

            actual_by_category = tracker.spend_by_category()

            # Same set of categories
            assert set(actual_by_category.keys()) == set(expected_by_category.keys()), (
                f"Category keys mismatch.\n"
                f"Actual: {set(actual_by_category.keys())}\n"
                f"Expected: {set(expected_by_category.keys())}"
            )

            # Each category total matches
            for category in expected_by_category:
                assert abs(actual_by_category[category] - expected_by_category[category]) < 1e-9, (
                    f"Category '{category}' spend mismatch.\n"
                    f"Actual: {actual_by_category[category]}\n"
                    f"Expected: {expected_by_category[category]}"
                )
        finally:
            self._cleanup(db_path)

    @given(records=api_call_sequences)
    @settings(max_examples=200, deadline=None)
    def test_category_breakdown_sums_to_total(self, records):
        """Sum of all per-category costs equals the cumulative total.

        **Validates: Requirements 6.6**
        """
        tracker, db_path = self._make_tracker()
        try:
            for record in records:
                tracker.record_call(record)

            category_sum = sum(tracker.spend_by_category().values())
            assert abs(category_sum - tracker.total_spend) < 1e-9, (
                f"Category breakdown sum {category_sum} != total_spend {tracker.total_spend}.\n"
                f"Breakdown: {tracker.spend_by_category()}"
            )
        finally:
            self._cleanup(db_path)


# === Property 9: Budget Threshold Tier Switch ===


class TestBudgetThresholdTierSwitch:
    """Property 9: Budget Threshold Tier Switch.

    For any budget amount B and sequence of API calls whose cumulative cost
    reaches or exceeds 0.9 * B, the is_budget_critical flag SHALL become True.
    Events processed before the threshold is crossed SHALL have is_budget_critical
    as False.

    **Validates: Requirements 6.7**
    """

    @settings(max_examples=200, deadline=None)
    @given(
        budget=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        calls=api_call_sequences,
    )
    def test_is_budget_critical_becomes_true_at_threshold(self, budget, calls):
        """When cumulative cost reaches 90% of budget, is_budget_critical becomes True.

        **Validates: Requirements 6.7**
        """
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            tracker = CostTracker(budget_usd=budget, alert_threshold=0.90, db_path=db_path)
            threshold = budget * 0.90

            for call in calls:
                tracker.record_call(call)

            if tracker.total_spend >= threshold:
                assert tracker.is_budget_critical is True, (
                    f"Expected is_budget_critical=True when spend "
                    f"({tracker.total_spend:.4f}) >= threshold ({threshold:.4f})"
                )
            else:
                assert tracker.is_budget_critical is False, (
                    f"Expected is_budget_critical=False when spend "
                    f"({tracker.total_spend:.4f}) < threshold ({threshold:.4f})"
                )
        finally:
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                pass

    @settings(max_examples=200, deadline=None)
    @given(
        budget=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    def test_threshold_transition_point(self, budget):
        """The transition from non-critical to critical happens exactly at 90% of budget.

        **Validates: Requirements 6.7**
        """
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            tracker = CostTracker(budget_usd=budget, alert_threshold=0.90, db_path=db_path)
            threshold = budget * 0.90

            below_cost = threshold * 0.99
            below_record = APICallRecord(
                timestamp=datetime.now(),
                service="anthropic",
                model="claude-sonnet-4",
                input_tokens=1000,
                output_tokens=500,
                estimated_cost_usd=below_cost,
                event_ticker="EVT-001",
                category="sports",
            )
            tracker.record_call(below_record)

            assert tracker.is_budget_critical is False, (
                f"Should not be critical at spend={tracker.total_spend:.4f}, "
                f"threshold={threshold:.4f}"
            )

            push_over_cost = threshold - below_cost + 0.01
            push_record = APICallRecord(
                timestamp=datetime.now(),
                service="openai",
                model="gpt-4o-mini",
                input_tokens=500,
                output_tokens=200,
                estimated_cost_usd=push_over_cost,
                event_ticker="EVT-002",
                category="economics",
            )
            tracker.record_call(push_record)

            assert tracker.is_budget_critical is True, (
                f"Should be critical at spend={tracker.total_spend:.4f}, "
                f"threshold={threshold:.4f}"
            )
        finally:
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                pass

    @settings(max_examples=200, deadline=None)
    @given(
        budget=st.floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        num_calls=st.integers(min_value=1, max_value=30),
    )
    def test_critical_flag_monotonic(self, budget, num_calls):
        """Once is_budget_critical becomes True, it stays True for all subsequent calls.

        **Validates: Requirements 6.7**
        """
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            tracker = CostTracker(budget_usd=budget, alert_threshold=0.90, db_path=db_path)
            became_critical = False

            for i in range(num_calls):
                cost_per_call = (budget * 0.95) / num_calls
                record = APICallRecord(
                    timestamp=datetime.now() + timedelta(seconds=i),
                    service="anthropic",
                    model="claude-sonnet-4",
                    input_tokens=1000,
                    output_tokens=500,
                    estimated_cost_usd=cost_per_call,
                    event_ticker=f"EVT-{i:03d}",
                    category="sports",
                )
                tracker.record_call(record)

                if tracker.is_budget_critical:
                    became_critical = True

                if became_critical:
                    assert tracker.is_budget_critical is True, (
                        f"is_budget_critical went from True to False after call {i}. "
                        f"Spend={tracker.total_spend:.4f}, threshold={budget * 0.90:.4f}"
                    )
        finally:
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                pass
