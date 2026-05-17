"""Unit tests for the ensemble aggregator module."""

import pytest

from src.aggregator import aggregate_predictions, weighted_aggregate
from src.models import PredictionResult, ReasoningTrace


def _make_prediction(probabilities: dict) -> PredictionResult:
    """Helper to create a PredictionResult with given probabilities."""
    trace = ReasoningTrace(
        evidence_considered=["source1"],
        base_rate=0.5,
        supporting_factors=["factor1", "factor2"],
        conflicting_evidence=[],
        conflict_resolution="none",
        confidence_level="medium",
    )
    return PredictionResult(
        event_ticker="EVT-001",
        probabilities=probabilities,
        reasoning_trace=trace,
        duration_seconds=1.0,
    )


class TestAggregatePredictions:
    """Tests for aggregate_predictions function."""

    def test_single_agent_returns_normalized(self):
        predictions = [_make_prediction({"Yes": 0.7, "No": 0.3})]
        result = aggregate_predictions(predictions, ["Yes", "No"])
        assert abs(result["Yes"] - 0.7) < 1e-9
        assert abs(result["No"] - 0.3) < 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_two_agents_mean(self):
        predictions = [
            _make_prediction({"Yes": 0.8, "No": 0.2}),
            _make_prediction({"Yes": 0.6, "No": 0.4}),
        ]
        result = aggregate_predictions(predictions, ["Yes", "No"])
        assert abs(result["Yes"] - 0.7) < 1e-9
        assert abs(result["No"] - 0.3) < 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_three_agents_mean(self):
        predictions = [
            _make_prediction({"A": 0.5, "B": 0.3, "C": 0.2}),
            _make_prediction({"A": 0.4, "B": 0.4, "C": 0.2}),
            _make_prediction({"A": 0.6, "B": 0.2, "C": 0.2}),
        ]
        result = aggregate_predictions(predictions, ["A", "B", "C"])
        assert abs(result["A"] - 0.5) < 1e-9
        assert abs(result["B"] - 0.3) < 1e-9
        assert abs(result["C"] - 0.2) < 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_predictions_returns_uniform(self):
        result = aggregate_predictions([], ["Yes", "No"])
        assert abs(result["Yes"] - 0.5) < 1e-9
        assert abs(result["No"] - 0.5) < 1e-9

    def test_empty_predictions_three_outcomes_uniform(self):
        result = aggregate_predictions([], ["A", "B", "C"])
        expected = 1.0 / 3
        for outcome in ["A", "B", "C"]:
            assert abs(result[outcome] - expected) < 1e-9

    def test_missing_outcome_in_prediction_treated_as_zero(self):
        # Agent only provides probability for "Yes", missing "No"
        predictions = [_make_prediction({"Yes": 0.8})]
        result = aggregate_predictions(predictions, ["Yes", "No"])
        # Mean: Yes=0.8, No=0.0, then normalized: Yes=1.0, No=0.0
        # Actually: 0.8/(0.8+0.0) = 1.0 and 0.0/(0.8) = 0.0
        assert abs(result["Yes"] - 1.0) < 1e-9
        assert abs(result["No"] - 0.0) < 1e-9

    def test_normalization_when_agents_dont_sum_to_one(self):
        # Agents provide probabilities that don't sum to 1
        predictions = [
            _make_prediction({"Yes": 0.6, "No": 0.6}),
            _make_prediction({"Yes": 0.4, "No": 0.4}),
        ]
        result = aggregate_predictions(predictions, ["Yes", "No"])
        # Mean: Yes=0.5, No=0.5, normalized: Yes=0.5, No=0.5
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_result_always_sums_to_one(self):
        predictions = [
            _make_prediction({"X": 0.33, "Y": 0.33, "Z": 0.34}),
            _make_prediction({"X": 0.5, "Y": 0.25, "Z": 0.25}),
        ]
        result = aggregate_predictions(predictions, ["X", "Y", "Z"])
        assert abs(sum(result.values()) - 1.0) < 1e-9


class TestWeightedAggregate:
    """Tests for weighted_aggregate function."""

    def test_equal_weights_matches_mean(self):
        predictions = [
            _make_prediction({"Yes": 0.8, "No": 0.2}),
            _make_prediction({"Yes": 0.6, "No": 0.4}),
        ]
        mean_result = aggregate_predictions(predictions, ["Yes", "No"])
        weighted_result = weighted_aggregate(predictions, [1.0, 1.0], ["Yes", "No"])
        for outcome in ["Yes", "No"]:
            assert abs(mean_result[outcome] - weighted_result[outcome]) < 1e-9

    def test_higher_weight_favors_agent(self):
        predictions = [
            _make_prediction({"Yes": 0.9, "No": 0.1}),
            _make_prediction({"Yes": 0.5, "No": 0.5}),
        ]
        # Weight first agent much more heavily
        result = weighted_aggregate(predictions, [9.0, 1.0], ["Yes", "No"])
        # Weighted mean: Yes = (0.9*9 + 0.5*1)/10 = 8.6/10 = 0.86
        assert result["Yes"] > 0.8
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_predictions_returns_uniform(self):
        result = weighted_aggregate([], [1.0], ["Yes", "No"])
        assert abs(result["Yes"] - 0.5) < 1e-9
        assert abs(result["No"] - 0.5) < 1e-9

    def test_empty_weights_returns_uniform(self):
        predictions = [_make_prediction({"Yes": 0.7, "No": 0.3})]
        result = weighted_aggregate(predictions, [], ["Yes", "No"])
        assert abs(result["Yes"] - 0.5) < 1e-9
        assert abs(result["No"] - 0.5) < 1e-9

    def test_zero_weights_falls_back_to_mean(self):
        predictions = [
            _make_prediction({"Yes": 0.8, "No": 0.2}),
            _make_prediction({"Yes": 0.6, "No": 0.4}),
        ]
        result = weighted_aggregate(predictions, [0.0, 0.0], ["Yes", "No"])
        # Falls back to aggregate_predictions (mean)
        assert abs(result["Yes"] - 0.7) < 1e-9
        assert abs(result["No"] - 0.3) < 1e-9

    def test_result_always_sums_to_one(self):
        predictions = [
            _make_prediction({"A": 0.5, "B": 0.3, "C": 0.2}),
            _make_prediction({"A": 0.3, "B": 0.4, "C": 0.3}),
            _make_prediction({"A": 0.6, "B": 0.2, "C": 0.2}),
        ]
        result = weighted_aggregate(predictions, [2.0, 1.0, 3.0], ["A", "B", "C"])
        assert abs(sum(result.values()) - 1.0) < 1e-9
