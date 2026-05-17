"""Tests for response validation and correction (src/validator.py)."""

import pytest

from src.validator import ResponseValidator


class TestValidate:
    """Tests for ResponseValidator.validate()."""

    def setup_method(self):
        self.validator = ResponseValidator()

    def test_valid_binary_prediction(self):
        """Valid prediction with 2 outcomes passes all checks."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.6, "No": 0.4}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is True
        assert violations == []

    def test_valid_multi_outcome_prediction(self):
        """Valid prediction with 4 outcomes passes all checks."""
        outcomes = ["A", "B", "C", "D"]
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is True
        assert violations == []

    def test_valid_at_boundary_values(self):
        """Probabilities at exact boundary values (0.01, 0.99) are valid."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.99, "No": 0.01}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is True
        assert violations == []

    def test_valid_within_sum_tolerance(self):
        """Sum within 0.001 tolerance is valid."""
        outcomes = ["A", "B", "C"]
        # Sum = 1.0005 which is within 0.001 tolerance
        probs = {"A": 0.3335, "B": 0.3335, "C": 0.3335}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is True
        assert violations == []

    def test_wrong_outcome_count(self):
        """Detects mismatch between probability count and outcome count."""
        outcomes = ["Yes", "No", "Maybe"]
        probs = {"Yes": 0.5, "No": 0.5}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("Expected 3 outcomes, got 2" in v for v in violations)

    def test_probability_below_range(self):
        """Detects probability below 0.01."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.995, "No": 0.005}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("0.005" in v and "0.01" in v for v in violations)

    def test_probability_above_range(self):
        """Detects probability above 0.99."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.995, "No": 0.005}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("0.99" in v for v in violations)

    def test_market_field_mismatch(self):
        """Detects market field that doesn't match any outcome."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.6, "Wrong": 0.4}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("Wrong" in v and "does not match" in v for v in violations)

    def test_missing_outcome(self):
        """Detects missing outcome in probabilities."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.6, "Wrong": 0.4}

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("Missing" in v and "No" in v for v in violations)

    def test_sum_exceeds_tolerance(self):
        """Detects sum that deviates more than 0.001 from 1.0."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.6, "No": 0.5}  # Sum = 1.1

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("sum" in v.lower() for v in violations)

    def test_sum_below_tolerance(self):
        """Detects sum below 1.0 - 0.001."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.4, "No": 0.4}  # Sum = 0.8

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert any("sum" in v.lower() for v in violations)

    def test_multiple_violations_reported(self):
        """Reports all violations, not just the first one."""
        outcomes = ["A", "B", "C"]
        probs = {"A": 0.005, "X": 0.5}  # Wrong count, out of range, wrong market, bad sum

        is_valid, violations = self.validator.validate(probs, outcomes)

        assert is_valid is False
        assert len(violations) >= 3  # Multiple issues detected


class TestCorrect:
    """Tests for ResponseValidator.correct()."""

    def setup_method(self):
        self.validator = ResponseValidator()

    def test_adds_missing_outcomes(self):
        """Adds missing outcomes with some probability."""
        outcomes = ["A", "B", "C"]
        probs = {"A": 0.5}

        corrected = self.validator.correct(probs, outcomes)

        assert set(corrected.keys()) == set(outcomes)
        assert all(outcome in corrected for outcome in outcomes)

    def test_clamps_below_minimum(self):
        """Clamps values below 0.01 upward."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.999, "No": 0.001}

        corrected = self.validator.correct(probs, outcomes)

        # After clamping and normalization, all values should be >= 0.01
        for v in corrected.values():
            assert v >= 0.01 - 1e-9  # Small epsilon for float comparison

    def test_clamps_above_maximum(self):
        """Clamps values above 0.99 downward."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 1.5, "No": 0.5}

        corrected = self.validator.correct(probs, outcomes)

        for v in corrected.values():
            assert v <= 0.99 + 1e-9  # Small epsilon for float comparison

    def test_normalizes_to_sum_one(self):
        """Corrected probabilities sum to approximately 1.0."""
        outcomes = ["A", "B", "C"]
        probs = {"A": 0.5, "B": 0.3, "C": 0.1}  # Sum = 0.9

        corrected = self.validator.correct(probs, outcomes)

        assert abs(sum(corrected.values()) - 1.0) < 0.001

    def test_corrected_output_passes_validation(self):
        """Corrected output passes the validate() method."""
        outcomes = ["A", "B", "C", "D"]
        probs = {"A": 0.005, "B": 1.5, "C": 0.3}  # Missing D, out of range, bad sum

        corrected = self.validator.correct(probs, outcomes)
        is_valid, violations = self.validator.validate(corrected, outcomes)

        assert is_valid is True, f"Violations: {violations}"

    def test_ignores_extra_outcomes_not_in_list(self):
        """Only includes outcomes from the outcomes list."""
        outcomes = ["A", "B"]
        probs = {"A": 0.5, "B": 0.3, "X": 0.2}

        corrected = self.validator.correct(probs, outcomes)

        assert "X" not in corrected
        assert set(corrected.keys()) == {"A", "B"}

    def test_all_missing_outcomes(self):
        """Handles case where no probabilities match outcomes."""
        outcomes = ["A", "B", "C"]
        probs = {"X": 0.5, "Y": 0.3, "Z": 0.2}

        corrected = self.validator.correct(probs, outcomes)

        assert set(corrected.keys()) == set(outcomes)
        assert abs(sum(corrected.values()) - 1.0) < 0.001

    def test_already_valid_input(self):
        """Already valid input is returned normalized."""
        outcomes = ["Yes", "No"]
        probs = {"Yes": 0.7, "No": 0.3}

        corrected = self.validator.correct(probs, outcomes)

        assert abs(sum(corrected.values()) - 1.0) < 0.001
        assert set(corrected.keys()) == set(outcomes)


class TestFallbackUniform:
    """Tests for ResponseValidator.fallback_uniform()."""

    def setup_method(self):
        self.validator = ResponseValidator()

    def test_binary_outcomes(self):
        """Returns 0.5 for each of 2 outcomes."""
        outcomes = ["Yes", "No"]

        result = self.validator.fallback_uniform(outcomes)

        assert result == {"Yes": 0.5, "No": 0.5}

    def test_three_outcomes(self):
        """Returns 1/3 for each of 3 outcomes."""
        outcomes = ["A", "B", "C"]

        result = self.validator.fallback_uniform(outcomes)

        for outcome in outcomes:
            assert abs(result[outcome] - 1.0 / 3) < 1e-10

    def test_four_outcomes(self):
        """Returns 0.25 for each of 4 outcomes."""
        outcomes = ["A", "B", "C", "D"]

        result = self.validator.fallback_uniform(outcomes)

        assert result == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    def test_sums_to_one(self):
        """Uniform distribution sums to 1.0."""
        outcomes = ["A", "B", "C", "D", "E"]

        result = self.validator.fallback_uniform(outcomes)

        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_all_outcomes_present(self):
        """All outcomes are present in the result."""
        outcomes = ["Win", "Lose", "Draw"]

        result = self.validator.fallback_uniform(outcomes)

        assert set(result.keys()) == set(outcomes)

    def test_single_outcome(self):
        """Single outcome gets probability 1.0."""
        outcomes = ["Only"]

        result = self.validator.fallback_uniform(outcomes)

        assert result == {"Only": 1.0}
