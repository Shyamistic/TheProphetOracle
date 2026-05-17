"""Tests for calibration module (src/calibrator.py)."""

import math
import pytest

from src.calibrator import (
    CalibrationModule,
    DEFAULT_SHRINKAGE,
    PLATT_COEFFICIENT,
)


class TestConstants:
    """Tests for module-level constants."""

    def test_platt_coefficient_is_sqrt3(self):
        """PLATT_COEFFICIENT equals √3."""
        assert PLATT_COEFFICIENT == pytest.approx(math.sqrt(3), rel=1e-9)

    def test_default_shrinkage_is_015(self):
        """DEFAULT_SHRINKAGE is 0.15."""
        assert DEFAULT_SHRINKAGE == 0.15


class TestCalibrationModuleInit:
    """Tests for CalibrationModule initialization."""

    def test_default_parameters(self):
        """Uses default shrinkage and platt coefficient."""
        cal = CalibrationModule()
        assert cal.shrinkage_factor == DEFAULT_SHRINKAGE
        assert cal.platt_coefficient == PLATT_COEFFICIENT

    def test_custom_parameters(self):
        """Accepts custom shrinkage and platt coefficient."""
        cal = CalibrationModule(shrinkage_factor=0.20, platt_coefficient=2.0)
        assert cal.shrinkage_factor == 0.20
        assert cal.platt_coefficient == 2.0


class TestPlattScale:
    """Tests for platt_scale method."""

    def test_p_050_unchanged(self):
        """p=0.5 maps to 0.5 (log-odds of 0.5 is 0, scaling 0 stays 0)."""
        cal = CalibrationModule()
        result = cal.platt_scale(0.5)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_p_above_05_extremized_higher(self):
        """p > 0.5 is pushed further from 0.5 (higher)."""
        cal = CalibrationModule()
        p = 0.6
        result = cal.platt_scale(p)
        assert result > p

    def test_p_below_05_extremized_lower(self):
        """p < 0.5 is pushed further from 0.5 (lower)."""
        cal = CalibrationModule()
        p = 0.4
        result = cal.platt_scale(p)
        assert result < p

    def test_known_value_p060(self):
        """Verify platt_scale(0.6) against manual calculation."""
        cal = CalibrationModule()
        p = 0.6
        log_odds = math.log(p / (1 - p))  # ~0.4055
        scaled = log_odds * math.sqrt(3)  # ~0.7023
        expected = 1.0 / (1.0 + math.exp(-scaled))  # ~0.669
        result = cal.platt_scale(p)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_known_value_p080(self):
        """Verify platt_scale(0.8) against manual calculation."""
        cal = CalibrationModule()
        p = 0.8
        log_odds = math.log(p / (1 - p))  # ~1.3863
        scaled = log_odds * math.sqrt(3)  # ~2.4010
        expected = 1.0 / (1.0 + math.exp(-scaled))  # ~0.917
        result = cal.platt_scale(p)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_symmetric_around_05(self):
        """platt_scale(0.5 + d) and platt_scale(0.5 - d) are symmetric around 0.5."""
        cal = CalibrationModule()
        d = 0.2
        high = cal.platt_scale(0.5 + d)
        low = cal.platt_scale(0.5 - d)
        assert high + low == pytest.approx(1.0, abs=1e-9)

    def test_extreme_low_input_clamped(self):
        """Very low input (0.0) is clamped to avoid log(0)."""
        cal = CalibrationModule()
        result = cal.platt_scale(0.0)
        # Should not raise, and should return a valid probability
        assert 0.0 < result < 1.0

    def test_extreme_high_input_clamped(self):
        """Very high input (1.0) is clamped to avoid log(0)."""
        cal = CalibrationModule()
        result = cal.platt_scale(1.0)
        assert 0.0 < result < 1.0

    def test_output_always_in_01(self):
        """Output is always in (0, 1) for any valid input."""
        cal = CalibrationModule()
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            result = cal.platt_scale(p)
            assert 0.0 < result < 1.0


class TestShrinkExtreme:
    """Tests for shrink_extreme method."""

    def test_value_in_range_unchanged(self):
        """Values in [0.10, 0.90] are not modified."""
        cal = CalibrationModule()
        for p in [0.10, 0.30, 0.50, 0.70, 0.90]:
            assert cal.shrink_extreme(p) == p

    def test_high_value_shrunk_toward_05(self):
        """Values > 0.90 are shrunk toward 0.5."""
        cal = CalibrationModule()
        p = 0.95
        result = cal.shrink_extreme(p)
        # result = 0.95 - 0.15 * (0.95 - 0.5) = 0.95 - 0.0675 = 0.8825
        expected = 0.95 - 0.15 * (0.95 - 0.5)
        assert result == pytest.approx(expected, rel=1e-9)
        assert result < p
        assert result > 0.5

    def test_low_value_shrunk_toward_05(self):
        """Values < 0.10 are shrunk toward 0.5."""
        cal = CalibrationModule()
        p = 0.05
        result = cal.shrink_extreme(p)
        # result = 0.05 + 0.15 * (0.5 - 0.05) = 0.05 + 0.0675 = 0.1175
        expected = 0.05 + 0.15 * (0.5 - 0.05)
        assert result == pytest.approx(expected, rel=1e-9)
        assert result > p
        assert result < 0.5

    def test_custom_shrinkage_factor(self):
        """Custom shrinkage factor is applied correctly."""
        cal = CalibrationModule(shrinkage_factor=0.30)
        p = 0.95
        result = cal.shrink_extreme(p)
        expected = 0.95 - 0.30 * (0.95 - 0.5)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_boundary_090_not_shrunk(self):
        """Exactly 0.90 is NOT shrunk (only > 0.90)."""
        cal = CalibrationModule()
        assert cal.shrink_extreme(0.90) == 0.90

    def test_boundary_010_not_shrunk(self):
        """Exactly 0.10 is NOT shrunk (only < 0.10)."""
        cal = CalibrationModule()
        assert cal.shrink_extreme(0.10) == 0.10


class TestClamp:
    """Tests for clamp method."""

    def test_value_in_range_unchanged(self):
        """Values in [0.01, 0.99] are unchanged."""
        cal = CalibrationModule()
        assert cal.clamp(0.5) == 0.5
        assert cal.clamp(0.01) == 0.01
        assert cal.clamp(0.99) == 0.99

    def test_value_below_range_clamped(self):
        """Values below 0.01 are clamped to 0.01."""
        cal = CalibrationModule()
        assert cal.clamp(0.0) == 0.01
        assert cal.clamp(-0.5) == 0.01
        assert cal.clamp(0.005) == 0.01

    def test_value_above_range_clamped(self):
        """Values above 0.99 are clamped to 0.99."""
        cal = CalibrationModule()
        assert cal.clamp(1.0) == 0.99
        assert cal.clamp(1.5) == 0.99
        assert cal.clamp(0.995) == 0.99


class TestNormalize:
    """Tests for normalize method."""

    def test_already_normalized(self):
        """Dict already summing to 1.0 is unchanged."""
        cal = CalibrationModule()
        probs = {"A": 0.6, "B": 0.4}
        result = cal.normalize(probs)
        assert result["A"] == pytest.approx(0.6, rel=1e-9)
        assert result["B"] == pytest.approx(0.4, rel=1e-9)

    def test_sums_to_one(self):
        """Output always sums to 1.0."""
        cal = CalibrationModule()
        probs = {"A": 0.3, "B": 0.5, "C": 0.8}
        result = cal.normalize(probs)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_preserves_ratios(self):
        """Normalization preserves relative ratios."""
        cal = CalibrationModule()
        probs = {"A": 2.0, "B": 3.0}
        result = cal.normalize(probs)
        assert result["A"] == pytest.approx(0.4, rel=1e-9)
        assert result["B"] == pytest.approx(0.6, rel=1e-9)

    def test_all_zeros_returns_uniform(self):
        """All-zero input returns uniform distribution."""
        cal = CalibrationModule()
        probs = {"A": 0.0, "B": 0.0, "C": 0.0}
        result = cal.normalize(probs)
        for v in result.values():
            assert v == pytest.approx(1.0 / 3, rel=1e-9)

    def test_single_outcome(self):
        """Single outcome normalizes to 1.0."""
        cal = CalibrationModule()
        probs = {"A": 0.7}
        result = cal.normalize(probs)
        assert result["A"] == pytest.approx(1.0, rel=1e-9)


class TestCalibrate:
    """Tests for the full calibrate pipeline."""

    def test_output_sums_to_one(self):
        """Calibrated output always sums to 1.0."""
        cal = CalibrationModule()
        probs = {"A": 0.6, "B": 0.3, "C": 0.1}
        result = cal.calibrate(probs)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_output_in_valid_range(self):
        """All calibrated values are in [0.01, 0.99]."""
        cal = CalibrationModule()
        probs = {"A": 0.95, "B": 0.03, "C": 0.02}
        result = cal.calibrate(probs)
        for v in result.values():
            assert 0.01 <= v <= 0.99

    def test_preserves_outcome_keys(self):
        """Output has same keys as input."""
        cal = CalibrationModule()
        probs = {"Yes": 0.7, "No": 0.3}
        result = cal.calibrate(probs)
        assert set(result.keys()) == {"Yes", "No"}

    def test_two_outcomes_binary(self):
        """Binary event calibration produces valid output."""
        cal = CalibrationModule()
        probs = {"Yes": 0.7, "No": 0.3}
        result = cal.calibrate(probs)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.01 <= v <= 0.99 for v in result.values())

    def test_uniform_input_stays_uniform(self):
        """Uniform input (all 0.5) stays uniform after calibration."""
        cal = CalibrationModule()
        probs = {"A": 0.5, "B": 0.5}
        result = cal.calibrate(probs)
        assert result["A"] == pytest.approx(0.5, abs=1e-6)
        assert result["B"] == pytest.approx(0.5, abs=1e-6)

    def test_pipeline_order_platt_then_shrink_then_clamp_then_normalize(self):
        """Verify the pipeline applies steps in correct order."""
        cal = CalibrationModule()
        p = 0.8
        # Step 1: Platt scale
        scaled = cal.platt_scale(p)
        # Step 2: Shrink extreme (if applicable)
        shrunk = cal.shrink_extreme(scaled)
        # Step 3: Clamp
        clamped = cal.clamp(shrunk)

        # For a single-outcome dict, normalize returns 1.0
        # For two outcomes, verify the pipeline manually
        probs = {"A": 0.8, "B": 0.2}
        result = cal.calibrate(probs)

        # Manually compute expected
        a_scaled = cal.platt_scale(0.8)
        a_shrunk = cal.shrink_extreme(a_scaled)
        a_clamped = cal.clamp(a_shrunk)

        b_scaled = cal.platt_scale(0.2)
        b_shrunk = cal.shrink_extreme(b_scaled)
        b_clamped = cal.clamp(b_shrunk)

        total = a_clamped + b_clamped
        expected_a = a_clamped / total
        expected_b = b_clamped / total

        assert result["A"] == pytest.approx(expected_a, rel=1e-9)
        assert result["B"] == pytest.approx(expected_b, rel=1e-9)

    def test_error_fallback_returns_clamped_normalized(self):
        """On error, returns original estimates clamped and normalized."""
        cal = CalibrationModule()

        # Monkey-patch platt_scale to raise an error
        def broken_platt(p):
            raise ValueError("Simulated error")

        cal.platt_scale = broken_platt

        probs = {"A": 0.7, "B": 0.3}
        result = cal.calibrate(probs)

        # Should return clamped and normalized originals
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.01 <= v <= 0.99 for v in result.values())

    def test_error_fallback_with_extreme_values(self):
        """Fallback clamps extreme values before normalizing."""
        cal = CalibrationModule()

        def broken_platt(p):
            raise ValueError("Simulated error")

        cal.platt_scale = broken_platt

        probs = {"A": 1.5, "B": -0.5}  # Out of range values
        result = cal.calibrate(probs)

        # Both should be clamped: A→0.99, B→0.01, then normalized
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.01 <= v <= 0.99 for v in result.values())

    def test_many_outcomes(self):
        """Works correctly with many outcomes."""
        cal = CalibrationModule()
        probs = {f"outcome_{i}": 1.0 / 10 for i in range(10)}
        result = cal.calibrate(probs)
        assert len(result) == 10
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.01 <= v <= 0.99 for v in result.values())
