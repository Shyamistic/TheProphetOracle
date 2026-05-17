"""Property-based tests for calibration module (src/calibrator.py).

Uses Hypothesis to verify universal properties of the calibration pipeline.
"""

from hypothesis import given, strategies as st, assume

from src.calibrator import CalibrationModule


class TestCalibrationExtremization:
    """Property 2: Calibration Extremization.

    For any p ≠ 0.5 in (0.01, 0.99), Platt scaling produces output farther
    from 0.5 than input. This validates that the √3 coefficient correctly
    extremizes predictions (moves them away from 0.5).

    **Validates: Requirements 4.1**
    """

    @given(p=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False))
    def test_platt_scale_extremizes_away_from_half(self, p: float):
        """For any p ≠ 0.5, platt_scale(p) is farther from 0.5 than p."""
        assume(abs(p - 0.5) > 1e-9)  # Exclude p = 0.5

        cal = CalibrationModule()
        result = cal.platt_scale(p)

        distance_input = abs(p - 0.5)
        distance_output = abs(result - 0.5)

        assert distance_output > distance_input, (
            f"Expected platt_scale({p}) = {result} to be farther from 0.5 "
            f"than input. Input distance: {distance_input}, "
            f"output distance: {distance_output}"
        )



