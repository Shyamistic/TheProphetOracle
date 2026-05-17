"""Property-based tests for ResponseValidator (src/validator.py).

Property 11: Validation Correction Produces Valid Output
For any invalid prediction (probabilities out of range, missing outcomes, incorrect sum),
the correction function SHALL produce output that passes all validation checks
(correct outcome count, values in [0.01, 0.99], sum to 1.0).

Validates: Requirements 9.5, 9.6, 5.2
"""

import hypothesis
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.validator import ResponseValidator


# --- Strategies ---

# Generate outcome lists with 2-10 unique string labels
outcome_labels = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
    min_size=1,
    max_size=10,
)

outcomes_strategy = st.lists(
    outcome_labels,
    min_size=2,
    max_size=10,
    unique=True,
)


def invalid_probabilities_strategy(outcomes):
    """Generate probability dicts that are invalid in various ways.

    Combines three kinds of invalidity:
    - Out-of-range values (below 0.01 or above 0.99)
    - Missing outcomes (subset of outcomes present)
    - Incorrect sums (not summing to 1.0)
    """
    # Strategy: generate a dict with a random subset of outcomes (possibly with extras)
    # and random float values that may be out of range
    out_of_range_floats = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)

    # Pick a random subset of outcomes (may be empty, partial, or full)
    subset = st.lists(
        st.sampled_from(outcomes),
        min_size=0,
        max_size=len(outcomes),
        unique=True,
    )

    # Optionally add extra keys not in outcomes
    extra_keys = st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L",), min_codepoint=65, max_codepoint=90),
            min_size=1,
            max_size=5,
        ).filter(lambda k: k not in outcomes),
        min_size=0,
        max_size=3,
        unique=True,
    )

    @st.composite
    def build_invalid_probs(draw):
        keys_from_outcomes = draw(subset)
        extra = draw(extra_keys)
        all_keys = keys_from_outcomes + extra

        # Need at least one key to have a non-empty dict
        if not all_keys:
            all_keys = [outcomes[0]]

        probs = {}
        for key in all_keys:
            probs[key] = draw(out_of_range_floats)

        # Ensure the dict is actually invalid for the given outcomes
        validator = ResponseValidator()
        is_valid, _ = validator.validate(probs, outcomes)
        assume(not is_valid)

        return probs

    return build_invalid_probs()


# --- Property Test ---


class TestValidationCorrectionProperty:
    """Property 11: Validation Correction Produces Valid Output.

    **Validates: Requirements 9.5, 9.6, 5.2**
    """

    def setup_method(self):
        self.validator = ResponseValidator()

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_correction_produces_valid_output(self, data):
        """For any invalid probability set, correction produces output passing all validation checks.

        **Validates: Requirements 9.5, 9.6, 5.2**
        """
        # Generate outcomes first, then invalid probabilities for those outcomes
        outcomes = data.draw(outcomes_strategy, label="outcomes")
        probs = data.draw(invalid_probabilities_strategy(outcomes), label="invalid_probs")

        # Apply correction
        corrected = self.validator.correct(probs, outcomes)

        # Validate the corrected output passes ALL checks
        is_valid, violations = self.validator.validate(corrected, outcomes)

        # Property: corrected output must always be valid
        assert is_valid, (
            f"Correction failed to produce valid output.\n"
            f"Input probs: {probs}\n"
            f"Outcomes: {outcomes}\n"
            f"Corrected: {corrected}\n"
            f"Violations: {violations}"
        )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_correction_has_correct_outcome_count(self, data):
        """Corrected output contains exactly one entry per outcome.

        **Validates: Requirements 9.5, 9.6, 5.2**
        """
        outcomes = data.draw(outcomes_strategy, label="outcomes")
        probs = data.draw(invalid_probabilities_strategy(outcomes), label="invalid_probs")

        corrected = self.validator.correct(probs, outcomes)

        assert len(corrected) == len(outcomes), (
            f"Expected {len(outcomes)} entries, got {len(corrected)}.\n"
            f"Outcomes: {outcomes}\n"
            f"Corrected keys: {list(corrected.keys())}"
        )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_correction_values_in_valid_range(self, data):
        """All corrected probability values are in [0.01, 0.99].

        **Validates: Requirements 9.5, 9.6, 5.2**
        """
        outcomes = data.draw(outcomes_strategy, label="outcomes")
        probs = data.draw(invalid_probabilities_strategy(outcomes), label="invalid_probs")

        corrected = self.validator.correct(probs, outcomes)

        for outcome, prob in corrected.items():
            assert 0.01 - 1e-9 <= prob <= 0.99 + 1e-9, (
                f"Probability for '{outcome}' is {prob}, outside [0.01, 0.99].\n"
                f"Input: {probs}\n"
                f"Corrected: {corrected}"
            )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_correction_sums_to_one(self, data):
        """Corrected probabilities sum to 1.0 within tolerance.

        **Validates: Requirements 9.5, 9.6, 5.2**
        """
        outcomes = data.draw(outcomes_strategy, label="outcomes")
        probs = data.draw(invalid_probabilities_strategy(outcomes), label="invalid_probs")

        corrected = self.validator.correct(probs, outcomes)

        total = sum(corrected.values())
        assert abs(total - 1.0) <= 0.001, (
            f"Corrected probabilities sum to {total}, expected 1.0.\n"
            f"Input: {probs}\n"
            f"Corrected: {corrected}"
        )

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_correction_keys_match_outcomes(self, data):
        """Corrected output keys exactly match the outcomes list.

        **Validates: Requirements 9.5, 9.6, 5.2**
        """
        outcomes = data.draw(outcomes_strategy, label="outcomes")
        probs = data.draw(invalid_probabilities_strategy(outcomes), label="invalid_probs")

        corrected = self.validator.correct(probs, outcomes)

        assert set(corrected.keys()) == set(outcomes), (
            f"Corrected keys {set(corrected.keys())} don't match outcomes {set(outcomes)}.\n"
            f"Input: {probs}"
        )
