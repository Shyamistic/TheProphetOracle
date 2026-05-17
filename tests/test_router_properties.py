"""Property-based tests for the event router and complexity classifier.

Uses Hypothesis to verify universal properties of the routing system.
"""

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import ComplexityTier, EventRequest
from src.router import assess_complexity


# === Strategies ===


def event_request_strategy(
    min_outcomes: int = 2,
    max_outcomes: int = 10,
    min_desc_len: int = 0,
    max_desc_len: int = 800,
):
    """Generate random EventRequest objects with controlled outcome counts and description lengths."""
    return st.builds(
        EventRequest,
        event_ticker=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=3, max_size=10),
        market_ticker=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=3, max_size=10),
        title=st.text(alphabet=string.ascii_letters + " ", min_size=5, max_size=50),
        description=st.text(
            alphabet=string.ascii_letters + " .,",
            min_size=min_desc_len,
            max_size=max_desc_len,
        ),
        category=st.sampled_from(["Sports", "Economics", "Geopolitics", "Technology", "Science", "General"]),
        rules=st.text(alphabet=string.ascii_letters + " ", min_size=5, max_size=50),
        close_time=st.just("2026-05-30T00:00:00Z"),
        outcomes=st.lists(
            st.text(alphabet=string.ascii_letters, min_size=2, max_size=15),
            min_size=min_outcomes,
            max_size=max_outcomes,
        ),
        resolved_outcome=st.none(),
    )


# === Property 6: Complexity Classification Determinism ===
# Validates: Requirements 6.1


class TestComplexityClassificationDeterminism:
    """Property 6: Complexity Classification Determinism.

    **Validates: Requirements 6.1**

    Tests that classification is deterministic and matches the defined rules:
    - LOW: 2 outcomes AND description < 200 chars
    - MEDIUM: 2-3 outcomes AND description < 500 chars (but not LOW)
    - HIGH: 4+ outcomes OR description >= 500 chars
    """

    @given(event=event_request_strategy())
    @settings(max_examples=200)
    def test_classification_is_deterministic(self, event: EventRequest):
        """Same event always produces the same complexity tier."""
        result1 = assess_complexity(event)
        result2 = assess_complexity(event)
        assert result1 == result2, (
            f"Non-deterministic classification: got {result1} and {result2} for same event"
        )

    @given(event=event_request_strategy(min_outcomes=4, max_outcomes=10))
    @settings(max_examples=200)
    def test_high_complexity_with_many_outcomes(self, event: EventRequest):
        """Events with 4+ outcomes are always classified as HIGH regardless of description length."""
        result = assess_complexity(event)
        assert result == ComplexityTier.HIGH, (
            f"Expected HIGH for {len(event.outcomes)} outcomes, got {result}"
        )

    @given(event=event_request_strategy(min_outcomes=2, max_outcomes=3, min_desc_len=500, max_desc_len=800))
    @settings(max_examples=200)
    def test_high_complexity_with_long_description(self, event: EventRequest):
        """Events with description >= 500 chars are always classified as HIGH regardless of outcome count."""
        result = assess_complexity(event)
        assert result == ComplexityTier.HIGH, (
            f"Expected HIGH for description length {len(event.description)}, got {result}"
        )

    @given(event=event_request_strategy(min_outcomes=2, max_outcomes=2, min_desc_len=0, max_desc_len=199))
    @settings(max_examples=200)
    def test_low_complexity_classification(self, event: EventRequest):
        """Events with exactly 2 outcomes AND description < 200 chars are classified as LOW."""
        result = assess_complexity(event)
        assert result == ComplexityTier.LOW, (
            f"Expected LOW for 2 outcomes and desc length {len(event.description)}, got {result}"
        )

    @given(event=event_request_strategy(min_outcomes=2, max_outcomes=3, min_desc_len=200, max_desc_len=499))
    @settings(max_examples=200)
    def test_medium_complexity_classification(self, event: EventRequest):
        """Events with 2-3 outcomes AND description 200-499 chars are classified as MEDIUM."""
        result = assess_complexity(event)
        assert result == ComplexityTier.MEDIUM, (
            f"Expected MEDIUM for {len(event.outcomes)} outcomes and desc length {len(event.description)}, got {result}"
        )

    @given(event=event_request_strategy(min_outcomes=3, max_outcomes=3, min_desc_len=0, max_desc_len=199))
    @settings(max_examples=200)
    def test_medium_complexity_three_outcomes_short_desc(self, event: EventRequest):
        """Events with 3 outcomes AND description < 200 chars are classified as MEDIUM (not LOW)."""
        result = assess_complexity(event)
        assert result == ComplexityTier.MEDIUM, (
            f"Expected MEDIUM for 3 outcomes and desc length {len(event.description)}, got {result}"
        )

    @given(event=event_request_strategy())
    @settings(max_examples=300)
    def test_classification_matches_defined_rules(self, event: EventRequest):
        """Classification always matches the defined rules for LOW/MEDIUM/HIGH."""
        result = assess_complexity(event)
        num_outcomes = len(event.outcomes)
        desc_length = len(event.description)

        # Verify the result matches the expected tier based on rules
        if num_outcomes >= 4 or desc_length >= 500:
            assert result == ComplexityTier.HIGH, (
                f"Expected HIGH: outcomes={num_outcomes}, desc_len={desc_length}, got {result}"
            )
        elif num_outcomes == 2 and desc_length < 200:
            assert result == ComplexityTier.LOW, (
                f"Expected LOW: outcomes={num_outcomes}, desc_len={desc_length}, got {result}"
            )
        else:
            assert result == ComplexityTier.MEDIUM, (
                f"Expected MEDIUM: outcomes={num_outcomes}, desc_len={desc_length}, got {result}"
            )


# === Additional imports for category routing tests ===
from src.models import EventCategory
from src.router import CATEGORY_STRATEGIES, detect_category


# Known category values (the string values of the enum, excluding GENERAL)
KNOWN_CATEGORIES = [cat.value for cat in EventCategory if cat != EventCategory.GENERAL]


# === Strategies for category routing ===


def case_variant_strategy():
    """Generate case variants of known category strings (upper, lower, mixed, with whitespace)."""
    return st.sampled_from(KNOWN_CATEGORIES).flatmap(
        lambda cat: st.sampled_from([
            cat.lower(),
            cat.upper(),
            cat.capitalize(),
            cat.title(),
            cat.swapcase(),
            f"  {cat}  ",
            f" {cat.upper()} ",
        ])
    )


def unrecognized_category_strategy():
    """Generate strings that do NOT match any known category."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s.strip().lower() not in [cat.value for cat in EventCategory])


# === Property 5: Category Routing Correctness ===
# Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7


class TestCategoryRoutingCorrectness:
    """Property 5: Category Routing Correctness

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**
    """

    @given(category_input=case_variant_strategy())
    @settings(max_examples=200)
    def test_case_insensitive_matching_routes_to_correct_category(self, category_input: str):
        """For any case variant of a known category, detect_category returns the correct enum value.

        This validates Requirement 7.7: case-insensitive matching on the event category field.
        """
        result = detect_category(category_input)

        # The normalized (stripped, lowered) input should match the result's value
        expected_value = category_input.strip().lower()
        assert result.value == expected_value, (
            f"Expected category '{expected_value}' but got '{result.value}' "
            f"for input '{category_input}'"
        )
        assert result != EventCategory.GENERAL, (
            f"Known category input '{category_input}' should not route to GENERAL"
        )

    @given(category_input=unrecognized_category_strategy())
    @settings(max_examples=200)
    def test_unrecognized_strings_route_to_general(self, category_input: str):
        """For any string not matching a known category, detect_category returns GENERAL.

        This validates Requirement 7.6: unrecognized categories use general-purpose strategy.
        """
        result = detect_category(category_input)

        assert result == EventCategory.GENERAL, (
            f"Unrecognized input '{category_input}' should route to GENERAL "
            f"but got '{result.value}'"
        )

    @given(category_input=case_variant_strategy())
    @settings(max_examples=200)
    def test_known_categories_have_at_least_2_source_types(self, category_input: str):
        """For any known category, the strategy specifies at least 2 source types.

        This validates Requirements 7.1-7.5: each known category targets at least two
        source types (e.g., Sports targets team_statistics, recent_performance, head_to_head).
        """
        category = detect_category(category_input)

        # Known categories must not be GENERAL
        assert category != EventCategory.GENERAL

        strategy = CATEGORY_STRATEGIES[category]
        source_types = strategy["source_types"]

        assert len(source_types) >= 2, (
            f"Category '{category.value}' has only {len(source_types)} source types: "
            f"{source_types}. Requirements 7.1-7.5 mandate at least 2."
        )

    @given(category_input=st.sampled_from(KNOWN_CATEGORIES))
    @settings(max_examples=50)
    def test_all_known_categories_present_in_strategies(self, category_input: str):
        """Every known category enum value has a corresponding entry in CATEGORY_STRATEGIES."""
        category = detect_category(category_input)
        assert category in CATEGORY_STRATEGIES, (
            f"Category '{category.value}' missing from CATEGORY_STRATEGIES"
        )

    @given(category_input=unrecognized_category_strategy())
    @settings(max_examples=100)
    def test_general_fallback_has_at_least_one_source_type(self, category_input: str):
        """GENERAL category (fallback) has at least 1 source type for general-purpose search.

        This validates Requirement 7.6: general-purpose strategy executes at least one
        web search query.
        """
        category = detect_category(category_input)
        assert category == EventCategory.GENERAL

        strategy = CATEGORY_STRATEGIES[EventCategory.GENERAL]
        source_types = strategy["source_types"]

        assert len(source_types) >= 1, (
            f"GENERAL category must have at least 1 source type but has {len(source_types)}"
        )
