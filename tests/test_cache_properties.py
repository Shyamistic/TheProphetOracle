"""Property-based tests for prediction cache (src/cache.py).

Uses Hypothesis to verify universal properties of the cache round-trip behavior.
"""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio
from hypothesis import given, strategies as st, settings

from src.cache import PredictionCache
from src.models import EventRequest


# --- Strategies ---

def event_request_strategy():
    """Generate random EventRequest instances for property testing."""
    return st.builds(
        EventRequest,
        event_ticker=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ),
        market_ticker=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
            min_size=1,
            max_size=20,
        ),
        title=st.text(min_size=1, max_size=100),
        description=st.text(min_size=1, max_size=500),
        category=st.sampled_from(["Sports", "Economics", "Geopolitics", "Technology", "Science", "General"]),
        rules=st.text(min_size=1, max_size=200),
        close_time=st.sampled_from([
            "2026-05-20T00:00:00Z",
            "2026-06-15T12:00:00Z",
            "2026-07-01T18:30:00Z",
            "2026-08-10T06:00:00Z",
        ]),
        outcomes=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=" -_"),
                min_size=1,
                max_size=30,
            ),
            min_size=2,
            max_size=6,
            unique=True,
        ),
        resolved_outcome=st.none(),
    )


def probability_dict_strategy(outcomes):
    """Generate a valid probability dict for given outcomes.

    Produces probabilities in [0.01, 0.99] that sum to 1.0.
    """
    n = len(outcomes)
    # Generate n random floats and normalize
    raw = [max(0.01, min(0.99, v)) for v in [1.0 / n] * n]
    # We'll use a different approach: generate random weights and normalize
    return st.lists(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        min_size=n,
        max_size=n,
    ).map(lambda vals: _normalize_to_dict(outcomes, vals))


def _normalize_to_dict(outcomes, values):
    """Normalize values to sum to 1.0 and map to outcomes."""
    total = sum(values)
    if total == 0:
        # Fallback to uniform
        n = len(outcomes)
        return {o: 1.0 / n for o in outcomes}
    normalized = [v / total for v in values]
    # Clamp to [0.01, 0.99] after normalization
    clamped = [max(0.01, min(0.99, v)) for v in normalized]
    # Re-normalize after clamping
    total2 = sum(clamped)
    final = [v / total2 for v in clamped]
    return {outcomes[i]: final[i] for i in range(len(outcomes))}


# Composite strategy that generates an event and matching probabilities together
@st.composite
def event_and_probabilities(draw):
    """Generate a random event and a matching probability dict."""
    event = draw(event_request_strategy())
    probs_list = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
            min_size=len(event.outcomes),
            max_size=len(event.outcomes),
        )
    )
    probabilities = _normalize_to_dict(event.outcomes, probs_list)
    return event, probabilities


class TestCacheRoundTrip:
    """Property 10: Cache Round-Trip.

    For any event processed by the agent, if the same event (identical
    event_ticker, description, outcomes, close_time) is submitted again
    within the cache TTL (6 hours), the agent SHALL return the identical
    prediction without making additional LLM or search API calls.

    **Validates: Requirements 6.2**
    """

    @settings(max_examples=50, deadline=None)
    @given(data=event_and_probabilities())
    def test_store_and_retrieve_returns_identical_predictions(self, data):
        """Storing and retrieving the same event within TTL returns identical predictions."""
        event, probabilities = data

        # Create a fresh temp db for each test case
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            cache = PredictionCache(db_path=db_path, ttl_hours=6)

            async def _run():
                await cache.set(event, probabilities)
                result = await cache.get(event)
                return result

            result = asyncio.run(_run())

            # The retrieved prediction must be identical to what was stored
            assert result is not None, (
                f"Cache returned None for event {event.event_ticker} "
                f"immediately after storing"
            )
            assert set(result.keys()) == set(probabilities.keys()), (
                f"Keys mismatch: stored {set(probabilities.keys())}, "
                f"got {set(result.keys())}"
            )
            for outcome in probabilities:
                assert abs(result[outcome] - probabilities[outcome]) < 1e-10, (
                    f"Probability mismatch for outcome '{outcome}': "
                    f"stored {probabilities[outcome]}, got {result[outcome]}"
                )
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @settings(max_examples=50, deadline=None)
    @given(data=event_and_probabilities())
    def test_cache_key_determinism(self, data):
        """The same event always produces the same cache key."""
        event, _ = data

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            cache = PredictionCache(db_path=db_path, ttl_hours=6)
            key1 = cache.cache_key(event)
            key2 = cache.cache_key(event)
            assert key1 == key2, (
                f"Cache key not deterministic for event {event.event_ticker}: "
                f"{key1} != {key2}"
            )
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @settings(max_examples=50, deadline=None)
    @given(data=event_and_probabilities())
    def test_multiple_stores_last_write_wins(self, data):
        """Storing the same event twice returns the last stored prediction."""
        event, probabilities = data

        # Create a second set of probabilities (shifted)
        outcomes = list(probabilities.keys())
        n = len(outcomes)
        second_probs = {outcomes[i]: 1.0 / n for i in range(n)}

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            cache = PredictionCache(db_path=db_path, ttl_hours=6)

            async def _run():
                await cache.set(event, probabilities)
                await cache.set(event, second_probs)
                result = await cache.get(event)
                return result

            result = asyncio.run(_run())

            assert result is not None
            for outcome in second_probs:
                assert abs(result[outcome] - second_probs[outcome]) < 1e-10, (
                    f"Expected last-write-wins for outcome '{outcome}': "
                    f"expected {second_probs[outcome]}, got {result[outcome]}"
                )
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
