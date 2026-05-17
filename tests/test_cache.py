"""Tests for the SQLite-backed prediction cache."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import pytest_asyncio

from src.cache import PredictionCache
from src.models import EventRequest


@pytest.fixture
def sample_event() -> EventRequest:
    """Create a sample event for testing."""
    return EventRequest(
        event_ticker="EVT-001",
        market_ticker="MKT-001",
        title="Will it rain tomorrow?",
        description="Will it rain in Chicago on May 20, 2026?",
        category="Science",
        rules="Resolves YES if measurable precipitation recorded.",
        close_time="2026-05-20T00:00:00Z",
        outcomes=["Yes", "No"],
    )


@pytest.fixture
def sample_probabilities() -> dict:
    """Sample probability dict."""
    return {"Yes": 0.65, "No": 0.35}


@pytest.fixture
def temp_db_path():
    """Provide a temporary database file path and clean up after test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest_asyncio.fixture
async def cache(temp_db_path) -> PredictionCache:
    """Create a PredictionCache instance with a temp database."""
    return PredictionCache(db_path=temp_db_path, ttl_hours=6)


class TestCacheKey:
    """Tests for cache_key generation."""

    def test_cache_key_is_sha256_hex(self, sample_event, temp_db_path):
        cache = PredictionCache(db_path=temp_db_path)
        key = cache.cache_key(sample_event)
        # SHA-256 hex digest is 64 characters
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_deterministic(self, sample_event, temp_db_path):
        cache = PredictionCache(db_path=temp_db_path)
        key1 = cache.cache_key(sample_event)
        key2 = cache.cache_key(sample_event)
        assert key1 == key2

    def test_cache_key_different_for_different_events(self, temp_db_path):
        cache = PredictionCache(db_path=temp_db_path)
        event1 = EventRequest(
            event_ticker="EVT-001",
            market_ticker="MKT-001",
            title="Event 1",
            description="Description 1",
            category="Science",
            rules="Rules",
            close_time="2026-05-20T00:00:00Z",
            outcomes=["Yes", "No"],
        )
        event2 = EventRequest(
            event_ticker="EVT-002",
            market_ticker="MKT-002",
            title="Event 2",
            description="Description 2",
            category="Science",
            rules="Rules",
            close_time="2026-05-20T00:00:00Z",
            outcomes=["Yes", "No"],
        )
        assert cache.cache_key(event1) != cache.cache_key(event2)

    def test_cache_key_outcome_order_independent(self, temp_db_path):
        """Outcomes are sorted before hashing, so order shouldn't matter."""
        cache = PredictionCache(db_path=temp_db_path)
        event1 = EventRequest(
            event_ticker="EVT-001",
            market_ticker="MKT-001",
            title="Test",
            description="Test event",
            category="Science",
            rules="Rules",
            close_time="2026-05-20T00:00:00Z",
            outcomes=["Yes", "No", "Maybe"],
        )
        event2 = EventRequest(
            event_ticker="EVT-001",
            market_ticker="MKT-001",
            title="Test",
            description="Test event",
            category="Science",
            rules="Rules",
            close_time="2026-05-20T00:00:00Z",
            outcomes=["Maybe", "No", "Yes"],
        )
        assert cache.cache_key(event1) == cache.cache_key(event2)

    def test_cache_key_uses_correct_fields(self, temp_db_path):
        """Changing non-key fields (title, rules, category) should NOT change the key."""
        cache = PredictionCache(db_path=temp_db_path)
        event1 = EventRequest(
            event_ticker="EVT-001",
            market_ticker="MKT-001",
            title="Title A",
            description="Same description",
            category="Science",
            rules="Rules A",
            close_time="2026-05-20T00:00:00Z",
            outcomes=["Yes", "No"],
        )
        event2 = EventRequest(
            event_ticker="EVT-001",
            market_ticker="MKT-002",
            title="Title B",
            description="Same description",
            category="Sports",
            rules="Rules B",
            close_time="2026-05-20T00:00:00Z",
            outcomes=["Yes", "No"],
        )
        # Same event_ticker, description, outcomes, close_time → same key
        assert cache.cache_key(event1) == cache.cache_key(event2)


class TestCacheGetSet:
    """Tests for get and set operations."""

    @pytest.mark.asyncio
    async def test_get_returns_none_for_empty_cache(self, cache, sample_event):
        result = await cache.get(sample_event)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_round_trip(self, cache, sample_event, sample_probabilities):
        await cache.set(sample_event, sample_probabilities)
        result = await cache.get(sample_event)
        assert result == sample_probabilities

    @pytest.mark.asyncio
    async def test_get_returns_none_for_expired_entry(self, temp_db_path, sample_event, sample_probabilities):
        # Use a very short TTL
        cache = PredictionCache(db_path=temp_db_path, ttl_hours=0)
        # Manually insert an expired entry
        import aiosqlite

        now = datetime.now(timezone.utc)
        expired_at = now - timedelta(hours=1)

        async with aiosqlite.connect(temp_db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    cache_key TEXT PRIMARY KEY,
                    event_ticker TEXT NOT NULL,
                    probabilities TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            key = cache.cache_key(sample_event)
            await db.execute(
                """
                INSERT INTO prediction_cache
                (cache_key, event_ticker, probabilities, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    sample_event.event_ticker,
                    json.dumps(sample_probabilities),
                    (now - timedelta(hours=7)).isoformat(),
                    expired_at.isoformat(),
                ),
            )
            await db.commit()

        cache._initialized = True
        result = await cache.get(sample_event)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_overwrites_existing_entry(self, cache, sample_event):
        probs1 = {"Yes": 0.7, "No": 0.3}
        probs2 = {"Yes": 0.4, "No": 0.6}

        await cache.set(sample_event, probs1)
        await cache.set(sample_event, probs2)

        result = await cache.get(sample_event)
        assert result == probs2


class TestInvalidateExpired:
    """Tests for invalidate_expired."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_expired_entries(self, temp_db_path, sample_event, sample_probabilities):
        import aiosqlite

        cache = PredictionCache(db_path=temp_db_path, ttl_hours=6)
        await cache._ensure_table()

        # Insert an expired entry directly
        now = datetime.now(timezone.utc)
        expired_at = now - timedelta(hours=1)
        key = cache.cache_key(sample_event)

        async with aiosqlite.connect(temp_db_path) as db:
            await db.execute(
                """
                INSERT INTO prediction_cache
                (cache_key, event_ticker, probabilities, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    sample_event.event_ticker,
                    json.dumps(sample_probabilities),
                    (now - timedelta(hours=7)).isoformat(),
                    expired_at.isoformat(),
                ),
            )
            await db.commit()

        removed = await cache.invalidate_expired()
        assert removed == 1

    @pytest.mark.asyncio
    async def test_invalidate_keeps_valid_entries(self, cache, sample_event, sample_probabilities):
        await cache.set(sample_event, sample_probabilities)
        removed = await cache.invalidate_expired()
        assert removed == 0

        # Entry should still be retrievable
        result = await cache.get(sample_event)
        assert result == sample_probabilities

    @pytest.mark.asyncio
    async def test_invalidate_returns_zero_on_empty_cache(self, cache):
        removed = await cache.invalidate_expired()
        assert removed == 0


class TestTableCreation:
    """Tests for table initialization."""

    @pytest.mark.asyncio
    async def test_table_created_on_first_operation(self, temp_db_path, sample_event):
        cache = PredictionCache(db_path=temp_db_path)
        assert cache._initialized is False

        # First get triggers table creation
        await cache.get(sample_event)
        assert cache._initialized is True

    @pytest.mark.asyncio
    async def test_multiple_operations_dont_recreate_table(self, cache, sample_event, sample_probabilities):
        await cache.get(sample_event)
        await cache.set(sample_event, sample_probabilities)
        await cache.get(sample_event)
        # Should not raise any errors
        assert cache._initialized is True
