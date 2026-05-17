"""SQLite-backed prediction cache with configurable TTL.

Caches prediction results keyed by a SHA-256 hash of event parameters
to avoid redundant API calls for identical events within the TTL window.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import aiosqlite

from src.models import EventRequest


class PredictionCache:
    """SQLite-backed prediction cache with configurable TTL (default 6 hours)."""

    def __init__(self, db_path: str = "cache.db", ttl_hours: int = 6):
        self.db_path = db_path
        self.ttl = timedelta(hours=ttl_hours)
        self._initialized = False

    async def _ensure_table(self) -> None:
        """Create the prediction_cache table if it doesn't exist."""
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
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
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON prediction_cache(expires_at)
                """
            )
            await db.commit()
        self._initialized = True

    def cache_key(self, event: EventRequest) -> str:
        """Generate cache key as SHA-256 of event_ticker + description + sorted outcomes + close_time."""
        payload = json.dumps(
            {
                "event_ticker": event.event_ticker,
                "description": event.description,
                "outcomes": sorted(event.outcomes),
                "close_time": event.close_time,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, event: EventRequest) -> Optional[Dict[str, float]]:
        """Retrieve cached prediction if it exists and has not expired.

        Returns the probabilities dict if a valid cache entry is found, else None.
        """
        await self._ensure_table()
        key = self.cache_key(event)
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT probabilities FROM prediction_cache
                WHERE cache_key = ? AND expires_at > ?
                """,
                (key, now),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    async def set(self, event: EventRequest, probabilities: Dict[str, float]) -> None:
        """Store a prediction in the cache with the configured TTL."""
        await self._ensure_table()
        key = self.cache_key(event)
        now = datetime.now(timezone.utc)
        expires_at = now + self.ttl

        probabilities_json = json.dumps(probabilities, sort_keys=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO prediction_cache
                (cache_key, event_ticker, probabilities, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    event.event_ticker,
                    probabilities_json,
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            await db.commit()

    async def invalidate_expired(self) -> int:
        """Remove all entries past their expires_at timestamp.

        Returns the number of entries removed.
        """
        await self._ensure_table()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM prediction_cache WHERE expires_at <= ?
                """,
                (now,),
            )
            await db.commit()
            return cursor.rowcount
