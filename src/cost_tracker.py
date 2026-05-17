"""API cost monitoring and budget enforcement for the Prophet Forecasting Agent.

Tracks cumulative API costs, persists records to SQLite, and enforces
configurable budget limits with alert thresholds.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List

from src.models import APICallRecord, RoutingConfig

logger = logging.getLogger(__name__)

# Average cost estimates for cost projection
AVG_LLM_CALL_COST_USD = 0.003  # ~1000 input + 500 output tokens on Claude Sonnet
AVG_SEARCH_COST_USD = 0.001  # Tavily per-search cost estimate


class CostTracker:
    """Tracks cumulative API costs and enforces budget limits.

    Persists all cost records to a SQLite database for durability across
    restarts. Provides budget monitoring, per-category breakdowns, and
    cost estimation for event processing.
    """

    def __init__(
        self,
        budget_usd: float = 50.0,
        alert_threshold: float = 0.90,
        db_path: str = "cost_tracker.db",
    ):
        """Initialize the CostTracker.

        Args:
            budget_usd: Total budget in USD for the evaluation period.
            alert_threshold: Fraction of budget at which to trigger critical alert (0.0-1.0).
            db_path: Path to the SQLite database file for persistence.
        """
        self.budget_usd = budget_usd
        self.alert_threshold = alert_threshold
        self.db_path = db_path
        self.records: List[APICallRecord] = []
        self._init_db()
        self._load_records()

    def _init_db(self) -> None:
        """Create the cost_records table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    service TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    event_ticker TEXT NOT NULL,
                    category TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cost_category
                ON cost_records(category)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cost_timestamp
                ON cost_records(timestamp)
                """
            )
            conn.commit()

    def _load_records(self) -> None:
        """Load existing records from the database into memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT timestamp, service, model, input_tokens, output_tokens,
                       estimated_cost_usd, event_ticker, category
                FROM cost_records
                ORDER BY timestamp ASC
                """
            )
            for row in cursor:
                record = APICallRecord(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    service=row["service"],
                    model=row["model"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    estimated_cost_usd=row["estimated_cost_usd"],
                    event_ticker=row["event_ticker"],
                    category=row["category"],
                )
                self.records.append(record)

    def _persist_record(self, call: APICallRecord) -> None:
        """Persist a single record to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO cost_records
                    (timestamp, service, model, input_tokens, output_tokens,
                     estimated_cost_usd, event_ticker, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.timestamp.isoformat(),
                    call.service,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.estimated_cost_usd,
                    call.event_ticker,
                    call.category,
                ),
            )
            conn.commit()

    def record_call(self, call: APICallRecord) -> None:
        """Record an API call and check budget threshold.

        Appends the record to the in-memory list, persists to SQLite,
        and logs a warning if the budget threshold is reached.

        Args:
            call: The API call record to track.
        """
        self.records.append(call)
        self._persist_record(call)

        if self.is_budget_critical:
            logger.warning(
                f"Budget critical! Spend ${self.total_spend:.4f} "
                f"has reached {self.alert_threshold * 100:.0f}% of "
                f"${self.budget_usd:.2f} budget. "
                f"Remaining: ${self.budget_remaining:.4f}"
            )

    @property
    def total_spend(self) -> float:
        """Total cumulative spend in USD."""
        return sum(r.estimated_cost_usd for r in self.records)

    @property
    def budget_remaining(self) -> float:
        """Remaining budget in USD."""
        return self.budget_usd - self.total_spend

    @property
    def is_budget_critical(self) -> bool:
        """True if spend >= alert_threshold fraction of budget."""
        return self.total_spend >= self.budget_usd * self.alert_threshold

    def spend_by_category(self) -> Dict[str, float]:
        """Breakdown of spend per event category.

        Returns:
            Dict mapping category name to total spend in USD for that category.
        """
        breakdown: Dict[str, float] = {}
        for record in self.records:
            category = record.category
            breakdown[category] = breakdown.get(category, 0.0) + record.estimated_cost_usd
        return breakdown

    def estimate_event_cost(self, config: RoutingConfig) -> float:
        """Estimate cost for processing one event given its routing config.

        Estimates based on:
        - num_agents × average LLM call cost
        - max_searches × average search API cost

        Args:
            config: The routing configuration for the event.

        Returns:
            Estimated cost in USD.
        """
        llm_cost = config.num_agents * config.max_llm_calls * AVG_LLM_CALL_COST_USD
        search_cost = config.max_searches * AVG_SEARCH_COST_USD
        return llm_cost + search_cost

    def get_summary(self) -> dict:
        """Return full cost summary for the /costs endpoint.

        Returns:
            Dict containing total_spend, budget_usd, budget_remaining,
            is_budget_critical, alert_threshold, num_records,
            spend_by_category, and spend_by_service breakdowns.
        """
        spend_by_service: Dict[str, float] = {}
        for record in self.records:
            service = record.service
            spend_by_service[service] = (
                spend_by_service.get(service, 0.0) + record.estimated_cost_usd
            )

        return {
            "total_spend_usd": round(self.total_spend, 6),
            "budget_usd": self.budget_usd,
            "budget_remaining_usd": round(self.budget_remaining, 6),
            "is_budget_critical": self.is_budget_critical,
            "alert_threshold": self.alert_threshold,
            "num_records": len(self.records),
            "spend_by_category": {
                k: round(v, 6) for k, v in self.spend_by_category().items()
            },
            "spend_by_service": {
                k: round(v, 6) for k, v in spend_by_service.items()
            },
        }
