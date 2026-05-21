"""Lightweight Calibration Refit Module.

Implements Brier Patch's key insight: continuously recalibrate predictions
based on resolved outcomes. Uses a simple Platt scaling approach.

Strategy:
1. After each prediction, store (predicted_prob, event_ticker, outcome) in a local DB
2. Periodically check Kalshi for resolved markets
3. When we have ≥15 resolved predictions, fit a Platt scalar
4. Apply the scalar to all future predictions as a final calibration step

The Platt scalar corrects systematic over/under-confidence:
- If a > 1.0: we're under-confident (predictions too close to 0.5)
- If a < 1.0: we're over-confident (predictions too extreme)
- If a ≈ 1.0: we're well-calibrated

Formula: p_calibrated = sigmoid(a * logit(p))
"""

import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "calibration_data.db"
KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# Minimum resolved predictions before we start applying calibration
MIN_RESOLVED_FOR_REFIT = 15


def _init_db():
    """Initialize the calibration database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_ticker TEXT NOT NULL,
            outcome TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            timestamp TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            actual_outcome INTEGER DEFAULT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            platt_a REAL DEFAULT 1.0,
            last_refit_time TEXT DEFAULT NULL,
            num_resolved INTEGER DEFAULT 0,
            brier_score REAL DEFAULT NULL
        )
    """)
    # Ensure calibration_state has a row
    conn.execute("""
        INSERT OR IGNORE INTO calibration_state (id, platt_a, num_resolved)
        VALUES (1, 1.0, 0)
    """)
    conn.commit()
    conn.close()


def store_prediction(event_ticker: str, probabilities: Dict[str, float]):
    """Store a prediction for later calibration analysis.
    
    Args:
        event_ticker: The event ticker (e.g., KXAAAGASD-26MAY21)
        probabilities: Dict mapping outcome -> predicted probability
    """
    try:
        _init_db()
        conn = sqlite3.connect(str(DB_PATH))
        timestamp = datetime.now(timezone.utc).isoformat()
        for outcome, prob in probabilities.items():
            conn.execute(
                "INSERT INTO predictions (event_ticker, outcome, predicted_prob, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (event_ticker, outcome, prob, timestamp),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"Failed to store prediction for calibration: {e}")


async def check_resolutions():
    """Check Kalshi for resolved markets and update our calibration DB.
    
    Looks at our stored predictions and checks if the corresponding
    Kalshi markets have settled.
    """
    try:
        _init_db()
        conn = sqlite3.connect(str(DB_PATH))
        
        # Get unresolved event tickers
        cursor = conn.execute(
            "SELECT DISTINCT event_ticker FROM predictions WHERE resolved = 0"
        )
        unresolved_tickers = [row[0] for row in cursor.fetchall()]
        
        if not unresolved_tickers:
            conn.close()
            return
        
        resolved_count = 0
        async with httpx.AsyncClient(timeout=10.0) as client:
            for ticker in unresolved_tickers[:20]:  # Check max 20 per call
                try:
                    response = await client.get(
                        f"{KALSHI_BASE_URL}/events/{ticker}",
                        params={"with_nested_markets": "true"},
                    )
                    if response.status_code != 200:
                        continue
                    
                    data = response.json()
                    event = data.get("event", {})
                    markets = event.get("markets", [])
                    
                    for mkt in markets:
                        result = mkt.get("result")
                        if result in ("yes", "no"):
                            # Market is resolved
                            label = (
                                mkt.get("yes_sub_title") or
                                mkt.get("subtitle") or
                                mkt.get("title") or
                                ""
                            )
                            if not label:
                                continue
                            
                            # Mark as resolved: actual_outcome = 1 if "yes", 0 if "no"
                            actual = 1 if result == "yes" else 0
                            conn.execute(
                                "UPDATE predictions SET resolved = 1, actual_outcome = ? "
                                "WHERE event_ticker = ? AND outcome = ?",
                                (actual, ticker, label),
                            )
                            resolved_count += 1
                    
                    # For binary events (Yes/No), check the event-level result
                    if len(markets) == 1 and markets[0].get("result"):
                        result = markets[0]["result"]
                        # Mark "Yes" outcome
                        conn.execute(
                            "UPDATE predictions SET resolved = 1, actual_outcome = ? "
                            "WHERE event_ticker = ? AND outcome = 'Yes'",
                            (1 if result == "yes" else 0, ticker),
                        )
                        conn.execute(
                            "UPDATE predictions SET resolved = 1, actual_outcome = ? "
                            "WHERE event_ticker = ? AND outcome = 'No'",
                            (1 if result == "no" else 0, ticker),
                        )
                        resolved_count += 2
                        
                except Exception as e:
                    logger.debug(f"Failed to check resolution for {ticker}: {e}")
                    continue
        
        conn.commit()
        
        if resolved_count > 0:
            logger.info(f"Calibration refit: resolved {resolved_count} new predictions")
            # Trigger refit if we have enough data
            _maybe_refit(conn)
        
        conn.close()
        
    except Exception as e:
        logger.debug(f"Calibration resolution check failed: {e}")


def _maybe_refit(conn: sqlite3.Connection):
    """Refit the Platt scalar if we have enough resolved data."""
    cursor = conn.execute(
        "SELECT predicted_prob, actual_outcome FROM predictions "
        "WHERE resolved = 1 AND actual_outcome IS NOT NULL"
    )
    data = cursor.fetchall()
    
    if len(data) < MIN_RESOLVED_FOR_REFIT:
        logger.info(
            f"Calibration refit: only {len(data)} resolved, need {MIN_RESOLVED_FOR_REFIT}"
        )
        return
    
    # Fit Platt scalar using grid search (simple but effective)
    # We want to find 'a' that minimizes Brier score:
    # p_cal = sigmoid(a * logit(p))
    best_a = 1.0
    best_brier = float("inf")
    
    for a_candidate in [x / 100.0 for x in range(70, 151, 2)]:  # 0.70 to 1.50
        brier_sum = 0.0
        for pred_prob, actual in data:
            # Clamp to avoid log(0)
            p = max(0.01, min(0.99, pred_prob))
            logit_p = math.log(p / (1.0 - p))
            p_cal = 1.0 / (1.0 + math.exp(-a_candidate * logit_p))
            brier_sum += (p_cal - actual) ** 2
        
        avg_brier = brier_sum / len(data)
        if avg_brier < best_brier:
            best_brier = avg_brier
            best_a = a_candidate
    
    # Update calibration state
    conn.execute(
        "UPDATE calibration_state SET platt_a = ?, last_refit_time = ?, "
        "num_resolved = ?, brier_score = ? WHERE id = 1",
        (best_a, datetime.now(timezone.utc).isoformat(), len(data), best_brier),
    )
    conn.commit()
    
    logger.info(
        f"Calibration refit complete: a={best_a:.3f}, "
        f"brier={best_brier:.4f}, n={len(data)}"
    )


def get_platt_scalar() -> float:
    """Get the current Platt scalar for calibration.
    
    Returns 1.0 (no-op) if not enough data for refit.
    """
    try:
        _init_db()
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute(
            "SELECT platt_a, num_resolved FROM calibration_state WHERE id = 1"
        )
        row = cursor.fetchone()
        conn.close()
        
        if row and row[1] >= MIN_RESOLVED_FOR_REFIT:
            return row[0]
        return 1.0
    except Exception:
        return 1.0


def apply_calibration(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Apply Platt calibration to a set of probabilities.
    
    If we don't have enough resolved data yet, returns probabilities unchanged.
    
    Args:
        probabilities: Dict mapping outcome -> probability
        
    Returns:
        Calibrated probabilities (same keys, adjusted values)
    """
    platt_a = get_platt_scalar()
    
    if abs(platt_a - 1.0) < 0.01:
        # No meaningful calibration to apply
        return probabilities
    
    calibrated = {}
    for outcome, p in probabilities.items():
        p = max(0.01, min(0.99, p))
        logit_p = math.log(p / (1.0 - p))
        p_cal = 1.0 / (1.0 + math.exp(-platt_a * logit_p))
        # Clamp to safe range
        p_cal = max(0.02, min(0.98, p_cal))
        calibrated[outcome] = p_cal
    
    # Normalize (maintain sum = 1 for mutually exclusive)
    total = sum(calibrated.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        calibrated = {k: v / total for k, v in calibrated.items()}
    
    logger.info(f"Calibration applied: platt_a={platt_a:.3f}")
    return calibrated


def get_calibration_status() -> Dict:
    """Get current calibration status for monitoring."""
    try:
        _init_db()
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute(
            "SELECT platt_a, last_refit_time, num_resolved, brier_score "
            "FROM calibration_state WHERE id = 1"
        )
        row = cursor.fetchone()
        
        total_predictions = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        
        conn.close()
        
        if row:
            return {
                "platt_a": row[0],
                "last_refit": row[1],
                "num_resolved": row[2],
                "brier_score": row[3],
                "total_stored": total_predictions,
                "active": row[2] >= MIN_RESOLVED_FOR_REFIT,
            }
        return {"active": False, "total_stored": 0}
    except Exception:
        return {"active": False, "error": "DB not initialized"}
