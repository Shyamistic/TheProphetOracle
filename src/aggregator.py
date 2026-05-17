"""Ensemble aggregator for combining multiple agent predictions.

Computes mean (or weighted mean) probability per outcome across agents.
Handles both mutually exclusive events (normalizes to sum 1.0) and
non-mutually-exclusive events (keeps independent probabilities).
"""

import logging
from typing import Dict, List

from src.models import PredictionResult

logger = logging.getLogger(__name__)


def _is_mutually_exclusive(predictions: List[PredictionResult], outcomes: List[str]) -> bool:
    """Detect whether predictions represent mutually exclusive outcomes.
    
    If the models returned probabilities that sum close to 1.0, the event
    is mutually exclusive. If they sum significantly above 1.0, it's a
    non-mutually-exclusive (top-K) event.
    """
    if not predictions:
        return True
    
    sums = []
    for pred in predictions:
        total = sum(pred.probabilities.get(o, 0.0) for o in outcomes)
        sums.append(total)
    
    avg_sum = sum(sums) / len(sums)
    # If average sum is within 30% of 1.0, treat as mutually exclusive
    return avg_sum <= 1.3


def aggregate_predictions(
    predictions: List[PredictionResult], outcomes: List[str],
    mutually_exclusive: bool = True,
) -> Dict[str, float]:
    """Compute mean probability across all agent predictions.

    For each outcome, average the probabilities from all agents.
    Only normalizes to sum 1.0 for mutually exclusive events.
    For non-mutually-exclusive events (top-K), keeps independent probabilities.

    Args:
        predictions: List of PredictionResult from parallel agents.
        outcomes: List of outcome labels.
        mutually_exclusive: Whether outcomes are mutually exclusive. If False,
            skips normalization regardless of what models returned.

    Returns:
        Dict mapping outcome -> mean probability.
    """
    if not predictions:
        # Fallback to uniform distribution if no predictions available
        n = len(outcomes)
        return {outcome: 1.0 / n for outcome in outcomes}

    num_agents = len(predictions)
    mean_probs: Dict[str, float] = {}

    for outcome in outcomes:
        total = 0.0
        for pred in predictions:
            total += pred.probabilities.get(outcome, 0.0)
        mean_probs[outcome] = total / num_agents

    # Only normalize for mutually exclusive events
    if mutually_exclusive:
        prob_sum = sum(mean_probs.values())
        if prob_sum > 0:
            mean_probs = {k: v / prob_sum for k, v in mean_probs.items()}
        else:
            n = len(outcomes)
            mean_probs = {outcome: 1.0 / n for outcome in outcomes}
    else:
        logger.info(
            f"Non-mutually-exclusive event: skipping normalization "
            f"(raw sum={sum(mean_probs.values()):.2f})"
        )

    return mean_probs


def weighted_aggregate(
    predictions: List[PredictionResult],
    weights: List[float],
    outcomes: List[str],
) -> Dict[str, float]:
    """Weighted mean aggregation for combining agent predictions.

    Computes a weighted average of probabilities per outcome, then normalizes
    to ensure the result sums to 1.0. Intended for future use with agent
    quality scores.

    Args:
        predictions: List of PredictionResult from parallel agents.
        weights: List of floats (one per prediction) representing agent quality.
        outcomes: List of outcome labels.

    Returns:
        Dict mapping outcome -> weighted mean probability (normalized to sum 1.0).
    """
    if not predictions or not weights:
        n = len(outcomes)
        return {outcome: 1.0 / n for outcome in outcomes}

    weight_sum = sum(weights)
    if weight_sum <= 0:
        # Fall back to uniform aggregation if weights are invalid
        return aggregate_predictions(predictions, outcomes)

    weighted_probs: Dict[str, float] = {}

    for outcome in outcomes:
        total = 0.0
        for pred, weight in zip(predictions, weights):
            total += pred.probabilities.get(outcome, 0.0) * weight
        weighted_probs[outcome] = total / weight_sum

    # Normalize to sum to 1.0
    prob_sum = sum(weighted_probs.values())
    if prob_sum > 0:
        weighted_probs = {k: v / prob_sum for k, v in weighted_probs.items()}
    else:
        n = len(outcomes)
        weighted_probs = {outcome: 1.0 / n for outcome in outcomes}

    return weighted_probs
