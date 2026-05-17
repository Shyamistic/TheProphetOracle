"""Ensemble aggregator for combining multiple agent predictions.

Computes mean (or weighted mean) probability per outcome across agents,
then normalizes to ensure probabilities sum to 1.0.
"""

from typing import Dict, List

from src.models import PredictionResult


def aggregate_predictions(
    predictions: List[PredictionResult], outcomes: List[str]
) -> Dict[str, float]:
    """Compute mean probability across all agent predictions.

    For each outcome, average the probabilities from all agents.
    Normalize to ensure sum = 1.0.

    Args:
        predictions: List of PredictionResult from parallel agents.
        outcomes: List of outcome labels.

    Returns:
        Dict mapping outcome -> mean probability (normalized to sum 1.0).
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

    # Normalize to sum to 1.0
    prob_sum = sum(mean_probs.values())
    if prob_sum > 0:
        mean_probs = {k: v / prob_sum for k, v in mean_probs.items()}
    else:
        # If all probabilities are zero, fall back to uniform
        n = len(outcomes)
        mean_probs = {outcome: 1.0 / n for outcome in outcomes}

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
