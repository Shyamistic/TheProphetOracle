"""Supervisor agent for final prediction reconciliation.

Inspired by the AIA Forecaster paper: a lightweight second-pass LLM call that
reconciles the agent's prediction with market prices and checks for common biases.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Reconciles predictions with market data and checks for biases.

    Makes a single lightweight LLM call to:
    1. Compare our prediction against market consensus
    2. Check for overconfidence/underconfidence
    3. Produce a final adjusted prediction
    """

    def __init__(self, api_key: str, model: str = "anthropic/claude-sonnet-4"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    @classmethod
    def from_config(cls, config) -> "SupervisorAgent":
        """Create from AgentConfig."""
        return cls(api_key=config.anthropic_api_key, model=config.primary_model)

    async def reconcile(
        self,
        predictions: Dict[str, float],
        market_stats: Optional[Dict] = None,
        evidence_summary: str = "",
        event_title: str = "",
        outcomes: Optional[List[str]] = None,
        mutually_exclusive: bool = True,
    ) -> Dict[str, float]:
        """Reconcile our prediction with market prices.

        Args:
            predictions: Our agent's aggregated prediction.
            market_stats: Raw market_stats dict from request.
            evidence_summary: Brief summary of key evidence found.
            event_title: The event question.
            outcomes: List of outcome labels.
            mutually_exclusive: Whether outcomes are mutually exclusive.

        Returns:
            Adjusted prediction dict.
        """
        if outcomes is None:
            outcomes = list(predictions.keys())

        # Extract market prices from market_stats
        market_prices = self._extract_prices(market_stats)

        if not market_prices:
            return predictions

        prompt = self._build_prompt(
            event_title, outcomes, predictions, market_prices, evidence_summary,
            mutually_exclusive=mutually_exclusive,
        )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30,  # 30 second timeout for supervisor
            )

            text = response.choices[0].message.content if response.choices else ""
            return self._parse_response(text, outcomes, predictions, market_prices, mutually_exclusive=mutually_exclusive)

        except (Exception, asyncio.TimeoutError) as e:
            logger.warning(f"Supervisor reconciliation failed: {e}. Using weighted blend.")
            return self._weighted_blend(predictions, market_prices, weight=0.7, mutually_exclusive=mutually_exclusive)

    def _extract_prices(self, market_stats: Optional[Dict]) -> Optional[Dict[str, float]]:
        """Extract outcome -> last_price mapping from market_stats."""
        if not market_stats:
            return None
        prices = {}
        for outcome, stats in market_stats.items():
            if isinstance(stats, dict):
                price = stats.get("last_price")
                if price is not None:
                    try:
                        prices[outcome] = float(price)
                    except (TypeError, ValueError):
                        pass
        return prices if prices else None

    def _build_prompt(
        self,
        event_title: str,
        outcomes: List[str],
        our_prediction: Dict[str, float],
        market_prices: Dict[str, float],
        evidence_summary: str,
        mutually_exclusive: bool = True,
    ) -> str:
        our_str = ", ".join(f"{k}: {v:.1%}" for k, v in our_prediction.items())
        market_str = ", ".join(f"{k}: {v:.1%}" for k, v in market_prices.items())

        normalization_rule = "7. Probabilities must sum to 1.0." if mutually_exclusive else "7. Probabilities are INDEPENDENT (non-mutually-exclusive event). They do NOT need to sum to 1.0. Each represents P(this outcome is in the winning set)."

        return f"""You are a forecasting supervisor. Your job is to produce the FINAL probability estimate by reconciling two sources:

EVENT: {event_title}
OUTCOMES: {', '.join(outcomes)}

OUR RESEARCH-BASED PREDICTION: {our_str}
KALSHI MARKET PRICE: {market_str}

KEY EVIDENCE: {evidence_summary[:500]}

RULES FOR RECONCILIATION:
1. Market prices reflect collective wisdom of traders with real money at stake. They are usually well-calibrated.
2. Our research may have found NEW information the market hasn't priced in yet.
3. If our prediction and market agree (within 10%), trust the consensus.
4. If they disagree significantly, ask: does our evidence justify deviating from the market?
5. Be skeptical of large deviations from market — the market is usually right.
6. Never go below 0.05 or above 0.95 for any outcome.
{normalization_rule}

Return ONLY a JSON object with your final probabilities:
{{"probabilities": {{{", ".join(f'"{o}": <float>' for o in outcomes)}}}}}"""

    def _parse_response(
        self,
        text: str,
        outcomes: List[str],
        our_prediction: Dict[str, float],
        market_prices: Dict[str, float],
        mutually_exclusive: bool = True,
    ) -> Dict[str, float]:
        """Parse supervisor response, fallback to weighted blend on failure."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            data = json.loads(cleaned)
            probs = data.get("probabilities", {})

            result = {}
            for outcome in outcomes:
                p = float(probs.get(outcome, our_prediction.get(outcome, 1.0 / len(outcomes))))
                result[outcome] = max(0.05, min(0.95, p))

            # Normalize only for mutually exclusive
            if mutually_exclusive:
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}

            return result

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse supervisor response: {e}")
            return self._weighted_blend(our_prediction, market_prices, weight=0.7, mutually_exclusive=mutually_exclusive)

    def _weighted_blend(
        self,
        our_prediction: Dict[str, float],
        market_prices: Dict[str, float],
        weight: float = 0.7,
        mutually_exclusive: bool = True,
    ) -> Dict[str, float]:
        """Simple weighted average: weight * ours + (1-weight) * market."""
        result = {}
        for outcome in our_prediction:
            our_p = our_prediction.get(outcome, 0.5)
            market_p = market_prices.get(outcome, 0.5)
            result[outcome] = our_p * weight + market_p * (1 - weight)

        # Normalize only for mutually exclusive
        if mutually_exclusive:
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}

        return result
