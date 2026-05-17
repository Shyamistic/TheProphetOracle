"""LLM-based probabilistic reasoning engine for the Prophet Forecasting Agent.

Synthesizes research evidence into probability estimates using structured
prompting via OpenRouter (OpenAI-compatible API). Includes timeout handling,
probability normalization, and reasoning trace generation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import AgentConfig
from src.models import (
    EventRequest,
    EvidenceItem,
    PredictionResult,
    ReasoningTrace,
    ResearchResult,
)

logger = logging.getLogger(__name__)

# Default timeout for a single prediction call (seconds)
DEFAULT_PREDICTION_TIMEOUT = 120


class ReasoningEngine:
    """Synthesizes research evidence into probability estimates.

    Uses the OpenAI SDK (via OpenRouter) to prompt an LLM with event context
    and gathered evidence, then parses structured JSON output into calibrated
    probability estimates with a full reasoning trace.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: int = DEFAULT_PREDICTION_TIMEOUT,
    ):
        """Initialize the reasoning engine.

        Args:
            api_key: API key for the LLM provider (OpenRouter).
            model: Model identifier to use for predictions.
            base_url: Base URL for the API (OpenRouter endpoint).
            timeout_seconds: Maximum time allowed for a single prediction.
        """
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @classmethod
    def from_config(cls, config: AgentConfig) -> "ReasoningEngine":
        """Create a ReasoningEngine from application configuration.

        Args:
            config: The AgentConfig instance with API keys and model settings.

        Returns:
            A configured ReasoningEngine instance.
        """
        return cls(
            api_key=config.anthropic_api_key,
            model=config.primary_model,
            timeout_seconds=DEFAULT_PREDICTION_TIMEOUT,
        )

    def build_prompt(self, event: EventRequest, research: ResearchResult,
                     market_stats: dict = None) -> str:
        """Construct a structured reasoning prompt with YES/NO thesis format.

        Uses FutureSearch-inspired approach that forces the LLM to:
        1. Consider base rates and historical precedents
        2. Steelman the YES thesis (strongest case for outcome 1)
        3. Steelman the NO thesis (strongest case against outcome 1)
        4. Identify key determining factors
        5. Synthesize into calibrated probabilities

        Args:
            event: The event to predict.
            research: Research results containing gathered evidence.
            market_stats: Optional dict of outcome -> market price info.

        Returns:
            A formatted prompt string for the LLM.
        """
        # Format evidence items
        evidence_text = self._format_evidence(research.evidence)

        # Format market information if available
        market_text = self._format_market_stats(market_stats)

        outcomes_str = ", ".join(f'"{o}"' for o in event.outcomes)
        outcome_1 = event.outcomes[0] if event.outcomes else "Yes"

        prompt = f"""You are an expert probabilistic forecaster. Analyze this event systematically.

## Event Information
- **Title**: {event.title}
- **Description**: {event.description}
- **Category**: {event.category}
- **Rules**: {event.rules}
- **Close Time**: {event.close_time}
- **Outcomes**: [{outcomes_str}]
{market_text}
## Research Evidence
{evidence_text}

## Your Analysis (follow these steps):

1. **BASE RATES**: What are the historical base rates for this type of event? How often do similar things happen?

2. **CURRENT STATE**: What is the current state of the world relevant to this question?

3. **YES THESIS**: What is the STRONGEST case that "{outcome_1}" will happen? Steelman this position with the best available evidence.

4. **NO THESIS**: What is the STRONGEST case that "{outcome_1}" will NOT happen? Steelman this position with the best available evidence.

5. **KEY FACTORS**: What 2-3 factors will most determine the outcome? Which direction does each factor point?

6. **SYNTHESIS**: Weighing all evidence — base rates, current state, both theses, and key factors — what probability do you assign to each outcome?

## Critical Rules
- Probabilities MUST be between 0.01 and 0.99 (never express absolute certainty)
- Probabilities MUST sum to exactly 1.0 across all outcomes
- If market prices are available, use them as your starting prior
- Be well-calibrated: when you say 70%, events should happen ~70% of the time

## Required Output Format

Respond with ONLY a JSON object in the following format (no markdown, no extra text):

{{
  "probabilities": {{
    "<outcome_1>": <float>,
    "<outcome_2>": <float>
  }},
  "reasoning_trace": {{
    "evidence_considered": ["<summary of each key evidence item considered>"],
    "base_rate": <float between 0 and 1>,
    "supporting_factors": ["<yes thesis key point 1>", "<yes thesis key point 2>"],
    "conflicting_evidence": ["<no thesis key point 1>", "<no thesis key point 2>"],
    "conflict_resolution": "<how you weighed yes vs no thesis>",
    "confidence_level": "<low|medium|high>"
  }}
}}

The outcomes are: [{outcomes_str}]
Ensure your probabilities keys match these outcome labels exactly and sum to 1.0."""

        return prompt

    def _format_market_stats(self, market_stats: dict = None) -> str:
        """Format market statistics into a prompt section.

        Args:
            market_stats: Dict of outcome -> price info.

        Returns:
            Formatted market information string, or empty string if unavailable.
        """
        if not market_stats:
            return ""

        lines = ["\n## Market Information",
                 "Current Kalshi market prices:"]

        for outcome, stats in market_stats.items():
            if isinstance(stats, dict):
                last_price = stats.get("last_price")
                if last_price is not None:
                    lines.append(f"- {outcome}: {last_price * 100:.0f}% (last traded)")

        lines.append("")
        lines.append("IMPORTANT: The market price represents the collective wisdom of traders with real money at stake.")
        lines.append("Use this as your starting prior. Only deviate significantly if your research provides strong")
        lines.append("evidence that the market is mispricing this event. Most of the time, your prediction should be")
        lines.append("within 10-15 percentage points of the market price.")
        lines.append("")

        return "\n".join(lines)

    def _format_evidence(self, evidence: List[EvidenceItem]) -> str:
        """Format evidence items into a readable text block for the prompt.

        Args:
            evidence: List of evidence items from research.

        Returns:
            Formatted string describing all evidence.
        """
        if not evidence:
            return "No research evidence was found for this event. Please rely on base rates and prior knowledge."

        lines = []
        for i, item in enumerate(evidence, 1):
            date_str = (
                item.publication_date.strftime("%Y-%m-%d")
                if item.publication_date
                else "Unknown date"
            )
            corr_str = "Corroborated" if item.corroborated else "Uncorroborated"
            lines.append(
                f"{i}. [{date_str}] ({corr_str}, relevance: {item.relevance_score:.2f})\n"
                f"   Source: {item.source_url}\n"
                f"   Summary: {item.summary}"
            )

        return "\n\n".join(lines)

    def parse_prediction(
        self, llm_response: str, outcomes: List[str]
    ) -> PredictionResult:
        """Parse structured LLM output into a PredictionResult.

        Extracts probabilities and reasoning trace from the JSON response.
        Applies normalization and clamping to ensure valid output.

        Args:
            llm_response: Raw text response from the LLM.
            outcomes: List of expected outcome labels.

        Returns:
            A PredictionResult with probabilities and reasoning trace.

        Raises:
            ValueError: If the response cannot be parsed into valid predictions.
        """
        # Clean the response - strip markdown code fences if present
        cleaned = llm_response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        # Extract probabilities
        raw_probs = data.get("probabilities", {})
        if not raw_probs:
            raise ValueError("No probabilities found in LLM response")

        # Map probabilities to outcomes, handling missing/extra keys
        probabilities = self._normalize_probabilities(raw_probs, outcomes)

        # Extract reasoning trace
        trace_data = data.get("reasoning_trace", {})
        reasoning_trace = self._parse_reasoning_trace(trace_data)

        return PredictionResult(
            event_ticker="",  # Will be set by caller
            probabilities=probabilities,
            reasoning_trace=reasoning_trace,
            duration_seconds=0.0,  # Will be set by caller
        )

    def _normalize_probabilities(
        self, raw_probs: Dict[str, float], outcomes: List[str]
    ) -> Dict[str, float]:
        """Normalize raw probabilities to satisfy constraints.

        Ensures:
        - All outcomes have a probability
        - Each probability is in [0.01, 0.99]
        - Probabilities sum to 1.0

        Args:
            raw_probs: Raw probability dict from LLM output.
            outcomes: Expected outcome labels.

        Returns:
            Normalized probability dict.
        """
        probs: Dict[str, float] = {}

        for outcome in outcomes:
            p = raw_probs.get(outcome, 0.0)
            try:
                p = float(p)
            except (TypeError, ValueError):
                p = 0.0
            # Clamp to [0.01, 0.99]
            p = max(0.01, min(0.99, p))
            probs[outcome] = p

        # Normalize to sum to 1.0
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            # Fallback to uniform
            n = len(outcomes)
            probs = {o: 1.0 / n for o in outcomes}

        # Final clamp after normalization (in case normalization pushed values out)
        probs = {k: max(0.01, min(0.99, v)) for k, v in probs.items()}

        # Re-normalize after clamping
        total = sum(probs.values())
        if abs(total - 1.0) > 0.001:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def _parse_reasoning_trace(self, trace_data: dict) -> ReasoningTrace:
        """Parse reasoning trace data from LLM output.

        Provides defaults for any missing fields to ensure a complete trace.

        Args:
            trace_data: Raw trace dict from LLM JSON output.

        Returns:
            A ReasoningTrace dataclass instance.
        """
        evidence_considered = trace_data.get("evidence_considered", [])
        if not isinstance(evidence_considered, list):
            evidence_considered = []

        base_rate = trace_data.get("base_rate", 0.5)
        try:
            base_rate = float(base_rate)
            base_rate = max(0.0, min(1.0, base_rate))
        except (TypeError, ValueError):
            base_rate = 0.5

        supporting_factors = trace_data.get("supporting_factors", [])
        if not isinstance(supporting_factors, list):
            supporting_factors = []
        # Ensure at least 2 supporting factors
        while len(supporting_factors) < 2:
            supporting_factors.append("Insufficient evidence for additional factors")

        conflicting_evidence = trace_data.get("conflicting_evidence", [])
        if not isinstance(conflicting_evidence, list):
            conflicting_evidence = []

        conflict_resolution = trace_data.get("conflict_resolution", "")
        if not isinstance(conflict_resolution, str):
            conflict_resolution = "No conflict resolution provided"

        confidence_level = trace_data.get("confidence_level", "low")
        if confidence_level not in ("low", "medium", "high"):
            confidence_level = "low"

        return ReasoningTrace(
            evidence_considered=evidence_considered,
            base_rate=base_rate,
            supporting_factors=supporting_factors,
            conflicting_evidence=conflicting_evidence,
            conflict_resolution=conflict_resolution,
            confidence_level=confidence_level,
        )

    async def predict(
        self, event: EventRequest, research: ResearchResult,
        market_stats: dict = None,
    ) -> PredictionResult:
        """Generate probability estimates from event and research evidence.

        Runs the full reasoning pipeline:
        1. Build structured prompt
        2. Call LLM with timeout
        3. Parse response into probabilities and trace
        4. Apply normalization constraints

        On timeout or error, returns a uniform distribution with low-confidence trace.

        Args:
            event: The event to predict.
            research: Research results with gathered evidence.
            market_stats: Optional dict of outcome -> market price info.

        Returns:
            A PredictionResult with probabilities and reasoning trace.
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._call_llm(event, research, market_stats),
                timeout=self.timeout_seconds,
            )
            result.event_ticker = event.event_ticker
            result.duration_seconds = time.time() - start_time
            return result

        except asyncio.TimeoutError:
            logger.warning(
                f"Reasoning timeout for event {event.event_ticker} "
                f"after {self.timeout_seconds}s"
            )
            return self._fallback_prediction(event, time.time() - start_time)

        except Exception as e:
            logger.error(
                f"Reasoning error for event {event.event_ticker}: {e}",
                exc_info=True,
            )
            return self._fallback_prediction(event, time.time() - start_time)

    async def _call_llm(
        self, event: EventRequest, research: ResearchResult,
        market_stats: dict = None,
    ) -> PredictionResult:
        """Make the actual LLM API call and parse the response.

        Args:
            event: The event to predict.
            research: Research results with gathered evidence.
            market_stats: Optional dict of outcome -> market price info.

        Returns:
            Parsed PredictionResult from LLM response.
        """
        prompt = self.build_prompt(event, research, market_stats)

        # Run the synchronous OpenAI client call in a thread pool
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Extract text content from the response
        llm_text = response.choices[0].message.content if response.choices else ""

        if not llm_text:
            raise ValueError("Empty response from LLM")

        result = self.parse_prediction(llm_text, event.outcomes)
        return result

    def _fallback_prediction(
        self, event: EventRequest, duration: float
    ) -> PredictionResult:
        """Generate a fallback uniform distribution prediction.

        Used when the LLM call times out or encounters an error.

        Args:
            event: The event that failed prediction.
            duration: Time elapsed before fallback was triggered.

        Returns:
            A PredictionResult with uniform probabilities and low-confidence trace.
        """
        n = len(event.outcomes)
        uniform_prob = 1.0 / n
        probabilities = {outcome: uniform_prob for outcome in event.outcomes}

        reasoning_trace = ReasoningTrace(
            evidence_considered=["Fallback: LLM call failed or timed out"],
            base_rate=uniform_prob,
            supporting_factors=[
                "Uniform distribution used as fallback",
                "No model-based reasoning available",
            ],
            conflicting_evidence=[],
            conflict_resolution="N/A - fallback prediction",
            confidence_level="low",
        )

        return PredictionResult(
            event_ticker=event.event_ticker,
            probabilities=probabilities,
            reasoning_trace=reasoning_trace,
            duration_seconds=duration,
        )
