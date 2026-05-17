"""Multi-model ensemble reasoning engine for improved forecasting accuracy.

Implements FutureSearch-inspired techniques:
- Structured YES/NO thesis prompting that forces consideration of both sides
- 3-model ensemble via OpenRouter (Claude, Gemini, GPT) run in parallel
- Featherless (Qwen 72B) as a 4th tiebreaker when models disagree >15%
- Median probability aggregation for robustness
- Graceful degradation: 2 models → single model → fallback
"""

import asyncio
import json
import logging
import statistics
import time
from typing import Dict, List, Optional, Tuple

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

# Timeout for each individual model call (seconds)
MODEL_CALL_TIMEOUT = 60


def _get_category_base_rates(category: str, outcomes: List[str]) -> str:
    """Provide category-specific base rate guidance for the LLM."""
    cat = category.lower() if category else "general"
    n = len(outcomes)

    if n == 2:  # Binary events
        guidance = {
            "sports": "For head-to-head sports matchups, the favorite wins ~60-65% of the time. Home advantage adds ~5%.",
            "economics": "For economic policy questions (rate cuts, etc.), the status quo (no change) occurs ~65% of the time. Markets tend toward stability.",
            "geopolitics": "For political events, incumbents/favorites win ~60% of the time. Dramatic changes are less common than continuity.",
            "technology": "For product launch/milestone questions, announced products ship ~75% of the time. Delays are common but cancellations are rare.",
            "science": "For space/science mission questions, scheduled missions launch on time ~40% of the time. Delays are very common.",
            "entertainment": "For entertainment outcomes, favorites/frontrunners win ~50% of the time. These are genuinely unpredictable.",
        }
        return f"\n## Base Rate Guidance\n{guidance.get(cat, 'No specific base rate available for this category.')}\n"
    else:
        return f"\n## Base Rate Guidance\nWith {n} outcomes, the uniform base rate is {100/n:.1f}% per outcome. Favorites typically have 2-3x the base rate.\n"


def build_structured_prompt(
    event: EventRequest,
    research: ResearchResult,
    market_stats: Optional[dict] = None,
) -> str:
    """Build a FutureSearch-style structured reasoning prompt.

    Forces the LLM to consider base rates, steelman both YES and NO theses,
    identify key factors, and only then synthesize a probability estimate.

    Args:
        event: The event to predict.
        research: Research results with gathered evidence.
        market_stats: Optional market price information.

    Returns:
        Formatted prompt string.
    """
    # Format evidence
    evidence_text = _format_evidence(research.evidence)

    # Format market info
    market_text = _format_market_info(market_stats)

    # Add category-specific base rate guidance
    base_rate_text = _get_category_base_rates(event.category, event.outcomes)

    # Format outcomes
    outcomes_str = ", ".join(f'"{o}"' for o in event.outcomes)
    outcome_1 = event.outcomes[0] if event.outcomes else "Yes"

    prompt = f"""You are an expert probabilistic forecaster. Analyze this event systematically.

## Event
Title: {event.title}
Description: {event.description}
Category: {event.category}
Rules: {event.rules}
Outcomes: [{outcomes_str}]
Close Time: {event.close_time}
{market_text}
{base_rate_text}
## Research Evidence
{evidence_text}

## Your Analysis (follow these steps):

1. BASE RATES: What are the historical base rates for this type of event? How often do similar things happen?

2. CURRENT STATE: What is the current state of the world relevant to this question?

3. YES THESIS: What is the STRONGEST case that "{outcome_1}" will happen? Steelman this position with the best available evidence.

4. NO THESIS: What is the STRONGEST case that "{outcome_1}" will NOT happen? Steelman this position with the best available evidence.

5. KEY FACTORS: What 2-3 factors will most determine the outcome? Which direction does each factor point?

6. SYNTHESIS: Weighing all evidence — base rates, current state, both theses, and key factors — what probability do you assign to each outcome?

## Critical Rules
- Probabilities MUST be between 0.02 and 0.98 (never express near-certainty)
- For mutually exclusive outcomes (e.g., "who wins"), probabilities should sum to approximately 1.0
- For non-mutually-exclusive outcomes (e.g., "which of these will qualify/finish top K"), each probability is INDEPENDENT and they do NOT need to sum to 1.0 — each represents P(this outcome is in the winning set)
- If market prices are available, use them as your starting prior — only deviate if evidence strongly justifies it
- Be well-calibrated: when you say 70%, events should happen ~70% of the time

## Output Format
Return ONLY a JSON object (no markdown, no extra text):

{{
  "probabilities": {{{", ".join(f'"{o}": <float>' for o in event.outcomes)}}},
  "reasoning_trace": {{
    "evidence_considered": ["<key evidence item 1>", "<key evidence item 2>"],
    "base_rate": <float between 0 and 1>,
    "supporting_factors": ["<yes thesis key point 1>", "<yes thesis key point 2>"],
    "conflicting_evidence": ["<no thesis key point 1>", "<no thesis key point 2>"],
    "conflict_resolution": "<how you weighed yes vs no thesis>",
    "confidence_level": "<low|medium|high>"
  }},
  "confidence": "<low|medium|high>"
}}

The outcomes are: [{outcomes_str}]
Ensure your probability keys match these outcome labels exactly."""

    return prompt


def _format_evidence(evidence: List[EvidenceItem]) -> str:
    """Format evidence items for the prompt."""
    if not evidence:
        return "No research evidence was found. Rely on base rates and prior knowledge."

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


def _format_market_info(market_stats: Optional[dict]) -> str:
    """Format market statistics into a prompt section."""
    if not market_stats:
        return ""

    lines = ["\n## Market Prices (use as Bayesian prior)"]
    for outcome, stats in market_stats.items():
        if isinstance(stats, dict):
            last_price = stats.get("last_price")
            if last_price is not None:
                lines.append(f"- {outcome}: {last_price * 100:.0f}%")

    lines.append("")
    lines.append("These prices reflect collective wisdom of traders with real money at stake.")
    lines.append("Use as your starting point. Only deviate significantly with strong evidence.")
    lines.append("")

    return "\n".join(lines)


class EnsembleReasoner:
    """Multi-model ensemble that runs structured prompts through 3 OpenRouter models.

    Uses Claude, Gemini, and GPT via OpenRouter in parallel, takes the median
    probability for robustness. If models disagree by >15% on any outcome and
    Featherless is available, adds Qwen 72B as a 4th tiebreaker opinion.
    """

    def __init__(
        self,
        openrouter_api_key: str,
        models: List[str],
        featherless_api_key: Optional[str] = None,
        featherless_model: str = "Qwen/Qwen2.5-72B-Instruct",
    ):
        """Initialize the ensemble reasoner.

        Args:
            openrouter_api_key: API key for OpenRouter (all 3 models).
            models: List of model IDs to run via OpenRouter.
            featherless_api_key: API key for Featherless (tiebreaker). If None, no tiebreaker.
            featherless_model: Model to use via Featherless for tiebreaking.
        """
        # All 3 primary models use the same OpenRouter client
        self.openrouter_client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.models = models  # e.g., ["anthropic/claude-sonnet-4", "google/gemini-3.1-pro-preview", "openai/gpt-5"]

        # Tiebreaker: Qwen via Featherless (if configured)
        self.featherless_client: Optional[OpenAI] = None
        self.featherless_model = featherless_model
        if featherless_api_key:
            self.featherless_client = OpenAI(
                api_key=featherless_api_key,
                base_url="https://api.featherless.ai/v1",
            )

    @classmethod
    def from_config(cls, config: AgentConfig) -> "EnsembleReasoner":
        """Create an EnsembleReasoner from application configuration.

        Args:
            config: The AgentConfig instance.

        Returns:
            A configured EnsembleReasoner instance.
        """
        return cls(
            openrouter_api_key=config.anthropic_api_key,
            models=[config.ensemble_model_1, config.ensemble_model_2, config.ensemble_model_3],
            featherless_api_key=config.featherless_api_key if config.use_ensemble else None,
            featherless_model=config.featherless_model,
        )

    async def predict(
        self,
        event: EventRequest,
        research: ResearchResult,
        market_stats: Optional[dict] = None,
    ) -> PredictionResult:
        """Generate probability estimates using 3-model ensemble.

        Runs the structured prompt through all 3 OpenRouter models in parallel,
        takes the median probability, and returns a unified result.

        If models disagree significantly (>15% spread) and Featherless is available,
        adds a 4th tiebreaker opinion before computing the median.

        Graceful degradation:
        - 3 models succeed → median of 3
        - 2 models succeed → median of 2
        - 1 model succeeds → use single result
        - 0 models succeed → fallback uniform distribution

        Args:
            event: The event to predict.
            research: Research results with gathered evidence.
            market_stats: Optional market price information.

        Returns:
            A PredictionResult with ensemble-aggregated probabilities.
        """
        start_time = time.time()
        prompt = build_structured_prompt(event, research, market_stats)

        # Run all 3 OpenRouter models in parallel
        model_tasks = [
            self._call_model(
                self.openrouter_client, model, prompt, event.outcomes, f"openrouter/{model}"
            )
            for model in self.models
        ]
        results = await asyncio.gather(*model_tasks, return_exceptions=True)

        # Collect successful results
        successful_results: List[Tuple[Dict[str, float], ReasoningTrace]] = []
        for i, result in enumerate(results):
            model_name = self.models[i]
            if isinstance(result, Exception):
                logger.warning(f"Ensemble model {model_name} failed: {result}")
            else:
                successful_results.append(result)
                logger.info(f"Ensemble model {model_name} succeeded")

        if not successful_results:
            # All models failed — return fallback
            logger.error("All ensemble models failed, using fallback prediction")
            return self._fallback_prediction(event, time.time() - start_time)

        had_disagreement = False

        if len(successful_results) == 1:
            # Only one model succeeded — use its result directly
            probs, trace = successful_results[0]
            logger.info("Ensemble: only 1 model succeeded, using single result")
        else:
            # Multiple models succeeded — check for disagreement
            if (
                self.featherless_client is not None
                and self._models_disagree(successful_results, event.outcomes)
            ):
                had_disagreement = True
                # Models disagree significantly — add Featherless tiebreaker
                logger.info(
                    "Ensemble: models disagree >15%, invoking Featherless tiebreaker "
                    f"({self.featherless_model})"
                )
                try:
                    tiebreaker_result = await self._call_model(
                        self.featherless_client,
                        self.featherless_model,
                        prompt,
                        event.outcomes,
                        "featherless/tiebreaker",
                    )
                    successful_results.append(tiebreaker_result)
                    logger.info("Featherless tiebreaker succeeded, now have "
                                f"{len(successful_results)} opinions")
                except Exception as e:
                    logger.warning(f"Featherless tiebreaker failed: {e}")
            elif self._models_disagree(successful_results, event.outcomes):
                # Disagreement detected but no Featherless client
                had_disagreement = True
                logger.info("Ensemble: models disagree >15% (no tiebreaker available)")

            # Aggregate using logit-space averaging (BLF method)
            probs, trace = self._aggregate_logit_average(successful_results, event.outcomes)

            # Log the spread
            spread_info = self._compute_spread(successful_results, event.outcomes)
            logger.info(
                f"Ensemble: {len(successful_results)} models succeeded, "
                f"logit-avg probabilities: {{{', '.join(f'{k}: {v:.3f}' for k, v in probs.items())}}}, "
                f"max spread: {spread_info:.3f}"
            )

        duration = time.time() - start_time
        return PredictionResult(
            event_ticker=event.event_ticker,
            probabilities=probs,
            reasoning_trace=trace,
            duration_seconds=duration,
            had_disagreement=had_disagreement,
        )

    def _models_disagree(
        self,
        results: List[Tuple[Dict[str, float], ReasoningTrace]],
        outcomes: List[str],
    ) -> bool:
        """Check if models disagree significantly (>15% spread on any outcome).

        Args:
            results: List of (probabilities, trace) tuples from each model.
            outcomes: Expected outcome labels.

        Returns:
            True if any outcome has a spread > 0.15 across models.
        """
        for outcome in outcomes:
            probs = [r[0].get(outcome, 0.5) for r in results]
            if max(probs) - min(probs) > 0.15:
                logger.debug(
                    f"Disagreement on '{outcome}': "
                    f"min={min(probs):.3f}, max={max(probs):.3f}, "
                    f"spread={max(probs) - min(probs):.3f}"
                )
                return True
        return False

    def _compute_spread(
        self,
        results: List[Tuple[Dict[str, float], ReasoningTrace]],
        outcomes: List[str],
    ) -> float:
        """Compute the maximum spread across all outcomes.

        Args:
            results: List of (probabilities, trace) tuples.
            outcomes: Expected outcome labels.

        Returns:
            Maximum spread value across all outcomes.
        """
        max_spread = 0.0
        for outcome in outcomes:
            probs = [r[0].get(outcome, 0.5) for r in results]
            spread = max(probs) - min(probs)
            max_spread = max(max_spread, spread)
        return max_spread

    async def _call_model(
        self,
        client: OpenAI,
        model: str,
        prompt: str,
        outcomes: List[str],
        model_label: str,
    ) -> Tuple[Dict[str, float], ReasoningTrace]:
        """Call a single model and parse its response.

        Args:
            client: OpenAI-compatible client.
            model: Model identifier.
            prompt: The structured prompt.
            outcomes: Expected outcome labels.
            model_label: Label for logging.

        Returns:
            Tuple of (probabilities dict, reasoning trace).

        Raises:
            Exception: If the model call or parsing fails.
        """
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=MODEL_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Model {model_label} timed out after {MODEL_CALL_TIMEOUT}s")

        llm_text = response.choices[0].message.content if response.choices else ""
        if not llm_text:
            raise ValueError(f"Empty response from {model_label}")

        logger.debug(f"Raw response from {model_label}: {llm_text[:200]}...")

        # Parse the response
        probs, trace = self._parse_response(llm_text, outcomes)
        logger.info(
            f"Model {model_label} ({model}) probabilities: "
            f"{{{', '.join(f'{k}: {v:.3f}' for k, v in probs.items())}}}"
        )
        return probs, trace

    def _parse_response(
        self, llm_text: str, outcomes: List[str]
    ) -> Tuple[Dict[str, float], ReasoningTrace]:
        """Parse LLM JSON response into probabilities and trace.

        Args:
            llm_text: Raw text response from the LLM.
            outcomes: Expected outcome labels.

        Returns:
            Tuple of (normalized probabilities, reasoning trace).

        Raises:
            ValueError: If parsing fails.
        """
        # Clean markdown code fences
        cleaned = llm_text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Try to extract JSON from the response (handle models that add text around JSON)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                try:
                    data = json.loads(cleaned[start_idx:end_idx])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON from response: {e}")
            else:
                raise ValueError("No JSON object found in response")

        # Extract probabilities
        raw_probs = data.get("probabilities", {})
        if not raw_probs:
            raise ValueError("No probabilities found in response")

        probs = self._normalize_probabilities(raw_probs, outcomes)

        # Extract reasoning trace
        trace_data = data.get("reasoning_trace", {})
        trace = self._parse_trace(trace_data)

        return probs, trace

    def _normalize_probabilities(
        self, raw_probs: Dict[str, float], outcomes: List[str]
    ) -> Dict[str, float]:
        """Normalize probabilities to valid range.

        For mutually exclusive events: normalizes to sum to 1.0
        For non-mutually-exclusive events: keeps raw values (each is independent)
        
        Detection: if raw probs sum to roughly 1.0 (within 0.3), treat as mutually exclusive.
        If they sum to significantly more (e.g., 3.0 for a top-5 event), keep as-is.
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

        # Detect if mutually exclusive: raw sum near 1.0
        total = sum(probs.values())
        n = len(outcomes)
        
        if total > 0 and abs(total - 1.0) < 0.3:
            # Looks mutually exclusive — normalize to sum to 1.0
            probs = {k: v / total for k, v in probs.items()}
            # Final clamp after normalization
            probs = {k: max(0.01, min(0.99, v)) for k, v in probs.items()}
            # Re-normalize after clamping
            total = sum(probs.values())
            if abs(total - 1.0) > 0.001:
                probs = {k: v / total for k, v in probs.items()}
        # else: non-mutually-exclusive — keep raw values, just clamped

        return probs

    def _parse_trace(self, trace_data: dict) -> ReasoningTrace:
        """Parse reasoning trace from LLM output."""
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

    def _aggregate_median(
        self,
        results: List[Tuple[Dict[str, float], ReasoningTrace]],
        outcomes: List[str],
    ) -> Tuple[Dict[str, float], ReasoningTrace]:
        """Aggregate multiple model results using element-wise median.

        For each outcome, takes the median probability across all models,
        then normalizes to sum to 1.0.

        Args:
            results: List of (probabilities, trace) tuples from each model.
            outcomes: Expected outcome labels.

        Returns:
            Tuple of (median probabilities, merged reasoning trace).
        """
        # Collect probabilities per outcome
        probs_per_outcome: Dict[str, List[float]] = {o: [] for o in outcomes}
        for probs, _ in results:
            for outcome in outcomes:
                probs_per_outcome[outcome].append(probs.get(outcome, 1.0 / len(outcomes)))

        # Take median for each outcome
        median_probs: Dict[str, float] = {}
        for outcome in outcomes:
            values = probs_per_outcome[outcome]
            median_probs[outcome] = statistics.median(values)

        # Normalize to sum to 1.0
        total = sum(median_probs.values())
        if total > 0:
            median_probs = {k: v / total for k, v in median_probs.items()}

        # Merge reasoning traces (combine evidence from all models)
        all_evidence = []
        all_supporting = []
        all_conflicting = []
        for _, trace in results:
            all_evidence.extend(trace.evidence_considered)
            all_supporting.extend(trace.supporting_factors)
            all_conflicting.extend(trace.conflicting_evidence)

        # Use the first model's trace as base, enriched with ensemble info
        base_trace = results[0][1]
        merged_trace = ReasoningTrace(
            evidence_considered=list(set(all_evidence))[:10],  # Deduplicate, cap at 10
            base_rate=base_trace.base_rate,
            supporting_factors=list(set(all_supporting))[:5],
            conflicting_evidence=list(set(all_conflicting))[:5],
            conflict_resolution=f"Ensemble median of {len(results)} models. {base_trace.conflict_resolution}",
            confidence_level=base_trace.confidence_level,
        )

        return median_probs, merged_trace

    def _aggregate_logit_average(
        self,
        results: List[Tuple[Dict[str, float], ReasoningTrace]],
        outcomes: List[str],
    ) -> Tuple[Dict[str, float], ReasoningTrace]:
        """Aggregate using logit-space averaging (BLF method).

        Instead of taking the median in probability space, converts to log-odds,
        averages in that space, then converts back. This properly handles extreme
        probabilities and is the state-of-the-art aggregation method.

        Formula: avg_logit = mean(log(p/(1-p))), final_p = sigmoid(avg_logit)
        """
        import math

        # Collect probabilities per outcome
        probs_per_outcome: Dict[str, List[float]] = {o: [] for o in outcomes}
        for probs, _ in results:
            for outcome in outcomes:
                p = probs.get(outcome, 1.0 / len(outcomes))
                # Clamp to avoid log(0) or log(inf)
                p = max(0.01, min(0.99, p))
                probs_per_outcome[outcome].append(p)

        # Logit-space averaging for each outcome
        aggregated_probs: Dict[str, float] = {}
        for outcome in outcomes:
            values = probs_per_outcome[outcome]
            # Convert to logits, average, convert back
            logits = [math.log(p / (1.0 - p)) for p in values]
            avg_logit = sum(logits) / len(logits)
            # Apply shrinkage toward 0 (slight pull toward 0.5) for robustness
            shrinkage = 0.95  # 5% shrinkage toward prior
            avg_logit = avg_logit * shrinkage
            # Convert back to probability
            aggregated_probs[outcome] = 1.0 / (1.0 + math.exp(-avg_logit))

        # Normalize to sum to 1.0
        total = sum(aggregated_probs.values())
        if total > 0:
            aggregated_probs = {k: v / total for k, v in aggregated_probs.items()}

        # Merge reasoning traces (same as before)
        all_evidence = []
        all_supporting = []
        all_conflicting = []
        for _, trace in results:
            all_evidence.extend(trace.evidence_considered)
            all_supporting.extend(trace.supporting_factors)
            all_conflicting.extend(trace.conflicting_evidence)

        base_trace = results[0][1]
        merged_trace = ReasoningTrace(
            evidence_considered=list(set(all_evidence))[:10],
            base_rate=base_trace.base_rate,
            supporting_factors=list(set(all_supporting))[:5],
            conflicting_evidence=list(set(all_conflicting))[:5],
            conflict_resolution=f"Logit-space average of {len(results)} models (BLF method). {base_trace.conflict_resolution}",
            confidence_level=base_trace.confidence_level,
        )

        return aggregated_probs, merged_trace

    def _fallback_prediction(
        self, event: EventRequest, duration: float
    ) -> PredictionResult:
        """Generate a fallback uniform distribution prediction.

        Args:
            event: The event that failed prediction.
            duration: Time elapsed before fallback.

        Returns:
            A PredictionResult with uniform probabilities.
        """
        n = len(event.outcomes)
        uniform_prob = 1.0 / n
        probabilities = {outcome: uniform_prob for outcome in event.outcomes}

        reasoning_trace = ReasoningTrace(
            evidence_considered=["Fallback: all ensemble models failed"],
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
