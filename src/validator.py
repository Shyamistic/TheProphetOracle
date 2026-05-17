"""Response validation and correction for prediction outputs.

Validates that prediction probabilities conform to the required format
(correct outcome count, range [0.01, 0.99], market field match).
For mutually exclusive events, also validates sum to 1.0.
For non-mutually-exclusive events (top-K), skips sum validation.
Provides correction and fallback mechanisms.
"""

from typing import Dict, List, Tuple


class ResponseValidator:
    """Validates and corrects prediction responses before sending."""

    def validate(
        self, probabilities: Dict[str, float], outcomes: List[str],
        mutually_exclusive: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Validate prediction against all rules.

        Checks:
        1. One entry per outcome (outcome count matches)
        2. Each probability in [0.01, 0.99]
        3. Each market field matches an outcome
        4. Sum equals 1.0 (within 0.001 tolerance) — ONLY for mutually exclusive events

        Args:
            probabilities: Dict mapping outcome label -> probability value.
            outcomes: List of expected outcome labels from the event.
            mutually_exclusive: Whether outcomes are mutually exclusive.

        Returns:
            Tuple of (is_valid, list_of_violations).
        """
        violations: List[str] = []

        # Check 1: One entry per outcome
        if len(probabilities) != len(outcomes):
            violations.append(
                f"Expected {len(outcomes)} outcomes, got {len(probabilities)}"
            )

        # Check 2: Each probability in [0.01, 0.99]
        for market, prob in probabilities.items():
            if prob < 0.01 - 1e-9 or prob > 0.99 + 1e-9:
                violations.append(
                    f"Probability for '{market}' is {prob}, "
                    f"must be in [0.01, 0.99]"
                )

        # Check 3: Each market field matches an outcome
        for market in probabilities:
            if market not in outcomes:
                violations.append(
                    f"Market '{market}' does not match any outcome"
                )

        # Also check for missing outcomes
        for outcome in outcomes:
            if outcome not in probabilities:
                violations.append(f"Missing probability for outcome '{outcome}'")

        # Check 4: Sum equals 1.0 within tolerance — ONLY for mutually exclusive
        if mutually_exclusive:
            total = sum(probabilities.values())
            if abs(total - 1.0) > 0.001:
                violations.append(
                    f"Probabilities sum to {total}, expected 1.0 (tolerance 0.001)"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def correct(
        self, probabilities: Dict[str, float], outcomes: List[str],
        mutually_exclusive: bool = True,
    ) -> Dict[str, float]:
        """Attempt to correct invalid predictions.

        Steps:
        1. Add missing outcomes with uniform probability (1/N)
        2. Clamp values to [0.01, 0.99]
        3. Normalize to sum to 1.0 (only for mutually exclusive events)

        Args:
            probabilities: Dict mapping outcome label -> probability value.
            outcomes: List of expected outcome labels from the event.
            mutually_exclusive: Whether outcomes are mutually exclusive.

        Returns:
            Corrected probabilities dict mapping outcome -> probability.
        """
        n = len(outcomes)
        uniform_prob = 1.0 / n

        # Step 1: Add missing outcomes with uniform probability
        corrected: Dict[str, float] = {}
        for outcome in outcomes:
            if outcome in probabilities:
                corrected[outcome] = probabilities[outcome]
            else:
                corrected[outcome] = uniform_prob

        # Step 2: Clamp values to [0.01, 0.99]
        for outcome in corrected:
            corrected[outcome] = max(0.01, min(0.99, corrected[outcome]))

        # Step 3: Normalize to sum to 1.0 (only for mutually exclusive)
        if not mutually_exclusive:
            return corrected

        total = sum(corrected.values())
        if total > 0:
            corrected = {k: v / total for k, v in corrected.items()}

        # Iteratively clamp and redistribute to ensure bounds are respected
        for _ in range(20):  # Max iterations to converge
            all_valid = True
            clamped_total = 0.0
            free_total = 0.0
            free_keys = []

            for k, v in corrected.items():
                if v < 0.01:
                    corrected[k] = 0.01
                    clamped_total += 0.01
                    all_valid = False
                elif v > 0.99:
                    corrected[k] = 0.99
                    clamped_total += 0.99
                    all_valid = False
                else:
                    free_keys.append(k)
                    free_total += v

            if all_valid and abs(sum(corrected.values()) - 1.0) <= 1e-9:
                break

            # Redistribute: free values must sum to (1.0 - clamped_total)
            target_free = 1.0 - clamped_total
            if free_keys and free_total > 0 and target_free > 0:
                scale = target_free / free_total
                for k in free_keys:
                    corrected[k] = corrected[k] * scale
            elif not free_keys:
                # All values are clamped; just normalize everything
                total = sum(corrected.values())
                if total > 0:
                    corrected = {k: v / total for k, v in corrected.items()}
                break

        return corrected

    def fallback_uniform(self, outcomes: List[str]) -> Dict[str, float]:
        """Return uniform distribution as last-resort fallback.

        Args:
            outcomes: List of outcome labels.

        Returns:
            Dict mapping each outcome to 1/N probability.
        """
        n = len(outcomes)
        return {outcome: 1.0 / n for outcome in outcomes}
