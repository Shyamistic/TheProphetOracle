"""Calibration module for adjusting raw probabilities to improve Brier score.

Implements Platt scaling with √3 coefficient (Neyman & Roughgarden 2022) and
overconfidence shrinkage to produce well-calibrated probability estimates.
Supports market-price anchoring when Kalshi prices are available.
"""

import math
import logging
from typing import Dict, Optional

# Platt scaling coefficient from Neyman & Roughgarden 2022
PLATT_COEFFICIENT = math.sqrt(3)

# Default shrinkage factor toward 0.5
DEFAULT_SHRINKAGE = 0.15

# Default market anchor weight
DEFAULT_MARKET_ANCHOR_WEIGHT = 0.3

logger = logging.getLogger(__name__)


class CalibrationModule:
    """Adjusts raw probabilities to improve calibration and minimize Brier score.

    The calibration pipeline applies:
    1. Platt scaling (extremize via log-odds transformation with √3 coefficient)
    2. Overconfidence shrinkage for extreme values (>0.90 or <0.10)
    3. Clamping to [0.01, 0.99]
    4. Normalization to sum to 1.0
    """

    def __init__(
        self,
        shrinkage_factor: float = DEFAULT_SHRINKAGE,
        platt_coefficient: float = PLATT_COEFFICIENT,
        market_anchor_weight: float = DEFAULT_MARKET_ANCHOR_WEIGHT,
    ):
        self.shrinkage_factor = shrinkage_factor
        self.platt_coefficient = platt_coefficient
        self.market_anchor_weight = market_anchor_weight

    def platt_scale(self, p: float) -> float:
        """Apply Platt scaling with √3 coefficient.

        Transforms probability to log-odds space, multiplies by √3,
        and transforms back. This extremizes predictions (moves away from 0.5).

        Formula:
            log_odds = log(p / (1 - p))
            scaled_log_odds = log_odds * √3
            result = 1 / (1 + exp(-scaled_log_odds))

        Args:
            p: Input probability in (0, 1).

        Returns:
            Platt-scaled probability.
        """
        # Clamp input to avoid log(0) or division by zero
        p = max(0.001, min(0.999, p))

        log_odds = math.log(p / (1.0 - p))
        scaled_log_odds = log_odds * self.platt_coefficient
        result = 1.0 / (1.0 + math.exp(-scaled_log_odds))
        return result

    def shrink_extreme(self, p: float) -> float:
        """Shrink extreme probabilities (>0.90 or <0.10) toward 0.5.

        Only applies to values outside the [0.10, 0.90] range.
        Moves the value toward 0.5 by shrinkage_factor.

        Formula:
            if p > 0.90: p = p - shrinkage_factor * (p - 0.5)
            if p < 0.10: p = p + shrinkage_factor * (0.5 - p)

        Args:
            p: Input probability.

        Returns:
            Shrunk probability (unchanged if within [0.10, 0.90]).
        """
        if p > 0.90:
            p = p - self.shrinkage_factor * (p - 0.5)
        elif p < 0.10:
            p = p + self.shrinkage_factor * (0.5 - p)
        return p

    def clamp(self, p: float) -> float:
        """Clamp probability to [0.01, 0.99] range.

        Args:
            p: Input probability.

        Returns:
            Clamped probability.
        """
        return max(0.01, min(0.99, p))

    def normalize(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Normalize probabilities to sum to 1.0.

        Args:
            probabilities: Dict mapping outcome labels to probability values.

        Returns:
            Normalized dict where all values sum to 1.0.
        """
        total = sum(probabilities.values())
        if total == 0:
            # Avoid division by zero — return uniform
            n = len(probabilities)
            return {k: 1.0 / n for k in probabilities}
        return {k: v / total for k, v in probabilities.items()}

    def calibrate(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Apply full calibration pipeline.

        Steps:
            1. Apply Platt scaling (extremize via log-odds transformation)
            2. Apply overconfidence shrinkage for values > 0.90 or < 0.10
            3. Clamp to [0.01, 0.99]
            4. Normalize to sum to 1.0

        On error, returns original estimates clamped and normalized.

        Args:
            probabilities: Dict mapping outcome labels to raw probability values.

        Returns:
            Calibrated and normalized probability dict.
        """
        try:
            calibrated = {}
            for outcome, p in probabilities.items():
                # Step 1: Platt scaling
                scaled = self.platt_scale(p)
                # Step 2: Overconfidence shrinkage
                shrunk = self.shrink_extreme(scaled)
                # Step 3: Clamp
                clamped = self.clamp(shrunk)
                calibrated[outcome] = clamped

            # Step 4: Normalize to sum to 1.0
            return self.normalize(calibrated)

        except Exception as e:
            logger.warning(
                "Calibration error, returning clamped/normalized originals: %s", e
            )
            # Fallback: clamp and normalize originals
            fallback = {k: self.clamp(v) for k, v in probabilities.items()}
            return self.normalize(fallback)

    def calibrate_with_market(
        self,
        probabilities: Dict[str, float],
        market_prices: Optional[Dict[str, float]] = None,
        anchor_weight: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calibrate with market price anchoring.

        When market prices are available, we SKIP Platt scaling (since the
        supervisor has already reconciled with market) and just apply light
        anchoring + clamping + normalization.

        This avoids the problem of Platt scaling undoing the supervisor's
        market-informed adjustment.

        Args:
            probabilities: Dict mapping outcome labels to probability values
                (already reconciled by supervisor when market data exists).
            market_prices: Optional dict mapping outcome labels to market prices (0-1).
            anchor_weight: How much to additionally anchor toward market (0-1).

        Returns:
            Final calibrated probability dict.
        """
        if anchor_weight is None:
            anchor_weight = self.market_anchor_weight

        # If no market prices, apply standard calibration (with Platt scaling)
        if not market_prices:
            return self.calibrate(probabilities)

        # With market prices: skip Platt scaling, just anchor + clamp + normalize
        # The supervisor already reconciled our prediction with market data,
        # so we only apply a light additional anchor to stay close to market.
        try:
            anchored = {}
            for outcome, pred in probabilities.items():
                market_price = market_prices.get(outcome)
                if market_price is not None and 0.0 < market_price < 1.0:
                    # Light anchor toward market (supervisor already did heavy lifting)
                    adjusted = pred * (1.0 - anchor_weight) + market_price * anchor_weight
                    anchored[outcome] = self.clamp(adjusted)
                else:
                    anchored[outcome] = self.clamp(pred)

            # Normalize after anchoring
            return self.normalize(anchored)

        except Exception as e:
            logger.warning(
                "Market anchoring failed, returning standard calibration: %s", e
            )
            return self.calibrate(probabilities)
