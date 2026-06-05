"""
src/scoring/setup_score_engine.py — Setup validity scoring engine
Produces the setup_score (0–100) that answers:

    "Is the chart setup valid?"

This score focuses on the quality of the technical setup itself.  It does
not approve trades, place orders, or decide position size.  It is one input
to trade_quality_gate.py.

Weights from the bot overview:
  Setup confirmation:        20 pts
  VWAP / key level behavior: 15 pts
  Volume / RVOL:             15 pts
  Market structure:          12 pts
  Opening range status:      10 pts
  Liquidity sweep / reclaim: 10 pts
  Fibonacci confirmation:     8 pts
  Risk / reward:             10 pts

Score labels:
  90–100 = elite
  80–89  = strong
  70–79  = decent
  60–69  = weak
  below 60 = reject

Design rules:
  - This file does not approve trades
  - This file does not place orders
  - Setup detectors only detect; this file scores the overall setup context
  - Hard rejection rules still belong in trade_quality_gate.py
  - Scores are transparent and include reasons/warnings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from confidence_labeler import label_score
from models import ConfidenceLabel, IndicatorSnapshot, SetupResult

log = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SetupScoreResult:
    """
    Result of setup validity scoring.
    Consumed by probability_engine.py and trade_quality_gate.py.
    """
    ticker:            str
    setup_name:        str = "none"
    setup_score:       float = 0.0
    confidence_label:  str = ConfidenceLabel.REJECT.value

    # Score component breakdown
    score_breakdown:   dict = field(default_factory=dict)

    # Context flags
    setup_confirmed:   bool = False
    strong_setup:      bool = False
    weak_setup:        bool = False

    reasons:           list[str] = field(default_factory=list)
    warnings:          list[str] = field(default_factory=list)

    scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "setup_name":       self.setup_name,
            "setup_score":      round(self.setup_score, 2),
            "confidence_label": self.confidence_label,
            "score_breakdown":  self.score_breakdown,
            "setup_confirmed":  self.setup_confirmed,
            "strong_setup":     self.strong_setup,
            "weak_setup":       self.weak_setup,
            "reasons":          self.reasons,
            "warnings":         self.warnings,
            "scored_at":        self.scored_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class SetupScoreEngine:
    """
    Scores the technical validity of a setup.

    Usage:
        engine = SetupScoreEngine(settings)
        result = engine.score(
            ticker="ABCD",
            setup_result=best_setup,
            indicators=indicators,
            context=context,
            risk_reward_result=rr_result,
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._entry    = settings.get("entry_rules", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        ticker:             str,
        setup_result:       Optional[SetupResult],
        indicators:         IndicatorSnapshot,
        context:            dict,
        risk_reward_result: Optional[object] = None,
    ) -> SetupScoreResult:
        """
        Produce the setup_score for a candidate.

        Args:
            ticker:             Ticker symbol.
            setup_result:       Best confirmed SetupResult from setups.best_setup().
            indicators:         IndicatorSnapshot from indicator_engine.
            context:            Analysis context dict.
            risk_reward_result: Optional RiskRewardResult.

        Returns:
            SetupScoreResult.
        """
        result = SetupScoreResult(ticker=ticker)

        if setup_result is None:
            result.warnings.append("No setup result provided")
            result.setup_score = 0.0
            result.confidence_label = label_score(0.0)
            result.weak_setup = True
            return result

        result.setup_name      = setup_result.setup_name
        result.setup_confirmed = bool(setup_result.confirmed)

        if not setup_result.confirmed:
            result.warnings.append("Setup is not confirmed")
            result.warnings.extend(setup_result.reasons)
            result.setup_score = 0.0
            result.confidence_label = label_score(0.0)
            result.weak_setup = True
            return result

        breakdown: dict[str, float] = {}

        # ── 1. Setup confirmation (20 pts) ────────────────────────────────────
        setup_pts = self._score_setup_confirmation(setup_result)
        breakdown["setup_confirmation"] = round(setup_pts, 2)

        # ── 2. VWAP / key level behavior (15 pts) ─────────────────────────────
        level_pts, level_reasons, level_warnings = self._score_vwap_key_levels(
            indicators, context
        )
        breakdown["vwap_key_levels"] = round(level_pts, 2)
        result.reasons.extend(level_reasons)
        result.warnings.extend(level_warnings)

        # ── 3. Volume / RVOL (15 pts) ─────────────────────────────────────────
        volume_pts, volume_reasons, volume_warnings = self._score_volume(
            indicators
        )
        breakdown["volume"] = round(volume_pts, 2)
        result.reasons.extend(volume_reasons)
        result.warnings.extend(volume_warnings)

        # ── 4. Market structure (12 pts) ──────────────────────────────────────
        structure_pts, structure_reasons, structure_warnings = self._score_structure(
            context
        )
        breakdown["market_structure"] = round(structure_pts, 2)
        result.reasons.extend(structure_reasons)
        result.warnings.extend(structure_warnings)

        # ── 5. Opening range (10 pts) ─────────────────────────────────────────
        or_pts, or_reasons, or_warnings = self._score_opening_range(context)
        breakdown["opening_range"] = round(or_pts, 2)
        result.reasons.extend(or_reasons)
        result.warnings.extend(or_warnings)

        # ── 6. Liquidity sweep / reclaim (10 pts) ─────────────────────────────
        sweep_pts, sweep_reasons, sweep_warnings = self._score_sweep(context)
        breakdown["liquidity_sweep"] = round(sweep_pts, 2)
        result.reasons.extend(sweep_reasons)
        result.warnings.extend(sweep_warnings)

        # ── 7. Fibonacci confirmation (8 pts) ─────────────────────────────────
        fib_pts, fib_reasons, fib_warnings = self._score_fibonacci(context)
        breakdown["fibonacci"] = round(fib_pts, 2)
        result.reasons.extend(fib_reasons)
        result.warnings.extend(fib_warnings)

        # ── 8. Risk / reward (10 pts) ─────────────────────────────────────────
        rr_pts, rr_reasons, rr_warnings = self._score_risk_reward(
            risk_reward_result
        )
        breakdown["risk_reward"] = round(rr_pts, 2)
        result.reasons.extend(rr_reasons)
        result.warnings.extend(rr_warnings)

        # ── Total ─────────────────────────────────────────────────────────────
        total = sum(breakdown.values())
        result.setup_score = round(max(0.0, min(total, 100.0)), 2)
        result.confidence_label = label_score(result.setup_score)
        result.score_breakdown = breakdown

        result.reasons.insert(
            0,
            f"{setup_result.setup_name} confirmed with detector score "
            f"{setup_result.score:.1f}",
        )
        result.reasons.extend(setup_result.reasons)
        result.warnings.extend(setup_result.warnings)

        result.strong_setup = result.setup_score >= 80
        result.weak_setup   = result.setup_score < 70

        log.debug(
            "[setup_score] %s setup=%s score=%.1f label=%s",
            ticker, result.setup_name, result.setup_score, result.confidence_label,
        )
        return result

    # ── Component scoring ─────────────────────────────────────────────────────

    def _score_setup_confirmation(self, setup_result: SetupResult) -> float:
        """
        Setup confirmation component, max 20 pts.
        Uses the setup detector's own 0–100 score.
        """
        if not setup_result or not setup_result.confirmed:
            return 0.0

        detector_score = max(0.0, min(float(setup_result.score or 0.0), 100.0))
        return detector_score / 100.0 * 20.0

    def _score_vwap_key_levels(
        self,
        indicators: IndicatorSnapshot,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score VWAP and key level behavior, max 15 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        current_price = float(context.get("current_price", 0) or 0)
        vwap = context.get("vwap") or indicators.vwap

        if vwap and current_price > vwap:
            score += 6
            reasons.append("Price is above VWAP")
        elif vwap:
            warnings.append("Price is below VWAP")

        if indicators.vwap_extended:
            warnings.append("Price is extended from VWAP")
            score -= 2
        elif vwap:
            score += 2

        key_levels = context.get("key_levels")
        if key_levels:
            if getattr(key_levels, "holding_support", False):
                score += 4
                reasons.append("Price is holding support")
            if getattr(key_levels, "breaking_out", False):
                score += 3
                reasons.append("Price is breaking above a key level")
            if getattr(key_levels, "rejecting_resistance", False):
                warnings.append("Price is rejecting resistance")
                score -= 3
            if getattr(key_levels, "price_near_support", False):
                score += 2
                reasons.append("Price is near support")
            if getattr(key_levels, "price_near_resistance", False):
                warnings.append("Price is near resistance")

        return max(0.0, min(score, 15.0)), reasons, warnings

    def _score_volume(
        self,
        indicators: IndicatorSnapshot,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score volume and RVOL quality, max 15 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        rvol = float(indicators.relative_volume or 0.0)

        if rvol >= 5.0:
            score += 10
            reasons.append(f"Very strong RVOL: {rvol:.1f}x")
        elif rvol >= 3.0:
            score += 8
            reasons.append(f"Strong RVOL: {rvol:.1f}x")
        elif rvol >= 2.0:
            score += 5
            reasons.append(f"Moderate RVOL: {rvol:.1f}x")
        else:
            warnings.append(f"Low RVOL: {rvol:.1f}x")

        if indicators.volume_trend == "increasing":
            score += 5
            reasons.append("Volume trend is increasing")
        elif indicators.volume_trend == "flat":
            score += 2
        elif indicators.volume_trend == "decreasing":
            warnings.append("Volume trend is decreasing")

        return max(0.0, min(score, 15.0)), reasons, warnings

    def _score_structure(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score market structure, max 12 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        structure = context.get("structure")
        if not structure:
            warnings.append("Structure result unavailable")
            return score, reasons, warnings

        structure_name = getattr(structure, "structure", "")
        trend_direction = getattr(structure, "trend_direction", "")
        structure_score = float(getattr(structure, "structure_score", 0.0) or 0.0)

        if trend_direction == "bullish":
            score += 5
            reasons.append(f"Structure is bullish: {structure_name}")
        elif trend_direction == "bearish":
            warnings.append(f"Structure is bearish: {structure_name}")
            score -= 2
        elif structure_name in ("sideways", "forming", "mixed"):
            score += 2
            reasons.append(f"Structure is {structure_name}")

        if structure_score > 0:
            score += min(structure_score / 100.0 * 7.0, 7.0)

        if getattr(structure, "structure_broken", False):
            direction = getattr(structure, "break_direction", "")
            if direction == "bearish_break":
                warnings.append("Bearish structure break detected")
                score -= 4
            elif direction == "bullish_break":
                reasons.append("Bullish structure break detected")
                score += 2

        return max(0.0, min(score, 12.0)), reasons, warnings

    def _score_opening_range(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score opening range status, max 10 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        or_result = context.get("or_result")
        if not or_result:
            warnings.append("Opening range result unavailable")
            return score, reasons, warnings

        state = getattr(or_result, "state", "unknown")

        if state == "breakout":
            score += 10
            reasons.append("Opening range breakout confirmed")
        elif state == "above":
            score += 8
            reasons.append("Price is above opening range")
        elif state == "inside":
            score += 3
            reasons.append("Price is inside opening range")
        elif state in ("failed_breakout", "breakdown", "below"):
            warnings.append(f"Opening range state is bearish: {state}")
            score -= 3

        if getattr(or_result, "breakout_confirmed", False):
            score += 2
            reasons.append("OR breakout confirmation present")

        if getattr(or_result, "failed_breakout", False):
            warnings.append("Failed opening range breakout detected")
            score -= 5

        return max(0.0, min(score, 10.0)), reasons, warnings

    def _score_sweep(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score liquidity sweep / reclaim context, max 10 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        sweep = context.get("sweep_result")
        if not sweep:
            return score, reasons, warnings

        if getattr(sweep, "any_confirmed", False):
            score += 4
            reasons.append("Confirmed liquidity sweep present")

        best = getattr(sweep, "best_sweep", None)
        if best:
            quality = float(getattr(best, "quality_score", 0.0) or 0.0)
            score += min(quality / 100.0 * 4.0, 4.0)

            if getattr(best, "higher_low", False):
                score += 2
                reasons.append("Higher low after sweep reclaim")

            if quality >= 80:
                reasons.append(f"High quality sweep: {quality:.1f}")
            elif quality > 0 and quality < 60:
                warnings.append(f"Low quality sweep: {quality:.1f}")

        return max(0.0, min(score, 10.0)), reasons, warnings

    def _score_fibonacci(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score Fibonacci confirmation, max 8 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        fib = context.get("fib_result")
        if not fib:
            return score, reasons, warnings

        if getattr(fib, "block_trade", False):
            warnings.append("Fibonacci engine is blocking trade")
            return 0.0, reasons, warnings

        if getattr(fib, "fib_trend_valid", False):
            score += 2
            reasons.append("Fibonacci trend is valid")

        if getattr(fib, "at_preferred_level", False):
            score += 2
            reasons.append("Price is at preferred Fibonacci level")

        if getattr(fib, "entry_confirmed_by_fib", False):
            score += 3
            reasons.append("Fibonacci confirms entry")

        if getattr(fib, "target_extensions", None):
            score += 1
            reasons.append("Fibonacci extension targets available")

        return max(0.0, min(score, 8.0)), reasons, warnings

    def _score_risk_reward(
        self,
        rr_result: Optional[object],
    ) -> tuple[float, list[str], list[str]]:
        """
        Score risk/reward context, max 10 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        if not rr_result:
            warnings.append("Risk/reward result unavailable")
            return score, reasons, warnings

        if getattr(rr_result, "hard_block", False):
            warnings.append("Risk/reward hard block present")
            return 0.0, reasons, warnings

        rr_score = float(getattr(rr_result, "risk_reward_score", 0.0) or 0.0)
        score = min(rr_score / 100.0 * 10.0, 10.0)

        rr = float(getattr(rr_result, "reward_to_risk", 0.0) or 0.0)
        if getattr(rr_result, "meets_preferred", False):
            reasons.append(f"Risk/reward is preferred: {rr:.2f}:1")
        elif getattr(rr_result, "meets_minimum", False):
            reasons.append(f"Risk/reward meets minimum: {rr:.2f}:1")
        else:
            warnings.append(f"Risk/reward below minimum: {rr:.2f}:1")

        return max(0.0, min(score, 10.0)), reasons, warnings
