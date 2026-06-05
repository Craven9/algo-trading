"""
src/scoring/probability_engine.py — Trade probability scoring engine
Produces the probability_score (0–100) that answers:

    "What is the chance this trade works right now?"

This score focuses on current trade conditions, not just whether a setup
exists.  It combines price action, volume, VWAP, opening range, market
structure, liquidity sweep, Fibonacci confirmation, risk/reward, move
potential, execution quality, historical edge, and session strength.

Weights from the bot overview:
  Price action confirmation:           15 pts
  Volume confirmation:                 12 pts
  VWAP status:                         10 pts
  Opening range status:                 8 pts
  Market structure:                    10 pts
  Liquidity sweep:                      8 pts
  Fibonacci:                            7 pts
  Risk/reward quality:                 10 pts
  Move potential:                       8 pts
  Spread/liquidity/execution quality:   5 pts
  Historical edge:                      5 pts
  Market/session strength:              2 pts

Minimum probability: 75
Preferred probability: 80+

Design rules:
  - This file does not approve trades
  - This file does not place orders
  - It only scores probability and explains the reasoning
  - Hard rejection rules still belong in trade_quality_gate.py
  - The final buy/no-buy decision still happens in trade_quality_gate.py
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
class ProbabilityResult:
    """
    Probability assessment for a trade candidate.
    Consumed by trade_quality_gate.py and the frontend dashboard.
    """
    ticker:              str
    probability_score:   float = 0.0
    confidence_label:    str   = ConfidenceLabel.REJECT.value

    score_breakdown:     dict  = field(default_factory=dict)

    minimum_required:    float = 75.0
    preferred_required:  float = 80.0
    meets_minimum:       bool  = False
    meets_preferred:     bool  = False

    reasons:             list[str] = field(default_factory=list)
    warnings:            list[str] = field(default_factory=list)

    scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":             self.ticker,
            "probability_score":  round(self.probability_score, 2),
            "confidence_label":   self.confidence_label,
            "score_breakdown":    self.score_breakdown,
            "minimum_required":   self.minimum_required,
            "preferred_required": self.preferred_required,
            "meets_minimum":      self.meets_minimum,
            "meets_preferred":    self.meets_preferred,
            "reasons":            self.reasons,
            "warnings":           self.warnings,
            "scored_at":          self.scored_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class ProbabilityEngine:
    """
    Scores the probability of a trade working right now.

    Usage:
        engine = ProbabilityEngine(settings)
        result = engine.score(
            ticker="ABCD",
            setup_result=best_setup,
            indicators=indicators,
            context=context,
            setup_score_result=setup_score,
            risk_reward_result=rr_result,
            move_potential_result=move_result,
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._entry    = settings.get("entry_rules", {})
        self._minimum  = float(self._entry.get("minimum_probability_score", 75))
        self._preferred = float(self._entry.get("preferred_probability_score", 80))

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        ticker:                 str,
        setup_result:           Optional[SetupResult],
        indicators:             IndicatorSnapshot,
        context:                dict,
        setup_score_result:     Optional[object] = None,
        risk_reward_result:     Optional[object] = None,
        move_potential_result:  Optional[object] = None,
        execution_quality_result: Optional[object] = None,
        historical_edge_result: Optional[object] = None,
    ) -> ProbabilityResult:
        """
        Produce the probability_score for a trade candidate.

        Args:
            ticker:                   Ticker symbol.
            setup_result:             Best confirmed setup result.
            indicators:               IndicatorSnapshot.
            context:                  Analysis context dict.
            setup_score_result:       Optional SetupScoreResult.
            risk_reward_result:       Optional RiskRewardResult.
            move_potential_result:    Optional MovePotentialResult.
            execution_quality_result: Optional execution quality result.
            historical_edge_result:   Optional historical edge result.

        Returns:
            ProbabilityResult.
        """
        result = ProbabilityResult(
            ticker             = ticker,
            minimum_required   = self._minimum,
            preferred_required = self._preferred,
        )

        breakdown: dict[str, float] = {}

        # ── 1. Price action confirmation (15 pts) ─────────────────────────────
        price_pts, price_reasons, price_warnings = self._score_price_action(
            setup_result, setup_score_result, indicators, context
        )
        breakdown["price_action"] = round(price_pts, 2)
        result.reasons.extend(price_reasons)
        result.warnings.extend(price_warnings)

        # ── 2. Volume confirmation (12 pts) ───────────────────────────────────
        volume_pts, volume_reasons, volume_warnings = self._score_volume(
            indicators
        )
        breakdown["volume"] = round(volume_pts, 2)
        result.reasons.extend(volume_reasons)
        result.warnings.extend(volume_warnings)

        # ── 3. VWAP status (10 pts) ───────────────────────────────────────────
        vwap_pts, vwap_reasons, vwap_warnings = self._score_vwap(
            indicators, context
        )
        breakdown["vwap"] = round(vwap_pts, 2)
        result.reasons.extend(vwap_reasons)
        result.warnings.extend(vwap_warnings)

        # ── 4. Opening range (8 pts) ──────────────────────────────────────────
        or_pts, or_reasons, or_warnings = self._score_opening_range(context)
        breakdown["opening_range"] = round(or_pts, 2)
        result.reasons.extend(or_reasons)
        result.warnings.extend(or_warnings)

        # ── 5. Market structure (10 pts) ──────────────────────────────────────
        structure_pts, structure_reasons, structure_warnings = self._score_structure(
            context
        )
        breakdown["market_structure"] = round(structure_pts, 2)
        result.reasons.extend(structure_reasons)
        result.warnings.extend(structure_warnings)

        # ── 6. Liquidity sweep (8 pts) ────────────────────────────────────────
        sweep_pts, sweep_reasons, sweep_warnings = self._score_sweep(context)
        breakdown["liquidity_sweep"] = round(sweep_pts, 2)
        result.reasons.extend(sweep_reasons)
        result.warnings.extend(sweep_warnings)

        # ── 7. Fibonacci (7 pts) ──────────────────────────────────────────────
        fib_pts, fib_reasons, fib_warnings = self._score_fibonacci(context)
        breakdown["fibonacci"] = round(fib_pts, 2)
        result.reasons.extend(fib_reasons)
        result.warnings.extend(fib_warnings)

        # ── 8. Risk/reward quality (10 pts) ───────────────────────────────────
        rr_pts, rr_reasons, rr_warnings = self._score_risk_reward(
            risk_reward_result
        )
        breakdown["risk_reward"] = round(rr_pts, 2)
        result.reasons.extend(rr_reasons)
        result.warnings.extend(rr_warnings)

        # ── 9. Move potential (8 pts) ─────────────────────────────────────────
        move_pts, move_reasons, move_warnings = self._score_move_potential(
            move_potential_result
        )
        breakdown["move_potential"] = round(move_pts, 2)
        result.reasons.extend(move_reasons)
        result.warnings.extend(move_warnings)

        # ── 10. Spread/liquidity/execution quality (5 pts) ────────────────────
        exec_pts, exec_reasons, exec_warnings = self._score_execution_quality(
            execution_quality_result, context
        )
        breakdown["execution_quality"] = round(exec_pts, 2)
        result.reasons.extend(exec_reasons)
        result.warnings.extend(exec_warnings)

        # ── 11. Historical edge (5 pts) ───────────────────────────────────────
        hist_pts, hist_reasons, hist_warnings = self._score_historical_edge(
            historical_edge_result
        )
        breakdown["historical_edge"] = round(hist_pts, 2)
        result.reasons.extend(hist_reasons)
        result.warnings.extend(hist_warnings)

        # ── 12. Market/session strength (2 pts) ───────────────────────────────
        session_pts, session_reasons, session_warnings = self._score_session_strength(
            context
        )
        breakdown["session_strength"] = round(session_pts, 2)
        result.reasons.extend(session_reasons)
        result.warnings.extend(session_warnings)

        # ── Total ─────────────────────────────────────────────────────────────
        total = sum(breakdown.values())
        result.probability_score = round(max(0.0, min(total, 100.0)), 2)
        result.confidence_label  = label_score(result.probability_score)
        result.score_breakdown   = breakdown
        result.meets_minimum     = result.probability_score >= self._minimum
        result.meets_preferred   = result.probability_score >= self._preferred

        if result.meets_preferred:
            result.reasons.insert(
                0,
                f"Probability score {result.probability_score:.1f} meets preferred threshold",
            )
        elif result.meets_minimum:
            result.reasons.insert(
                0,
                f"Probability score {result.probability_score:.1f} meets minimum threshold",
            )
        else:
            result.warnings.insert(
                0,
                f"Probability score {result.probability_score:.1f} is below minimum {self._minimum:.1f}",
            )

        log.debug(
            "[probability] %s score=%.1f label=%s min=%s",
            ticker, result.probability_score, result.confidence_label,
            result.meets_minimum,
        )
        return result

    # ── Component scoring ─────────────────────────────────────────────────────

    def _score_price_action(
        self,
        setup_result: Optional[SetupResult],
        setup_score_result: Optional[object],
        indicators: IndicatorSnapshot,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score price action confirmation, max 15 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        if setup_result and setup_result.confirmed:
            setup_score = float(setup_result.score or 0.0)
            score += min(setup_score / 100.0 * 8.0, 8.0)
            reasons.append(f"Confirmed setup: {setup_result.setup_name}")
        else:
            warnings.append("No confirmed setup for price action")
            return score, reasons, warnings

        if setup_score_result:
            ss = float(getattr(setup_score_result, "setup_score", 0.0) or 0.0)
            score += min(ss / 100.0 * 4.0, 4.0)

        if indicators.candle_strength >= 0.5:
            score += 2
            reasons.append("Latest candle shows bullish strength")
        elif indicators.candle_strength < 0:
            warnings.append("Latest candle is bearish")
            score -= 1

        key_levels = context.get("key_levels")
        if key_levels and getattr(key_levels, "breaking_out", False):
            score += 1
            reasons.append("Price action is breaking out through a key level")

        return max(0.0, min(score, 15.0)), reasons, warnings

    def _score_volume(
        self,
        indicators: IndicatorSnapshot,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score volume confirmation, max 12 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        rvol = float(indicators.relative_volume or 0.0)

        if rvol >= 5.0:
            score += 8
            reasons.append(f"Very strong RVOL: {rvol:.1f}x")
        elif rvol >= 3.0:
            score += 6
            reasons.append(f"Strong RVOL: {rvol:.1f}x")
        elif rvol >= 2.0:
            score += 4
            reasons.append(f"Moderate RVOL: {rvol:.1f}x")
        else:
            warnings.append(f"RVOL is weak: {rvol:.1f}x")

        if indicators.volume_trend == "increasing":
            score += 4
            reasons.append("Volume trend is increasing")
        elif indicators.volume_trend == "flat":
            score += 2
        else:
            warnings.append("Volume trend is decreasing")

        return max(0.0, min(score, 12.0)), reasons, warnings

    def _score_vwap(
        self,
        indicators: IndicatorSnapshot,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score VWAP status, max 10 pts.
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

        dist = abs(float(indicators.vwap_distance_pct or 0.0))
        if vwap and not indicators.vwap_extended and dist <= 5.0:
            score += 4
            reasons.append("Price is not badly extended from VWAP")
        elif indicators.vwap_extended:
            warnings.append("Price is extended from VWAP")
            score -= 2

        return max(0.0, min(score, 10.0)), reasons, warnings

    def _score_opening_range(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score opening range status, max 8 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        or_result = context.get("or_result")
        if not or_result:
            return score, reasons, warnings

        state = getattr(or_result, "state", "unknown")

        if state == "breakout":
            score += 8
            reasons.append("Opening range breakout supports probability")
        elif state == "above":
            score += 6
            reasons.append("Price is above opening range")
        elif state == "inside":
            score += 2
            reasons.append("Price is inside opening range")
        elif state in ("failed_breakout", "breakdown", "below"):
            warnings.append(f"Opening range status is bearish: {state}")
            score -= 2

        return max(0.0, min(score, 8.0)), reasons, warnings

    def _score_structure(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score market structure, max 10 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        structure = context.get("structure")
        if not structure:
            return score, reasons, warnings

        trend = getattr(structure, "trend_direction", "")
        structure_score = float(getattr(structure, "structure_score", 0.0) or 0.0)
        phase = getattr(structure, "momentum_phase", "")

        if trend == "bullish":
            score += 4
            reasons.append("Market structure is bullish")
        elif trend == "bearish":
            warnings.append("Market structure is bearish")
            score -= 2
        else:
            score += 1

        score += min(structure_score / 100.0 * 4.0, 4.0)

        if phase == "confirmed":
            score += 2
            reasons.append("Momentum phase is confirmed")
        elif phase == "early":
            score += 1
            reasons.append("Momentum phase is early")
        elif phase in ("extended", "exhausted"):
            warnings.append(f"Momentum phase is {phase}")
            score -= 1

        return max(0.0, min(score, 10.0)), reasons, warnings

    def _score_sweep(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score liquidity sweep context, max 8 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        sweep = context.get("sweep_result")
        if not sweep:
            return score, reasons, warnings

        if getattr(sweep, "any_confirmed", False):
            score += 3
            reasons.append("Confirmed liquidity sweep improves probability")

        best = getattr(sweep, "best_sweep", None)
        if best:
            quality = float(getattr(best, "quality_score", 0.0) or 0.0)
            score += min(quality / 100.0 * 3.0, 3.0)

            if getattr(best, "higher_low", False):
                score += 2
                reasons.append("Higher low formed after sweep")

        return max(0.0, min(score, 8.0)), reasons, warnings

    def _score_fibonacci(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score Fibonacci confirmation, max 7 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        fib = context.get("fib_result")
        if not fib:
            return score, reasons, warnings

        if getattr(fib, "block_trade", False):
            warnings.append("Fibonacci result is blocking trade")
            return 0.0, reasons, warnings

        if getattr(fib, "fib_trend_valid", False):
            score += 2
            reasons.append("Fibonacci trend is valid")

        if getattr(fib, "entry_confirmed_by_fib", False):
            score += 3
            reasons.append("Fibonacci confirms entry")

        if getattr(fib, "target_extensions", None):
            score += 2
            reasons.append("Fibonacci extension targets support upside")

        return max(0.0, min(score, 7.0)), reasons, warnings

    def _score_risk_reward(
        self,
        rr_result: Optional[object],
    ) -> tuple[float, list[str], list[str]]:
        """
        Score risk/reward quality, max 10 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        if not rr_result:
            return score, reasons, warnings

        if getattr(rr_result, "hard_block", False):
            warnings.append("Risk/reward has hard block")
            return 0.0, reasons, warnings

        rr_score = float(getattr(rr_result, "risk_reward_score", 0.0) or 0.0)
        score = min(rr_score / 100.0 * 10.0, 10.0)

        rr = float(getattr(rr_result, "reward_to_risk", 0.0) or 0.0)
        if getattr(rr_result, "meets_preferred", False):
            reasons.append(f"Preferred risk/reward: {rr:.2f}:1")
        elif getattr(rr_result, "meets_minimum", False):
            reasons.append(f"Minimum risk/reward met: {rr:.2f}:1")
        else:
            warnings.append(f"Risk/reward below minimum: {rr:.2f}:1")

        return max(0.0, min(score, 10.0)), reasons, warnings

    def _score_move_potential(
        self,
        move_result: Optional[object],
    ) -> tuple[float, list[str], list[str]]:
        """
        Score move potential, max 8 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        if not move_result:
            return score, reasons, warnings

        move_score = float(getattr(move_result, "move_potential_score", 0.0) or 0.0)
        score = min(move_score / 100.0 * 8.0, 8.0)

        label = getattr(move_result, "score_label", "weak")
        if label in ("strong", "moderate"):
            reasons.append(f"Move potential is {label}")
        elif label in ("limited", "weak"):
            warnings.append(f"Move potential is {label}")

        return max(0.0, min(score, 8.0)), reasons, warnings

    def _score_execution_quality(
        self,
        execution_result: Optional[object],
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score execution quality, max 5 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        if execution_result:
            if getattr(execution_result, "hard_block", False):
                warnings.append("Execution quality has hard block")
                return 0.0, reasons, warnings

            exec_score = float(getattr(execution_result, "execution_quality_score", 0.0) or 0.0)
            score = min(exec_score / 100.0 * 5.0, 5.0)
            reasons.append("Execution quality result included")
            return max(0.0, min(score, 5.0)), reasons, warnings

        candidate = context.get("candidate")
        spread = float(getattr(candidate, "spread_percent", 0.0) or 0.0) if candidate else 0.0

        if spread <= 0:
            score += 2
        elif spread <= 1.0:
            score += 5
            reasons.append("Spread is tight")
        elif spread <= 2.0:
            score += 3
            reasons.append("Spread is acceptable")
        else:
            warnings.append(f"Spread is elevated: {spread:.2f}%")
            score += 1

        return max(0.0, min(score, 5.0)), reasons, warnings

    def _score_historical_edge(
        self,
        historical_result: Optional[object],
    ) -> tuple[float, list[str], list[str]]:
        """
        Score historical edge, max 5 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        if not historical_result:
            score += 2.5
            reasons.append("No historical edge yet — neutral score")
            return score, reasons, warnings

        edge_score = float(getattr(historical_result, "historical_edge_score", 50.0) or 50.0)
        score = min(edge_score / 100.0 * 5.0, 5.0)

        if edge_score >= 70:
            reasons.append("Historical edge is positive")
        elif edge_score < 40:
            warnings.append("Historical edge is weak")

        return max(0.0, min(score, 5.0)), reasons, warnings

    def _score_session_strength(
        self,
        context: dict,
    ) -> tuple[float, list[str], list[str]]:
        """
        Score market/session strength, max 2 pts.
        """
        score = 0.0
        reasons: list[str] = []
        warnings: list[str] = []

        session_context = context.get("session_context")
        if not session_context:
            score += 1
            return score, reasons, warnings

        phase = getattr(session_context, "session_phase", "")
        high_risk = bool(getattr(session_context, "is_high_risk_window", False))

        if high_risk:
            warnings.append("Session is in a high-risk window")
            return 0.0, reasons, warnings

        if phase in ("active", "power_hour", "opening"):
            score += 2
            reasons.append(f"Session phase supports momentum: {phase}")
        elif phase in ("afternoon", "pre_market"):
            score += 1
        elif phase in ("midday", "after_hours"):
            warnings.append(f"Session phase is weaker: {phase}")

        return max(0.0, min(score, 2.0)), reasons, warnings
