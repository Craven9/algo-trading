"""
src/analysis/session_structure_analyzer.py — Intraday market structure analysis
Analyzes the current session's price structure to determine trend quality,
momentum direction, and structural health of a setup.

Responsibilities:
  - Detect higher highs / higher lows (uptrend structure)
  - Detect lower highs / lower lows (downtrend structure)
  - Classify overall trend structure as bullish, bearish, or mixed
  - Measure trend consistency (how clean is the structure)
  - Detect structure breaks (key level violations)
  - Track momentum phase (early, confirmed, extended, exhausted)
  - Provide structure quality score for the probability engine

Structure classification:
  "strong_uptrend"    — consecutive HH + HL pattern
  "weak_uptrend"      — mostly HH/HL but with one violation
  "sideways"          — no clear direction
  "weak_downtrend"    — mostly LH/LL but with one violation
  "strong_downtrend"  — consecutive LH + LL pattern
  "mixed"             — conflicting signals

Momentum phase:
  "early"      — first confirmed HH/HL (1-2 sequences)
  "confirmed"  — 2-3 sequences, most reliable entry zone
  "extended"   — 4+ sequences, consider reduced size
  "exhausted"  — structure weakening, exit risk rising
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from indicator_calculator import (
    detect_higher_lows,
    detect_lower_highs,
    find_swing_highs,
    find_swing_lows,
)

log = logging.getLogger(__name__)

# Minimum swing count to determine trend
_MIN_SWINGS = 2
# Lookback for swing detection
_SWING_LOOKBACK = 3


# ── Structure result ──────────────────────────────────────────────────────────

@dataclass
class SessionStructureResult:
    """
    Full session structure analysis for a ticker.
    Consumed by setup detectors and the probability engine.
    """
    ticker:          str
    current_price:   float

    # Swing data
    swing_highs:     list[float]  = field(default_factory=list)
    swing_lows:      list[float]  = field(default_factory=list)

    # Structure flags
    higher_highs:    bool   = False
    higher_lows:     bool   = False
    lower_highs:     bool   = False
    lower_lows:      bool   = False

    # Classification
    structure:       str    = "unknown"   # see module docstring
    momentum_phase:  str    = "unknown"   # early|confirmed|extended|exhausted
    trend_direction: str    = "neutral"   # bullish|bearish|neutral

    # Quality metrics
    structure_score:    float = 0.0   # 0–100, feeds probability engine
    consistency_pct:    float = 0.0   # % of swings that follow the trend
    hh_hl_count:        int   = 0     # consecutive HH+HL sequences
    lh_ll_count:        int   = 0     # consecutive LH+LL sequences

    # Structure break flags
    structure_broken:   bool  = False
    break_direction:    str   = ""    # "bullish_break" | "bearish_break"

    reasons:     list[str] = field(default_factory=list)
    warnings:    list[str] = field(default_factory=list)

    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def is_bullish(self) -> bool:
        return self.trend_direction == "bullish"

    def is_bearish(self) -> bool:
        return self.trend_direction == "bearish"

    def is_strong(self) -> bool:
        return self.structure in ("strong_uptrend", "strong_downtrend")

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "current_price":    self.current_price,
            "swing_highs":      [round(p, 4) for p in self.swing_highs],
            "swing_lows":       [round(p, 4) for p in self.swing_lows],
            "higher_highs":     self.higher_highs,
            "higher_lows":      self.higher_lows,
            "lower_highs":      self.lower_highs,
            "lower_lows":       self.lower_lows,
            "structure":        self.structure,
            "momentum_phase":   self.momentum_phase,
            "trend_direction":  self.trend_direction,
            "structure_score":  round(self.structure_score, 2),
            "consistency_pct":  round(self.consistency_pct, 2),
            "hh_hl_count":      self.hh_hl_count,
            "lh_ll_count":      self.lh_ll_count,
            "structure_broken": self.structure_broken,
            "break_direction":  self.break_direction,
            "reasons":          self.reasons,
            "warnings":         self.warnings,
            "analyzed_at":      self.analyzed_at,
        }


# ── Analyzer ──────────────────────────────────────────────────────────────────

class SessionStructureAnalyzer:
    """
    Analyzes intraday price structure from bar data.

    Usage:
        analyzer = SessionStructureAnalyzer(settings)
        result   = analyzer.analyze("ABCD", bars, current_price)
    """

    def __init__(self, settings: dict):
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        ticker:        str,
        bars:          list[dict],
        current_price: float,
    ) -> SessionStructureResult:
        """
        Analyze price structure for a ticker.

        Args:
            ticker:        Ticker symbol.
            bars:          Regular-session bars, oldest→newest.
            current_price: Latest close price.

        Returns:
            SessionStructureResult with full structure analysis.
        """
        result = SessionStructureResult(
            ticker        = ticker,
            current_price = current_price,
        )

        if not bars or current_price <= 0 or len(bars) < _SWING_LOOKBACK * 2 + 1:
            result.structure  = "insufficient_data"
            result.reasons.append("Insufficient bars for structure analysis")
            return result

        # ── Detect swings ─────────────────────────────────────────────────────
        highs = find_swing_highs(bars, _SWING_LOOKBACK)
        lows  = find_swing_lows(bars,  _SWING_LOOKBACK)

        result.swing_highs = highs
        result.swing_lows  = lows

        if len(highs) < _MIN_SWINGS or len(lows) < _MIN_SWINGS:
            result.structure = "forming"
            result.reasons.append("Not enough swings yet — structure still forming")
            return result

        # ── Higher highs / lower highs ────────────────────────────────────────
        result.higher_highs = _is_ascending(highs[-3:])
        result.lower_highs  = _is_descending(highs[-3:])

        # ── Higher lows / lower lows ──────────────────────────────────────────
        result.higher_lows = _is_ascending(lows[-3:])
        result.lower_lows  = _is_descending(lows[-3:])

        # ── Count HH+HL sequences ─────────────────────────────────────────────
        result.hh_hl_count = self._count_hh_hl(highs, lows)
        result.lh_ll_count = self._count_lh_ll(highs, lows)

        # ── Classify structure ────────────────────────────────────────────────
        result.structure, result.trend_direction = self._classify(result)

        # ── Momentum phase ────────────────────────────────────────────────────
        result.momentum_phase = self._momentum_phase(result)

        # ── Consistency ───────────────────────────────────────────────────────
        result.consistency_pct = self._consistency(highs, lows, result.trend_direction)

        # ── Structure break detection ─────────────────────────────────────────
        result = self._detect_break(result, bars, current_price, lows, highs)

        # ── Score ─────────────────────────────────────────────────────────────
        result.structure_score = self._score(result)

        # ── Reasons ───────────────────────────────────────────────────────────
        result.reasons  = self._build_reasons(result)
        result.warnings = self._build_warnings(result)

        log.debug(
            "[structure] %s: %s phase=%s score=%.1f hh_hl=%d",
            ticker, result.structure, result.momentum_phase,
            result.structure_score, result.hh_hl_count,
        )
        return result

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(
        self, r: SessionStructureResult
    ) -> tuple[str, str]:
        """Classify structure and trend direction from swing flags."""
        hh = r.higher_highs
        hl = r.higher_lows
        lh = r.lower_highs
        ll = r.lower_lows

        if hh and hl:
            return "strong_uptrend", "bullish"
        if hh and not hl and not ll:
            return "weak_uptrend", "bullish"
        if hl and not hh:
            return "weak_uptrend", "bullish"
        if lh and ll:
            return "strong_downtrend", "bearish"
        if lh and not hh and not hl:
            return "weak_downtrend", "bearish"
        if ll and not lh:
            return "weak_downtrend", "bearish"
        if (hh or hl) and (lh or ll):
            return "mixed", "neutral"
        return "sideways", "neutral"

    # ── HH/HL counter ────────────────────────────────────────────────────────

    def _count_hh_hl(
        self, highs: list[float], lows: list[float]
    ) -> int:
        """Count consecutive HH+HL sequences (pairs of higher high AND higher low)."""
        count   = 0
        n       = min(len(highs), len(lows))
        for i in range(1, n):
            if highs[i] > highs[i - 1] and lows[i] > lows[i - 1]:
                count += 1
            else:
                count = 0   # reset on any violation
        return count

    def _count_lh_ll(
        self, highs: list[float], lows: list[float]
    ) -> int:
        """Count consecutive LH+LL sequences."""
        count = 0
        n     = min(len(highs), len(lows))
        for i in range(1, n):
            if highs[i] < highs[i - 1] and lows[i] < lows[i - 1]:
                count += 1
            else:
                count = 0
        return count

    # ── Momentum phase ────────────────────────────────────────────────────────

    def _momentum_phase(self, r: SessionStructureResult) -> str:
        """Classify momentum maturity for sizing/entry decisions."""
        count = r.hh_hl_count if r.trend_direction == "bullish" else r.lh_ll_count

        if count == 0:
            return "unknown"
        if count <= 1:
            return "early"
        if count <= 3:
            return "confirmed"
        if count <= 5:
            return "extended"
        return "exhausted"

    # ── Consistency ──────────────────────────────────────────────────────────

    def _consistency(
        self,
        highs:     list[float],
        lows:      list[float],
        direction: str,
    ) -> float:
        """
        Percentage of swing pairs that follow the dominant trend direction.
        Returns 0.0 when direction is neutral or data is insufficient.
        """
        n = min(len(highs), len(lows))
        if n < 2 or direction == "neutral":
            return 0.0

        conforming = 0
        for i in range(1, n):
            if direction == "bullish":
                if highs[i] > highs[i - 1] or lows[i] > lows[i - 1]:
                    conforming += 1
            else:
                if highs[i] < highs[i - 1] or lows[i] < lows[i - 1]:
                    conforming += 1

        return round(conforming / (n - 1) * 100, 2)

    # ── Structure break ───────────────────────────────────────────────────────

    def _detect_break(
        self,
        result:        SessionStructureResult,
        bars:          list[dict],
        current_price: float,
        lows:          list[float],
        highs:         list[float],
    ) -> SessionStructureResult:
        """
        Detect if the most recent price action has broken the prevailing structure.

        Bullish break: price closes above the most recent swing high in a downtrend.
        Bearish break: price closes below the most recent swing low in an uptrend.
        """
        if not lows or not highs:
            return result

        last_swing_low  = lows[-1]
        last_swing_high = highs[-1]

        if result.trend_direction == "bullish":
            # Price losing last swing low = bearish structure break
            if current_price < last_swing_low:
                result.structure_broken = True
                result.break_direction  = "bearish_break"

        elif result.trend_direction == "bearish":
            # Price reclaiming last swing high = bullish structure break
            if current_price > last_swing_high:
                result.structure_broken = True
                result.break_direction  = "bullish_break"

        return result

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, r: SessionStructureResult) -> float:
        """
        Score structure quality 0–100 for the probability engine.

        Strong uptrend:   base 80
        Weak uptrend:     base 60
        Sideways/mixed:   base 40
        Weak downtrend:   base 25
        Strong downtrend: base 10

        Bonuses:
          + Consistency pct / 5  (max +20)
          + HH/HL count * 3      (max +15)
          - Structure broken: -30
          - Exhausted phase: -15
        """
        base_scores = {
            "strong_uptrend":   80,
            "weak_uptrend":     60,
            "sideways":         40,
            "mixed":            35,
            "weak_downtrend":   25,
            "strong_downtrend": 10,
            "forming":          30,
            "insufficient_data":20,
            "unknown":          20,
        }
        score = float(base_scores.get(r.structure, 30))

        # Consistency bonus
        score += r.consistency_pct / 5

        # HH/HL sequence bonus
        seq_count = r.hh_hl_count if r.trend_direction == "bullish" else r.lh_ll_count
        score += min(seq_count * 3, 15)

        # Penalties
        if r.structure_broken:
            score -= 30
        if r.momentum_phase == "exhausted":
            score -= 15
        if r.momentum_phase == "extended":
            score -= 5

        return round(max(0.0, min(score, 100.0)), 2)

    # ── Reason / warning builders ─────────────────────────────────────────────

    def _build_reasons(self, r: SessionStructureResult) -> list[str]:
        reasons = []
        if r.higher_highs and r.higher_lows:
            reasons.append("Higher highs and higher lows — clean uptrend structure")
        elif r.higher_lows:
            reasons.append("Higher lows forming — buyer support building")
        elif r.higher_highs:
            reasons.append("Higher highs forming — price making progress")
        if r.lower_highs and r.lower_lows:
            reasons.append("Lower highs and lower lows — downtrend structure")
        if r.hh_hl_count >= 2:
            reasons.append(f"{r.hh_hl_count} consecutive HH+HL sequences")
        if r.momentum_phase == "confirmed":
            reasons.append("Momentum in confirmed phase — ideal entry window")
        if r.structure_broken and r.break_direction == "bullish_break":
            reasons.append("Bullish structure break — price reclaimed last swing high")
        return reasons

    def _build_warnings(self, r: SessionStructureResult) -> list[str]:
        warnings = []
        if r.structure_broken and r.break_direction == "bearish_break":
            warnings.append("Structure broken — price lost last swing low")
        if r.momentum_phase == "exhausted":
            warnings.append("Momentum exhausted — 6+ HH/HL sequences, use smaller size")
        if r.momentum_phase == "extended":
            warnings.append("Momentum extended — consider reduced position size")
        if r.structure in ("mixed", "sideways"):
            warnings.append("No clear trend structure — wait for confirmation")
        if r.consistency_pct < 50 and r.trend_direction != "neutral":
            warnings.append("Low trend consistency — choppy price action")
        return warnings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_ascending(prices: list[float]) -> bool:
    """True when each price in the list is higher than the previous."""
    if len(prices) < 2:
        return False
    return all(prices[i] > prices[i - 1] for i in range(1, len(prices)))


def _is_descending(prices: list[float]) -> bool:
    """True when each price in the list is lower than the previous."""
    if len(prices) < 2:
        return False
    return all(prices[i] < prices[i - 1] for i in range(1, len(prices)))
