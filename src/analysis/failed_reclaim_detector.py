"""
src/analysis/failed_reclaim_detector.py — Failed reclaim detection
Detects when price attempts to reclaim a key level but fails to hold above it.
A failed reclaim is one of the strongest rejection signals and triggers a
hard block on new long entries per the design spec.

Design rules (from spec):
  - "block_failed_reclaims": true in entry_rules
  - "block_if_under_0618_fib_after_failed_reclaim": true in entry_rules
  - Failed reclaim is a HARD REJECTION signal — no override

A failed reclaim occurs when:
  1. Price moves above a key level (VWAP, OR high, swing high, etc.)
  2. Price fails to HOLD above that level (closes back below within N bars)
  3. The reclaim attempt volume was not convincing

Failed reclaim types:
  "failed_vwap_reclaim"    — price crossed VWAP but closed back below
  "failed_or_breakout"     — price cleared OR high but closed back inside
  "failed_key_level"       — price cleared a swing high / resistance but failed
  "failed_pm_high_reclaim" — price crossed premarket high but fell back

Severity:
  "critical" — failed VWAP or OR high reclaim (highest rejection weight)
  "moderate" — failed key level or swing high
  "minor"    — brief pop above level, immediately rejected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Number of bars to look back for a failed reclaim pattern
_DEFAULT_LOOKBACK = 15
# A reclaim that holds for fewer than this many bars is considered failed
_MIN_HOLD_BARS = 2
# Maximum bars above level for it to still count as a "brief" failed reclaim
_MAX_BRIEF_BARS = 5


# ── Failed reclaim result ─────────────────────────────────────────────────────

@dataclass
class FailedReclaimResult:
    """
    Result of a failed reclaim check for a single level.
    """
    ticker:         str
    level_price:    float
    level_type:     str    = ""

    # Detection flags
    detected:       bool   = False   # True when a failed reclaim is confirmed
    severity:       str    = "none"  # "critical" | "moderate" | "minor" | "none"

    # Pattern details
    reclaim_bar_idx:int    = -1      # bar index where price crossed above level
    fail_bar_idx:   int    = -1      # bar index where price closed back below
    bars_above:     int    = 0       # how many bars price held above level
    peak_above:     float  = 0.0     # highest price reached above the level
    overshoot_pct:  float  = 0.0     # how far above the level it got

    # Volume analysis
    reclaim_volume:    float = 0.0
    fail_volume:       float = 0.0
    volume_confirming: bool  = False  # True when reclaim vol > fail vol

    reasons:  list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "level_price":       round(self.level_price, 4),
            "level_type":        self.level_type,
            "detected":          self.detected,
            "severity":          self.severity,
            "bars_above":        self.bars_above,
            "peak_above":        round(self.peak_above,   4),
            "overshoot_pct":     round(self.overshoot_pct,4),
            "volume_confirming": self.volume_confirming,
            "reasons":           self.reasons,
            "warnings":          self.warnings,
            "detected_at":       self.detected_at,
        }


# ── Aggregate result ──────────────────────────────────────────────────────────

@dataclass
class FailedReclaimSummary:
    """
    Aggregated failed reclaim analysis across all key levels for a ticker.
    The block_long_entry flag is what the trade quality gate reads.
    """
    ticker:             str
    current_price:      float
    failed_reclaims:    list[FailedReclaimResult] = field(default_factory=list)

    # Top-level signals
    any_detected:       bool  = False
    block_long_entry:   bool  = False   # hard block per spec
    worst_severity:     str   = "none"  # most severe failure detected

    # Specific flags
    vwap_failed:        bool  = False
    or_failed:          bool  = False
    key_level_failed:   bool  = False

    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def critical_failures(self) -> list[FailedReclaimResult]:
        return [r for r in self.failed_reclaims if r.severity == "critical"]

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "current_price":    self.current_price,
            "any_detected":     self.any_detected,
            "block_long_entry": self.block_long_entry,
            "worst_severity":   self.worst_severity,
            "vwap_failed":      self.vwap_failed,
            "or_failed":        self.or_failed,
            "key_level_failed": self.key_level_failed,
            "failed_reclaims":  [r.to_dict() for r in self.failed_reclaims],
            "analyzed_at":      self.analyzed_at,
        }


# ── Detector ──────────────────────────────────────────────────────────────────

class FailedReclaimDetector:
    """
    Detects failed reclaim patterns across key price levels.

    Usage:
        detector = FailedReclaimDetector(settings)
        summary  = detector.analyze(
            "ABCD", bars, current_price=3.20,
            key_levels={"vwap": 3.30, "or_high": 3.50}
        )
        if summary.block_long_entry:
            # reject trade
    """

    def __init__(self, settings: dict):
        self._settings      = settings
        self._entry_cfg     = settings.get("entry_rules", {})
        self._block_enabled = bool(self._entry_cfg.get("block_failed_reclaims", True))
        self._lookback      = _DEFAULT_LOOKBACK
        self._min_hold      = _MIN_HOLD_BARS

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        ticker:        str,
        bars:          list[dict],
        current_price: float,
        key_levels:    Optional[dict[str, float]] = None,
    ) -> FailedReclaimSummary:
        """
        Detect failed reclaims across all provided key levels.

        Args:
            ticker:        Ticker symbol.
            bars:          Regular-session bars, oldest→newest.
            current_price: Latest close price.
            key_levels:    Dict of level_type → price.
                           Supported: "vwap", "or_high", "or_low",
                           "pm_high", "swing_high", "key_level", etc.

        Returns:
            FailedReclaimSummary — check block_long_entry before placing trade.
        """
        summary = FailedReclaimSummary(
            ticker        = ticker,
            current_price = current_price,
        )

        if not bars or not key_levels or current_price <= 0:
            return summary

        recent = bars[-self._lookback:]
        results: list[FailedReclaimResult] = []

        for level_type, level_price in key_levels.items():
            if not level_price or level_price <= 0:
                continue
            result = self._detect(ticker, recent, level_price, level_type)
            results.append(result)

            if result.detected:
                if level_type == "vwap":
                    summary.vwap_failed = True
                elif "or" in level_type.lower():
                    summary.or_failed = True
                else:
                    summary.key_level_failed = True

        summary.failed_reclaims = results
        summary.any_detected    = any(r.detected for r in results)

        # Determine worst severity
        severity_order = {"critical": 3, "moderate": 2, "minor": 1, "none": 0}
        if summary.any_detected:
            worst = max(
                (r for r in results if r.detected),
                key=lambda r: severity_order.get(r.severity, 0),
            )
            summary.worst_severity = worst.severity
        else:
            summary.worst_severity = "none"

        # Block long entry when any critical or moderate failure is present
        if self._block_enabled and summary.any_detected:
            if summary.worst_severity in ("critical", "moderate"):
                summary.block_long_entry = True

        log.debug(
            "[failed_reclaim] %s: detected=%s worst=%s block=%s",
            ticker, summary.any_detected, summary.worst_severity,
            summary.block_long_entry,
        )
        return summary

    # ── Core detection ────────────────────────────────────────────────────────

    def _detect(
        self,
        ticker:      str,
        bars:        list[dict],
        level_price: float,
        level_type:  str,
    ) -> FailedReclaimResult:
        """
        Scan bars for a failed reclaim of `level_price`.

        A failed reclaim is detected when:
          1. At least one bar closes ABOVE the level (reclaim attempt)
          2. A subsequent bar closes BELOW the level (failure)
          3. The total bars held above was < _MIN_HOLD_BARS
             OR the failure bar comes within _MAX_BRIEF_BARS of the reclaim
        """
        result = FailedReclaimResult(
            ticker      = ticker,
            level_price = level_price,
            level_type  = level_type,
        )

        reclaim_idx   = -1
        reclaim_vol   = 0.0
        bars_above    = 0
        peak_above    = level_price

        for i, bar in enumerate(bars):
            c   = bar.get("c", 0)
            h   = bar.get("h", 0)
            vol = float(bar.get("v", 0))

            if reclaim_idx == -1:
                # Looking for first bar that closes above the level
                if c > level_price:
                    reclaim_idx = i
                    reclaim_vol = vol
                    bars_above  = 1
                    peak_above  = max(peak_above, h)
                continue

            # Already in reclaim mode — tracking
            if c > level_price:
                bars_above += 1
                peak_above  = max(peak_above, h)
                continue

            # Price just closed back below the level — check if it failed
            bars_since_reclaim = i - reclaim_idx

            failed = (
                bars_above < self._min_hold
                or bars_since_reclaim <= _MAX_BRIEF_BARS
            )

            if failed:
                fail_vol   = vol
                overshoot  = (peak_above - level_price) / level_price * 100

                result.detected        = True
                result.reclaim_bar_idx = reclaim_idx
                result.fail_bar_idx    = i
                result.bars_above      = bars_above
                result.peak_above      = round(peak_above, 4)
                result.overshoot_pct   = round(overshoot,  4)
                result.reclaim_volume  = reclaim_vol
                result.fail_volume     = fail_vol
                result.volume_confirming = reclaim_vol > fail_vol

                result.severity  = self._severity(level_type, bars_above, overshoot)
                result.reasons   = self._build_reasons(result)
                result.warnings  = self._build_warnings(result)

                log.debug(
                    "[failed_reclaim] %s %s: FAILED — held %d bars, overshoot=%.2f%%",
                    ticker, level_type, bars_above, overshoot,
                )
                return result

            # Held long enough → not a failed reclaim
            # Reset and keep scanning for another attempt
            reclaim_idx = -1
            bars_above  = 0
            peak_above  = level_price

        return result

    # ── Severity classification ───────────────────────────────────────────────

    @staticmethod
    def _severity(level_type: str, bars_above: int, overshoot_pct: float) -> str:
        """
        Classify the severity of a failed reclaim.

        Critical: VWAP or OR — most impactful levels
        Moderate: swing high, key level
        Minor:    very brief overshoot (< 0.5%) or very fast failure (0 bars)
        """
        # Minor: barely crossed the level
        if overshoot_pct < 0.5 or bars_above == 0:
            return "minor"

        # Critical: VWAP or OR failure
        if level_type in ("vwap", "or_high", "or_low"):
            return "critical"

        # Moderate: other levels
        return "moderate"

    # ── Reason builders ───────────────────────────────────────────────────────

    @staticmethod
    def _build_reasons(r: FailedReclaimResult) -> list[str]:
        reasons = [
            f"Price attempted to reclaim {r.level_type} ({r.level_price:.4f})",
            f"Held above for only {r.bars_above} bar(s) before closing back below",
        ]
        if r.overshoot_pct > 0:
            reasons.append(f"Overshot level by {r.overshoot_pct:.2f}%")
        if not r.volume_confirming:
            reasons.append("Reclaim volume was not convincing (lower than fail bar)")
        return reasons

    @staticmethod
    def _build_warnings(r: FailedReclaimResult) -> list[str]:
        warnings = []
        if r.severity == "critical":
            warnings.append(
                f"Critical failed reclaim of {r.level_type} — "
                f"long entry blocked until valid reclaim confirmed"
            )
        if r.bars_above == 0:
            warnings.append("Immediate rejection — no close above level achieved")
        return warnings
