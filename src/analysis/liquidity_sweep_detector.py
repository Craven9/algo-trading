"""
src/analysis/liquidity_sweep_detector.py — Liquidity sweep detection
Detects when price briefly pierces below a key support level (sweeping stops
and liquidity) then quickly recovers above it — a powerful reversal signal.

A liquidity sweep is one of the strongest confirmation patterns in the bot
because it indicates:
  1. Weak hands were stopped out below the level
  2. Smart money absorbed the selling
  3. Price reclaimed the level quickly with intent

Responsibilities:
  - Detect sweeps below key levels (premarket low, OR low, swing low, VWAP)
  - Detect sweeps above key levels (premarket high, OR high, swing high)
  - Confirm whether price reclaimed the swept level
  - Detect higher low formation after the reclaim
  - Score sweep quality based on reclaim speed, volume, and structure

Sweep quality factors:
  - How far price swept below the level (deeper = stronger)
  - How quickly it recovered (faster = stronger)
  - Volume on the sweep bar vs reclaim bar
  - Whether a higher low formed after reclaim
  - Whether VWAP was reclaimed after the sweep

Output:
  SweepResult with confirmed flag, level swept, reclaim status,
  higher low detection, and a quality score (0–100).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# How many bars to look back for a sweep
_DEFAULT_LOOKBACK = 20
# Maximum bars allowed between sweep and reclaim for it to count
_MAX_RECLAIM_BARS = 5
# Minimum sweep depth as % below the level
_MIN_SWEEP_DEPTH_PCT = 0.1
# Minimum volume ratio on reclaim bar vs sweep bar
_MIN_RECLAIM_VOL_RATIO = 0.5


# ── Sweep result ──────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """
    Result of a liquidity sweep detection for a single level.
    """
    # Identification
    ticker:          str
    level_price:     float
    level_type:      str    = ""     # "vwap" | "premarket_low" | "or_low" | "swing_low" | etc.

    # Sweep details
    confirmed:       bool   = False  # True when sweep + reclaim both detected
    sweep_detected:  bool   = False  # True when price pierced below the level
    reclaimed:       bool   = False  # True when price recovered above the level
    higher_low:      bool   = False  # True when a higher low formed after reclaim

    # Sweep bar info
    sweep_bar_idx:   int    = -1
    sweep_low:       float  = 0.0    # how far price went below the level
    sweep_depth_pct: float  = 0.0    # (level - sweep_low) / level * 100

    # Reclaim info
    reclaim_bar_idx: int    = -1
    bars_to_reclaim: int    = 0      # bars between sweep and reclaim
    reclaim_volume_ratio: float = 0.0  # reclaim bar vol / sweep bar vol

    # Quality
    quality_score:   float  = 0.0   # 0–100
    quality_label:   str    = "none"  # "high" | "medium" | "low" | "none"
    reasons:         list[str] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)

    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "ticker":               self.ticker,
            "level_price":          round(self.level_price, 4),
            "level_type":           self.level_type,
            "confirmed":            self.confirmed,
            "sweep_detected":       self.sweep_detected,
            "reclaimed":            self.reclaimed,
            "higher_low":           self.higher_low,
            "sweep_low":            round(self.sweep_low, 4),
            "sweep_depth_pct":      round(self.sweep_depth_pct, 4),
            "bars_to_reclaim":      self.bars_to_reclaim,
            "reclaim_volume_ratio": round(self.reclaim_volume_ratio, 4),
            "quality_score":        round(self.quality_score, 2),
            "quality_label":        self.quality_label,
            "reasons":              self.reasons,
            "warnings":             self.warnings,
            "detected_at":          self.detected_at,
        }


# ── Full sweep analysis ───────────────────────────────────────────────────────

@dataclass
class LiquiditySweepResult:
    """
    Aggregated sweep analysis across all key levels for a ticker.
    """
    ticker:          str
    current_price:   float
    sweeps:          list[SweepResult]  = field(default_factory=list)
    best_sweep:      Optional[SweepResult] = None
    any_confirmed:   bool               = False
    vwap_swept:      bool               = False
    or_low_swept:    bool               = False
    pm_low_swept:    bool               = False
    swing_low_swept: bool               = False
    analyzed_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def confirmed_sweeps(self) -> list[SweepResult]:
        return [s for s in self.sweeps if s.confirmed]

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "current_price":   self.current_price,
            "any_confirmed":   self.any_confirmed,
            "vwap_swept":      self.vwap_swept,
            "or_low_swept":    self.or_low_swept,
            "pm_low_swept":    self.pm_low_swept,
            "swing_low_swept": self.swing_low_swept,
            "best_sweep":      self.best_sweep.to_dict() if self.best_sweep else None,
            "sweeps":          [s.to_dict() for s in self.sweeps],
            "analyzed_at":     self.analyzed_at,
        }


# ── Detector ──────────────────────────────────────────────────────────────────

class LiquiditySweepDetector:
    """
    Detects liquidity sweeps below key support levels in bar data.

    Usage:
        detector = LiquiditySweepDetector(settings)
        result   = detector.analyze(
            "ABCD", bars, current_price,
            key_levels={"vwap": 3.10, "or_low": 2.95, "pm_low": 2.80}
        )
    """

    def __init__(self, settings: dict):
        self._settings    = settings
        self._lookback    = _DEFAULT_LOOKBACK
        self._max_reclaim = _MAX_RECLAIM_BARS

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        ticker:        str,
        bars:          list[dict],
        current_price: float,
        key_levels:    Optional[dict[str, float]] = None,
    ) -> LiquiditySweepResult:
        """
        Detect liquidity sweeps across all provided key levels.

        Args:
            ticker:        Ticker symbol.
            bars:          Regular-session bars, oldest→newest.
            current_price: Latest close price.
            key_levels:    Dict of level_type → price.
                           Supported keys: "vwap", "or_low", "or_high",
                           "pm_low", "pm_high", "swing_low", "swing_high",
                           plus any custom label.

        Returns:
            LiquiditySweepResult with all detected sweeps.
        """
        result = LiquiditySweepResult(
            ticker        = ticker,
            current_price = current_price,
        )

        if not bars or not key_levels or current_price <= 0:
            return result

        # Work with the most recent lookback bars
        recent = bars[-self._lookback:]

        sweeps: list[SweepResult] = []

        for level_type, level_price in key_levels.items():
            if not level_price or level_price <= 0:
                continue

            sweep = self._detect_sweep(ticker, recent, level_price, level_type)
            sweeps.append(sweep)

            # Flag specific level types
            if sweep.confirmed:
                if level_type == "vwap":
                    result.vwap_swept = True
                elif level_type == "or_low":
                    result.or_low_swept = True
                elif level_type == "pm_low":
                    result.pm_low_swept = True
                elif "swing_low" in level_type:
                    result.swing_low_swept = True

        result.sweeps      = sweeps
        result.any_confirmed = any(s.confirmed for s in sweeps)

        # Best sweep = confirmed sweep with highest quality score
        confirmed = [s for s in sweeps if s.confirmed]
        if confirmed:
            result.best_sweep = max(confirmed, key=lambda s: s.quality_score)

        log.debug(
            "[sweep] %s: %d levels checked, %d confirmed sweeps",
            ticker, len(sweeps), len(confirmed),
        )
        return result

    # ── Core sweep detection ──────────────────────────────────────────────────

    def _detect_sweep(
        self,
        ticker:      str,
        bars:        list[dict],
        level_price: float,
        level_type:  str,
    ) -> SweepResult:
        """
        Detect a liquidity sweep below `level_price` in the bar sequence.

        A sweep is confirmed when:
          1. A bar's low goes below the level (sweep)
          2. That same bar OR a subsequent bar (within max_reclaim_bars)
             closes back ABOVE the level (reclaim)
          3. The sweep depth meets the minimum threshold
        """
        result = SweepResult(
            ticker       = ticker,
            level_price  = level_price,
            level_type   = level_type,
        )

        sweep_idx    = -1
        sweep_low    = level_price
        sweep_vol    = 0

        for i, bar in enumerate(bars):
            bar_low   = bar.get("l", 0)
            bar_close = bar.get("c", 0)
            bar_high  = bar.get("h", 0)
            bar_vol   = bar.get("v", 0)

            # ── Detect the sweep bar ──────────────────────────────────────────
            if sweep_idx == -1:
                if bar_low < level_price:
                    depth_pct = (level_price - bar_low) / level_price * 100
                    if depth_pct >= _MIN_SWEEP_DEPTH_PCT:
                        sweep_idx = i
                        sweep_low = bar_low
                        sweep_vol = bar_vol
                        result.sweep_detected = True
                        result.sweep_bar_idx  = i
                        result.sweep_low      = sweep_low
                        result.sweep_depth_pct = round(depth_pct, 4)
                continue   # keep scanning for reclaim

            # ── Check for reclaim within max_reclaim_bars ─────────────────────
            bars_since = i - sweep_idx
            if bars_since > self._max_reclaim:
                # Too slow to reclaim — sweep failed, reset and look again
                sweep_idx = -1
                continue

            if bar_close > level_price:
                # Reclaim confirmed
                result.reclaimed         = True
                result.reclaim_bar_idx   = i
                result.bars_to_reclaim   = bars_since

                reclaim_vol_ratio = (bar_vol / sweep_vol) if sweep_vol > 0 else 0.0
                result.reclaim_volume_ratio = round(reclaim_vol_ratio, 4)

                # Check for higher low in remaining bars
                remaining = bars[i + 1:]
                result.higher_low = self._detect_higher_low(remaining, sweep_low)

                # Score the sweep quality
                result.quality_score, result.quality_label = self._score_sweep(result)
                result.confirmed = True
                result.reasons   = self._build_reasons(result)
                result.warnings  = self._build_warnings(result)

                log.debug(
                    "[sweep] %s %s: CONFIRMED sweep=%.4f reclaim in %d bars score=%.1f",
                    ticker, level_type, sweep_low, bars_since, result.quality_score,
                )
                return result   # take the first confirmed sweep per level

        # Sweep detected but never reclaimed
        if result.sweep_detected:
            result.warnings.append(
                f"price swept below {level_type} ({level_price:.4f}) "
                f"but did not reclaim within {self._max_reclaim} bars"
            )

        return result

    # ── Higher low detection ──────────────────────────────────────────────────

    def _detect_higher_low(self, bars: list[dict], sweep_low: float) -> bool:
        """
        True when a bar after the reclaim forms a low that is higher than
        the sweep low — confirming the sweep held and buyers stepped in.
        """
        if not bars:
            return False
        return any(b.get("l", 0) > sweep_low for b in bars)

    # ── Quality scoring ───────────────────────────────────────────────────────

    def _score_sweep(self, sweep: SweepResult) -> tuple[float, str]:
        """
        Score sweep quality 0–100.

        Factors:
          Reclaim speed (30 pts): faster = better (1 bar = 30, 5 bars = 6)
          Sweep depth   (25 pts): deeper = more liquidity cleared
          Vol ratio     (25 pts): strong reclaim volume vs sweep volume
          Higher low    (20 pts): structural confirmation
        """
        score = 0.0

        # Reclaim speed (30 pts): 1 bar = 30, 2 bars = 20, 3 bars = 15, 4 = 8, 5 = 4
        speed_map = {0: 30, 1: 30, 2: 20, 3: 15, 4: 8, 5: 4}
        score += speed_map.get(sweep.bars_to_reclaim, 0)

        # Sweep depth (25 pts): min 0.1% = 5pts, 1%+ = 25pts
        depth_score = min(sweep.sweep_depth_pct / 1.0 * 25, 25)
        score += depth_score

        # Volume ratio (25 pts): reclaim vol >= sweep vol = 25pts
        vol_score = min(sweep.reclaim_volume_ratio * 25, 25)
        score += vol_score

        # Higher low (20 pts)
        if sweep.higher_low:
            score += 20

        score = round(min(score, 100), 2)

        if score >= 70:
            label = "high"
        elif score >= 45:
            label = "medium"
        elif score > 0:
            label = "low"
        else:
            label = "none"

        return score, label

    # ── Reason / warning builders ─────────────────────────────────────────────

    def _build_reasons(self, sweep: SweepResult) -> list[str]:
        reasons = [
            f"Price swept below {sweep.level_type} ({sweep.level_price:.4f})",
            f"Reclaimed in {sweep.bars_to_reclaim} bar(s)",
        ]
        if sweep.higher_low:
            reasons.append("Higher low confirmed after reclaim")
        if sweep.reclaim_volume_ratio >= 1.0:
            reasons.append("Strong volume on reclaim bar")
        return reasons

    def _build_warnings(self, sweep: SweepResult) -> list[str]:
        warnings = []
        if sweep.bars_to_reclaim >= 4:
            warnings.append("Slow reclaim — took 4+ bars")
        if sweep.reclaim_volume_ratio < _MIN_RECLAIM_VOL_RATIO:
            warnings.append("Weak volume on reclaim bar")
        if sweep.sweep_depth_pct < 0.3:
            warnings.append("Shallow sweep depth — may be noise")
        return warnings
