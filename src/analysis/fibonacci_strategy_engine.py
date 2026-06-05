"""
src/analysis/fibonacci_strategy_engine.py — Fibonacci retracement and extension engine
Calculates Fibonacci levels from swing high/low and determines whether
the current price is at a valid retracement entry or extension target.

This is one of the most important analysis modules — it feeds:
  - setup_score_engine.py  (entry confirmation bonus)
  - probability_engine.py  (probability bonus)
  - move_potential_engine.py (runner target levels)
  - exit_manager.py          (profit targets)
  - trade_quality_gate.py    (block if under 0.618 after failed reclaim)

Retracement levels (for pullback entries):
  0.236, 0.382, 0.500, 0.618, 0.786

Extension levels (for runner targets):
  1.272, 1.414, 1.618, 2.000, 2.618

Design rules per the spec:
  - Preferred retracements for entry: 0.382, 0.500, 0.618
  - If price holds near 0.382/0.5/0.618 with VWAP support + higher low → strong entry
  - If price breaks out and holds above prior high → use 1.272, 1.618, 2.0, 2.618 as targets
  - If price fails reclaim and stays below 0.618 → BLOCK long entry
  - Max distance from fib level: 2.0% (configurable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Default Fibonacci levels
_RETRACEMENT_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
_EXTENSION_LEVELS   = [1.272, 1.414, 1.618, 2.000, 2.618]
_PREFERRED_RETRACEMENTS = [0.382, 0.500, 0.618]
_PREFERRED_EXTENSIONS   = [1.272, 1.618, 2.000, 2.618]

# Hard block: if price is below 0.618 after a failed reclaim, block long entry
_FAILED_RECLAIM_BLOCK_LEVEL = 0.618


# ── Fibonacci level ───────────────────────────────────────────────────────────

@dataclass
class FibLevel:
    """A single Fibonacci level price point."""
    ratio:       float
    price:       float
    level_type:  str    = "retracement"   # "retracement" | "extension"
    is_preferred:bool   = False

    def distance_pct(self, current_price: float) -> float:
        """Percentage distance from current price to this level."""
        if self.price == 0:
            return float("inf")
        return abs(current_price - self.price) / self.price * 100

    def is_at(self, current_price: float, max_pct: float = 2.0) -> bool:
        """True when current price is within max_pct% of this level."""
        return self.distance_pct(current_price) <= max_pct

    def to_dict(self) -> dict:
        return {
            "ratio":        self.ratio,
            "price":        round(self.price, 4),
            "level_type":   self.level_type,
            "is_preferred": self.is_preferred,
        }


# ── Fibonacci result ──────────────────────────────────────────────────────────

@dataclass
class FibonacciResult:
    """
    Full Fibonacci analysis for a ticker.
    Produced by FibonacciStrategyEngine.analyze() and consumed by
    setup detectors, scoring engines, and the exit manager.
    """
    ticker:                  str
    current_price:           float

    # Swing points used for calculation
    swing_high:              Optional[float]   = None
    swing_low:               Optional[float]   = None
    swing_range:             float             = 0.0

    # Trend direction (determines retracement vs extension orientation)
    trend_up:                bool              = True

    # All computed levels
    retracement_levels:      list[FibLevel]    = field(default_factory=list)
    extension_levels:        list[FibLevel]    = field(default_factory=list)

    # Nearest retracement
    nearest_retracement:     Optional[FibLevel] = None
    distance_from_fib_pct:   float              = 0.0

    # Entry confirmation
    fib_trend_valid:         bool   = False
    entry_confirmed_by_fib:  bool   = False
    at_preferred_level:      bool   = False   # at 0.382, 0.5, or 0.618
    block_trade:             bool   = False   # True when below 0.618 after failed reclaim

    # Extension targets (for exits and runner management)
    target_extensions:       dict   = field(default_factory=dict)

    reasons:   list[str] = field(default_factory=list)
    warnings:  list[str] = field(default_factory=list)

    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def get_extension_price(self, ratio: float) -> Optional[float]:
        """Return the price for a specific extension ratio."""
        key = str(ratio)
        return self.target_extensions.get(key)

    def to_dict(self) -> dict:
        return {
            "ticker":                 self.ticker,
            "current_price":          self.current_price,
            "swing_high":             round(self.swing_high, 4) if self.swing_high else None,
            "swing_low":              round(self.swing_low,  4) if self.swing_low  else None,
            "swing_range":            round(self.swing_range,  4),
            "trend_up":               self.trend_up,
            "fib_trend_valid":        self.fib_trend_valid,
            "nearest_retracement":    self.nearest_retracement.to_dict() if self.nearest_retracement else None,
            "distance_from_fib_pct":  round(self.distance_from_fib_pct, 4),
            "entry_confirmed_by_fib": self.entry_confirmed_by_fib,
            "at_preferred_level":     self.at_preferred_level,
            "block_trade":            self.block_trade,
            "target_extensions":      {k: round(v, 4) for k, v in self.target_extensions.items()},
            "retracement_levels":     [l.to_dict() for l in self.retracement_levels],
            "extension_levels":       [l.to_dict() for l in self.extension_levels],
            "reasons":                self.reasons,
            "warnings":               self.warnings,
            "analyzed_at":            self.analyzed_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class FibonacciStrategyEngine:
    """
    Computes Fibonacci retracement and extension levels from swing data
    and determines entry quality and extension targets.

    Usage:
        engine = FibonacciStrategyEngine(settings)
        result = engine.analyze(
            "ABCD",
            current_price = 3.42,
            swing_high    = 4.00,
            swing_low     = 2.80,
            vwap          = 3.38,
            higher_low    = True,
            failed_reclaim= False,
        )
    """

    def __init__(self, settings: dict):
        self._settings    = settings
        self._fib_cfg     = settings.get("fibonacci_strategy", {})
        self._enabled     = self._fib_cfg.get("enabled", True)
        self._max_dist    = float(self._fib_cfg.get(
            "max_distance_from_fib_level_percent", 2.0))
        self._ret_levels  = list(self._fib_cfg.get(
            "retracement_levels", _RETRACEMENT_LEVELS))
        self._ext_levels  = list(self._fib_cfg.get(
            "extension_levels", _EXTENSION_LEVELS))
        self._pref_ret    = list(self._fib_cfg.get(
            "preferred_retracements", _PREFERRED_RETRACEMENTS))
        self._pref_ext    = list(self._fib_cfg.get(
            "preferred_extensions", _PREFERRED_EXTENSIONS))
        self._req_vwap    = bool(self._fib_cfg.get(
            "require_vwap_confirmation", True))
        self._req_hl      = bool(self._fib_cfg.get(
            "require_higher_low_confirmation", True))
        self._block_0618  = bool(self._fib_cfg.get(
            "block_if_under_0618_after_failed_reclaim", True))

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        ticker:         str,
        current_price:  float,
        swing_high:     Optional[float]  = None,
        swing_low:      Optional[float]  = None,
        vwap:           Optional[float]  = None,
        higher_low:     bool             = False,
        failed_reclaim: bool             = False,
        trend_up:       bool             = True,
    ) -> FibonacciResult:
        """
        Compute Fibonacci levels and determine entry/target quality.

        Args:
            ticker:         Ticker symbol.
            current_price:  Latest close price.
            swing_high:     Most significant recent swing high.
            swing_low:      Most significant recent swing low.
            vwap:           Current VWAP (used for entry confirmation).
            higher_low:     True when a higher low has formed.
            failed_reclaim: True when price failed to reclaim a key level.
            trend_up:       True for uptrend (retracement from high to low);
                            False for downtrend.

        Returns:
            FibonacciResult with all levels, entry confirmation, and targets.
        """
        result = FibonacciResult(
            ticker        = ticker,
            current_price = current_price,
            swing_high    = swing_high,
            swing_low     = swing_low,
            trend_up      = trend_up,
        )

        if not self._enabled:
            result.reasons.append("Fibonacci engine disabled in settings")
            return result

        if not swing_high or not swing_low or current_price <= 0:
            result.reasons.append("Insufficient swing data for Fibonacci calculation")
            return result

        if swing_high <= swing_low:
            result.warnings.append("swing_high <= swing_low — invalid swing data")
            return result

        swing_range = swing_high - swing_low
        result.swing_range = round(swing_range, 4)

        # ── Compute retracement levels ────────────────────────────────────────
        retracements = self._compute_retracements(
            swing_high, swing_low, swing_range, trend_up
        )
        result.retracement_levels = retracements

        # ── Compute extension levels ──────────────────────────────────────────
        extensions = self._compute_extensions(
            swing_high, swing_low, swing_range, trend_up
        )
        result.extension_levels = extensions

        # ── Build extension target dict ───────────────────────────────────────
        result.target_extensions = {
            str(l.ratio): l.price for l in extensions
        }

        # ── Find nearest retracement ──────────────────────────────────────────
        nearest, dist = self._nearest_retracement(current_price, retracements)
        result.nearest_retracement   = nearest
        result.distance_from_fib_pct = round(dist, 4)

        # ── Trend validity ────────────────────────────────────────────────────
        result.fib_trend_valid = self._trend_valid(
            current_price, swing_high, swing_low, trend_up
        )

        # ── Block trade check ─────────────────────────────────────────────────
        if failed_reclaim and self._block_0618:
            block_price = self._level_price(swing_high, swing_low, swing_range,
                                            _FAILED_RECLAIM_BLOCK_LEVEL, trend_up)
            if current_price < block_price:
                result.block_trade = True
                result.warnings.append(
                    f"Price below 0.618 Fibonacci ({block_price:.4f}) "
                    f"after failed reclaim — long entry blocked"
                )

        # ── Entry confirmation ────────────────────────────────────────────────
        at_preferred = nearest is not None and nearest.is_preferred and dist <= self._max_dist

        vwap_ok  = (not self._req_vwap)  or (vwap is not None and current_price >= vwap)
        hl_ok    = (not self._req_hl)    or higher_low

        result.at_preferred_level     = at_preferred
        result.entry_confirmed_by_fib = (
            at_preferred
            and vwap_ok
            and hl_ok
            and not result.block_trade
            and result.fib_trend_valid
        )

        # ── Reasons / warnings ────────────────────────────────────────────────
        result.reasons  = self._build_reasons(result, vwap, vwap_ok, hl_ok, higher_low)
        result.warnings = result.warnings + self._build_warnings(result, dist)

        log.debug(
            "[fib] %s: nearest=%.3f dist=%.2f%% confirmed=%s block=%s",
            ticker,
            nearest.ratio if nearest else 0,
            dist,
            result.entry_confirmed_by_fib,
            result.block_trade,
        )
        return result

    # ── Level computation ─────────────────────────────────────────────────────

    def _compute_retracements(
        self,
        swing_high: float,
        swing_low:  float,
        swing_range:float,
        trend_up:   bool,
    ) -> list[FibLevel]:
        """
        For an uptrend: retracements pull back from swing_high toward swing_low.
        Price = swing_high - (ratio * swing_range)

        For a downtrend: retracements pull back up from swing_low toward swing_high.
        Price = swing_low + (ratio * swing_range)
        """
        levels: list[FibLevel] = []
        for ratio in self._ret_levels:
            price = self._level_price(swing_high, swing_low, swing_range, ratio, trend_up)
            levels.append(FibLevel(
                ratio        = ratio,
                price        = round(price, 4),
                level_type   = "retracement",
                is_preferred = ratio in self._pref_ret,
            ))
        return levels

    def _compute_extensions(
        self,
        swing_high: float,
        swing_low:  float,
        swing_range:float,
        trend_up:   bool,
    ) -> list[FibLevel]:
        """
        For an uptrend: extensions project above swing_high.
        Price = swing_high + ((ratio - 1) * swing_range)  — but we use
        the standard formula: swing_low + (ratio * swing_range)

        Standard Fibonacci extension from swing_low with ratio > 1:
        Price = swing_low + (ratio * swing_range)
        """
        levels: list[FibLevel] = []
        for ratio in self._ext_levels:
            if trend_up:
                price = swing_low + ratio * swing_range
            else:
                price = swing_high - ratio * swing_range
            levels.append(FibLevel(
                ratio        = ratio,
                price        = round(price, 4),
                level_type   = "extension",
                is_preferred = ratio in self._pref_ext,
            ))
        return levels

    def _level_price(
        self,
        swing_high:  float,
        swing_low:   float,
        swing_range: float,
        ratio:       float,
        trend_up:    bool,
    ) -> float:
        if trend_up:
            return swing_high - ratio * swing_range
        return swing_low + ratio * swing_range

    # ── Nearest retracement ───────────────────────────────────────────────────

    def _nearest_retracement(
        self,
        current_price:  float,
        retracements:   list[FibLevel],
    ) -> tuple[Optional[FibLevel], float]:
        """Return (nearest_level, distance_pct)."""
        if not retracements:
            return None, float("inf")
        nearest = min(retracements, key=lambda l: l.distance_pct(current_price))
        return nearest, nearest.distance_pct(current_price)

    # ── Trend validity ────────────────────────────────────────────────────────

    def _trend_valid(
        self,
        current_price: float,
        swing_high:    float,
        swing_low:     float,
        trend_up:      bool,
    ) -> bool:
        """
        True when the current price is in the right zone for the stated trend.
        Uptrend: price should be above swing_low (in the retracement zone)
        Downtrend: price should be below swing_high
        """
        if trend_up:
            # Price must be between swing_low and swing_high (pullback zone)
            return swing_low < current_price < swing_high
        else:
            return swing_low < current_price < swing_high

    # ── Reason builders ───────────────────────────────────────────────────────

    def _build_reasons(
        self,
        r:          FibonacciResult,
        vwap:       Optional[float],
        vwap_ok:    bool,
        hl_ok:      bool,
        higher_low: bool,
    ) -> list[str]:
        reasons = []
        if r.nearest_retracement:
            reasons.append(
                f"Price near {r.nearest_retracement.ratio} Fibonacci retracement "
                f"({r.nearest_retracement.price:.4f})"
            )
        if r.at_preferred_level:
            reasons.append(
                f"At preferred retracement level ({r.nearest_retracement.ratio})"
            )
        if vwap_ok and vwap:
            reasons.append("VWAP support confirmed")
        if hl_ok and higher_low:
            reasons.append("Higher low confirmed near Fibonacci support")
        if r.entry_confirmed_by_fib:
            reasons.append("Fibonacci entry confirmed — all conditions met")
        return reasons

    def _build_warnings(self, r: FibonacciResult, dist: float) -> list[str]:
        warnings = []
        if dist > self._max_dist and r.nearest_retracement:
            warnings.append(
                f"Price is {dist:.2f}% from nearest Fibonacci level "
                f"(max {self._max_dist:.1f}%) — entry not ideal"
            )
        if r.nearest_retracement and r.nearest_retracement.ratio == 0.786:
            warnings.append("Price at deep 0.786 retracement — structure may be weakening")
        if not r.fib_trend_valid:
            warnings.append("Price outside expected Fibonacci trend zone")
        return warnings
