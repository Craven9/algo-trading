"""
src/analysis/key_level_engine.py — Key price level detection
Identifies significant support and resistance levels from bar data.
These levels feed setup detectors, scoring engines, and the exit system.

Responsibilities:
  - Detect intraday support and resistance from swing highs/lows
  - Identify premarket high and low as key levels
  - Identify prior day high and low as key levels
  - Detect round number levels near the current price
  - Classify each level by type, strength, and distance from current price
  - Determine whether price is holding above, breaking through, or
    failing at a key level

Key level types:
  "swing_high"      — intraday swing high (resistance)
  "swing_low"       — intraday swing low (support)
  "premarket_high"  — premarket session high
  "premarket_low"   — premarket session low
  "prior_day_high"  — previous day's high
  "prior_day_low"   — previous day's low
  "round_number"    — $1.00, $2.50, $5.00, etc.
  "vwap"            — VWAP treated as a dynamic key level

Level strength:
  "strong"   — tested multiple times, held
  "moderate" — tested once or near a confluence
  "weak"     — only appeared once, not tested
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from candle_builder import (
    filter_session_bars,
    premarket_high,
    premarket_low,
)
from indicator_calculator import find_swing_highs, find_swing_lows

log = logging.getLogger(__name__)

# Distance threshold — a price is "at" a level if within this % of it
_AT_LEVEL_PCT    = 0.5    # 0.5%
_NEAR_LEVEL_PCT  = 2.0    # 2.0%

# Round number increments to check
_ROUND_INCREMENTS = [0.25, 0.50, 1.00, 2.50, 5.00, 10.00]


# ── Key level dataclass ───────────────────────────────────────────────────────

@dataclass
class KeyLevel:
    """A single identified key price level."""
    price:      float
    level_type: str           # see module docstring for valid types
    strength:   str  = "moderate"
    touches:    int  = 1      # number of times price tested this level
    direction:  str  = "both" # "support" | "resistance" | "both"
    notes:      str  = ""

    def distance_pct(self, current_price: float) -> float:
        """Percentage distance from current price to this level."""
        if self.price == 0:
            return float("inf")
        return (current_price - self.price) / self.price * 100

    def is_above(self, current_price: float) -> bool:
        """True when this level is above the current price (resistance)."""
        return self.price > current_price

    def is_below(self, current_price: float) -> bool:
        """True when this level is below the current price (support)."""
        return self.price < current_price

    def is_at(self, current_price: float, pct: float = _AT_LEVEL_PCT) -> bool:
        """True when current price is within `pct`% of this level."""
        return abs(self.distance_pct(current_price)) <= pct

    def is_near(self, current_price: float, pct: float = _NEAR_LEVEL_PCT) -> bool:
        """True when current price is within `pct`% of this level."""
        return abs(self.distance_pct(current_price)) <= pct

    def to_dict(self) -> dict:
        return {
            "price":      round(self.price, 4),
            "level_type": self.level_type,
            "strength":   self.strength,
            "touches":    self.touches,
            "direction":  self.direction,
            "notes":      self.notes,
        }


# ── Key level result ──────────────────────────────────────────────────────────

@dataclass
class KeyLevelResult:
    """
    All identified key levels for a ticker plus derived signals.
    Produced by KeyLevelEngine.analyze() and consumed by setup detectors
    and scoring engines.
    """
    ticker:           str
    current_price:    float
    levels:           list[KeyLevel]     = field(default_factory=list)

    # Nearest levels
    nearest_support:  Optional[KeyLevel] = None
    nearest_resistance: Optional[KeyLevel] = None

    # Status flags
    price_at_key_level:    bool = False
    price_near_support:    bool = False
    price_near_resistance: bool = False
    breaking_out:          bool = False   # price just cleared resistance
    breaking_down:         bool = False   # price just lost support
    holding_support:       bool = False
    rejecting_resistance:  bool = False

    # Distance to nearest levels
    support_distance_pct:    float = 0.0
    resistance_distance_pct: float = 0.0

    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def support_levels(self) -> list[KeyLevel]:
        return [l for l in self.levels if l.is_below(self.current_price)]

    def resistance_levels(self) -> list[KeyLevel]:
        return [l for l in self.levels if l.is_above(self.current_price)]

    def strong_levels(self) -> list[KeyLevel]:
        return [l for l in self.levels if l.strength == "strong"]

    def to_dict(self) -> dict:
        return {
            "ticker":                  self.ticker,
            "current_price":           self.current_price,
            "levels":                  [l.to_dict() for l in self.levels],
            "nearest_support":         self.nearest_support.to_dict() if self.nearest_support else None,
            "nearest_resistance":      self.nearest_resistance.to_dict() if self.nearest_resistance else None,
            "price_at_key_level":      self.price_at_key_level,
            "price_near_support":      self.price_near_support,
            "price_near_resistance":   self.price_near_resistance,
            "breaking_out":            self.breaking_out,
            "breaking_down":           self.breaking_down,
            "holding_support":         self.holding_support,
            "rejecting_resistance":    self.rejecting_resistance,
            "support_distance_pct":    round(self.support_distance_pct, 4),
            "resistance_distance_pct": round(self.resistance_distance_pct, 4),
            "analyzed_at":             self.analyzed_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class KeyLevelEngine:
    """
    Detects and classifies key price levels from bar data.

    Usage:
        engine = KeyLevelEngine(settings)
        result = engine.analyze("ABCD", bars, current_price, vwap=3.10)
    """

    def __init__(self, settings: dict):
        self._settings    = settings
        self._swing_look  = 3    # bars on each side for swing detection
        self._at_pct      = _AT_LEVEL_PCT
        self._near_pct    = _NEAR_LEVEL_PCT

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        ticker:        str,
        bars:          list[dict],
        current_price: float,
        vwap:          Optional[float]  = None,
        prev_day_high: Optional[float]  = None,
        prev_day_low:  Optional[float]  = None,
    ) -> KeyLevelResult:
        """
        Identify all key levels and derive price-relative signals.

        Args:
            ticker:        Ticker symbol (for logging).
            bars:          All bars (all sessions) for the current day.
            current_price: Latest price.
            vwap:          Current VWAP (treated as a dynamic key level).
            prev_day_high: Prior day's high (optional).
            prev_day_low:  Prior day's low (optional).

        Returns:
            KeyLevelResult with all levels and derived signals.
        """
        if not bars or current_price <= 0:
            return KeyLevelResult(ticker=ticker, current_price=current_price)

        levels: list[KeyLevel] = []

        # ── Swing highs and lows (regular session only) ───────────────────────
        reg_bars = filter_session_bars(bars, "regular")
        if len(reg_bars) >= self._swing_look * 2 + 1:
            swing_highs = find_swing_highs(reg_bars, self._swing_look)
            swing_lows  = find_swing_lows(reg_bars,  self._swing_look)

            for price in swing_highs:
                levels.append(KeyLevel(
                    price      = price,
                    level_type = "swing_high",
                    strength   = self._swing_strength(price, swing_highs),
                    direction  = "resistance",
                ))
            for price in swing_lows:
                levels.append(KeyLevel(
                    price      = price,
                    level_type = "swing_low",
                    strength   = self._swing_strength(price, swing_lows),
                    direction  = "support",
                ))

        # ── Premarket levels ──────────────────────────────────────────────────
        pm_high = premarket_high(bars)
        pm_low  = premarket_low(bars)
        if pm_high:
            levels.append(KeyLevel(
                price      = pm_high,
                level_type = "premarket_high",
                strength   = "strong",
                direction  = "resistance",
                notes      = "premarket high — often acts as intraday resistance",
            ))
        if pm_low:
            levels.append(KeyLevel(
                price      = pm_low,
                level_type = "premarket_low",
                strength   = "strong",
                direction  = "support",
                notes      = "premarket low — break below is bearish",
            ))

        # ── Prior day levels ──────────────────────────────────────────────────
        if prev_day_high:
            levels.append(KeyLevel(
                price      = prev_day_high,
                level_type = "prior_day_high",
                strength   = "strong",
                direction  = "resistance",
                notes      = "prior day high",
            ))
        if prev_day_low:
            levels.append(KeyLevel(
                price      = prev_day_low,
                level_type = "prior_day_low",
                strength   = "strong",
                direction  = "support",
                notes      = "prior day low",
            ))

        # ── VWAP as key level ─────────────────────────────────────────────────
        if vwap and vwap > 0:
            levels.append(KeyLevel(
                price      = vwap,
                level_type = "vwap",
                strength   = "strong",
                direction  = "both",
                notes      = "VWAP — dynamic support/resistance",
            ))

        # ── Round number levels ───────────────────────────────────────────────
        levels.extend(self._round_number_levels(current_price))

        # ── De-duplicate levels within 0.3% of each other ────────────────────
        levels = self._deduplicate(levels)

        # ── Derive signals ────────────────────────────────────────────────────
        result = self._derive_signals(ticker, current_price, levels)
        log.debug(
            "[key_levels] %s: %d levels — support=%.4f resistance=%.4f",
            ticker,
            len(levels),
            result.nearest_support.price if result.nearest_support else 0,
            result.nearest_resistance.price if result.nearest_resistance else 0,
        )
        return result

    # ── Level builders ────────────────────────────────────────────────────────

    def _round_number_levels(self, price: float) -> list[KeyLevel]:
        """
        Find round number levels within 10% of the current price.
        Round numbers act as psychological support/resistance.
        """
        levels: list[KeyLevel] = []
        if price <= 0:
            return levels

        for increment in _ROUND_INCREMENTS:
            # Find the nearest round number above and below
            below = math.floor(price / increment) * increment
            above = math.ceil(price  / increment) * increment

            for rn_price in {below, above}:
                if rn_price <= 0:
                    continue
                dist_pct = abs((price - rn_price) / price * 100)
                if dist_pct > 10.0:
                    continue
                # Only include if not too close to current price (avoid noise)
                if dist_pct < 0.1:
                    continue
                levels.append(KeyLevel(
                    price      = round(rn_price, 4),
                    level_type = "round_number",
                    strength   = "weak",
                    direction  = "above" if rn_price > price else "below",
                    notes      = f"round number ${rn_price:.2f}",
                ))

        return levels

    def _swing_strength(self, price: float, all_prices: list[float]) -> str:
        """
        Rate a swing level's strength by how many times it appears
        (within 0.5%) in the full list of swing highs or lows.
        """
        touches = sum(
            1 for p in all_prices
            if abs(p - price) / price * 100 <= 0.5
        )
        if touches >= 3:
            return "strong"
        if touches == 2:
            return "moderate"
        return "weak"

    # ── De-duplication ────────────────────────────────────────────────────────

    def _deduplicate(self, levels: list[KeyLevel]) -> list[KeyLevel]:
        """
        Remove duplicate levels within 0.3% of each other.
        When two levels are very close, keep the one with higher strength.
        """
        if not levels:
            return []

        strength_order = {"strong": 3, "moderate": 2, "weak": 1}
        levels.sort(key=lambda l: l.price)
        kept: list[KeyLevel] = []

        for level in levels:
            merged = False
            for existing in kept:
                if existing.price > 0:
                    pct_diff = abs(level.price - existing.price) / existing.price * 100
                    if pct_diff <= 0.3:
                        # Keep the stronger one
                        if strength_order.get(level.strength, 0) > strength_order.get(existing.strength, 0):
                            existing.price      = level.price
                            existing.strength   = level.strength
                            existing.level_type = level.level_type
                        existing.touches += 1
                        merged = True
                        break
            if not merged:
                kept.append(level)

        return kept

    # ── Signal derivation ─────────────────────────────────────────────────────

    def _derive_signals(
        self,
        ticker:        str,
        current_price: float,
        levels:        list[KeyLevel],
    ) -> KeyLevelResult:
        """Compute price-relative signals from the level list."""

        # Nearest support (highest level below price)
        supports     = [l for l in levels if l.price < current_price]
        resistances  = [l for l in levels if l.price > current_price]

        nearest_sup  = max(supports,    key=lambda l: l.price) if supports    else None
        nearest_res  = min(resistances, key=lambda l: l.price) if resistances else None

        sup_dist  = abs(nearest_sup.distance_pct(current_price))  if nearest_sup else 0.0
        res_dist  = abs(nearest_res.distance_pct(current_price))  if nearest_res else 0.0

        # Price-at-level: within _AT_LEVEL_PCT of any level
        at_level = any(l.is_at(current_price, self._at_pct) for l in levels)

        # Near support / resistance
        near_sup = nearest_sup is not None and sup_dist <= self._near_pct
        near_res = nearest_res is not None and res_dist <= self._near_pct

        # Breaking out: price is above resistance but within 1% of it
        breaking_out = (
            nearest_res is not None
            and current_price > nearest_res.price
            and abs(current_price - nearest_res.price) / nearest_res.price * 100 <= 1.0
        )

        # Breaking down: price just lost support (within 1% below it)
        breaking_down = (
            nearest_sup is not None
            and current_price < nearest_sup.price
            and abs(current_price - nearest_sup.price) / nearest_sup.price * 100 <= 1.0
        )

        # Holding support: near support and not breaking down
        holding_sup = near_sup and not breaking_down

        # Rejecting resistance: near resistance from below
        rejecting_res = near_res and not breaking_out

        return KeyLevelResult(
            ticker                  = ticker,
            current_price           = current_price,
            levels                  = levels,
            nearest_support         = nearest_sup,
            nearest_resistance      = nearest_res,
            price_at_key_level      = at_level,
            price_near_support      = near_sup,
            price_near_resistance   = near_res,
            breaking_out            = breaking_out,
            breaking_down           = breaking_down,
            holding_support         = holding_sup,
            rejecting_resistance    = rejecting_res,
            support_distance_pct    = sup_dist,
            resistance_distance_pct = res_dist,
        )
