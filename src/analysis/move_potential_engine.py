"""
src/analysis/move_potential_engine.py — Move potential scoring engine
Scores whether a ticker has meaningful upside room for a profitable trade.
A strong setup with no room to move is still a bad trade.

Responsibilities:
  - Estimate how far price can realistically move before hitting resistance
  - Score the opportunity size (small scalp vs large momentum play)
  - Factor in float, RVOL, catalyst, and Fibonacci extension targets
  - Identify nearest resistance that would cap the move
  - Calculate potential reward in R-multiples and percent

Design rules:
  - Move potential is a SCORING factor — not a hard block
  - Low move potential reduces the final trade quality score
  - Scores feed directly into final_trade_quality_score via the spec formula:
      final = setup*0.35 + probability*0.35 + rr*0.15 + move_potential*0.10 + hist*0.05
  - A ticker up 200% with no room is NOT a good trade
  - A ticker up 30% with clear room to 100% IS a good trade

Move potential score labels:
  80-100 = "strong"   — significant upside with clear targets
  60-79  = "moderate" — reasonable move potential
  40-59  = "limited"  — small room, be cautious
  0-39   = "weak"     — little to no room, avoid

Scoring factors (total 100 pts):
  Room to nearest resistance:  30 pts — core factor
  Fibonacci extension targets: 25 pts — structured targets above
  RVOL strength:               20 pts — fuel for the move
  Day move context:            15 pts — how much of the day range used
  Catalyst presence:           10 pts — news can extend moves
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Score thresholds
_STRONG_THRESHOLD   = 80
_MODERATE_THRESHOLD = 60
_LIMITED_THRESHOLD  = 40


# ── Move potential result ─────────────────────────────────────────────────────

@dataclass
class MovePotentialResult:
    """
    Move potential assessment for a single ticker.
    Consumed by the final trade quality calculation.
    """
    ticker:           str
    current_price:    float

    # Core scores
    move_potential_score:  float  = 0.0    # 0–100
    score_label:           str    = "weak" # strong|moderate|limited|weak

    # Room analysis
    nearest_resistance:    Optional[float] = None
    room_to_resistance_pct: float          = 0.0
    room_to_resistance_r:   float          = 0.0   # in R-multiples

    # Extension targets
    fib_target_1:    Optional[float] = None   # 1.272
    fib_target_2:    Optional[float] = None   # 1.618
    runner_target:   Optional[float] = None   # 2.000
    parabolic_target:Optional[float] = None   # 2.618

    # Context
    rvol:                 float = 0.0
    day_change_pct:       float = 0.0
    day_range_used_pct:   float = 0.0   # how much of prior day range used
    catalyst_strength:    str   = "none"

    # Score breakdown
    score_breakdown:  dict = field(default_factory=dict)
    reasons:          list[str] = field(default_factory=list)
    warnings:         list[str] = field(default_factory=list)

    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def is_strong(self) -> bool:
        return self.move_potential_score >= _STRONG_THRESHOLD

    def is_adequate(self) -> bool:
        return self.move_potential_score >= _LIMITED_THRESHOLD

    def best_target(self) -> Optional[float]:
        """Return the most achievable extension target."""
        return self.fib_target_1 or self.fib_target_2 or self.runner_target

    def to_dict(self) -> dict:
        return {
            "ticker":                  self.ticker,
            "current_price":           self.current_price,
            "move_potential_score":    round(self.move_potential_score, 2),
            "score_label":             self.score_label,
            "nearest_resistance":      round(self.nearest_resistance, 4) if self.nearest_resistance else None,
            "room_to_resistance_pct":  round(self.room_to_resistance_pct, 4),
            "room_to_resistance_r":    round(self.room_to_resistance_r,   4),
            "fib_target_1":            round(self.fib_target_1,    4) if self.fib_target_1    else None,
            "fib_target_2":            round(self.fib_target_2,    4) if self.fib_target_2    else None,
            "runner_target":           round(self.runner_target,   4) if self.runner_target   else None,
            "parabolic_target":        round(self.parabolic_target,4) if self.parabolic_target else None,
            "rvol":                    self.rvol,
            "day_change_pct":          self.day_change_pct,
            "day_range_used_pct":      round(self.day_range_used_pct, 4),
            "catalyst_strength":       self.catalyst_strength,
            "score_breakdown":         self.score_breakdown,
            "reasons":                 self.reasons,
            "warnings":                self.warnings,
            "analyzed_at":             self.analyzed_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class MovePotentialEngine:
    """
    Scores move potential for scanner candidates.

    Usage:
        engine = MovePotentialEngine(settings)
        result = engine.score(
            ticker          = "ABCD",
            current_price   = 3.42,
            entry_price     = 3.42,
            stop_price      = 3.20,
            resistance_levels = [3.80, 4.20],
            fib_extensions  = {"1.272": 4.05, "1.618": 4.72, "2.0": 5.25},
            rvol            = 8.5,
            day_change_pct  = 42.0,
            day_high        = 3.55,
            prev_day_high   = 2.80,
            catalyst_strength = "strong",
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._entry    = settings.get("entry_rules", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        ticker:            str,
        current_price:     float,
        entry_price:       Optional[float]        = None,
        stop_price:        Optional[float]         = None,
        resistance_levels: Optional[list[float]]  = None,
        fib_extensions:    Optional[dict]         = None,
        rvol:              float                   = 0.0,
        day_change_pct:    float                   = 0.0,
        day_high:          Optional[float]         = None,
        prev_day_high:     Optional[float]         = None,
        catalyst_strength: str                     = "none",
    ) -> MovePotentialResult:
        """
        Score move potential for a ticker.

        Args:
            ticker:            Ticker symbol.
            current_price:     Latest price.
            entry_price:       Planned entry price (defaults to current_price).
            stop_price:        Planned stop loss price (used for R-multiple calc).
            resistance_levels: List of resistance prices above current price.
            fib_extensions:    Dict of {"ratio": price} from fibonacci_strategy_engine.
            rvol:              Relative volume.
            day_change_pct:    Today's % price change.
            day_high:          Today's session high.
            prev_day_high:     Prior day's high (key resistance level).
            catalyst_strength: "strong" | "moderate" | "weak" | "none"

        Returns:
            MovePotentialResult with score and targets.
        """
        result = MovePotentialResult(
            ticker        = ticker,
            current_price = current_price,
        )

        if current_price <= 0:
            result.warnings.append("Invalid price — cannot score move potential")
            return result

        entry = entry_price or current_price
        risk  = abs(entry - stop_price) if stop_price else 0.0

        # ── Nearest resistance ────────────────────────────────────────────────
        resistances = self._collect_resistances(
            current_price, resistance_levels, prev_day_high, fib_extensions
        )
        nearest_res = min(resistances) if resistances else None
        room_pct    = 0.0
        room_r      = 0.0

        if nearest_res and nearest_res > current_price:
            room_pct = (nearest_res - current_price) / current_price * 100
            room_r   = (nearest_res - entry) / risk if risk > 0 else 0.0

        result.nearest_resistance      = nearest_res
        result.room_to_resistance_pct  = round(room_pct, 4)
        result.room_to_resistance_r    = round(room_r,   4)

        # ── Fibonacci extension targets ───────────────────────────────────────
        fib_extensions = fib_extensions or {}
        result.fib_target_1     = _fib_price(fib_extensions, ["1.272","1.272"])
        result.fib_target_2     = _fib_price(fib_extensions, ["1.618","1.618"])
        result.runner_target    = _fib_price(fib_extensions, ["2.0","2.000","2.0"])
        result.parabolic_target = _fib_price(fib_extensions, ["2.618"])

        # ── Day range context ─────────────────────────────────────────────────
        day_range_used = 0.0
        if prev_day_high and prev_day_high > 0 and day_high:
            day_range_used = (day_high - entry) / prev_day_high * 100
        result.day_range_used_pct = round(day_range_used, 4)

        result.rvol              = rvol
        result.day_change_pct    = day_change_pct
        result.catalyst_strength = catalyst_strength

        # ── Compute component scores ──────────────────────────────────────────
        breakdown = {}

        # 1. Room to resistance (30 pts)
        # 5%+ room = 30pts, 3% = 18pts, 1% = 6pts, 0% = 0pts
        room_score = min(room_pct / 5.0 * 30, 30)
        breakdown["room_to_resistance"] = round(room_score, 2)

        # 2. Fibonacci extension targets (25 pts)
        fib_score = self._fib_score(current_price, fib_extensions)
        breakdown["fib_extensions"] = round(fib_score, 2)

        # 3. RVOL (20 pts): 3x=12, 5x=16, 10x+=20
        rvol_score = min(rvol / 10.0 * 20, 20)
        breakdown["rvol"] = round(rvol_score, 2)

        # 4. Day move context (15 pts)
        day_score = self._day_score(day_change_pct, day_range_used)
        breakdown["day_context"] = round(day_score, 2)

        # 5. Catalyst (10 pts)
        cat_scores = {"strong": 10, "moderate": 6, "weak": 2, "none": 0}
        cat_score  = float(cat_scores.get(catalyst_strength, 0))
        breakdown["catalyst"] = cat_score

        total_score = sum(breakdown.values())
        total_score = round(min(max(total_score, 0), 100), 2)

        result.move_potential_score = total_score
        result.score_breakdown      = breakdown
        result.score_label          = self._label(total_score)
        result.reasons              = self._build_reasons(result)
        result.warnings             = self._build_warnings(result, room_pct, day_change_pct)

        log.debug(
            "[move] %s score=%.1f label=%s room=%.2f%% rvol=%.1fx",
            ticker, total_score, result.score_label, room_pct, rvol,
        )
        return result

    # ── Component scorers ─────────────────────────────────────────────────────

    def _fib_score(
        self,
        current_price:  float,
        fib_extensions: dict,
    ) -> float:
        """
        Score based on how many valid Fibonacci extension targets exist
        above the current price and how much room they represent.

        25 pts for having all 4 preferred extensions with clear room.
        """
        if not fib_extensions:
            return 0.0

        preferred = ["1.272", "1.618", "2.0", "2.000", "2.618"]
        targets_above = []
        for key in preferred:
            price = _fib_price(fib_extensions, [key])
            if price and price > current_price:
                targets_above.append(price)

        # Deduplicate
        targets_above = sorted(set(targets_above))
        count = len(targets_above)

        if count == 0:
            return 0.0

        # Base score from count (0–15)
        count_score = min(count * 4, 15)

        # Bonus for highest target being well above price (0–10)
        if targets_above:
            highest  = targets_above[-1]
            ext_room = (highest - current_price) / current_price * 100
            ext_bonus = min(ext_room / 30.0 * 10, 10)
        else:
            ext_bonus = 0.0

        return round(count_score + ext_bonus, 2)

    def _day_score(self, day_change_pct: float, day_range_used: float) -> float:
        """
        Score based on today's move context.

        Big move (30%+) is good but if the day range is already exhausted,
        there may be little room left.

        15 pts: strong day move with room remaining.
        """
        # Base from day change magnitude
        change_score = min(day_change_pct / 50.0 * 10, 10)

        # Penalty if most of the day range is already used
        # day_range_used near 0 = early in the move (bonus)
        # day_range_used > 80 = late in the move (penalty)
        if day_range_used > 80:
            exhaustion_penalty = 5.0
        elif day_range_used > 60:
            exhaustion_penalty = 2.0
        else:
            exhaustion_penalty = 0.0

        # Bonus for early discovery (low range used, big RVOL)
        early_bonus = max(0.0, (100 - day_range_used) / 100 * 5)

        return round(max(0.0, change_score - exhaustion_penalty + early_bonus), 2)

    # ── Resistance collection ─────────────────────────────────────────────────

    def _collect_resistances(
        self,
        current_price:     float,
        resistance_levels: Optional[list[float]],
        prev_day_high:     Optional[float],
        fib_extensions:    Optional[dict],
    ) -> list[float]:
        """Collect all resistance prices above current price for room calculation."""
        candidates: list[float] = []

        if resistance_levels:
            candidates.extend(r for r in resistance_levels if r > current_price)

        if prev_day_high and prev_day_high > current_price:
            candidates.append(prev_day_high)

        # Lowest fib extension above price as soft resistance
        if fib_extensions:
            for key in ["1.272", "1.618"]:
                price = _fib_price(fib_extensions, [key])
                if price and price > current_price:
                    candidates.append(price)

        return candidates

    # ── Label / reasons ───────────────────────────────────────────────────────

    @staticmethod
    def _label(score: float) -> str:
        if score >= _STRONG_THRESHOLD:
            return "strong"
        if score >= _MODERATE_THRESHOLD:
            return "moderate"
        if score >= _LIMITED_THRESHOLD:
            return "limited"
        return "weak"

    def _build_reasons(self, r: MovePotentialResult) -> list[str]:
        reasons = []
        if r.room_to_resistance_pct >= 5.0:
            reasons.append(f"{r.room_to_resistance_pct:.1f}% room to nearest resistance")
        if r.fib_target_1:
            reasons.append(f"Fibonacci 1.272 target at ${r.fib_target_1:.2f}")
        if r.runner_target:
            reasons.append(f"Runner target at ${r.runner_target:.2f} (2.0 extension)")
        if r.rvol >= 5.0:
            reasons.append(f"RVOL {r.rvol:.1f}x — strong fuel for the move")
        if r.catalyst_strength == "strong":
            reasons.append("Strong catalyst — can extend the move further")
        return reasons

    def _build_warnings(
        self,
        r:              MovePotentialResult,
        room_pct:       float,
        day_change_pct: float,
    ) -> list[str]:
        warnings = []
        if room_pct < 2.0:
            warnings.append("Less than 2% room to nearest resistance — tight")
        if day_change_pct > 100.0:
            warnings.append(f"Up {day_change_pct:.0f}% today — late-stage, lower potential")
        if r.nearest_resistance is None:
            warnings.append("No clear resistance identified — use Fibonacci targets")
        return warnings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fib_price(fib_dict: dict, keys: list[str]) -> Optional[float]:
    """Return the first matching price from a list of key variants."""
    for key in keys:
        val = fib_dict.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None
