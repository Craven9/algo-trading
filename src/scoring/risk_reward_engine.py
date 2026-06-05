"""
src/scoring/risk_reward_engine.py — Risk / reward scoring engine
Computes reward-to-risk ratios for planned trades and converts them into
a standardized 0–100 score.

A setup can look strong but still be a bad trade if the stop is too far
away or the target is too close.  This engine answers:

    "Is the planned reward worth the planned risk?"

Responsibilities:
  - Validate entry, stop, and target prices
  - Compute risk per share
  - Compute reward per share
  - Compute reward-to-risk ratio
  - Score the ratio against minimum/preferred thresholds
  - Flag hard blocks when risk/reward is unacceptable
  - Provide clean output for setup_score_engine.py, probability_engine.py,
    and trade_quality_gate.py

Design rules:
  - This file does not approve trades
  - This file does not place orders
  - Minimum reward:risk is config-driven
  - Preferred reward:risk is config-driven
  - Long trades require target above entry and stop below entry
  - Scores are transparent and include reasons/warnings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MIN_RR       = 3.0
_DEFAULT_PREFERRED_RR = 5.0
_DEFAULT_HARD_BLOCK   = 2.0


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RiskRewardResult:
    """
    Risk/reward assessment for a planned trade.
    Consumed by setup_score_engine.py, probability_engine.py, and
    trade_quality_gate.py.
    """
    ticker:              str
    entry_price:         float = 0.0
    stop_price:          float = 0.0
    target_price:        float = 0.0

    # Core values
    risk_per_share:      float = 0.0
    reward_per_share:    float = 0.0
    reward_to_risk:      float = 0.0
    risk_percent:        float = 0.0
    reward_percent:      float = 0.0

    # Score / flags
    risk_reward_score:   float = 0.0
    meets_minimum:       bool  = False
    meets_preferred:     bool  = False
    hard_block:          bool  = False

    # Optional multi-target detail
    target_1_rr:         float = 0.0
    target_2_rr:         float = 0.0
    runner_rr:           float = 0.0

    reasons:             list[str] = field(default_factory=list)
    warnings:            list[str] = field(default_factory=list)

    scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":             self.ticker,
            "entry_price":        round(self.entry_price,      4),
            "stop_price":         round(self.stop_price,       4),
            "target_price":       round(self.target_price,     4),
            "risk_per_share":     round(self.risk_per_share,   4),
            "reward_per_share":   round(self.reward_per_share, 4),
            "reward_to_risk":     round(self.reward_to_risk,   4),
            "risk_percent":       round(self.risk_percent,     4),
            "reward_percent":     round(self.reward_percent,   4),
            "risk_reward_score":  round(self.risk_reward_score,2),
            "meets_minimum":      self.meets_minimum,
            "meets_preferred":    self.meets_preferred,
            "hard_block":         self.hard_block,
            "target_1_rr":        round(self.target_1_rr,      4),
            "target_2_rr":        round(self.target_2_rr,      4),
            "runner_rr":          round(self.runner_rr,        4),
            "reasons":            self.reasons,
            "warnings":           self.warnings,
            "scored_at":          self.scored_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class RiskRewardEngine:
    """
    Computes and scores reward-to-risk quality.

    Usage:
        engine = RiskRewardEngine(settings)
        result = engine.score(
            ticker="ABCD",
            entry_price=3.00,
            stop_price=2.85,
            target_price=3.60,
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._entry    = settings.get("entry_rules", {})
        self._risk     = settings.get("risk", {})

        self._min_rr = float(
            self._entry.get(
                "minimum_reward_to_risk",
                self._risk.get("minimum_reward_to_risk", _DEFAULT_MIN_RR),
            )
        )
        self._preferred_rr = float(
            self._entry.get(
                "preferred_reward_to_risk",
                self._risk.get("preferred_reward_to_risk", _DEFAULT_PREFERRED_RR),
            )
        )
        self._hard_block_rr = float(
            self._entry.get("min_reward_to_risk_hard_block", _DEFAULT_HARD_BLOCK)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        ticker:       str,
        entry_price:  float,
        stop_price:   float,
        target_price: float,
        target_1:     Optional[float] = None,
        target_2:     Optional[float] = None,
        runner_target:Optional[float] = None,
        side:         str = "long",
    ) -> RiskRewardResult:
        """
        Score the risk/reward quality for a planned trade.

        Args:
            ticker:        Ticker symbol.
            entry_price:   Planned entry price.
            stop_price:    Planned stop loss.
            target_price:  Primary target used for core R/R score.
            target_1:      Optional first partial target.
            target_2:      Optional second target.
            runner_target: Optional runner target.
            side:          "long" only for current bot design.

        Returns:
            RiskRewardResult with ratio, score, and block flags.
        """
        result = RiskRewardResult(
            ticker       = ticker,
            entry_price  = float(entry_price or 0.0),
            stop_price   = float(stop_price or 0.0),
            target_price = float(target_price or 0.0),
        )

        # ── Validate prices ───────────────────────────────────────────────────
        validation_error = self._validate_prices(
            entry_price=result.entry_price,
            stop_price=result.stop_price,
            target_price=result.target_price,
            side=side,
        )
        if validation_error:
            result.hard_block = True
            result.warnings.append(validation_error)
            result.risk_reward_score = 0.0
            return result

        # ── Compute core R/R ──────────────────────────────────────────────────
        risk, reward, rr = _compute_rr(
            entry_price  = result.entry_price,
            stop_price   = result.stop_price,
            target_price = result.target_price,
            side         = side,
        )

        result.risk_per_share   = risk
        result.reward_per_share = reward
        result.reward_to_risk   = rr

        result.risk_percent = _pct_distance(
            result.entry_price, result.stop_price
        )
        result.reward_percent = _pct_distance(
            result.entry_price, result.target_price
        )

        # ── Optional multi-target R/R values ──────────────────────────────────
        result.target_1_rr = self._target_rr(
            result.entry_price, result.stop_price, target_1, side
        )
        result.target_2_rr = self._target_rr(
            result.entry_price, result.stop_price, target_2, side
        )
        result.runner_rr = self._target_rr(
            result.entry_price, result.stop_price, runner_target, side
        )

        # ── Threshold flags ───────────────────────────────────────────────────
        result.meets_minimum   = rr >= self._min_rr
        result.meets_preferred = rr >= self._preferred_rr

        if rr < self._hard_block_rr:
            result.hard_block = True
            result.warnings.append(
                f"Reward:risk {rr:.2f}:1 is below hard block minimum "
                f"{self._hard_block_rr:.2f}:1"
            )
        elif rr < self._min_rr:
            result.warnings.append(
                f"Reward:risk {rr:.2f}:1 is below required minimum "
                f"{self._min_rr:.2f}:1"
            )
        elif rr >= self._preferred_rr:
            result.reasons.append(
                f"Reward:risk {rr:.2f}:1 meets preferred "
                f"{self._preferred_rr:.2f}:1"
            )
        else:
            result.reasons.append(
                f"Reward:risk {rr:.2f}:1 meets minimum "
                f"{self._min_rr:.2f}:1"
            )

        # ── Score ─────────────────────────────────────────────────────────────
        result.risk_reward_score = self._score_ratio(rr)

        # ── Extra context warnings ────────────────────────────────────────────
        if result.risk_percent > 8.0:
            result.warnings.append(
                f"Stop is {result.risk_percent:.2f}% from entry — risk is wide"
            )

        if result.reward_percent < 5.0:
            result.warnings.append(
                f"Primary target is only {result.reward_percent:.2f}% from entry"
            )

        log.debug(
            "[risk_reward] %s rr=%.2f score=%.1f hard_block=%s",
            ticker, result.reward_to_risk,
            result.risk_reward_score, result.hard_block,
        )
        return result

    def passes(
        self,
        entry_price: float,
        stop_price: float,
        target_price: float,
        side: str = "long",
    ) -> tuple[bool, str]:
        """
        Convenience check for callers that only need pass/fail.

        Returns:
            (True, "ok") or (False, reason)
        """
        validation_error = self._validate_prices(entry_price, stop_price, target_price, side)
        if validation_error:
            return False, validation_error

        _risk, _reward, rr = _compute_rr(entry_price, stop_price, target_price, side)

        if rr < self._min_rr:
            return False, f"reward:risk {rr:.2f}:1 below minimum {self._min_rr:.2f}:1"

        return True, "ok"

    # ── Internal scoring ──────────────────────────────────────────────────────

    def _score_ratio(self, rr: float) -> float:
        """
        Convert a reward-to-risk ratio into a 0–100 score.

        Scoring:
          Below hard block → 0–40
          Hard block to minimum → 40–70
          Minimum to preferred → 70–90
          Preferred+ → 90–100
        """
        if rr <= 0:
            return 0.0

        if rr < self._hard_block_rr:
            return max(0.0, min((rr / self._hard_block_rr) * 40, 40))

        if rr < self._min_rr:
            span = self._min_rr - self._hard_block_rr
            if span <= 0:
                return 60.0
            progress = (rr - self._hard_block_rr) / span
            return 40 + progress * 30

        if rr < self._preferred_rr:
            span = self._preferred_rr - self._min_rr
            if span <= 0:
                return 80.0
            progress = (rr - self._min_rr) / span
            return 70 + progress * 20

        # Preferred or better.  5:1 = 90, 8:1+ = 100
        extra = min((rr - self._preferred_rr) / 3.0, 1.0)
        return 90 + extra * 10

    def _validate_prices(
        self,
        entry_price:  float,
        stop_price:   float,
        target_price: float,
        side:         str,
    ) -> Optional[str]:
        """Validate that prices make sense for the trade side."""
        if entry_price <= 0:
            return "entry price is invalid"
        if stop_price <= 0:
            return "stop price is invalid"
        if target_price <= 0:
            return "target price is invalid"

        if side != "long":
            return "only long trades are supported by current bot design"

        if stop_price >= entry_price:
            return "long trade stop must be below entry"

        if target_price <= entry_price:
            return "long trade target must be above entry"

        return None

    def _target_rr(
        self,
        entry_price: float,
        stop_price:  float,
        target:      Optional[float],
        side:        str,
    ) -> float:
        """Compute R/R for an optional target."""
        if not target or target <= 0:
            return 0.0

        validation_error = self._validate_prices(entry_price, stop_price, target, side)
        if validation_error:
            return 0.0

        _risk, _reward, rr = _compute_rr(entry_price, stop_price, target, side)
        return rr


# ── Standalone helpers ────────────────────────────────────────────────────────

def compute_reward_to_risk(
    entry_price:  float,
    stop_price:   float,
    target_price: float,
    side:         str = "long",
) -> float:
    """
    Standalone helper to compute reward:risk ratio.
    Returns 0.0 when inputs are invalid.
    """
    try:
        _risk, _reward, rr = _compute_rr(entry_price, stop_price, target_price, side)
        return rr
    except Exception:
        return 0.0


def _compute_rr(
    entry_price:  float,
    stop_price:   float,
    target_price: float,
    side:         str = "long",
) -> tuple[float, float, float]:
    """
    Compute risk per share, reward per share, and reward:risk ratio.
    """
    if side != "long":
        return 0.0, 0.0, 0.0

    risk = entry_price - stop_price
    reward = target_price - entry_price

    if risk <= 0 or reward <= 0:
        return 0.0, 0.0, 0.0

    rr = reward / risk
    return round(risk, 4), round(reward, 4), round(rr, 4)


def _pct_distance(price_a: float, price_b: float) -> float:
    """
    Percentage distance between two prices relative to price_a.
    """
    if price_a <= 0:
        return 0.0
    return abs(price_b - price_a) / price_a * 100
