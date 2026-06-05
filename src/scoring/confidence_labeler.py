"""
src/scoring/confidence_labeler.py — Confidence label and size adjustment helper
Converts numeric trade scores into human-readable confidence labels and
position-size adjustment percentages.

This module is intentionally small and pure.  It is used by scoring engines,
risk modules, the trade quality gate, and the dashboard so every part of the
bot uses the same confidence language.

Confidence labels:
  "elite"   — 90–100
  "strong"  — 80–89
  "decent"  — 70–79
  "weak"    — 60–69
  "reject"  — below 60

Sizing rules from bot_settings.json → risk → confidence_sizing:
  90+     → 100% size
  85–89   → 75% size
  80–84   → 50% size
  below 80 → no trade when below_80_no_trade is true

Design rules:
  - This file does not approve trades
  - This file does not place orders
  - This file does not know about setups or tickers
  - Numeric scores are clamped to 0–100 before labeling
  - trade_quality_gate.py still makes the final buy/no-buy decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from models import ConfidenceLabel

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_SIZING = {
    "enabled": True,
    "score_90_plus_size_pct": 100,
    "score_85_to_89_size_pct": 75,
    "score_80_to_84_size_pct": 50,
    "below_80_no_trade": True,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConfidenceResult:
    """
    Result of converting a numeric score into a confidence label and
    size-adjustment recommendation.
    """
    score:              float
    label:              str   = ConfidenceLabel.REJECT.value
    size_pct:           float = 0.0
    no_trade:           bool  = True
    reason:             str   = ""
    sizing_enabled:     bool  = True
    labeled_at:         str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "score":          round(self.score, 2),
            "label":          self.label,
            "size_pct":       self.size_pct,
            "no_trade":       self.no_trade,
            "reason":         self.reason,
            "sizing_enabled": self.sizing_enabled,
            "labeled_at":     self.labeled_at,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def label_score(score: float) -> str:
    """
    Convert a numeric score into a ConfidenceLabel value.

    Args:
        score: Numeric score, usually 0–100.

    Returns:
        Confidence label string.
    """
    s = _clamp_score(score)

    if s >= 90:
        return ConfidenceLabel.ELITE.value
    if s >= 80:
        return ConfidenceLabel.STRONG.value
    if s >= 70:
        return ConfidenceLabel.DECENT.value
    if s >= 60:
        return ConfidenceLabel.WEAK.value
    return ConfidenceLabel.REJECT.value


def confidence_result(score: float, settings: dict | None = None) -> ConfidenceResult:
    """
    Return the full confidence result for a score.

    Args:
        score:    Numeric score, usually final_trade_quality_score.
        settings: Full bot_settings dict.  If omitted, defaults are used.

    Returns:
        ConfidenceResult containing label, size percent, and no_trade flag.
    """
    s = _clamp_score(score)
    label = label_score(s)
    size_pct, no_trade, reason, enabled = position_size_percent(s, settings)

    return ConfidenceResult(
        score          = s,
        label          = label,
        size_pct       = size_pct,
        no_trade       = no_trade,
        reason         = reason,
        sizing_enabled = enabled,
    )


def position_size_percent(
    score: float,
    settings: dict | None = None,
) -> tuple[float, bool, str, bool]:
    """
    Convert a score into a position size percentage.

    Args:
        score:    Numeric score, usually final_trade_quality_score.
        settings: Full bot_settings dict.

    Returns:
        (size_pct, no_trade, reason, sizing_enabled)

        size_pct:
            Percentage of normal calculated position size to use.
        no_trade:
            True when the score should block trading from a sizing standpoint.
        reason:
            Human-readable explanation.
        sizing_enabled:
            Whether confidence sizing was enabled in config.
    """
    s = _clamp_score(score)
    cfg = _sizing_cfg(settings)

    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return 100.0, False, "confidence sizing disabled — using full size", False

    if s >= 90:
        pct = float(cfg.get("score_90_plus_size_pct", 100))
        return pct, False, f"elite confidence — {pct:.0f}% size", True

    if 85 <= s < 90:
        pct = float(cfg.get("score_85_to_89_size_pct", 75))
        return pct, False, f"strong confidence — {pct:.0f}% size", True

    if 80 <= s < 85:
        pct = float(cfg.get("score_80_to_84_size_pct", 50))
        return pct, False, f"minimum confidence — {pct:.0f}% size", True

    below_80_no_trade = bool(cfg.get("below_80_no_trade", True))
    if below_80_no_trade:
        return 0.0, True, "score below 80 — no trade by confidence sizing", True

    return 25.0, False, "score below 80 — reduced size only", True


def should_allow_trade_by_score(
    score: float,
    settings: dict | None = None,
) -> tuple[bool, str]:
    """
    Convenience helper used by trade_quality_gate.py.

    Returns:
        (True, "ok") when confidence sizing allows trade.
        (False, reason) when score should block trade.
    """
    size_pct, no_trade, reason, _enabled = position_size_percent(score, settings)
    if no_trade or size_pct <= 0:
        return False, reason
    return True, "confidence score allows trade"


def apply_size_reduction(base_shares: int, score: float,
                         settings: dict | None = None) -> tuple[int, float, str]:
    """
    Apply confidence-based size reduction to a share count.

    Args:
        base_shares: Shares calculated by position_sizer.py before confidence adjustment.
        score:       Numeric confidence / final trade quality score.
        settings:    Full bot_settings dict.

    Returns:
        (adjusted_shares, size_pct, reason)
    """
    if base_shares <= 0:
        return 0, 0.0, "base share count is zero"

    size_pct, no_trade, reason, _enabled = position_size_percent(score, settings)

    if no_trade or size_pct <= 0:
        return 0, size_pct, reason

    adjusted = int(base_shares * (size_pct / 100.0))
    adjusted = max(adjusted, 1) if base_shares > 0 else 0

    return adjusted, size_pct, reason


def score_bucket(score: float) -> str:
    """
    Return a broader score bucket useful for dashboard grouping.

    Buckets:
      "tradeable"    — 80+
      "watch"        — 70–79
      "weak_watch"   — 60–69
      "reject"       — below 60
    """
    s = _clamp_score(score)

    if s >= 80:
        return "tradeable"
    if s >= 70:
        return "watch"
    if s >= 60:
        return "weak_watch"
    return "reject"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sizing_cfg(settings: dict | None) -> dict:
    """
    Extract risk.confidence_sizing from bot_settings.json.
    Falls back to safe defaults when settings are missing.
    """
    if not settings:
        return dict(_DEFAULT_SIZING)

    risk_cfg = settings.get("risk", {})
    sizing = risk_cfg.get("confidence_sizing", {})

    return {**_DEFAULT_SIZING, **sizing}


def _clamp_score(score: float) -> float:
    """Clamp score to the 0–100 range."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(s, 100.0))
