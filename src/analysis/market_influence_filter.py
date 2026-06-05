"""
src/analysis/market_influence_filter.py — Broad market influence filter
Checks whether the broader market is supporting or working against a
candidate trade.

A strong individual setup can still fail if the overall market is weak,
risk-off, or pulling against the ticker.  This module gives the bot a
market-context filter before the final trade quality gate.

Responsibilities:
  - Analyze broad market direction using SPY / QQQ / IWM data
  - Detect risk-on vs risk-off conditions
  - Detect whether indexes are above/below VWAP
  - Detect whether indexes are trending bullish, bearish, or mixed
  - Penalize long setups when the market is weak
  - Provide a market influence score for probability/trade quality logic
  - Provide hard warnings when market conditions are hostile

Design rules:
  - Market influence is usually a scoring/warning factor, not an automatic block
  - Severe risk-off conditions can set block_new_longs=True
  - This file does not place trades
  - This file does not approve trades
  - trade_quality_gate.py still makes the final buy/no-buy decision
  - Missing market index data returns a neutral result, not a crash

Market labels:
  "strong_bullish" — SPY/QQQ/IWM mostly bullish and above VWAP
  "bullish"        — market context supports longs
  "mixed"          — no strong edge either way
  "weak"           — market context is poor for longs
  "risk_off"       — broad market is actively hostile
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from models import IndicatorSnapshot

log = logging.getLogger(__name__)

# Default market tickers used for influence checks
_DEFAULT_MARKET_TICKERS = ["SPY", "QQQ", "IWM"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MarketInfluenceResult:
    """
    Broad market influence assessment.
    Consumed by probability_engine.py, trade_quality_gate.py, and dashboard.
    """
    market_label:          str   = "mixed"
    market_score:          float = 50.0   # 0–100
    long_supportive:       bool  = False
    block_new_longs:       bool  = False

    # Index-specific context
    index_scores:          dict  = field(default_factory=dict)
    index_states:          dict  = field(default_factory=dict)

    # Summary flags
    spy_bullish:           bool  = False
    qqq_bullish:           bool  = False
    iwm_bullish:           bool  = False
    majority_bullish:      bool  = False
    majority_bearish:      bool  = False

    # Price / VWAP context
    indexes_above_vwap:    int   = 0
    indexes_below_vwap:    int   = 0

    reasons:               list[str] = field(default_factory=list)
    warnings:              list[str] = field(default_factory=list)

    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "market_label":       self.market_label,
            "market_score":       round(self.market_score, 2),
            "long_supportive":    self.long_supportive,
            "block_new_longs":    self.block_new_longs,
            "index_scores":       self.index_scores,
            "index_states":       self.index_states,
            "spy_bullish":        self.spy_bullish,
            "qqq_bullish":        self.qqq_bullish,
            "iwm_bullish":        self.iwm_bullish,
            "majority_bullish":   self.majority_bullish,
            "majority_bearish":   self.majority_bearish,
            "indexes_above_vwap": self.indexes_above_vwap,
            "indexes_below_vwap": self.indexes_below_vwap,
            "reasons":            self.reasons,
            "warnings":           self.warnings,
            "analyzed_at":        self.analyzed_at,
        }


# ── Filter ────────────────────────────────────────────────────────────────────

class MarketInfluenceFilter:
    """
    Analyzes broad market context and returns a MarketInfluenceResult.

    Usage:
        filter_ = MarketInfluenceFilter(settings)
        result = filter_.analyze({
            "SPY": spy_indicators,
            "QQQ": qqq_indicators,
            "IWM": iwm_indicators,
        })
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._scanner  = settings.get("scanner", {})
        self._entry    = settings.get("entry_rules", {})

        self._market_tickers = list(
            settings.get("market_influence", {}).get(
                "market_tickers", _DEFAULT_MARKET_TICKERS
            )
        )
        self._enabled = bool(
            settings.get("market_influence", {}).get("enabled", True)
        )
        self._block_risk_off = bool(
            settings.get("market_influence", {}).get("block_new_longs_risk_off", True)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        market_indicators: Optional[dict[str, IndicatorSnapshot]] = None,
        market_data: Optional[dict[str, object]] = None,
    ) -> MarketInfluenceResult:
        """
        Analyze broad market influence.

        Args:
            market_indicators: Dict mapping ticker → IndicatorSnapshot.
            market_data:       Optional dict mapping ticker → TickerData-like object.
                               Used when indicator snapshots are nested on data objects.

        Returns:
            MarketInfluenceResult.
        """
        result = MarketInfluenceResult()

        if not self._enabled:
            result.reasons.append("Market influence filter disabled")
            return result

        indicators = self._normalize_market_inputs(market_indicators, market_data)

        if not indicators:
            result.reasons.append("Market indicator data unavailable — neutral market score")
            return result

        scores: dict[str, float] = {}
        states: dict[str, dict] = {}

        for ticker in self._market_tickers:
            snap = indicators.get(ticker)
            if not snap:
                continue

            index_score, state = self._score_index(ticker, snap)
            scores[ticker] = index_score
            states[ticker] = state

        if not scores:
            result.reasons.append("No supported market index snapshots found — neutral score")
            return result

        result.index_scores = {k: round(v, 2) for k, v in scores.items()}
        result.index_states = states

        # ── Aggregate broad market context ────────────────────────────────────
        bullish_count = sum(1 for s in states.values() if s.get("bullish"))
        bearish_count = sum(1 for s in states.values() if s.get("bearish"))

        result.indexes_above_vwap = sum(1 for s in states.values() if s.get("above_vwap"))
        result.indexes_below_vwap = sum(1 for s in states.values() if s.get("below_vwap"))

        result.spy_bullish = bool(states.get("SPY", {}).get("bullish", False))
        result.qqq_bullish = bool(states.get("QQQ", {}).get("bullish", False))
        result.iwm_bullish = bool(states.get("IWM", {}).get("bullish", False))

        total_indexes = len(states)
        result.majority_bullish = bullish_count >= max(1, (total_indexes // 2) + 1)
        result.majority_bearish = bearish_count >= max(1, (total_indexes // 2) + 1)

        result.market_score = round(sum(scores.values()) / len(scores), 2)
        result.market_label = self._label_market(result.market_score, result)

        result.long_supportive = result.market_label in ("strong_bullish", "bullish")

        if result.market_label == "risk_off" and self._block_risk_off:
            result.block_new_longs = True
            result.warnings.append("Risk-off market detected — blocking new long entries")

        # ── Reasons / warnings ────────────────────────────────────────────────
        if result.majority_bullish:
            result.reasons.append("Majority of market indexes are bullish")
        if result.majority_bearish:
            result.warnings.append("Majority of market indexes are bearish")

        if result.indexes_above_vwap >= 2:
            result.reasons.append("Most market indexes are above VWAP")
        elif result.indexes_below_vwap >= 2:
            result.warnings.append("Most market indexes are below VWAP")

        result.reasons.append(
            f"Market influence score: {result.market_score:.1f} ({result.market_label})"
        )

        log.debug(
            "[market_influence] label=%s score=%.1f block=%s",
            result.market_label, result.market_score, result.block_new_longs,
        )
        return result

    def score_for_probability(
        self,
        result: MarketInfluenceResult,
        max_points: float = 5.0,
    ) -> float:
        """
        Convert market influence result into a probability-score component.

        Args:
            result:     MarketInfluenceResult.
            max_points: Maximum points to award.

        Returns:
            Score contribution from 0 to max_points.
        """
        if not result:
            return max_points * 0.5

        if result.market_label == "strong_bullish":
            return max_points
        if result.market_label == "bullish":
            return max_points * 0.8
        if result.market_label == "mixed":
            return max_points * 0.5
        if result.market_label == "weak":
            return max_points * 0.2
        return 0.0

    # ── Input normalization ───────────────────────────────────────────────────

    def _normalize_market_inputs(
        self,
        market_indicators: Optional[dict[str, IndicatorSnapshot]],
        market_data: Optional[dict[str, object]],
    ) -> dict[str, IndicatorSnapshot]:
        """
        Normalize either direct IndicatorSnapshot input or TickerData-like input.
        """
        if market_indicators:
            return {
                ticker.upper(): snap
                for ticker, snap in market_indicators.items()
                if snap is not None
            }

        if not market_data:
            return {}

        normalized: dict[str, IndicatorSnapshot] = {}
        for ticker, data in market_data.items():
            snap = getattr(data, "indicators", None)
            if snap:
                normalized[ticker.upper()] = snap

        return normalized

    # ── Per-index scoring ─────────────────────────────────────────────────────

    def _score_index(
        self,
        ticker: str,
        snap: IndicatorSnapshot,
    ) -> tuple[float, dict]:
        """
        Score a single market index / ETF from 0–100.
        """
        score = 50.0
        state = {
            "bullish": False,
            "bearish": False,
            "above_vwap": False,
            "below_vwap": False,
            "ma_trend": snap.ma_trend,
            "price_vs_vwap": snap.price_vs_vwap,
            "rsi": snap.rsi,
            "macd_bullish": bool(snap.macd and snap.macd.bullish),
            "relative_volume": snap.relative_volume,
        }

        # VWAP context
        if snap.price_vs_vwap == "above":
            score += 15
            state["above_vwap"] = True
        elif snap.price_vs_vwap == "below":
            score -= 15
            state["below_vwap"] = True

        # MA trend
        if snap.ma_trend == "bullish":
            score += 15
        elif snap.ma_trend == "bearish":
            score -= 15

        # MACD
        if snap.macd and snap.macd.bullish:
            score += 10
        elif snap.macd and snap.macd.bearish_crossover:
            score -= 10

        # Candle strength
        if snap.candle_strength >= 0.4:
            score += 8
        elif snap.candle_strength <= -0.4:
            score -= 8

        # Structure
        if snap.higher_lows:
            score += 7
        if snap.lower_highs:
            score -= 7

        # RSI context
        if snap.rsi is not None:
            if 45 <= snap.rsi <= 70:
                score += 5
            elif snap.rsi < 35:
                score -= 5
            elif snap.rsi > 75:
                score -= 3

        score = max(0.0, min(score, 100.0))

        state["score"] = round(score, 2)
        state["bullish"] = score >= 60
        state["bearish"] = score <= 40

        return score, state

    # ── Labeling ──────────────────────────────────────────────────────────────

    def _label_market(
        self,
        market_score: float,
        result: MarketInfluenceResult,
    ) -> str:
        """
        Convert market score and breadth into a label.
        """
        if market_score >= 75 and result.majority_bullish:
            return "strong_bullish"
        if market_score >= 60:
            return "bullish"
        if market_score >= 45:
            return "mixed"
        if market_score >= 30:
            return "weak"
        return "risk_off"


# ── Convenience wrapper ───────────────────────────────────────────────────────

def analyze_market_influence(
    settings: dict,
    market_indicators: Optional[dict[str, IndicatorSnapshot]] = None,
    market_data: Optional[dict[str, object]] = None,
) -> MarketInfluenceResult:
    """
    Convenience function for bot_runner.py.
    """
    filter_ = MarketInfluenceFilter(settings)
    return filter_.analyze(
        market_indicators = market_indicators,
        market_data       = market_data,
    )
