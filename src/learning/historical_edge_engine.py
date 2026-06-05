"""
src/learning/historical_edge_engine.py — Historical setup edge scoring
Calculates whether a setup has been working recently based on closed
trade history.

The bot should not treat every setup equally forever.  If VWAP reclaims
are working and breakouts are failing, the bot should know that.  This
module produces a historical_edge_score that can feed probability_engine.py
and trade_quality_gate.py.

Responsibilities:
  - Read closed trade JSON files
  - Filter by setup type, ticker, or source when available
  - Calculate recent win rate
  - Calculate average P/L
  - Calculate average percent return
  - Calculate profit factor
  - Detect whether edge is improving or weakening
  - Return a 0–100 historical_edge_score
  - Return neutral score when not enough data exists

Design rules:
  - This file only reads closed trade records
  - This file does not approve trades
  - This file does not place orders
  - Small sample sizes should not over-influence the bot
  - Missing history returns neutral score, not rejection
  - trade_quality_gate.py still makes the final buy/no-buy decision
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_CLOSED_DIR = Path("data") / "trades" / "closed"
_NEUTRAL_EDGE_SCORE = 50.0
_MIN_TRADES_FOR_EDGE = 5


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class HistoricalEdgeResult:
    """
    Historical edge score for a setup/ticker/source.
    Consumed by probability_engine.py and trade_quality_gate.py.
    """
    historical_edge_score: float = _NEUTRAL_EDGE_SCORE
    edge_label:            str   = "neutral"  # strong|positive|neutral|weak|bad

    setup_type:            str   = ""
    ticker:                str   = ""
    source:                str   = ""

    trade_count:           int   = 0
    wins:                  int   = 0
    losses:                int   = 0
    breakevens:            int   = 0

    win_rate:              float = 0.0
    total_pl:              float = 0.0
    average_pl:            float = 0.0
    average_return_pct:    float = 0.0
    profit_factor:         float = 0.0

    recent_trade_count:    int   = 0
    recent_win_rate:       float = 0.0
    recent_average_pl:     float = 0.0
    edge_trend:            str   = "unknown"  # improving|weakening|stable|unknown

    enough_data:           bool  = False

    reasons:               list[str] = field(default_factory=list)
    warnings:              list[str] = field(default_factory=list)

    calculated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "historical_edge_score": round(self.historical_edge_score, 2),
            "edge_label":            self.edge_label,
            "setup_type":            self.setup_type,
            "ticker":                self.ticker,
            "source":                self.source,
            "trade_count":           self.trade_count,
            "wins":                  self.wins,
            "losses":                self.losses,
            "breakevens":            self.breakevens,
            "win_rate":              round(self.win_rate, 2),
            "total_pl":              round(self.total_pl, 2),
            "average_pl":            round(self.average_pl, 2),
            "average_return_pct":    round(self.average_return_pct, 4),
            "profit_factor":         round(self.profit_factor, 4),
            "recent_trade_count":    self.recent_trade_count,
            "recent_win_rate":       round(self.recent_win_rate, 2),
            "recent_average_pl":     round(self.recent_average_pl, 2),
            "edge_trend":            self.edge_trend,
            "enough_data":           self.enough_data,
            "reasons":               self.reasons,
            "warnings":              self.warnings,
            "calculated_at":         self.calculated_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class HistoricalEdgeEngine:
    """
    Calculates historical edge from closed trade files.

    Usage:
        engine = HistoricalEdgeEngine(settings)
        result = engine.score(setup_type="vwap_reclaim")
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._learning = settings.get("learning", {})
        paths = settings.get("paths", {})

        trades_dir = Path(paths.get("trades_dir", str(Path("data") / "trades")))
        self._closed_dir = Path(
            paths.get("closed_trades_dir", str(trades_dir / "closed"))
        )

        self._closed_dir.mkdir(parents=True, exist_ok=True)

        self._lookback = int(self._learning.get("historical_edge_lookback", 50))
        self._recent_lookback = int(self._learning.get("recent_edge_lookback", 10))
        self._min_trades = int(self._learning.get("min_trades_for_historical_edge", _MIN_TRADES_FOR_EDGE))

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        setup_type: str = "",
        ticker: str = "",
        source: str = "",
        lookback: Optional[int] = None,
    ) -> HistoricalEdgeResult:
        """
        Calculate historical edge score.

        Args:
            setup_type: Optional setup type filter.
            ticker:     Optional ticker filter.
            source:     Optional scanner/manual source filter.
            lookback:   Optional max number of recent closed trades.

        Returns:
            HistoricalEdgeResult.
        """
        result = HistoricalEdgeResult(
            setup_type = setup_type,
            ticker     = ticker.upper() if ticker else "",
            source     = source,
        )

        trades = self._load_trades(limit=lookback or self._lookback)
        trades = self._filter_trades(trades, setup_type, ticker, source)

        result.trade_count = len(trades)

        if not trades:
            result.historical_edge_score = _NEUTRAL_EDGE_SCORE
            result.edge_label = "neutral"
            result.warnings.append("No historical trades found — neutral edge score")
            return result

        self._calculate_metrics(result, trades)
        self._calculate_recent_metrics(result, trades)

        result.enough_data = result.trade_count >= self._min_trades

        if not result.enough_data:
            result.historical_edge_score = self._small_sample_score(result)
            result.edge_label = self._label(result.historical_edge_score)
            result.warnings.append(
                f"Only {result.trade_count} trade(s) found — using dampened edge score"
            )
            return result

        result.historical_edge_score = self._score_edge(result)
        result.edge_label = self._label(result.historical_edge_score)
        result.edge_trend = self._edge_trend(result)

        self._build_reasons(result)

        log.debug(
            "[historical_edge] setup=%s ticker=%s score=%.1f trades=%d win_rate=%.1f%%",
            setup_type or "all",
            ticker or "all",
            result.historical_edge_score,
            result.trade_count,
            result.win_rate,
        )
        return result

    def score_for_setup(self, setup_type: str) -> HistoricalEdgeResult:
        """Convenience wrapper for setup-only edge."""
        return self.score(setup_type=setup_type)

    def score_for_ticker(self, ticker: str) -> HistoricalEdgeResult:
        """Convenience wrapper for ticker-only edge."""
        return self.score(ticker=ticker)

    # ── Load / filter ─────────────────────────────────────────────────────────

    def _load_trades(self, limit: int) -> list[dict]:
        """
        Load closed trades from disk newest first.
        """
        files = sorted(
            self._closed_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        trades: list[dict] = []
        for path in files:
            if len(trades) >= limit:
                break

            data = _read_json(path)
            if not data:
                continue
            if str(data.get("status", "")).lower() != "closed":
                continue

            data["_file_path"] = str(path)
            trades.append(data)

        return trades

    @staticmethod
    def _filter_trades(
        trades: list[dict],
        setup_type: str,
        ticker: str,
        source: str,
    ) -> list[dict]:
        """
        Filter trades by setup type, ticker, and source.
        """
        result = trades

        if setup_type:
            wanted = setup_type.lower()
            result = [
                t for t in result
                if str(t.get("setup_type", t.get("setup", ""))).lower() == wanted
            ]

        if ticker:
            wanted_ticker = ticker.upper()
            result = [
                t for t in result
                if str(t.get("ticker", "")).upper() == wanted_ticker
            ]

        if source:
            wanted_source = source.lower()
            result = [
                t for t in result
                if str(t.get("source", t.get("candidate_source", ""))).lower() == wanted_source
            ]

        return result

    # ── Metric calculations ───────────────────────────────────────────────────

    def _calculate_metrics(
        self,
        result: HistoricalEdgeResult,
        trades: list[dict],
    ) -> None:
        """
        Calculate all-time metrics over filtered trades.
        """
        pls = [_realized_pl(t) for t in trades]
        returns = [_return_pct(t) for t in trades]

        wins = [p for p in pls if p > 0]
        losses = [p for p in pls if p < 0]
        breakevens = [p for p in pls if p == 0]

        result.wins = len(wins)
        result.losses = len(losses)
        result.breakevens = len(breakevens)
        result.win_rate = result.wins / result.trade_count * 100 if result.trade_count else 0.0
        result.total_pl = sum(pls)
        result.average_pl = sum(pls) / len(pls) if pls else 0.0
        result.average_return_pct = sum(returns) / len(returns) if returns else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit

    def _calculate_recent_metrics(
        self,
        result: HistoricalEdgeResult,
        trades: list[dict],
    ) -> None:
        """
        Calculate recent lookback metrics.
        """
        recent = trades[:self._recent_lookback]
        result.recent_trade_count = len(recent)

        if not recent:
            return

        pls = [_realized_pl(t) for t in recent]
        wins = [p for p in pls if p > 0]

        result.recent_win_rate = len(wins) / len(recent) * 100
        result.recent_average_pl = sum(pls) / len(pls)

    # ── Edge scoring ──────────────────────────────────────────────────────────

    def _score_edge(self, result: HistoricalEdgeResult) -> float:
        """
        Convert historical metrics into a 0–100 edge score.

        Components:
          Win rate:          35 pts
          Average P/L:       25 pts
          Profit factor:     20 pts
          Recent trend:      15 pts
          Sample size:        5 pts
        """
        score = 0.0

        # Win rate (35 pts)
        score += min(result.win_rate / 70.0 * 35.0, 35.0)

        # Average P/L (25 pts)
        if result.average_pl > 0:
            score += min(result.average_pl / 25.0 * 25.0, 25.0)
        else:
            score += max(result.average_pl / 25.0 * 20.0, -15.0)

        # Profit factor (20 pts)
        if result.profit_factor >= 2.0:
            score += 20
        elif result.profit_factor >= 1.5:
            score += 16
        elif result.profit_factor >= 1.0:
            score += 10
        elif result.profit_factor > 0:
            score += 4

        # Recent trend (15 pts)
        if result.recent_average_pl > result.average_pl and result.recent_win_rate >= result.win_rate:
            score += 15
        elif result.recent_win_rate >= result.win_rate:
            score += 10
        elif result.recent_average_pl < 0:
            score -= 8
        else:
            score += 5

        # Sample size (5 pts)
        if result.trade_count >= self._min_trades * 3:
            score += 5
        elif result.trade_count >= self._min_trades:
            score += 3

        return round(max(0.0, min(score, 100.0)), 2)

    def _small_sample_score(self, result: HistoricalEdgeResult) -> float:
        """
        Dampened score for small samples.
        Keeps result near neutral until enough trades exist.
        """
        if result.trade_count <= 0:
            return _NEUTRAL_EDGE_SCORE

        raw = 50.0

        if result.win_rate >= 70:
            raw += 10
        elif result.win_rate >= 50:
            raw += 5
        elif result.win_rate < 35:
            raw -= 8

        if result.average_pl > 0:
            raw += min(result.average_pl / 25.0 * 8, 8)
        elif result.average_pl < 0:
            raw += max(result.average_pl / 25.0 * 8, -8)

        return round(max(35.0, min(raw, 65.0)), 2)

    @staticmethod
    def _label(score: float) -> str:
        """Label historical edge score."""
        if score >= 80:
            return "strong"
        if score >= 65:
            return "positive"
        if score >= 45:
            return "neutral"
        if score >= 30:
            return "weak"
        return "bad"

    @staticmethod
    def _edge_trend(result: HistoricalEdgeResult) -> str:
        """Classify whether recent edge is improving or weakening."""
        if result.recent_trade_count < 3:
            return "unknown"

        if (
            result.recent_win_rate > result.win_rate + 10
            and result.recent_average_pl > result.average_pl
        ):
            return "improving"

        if (
            result.recent_win_rate < result.win_rate - 10
            or result.recent_average_pl < 0
        ):
            return "weakening"

        return "stable"

    @staticmethod
    def _build_reasons(result: HistoricalEdgeResult) -> None:
        """Add human-readable reasons/warnings."""
        result.reasons.append(
            f"Historical edge based on {result.trade_count} closed trade(s)"
        )
        result.reasons.append(f"Win rate: {result.win_rate:.1f}%")
        result.reasons.append(f"Average P/L: ${result.average_pl:.2f}")

        if result.edge_label in ("strong", "positive"):
            result.reasons.append(f"Historical edge is {result.edge_label}")
        elif result.edge_label in ("weak", "bad"):
            result.warnings.append(f"Historical edge is {result.edge_label}")

        if result.edge_trend == "improving":
            result.reasons.append("Recent edge is improving")
        elif result.edge_trend == "weakening":
            result.warnings.append("Recent edge is weakening")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def score_historical_edge(
    settings: dict,
    setup_type: str = "",
    ticker: str = "",
    source: str = "",
    lookback: Optional[int] = None,
) -> HistoricalEdgeResult:
    """
    Convenience function for probability_engine.py / bot_runner.py.
    """
    engine = HistoricalEdgeEngine(settings)
    return engine.score(
        setup_type = setup_type,
        ticker     = ticker,
        source     = source,
        lookback   = lookback,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[historical_edge] Failed to read %s: %s", path, exc)
        return {}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _realized_pl(trade: dict) -> float:
    value = trade.get("realized_pl", None)
    if value is not None:
        return _safe_float(value)

    entry = _safe_float(trade.get("entry_price", 0.0))
    exit_price = _safe_float(trade.get("exit_price", 0.0))
    qty = _safe_float(trade.get("exit_quantity", trade.get("quantity", 0.0)))

    if entry > 0 and exit_price > 0 and qty > 0:
        return (exit_price - entry) * qty

    return 0.0


def _return_pct(trade: dict) -> float:
    value = trade.get("realized_pl_percent", None)
    if value is not None:
        return _safe_float(value)

    entry = _safe_float(trade.get("entry_price", 0.0))
    exit_price = _safe_float(trade.get("exit_price", 0.0))

    if entry > 0 and exit_price > 0:
        return (exit_price - entry) / entry * 100

    return 0.0
