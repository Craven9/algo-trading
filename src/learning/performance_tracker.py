"""
src/learning/performance_tracker.py — Trade performance tracking
Reads closed trade files and calculates performance statistics for the bot.

The bot needs to know what is working and what is failing.  This module
turns closed trade JSON files into clean performance metrics for:
  - Dashboard display
  - Setup review
  - Probability engine historical edge
  - Backtest review
  - Learning system

Responsibilities:
  - Read closed trade JSON files
  - Calculate total realized P/L
  - Calculate win rate
  - Calculate average win and average loss
  - Calculate best/worst trade
  - Calculate best/worst setup type
  - Calculate daily and weekly P/L
  - Calculate scanner vs manual performance if source exists
  - Return clean PerformanceSummary object
  - Never place trades or change broker state

Design rules:
  - This file only reads trade records
  - This file does not approve trades
  - This file does not place orders
  - Missing or incomplete closed trades are skipped safely
  - All metrics are based on local closed trade JSON files
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


# ── Setup stats ───────────────────────────────────────────────────────────────

@dataclass
class SetupPerformance:
    """
    Performance metrics for one setup type.
    """
    setup_type:       str
    trade_count:      int   = 0
    wins:             int   = 0
    losses:           int   = 0
    breakevens:       int   = 0
    total_pl:         float = 0.0
    win_rate:         float = 0.0
    avg_win:          float = 0.0
    avg_loss:         float = 0.0
    best_trade_pl:    float = 0.0
    worst_trade_pl:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "setup_type":     self.setup_type,
            "trade_count":    self.trade_count,
            "wins":           self.wins,
            "losses":         self.losses,
            "breakevens":     self.breakevens,
            "total_pl":       round(self.total_pl, 2),
            "win_rate":       round(self.win_rate, 2),
            "avg_win":        round(self.avg_win, 2),
            "avg_loss":       round(self.avg_loss, 2),
            "best_trade_pl":  round(self.best_trade_pl, 2),
            "worst_trade_pl": round(self.worst_trade_pl, 2),
        }


# ── Summary dataclass ─────────────────────────────────────────────────────────

@dataclass
class PerformanceSummary:
    """
    Overall closed-trade performance summary.
    """
    trade_count:           int   = 0
    wins:                  int   = 0
    losses:                int   = 0
    breakevens:            int   = 0

    total_realized_pl:     float = 0.0
    win_rate:              float = 0.0
    average_win:           float = 0.0
    average_loss:          float = 0.0
    average_trade_pl:      float = 0.0
    profit_factor:         float = 0.0

    best_trade:            dict  = field(default_factory=dict)
    worst_trade:           dict  = field(default_factory=dict)

    best_setup:            Optional[str] = None
    worst_setup:           Optional[str] = None
    setup_performance:     dict = field(default_factory=dict)

    daily_pl:              dict = field(default_factory=dict)
    weekly_pl:             dict = field(default_factory=dict)

    scanner_performance:   dict = field(default_factory=dict)
    manual_performance:    dict = field(default_factory=dict)

    source_file_count:     int = 0
    skipped_file_count:    int = 0

    calculated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "trade_count":         self.trade_count,
            "wins":                self.wins,
            "losses":              self.losses,
            "breakevens":          self.breakevens,
            "total_realized_pl":   round(self.total_realized_pl, 2),
            "win_rate":            round(self.win_rate, 2),
            "average_win":         round(self.average_win, 2),
            "average_loss":        round(self.average_loss, 2),
            "average_trade_pl":    round(self.average_trade_pl, 2),
            "profit_factor":       round(self.profit_factor, 4),
            "best_trade":          self.best_trade,
            "worst_trade":         self.worst_trade,
            "best_setup":          self.best_setup,
            "worst_setup":         self.worst_setup,
            "setup_performance":   self.setup_performance,
            "daily_pl":            self.daily_pl,
            "weekly_pl":           self.weekly_pl,
            "scanner_performance": self.scanner_performance,
            "manual_performance":  self.manual_performance,
            "source_file_count":   self.source_file_count,
            "skipped_file_count":  self.skipped_file_count,
            "calculated_at":       self.calculated_at,
        }


# ── Tracker ──────────────────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Reads closed trade files and calculates performance.

    Usage:
        tracker = PerformanceTracker(settings)
        summary = tracker.calculate()
    """

    def __init__(self, settings: dict):
        self._settings = settings
        paths = settings.get("paths", {})

        trades_dir = Path(paths.get("trades_dir", str(Path("data") / "trades")))
        self._closed_dir = Path(
            paths.get("closed_trades_dir", str(trades_dir / "closed"))
        )

        self._closed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(self, limit: Optional[int] = None) -> PerformanceSummary:
        """
        Calculate performance from closed trade files.

        Args:
            limit: Optional number of most recent trades to include.

        Returns:
            PerformanceSummary.
        """
        trades, skipped = self._load_closed_trades(limit=limit)

        summary = PerformanceSummary(
            source_file_count  = len(trades),
            skipped_file_count = skipped,
        )

        if not trades:
            return summary

        pls = [_realized_pl(t) for t in trades]
        wins = [p for p in pls if p > 0]
        losses = [p for p in pls if p < 0]
        breakevens = [p for p in pls if p == 0]

        summary.trade_count = len(trades)
        summary.wins = len(wins)
        summary.losses = len(losses)
        summary.breakevens = len(breakevens)
        summary.total_realized_pl = round(sum(pls), 2)
        summary.win_rate = (summary.wins / summary.trade_count * 100) if summary.trade_count else 0.0
        summary.average_win = sum(wins) / len(wins) if wins else 0.0
        summary.average_loss = sum(losses) / len(losses) if losses else 0.0
        summary.average_trade_pl = sum(pls) / len(pls) if pls else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        summary.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit

        summary.best_trade = self._trade_card(max(trades, key=_realized_pl))
        summary.worst_trade = self._trade_card(min(trades, key=_realized_pl))

        setup_stats = self._setup_performance(trades)
        summary.setup_performance = {
            name: stats.to_dict() for name, stats in setup_stats.items()
        }

        if setup_stats:
            summary.best_setup = max(setup_stats.values(), key=lambda s: s.total_pl).setup_type
            summary.worst_setup = min(setup_stats.values(), key=lambda s: s.total_pl).setup_type

        summary.daily_pl = self._daily_pl(trades)
        summary.weekly_pl = self._weekly_pl(trades)

        summary.scanner_performance = self._source_performance(trades, "scanner")
        summary.manual_performance = self._source_performance(trades, "manual")

        log.info(
            "[performance] trades=%d total_pl=%.2f win_rate=%.1f%%",
            summary.trade_count,
            summary.total_realized_pl,
            summary.win_rate,
        )
        return summary

    def recent_trades(self, limit: int = 20) -> list[dict]:
        """
        Return most recent closed trades as simple dashboard cards.
        """
        trades, _skipped = self._load_closed_trades(limit=limit)
        return [self._trade_card(t) for t in trades]

    def setup_edge_score(self, setup_type: str, lookback: int = 50) -> float:
        """
        Return a 0–100 historical edge score for one setup type.

        This is used by probability_engine.py / trade_quality_gate.py.
        Neutral score is 50 when there is not enough history.
        """
        setup_type = str(setup_type or "").lower()
        trades, _skipped = self._load_closed_trades(limit=lookback)

        relevant = [
            t for t in trades
            if str(t.get("setup_type", t.get("setup", ""))).lower() == setup_type
        ]

        if len(relevant) < 3:
            return 50.0

        pls = [_realized_pl(t) for t in relevant]
        wins = [p for p in pls if p > 0]
        losses = [p for p in pls if p < 0]

        win_rate = len(wins) / len(relevant) * 100
        avg_trade = sum(pls) / len(pls)

        # Simple balanced edge score:
        # win rate contributes up to 60 pts, average P/L contributes +/-40.
        score = win_rate * 0.60

        if avg_trade > 0:
            score += min(avg_trade / 25.0 * 40, 40)
        else:
            score += max(avg_trade / 25.0 * 40, -30)

        return round(max(0.0, min(score, 100.0)), 2)

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_closed_trades(self, limit: Optional[int] = None) -> tuple[list[dict], int]:
        """
        Load closed trade JSON files.

        Returns:
            (trades, skipped_count)
        """
        files = sorted(
            self._closed_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if limit:
            files = files[:limit]

        trades: list[dict] = []
        skipped = 0

        for path in files:
            data = _read_json(path)
            if not data:
                skipped += 1
                continue

            if str(data.get("status", "")).lower() != "closed":
                skipped += 1
                continue

            if _realized_pl(data) is None:
                skipped += 1
                continue

            data["_file_path"] = str(path)
            trades.append(data)

        return trades, skipped

    # ── Metric builders ───────────────────────────────────────────────────────

    def _setup_performance(self, trades: list[dict]) -> dict[str, SetupPerformance]:
        """Calculate metrics by setup type."""
        grouped: dict[str, list[dict]] = {}

        for trade in trades:
            setup = str(trade.get("setup_type", trade.get("setup", "unknown")) or "unknown")
            grouped.setdefault(setup, []).append(trade)

        results: dict[str, SetupPerformance] = {}

        for setup, items in grouped.items():
            pls = [_realized_pl(t) for t in items]
            wins = [p for p in pls if p > 0]
            losses = [p for p in pls if p < 0]
            breakevens = [p for p in pls if p == 0]

            stats = SetupPerformance(
                setup_type    = setup,
                trade_count   = len(items),
                wins          = len(wins),
                losses        = len(losses),
                breakevens    = len(breakevens),
                total_pl      = sum(pls),
                win_rate      = (len(wins) / len(items) * 100) if items else 0.0,
                avg_win       = (sum(wins) / len(wins)) if wins else 0.0,
                avg_loss      = (sum(losses) / len(losses)) if losses else 0.0,
                best_trade_pl = max(pls) if pls else 0.0,
                worst_trade_pl= min(pls) if pls else 0.0,
            )
            results[setup] = stats

        return results

    def _daily_pl(self, trades: list[dict]) -> dict:
        """Calculate realized P/L by close date."""
        daily: dict[str, float] = {}

        for trade in trades:
            day = _trade_day(trade)
            daily[day] = daily.get(day, 0.0) + _realized_pl(trade)

        return {k: round(v, 2) for k, v in sorted(daily.items())}

    def _weekly_pl(self, trades: list[dict]) -> dict:
        """Calculate realized P/L by ISO week."""
        weekly: dict[str, float] = {}

        for trade in trades:
            dt = _trade_datetime(trade)
            key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
            weekly[key] = weekly.get(key, 0.0) + _realized_pl(trade)

        return {k: round(v, 2) for k, v in sorted(weekly.items())}

    def _source_performance(self, trades: list[dict], source_name: str) -> dict:
        """Calculate basic stats for scanner/manual source."""
        source_name = source_name.lower()
        relevant = [
            t for t in trades
            if str(t.get("source", t.get("candidate_source", ""))).lower() == source_name
        ]

        if not relevant:
            return {
                "source": source_name,
                "trade_count": 0,
                "total_pl": 0.0,
                "win_rate": 0.0,
            }

        pls = [_realized_pl(t) for t in relevant]
        wins = [p for p in pls if p > 0]

        return {
            "source": source_name,
            "trade_count": len(relevant),
            "total_pl": round(sum(pls), 2),
            "win_rate": round(len(wins) / len(relevant) * 100, 2),
        }

    @staticmethod
    def _trade_card(trade: dict) -> dict:
        """Return simple dashboard card for one trade."""
        return {
            "trade_id": trade.get("trade_id", ""),
            "ticker": trade.get("ticker", ""),
            "setup_type": trade.get("setup_type", trade.get("setup", "")),
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", ""),
            "entry_price": _safe_float(trade.get("entry_price", 0.0)),
            "exit_price": _safe_float(trade.get("exit_price", 0.0)),
            "quantity": int(_safe_float(trade.get("exit_quantity", trade.get("quantity", 0)))),
            "realized_pl": round(_realized_pl(trade), 2),
            "realized_pl_percent": round(_safe_float(trade.get("realized_pl_percent", 0.0)), 4),
            "close_reason": trade.get("close_reason", ""),
            "file_path": trade.get("_file_path", ""),
        }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def calculate_performance(settings: dict, limit: Optional[int] = None) -> PerformanceSummary:
    """
    Convenience function for dashboard/backend.
    """
    tracker = PerformanceTracker(settings)
    return tracker.calculate(limit=limit)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[performance] Failed to read %s: %s", path, exc)
        return {}


def _realized_pl(trade: dict) -> float:
    value = trade.get("realized_pl", None)
    if value is None:
        entry = _safe_float(trade.get("entry_price", 0.0))
        exit_price = _safe_float(trade.get("exit_price", 0.0))
        qty = _safe_float(trade.get("exit_quantity", trade.get("quantity", 0.0)))
        if entry > 0 and exit_price > 0 and qty > 0:
            return (exit_price - entry) * qty
        return 0.0
    return _safe_float(value)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trade_datetime(trade: dict) -> datetime:
    raw = trade.get("exit_time") or trade.get("closed_at") or trade.get("updated_at") or trade.get("entry_time")
    if not raw:
        return datetime.now(timezone.utc)

    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _trade_day(trade: dict) -> str:
    return _trade_datetime(trade).date().isoformat()
