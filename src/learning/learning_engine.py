"""
src/learning/learning_engine.py — Trade learning and rule suggestion engine
Combines closed trade reviews, performance statistics, and historical edge
to produce lessons the bot can use for future decisions.

The bot should learn from its paper trades, but it should not randomly
change rules without structure.  This module creates transparent learning
summaries and suggested rule adjustments.  The suggestions can be reviewed
by the user or used as soft scoring context later.

Responsibilities:
  - Read backtest reviews from closed trade files
  - Summarize repeated mistake patterns
  - Summarize repeated winning patterns
  - Identify setups that are working or failing
  - Identify weak conditions to avoid
  - Produce learning notes for dashboard display
  - Produce optional rule suggestions for future config changes
  - Save a learning summary JSON file

Design rules:
  - This file does not place trades
  - This file does not approve trades
  - This file does not directly rewrite bot_settings.json
  - Suggestions are advisory unless the user manually applies them
  - Learning should be conservative with small samples
  - Missing trade history returns a neutral summary
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_CLOSED_DIR = Path("data") / "trades" / "closed"
_DEFAULT_LEARNING_DIR = Path("data") / "learning"
_DEFAULT_SUMMARY_FILE = _DEFAULT_LEARNING_DIR / "learning_summary.json"
_MIN_SAMPLE_SIZE = 5


# ── Rule suggestion ───────────────────────────────────────────────────────────

@dataclass
class RuleSuggestion:
    """
    Suggested rule change or scoring adjustment.
    These are advisory and should not be auto-applied without review.
    """
    category:       str
    suggestion:     str
    reason:         str
    confidence:     str = "low"  # low|medium|high
    affected_setup: str = ""
    metric:         str = ""
    value:          float = 0.0

    def to_dict(self) -> dict:
        return {
            "category":       self.category,
            "suggestion":     self.suggestion,
            "reason":         self.reason,
            "confidence":     self.confidence,
            "affected_setup": self.affected_setup,
            "metric":         self.metric,
            "value":          round(self.value, 4),
        }


# ── Learning summary ──────────────────────────────────────────────────────────

@dataclass
class LearningSummary:
    """
    Full learning summary for dashboard and future scoring context.
    """
    trade_count:              int = 0
    reviewed_trade_count:     int = 0

    repeated_mistakes:        dict = field(default_factory=dict)
    winning_patterns:         dict = field(default_factory=dict)
    setup_scores:             dict = field(default_factory=dict)

    strongest_setup:          str = ""
    weakest_setup:            str = ""
    avoid_conditions:         list[str] = field(default_factory=list)
    reinforce_conditions:     list[str] = field(default_factory=list)

    rule_suggestions:         list[RuleSuggestion] = field(default_factory=list)
    learning_notes:           list[str] = field(default_factory=list)
    warnings:                 list[str] = field(default_factory=list)

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "trade_count":          self.trade_count,
            "reviewed_trade_count": self.reviewed_trade_count,
            "repeated_mistakes":    self.repeated_mistakes,
            "winning_patterns":     self.winning_patterns,
            "setup_scores":         self.setup_scores,
            "strongest_setup":      self.strongest_setup,
            "weakest_setup":        self.weakest_setup,
            "avoid_conditions":     self.avoid_conditions,
            "reinforce_conditions": self.reinforce_conditions,
            "rule_suggestions":     [s.to_dict() for s in self.rule_suggestions],
            "learning_notes":       self.learning_notes,
            "warnings":             self.warnings,
            "generated_at":         self.generated_at,
        }


# ── Learning engine ───────────────────────────────────────────────────────────

class LearningEngine:
    """
    Builds a learning summary from closed trade reviews.

    Usage:
        engine = LearningEngine(settings)
        summary = engine.generate_summary()
        engine.save_summary(summary)
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._learning = settings.get("learning", {})
        paths = settings.get("paths", {})

        trades_dir = Path(paths.get("trades_dir", str(Path("data") / "trades")))
        self._closed_dir = Path(
            paths.get("closed_trades_dir", str(trades_dir / "closed"))
        )

        self._learning_dir = Path(
            paths.get("learning_dir", str(_DEFAULT_LEARNING_DIR))
        )
        self._summary_file = Path(
            paths.get("learning_summary_path", str(self._learning_dir / "learning_summary.json"))
        )

        self._closed_dir.mkdir(parents=True, exist_ok=True)
        self._learning_dir.mkdir(parents=True, exist_ok=True)

        self._min_sample = int(
            self._learning.get("min_sample_size_for_learning", _MIN_SAMPLE_SIZE)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_summary(self, limit: Optional[int] = None) -> LearningSummary:
        """
        Generate a learning summary from closed trades.

        Args:
            limit: Optional max number of recent closed trades to review.

        Returns:
            LearningSummary.
        """
        trades = self._load_closed_trades(limit=limit)
        summary = LearningSummary(trade_count=len(trades))

        if not trades:
            summary.warnings.append("No closed trades found — no learning summary available")
            summary.learning_notes.append("Need closed paper trades before learning can improve the bot")
            return summary

        reviewed = [t for t in trades if t.get("backtest_review")]
        summary.reviewed_trade_count = len(reviewed)

        if not reviewed:
            summary.warnings.append("No reviewed trades found — run backtest_reviewer.py first")
            summary.learning_notes.append("Backtest reviews are needed before rule suggestions can be trusted")
            return summary

        summary.repeated_mistakes = self._mistake_counts(reviewed)
        summary.winning_patterns = self._winning_pattern_counts(reviewed)
        summary.setup_scores = self._setup_scores(reviewed)

        self._assign_best_worst_setup(summary)
        summary.avoid_conditions = self._avoid_conditions(summary)
        summary.reinforce_conditions = self._reinforce_conditions(summary)
        summary.rule_suggestions = self._rule_suggestions(summary, reviewed)
        summary.learning_notes = self._learning_notes(summary)

        log.info(
            "[learning] trades=%d reviewed=%d suggestions=%d",
            summary.trade_count,
            summary.reviewed_trade_count,
            len(summary.rule_suggestions),
        )
        return summary

    def save_summary(self, summary: LearningSummary) -> str:
        """
        Save learning summary to JSON.

        Returns:
            File path.
        """
        _write_json(self._summary_file, summary.to_dict())
        log.info("[learning] Summary saved: %s", self._summary_file)
        return str(self._summary_file)

    def generate_and_save(self, limit: Optional[int] = None) -> LearningSummary:
        """
        Generate and save learning summary.
        """
        summary = self.generate_summary(limit=limit)
        self.save_summary(summary)
        return summary

    def load_latest_summary(self) -> LearningSummary:
        """
        Load the latest saved learning summary.
        Returns an empty neutral summary when missing.
        """
        data = _read_json(self._summary_file)
        if not data:
            return LearningSummary(
                warnings=["No saved learning summary found"]
            )

        summary = LearningSummary(
            trade_count          = int(data.get("trade_count", 0)),
            reviewed_trade_count = int(data.get("reviewed_trade_count", 0)),
            repeated_mistakes    = data.get("repeated_mistakes", {}),
            winning_patterns     = data.get("winning_patterns", {}),
            setup_scores         = data.get("setup_scores", {}),
            strongest_setup      = data.get("strongest_setup", ""),
            weakest_setup        = data.get("weakest_setup", ""),
            avoid_conditions     = data.get("avoid_conditions", []),
            reinforce_conditions = data.get("reinforce_conditions", []),
            learning_notes       = data.get("learning_notes", []),
            warnings             = data.get("warnings", []),
            generated_at         = data.get("generated_at", datetime.now(timezone.utc).isoformat()),
        )

        suggestions = []
        for raw in data.get("rule_suggestions", []):
            suggestions.append(RuleSuggestion(
                category       = raw.get("category", ""),
                suggestion     = raw.get("suggestion", ""),
                reason         = raw.get("reason", ""),
                confidence     = raw.get("confidence", "low"),
                affected_setup = raw.get("affected_setup", ""),
                metric         = raw.get("metric", ""),
                value          = _safe_float(raw.get("value", 0.0)),
            ))
        summary.rule_suggestions = suggestions
        return summary

    # ── Pattern builders ──────────────────────────────────────────────────────

    def _mistake_counts(self, trades: list[dict]) -> dict:
        """Count repeated mistake categories from backtest reviews."""
        counter: Counter[str] = Counter()

        for trade in trades:
            review = trade.get("backtest_review", {}) or {}
            for mistake in review.get("mistakes", []) or []:
                counter[str(mistake)] += 1

        return dict(counter.most_common())

    def _winning_pattern_counts(self, trades: list[dict]) -> dict:
        """Count repeated positive observations from winning trades."""
        counter: Counter[str] = Counter()

        for trade in trades:
            review = trade.get("backtest_review", {}) or {}
            if review.get("outcome") != "win":
                continue
            for positive in review.get("positives", []) or []:
                counter[str(positive)] += 1

        return dict(counter.most_common())

    def _setup_scores(self, trades: list[dict]) -> dict:
        """Build simple setup performance stats from reviewed trades."""
        grouped: dict[str, list[dict]] = defaultdict(list)

        for trade in trades:
            setup = str(
                trade.get("setup_type")
                or trade.get("setup")
                or trade.get("backtest_review", {}).get("setup_type")
                or "unknown"
            )
            grouped[setup].append(trade)

        scores: dict[str, dict] = {}

        for setup, items in grouped.items():
            pls = [_realized_pl(t) for t in items]
            wins = [p for p in pls if p > 0]
            losses = [p for p in pls if p < 0]

            trade_count = len(items)
            win_rate = len(wins) / trade_count * 100 if trade_count else 0.0
            avg_pl = sum(pls) / trade_count if trade_count else 0.0
            total_pl = sum(pls)

            scores[setup] = {
                "trade_count": trade_count,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(win_rate, 2),
                "average_pl": round(avg_pl, 2),
                "total_pl": round(total_pl, 2),
                "sample_ok": trade_count >= self._min_sample,
            }

        return scores

    def _assign_best_worst_setup(self, summary: LearningSummary) -> None:
        """Set strongest and weakest setup from setup_scores."""
        if not summary.setup_scores:
            return

        valid = {
            setup: stats for setup, stats in summary.setup_scores.items()
            if stats.get("sample_ok")
        }

        # If no setup has enough sample size, use all but warn.
        if not valid:
            valid = summary.setup_scores
            summary.warnings.append("Setup samples are small — strongest/weakest setup is low confidence")

        summary.strongest_setup = max(
            valid.items(),
            key=lambda item: (item[1].get("total_pl", 0), item[1].get("win_rate", 0)),
        )[0]

        summary.weakest_setup = min(
            valid.items(),
            key=lambda item: (item[1].get("total_pl", 0), item[1].get("win_rate", 0)),
        )[0]

    # ── Conditions / suggestions ──────────────────────────────────────────────

    def _avoid_conditions(self, summary: LearningSummary) -> list[str]:
        """Create avoid-condition list from repeated mistakes."""
        avoid: list[str] = []

        for mistake, count in summary.repeated_mistakes.items():
            if count < 2:
                continue

            lower = mistake.lower()

            if "chase" in lower or "extended" in lower:
                avoid.append("Avoid chasing extended moves without a pullback")
            elif "vwap" in lower:
                avoid.append("Avoid longs below VWAP or without confirmed VWAP reclaim")
            elif "volume" in lower or "rvol" in lower:
                avoid.append("Avoid entries without strong volume/RVOL confirmation")
            elif "risk/reward" in lower:
                avoid.append("Avoid trades with weak reward-to-risk")
            elif "probability" in lower:
                avoid.append("Avoid trades with probability score below minimum")
            elif "failed breakout" in lower:
                avoid.append("Avoid breakouts that fail to hold the broken level")
            elif "failed reclaim" in lower:
                avoid.append("Avoid longs after failed reclaim unless price reclaims again with volume")
            else:
                avoid.append(mistake)

        if summary.weakest_setup:
            avoid.append(f"Be more selective with weakest setup: {summary.weakest_setup}")

        return _dedupe(avoid)

    def _reinforce_conditions(self, summary: LearningSummary) -> list[str]:
        """Create reinforce-condition list from winning patterns."""
        reinforce: list[str] = []

        for pattern, count in summary.winning_patterns.items():
            if count < 2:
                continue
            reinforce.append(pattern)

        if summary.strongest_setup:
            reinforce.append(f"Prioritize strongest setup when confirmed: {summary.strongest_setup}")

        return _dedupe(reinforce)

    def _rule_suggestions(
        self,
        summary: LearningSummary,
        trades: list[dict],
    ) -> list[RuleSuggestion]:
        """Build advisory rule suggestions."""
        suggestions: list[RuleSuggestion] = []

        if summary.reviewed_trade_count < self._min_sample:
            suggestions.append(RuleSuggestion(
                category   = "sample_size",
                suggestion = "Do not change config yet",
                reason     = f"Only {summary.reviewed_trade_count} reviewed trades available",
                confidence = "low",
                metric     = "reviewed_trade_count",
                value      = summary.reviewed_trade_count,
            ))
            return suggestions

        for condition in summary.avoid_conditions:
            lower = condition.lower()

            if "vwap" in lower:
                suggestions.append(RuleSuggestion(
                    category   = "entry_filter",
                    suggestion = "Require VWAP confirmation more strictly",
                    reason     = condition,
                    confidence = self._confidence_from_count(summary, "vwap"),
                    metric     = "mistake_count",
                    value      = self._count_matching(summary.repeated_mistakes, "vwap"),
                ))

            elif "volume" in lower or "rvol" in lower:
                suggestions.append(RuleSuggestion(
                    category   = "volume_filter",
                    suggestion = "Increase minimum RVOL or require stronger volume confirmation",
                    reason     = condition,
                    confidence = self._confidence_from_count(summary, "volume"),
                    metric     = "mistake_count",
                    value      = self._count_matching(summary.repeated_mistakes, "volume"),
                ))

            elif "risk" in lower:
                suggestions.append(RuleSuggestion(
                    category   = "risk_reward",
                    suggestion = "Tighten reward-to-risk requirements or wait for better entry",
                    reason     = condition,
                    confidence = self._confidence_from_count(summary, "risk"),
                    metric     = "mistake_count",
                    value      = self._count_matching(summary.repeated_mistakes, "risk"),
                ))

            elif "extended" in lower or "chasing" in lower:
                suggestions.append(RuleSuggestion(
                    category   = "entry_timing",
                    suggestion = "Reduce chasing by requiring pullback/retest after large moves",
                    reason     = condition,
                    confidence = self._confidence_from_count(summary, "chase"),
                    metric     = "mistake_count",
                    value      = self._count_matching(summary.repeated_mistakes, "chase"),
                ))

        if summary.weakest_setup:
            weak_stats = summary.setup_scores.get(summary.weakest_setup, {})
            if weak_stats.get("sample_ok") and weak_stats.get("total_pl", 0) < 0:
                suggestions.append(RuleSuggestion(
                    category       = "setup_weighting",
                    suggestion     = "Reduce priority or require stronger confirmation for this setup",
                    reason         = f"{summary.weakest_setup} is currently weakest by P/L",
                    confidence     = "medium",
                    affected_setup = summary.weakest_setup,
                    metric         = "total_pl",
                    value          = _safe_float(weak_stats.get("total_pl", 0)),
                ))

        if summary.strongest_setup:
            strong_stats = summary.setup_scores.get(summary.strongest_setup, {})
            if strong_stats.get("sample_ok") and strong_stats.get("total_pl", 0) > 0:
                suggestions.append(RuleSuggestion(
                    category       = "setup_weighting",
                    suggestion     = "Allow this setup to keep normal priority when other gates confirm",
                    reason         = f"{summary.strongest_setup} is currently strongest by P/L",
                    confidence     = "medium",
                    affected_setup = summary.strongest_setup,
                    metric         = "total_pl",
                    value          = _safe_float(strong_stats.get("total_pl", 0)),
                ))

        return suggestions

    def _learning_notes(self, summary: LearningSummary) -> list[str]:
        """Build human-readable learning notes."""
        notes: list[str] = []

        if summary.reviewed_trade_count < self._min_sample:
            notes.append(
                f"Only {summary.reviewed_trade_count} reviewed trades — keep learning conservative"
            )

        if summary.strongest_setup:
            notes.append(f"Strongest setup so far: {summary.strongest_setup}")
        if summary.weakest_setup:
            notes.append(f"Weakest setup so far: {summary.weakest_setup}")

        for condition in summary.avoid_conditions[:5]:
            notes.append(f"Avoid condition: {condition}")

        for condition in summary.reinforce_conditions[:5]:
            notes.append(f"Reinforce condition: {condition}")

        if not notes:
            notes.append("No clear learning pattern yet")

        return _dedupe(notes)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _count_matching(counts: dict, keyword: str) -> int:
        total = 0
        keyword = keyword.lower()
        for key, value in counts.items():
            if keyword in str(key).lower():
                total += int(value)
        return total

    def _confidence_from_count(self, summary: LearningSummary, keyword: str) -> str:
        count = self._count_matching(summary.repeated_mistakes, keyword)
        if count >= 5:
            return "high"
        if count >= 3:
            return "medium"
        return "low"

    def _load_closed_trades(self, limit: Optional[int]) -> list[dict]:
        files = sorted(
            self._closed_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if limit:
            files = files[:limit]

        trades: list[dict] = []
        for path in files:
            data = _read_json(path)
            if not data:
                continue
            if str(data.get("status", "")).lower() != "closed":
                continue
            data["_file_path"] = str(path)
            trades.append(data)

        return trades


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_learning_summary(
    settings: dict,
    limit: Optional[int] = None,
    save: bool = True,
) -> LearningSummary:
    """
    Convenience function for bot_runner.py or dashboard backend.
    """
    engine = LearningEngine(settings)
    summary = engine.generate_summary(limit=limit)
    if save:
        engine.save_summary(summary)
    return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[learning] Failed to read %s: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


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


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
