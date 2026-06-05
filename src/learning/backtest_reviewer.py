"""
src/learning/backtest_reviewer.py — Closed trade review and lesson builder
Reviews completed trades after they close and creates structured feedback
for the bot's learning system.

The bot should not just log wins and losses.  It should review whether the
entry was good, whether the exit made sense, whether the setup followed the
rules, and what should be improved next time.

Responsibilities:
  - Read closed trade JSON files that are ready_for_backtest_review
  - Review entry quality using saved scores/reasons
  - Review exit quality using close reason and realized P/L
  - Classify trade outcome as win, loss, breakeven, or unknown
  - Identify likely mistake categories
  - Produce lessons for future scoring/risk decisions
  - Mark reviewed trades so they are not reviewed repeatedly
  - Never place trades or change broker state

Design rules:
  - This file only reviews closed local trade records
  - This file does not approve trades
  - This file does not place orders
  - Reviews are rule-based and transparent
  - Missing trade data should not crash the review process
  - Review output is saved back into the trade JSON file
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


# ── Review result ─────────────────────────────────────────────────────────────

@dataclass
class BacktestReviewResult:
    """
    Structured review for one closed trade.
    """
    trade_id:             str
    ticker:               str
    setup_type:           str = ""
    outcome:              str = "unknown"  # win|loss|breakeven|unknown

    entry_quality:        str = "unknown"  # good|acceptable|weak|bad|unknown
    exit_quality:         str = "unknown"  # good|acceptable|weak|bad|unknown
    overall_grade:        str = "unknown"  # A|B|C|D|F|unknown

    realized_pl:          float = 0.0
    realized_pl_percent:  float = 0.0
    final_score:          float = 0.0
    probability_score:    float = 0.0
    setup_score:          float = 0.0
    risk_reward_score:    float = 0.0

    mistakes:             list[str] = field(default_factory=list)
    positives:            list[str] = field(default_factory=list)
    lessons:              list[str] = field(default_factory=list)
    warnings:             list[str] = field(default_factory=list)

    reviewed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "trade_id":            self.trade_id,
            "ticker":              self.ticker,
            "setup_type":          self.setup_type,
            "outcome":             self.outcome,
            "entry_quality":       self.entry_quality,
            "exit_quality":        self.exit_quality,
            "overall_grade":       self.overall_grade,
            "realized_pl":         round(self.realized_pl, 2),
            "realized_pl_percent": round(self.realized_pl_percent, 4),
            "final_score":         round(self.final_score, 2),
            "probability_score":   round(self.probability_score, 2),
            "setup_score":         round(self.setup_score, 2),
            "risk_reward_score":   round(self.risk_reward_score, 2),
            "mistakes":            self.mistakes,
            "positives":           self.positives,
            "lessons":             self.lessons,
            "warnings":            self.warnings,
            "reviewed_at":         self.reviewed_at,
        }


# ── Reviewer ─────────────────────────────────────────────────────────────────

class BacktestReviewer:
    """
    Reviews closed trade files.

    Usage:
        reviewer = BacktestReviewer(settings)
        results = reviewer.review_ready_trades()
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

    def review_ready_trades(self, limit: Optional[int] = None) -> list[BacktestReviewResult]:
        """
        Review all closed trades that are ready for backtest review.

        Args:
            limit: Optional maximum number of files to review.

        Returns:
            List of BacktestReviewResult.
        """
        files = self._ready_trade_files()
        if limit:
            files = files[:limit]

        results: list[BacktestReviewResult] = []

        for path in files:
            result = self.review_trade_file(path)
            if result:
                results.append(result)

        log.info("[backtest_reviewer] Reviewed %d trade(s)", len(results))
        return results

    def review_trade_file(self, trade_path: str | Path) -> Optional[BacktestReviewResult]:
        """
        Review one closed trade file and save review output back into the file.
        """
        path = Path(trade_path)
        trade = _read_json(path)

        if not trade:
            log.warning("[backtest_reviewer] Missing/unreadable trade file: %s", path)
            return None

        if str(trade.get("status", "")).lower() != "closed":
            log.debug("[backtest_reviewer] Skipping non-closed trade: %s", path)
            return None

        review = self._review_trade(trade)

        trade["backtest_review"] = review.to_dict()
        trade["ready_for_backtest_review"] = False
        trade["backtest_reviewed"] = True
        trade["backtest_reviewed_at"] = review.reviewed_at
        trade["updated_at"] = _now_iso()

        _write_json(path, trade)

        log.info(
            "[backtest_reviewer] %s reviewed: outcome=%s grade=%s",
            review.ticker, review.outcome, review.overall_grade,
        )
        return review

    def review_trade_dict(self, trade: dict) -> BacktestReviewResult:
        """
        Review a trade dict without writing to disk.
        Useful for tests.
        """
        return self._review_trade(trade)

    # ── Core review ───────────────────────────────────────────────────────────

    def _review_trade(self, trade: dict) -> BacktestReviewResult:
        """
        Build a structured review from one closed trade.
        """
        review = BacktestReviewResult(
            trade_id            = str(trade.get("trade_id", "")),
            ticker              = str(trade.get("ticker", "")).upper(),
            setup_type          = str(trade.get("setup_type", trade.get("setup", ""))),
            realized_pl         = _safe_float(trade.get("realized_pl", 0.0)),
            realized_pl_percent = _safe_float(trade.get("realized_pl_percent", 0.0)),
        )

        scores = trade.get("scores", {}) or {}
        review.final_score = _safe_float(scores.get("final_trade_quality_score", 0.0))
        review.probability_score = _safe_float(scores.get("probability_score", 0.0))
        review.setup_score = _safe_float(scores.get("setup_score", 0.0))
        review.risk_reward_score = _safe_float(scores.get("risk_reward_score", 0.0))

        review.outcome = self._classify_outcome(review)
        review.entry_quality = self._entry_quality(review, trade)
        review.exit_quality = self._exit_quality(review, trade)
        review.positives = self._positives(review, trade)
        review.mistakes = self._mistakes(review, trade)
        review.lessons = self._lessons(review, trade)
        review.overall_grade = self._overall_grade(review)
        review.warnings = self._warnings(review, trade)

        return review

    # ── Classifiers ───────────────────────────────────────────────────────────

    @staticmethod
    def _classify_outcome(review: BacktestReviewResult) -> str:
        """Classify trade outcome from realized P/L."""
        if review.realized_pl > 0.01:
            return "win"
        if review.realized_pl < -0.01:
            return "loss"
        if abs(review.realized_pl) <= 0.01:
            return "breakeven"
        return "unknown"

    @staticmethod
    def _entry_quality(review: BacktestReviewResult, trade: dict) -> str:
        """Classify entry quality from saved scores and warnings."""
        warning_count = len(trade.get("entry_warnings", []) or [])

        if review.final_score >= 90 and review.probability_score >= 80 and warning_count <= 2:
            return "good"
        if review.final_score >= 80 and review.probability_score >= 75:
            return "acceptable"
        if review.final_score >= 70:
            return "weak"
        if review.final_score > 0:
            return "bad"
        return "unknown"

    @staticmethod
    def _exit_quality(review: BacktestReviewResult, trade: dict) -> str:
        """Classify exit quality from close reason and outcome."""
        reason = str(trade.get("close_reason", "")).lower()

        if "hard stop" in reason or "max loss" in reason:
            if review.outcome == "loss":
                return "acceptable"
            return "unknown"

        if "partial" in reason or "target" in reason or "runner" in reason:
            if review.outcome == "win":
                return "good"
            return "acceptable"

        if "failed" in reason or "vwap" in reason or "breakout failed" in reason:
            return "acceptable"

        if review.outcome == "win":
            return "acceptable"
        if review.outcome == "loss":
            return "weak"
        return "unknown"

    # ── Explanation builders ──────────────────────────────────────────────────

    def _positives(self, review: BacktestReviewResult, trade: dict) -> list[str]:
        """Build positive observations."""
        positives: list[str] = []

        if review.final_score >= 85:
            positives.append("Trade had a strong final quality score")
        if review.probability_score >= 80:
            positives.append("Probability score was strong")
        if review.setup_score >= 80:
            positives.append("Setup score was strong")
        if review.risk_reward_score >= 80:
            positives.append("Risk/reward quality was strong")
        if review.outcome == "win":
            positives.append("Trade closed profitable")

        close_reason = str(trade.get("close_reason", "")).lower()
        if "target" in close_reason or "partial" in close_reason:
            positives.append("Exit followed profit-taking logic")
        if "stop" in close_reason and review.outcome == "loss":
            positives.append("Loss was controlled by stop logic")

        return _dedupe(positives)

    def _mistakes(self, review: BacktestReviewResult, trade: dict) -> list[str]:
        """Build likely mistake categories."""
        mistakes: list[str] = []

        if review.outcome == "loss":
            if review.final_score < 80:
                mistakes.append("Trade quality score was below ideal level")
            if review.probability_score < 75:
                mistakes.append("Probability score was below minimum")
            if review.setup_score < 75:
                mistakes.append("Setup confirmation was weak")
            if review.risk_reward_score < 70:
                mistakes.append("Risk/reward quality was weak")

        warnings = trade.get("entry_warnings", []) or []
        warning_text = " ".join(str(w).lower() for w in warnings)

        if "overextended" in warning_text or "chase" in warning_text:
            mistakes.append("Entry may have chased an extended move")
        if "below vwap" in warning_text:
            mistakes.append("Trade had weak VWAP context")
        if "spread" in warning_text:
            mistakes.append("Execution spread was a concern")
        if "volume" in warning_text or "rvol" in warning_text:
            mistakes.append("Volume confirmation was weak")
        if "bearish" in warning_text:
            mistakes.append("Trade had bearish context warning")

        close_reason = str(trade.get("close_reason", "")).lower()
        if "failed breakout" in close_reason:
            mistakes.append("Breakout failed after entry")
        if "failed reclaim" in close_reason:
            mistakes.append("Failed reclaim appeared after entry")
        if "max loss" in close_reason:
            mistakes.append("Trade reached max loss limit")

        return _dedupe(mistakes)

    def _lessons(self, review: BacktestReviewResult, trade: dict) -> list[str]:
        """Build reusable lessons for future trades."""
        lessons: list[str] = []

        if review.outcome == "win":
            lessons.append(
                f"{review.setup_type or 'setup'} can work when score/probability conditions align"
            )
        elif review.outcome == "loss":
            lessons.append(
                f"Be more selective with {review.setup_type or 'this setup'} when warnings are present"
            )

        if review.probability_score < 75:
            lessons.append("Do not enter when probability score is below minimum")
        if review.final_score < 80:
            lessons.append("Avoid trades below the final trade quality threshold")
        if review.risk_reward_score < 70:
            lessons.append("Require cleaner reward-to-risk before entry")

        for mistake in review.mistakes:
            if "chased" in mistake.lower() or "extended" in mistake.lower():
                lessons.append("Wait for pullback instead of chasing extended entries")
            if "vwap" in mistake.lower():
                lessons.append("Require price to reclaim and hold VWAP before long entry")
            if "volume" in mistake.lower():
                lessons.append("Require stronger volume/RVOL confirmation")

        if not lessons:
            lessons.append("No major rule adjustment identified")

        return _dedupe(lessons)

    @staticmethod
    def _warnings(review: BacktestReviewResult, trade: dict) -> list[str]:
        """Build review warnings."""
        warnings: list[str] = []

        if not review.trade_id:
            warnings.append("Trade ID missing")
        if not review.ticker:
            warnings.append("Ticker missing")
        if review.final_score == 0:
            warnings.append("Final score missing from trade record")
        if review.realized_pl == 0 and review.outcome == "breakeven":
            warnings.append("Trade appears breakeven or realized P/L missing")

        return warnings

    @staticmethod
    def _overall_grade(review: BacktestReviewResult) -> str:
        """Assign review grade from outcome, quality, and mistakes."""
        score = 50.0

        if review.outcome == "win":
            score += 25
        elif review.outcome == "breakeven":
            score += 5
        elif review.outcome == "loss":
            score -= 15

        if review.entry_quality == "good":
            score += 15
        elif review.entry_quality == "acceptable":
            score += 8
        elif review.entry_quality == "weak":
            score -= 5
        elif review.entry_quality == "bad":
            score -= 15

        if review.exit_quality == "good":
            score += 10
        elif review.exit_quality == "acceptable":
            score += 5
        elif review.exit_quality == "weak":
            score -= 5
        elif review.exit_quality == "bad":
            score -= 10

        score -= min(len(review.mistakes) * 5, 20)

        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    # ── File helpers ──────────────────────────────────────────────────────────

    def _ready_trade_files(self) -> list[Path]:
        """
        Return closed trade files ready for review.
        """
        files = sorted(
            self._closed_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        ready: list[Path] = []
        for path in files:
            data = _read_json(path)
            if not data:
                continue
            if str(data.get("status", "")).lower() != "closed":
                continue
            if bool(data.get("backtest_reviewed", False)):
                continue
            if bool(data.get("ready_for_backtest_review", True)):
                ready.append(path)

        return ready


# ── Convenience wrapper ───────────────────────────────────────────────────────

def review_ready_trades(
    settings: dict,
    limit: Optional[int] = None,
) -> list[BacktestReviewResult]:
    """
    Convenience function for bot_runner.py or scheduled review.
    """
    reviewer = BacktestReviewer(settings)
    return reviewer.review_ready_trades(limit=limit)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[backtest_reviewer] Failed to read %s: %s", path, exc)
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
