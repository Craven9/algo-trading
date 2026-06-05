"""
src/frontend/dashboard_state_writer.py — Dashboard state JSON writer
Builds and writes the dashboard state used by the frontend.

The frontend should not need to understand every internal bot module.
It should read one clean state object that explains:
  - Bot status
  - Current mode / safety settings
  - Scanner results
  - Ranked candidates
  - Open trades
  - Closed trades
  - Performance summary
  - Learning summary
  - Recent decisions
  - Warnings/errors

Responsibilities:
  - Build dashboard state from bot outputs
  - Read open and closed trade JSON files
  - Include performance and learning summaries when available
  - Write a clean dashboard_state.json file
  - Keep frontend data stable and easy to render
  - Never place orders or approve trades

Design rules:
  - This file only writes dashboard state
  - It does not scan
  - It does not trade
  - It does not approve decisions
  - Missing data should produce empty lists, not crashes
  - JSON output should be human-readable
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_DASHBOARD_DIR = _DEFAULT_DATA_DIR / "dashboard"
_DEFAULT_DASHBOARD_STATE = _DEFAULT_DASHBOARD_DIR / "dashboard_state.json"


# ── Writer ────────────────────────────────────────────────────────────────────

class DashboardStateWriter:
    """
    Builds and writes dashboard state.

    Usage:
        writer = DashboardStateWriter(settings)
        writer.write_state(
            scanner_results=scanner_candidates,
            ranked_candidates=ranked,
            recent_decisions=decisions,
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        paths = settings.get("paths", {})

        self._dashboard_dir = Path(
            paths.get("dashboard_dir", str(_DEFAULT_DASHBOARD_DIR))
        )
        self._state_path = Path(
            paths.get("dashboard_state_path", str(_DEFAULT_DASHBOARD_STATE))
        )

        trades_dir = Path(paths.get("trades_dir", str(Path("data") / "trades")))
        self._open_dir = Path(paths.get("open_trades_dir", str(trades_dir / "open")))
        self._closed_dir = Path(paths.get("closed_trades_dir", str(trades_dir / "closed")))

        self._learning_summary_path = Path(
            paths.get("learning_summary_path", str(Path("data") / "learning" / "learning_summary.json"))
        )

        self._dashboard_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._open_dir.mkdir(parents=True, exist_ok=True)
        self._closed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def build_state(
        self,
        bot_status: str = "idle",
        scanner_results: Optional[list] = None,
        ranked_candidates: Optional[list] = None,
        recent_decisions: Optional[list] = None,
        performance_summary: Optional[object] = None,
        learning_summary: Optional[object] = None,
        errors: Optional[list[str]] = None,
        warnings: Optional[list[str]] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Build dashboard state object.

        Args:
            bot_status:          idle|running|paused|error.
            scanner_results:     Raw scanner candidate objects.
            ranked_candidates:   CandidateRank objects.
            recent_decisions:    TradeDecisionResult objects.
            performance_summary: Optional PerformanceSummary.
            learning_summary:    Optional LearningSummary.
            errors:              Recent error strings.
            warnings:            Recent warning strings.
            extra:               Optional extra dashboard fields.

        Returns:
            Dict ready to write as JSON.
        """
        state = {
            "bot": self._bot_state(bot_status),
            "mode": self._mode_state(),
            "scanner": {
                "candidates": _to_list(scanner_results),
                "ranked_candidates": _to_list(ranked_candidates),
                "candidate_count": len(scanner_results or []),
                "ranked_count": len(ranked_candidates or []),
            },
            "trades": {
                "open": self._load_open_trades(),
                "closed": self._load_closed_trades(limit=50),
            },
            "recent_decisions": _to_list(recent_decisions),
            "performance": _to_dict(performance_summary),
            "learning": _to_dict(learning_summary) if learning_summary else self._load_learning_summary(),
            "warnings": warnings or [],
            "errors": errors or [],
            "extra": extra or {},
            "updated_at": _now_iso(),
        }

        state["trades"]["open_count"] = len(state["trades"]["open"])
        state["trades"]["closed_count"] = len(state["trades"]["closed"])

        return state

    def write_state(
        self,
        bot_status: str = "idle",
        scanner_results: Optional[list] = None,
        ranked_candidates: Optional[list] = None,
        recent_decisions: Optional[list] = None,
        performance_summary: Optional[object] = None,
        learning_summary: Optional[object] = None,
        errors: Optional[list[str]] = None,
        warnings: Optional[list[str]] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Build and write dashboard state JSON.

        Returns:
            Path to dashboard state JSON file.
        """
        state = self.build_state(
            bot_status          = bot_status,
            scanner_results     = scanner_results,
            ranked_candidates   = ranked_candidates,
            recent_decisions    = recent_decisions,
            performance_summary = performance_summary,
            learning_summary    = learning_summary,
            errors              = errors,
            warnings            = warnings,
            extra               = extra,
        )

        _write_json(self._state_path, state)

        log.debug("[dashboard_state] State written: %s", self._state_path)
        return str(self._state_path)

    def read_state(self) -> dict:
        """
        Read the current dashboard state file.
        """
        return _read_json(self._state_path)

    # ── State sections ────────────────────────────────────────────────────────

    def _bot_state(self, bot_status: str) -> dict:
        """Build bot status section."""
        return {
            "status": bot_status,
            "name": self._settings.get("bot_name", "AI Trading Assistant"),
            "version": self._settings.get("version", ""),
            "updated_at": _now_iso(),
        }

    def _mode_state(self) -> dict:
        """Build mode/safety section."""
        mode = self._settings.get("mode", {})
        return {
            "bot_enabled": mode.get("bot_enabled", True),
            "dry_run": mode.get("dry_run", True),
            "paper_trading_only": mode.get("paper_trading_only", True),
            "allow_live_money": mode.get("allow_live_money", False),
            "safety_lock": mode.get("safety_lock", False),
            "after_hours_enabled": mode.get("after_hours_enabled", False),
        }

    def _load_open_trades(self) -> list[dict]:
        """Load open trade JSON files for dashboard."""
        trades = []
        for path in sorted(self._open_dir.glob("*.json")):
            data = _read_json(path)
            if not data:
                continue
            data["_file_path"] = str(path)
            trades.append(self._trade_card(data, open_trade=True))
        return trades

    def _load_closed_trades(self, limit: int = 50) -> list[dict]:
        """Load recent closed trade JSON files for dashboard."""
        files = sorted(
            self._closed_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        trades = []
        for path in files:
            data = _read_json(path)
            if not data:
                continue
            data["_file_path"] = str(path)
            trades.append(self._trade_card(data, open_trade=False))
        return trades

    def _load_learning_summary(self) -> dict:
        """Load latest learning summary if available."""
        return _read_json(self._learning_summary_path)

    @staticmethod
    def _trade_card(trade: dict, open_trade: bool) -> dict:
        """
        Convert trade JSON into frontend-friendly card.
        """
        entry = _safe_float(trade.get("entry_price", 0.0))
        current = _safe_float(trade.get("current_price", trade.get("exit_price", entry)))
        exit_price = _safe_float(trade.get("exit_price", 0.0))
        qty = int(_safe_float(trade.get("quantity", trade.get("exit_quantity", 0))))

        if open_trade:
            pl = _safe_float(trade.get("unrealized_pl", 0.0))
            pl_pct = _safe_float(trade.get("unrealized_pl_percent", 0.0))
        else:
            pl = _safe_float(trade.get("realized_pl", 0.0))
            pl_pct = _safe_float(trade.get("realized_pl_percent", 0.0))

        return {
            "trade_id": trade.get("trade_id", ""),
            "ticker": trade.get("ticker", ""),
            "status": trade.get("status", "open" if open_trade else "closed"),
            "setup_type": trade.get("setup_type", trade.get("setup", "")),
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", ""),
            "entry_price": entry,
            "current_price": current,
            "exit_price": exit_price if exit_price > 0 else None,
            "quantity": qty,
            "stop_loss": trade.get("stop_loss"),
            "target_1": trade.get("target_1"),
            "target_2": trade.get("target_2"),
            "runner_target": trade.get("runner_target"),
            "pl_dollars": round(pl, 2),
            "pl_percent": round(pl_pct, 4),
            "close_reason": trade.get("close_reason", ""),
            "entry_reasons": trade.get("entry_reasons", []),
            "entry_warnings": trade.get("entry_warnings", []),
            "scores": trade.get("scores", {}),
            "file_path": trade.get("_file_path", ""),
        }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def write_dashboard_state(
    settings: dict,
    bot_status: str = "idle",
    scanner_results: Optional[list] = None,
    ranked_candidates: Optional[list] = None,
    recent_decisions: Optional[list] = None,
    performance_summary: Optional[object] = None,
    learning_summary: Optional[object] = None,
    errors: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
    extra: Optional[dict] = None,
) -> str:
    """
    Convenience function for bot_runner.py.
    """
    writer = DashboardStateWriter(settings)
    return writer.write_state(
        bot_status          = bot_status,
        scanner_results     = scanner_results,
        ranked_candidates   = ranked_candidates,
        recent_decisions    = recent_decisions,
        performance_summary = performance_summary,
        learning_summary    = learning_summary,
        errors              = errors,
        warnings            = warnings,
        extra               = extra,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(obj: object) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    try:
        return asdict(obj)
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _to_list(items: Optional[list]) -> list:
    if not items:
        return []
    return [_to_dict(item) for item in items]


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[dashboard_state] Failed to read %s: %s", path, exc)
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
