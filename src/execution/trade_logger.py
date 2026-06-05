"""
src/execution/trade_logger.py — Trade logging and trade file writer
Writes every paper trade decision, entry, exit, and outcome to JSON files.

The bot needs clean trade records so it can:
  - Display open and closed trades on the frontend
  - Track P/L
  - Review why a trade was entered
  - Review why a trade was exited
  - Run backtest review after the trade closes
  - Learn which setups are working or failing

Responsibilities:
  - Create a new trade JSON file after an approved paper buy
  - Update an open trade after broker order submission
  - Update an open trade with current price / unrealized P/L
  - Mark a trade closed after an exit fills
  - Store setup score, probability score, risk/reward, move potential,
    confidence label, entry reasons, warnings, and broker order data
  - Keep open and closed trade records consistent
  - Never place orders

Design rules:
  - This file only writes/updates trade records
  - It does not approve trades
  - It does not submit orders
  - Every trade gets a unique trade_id
  - JSON files are human-readable
  - Missing folders are created automatically
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import TradeDecisionResult

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_TRADES_DIR = os.path.join("data", "trades")
_DEFAULT_OPEN_DIR   = os.path.join(_DEFAULT_TRADES_DIR, "open")
_DEFAULT_CLOSED_DIR = os.path.join(_DEFAULT_TRADES_DIR, "closed")


# ── Logger ────────────────────────────────────────────────────────────────────

class TradeLogger:
    """
    Creates and updates trade JSON records.

    Usage:
        logger = TradeLogger(settings)
        path = logger.log_new_trade(decision, order_result)
    """

    def __init__(self, settings: dict):
        self._settings = settings
        paths = settings.get("paths", {})

        trades_dir = paths.get("trades_dir", _DEFAULT_TRADES_DIR)
        self._open_dir = Path(paths.get("open_trades_dir", os.path.join(trades_dir, "open")))
        self._closed_dir = Path(paths.get("closed_trades_dir", os.path.join(trades_dir, "closed")))

        self._open_dir.mkdir(parents=True, exist_ok=True)
        self._closed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_new_trade(
        self,
        decision: TradeDecisionResult,
        order_result: Optional[object] = None,
        extra_context: Optional[dict] = None,
    ) -> str:
        """
        Create a new open trade file after a paper buy is approved/submitted.

        Args:
            decision:      TradeDecisionResult from trade_quality_gate.py.
            order_result:  Optional OrderExecutionResult from order_executor.py.
            extra_context: Optional extra analysis context for debugging.

        Returns:
            Path to the created trade JSON file.
        """
        trade_id = _make_trade_id()
        ticker = decision.ticker.upper()
        now = _now_iso()

        order_dict = _to_dict(order_result)

        data = {
            "trade_id": trade_id,
            "ticker": ticker,
            "status": "open",
            "outcome_complete": False,
            "ready_for_backtest_review": False,

            # Entry
            "entry_time": now,
            "entry_price": decision.entry_price,
            "quantity": _extract_quantity(decision),
            "setup_type": decision.setup,
            "stop_loss": decision.stop_loss,
            "target_1": decision.target_1,
            "target_2": decision.target_2,
            "runner_target": decision.runner_target,

            # Scores / reasons
            "decision": decision.decision,
            "scores": decision.scores.to_dict() if hasattr(decision.scores, "to_dict") else _to_dict(decision.scores),
            "entry_reasons": list(decision.reasons or []),
            "entry_warnings": list(decision.warnings or []),
            "what_would_make_valid": list(decision.what_would_make_valid or []),

            # Position sizing
            "position_size": _to_dict(decision.position_size),

            # Broker order
            "broker_entry_order": order_dict,
            "broker_entry_order_id": order_dict.get("order_id") or order_dict.get("id", ""),
            "broker_entry_order_status": order_dict.get("status", ""),
            "broker_entry_submitted": bool(order_dict.get("submitted", False)),
            "dry_run": bool(order_dict.get("dry_run", False)),

            # Live tracking
            "current_price": decision.entry_price,
            "highest_price": decision.entry_price,
            "lowest_price": decision.entry_price,
            "unrealized_pl": 0.0,
            "unrealized_pl_percent": 0.0,
            "partial_1_taken": False,
            "breakeven_stop_active": False,
            "trailing_stop_active": False,

            # Exit fields
            "exit_time": "",
            "exit_price": None,
            "exit_quantity": 0,
            "close_reason": "",
            "realized_pl": None,
            "realized_pl_percent": None,
            "broker_exit_order": {},
            "broker_exit_order_id": "",
            "broker_exit_order_status": "",

            # Review / learning
            "backtest_review": {},
            "learning_notes": [],

            # Metadata
            "created_at": now,
            "updated_at": now,
            "extra_context": extra_context or {},
        }

        path = self._open_trade_path(trade_id, ticker)
        _write_json(path, data)

        log.info("[trade_logger] New trade logged: %s -> %s", ticker, path)
        return str(path)

    def update_trade(
        self,
        trade_path: str,
        updates: dict,
    ) -> bool:
        """
        Update an existing trade file with partial updates.
        """
        path = Path(trade_path)
        data = _read_json(path)
        if not data:
            log.warning("[trade_logger] Cannot update missing trade file: %s", path)
            return False

        data.update(updates)
        data["updated_at"] = _now_iso()

        _write_json(path, data)
        return True

    def update_open_trade_price(
        self,
        trade_path: str,
        current_price: float,
    ) -> bool:
        """
        Update current price, high/low, and unrealized P/L for an open trade.
        """
        path = Path(trade_path)
        data = _read_json(path)
        if not data:
            return False

        entry = _safe_float(data.get("entry_price", 0.0))
        qty = _safe_float(data.get("quantity", 0.0))
        price = _safe_float(current_price)

        if entry <= 0 or qty <= 0 or price <= 0:
            return False

        highest = max(_safe_float(data.get("highest_price", entry)), price)
        lowest = min(_safe_float(data.get("lowest_price", entry)), price)

        unrealized = (price - entry) * qty
        unrealized_pct = (price - entry) / entry * 100

        data.update({
            "current_price": price,
            "highest_price": highest,
            "lowest_price": lowest,
            "unrealized_pl": round(unrealized, 2),
            "unrealized_pl_percent": round(unrealized_pct, 4),
            "updated_at": _now_iso(),
        })

        _write_json(path, data)
        return True

    def mark_partial_taken(
        self,
        trade_path: str,
        exit_result: object,
        order_result: Optional[object] = None,
    ) -> bool:
        """
        Mark partial profit as taken and reduce open quantity.
        """
        path = Path(trade_path)
        data = _read_json(path)
        if not data:
            return False

        sell_qty = int(_safe_float(getattr(exit_result, "quantity_to_sell", 0)))
        old_qty = int(_safe_float(data.get("quantity", 0)))
        new_qty = max(old_qty - sell_qty, 0)

        data["quantity"] = new_qty
        data["partial_1_taken"] = True
        data["last_partial_exit"] = _to_dict(exit_result)
        data["last_partial_order"] = _to_dict(order_result)
        data["updated_at"] = _now_iso()

        _write_json(path, data)
        log.info("[trade_logger] Partial exit logged for %s qty=%d", data.get("ticker"), sell_qty)
        return True

    def update_stop(
        self,
        trade_path: str,
        new_stop_price: float,
        reason: str = "",
    ) -> bool:
        """
        Update stop loss after break-even/trailing stop decision.
        """
        path = Path(trade_path)
        data = _read_json(path)
        if not data:
            return False

        data["stop_loss"] = round(float(new_stop_price), 4)
        if reason:
            data.setdefault("stop_updates", []).append({
                "new_stop_price": round(float(new_stop_price), 4),
                "reason": reason,
                "updated_at": _now_iso(),
            })

        if reason.lower().find("breakeven") >= 0:
            data["breakeven_stop_active"] = True
        if reason.lower().find("trail") >= 0:
            data["trailing_stop_active"] = True

        data["updated_at"] = _now_iso()
        _write_json(path, data)
        return True

    def close_trade(
        self,
        trade_path: str,
        exit_price: float,
        exit_quantity: int,
        close_reason: str,
        order_result: Optional[object] = None,
        exit_result: Optional[object] = None,
    ) -> str:
        """
        Mark an open trade closed and move it to the closed folder.

        Returns:
            Path to the closed trade file.
        """
        open_path = Path(trade_path)
        data = _read_json(open_path)
        if not data:
            log.warning("[trade_logger] Cannot close missing trade file: %s", open_path)
            return ""

        entry = _safe_float(data.get("entry_price", 0.0))
        original_qty = _safe_float(data.get("quantity", exit_quantity))
        qty = _safe_float(exit_quantity or original_qty)
        exit_px = _safe_float(exit_price)

        realized = None
        realized_pct = None
        if entry > 0 and qty > 0 and exit_px > 0:
            realized = (exit_px - entry) * qty
            realized_pct = (exit_px - entry) / entry * 100

        now = _now_iso()

        order_dict = _to_dict(order_result)

        data.update({
            "status": "closed",
            "outcome_complete": True,
            "ready_for_backtest_review": True,
            "exit_time": now,
            "exit_price": exit_px,
            "exit_quantity": int(qty),
            "close_reason": close_reason,
            "realized_pl": round(realized, 2) if realized is not None else None,
            "realized_pl_percent": round(realized_pct, 4) if realized_pct is not None else None,
            "broker_exit_order": order_dict,
            "broker_exit_order_id": order_dict.get("order_id") or order_dict.get("id", ""),
            "broker_exit_order_status": order_dict.get("status", ""),
            "exit_decision": _to_dict(exit_result),
            "updated_at": now,
            "closed_at": now,
        })

        closed_path = self._closed_trade_path(
            data.get("trade_id", _make_trade_id()),
            data.get("ticker", "UNKNOWN"),
        )
        _write_json(closed_path, data)

        try:
            open_path.unlink(missing_ok=True)
        except Exception:
            log.warning("[trade_logger] Could not remove open trade file: %s", open_path)

        log.info("[trade_logger] Trade closed: %s -> %s", data.get("ticker"), closed_path)
        return str(closed_path)

    def list_open_trades(self) -> list[dict]:
        """Return all open trade JSON records."""
        return self._list_trades(self._open_dir)

    def list_closed_trades(self) -> list[dict]:
        """Return all closed trade JSON records."""
        return self._list_trades(self._closed_dir)

    def find_open_trade(self, ticker: str) -> Optional[tuple[str, dict]]:
        """
        Find open trade file for a ticker.
        Returns (path, data) or None.
        """
        ticker = ticker.upper()
        for path in self._open_dir.glob("*.json"):
            data = _read_json(path)
            if data and str(data.get("ticker", "")).upper() == ticker:
                return str(path), data
        return None

    # ── Internal path helpers ─────────────────────────────────────────────────

    def _open_trade_path(self, trade_id: str, ticker: str) -> Path:
        return self._open_dir / f"{trade_id}_{ticker.upper()}.json"

    def _closed_trade_path(self, trade_id: str, ticker: str) -> Path:
        return self._closed_dir / f"{trade_id}_{ticker.upper()}.json"

    @staticmethod
    def _list_trades(folder: Path) -> list[dict]:
        trades: list[dict] = []
        for path in sorted(folder.glob("*.json")):
            data = _read_json(path)
            if data:
                data["_file_path"] = str(path)
                trades.append(data)
        return trades


# ── Convenience wrappers ──────────────────────────────────────────────────────

def log_new_trade(
    settings: dict,
    decision: TradeDecisionResult,
    order_result: Optional[object] = None,
    extra_context: Optional[dict] = None,
) -> str:
    """Convenience function for bot_runner.py."""
    logger = TradeLogger(settings)
    return logger.log_new_trade(decision, order_result, extra_context)


def close_trade(
    settings: dict,
    trade_path: str,
    exit_price: float,
    exit_quantity: int,
    close_reason: str,
    order_result: Optional[object] = None,
    exit_result: Optional[object] = None,
) -> str:
    """Convenience function for exit flow."""
    logger = TradeLogger(settings)
    return logger.close_trade(
        trade_path     = trade_path,
        exit_price     = exit_price,
        exit_quantity  = exit_quantity,
        close_reason   = close_reason,
        order_result   = order_result,
        exit_result    = exit_result,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trade_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"TRADE-{ts}-{suffix}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[trade_logger] Failed to read %s: %s", path, exc)
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


def _extract_quantity(decision: TradeDecisionResult) -> int:
    if not decision.position_size:
        return 0
    return int(getattr(decision.position_size, "shares", 0) or 0)


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
