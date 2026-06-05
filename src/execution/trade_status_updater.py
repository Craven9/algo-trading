"""
src/execution/trade_status_updater.py — Trade status synchronization
Synchronizes local trade JSON files with Alpaca paper account positions
and broker order statuses.

The bot writes trade records locally, but the broker is the source of truth
for whether an order filled and whether a position is still open.  This
module checks Alpaca positions/orders and updates local trade files.

Responsibilities:
  - Read open trade JSON files
  - Check whether a ticker still has an open Alpaca position
  - Check entry and exit broker order statuses
  - Normalize Alpaca order status values like "orderstatus.filled"
  - Mark trades closed when exit orders are filled
  - Mark outcome_complete and ready_for_backtest_review
  - Update current price and unrealized P/L while trades remain open
  - Never place orders

Design rules:
  - This file does not submit orders
  - This file does not approve trades
  - This file only updates local trade records
  - Alpaca positions/orders are treated as the source of truth
  - Missing broker data should not delete a trade
  - Filled exit orders should close the trade file
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_OPEN_DIR = Path("data") / "trades" / "open"
_DEFAULT_CLOSED_DIR = Path("data") / "trades" / "closed"

_FILLED_STATUSES = {"filled", "done_for_day"}
_OPEN_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "open"}
_CANCELLED_STATUSES = {"canceled", "cancelled", "expired", "rejected", "stopped"}


# ── Update result ─────────────────────────────────────────────────────────────

class TradeStatusUpdateResult:
    """
    Lightweight result object for one trade update.
    Kept as a simple class so it matches the project's plain-Python style.
    """

    def __init__(
        self,
        trade_id: str = "",
        ticker: str = "",
        old_status: str = "",
        new_status: str = "",
        outcome_complete: bool = False,
        ready_for_backtest_review: bool = False,
        message: str = "",
        updated_path: str = "",
    ):
        self.trade_id = trade_id
        self.ticker = ticker
        self.old_status = old_status
        self.new_status = new_status
        self.outcome_complete = outcome_complete
        self.ready_for_backtest_review = ready_for_backtest_review
        self.message = message
        self.updated_path = updated_path

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "outcome_complete": self.outcome_complete,
            "ready_for_backtest_review": self.ready_for_backtest_review,
            "message": self.message,
            "updated_path": self.updated_path,
        }


# ── Updater ───────────────────────────────────────────────────────────────────

class TradeStatusUpdater:
    """
    Updates local trade JSON files from broker state.

    Usage:
        updater = TradeStatusUpdater(settings, alpaca_client)
        results = updater.update_all_open_trades()
    """

    def __init__(self, settings: dict, alpaca_client: Optional[object] = None):
        self._settings = settings
        self._client = alpaca_client

        paths = settings.get("paths", {})
        trades_dir = Path(paths.get("trades_dir", str(Path("data") / "trades")))

        self._open_dir = Path(paths.get("open_trades_dir", str(trades_dir / "open")))
        self._closed_dir = Path(paths.get("closed_trades_dir", str(trades_dir / "closed")))

        self._open_dir.mkdir(parents=True, exist_ok=True)
        self._closed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_all_open_trades(self) -> list[TradeStatusUpdateResult]:
        """
        Update every open trade file.

        Returns:
            List of TradeStatusUpdateResult.
        """
        trade_files = sorted(self._open_dir.glob("*.json"))
        results: list[TradeStatusUpdateResult] = []

        log.info("[trade_status] Open trade file(s) found: %d", len(trade_files))

        positions = self._fetch_positions()

        for path in trade_files:
            try:
                result = self.update_trade_file(path, positions=positions)
                results.append(result)
            except Exception as exc:
                log.error("[trade_status] Failed to update %s: %s", path, exc)
                results.append(TradeStatusUpdateResult(
                    updated_path=str(path),
                    message=f"update failed: {exc}",
                ))

        return results

    def update_trade_file(
        self,
        trade_path: str | Path,
        positions: Optional[list[dict]] = None,
    ) -> TradeStatusUpdateResult:
        """
        Update one trade file from broker state.

        Args:
            trade_path: Open trade JSON file path.
            positions: Optional already-fetched positions list.

        Returns:
            TradeStatusUpdateResult.
        """
        path = Path(trade_path)
        trade = _read_json(path)

        if not trade:
            return TradeStatusUpdateResult(
                updated_path=str(path),
                message="trade file missing or unreadable",
            )

        trade_id = str(trade.get("trade_id", ""))
        ticker = str(trade.get("ticker", "")).upper()
        old_status = str(trade.get("status", "open"))

        result = TradeStatusUpdateResult(
            trade_id=trade_id,
            ticker=ticker,
            old_status=old_status,
            new_status=old_status,
            outcome_complete=bool(trade.get("outcome_complete", False)),
            ready_for_backtest_review=bool(trade.get("ready_for_backtest_review", False)),
            updated_path=str(path),
        )

        if not ticker:
            result.message = "missing ticker"
            return result

        positions = positions if positions is not None else self._fetch_positions()
        position = _find_position(positions, ticker)

        # Update live price/P&L while position exists.
        if position:
            self._update_open_position_fields(trade, position)
            trade["status"] = "open"
            trade["outcome_complete"] = False
            trade["ready_for_backtest_review"] = False
            trade["updated_at"] = _now_iso()
            _write_json(path, trade)

            result.new_status = "open"
            result.outcome_complete = False
            result.ready_for_backtest_review = False
            result.message = "Open position found."
            return result

        # No position found.  Check exit order status.
        exit_order_id = str(trade.get("broker_exit_order_id", "") or "")
        exit_order_status = normalize_order_status(
            trade.get("broker_exit_order_status", "")
        )

        if exit_order_id and self._client:
            broker_order = self._fetch_order(exit_order_id)
            if broker_order:
                exit_order_status = normalize_order_status(
                    get_order_field(broker_order, "status", "")
                )
                trade["broker_exit_order"] = _order_to_dict(broker_order)
                trade["broker_exit_order_status"] = exit_order_status

                filled_qty = _safe_float(get_order_field(broker_order, "filled_qty", 0.0))
                avg_fill = _safe_float(get_order_field(broker_order, "filled_avg_price", 0.0))

                if filled_qty > 0:
                    trade["exit_quantity"] = int(filled_qty)
                if avg_fill > 0:
                    trade["exit_price"] = avg_fill

        # If the exit order is filled, close the trade.
        if exit_order_status in _FILLED_STATUSES:
            closed_path = self._mark_closed(path, trade, reason="Exit order filled")
            result.new_status = "closed"
            result.outcome_complete = True
            result.ready_for_backtest_review = True
            result.updated_path = str(closed_path)
            result.message = "Exit order filled. Trade is marked closed."
            return result

        # If no position exists and no exit order is known, mark as needs review.
        if not position:
            entry_order_status = normalize_order_status(
                trade.get("broker_entry_order_status", "")
            )

            if entry_order_status in _FILLED_STATUSES:
                trade["status"] = "needs_review"
                trade["outcome_complete"] = False
                trade["ready_for_backtest_review"] = False
                trade["review_warning"] = (
                    "No open position found, but no filled exit order was found. "
                    "Manual review needed."
                )
                trade["updated_at"] = _now_iso()
                _write_json(path, trade)

                result.new_status = "needs_review"
                result.message = "No open position found and no filled exit order found."
                return result

        trade["updated_at"] = _now_iso()
        _write_json(path, trade)

        result.new_status = str(trade.get("status", old_status))
        result.outcome_complete = bool(trade.get("outcome_complete", False))
        result.ready_for_backtest_review = bool(trade.get("ready_for_backtest_review", False))
        result.message = "Trade checked; no closure detected."
        return result

    # ── Broker fetch helpers ──────────────────────────────────────────────────

    def _fetch_positions(self) -> list[dict]:
        """Fetch open positions from Alpaca client."""
        if not self._client:
            return []

        try:
            positions = self._client.get_positions()
            return positions if isinstance(positions, list) else []
        except Exception as exc:
            log.warning("[trade_status] Failed to fetch positions: %s", exc)
            return []

    def _fetch_order(self, order_id: str) -> Optional[object]:
        """Fetch one broker order by ID."""
        if not self._client or not order_id:
            return None

        try:
            return self._client.get_order(order_id)
        except Exception as exc:
            log.warning("[trade_status] Failed to fetch order %s: %s", order_id, exc)
            return None

    # ── Trade update helpers ──────────────────────────────────────────────────

    def _update_open_position_fields(self, trade: dict, position: dict) -> None:
        """Update current price and unrealized P/L from a position dict."""
        current_price = _safe_float(position.get("current_price", 0.0))
        qty = _safe_float(position.get("qty", trade.get("quantity", 0.0)))
        entry = _safe_float(
            trade.get("entry_price")
            or position.get("avg_entry_price")
            or 0.0
        )

        if current_price > 0:
            trade["current_price"] = current_price
            trade["highest_price"] = max(
                _safe_float(trade.get("highest_price", current_price)),
                current_price,
            )
            trade["lowest_price"] = min(
                _safe_float(trade.get("lowest_price", current_price)),
                current_price,
            )

        if entry > 0 and qty > 0 and current_price > 0:
            trade["unrealized_pl"] = round((current_price - entry) * qty, 2)
            trade["unrealized_pl_percent"] = round(
                (current_price - entry) / entry * 100,
                4,
            )

        trade["quantity"] = int(qty) if qty > 0 else trade.get("quantity", 0)
        trade["position_snapshot"] = dict(position)

    def _mark_closed(self, open_path: Path, trade: dict, reason: str) -> Path:
        """Mark a trade closed and move it to closed folder."""
        entry = _safe_float(trade.get("entry_price", 0.0))
        exit_price = _safe_float(trade.get("exit_price", trade.get("current_price", 0.0)))
        qty = _safe_float(trade.get("exit_quantity", trade.get("quantity", 0.0)))

        if exit_price <= 0:
            exit_price = _safe_float(trade.get("current_price", 0.0))

        realized = None
        realized_pct = None
        if entry > 0 and exit_price > 0 and qty > 0:
            realized = (exit_price - entry) * qty
            realized_pct = (exit_price - entry) / entry * 100

        now = _now_iso()

        trade.update({
            "status": "closed",
            "outcome_complete": True,
            "ready_for_backtest_review": True,
            "close_reason": trade.get("close_reason") or reason,
            "exit_time": trade.get("exit_time") or now,
            "exit_price": exit_price if exit_price > 0 else trade.get("exit_price"),
            "exit_quantity": int(qty) if qty > 0 else trade.get("exit_quantity", 0),
            "realized_pl": round(realized, 2) if realized is not None else trade.get("realized_pl"),
            "realized_pl_percent": (
                round(realized_pct, 4)
                if realized_pct is not None
                else trade.get("realized_pl_percent")
            ),
            "updated_at": now,
            "closed_at": now,
        })

        trade_id = str(trade.get("trade_id") or open_path.stem)
        ticker = str(trade.get("ticker") or "UNKNOWN").upper()
        closed_path = self._closed_dir / f"{trade_id}_{ticker}.json"

        _write_json(closed_path, trade)

        try:
            open_path.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("[trade_status] Could not remove open trade file %s: %s", open_path, exc)

        return closed_path


# ── Convenience wrapper ───────────────────────────────────────────────────────

def update_trade_statuses(
    settings: dict,
    alpaca_client: Optional[object] = None,
) -> list[TradeStatusUpdateResult]:
    """
    Convenience function for bot_runner.py or a CLI script.
    """
    updater = TradeStatusUpdater(settings, alpaca_client)
    return updater.update_all_open_trades()


# ── Normalization helpers ─────────────────────────────────────────────────────

def normalize_order_status(order_or_status: Any) -> str:
    """
    Normalize Alpaca order status values.

    Handles examples:
      "filled"
      "OrderStatus.FILLED"
      "orderstatus.filled"
      "OrderStatus.PARTIALLY_FILLED"

    Returns:
        lowercase status like "filled", "partially_filled", "new"
    """
    if order_or_status is None:
        return ""

    if isinstance(order_or_status, dict):
        raw = str(order_or_status.get("status", ""))
    else:
        raw = str(order_or_status)

    raw = raw.strip()
    if not raw:
        return ""

    # Convert enum-like strings.
    if "." in raw:
        raw = raw.split(".")[-1]

    raw = raw.lower().replace(" ", "_").replace("-", "_")

    # Common normalize cases.
    if raw == "partiallyfilled":
        raw = "partially_filled"
    if raw == "doneforday":
        raw = "done_for_day"

    return raw


def get_order_field(order: Any, field: str, default: Any = "") -> Any:
    """
    Read a field from an Alpaca order object or dict.
    """
    if order is None:
        return default

    if isinstance(order, dict):
        return order.get(field, default)

    return getattr(order, field, default)


def _order_to_dict(order: Any) -> dict:
    """Convert order object/dict to plain dict."""
    if order is None:
        return {}
    if isinstance(order, dict):
        return dict(order)

    if hasattr(order, "model_dump"):
        try:
            return order.model_dump()
        except Exception:
            pass

    if hasattr(order, "dict"):
        try:
            return order.dict()
        except Exception:
            pass

    data = {}
    for field in [
        "id", "client_order_id", "symbol", "qty", "filled_qty",
        "filled_avg_price", "side", "type", "status", "limit_price",
        "submitted_at", "filled_at", "canceled_at", "failed_at",
    ]:
        value = getattr(order, field, None)
        if value is not None:
            data[field] = str(value)
    return data


def _find_position(positions: list[dict], ticker: str) -> Optional[dict]:
    """Return position dict for ticker if found."""
    ticker = ticker.upper()
    for pos in positions:
        symbol = str(pos.get("symbol") or pos.get("ticker") or "").upper()
        qty = _safe_float(pos.get("qty", 0.0))
        if symbol == ticker and abs(qty) > 0:
            return pos
    return None


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[trade_status] Failed to read %s: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
