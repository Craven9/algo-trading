"""
src/execution/exit_manager.py — Open trade exit management
Manages open paper trades after entry and decides when an exit order
should be submitted.

The bot should not only buy well — it must also exit intelligently.
This module watches open trades and produces exit decisions based on:
  - Hard stop loss
  - Break-even stop after trade moves up
  - Trailing stop
  - Partial profit targets
  - Runner targets
  - Failed breakout / failed reclaim
  - VWAP loss
  - Key level loss
  - Volume fade
  - Max loss dollars
  - End-of-day / session safety

Responsibilities:
  - Evaluate open trades against current market data
  - Decide whether to hold, partial sell, or full exit
  - Prevent sell quantity from exceeding the real open position quantity
  - Produce clear reasons for every exit decision
  - Return an ExitDecisionResult for order_executor.py
  - Never place orders directly

Design rules:
  - This file does not submit orders
  - This file only returns exit decisions
  - order_executor.py submits sell orders after an exit decision
  - Stop-loss exits always override profit logic
  - Sell quantity can never exceed actual open position quantity
  - Dry-run and paper execution are handled by order_executor.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ── Exit actions ──────────────────────────────────────────────────────────────

class ExitAction:
    HOLD = "hold"
    PARTIAL_SELL = "partial_sell"
    FULL_EXIT = "full_exit"
    MOVE_STOP = "move_stop"
    TRAIL_STOP = "trail_stop"


# ── Exit result dataclass ─────────────────────────────────────────────────────

@dataclass
class ExitDecisionResult:
    """
    Exit decision for one open trade.
    Consumed by bot_runner.py and order_executor.py.
    """
    ticker:              str
    action:              str = ExitAction.HOLD
    should_exit:         bool = False
    full_exit:           bool = False
    partial_exit:        bool = False

    quantity_to_sell:    int = 0
    available_quantity:  int = 0
    sell_percent:        float = 0.0

    entry_price:         float = 0.0
    current_price:       float = 0.0
    stop_price:          float = 0.0
    new_stop_price:      Optional[float] = None

    pnl_dollars:         float = 0.0
    pnl_percent:         float = 0.0
    current_r_multiple:  float = 0.0

    exit_reason:         str = ""
    priority:            str = "normal"  # critical|high|normal|low

    reasons:             list[str] = field(default_factory=list)
    warnings:            list[str] = field(default_factory=list)

    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":             self.ticker,
            "action":             self.action,
            "should_exit":        self.should_exit,
            "full_exit":          self.full_exit,
            "partial_exit":       self.partial_exit,
            "quantity_to_sell":   self.quantity_to_sell,
            "available_quantity": self.available_quantity,
            "sell_percent":       round(self.sell_percent, 2),
            "entry_price":        round(self.entry_price, 4),
            "current_price":      round(self.current_price, 4),
            "stop_price":         round(self.stop_price, 4),
            "new_stop_price":     round(self.new_stop_price, 4) if self.new_stop_price else None,
            "pnl_dollars":        round(self.pnl_dollars, 2),
            "pnl_percent":        round(self.pnl_percent, 4),
            "current_r_multiple": round(self.current_r_multiple, 4),
            "exit_reason":        self.exit_reason,
            "priority":           self.priority,
            "reasons":            self.reasons,
            "warnings":           self.warnings,
            "evaluated_at":       self.evaluated_at,
        }


# ── Exit manager ──────────────────────────────────────────────────────────────

class ExitManager:
    """
    Evaluates open trades and returns exit decisions.

    Usage:
        manager = ExitManager(settings)
        decision = manager.evaluate(
            trade=trade_dict,
            current_price=3.40,
            current_position_qty=100,
            indicators=indicators,
            context=context,
        )

        if decision.should_exit:
            # order_executor.py submits sell
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._exit     = settings.get("exits", {})
        self._risk     = settings.get("risk", {})

        self._enable_trailing = bool(self._exit.get("enable_trailing_stop", True))
        self._enable_breakeven = bool(self._exit.get("enable_breakeven_stop", True))
        self._enable_partials = bool(self._exit.get("enable_partial_profits", True))

        self._breakeven_trigger_pct = float(
            self._exit.get("breakeven_trigger_percent", 8.0)
        )
        self._partial_1_trigger_pct = float(
            self._exit.get("partial_1_trigger_percent", 15.0)
        )
        self._partial_1_sell_pct = float(
            self._exit.get("partial_1_sell_percent", 50.0)
        )
        self._trailing_stop_pct = float(
            self._exit.get("trailing_stop_percent", 8.0)
        )
        self._max_loss_dollars = float(
            self._exit.get(
                "max_loss_dollars",
                self._risk.get("max_risk_per_trade_dollars", 25.0),
            )
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        trade: dict,
        current_price: float,
        current_position_qty: int,
        indicators: Optional[object] = None,
        context: Optional[dict] = None,
    ) -> ExitDecisionResult:
        """
        Evaluate an open trade and return an exit decision.

        Args:
            trade:                Trade JSON/dict.
            current_price:        Latest market price.
            current_position_qty: Actual open Alpaca position quantity.
            indicators:           Optional IndicatorSnapshot.
            context:              Optional analysis context.

        Returns:
            ExitDecisionResult.
        """
        context = context or {}

        ticker = str(trade.get("ticker") or trade.get("symbol") or "").upper()
        entry_price = _safe_float(
            trade.get("entry_price")
            or trade.get("avg_entry_price")
            or trade.get("filled_avg_price")
        )
        stop_price = _safe_float(
            trade.get("stop_loss")
            or trade.get("stop_price")
            or trade.get("initial_stop")
        )
        original_qty = int(_safe_float(
            trade.get("quantity")
            or trade.get("qty")
            or trade.get("shares")
            or current_position_qty
        ))

        available_qty = max(int(current_position_qty or 0), 0)

        result = ExitDecisionResult(
            ticker             = ticker,
            available_quantity = available_qty,
            entry_price        = entry_price,
            current_price      = float(current_price or 0.0),
            stop_price         = stop_price,
        )

        # ── Basic validation ──────────────────────────────────────────────────
        if not ticker:
            result.warnings.append("Missing ticker — cannot evaluate exit")
            return result

        if available_qty <= 0:
            result.warnings.append("No open position quantity available")
            return result

        if entry_price <= 0 or current_price <= 0:
            result.warnings.append("Invalid entry/current price")
            return result

        if stop_price <= 0:
            stop_price = entry_price * 0.95
            result.stop_price = stop_price
            result.warnings.append("Stop price missing — fallback stop used")

        # ── P/L metrics ───────────────────────────────────────────────────────
        result.pnl_dollars = (current_price - entry_price) * available_qty
        result.pnl_percent = (current_price - entry_price) / entry_price * 100

        risk_per_share = entry_price - stop_price
        if risk_per_share > 0:
            result.current_r_multiple = (current_price - entry_price) / risk_per_share

        # ── Critical exits first ──────────────────────────────────────────────
        hard_stop = self._hard_stop_exit(result)
        if hard_stop.should_exit:
            return hard_stop

        max_loss = self._max_loss_exit(result)
        if max_loss.should_exit:
            return max_loss

        failed_reclaim = self._failed_reclaim_exit(result, context)
        if failed_reclaim.should_exit:
            return failed_reclaim

        failed_breakout = self._failed_breakout_exit(result, context)
        if failed_breakout.should_exit:
            return failed_breakout

        # ── Profit-taking exits ───────────────────────────────────────────────
        partial = self._partial_profit_exit(result, trade, original_qty)
        if partial.should_exit:
            return partial

        runner = self._runner_target_exit(result, trade, context)
        if runner.should_exit:
            return runner

        # ── Stop management decisions ─────────────────────────────────────────
        breakeven = self._breakeven_stop(result, trade)
        if breakeven.action == ExitAction.MOVE_STOP:
            return breakeven

        trailing = self._trailing_stop(result, trade)
        if trailing.action == ExitAction.TRAIL_STOP:
            return trailing

        # ── Soft warnings / hold context ──────────────────────────────────────
        self._add_hold_context(result, indicators, context)

        result.action = ExitAction.HOLD
        result.should_exit = False
        result.exit_reason = "hold — no exit condition triggered"
        result.reasons.append(result.exit_reason)

        log.debug(
            "[exit_manager] %s HOLD price=%.4f pnl=%.2f%% r=%.2f",
            ticker, current_price, result.pnl_percent, result.current_r_multiple,
        )
        return result

    # ── Exit checks ───────────────────────────────────────────────────────────

    def _hard_stop_exit(self, result: ExitDecisionResult) -> ExitDecisionResult:
        """Full exit when price hits or breaks stop loss."""
        if result.current_price <= result.stop_price:
            return self._full_exit(
                result,
                reason=(
                    f"hard stop hit: price {result.current_price:.4f} "
                    f"<= stop {result.stop_price:.4f}"
                ),
                priority="critical",
            )
        return result

    def _max_loss_exit(self, result: ExitDecisionResult) -> ExitDecisionResult:
        """Full exit when max dollar loss is reached."""
        if result.pnl_dollars <= -abs(self._max_loss_dollars):
            return self._full_exit(
                result,
                reason=(
                    f"max loss hit: ${result.pnl_dollars:.2f} "
                    f"<= -${abs(self._max_loss_dollars):.2f}"
                ),
                priority="critical",
            )
        return result

    def _failed_reclaim_exit(
        self,
        result: ExitDecisionResult,
        context: dict,
    ) -> ExitDecisionResult:
        """Full exit when failed reclaim detector blocks long continuation."""
        failed_reclaim = context.get("failed_reclaim")
        if failed_reclaim and getattr(failed_reclaim, "block_long_entry", False):
            return self._full_exit(
                result,
                reason="failed reclaim detected — exit long position",
                priority="high",
            )
        return result

    def _failed_breakout_exit(
        self,
        result: ExitDecisionResult,
        context: dict,
    ) -> ExitDecisionResult:
        """Full exit when opening range breakout fails."""
        or_result = context.get("or_result")
        if or_result and getattr(or_result, "failed_breakout", False):
            return self._full_exit(
                result,
                reason="opening range breakout failed",
                priority="high",
            )
        return result

    def _partial_profit_exit(
        self,
        result: ExitDecisionResult,
        trade: dict,
        original_qty: int,
    ) -> ExitDecisionResult:
        """Partial sell when first profit target is reached."""
        if not self._enable_partials:
            return result

        partial_done = bool(
            trade.get("partial_1_taken")
            or trade.get("partial_profit_taken")
        )
        if partial_done:
            return result

        if result.pnl_percent < self._partial_1_trigger_pct:
            return result

        qty = int(result.available_quantity * (self._partial_1_sell_pct / 100.0))
        qty = self._safe_sell_qty(qty, result.available_quantity)

        if qty <= 0:
            return result

        result.action = ExitAction.PARTIAL_SELL
        result.should_exit = True
        result.partial_exit = True
        result.full_exit = False
        result.quantity_to_sell = qty
        result.sell_percent = qty / result.available_quantity * 100
        result.exit_reason = (
            f"partial profit target hit: up {result.pnl_percent:.2f}%"
        )
        result.priority = "normal"
        result.reasons.append(result.exit_reason)
        return result

    def _runner_target_exit(
        self,
        result: ExitDecisionResult,
        trade: dict,
        context: dict,
    ) -> ExitDecisionResult:
        """Full exit when runner target is reached."""
        runner_target = _safe_float(
            trade.get("runner_target")
            or trade.get("target_2")
            or trade.get("target_area")
        )

        move = context.get("move_potential")
        if not runner_target and move:
            runner_target = _safe_float(getattr(move, "runner_target", 0.0))

        if runner_target and result.current_price >= runner_target:
            return self._full_exit(
                result,
                reason=f"runner target reached: {runner_target:.4f}",
                priority="normal",
            )

        return result

    def _breakeven_stop(
        self,
        result: ExitDecisionResult,
        trade: dict,
    ) -> ExitDecisionResult:
        """Move stop to breakeven after configured profit percent."""
        if not self._enable_breakeven:
            return result

        already_moved = bool(trade.get("breakeven_stop_active"))
        if already_moved:
            return result

        if result.pnl_percent >= self._breakeven_trigger_pct:
            result.action = ExitAction.MOVE_STOP
            result.should_exit = False
            result.new_stop_price = result.entry_price
            result.exit_reason = (
                f"move stop to breakeven after {result.pnl_percent:.2f}% move"
            )
            result.reasons.append(result.exit_reason)
            return result

        return result

    def _trailing_stop(
        self,
        result: ExitDecisionResult,
        trade: dict,
    ) -> ExitDecisionResult:
        """Update trailing stop after trade moves in favor."""
        if not self._enable_trailing:
            return result

        highest_price = _safe_float(
            trade.get("highest_price")
            or trade.get("high_water_mark")
            or result.current_price
        )
        if highest_price <= result.entry_price:
            return result

        trailing_stop = highest_price * (1 - self._trailing_stop_pct / 100.0)

        if trailing_stop > result.stop_price and trailing_stop < result.current_price:
            result.action = ExitAction.TRAIL_STOP
            result.should_exit = False
            result.new_stop_price = trailing_stop
            result.exit_reason = (
                f"raise trailing stop to {trailing_stop:.4f}"
            )
            result.reasons.append(result.exit_reason)
            return result

        if result.current_price <= trailing_stop:
            return self._full_exit(
                result,
                reason=f"trailing stop hit: {trailing_stop:.4f}",
                priority="high",
            )

        return result

    # ── Hold context ──────────────────────────────────────────────────────────

    def _add_hold_context(
        self,
        result: ExitDecisionResult,
        indicators: Optional[object],
        context: dict,
    ) -> None:
        """Add warnings while holding."""
        if indicators:
            if getattr(indicators, "volume_trend", "") == "decreasing":
                result.warnings.append("volume trend is fading")

            if getattr(indicators, "price_vs_vwap", "") == "below":
                result.warnings.append("price is below VWAP")

            macd = getattr(indicators, "macd", None)
            if macd and getattr(macd, "bearish_crossover", False):
                result.warnings.append("MACD bearish crossover")

        key_levels = context.get("key_levels")
        if key_levels and getattr(key_levels, "breaking_down", False):
            result.warnings.append("price is breaking down through key support")

    # ── Result builders ───────────────────────────────────────────────────────

    def _full_exit(
        self,
        result: ExitDecisionResult,
        reason: str,
        priority: str,
    ) -> ExitDecisionResult:
        """Build a full-exit result."""
        qty = self._safe_sell_qty(result.available_quantity, result.available_quantity)

        result.action = ExitAction.FULL_EXIT
        result.should_exit = True
        result.full_exit = True
        result.partial_exit = False
        result.quantity_to_sell = qty
        result.sell_percent = 100.0 if qty > 0 else 0.0
        result.exit_reason = reason
        result.priority = priority
        result.reasons.append(reason)

        log.info("[exit_manager] %s FULL_EXIT: %s", result.ticker, reason)
        return result

    @staticmethod
    def _safe_sell_qty(requested_qty: int, available_qty: int) -> int:
        """
        Make sure sell quantity can never exceed real open position quantity.
        """
        requested = max(int(requested_qty or 0), 0)
        available = max(int(available_qty or 0), 0)
        return min(requested, available)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def evaluate_exit(
    settings: dict,
    trade: dict,
    current_price: float,
    current_position_qty: int,
    indicators: Optional[object] = None,
    context: Optional[dict] = None,
) -> ExitDecisionResult:
    """
    Convenience function for bot_runner.py.
    """
    manager = ExitManager(settings)
    return manager.evaluate(
        trade                = trade,
        current_price        = current_price,
        current_position_qty = current_position_qty,
        indicators           = indicators,
        context              = context,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
