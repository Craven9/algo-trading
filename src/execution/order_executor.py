"""
src/execution/order_executor.py — Paper order execution layer
Submits paper-trading orders to Alpaca only after trade_quality_gate.py
has approved the trade.

This file is intentionally strict.  It should never decide whether a trade
is good.  It only executes an already-approved TradeDecisionResult.

Responsibilities:
  - Refuse execution unless decision is APPROVED_FOR_PAPER_BUY
  - Refuse execution when paper trading safety is not confirmed
  - Submit buy orders to Alpaca paper account
  - Support dry-run mode for testing without broker submission
  - Use marketable limit orders by default to reduce bad fills
  - Normalize broker response into OrderExecutionResult
  - Log submitted, rejected, failed, and dry-run orders
  - Never submit live-money orders

Design rules:
  - This file does not scan
  - This file does not score
  - This file does not approve trades
  - This file does not size trades
  - The trade decision must already be approved
  - Paper trading safety always overrides execution
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    TradingClient = None
    LimitOrderRequest = None
    MarketOrderRequest = None
    OrderSide = None
    TimeInForce = None

from models import TradeDecision, TradeDecisionResult

log = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OrderExecutionResult:
    """
    Result returned after attempting to execute a paper order.
    """
    ticker:          str
    side:            str = "buy"
    quantity:        int = 0
    order_type:      str = "marketable_limit"

    submitted:       bool = False
    dry_run:         bool = False
    order_id:        str = ""
    status:          str = "not_submitted"
    message:         str = ""

    requested_price: Optional[float] = None
    limit_price:     Optional[float] = None

    raw_order:       dict = field(default_factory=dict)
    error:           str  = ""

    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "side":            self.side,
            "quantity":        self.quantity,
            "order_type":      self.order_type,
            "submitted":       self.submitted,
            "dry_run":         self.dry_run,
            "order_id":        self.order_id,
            "status":          self.status,
            "message":         self.message,
            "requested_price": round(self.requested_price, 4) if self.requested_price else None,
            "limit_price":     round(self.limit_price, 4) if self.limit_price else None,
            "raw_order":       self.raw_order,
            "error":           self.error,
            "submitted_at":    self.submitted_at,
        }


# ── Executor ──────────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Executes approved paper orders.

    Usage:
        executor = OrderExecutor(settings)
        result = executor.execute_buy(decision)
    """

    def __init__(self, settings: dict, trading_client: Optional[object] = None):
        self._settings = settings
        self._mode     = settings.get("mode", {})
        self._exec     = settings.get("execution", {})

        self._dry_run = bool(self._mode.get("dry_run", True))
        self._paper_only = bool(self._mode.get("paper_trading_only", True))
        self._allow_live_money = bool(self._mode.get("allow_live_money", False))

        self._order_type = str(
            self._exec.get("default_order_type", "marketable_limit")
        )
        self._limit_buffer_pct = float(
            self._exec.get("marketable_limit_buffer_percent", 0.30)
        )
        self._time_in_force = str(
            self._exec.get("time_in_force", "day")
        )

        self._client = trading_client

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_buy(
        self,
        decision: TradeDecisionResult,
        current_ask: Optional[float] = None,
    ) -> OrderExecutionResult:
        """
        Execute an approved buy decision.

        Args:
            decision:    TradeDecisionResult from trade_quality_gate.py.
            current_ask: Optional current ask price for limit calculation.

        Returns:
            OrderExecutionResult.
        """
        ticker = decision.ticker
        qty = self._extract_quantity(decision)
        entry_price = float(decision.entry_price or 0.0)

        result = OrderExecutionResult(
            ticker          = ticker,
            side            = "buy",
            quantity        = qty,
            order_type      = self._order_type,
            dry_run         = self._dry_run,
            requested_price = entry_price,
        )

        # ── Safety checks ─────────────────────────────────────────────────────
        safety_error = self._safety_error(decision, qty)
        if safety_error:
            result.status = "blocked"
            result.message = safety_error
            result.error = safety_error
            log.warning("[order_executor] %s blocked: %s", ticker, safety_error)
            return result

        # ── Dry run path ──────────────────────────────────────────────────────
        if self._dry_run:
            result.submitted = False
            result.status = "dry_run"
            result.order_id = self._dry_run_order_id("BUY", ticker)
            result.limit_price = self._limit_price(entry_price, current_ask)
            result.message = (
                f"DRY RUN: BUY {self._order_type} order was simulated for "
                f"{qty} share(s) of {ticker}. No Alpaca order was submitted."
            )
            result.raw_order = {
                "dry_run": True,
                "id": result.order_id,
                "symbol": ticker,
                "qty": qty,
                "side": "buy",
                "order_type": self._order_type,
                "requested_price": entry_price,
                "limit_price": result.limit_price,
            }
            log.info("[order_executor] %s", result.message)
            return result

        # ── Live paper submit ─────────────────────────────────────────────────
        client = self._get_client()
        if client is None:
            result.status = "failed"
            result.error = "Trading client unavailable"
            result.message = result.error
            return result

        try:
            order_request = self._build_buy_order_request(
                ticker=ticker,
                qty=qty,
                entry_price=entry_price,
                current_ask=current_ask,
                result=result,
            )

            order = client.submit_order(order_request)

            result.submitted = True
            result.status = _get_order_field(order, "status", "submitted")
            result.order_id = _get_order_field(order, "id", "")
            result.message = (
                f"Submitted BUY order for {qty} share(s) of {ticker}"
            )
            result.raw_order = _order_to_dict(order)

            log.info(
                "[order_executor] Submitted BUY %s qty=%d order_id=%s status=%s",
                ticker, qty, result.order_id, result.status,
            )
            return result

        except Exception as exc:
            result.submitted = False
            result.status = "failed"
            result.error = str(exc)
            result.message = f"Order submission failed: {exc}"
            log.error("[order_executor] %s order failed: %s", ticker, exc)
            return result

    def execute_sell(
        self,
        ticker: str,
        quantity: int,
        reason: str = "",
        current_bid: Optional[float] = None,
    ) -> OrderExecutionResult:
        """
        Execute a paper sell order.  Used by exit_manager.py.
        """
        result = OrderExecutionResult(
            ticker     = ticker.upper(),
            side       = "sell",
            quantity   = int(quantity or 0),
            order_type = self._order_type,
            dry_run    = self._dry_run,
        )

        if result.quantity <= 0:
            result.status = "blocked"
            result.error = "sell quantity must be greater than 0"
            result.message = result.error
            return result

        if not self._paper_safety_ok():
            result.status = "blocked"
            result.error = "paper trading safety check failed"
            result.message = result.error
            return result

        if self._dry_run:
            result.status = "dry_run"
            result.order_id = self._dry_run_order_id("SELL", ticker)
            result.limit_price = self._sell_limit_price(current_bid)
            result.message = (
                f"DRY RUN: SELL {self._order_type} order was simulated for "
                f"{quantity} share(s) of {ticker}. Reason: {reason}"
            )
            result.raw_order = {
                "dry_run": True,
                "id": result.order_id,
                "symbol": ticker.upper(),
                "qty": quantity,
                "side": "sell",
                "order_type": self._order_type,
                "reason": reason,
                "limit_price": result.limit_price,
            }
            log.info("[order_executor] %s", result.message)
            return result

        client = self._get_client()
        if client is None:
            result.status = "failed"
            result.error = "Trading client unavailable"
            result.message = result.error
            return result

        try:
            order_request = self._build_sell_order_request(
                ticker=ticker.upper(),
                qty=quantity,
                current_bid=current_bid,
                result=result,
            )

            order = client.submit_order(order_request)

            result.submitted = True
            result.status = _get_order_field(order, "status", "submitted")
            result.order_id = _get_order_field(order, "id", "")
            result.message = (
                f"Submitted SELL order for {quantity} share(s) of {ticker.upper()}"
            )
            result.raw_order = _order_to_dict(order)

            log.info(
                "[order_executor] Submitted SELL %s qty=%d order_id=%s status=%s",
                ticker.upper(), quantity, result.order_id, result.status,
            )
            return result

        except Exception as exc:
            result.submitted = False
            result.status = "failed"
            result.error = str(exc)
            result.message = f"Sell order submission failed: {exc}"
            log.error("[order_executor] %s sell failed: %s", ticker.upper(), exc)
            return result

    # ── Order builders ────────────────────────────────────────────────────────

    def _build_buy_order_request(
        self,
        ticker: str,
        qty: int,
        entry_price: float,
        current_ask: Optional[float],
        result: OrderExecutionResult,
    ):
        """Build Alpaca buy order request."""
        tif = _time_in_force(self._time_in_force)

        if self._order_type == "market":
            return MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=tif,
            )

        limit_price = self._limit_price(entry_price, current_ask)
        result.limit_price = limit_price

        return LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=tif,
            limit_price=limit_price,
        )

    def _build_sell_order_request(
        self,
        ticker: str,
        qty: int,
        current_bid: Optional[float],
        result: OrderExecutionResult,
    ):
        """Build Alpaca sell order request."""
        tif = _time_in_force(self._time_in_force)

        if self._order_type == "market":
            return MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=tif,
            )

        limit_price = self._sell_limit_price(current_bid)
        result.limit_price = limit_price

        return LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=tif,
            limit_price=limit_price,
        )

    # ── Safety helpers ────────────────────────────────────────────────────────

    def _safety_error(self, decision: TradeDecisionResult, qty: int) -> Optional[str]:
        """
        Return safety error string, or None if safe to submit.
        """
        if decision.decision != TradeDecision.APPROVED_FOR_PAPER_BUY.value:
            return f"trade decision is not approved: {decision.decision}"

        if qty <= 0:
            return "quantity must be greater than 0"

        if not self._paper_safety_ok():
            return "paper trading safety check failed"

        if decision.entry_price is None or decision.entry_price <= 0:
            return "entry price is invalid"

        return None

    def _paper_safety_ok(self) -> bool:
        """
        True only when live-money trading is not allowed and paper-only mode is active.
        """
        if self._allow_live_money:
            return False
        if not self._paper_only:
            return False
        return True

    def _get_client(self):
        """
        Return an Alpaca TradingClient.
        """
        if self._client is not None:
            return self._client

        if TradingClient is None:
            log.error("[order_executor] alpaca-py package unavailable")
            return None

        try:
            api_key = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

            if not api_key or not secret_key:
                log.error("[order_executor] ALPACA_API_KEY / ALPACA_SECRET_KEY missing")
                return None

            self._client = TradingClient(api_key, secret_key, paper=True)
            return self._client

        except Exception as exc:
            log.error("[order_executor] Failed to create TradingClient: %s", exc)
            return None

    # ── Price helpers ─────────────────────────────────────────────────────────

    def _limit_price(
        self,
        entry_price: float,
        current_ask: Optional[float],
    ) -> float:
        """
        Buy marketable limit price.
        Uses current ask if available, else planned entry.
        """
        base_price = float(current_ask or entry_price or 0.0)
        if base_price <= 0:
            return 0.0
        limit = base_price * (1 + self._limit_buffer_pct / 100.0)
        return round(limit, 4)

    def _sell_limit_price(self, current_bid: Optional[float]) -> Optional[float]:
        """
        Sell marketable limit price.
        Uses a small buffer below bid.
        """
        if not current_bid or current_bid <= 0:
            return None
        limit = current_bid * (1 - self._limit_buffer_pct / 100.0)
        return round(limit, 4)

    @staticmethod
    def _extract_quantity(decision: TradeDecisionResult) -> int:
        """
        Extract quantity from decision.position_size.
        """
        if not decision.position_size:
            return 0
        return int(getattr(decision.position_size, "shares", 0) or 0)

    @staticmethod
    def _dry_run_order_id(side: str, ticker: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"DRY-RUN-{side}-{ticker.upper()}-{ts}"


# ── Convenience wrappers ──────────────────────────────────────────────────────

def execute_buy_order(
    settings: dict,
    decision: TradeDecisionResult,
    current_ask: Optional[float] = None,
    trading_client: Optional[object] = None,
) -> OrderExecutionResult:
    """
    Convenience function for bot_runner.py.
    """
    executor = OrderExecutor(settings, trading_client=trading_client)
    return executor.execute_buy(decision, current_ask=current_ask)


def execute_sell_order(
    settings: dict,
    ticker: str,
    quantity: int,
    reason: str = "",
    current_bid: Optional[float] = None,
    trading_client: Optional[object] = None,
) -> OrderExecutionResult:
    """
    Convenience function for exit_manager.py.
    """
    executor = OrderExecutor(settings, trading_client=trading_client)
    return executor.execute_sell(
        ticker=ticker,
        quantity=quantity,
        reason=reason,
        current_bid=current_bid,
    )


# ── Generic helpers ───────────────────────────────────────────────────────────

def _time_in_force(value: str):
    """Convert config time-in-force string to Alpaca enum."""
    if TimeInForce is None:
        return value

    value = (value or "day").lower()

    if value == "gtc":
        return TimeInForce.GTC
    if value == "ioc":
        return TimeInForce.IOC
    if value == "fok":
        return TimeInForce.FOK
    return TimeInForce.DAY


def _get_order_field(order: object, field: str, default: str = "") -> str:
    """Read a field from Alpaca order object or dict."""
    if order is None:
        return default
    if isinstance(order, dict):
        return str(order.get(field, default))
    return str(getattr(order, field, default))


def _order_to_dict(order: object) -> dict:
    """
    Convert Alpaca order object to dict as safely as possible.
    """
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

    raw = {}
    for field in [
        "id", "client_order_id", "created_at", "updated_at", "submitted_at",
        "filled_at", "expired_at", "canceled_at", "failed_at", "asset_id",
        "symbol", "qty", "filled_qty", "type", "side", "time_in_force",
        "limit_price", "stop_price", "filled_avg_price", "status",
    ]:
        value = getattr(order, field, None)
        if value is not None:
            raw[field] = str(value)
    return raw
