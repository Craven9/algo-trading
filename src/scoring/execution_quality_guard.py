"""
src/scoring/execution_quality_guard.py — Final execution quality check
Checks execution-level conditions immediately before an order is placed.

A trade can have a strong setup and high probability but still be a bad
execution if the spread widens, the quote is stale, price runs too far
from the planned entry, buying power is unavailable, or the bot already
has a position.

Responsibilities:
  - Confirm spread is still within limits
  - Confirm quote is fresh enough to trade
  - Confirm price has not moved too far from planned entry
  - Confirm no existing position is already open
  - Confirm paper trading is active
  - Confirm buying power is available
  - Produce an execution_quality_score for probability_engine.py
  - Produce hard_block when execution is unsafe

Design rules:
  - This file does not approve trades by itself
  - This file does not place orders
  - This file is the last quality check before order_executor.py
  - trade_quality_gate.py reads hard_block before allowing execution
  - Safety checks always win over score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExecutionQualityResult:
    """
    Execution quality assessment for a planned trade.
    Consumed by probability_engine.py and trade_quality_gate.py.
    """
    ticker:                  str
    execution_quality_score: float = 0.0

    # Core fields
    planned_entry:           float = 0.0
    current_price:           float = 0.0
    bid:                     float = 0.0
    ask:                     float = 0.0
    spread_pct:              float = 0.0
    quote_age_seconds:       float = 0.0

    # Account / position state
    paper_trading_confirmed: bool = False
    position_already_open:   bool = False
    buying_power_available:  bool = True

    # Flags
    quote_fresh:             bool = False
    spread_ok:               bool = False
    price_chase_ok:          bool = False
    hard_block:              bool = False

    reasons:                 list[str] = field(default_factory=list)
    warnings:                list[str] = field(default_factory=list)

    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker":                  self.ticker,
            "execution_quality_score": round(self.execution_quality_score, 2),
            "planned_entry":           round(self.planned_entry,           4),
            "current_price":           round(self.current_price,           4),
            "bid":                     round(self.bid,                     4),
            "ask":                     round(self.ask,                     4),
            "spread_pct":              round(self.spread_pct,              4),
            "quote_age_seconds":       round(self.quote_age_seconds,       2),
            "paper_trading_confirmed": self.paper_trading_confirmed,
            "position_already_open":   self.position_already_open,
            "buying_power_available":  self.buying_power_available,
            "quote_fresh":             self.quote_fresh,
            "spread_ok":               self.spread_ok,
            "price_chase_ok":          self.price_chase_ok,
            "hard_block":              self.hard_block,
            "reasons":                 self.reasons,
            "warnings":                self.warnings,
            "checked_at":              self.checked_at,
        }


# ── Guard ─────────────────────────────────────────────────────────────────────

class ExecutionQualityGuard:
    """
    Checks final execution quality before order execution.

    Usage:
        guard = ExecutionQualityGuard(settings)
        result = guard.check(
            ticker="ABCD",
            planned_entry=3.00,
            current_price=3.02,
            bid=3.01,
            ask=3.03,
            quote_timestamp="2026-06-05T14:30:00+00:00",
            open_positions=["WXYZ"],
            account={"buying_power": "10000"},
            paper_trading=True,
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._entry    = settings.get("entry_rules", {})
        self._exec     = settings.get("execution", {})
        self._mode     = settings.get("mode", {})

        self._max_spread = float(
            self._exec.get(
                "max_spread_percent_at_execution",
                self._entry.get("max_spread_percent_at_execution", 3.0),
            )
        )
        self._max_quote_age = float(
            self._exec.get(
                "max_quote_age_seconds",
                self._entry.get("max_quote_age_seconds", 20),
            )
        )
        self._max_chase_pct = float(
            self._entry.get("max_entry_extension_percent", 8.0)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        ticker:            str,
        planned_entry:     float,
        current_price:     float,
        bid:               float = 0.0,
        ask:               float = 0.0,
        quote_timestamp:   Optional[str | datetime] = None,
        open_positions:    Optional[list[str]] = None,
        account:           Optional[dict] = None,
        paper_trading:     Optional[bool] = None,
        required_buying_power: float = 0.0,
    ) -> ExecutionQualityResult:
        """
        Run all execution quality checks.

        Args:
            ticker:                Ticker symbol.
            planned_entry:         Entry price approved by trade_quality_gate.py.
            current_price:         Latest price right before execution.
            bid:                   Latest bid.
            ask:                   Latest ask.
            quote_timestamp:       Quote timestamp, ISO string or datetime.
            open_positions:        Current open position tickers.
            account:               Alpaca account dict.
            paper_trading:         True when Alpaca client/account is paper.
            required_buying_power: Estimated cash needed for order.

        Returns:
            ExecutionQualityResult.
        """
        result = ExecutionQualityResult(
            ticker        = ticker,
            planned_entry = float(planned_entry or 0.0),
            current_price = float(current_price or 0.0),
            bid           = float(bid or 0.0),
            ask           = float(ask or 0.0),
        )

        # ── Basic price validation ────────────────────────────────────────────
        if result.planned_entry <= 0:
            return self._block(result, "planned entry price is invalid")
        if result.current_price <= 0:
            return self._block(result, "current price is invalid")

        # ── Spread check ──────────────────────────────────────────────────────
        result.spread_pct = _spread_percent(result.bid, result.ask)
        result.spread_ok = self._check_spread(result)

        # ── Quote freshness check ─────────────────────────────────────────────
        result.quote_age_seconds = _quote_age_seconds(quote_timestamp)
        result.quote_fresh = self._check_quote_fresh(result, quote_timestamp)

        # ── Chasing check ─────────────────────────────────────────────────────
        result.price_chase_ok = self._check_price_chase(result)

        # ── Existing position check ───────────────────────────────────────────
        result.position_already_open = self._position_exists(ticker, open_positions)
        if result.position_already_open:
            result.warnings.append("position already open")
            result.hard_block = True

        # ── Paper trading safety check ────────────────────────────────────────
        result.paper_trading_confirmed = self._paper_trading_confirmed(paper_trading)
        if not result.paper_trading_confirmed:
            result.warnings.append("paper trading is not confirmed")
            result.hard_block = True

        # ── Buying power check ────────────────────────────────────────────────
        result.buying_power_available = self._check_buying_power(
            result, account, required_buying_power
        )

        # ── Score ─────────────────────────────────────────────────────────────
        result.execution_quality_score = self._score(result)

        if result.hard_block:
            result.execution_quality_score = min(result.execution_quality_score, 40.0)

        log.debug(
            "[execution_quality] %s score=%.1f spread=%.2f quote_age=%.1f block=%s",
            ticker, result.execution_quality_score, result.spread_pct,
            result.quote_age_seconds, result.hard_block,
        )
        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_spread(self, result: ExecutionQualityResult) -> bool:
        """
        Confirm spread is within execution limits.
        """
        if result.bid <= 0 or result.ask <= 0:
            result.warnings.append("bid/ask unavailable — cannot validate spread")
            result.hard_block = True
            return False

        if result.spread_pct > self._max_spread:
            result.warnings.append(
                f"spread {result.spread_pct:.2f}% exceeds max {self._max_spread:.2f}%"
            )
            result.hard_block = True
            return False

        result.reasons.append(f"Spread acceptable: {result.spread_pct:.2f}%")
        return True

    def _check_quote_fresh(
        self,
        result: ExecutionQualityResult,
        quote_timestamp: Optional[str | datetime],
    ) -> bool:
        """
        Confirm quote is fresh enough to execute.
        """
        if quote_timestamp is None:
            result.warnings.append("quote timestamp unavailable")
            result.hard_block = True
            return False

        if result.quote_age_seconds < 0:
            result.warnings.append("quote timestamp is in the future or invalid")
            result.hard_block = True
            return False

        if result.quote_age_seconds > self._max_quote_age:
            result.warnings.append(
                f"quote is stale: {result.quote_age_seconds:.1f}s old "
                f"(max {self._max_quote_age:.1f}s)"
            )
            result.hard_block = True
            return False

        result.reasons.append(f"Quote fresh: {result.quote_age_seconds:.1f}s old")
        return True

    def _check_price_chase(self, result: ExecutionQualityResult) -> bool:
        """
        Confirm current price has not moved too far from planned entry.
        """
        move_pct = (result.current_price - result.planned_entry) / result.planned_entry * 100

        if move_pct > self._max_chase_pct:
            result.warnings.append(
                f"price moved {move_pct:.2f}% above planned entry — chasing risk"
            )
            result.hard_block = True
            return False

        if move_pct > self._max_chase_pct * 0.5:
            result.warnings.append(
                f"price moved {move_pct:.2f}% above planned entry"
            )
        else:
            result.reasons.append("Current price is close to planned entry")

        return True

    def _position_exists(
        self,
        ticker: str,
        open_positions: Optional[list[str]],
    ) -> bool:
        """True when ticker is already in open positions."""
        positions = [t.upper() for t in (open_positions or [])]
        return ticker.upper() in positions

    def _paper_trading_confirmed(self, paper_trading: Optional[bool]) -> bool:
        """
        Confirm paper trading mode.
        Falls back to bot_settings mode safety flags when explicit value missing.
        """
        if paper_trading is not None:
            return bool(paper_trading)

        paper_only = bool(self._mode.get("paper_trading_only", True))
        allow_live = bool(self._mode.get("allow_live_money", False))
        return paper_only and not allow_live

    def _check_buying_power(
        self,
        result: ExecutionQualityResult,
        account: Optional[dict],
        required_buying_power: float,
    ) -> bool:
        """
        Confirm account has enough buying power for planned order.
        """
        if required_buying_power <= 0:
            result.reasons.append("Buying power check skipped — no required amount provided")
            return True

        if not account:
            result.warnings.append("account unavailable — cannot confirm buying power")
            result.hard_block = True
            return False

        buying_power = _safe_float(account.get("buying_power", 0.0))
        if buying_power < required_buying_power:
            result.warnings.append(
                f"buying power ${buying_power:,.2f} below required "
                f"${required_buying_power:,.2f}"
            )
            result.hard_block = True
            return False

        result.reasons.append(
            f"Buying power available: ${buying_power:,.2f}"
        )
        return True

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, result: ExecutionQualityResult) -> float:
        """
        Score execution quality 0–100.

        Weights:
          Spread quality:       30 pts
          Quote freshness:      25 pts
          Price chase quality:  20 pts
          Paper safety:         15 pts
          Buying power:         10 pts
        """
        score = 0.0

        # Spread quality (30 pts)
        if result.spread_ok:
            if result.spread_pct <= 0.5:
                score += 30
            elif result.spread_pct <= 1.0:
                score += 25
            elif result.spread_pct <= 2.0:
                score += 18
            else:
                score += 10

        # Quote freshness (25 pts)
        if result.quote_fresh:
            if result.quote_age_seconds <= 5:
                score += 25
            elif result.quote_age_seconds <= 10:
                score += 20
            else:
                score += 14

        # Price chase quality (20 pts)
        if result.price_chase_ok:
            move_pct = abs(
                (result.current_price - result.planned_entry) / result.planned_entry * 100
            )
            if move_pct <= 1.0:
                score += 20
            elif move_pct <= 3.0:
                score += 15
            elif move_pct <= self._max_chase_pct:
                score += 8

        # Paper safety (15 pts)
        if result.paper_trading_confirmed:
            score += 15

        # Buying power (10 pts)
        if result.buying_power_available:
            score += 10

        return max(0.0, min(score, 100.0))

    # ── Rejection helper ──────────────────────────────────────────────────────

    @staticmethod
    def _block(result: ExecutionQualityResult, reason: str) -> ExecutionQualityResult:
        result.hard_block = True
        result.warnings.append(reason)
        result.execution_quality_score = 0.0
        return result


# ── Standalone helpers ────────────────────────────────────────────────────────

def _spread_percent(bid: float, ask: float) -> float:
    """
    Bid-ask spread as a percentage of ask.
    """
    if bid <= 0 or ask <= 0:
        return 0.0
    return (ask - bid) / ask * 100


def _quote_age_seconds(timestamp: Optional[str | datetime]) -> float:
    """
    Return quote age in seconds.
    Returns infinity when timestamp is missing or invalid.
    """
    if timestamp is None:
        return float("inf")

    try:
        if isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        else:
            ts = timestamp

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
    except Exception:
        return float("inf")


def _safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
