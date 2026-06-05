"""
src/risk/account_risk_guard.py — Account-level risk protection
Checks whether the account is allowed to take a new trade before any
paper buy is submitted.

A trade can have a strong setup and good probability, but the account
should still block the buy if exposure is too high, daily loss is hit,
too many positions are open, buying power is too low, or the requested
position is too large.

Responsibilities:
  - Block new buys if max open positions is reached
  - Block new buys if daily loss limit is reached
  - Block new buys if requested position size is too large
  - Block new buys if total exposure would exceed configured limit
  - Block new buys if buying power is insufficient
  - Block new buys if account drawdown is too high
  - Confirm paper trading safety
  - Return a clean AccountRiskResult for trade_quality_gate.py

Design rules:
  - This file does not place orders
  - This file does not approve trades by itself
  - This file only protects the account from bad risk conditions
  - Any hard block must stop order_executor.py from running
  - Safety checks always override score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_OPEN_POSITIONS = 5
_DEFAULT_MAX_DAILY_LOSS_DOLLARS = 100.0
_DEFAULT_MAX_POSITION_SIZE_DOLLARS = 250.0
_DEFAULT_MAX_TOTAL_EXPOSURE_PERCENT = 30.0
_DEFAULT_MAX_ACCOUNT_DRAWDOWN_PERCENT = 10.0
_DEFAULT_MIN_BUYING_POWER_AFTER_TRADE = 25.0


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AccountRiskResult:
    """
    Account-level risk result.
    Consumed by trade_quality_gate.py before order execution.
    """
    approved:                    bool  = True
    hard_block:                  bool  = False
    risk_score:                  float = 100.0

    # Account values
    buying_power:                float = 0.0
    cash:                        float = 0.0
    portfolio_value:             float = 0.0
    equity:                      float = 0.0
    last_equity:                 float = 0.0

    # Position / exposure
    open_position_count:         int   = 0
    max_open_positions:          int   = _DEFAULT_MAX_OPEN_POSITIONS
    current_exposure_dollars:    float = 0.0
    new_position_value:          float = 0.0
    projected_exposure_dollars:  float = 0.0
    projected_exposure_percent:  float = 0.0

    # Daily / drawdown risk
    daily_pnl_dollars:           float = 0.0
    daily_loss_limit:            float = _DEFAULT_MAX_DAILY_LOSS_DOLLARS
    account_drawdown_percent:    float = 0.0

    # Safety
    paper_trading_confirmed:     bool  = False

    reasons:                     list[str] = field(default_factory=list)
    warnings:                    list[str] = field(default_factory=list)

    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "approved":                   self.approved,
            "hard_block":                 self.hard_block,
            "risk_score":                 round(self.risk_score, 2),
            "buying_power":               round(self.buying_power, 2),
            "cash":                       round(self.cash, 2),
            "portfolio_value":            round(self.portfolio_value, 2),
            "equity":                     round(self.equity, 2),
            "last_equity":                round(self.last_equity, 2),
            "open_position_count":        self.open_position_count,
            "max_open_positions":         self.max_open_positions,
            "current_exposure_dollars":   round(self.current_exposure_dollars, 2),
            "new_position_value":         round(self.new_position_value, 2),
            "projected_exposure_dollars": round(self.projected_exposure_dollars, 2),
            "projected_exposure_percent": round(self.projected_exposure_percent, 2),
            "daily_pnl_dollars":          round(self.daily_pnl_dollars, 2),
            "daily_loss_limit":           round(self.daily_loss_limit, 2),
            "account_drawdown_percent":   round(self.account_drawdown_percent, 2),
            "paper_trading_confirmed":    self.paper_trading_confirmed,
            "reasons":                    self.reasons,
            "warnings":                   self.warnings,
            "checked_at":                 self.checked_at,
        }


# ── Guard ─────────────────────────────────────────────────────────────────────

class AccountRiskGuard:
    """
    Protects the account from unsafe new entries.

    Usage:
        guard = AccountRiskGuard(settings)
        result = guard.check(
            account=alpaca_account,
            positions=alpaca_positions,
            new_position_value=225.00,
            daily_pnl_dollars=-20.00,
            paper_trading=True,
        )

        if result.hard_block:
            # do not submit order
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._risk     = settings.get("risk", {})
        self._mode     = settings.get("mode", {})

        self._max_open_positions = int(
            self._risk.get("max_open_positions", _DEFAULT_MAX_OPEN_POSITIONS)
        )
        self._max_daily_loss = float(
            self._risk.get("max_daily_loss_dollars", _DEFAULT_MAX_DAILY_LOSS_DOLLARS)
        )
        self._max_position_value = float(
            self._risk.get("max_position_size_dollars", _DEFAULT_MAX_POSITION_SIZE_DOLLARS)
        )
        self._max_exposure_pct = float(
            self._risk.get("max_total_exposure_percent", _DEFAULT_MAX_TOTAL_EXPOSURE_PERCENT)
        )
        self._max_drawdown_pct = float(
            self._risk.get("max_account_drawdown_percent", _DEFAULT_MAX_ACCOUNT_DRAWDOWN_PERCENT)
        )
        self._min_bp_after_trade = float(
            self._risk.get("min_buying_power_after_trade", _DEFAULT_MIN_BUYING_POWER_AFTER_TRADE)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        account: Optional[dict],
        positions: Optional[list[dict]] = None,
        new_position_value: float = 0.0,
        daily_pnl_dollars: float = 0.0,
        paper_trading: Optional[bool] = None,
    ) -> AccountRiskResult:
        """
        Check account-level risk before allowing a new buy.

        Args:
            account:            Alpaca account dict.
            positions:          Current Alpaca positions list.
            new_position_value: Planned dollar value of new position.
            daily_pnl_dollars:  Realized/unrealized daily P/L if available.
            paper_trading:      Explicit paper trading confirmation.

        Returns:
            AccountRiskResult.
        """
        result = AccountRiskResult(
            max_open_positions = self._max_open_positions,
            daily_loss_limit   = self._max_daily_loss,
            daily_pnl_dollars  = float(daily_pnl_dollars or 0.0),
            new_position_value = float(new_position_value or 0.0),
        )

        if not account:
            return self._block(result, "account data unavailable")

        positions = positions or []

        self._load_account_values(result, account)
        self._load_position_values(result, positions)

        result.projected_exposure_dollars = (
            result.current_exposure_dollars + result.new_position_value
        )
        if result.portfolio_value > 0:
            result.projected_exposure_percent = (
                result.projected_exposure_dollars / result.portfolio_value * 100
            )

        result.account_drawdown_percent = self._account_drawdown_percent(result)
        result.paper_trading_confirmed = self._paper_trading_confirmed(
            paper_trading, account
        )

        # ── Safety / hard checks ──────────────────────────────────────────────
        self._check_paper_safety(result)
        self._check_open_positions(result)
        self._check_daily_loss(result)
        self._check_position_size(result)
        self._check_total_exposure(result)
        self._check_buying_power(result)
        self._check_drawdown(result)

        result.risk_score = self._score(result)

        if result.hard_block:
            result.approved = False
            result.risk_score = min(result.risk_score, 40.0)
        else:
            result.approved = True
            result.reasons.append("Account risk guard passed")

        log.debug(
            "[account_risk] approved=%s score=%.1f open=%d exposure=%.1f%% daily_pnl=%.2f",
            result.approved,
            result.risk_score,
            result.open_position_count,
            result.projected_exposure_percent,
            result.daily_pnl_dollars,
        )
        return result

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_account_values(self, result: AccountRiskResult, account: dict) -> None:
        """Load numeric account values from Alpaca account dict."""
        result.buying_power = _safe_float(account.get("buying_power", 0.0))
        result.cash = _safe_float(account.get("cash", 0.0))
        result.portfolio_value = _safe_float(account.get("portfolio_value", 0.0))
        result.equity = _safe_float(account.get("equity", result.portfolio_value))
        result.last_equity = _safe_float(account.get("last_equity", result.equity))

        if result.portfolio_value <= 0 and result.equity > 0:
            result.portfolio_value = result.equity

    def _load_position_values(
        self,
        result: AccountRiskResult,
        positions: list[dict],
    ) -> None:
        """Load exposure and position count from open positions."""
        result.open_position_count = len(positions)

        exposure = 0.0
        for pos in positions:
            market_value = _safe_float(pos.get("market_value", 0.0))
            if market_value == 0:
                qty = abs(_safe_float(pos.get("qty", 0.0)))
                price = _safe_float(pos.get("current_price", 0.0))
                market_value = qty * price
            exposure += abs(market_value)

        result.current_exposure_dollars = exposure

    # ── Hard checks ───────────────────────────────────────────────────────────

    def _check_paper_safety(self, result: AccountRiskResult) -> None:
        """Confirm paper trading safety."""
        if not result.paper_trading_confirmed:
            self._block(result, "paper trading is not confirmed")
            return
        result.reasons.append("Paper trading confirmed")

    def _check_open_positions(self, result: AccountRiskResult) -> None:
        """Block when max open positions would be exceeded."""
        if result.open_position_count >= self._max_open_positions:
            self._block(
                result,
                f"max open positions reached "
                f"({result.open_position_count}/{self._max_open_positions})",
            )
            return
        result.reasons.append(
            f"Open positions ok: {result.open_position_count}/{self._max_open_positions}"
        )

    def _check_daily_loss(self, result: AccountRiskResult) -> None:
        """Block when daily loss limit is hit."""
        if result.daily_pnl_dollars <= -abs(self._max_daily_loss):
            self._block(
                result,
                f"daily loss limit hit: ${result.daily_pnl_dollars:,.2f}",
            )
            return
        result.reasons.append(
            f"Daily P/L within limit: ${result.daily_pnl_dollars:,.2f}"
        )

    def _check_position_size(self, result: AccountRiskResult) -> None:
        """Block when planned position value exceeds max allowed size."""
        if result.new_position_value <= 0:
            self._block(result, "new position value is invalid")
            return

        if result.new_position_value > self._max_position_value:
            self._block(
                result,
                f"position value ${result.new_position_value:,.2f} exceeds max "
                f"${self._max_position_value:,.2f}",
            )
            return

        result.reasons.append(
            f"Position size ok: ${result.new_position_value:,.2f}"
        )

    def _check_total_exposure(self, result: AccountRiskResult) -> None:
        """Block when total projected exposure is too high."""
        if result.portfolio_value <= 0:
            result.warnings.append("portfolio value unavailable — exposure check limited")
            return

        if result.projected_exposure_percent > self._max_exposure_pct:
            self._block(
                result,
                f"projected exposure {result.projected_exposure_percent:.1f}% "
                f"exceeds max {self._max_exposure_pct:.1f}%",
            )
            return

        result.reasons.append(
            f"Projected exposure ok: {result.projected_exposure_percent:.1f}%"
        )

    def _check_buying_power(self, result: AccountRiskResult) -> None:
        """Block when buying power is insufficient."""
        remaining = result.buying_power - result.new_position_value

        if result.buying_power < result.new_position_value:
            self._block(
                result,
                f"buying power ${result.buying_power:,.2f} below required "
                f"${result.new_position_value:,.2f}",
            )
            return

        if remaining < self._min_bp_after_trade:
            self._block(
                result,
                f"buying power after trade would be ${remaining:,.2f}, below "
                f"minimum ${self._min_bp_after_trade:,.2f}",
            )
            return

        result.reasons.append(
            f"Buying power ok: ${result.buying_power:,.2f}"
        )

    def _check_drawdown(self, result: AccountRiskResult) -> None:
        """Block when account drawdown is too high."""
        if result.account_drawdown_percent >= self._max_drawdown_pct:
            self._block(
                result,
                f"account drawdown {result.account_drawdown_percent:.2f}% exceeds "
                f"max {self._max_drawdown_pct:.2f}%",
            )
            return

        if result.account_drawdown_percent > 0:
            result.reasons.append(
                f"Account drawdown within limit: {result.account_drawdown_percent:.2f}%"
            )

    # ── Calculations ──────────────────────────────────────────────────────────

    def _account_drawdown_percent(self, result: AccountRiskResult) -> float:
        """
        Estimate account drawdown from last_equity to equity.
        """
        if result.last_equity <= 0:
            return 0.0
        if result.equity >= result.last_equity:
            return 0.0
        return (result.last_equity - result.equity) / result.last_equity * 100

    def _paper_trading_confirmed(
        self,
        paper_trading: Optional[bool],
        account: dict,
    ) -> bool:
        """
        Confirm paper trading using explicit value, account flag, or settings.
        """
        if paper_trading is not None:
            return bool(paper_trading)

        if "paper_trading" in account:
            return bool(account.get("paper_trading"))

        paper_only = bool(self._mode.get("paper_trading_only", True))
        allow_live = bool(self._mode.get("allow_live_money", False))
        return paper_only and not allow_live

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, result: AccountRiskResult) -> float:
        """
        Score account risk quality 0–100.

        Weights:
          Open position room:     20 pts
          Daily loss room:        20 pts
          Position size quality:  20 pts
          Exposure room:          20 pts
          Buying power room:      10 pts
          Drawdown room:          10 pts
        """
        score = 0.0

        # Open position room (20)
        if self._max_open_positions > 0:
            room = max(self._max_open_positions - result.open_position_count, 0)
            score += min(room / self._max_open_positions * 20, 20)

        # Daily loss room (20)
        if self._max_daily_loss > 0:
            used_loss = abs(min(result.daily_pnl_dollars, 0.0))
            loss_room = max(self._max_daily_loss - used_loss, 0.0)
            score += min(loss_room / self._max_daily_loss * 20, 20)

        # Position size quality (20)
        if self._max_position_value > 0 and result.new_position_value > 0:
            size_ratio = result.new_position_value / self._max_position_value
            if size_ratio <= 0.50:
                score += 20
            elif size_ratio <= 0.75:
                score += 15
            elif size_ratio <= 1.0:
                score += 10

        # Exposure room (20)
        if self._max_exposure_pct > 0:
            exposure_room = max(
                self._max_exposure_pct - result.projected_exposure_percent,
                0.0,
            )
            score += min(exposure_room / self._max_exposure_pct * 20, 20)

        # Buying power room (10)
        if result.buying_power > 0 and result.new_position_value > 0:
            remaining = result.buying_power - result.new_position_value
            if remaining >= result.new_position_value:
                score += 10
            elif remaining > 0:
                score += 5

        # Drawdown room (10)
        if self._max_drawdown_pct > 0:
            dd_room = max(
                self._max_drawdown_pct - result.account_drawdown_percent,
                0.0,
            )
            score += min(dd_room / self._max_drawdown_pct * 10, 10)

        return max(0.0, min(score, 100.0))

    # ── Rejection helper ──────────────────────────────────────────────────────

    @staticmethod
    def _block(result: AccountRiskResult, reason: str) -> AccountRiskResult:
        result.approved = False
        result.hard_block = True
        result.warnings.append(reason)
        return result


# ── Convenience wrapper ───────────────────────────────────────────────────────

def check_account_risk(
    settings: dict,
    account: Optional[dict],
    positions: Optional[list[dict]] = None,
    new_position_value: float = 0.0,
    daily_pnl_dollars: float = 0.0,
    paper_trading: Optional[bool] = None,
) -> AccountRiskResult:
    """
    Convenience function for bot_runner.py.
    """
    guard = AccountRiskGuard(settings)
    return guard.check(
        account            = account,
        positions          = positions,
        new_position_value = new_position_value,
        daily_pnl_dollars  = daily_pnl_dollars,
        paper_trading      = paper_trading,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
