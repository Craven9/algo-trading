"""
src/risk/position_sizer.py — Position sizing engine
Calculates the correct number of shares for a planned paper trade.

The bot should never guess position size.  Position size must be based on:
  - Entry price
  - Stop loss price
  - Maximum dollars allowed to risk
  - Maximum position value allowed
  - Account buying power
  - Final trade quality / confidence score
  - Risk reduction after losses or weak confidence

Responsibilities:
  - Calculate risk per share
  - Calculate max shares based on risk dollars
  - Calculate max shares based on max position value
  - Calculate max shares based on buying power
  - Apply confidence-based size reduction
  - Return PositionSize dataclass from models.py
  - Explain every sizing decision with reasons

Design rules:
  - This file does not approve trades
  - This file does not place orders
  - This file only calculates size
  - trade_quality_gate.py still makes the final buy/no-buy decision
  - account_risk_guard.py still checks account-level limits
  - Share count is always an integer
  - If inputs are invalid, shares must be 0
"""

from __future__ import annotations

import logging
from typing import Optional

from confidence_labeler import apply_size_reduction
from models import PositionSize

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_RISK_DOLLARS = 25.0
_DEFAULT_MAX_POSITION_VALUE = 250.0
_DEFAULT_MIN_SHARES = 1


# ── Sizer ─────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Calculates paper-trade position size.

    Usage:
        sizer = PositionSizer(settings)
        size = sizer.calculate(
            ticker="ABCD",
            entry_price=3.00,
            stop_price=2.85,
            account={"buying_power": "10000"},
            final_trade_quality_score=88,
        )
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._risk     = settings.get("risk", {})
        self._entry    = settings.get("entry_rules", {})

        self._max_risk_dollars = float(
            self._risk.get("max_risk_per_trade_dollars", _DEFAULT_MAX_RISK_DOLLARS)
        )
        self._max_position_value = float(
            self._risk.get("max_position_size_dollars", _DEFAULT_MAX_POSITION_VALUE)
        )
        self._min_shares = int(
            self._risk.get("min_shares", _DEFAULT_MIN_SHARES)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(
        self,
        ticker: str,
        entry_price: float,
        stop_price: float,
        account: Optional[dict] = None,
        final_trade_quality_score: float = 0.0,
        max_risk_dollars: Optional[float] = None,
        max_position_value: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size for a planned trade.

        Args:
            ticker:                    Ticker symbol.
            entry_price:               Planned entry price.
            stop_price:                Planned stop loss.
            account:                   Alpaca account dict, optional.
            final_trade_quality_score: Final score from trade_quality_gate.py.
            max_risk_dollars:          Optional override for max risk dollars.
            max_position_value:        Optional override for max position value.

        Returns:
            PositionSize.
        """
        entry = float(entry_price or 0.0)
        stop  = float(stop_price or 0.0)

        result = PositionSize(
            shares           = 0,
            entry_price      = entry,
            stop_price       = stop,
            risk_per_share   = 0.0,
            max_risk_dollars = float(max_risk_dollars or self._max_risk_dollars),
            position_value   = 0.0,
            size_reduction_pct = 0.0,
            reasons          = [],
        )

        # ── Validate prices ───────────────────────────────────────────────────
        if entry <= 0:
            result.reasons.append("Invalid entry price — shares set to 0")
            return result

        if stop <= 0:
            result.reasons.append("Invalid stop price — shares set to 0")
            return result

        if stop >= entry:
            result.reasons.append("Stop price must be below entry for long trades")
            return result

        risk_per_share = entry - stop
        result.risk_per_share = round(risk_per_share, 4)

        if risk_per_share <= 0:
            result.reasons.append("Risk per share is invalid — shares set to 0")
            return result

        # ── Determine limits ──────────────────────────────────────────────────
        risk_dollars = float(max_risk_dollars or self._max_risk_dollars)
        position_limit = float(max_position_value or self._max_position_value)
        buying_power = _buying_power(account)

        if risk_dollars <= 0:
            result.reasons.append("Max risk dollars is invalid — shares set to 0")
            return result

        if position_limit <= 0:
            result.reasons.append("Max position value is invalid — shares set to 0")
            return result

        # ── Shares from risk dollars ──────────────────────────────────────────
        shares_by_risk = int(risk_dollars // risk_per_share)

        # ── Shares from max position value ────────────────────────────────────
        shares_by_position_value = int(position_limit // entry)

        # ── Shares from buying power ──────────────────────────────────────────
        if buying_power > 0:
            shares_by_buying_power = int(buying_power // entry)
        else:
            shares_by_buying_power = shares_by_position_value

        raw_shares = min(
            shares_by_risk,
            shares_by_position_value,
            shares_by_buying_power,
        )

        if raw_shares < self._min_shares:
            result.reasons.append(
                f"Calculated shares {raw_shares} below minimum {self._min_shares}"
            )
            return result

        result.reasons.append(
            f"Risk-based shares: {shares_by_risk} "
            f"(${risk_dollars:.2f} max risk / ${risk_per_share:.4f} risk per share)"
        )
        result.reasons.append(
            f"Position-value shares: {shares_by_position_value} "
            f"(${position_limit:.2f} max position / ${entry:.4f} entry)"
        )

        if buying_power > 0:
            result.reasons.append(
                f"Buying-power shares: {shares_by_buying_power} "
                f"(${buying_power:.2f} buying power)"
            )

        # ── Apply confidence-based size reduction ─────────────────────────────
        adjusted_shares, size_pct, size_reason = apply_size_reduction(
            base_shares = raw_shares,
            score       = final_trade_quality_score,
            settings    = self._settings,
        )

        result.size_reduction_pct = 100.0 - size_pct
        result.reasons.append(size_reason)

        if adjusted_shares <= 0:
            result.reasons.append("Confidence sizing reduced shares to 0")
            return result

        result.shares = adjusted_shares
        result.position_value = round(result.shares * entry, 2)

        # ── Final sanity check ────────────────────────────────────────────────
        actual_risk = result.shares * risk_per_share
        if actual_risk > risk_dollars + 0.01:
            result.shares = max(int(risk_dollars // risk_per_share), 0)
            result.position_value = round(result.shares * entry, 2)
            result.reasons.append("Share count reduced to respect max risk dollars")

        if result.position_value > position_limit + 0.01:
            result.shares = max(int(position_limit // entry), 0)
            result.position_value = round(result.shares * entry, 2)
            result.reasons.append("Share count reduced to respect max position value")

        if buying_power > 0 and result.position_value > buying_power + 0.01:
            result.shares = max(int(buying_power // entry), 0)
            result.position_value = round(result.shares * entry, 2)
            result.reasons.append("Share count reduced to respect buying power")

        if result.shares <= 0:
            result.position_value = 0.0
            result.reasons.append("Final share count is 0 after safety adjustments")
            return result

        result.reasons.append(
            f"Final size: {result.shares} share(s), "
            f"${result.position_value:.2f} position value"
        )

        log.debug(
            "[position_sizer] %s shares=%d entry=%.4f stop=%.4f value=%.2f",
            ticker, result.shares, result.entry_price,
            result.stop_price, result.position_value,
        )
        return result

    def estimate_position_value(self, shares: int, entry_price: float) -> float:
        """Return position value for a given share count and entry."""
        if shares <= 0 or entry_price <= 0:
            return 0.0
        return round(shares * entry_price, 2)

    def estimate_risk_dollars(
        self,
        shares: int,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Return dollar risk for a given share count."""
        if shares <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0.0
        if stop_price >= entry_price:
            return 0.0
        return round((entry_price - stop_price) * shares, 2)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def calculate_position_size(
    settings: dict,
    ticker: str,
    entry_price: float,
    stop_price: float,
    account: Optional[dict] = None,
    final_trade_quality_score: float = 0.0,
    max_risk_dollars: Optional[float] = None,
    max_position_value: Optional[float] = None,
) -> PositionSize:
    """
    Convenience function for bot_runner.py.
    """
    sizer = PositionSizer(settings)
    return sizer.calculate(
        ticker                    = ticker,
        entry_price               = entry_price,
        stop_price                = stop_price,
        account                   = account,
        final_trade_quality_score = final_trade_quality_score,
        max_risk_dollars          = max_risk_dollars,
        max_position_value        = max_position_value,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _buying_power(account: Optional[dict]) -> float:
    """Extract buying power from an Alpaca account dict."""
    if not account:
        return 0.0
    try:
        return float(account.get("buying_power", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
