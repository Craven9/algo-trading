"""
src/risk/position_tracker.py — Alpaca position tracking helper
Tracks current open positions in the Alpaca paper account.

The bot should never buy a ticker blindly.  Before any new entry, it must
know whether the account already has a position in that ticker and what
the current exposure looks like.

Responsibilities:
  - Fetch all open positions from Alpaca paper account
  - Fetch a single ticker position
  - Normalize Alpaca position fields into a clean PositionSnapshot
  - Check whether a ticker already has an open position
  - Calculate total open exposure
  - Provide position summaries for account_risk_guard.py and frontend
  - Fail safely when Alpaca data is unavailable

Design rules:
  - This file does not place orders
  - This file does not approve trades
  - This file only reads and normalizes position data
  - Duplicate-position checks should happen before order_executor.py
  - Missing broker data should be treated as unsafe by caller
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ── Position snapshot ─────────────────────────────────────────────────────────

@dataclass
class PositionSnapshot:
    """
    Normalized open position snapshot.
    Built from Alpaca position dicts.
    """
    ticker:              str
    qty:                 float = 0.0
    side:                str   = "long"
    avg_entry_price:     float = 0.0
    current_price:       float = 0.0
    market_value:        float = 0.0
    cost_basis:          float = 0.0
    unrealized_pl:       float = 0.0
    unrealized_plpc:     float = 0.0
    unrealized_intraday_pl:   float = 0.0
    unrealized_intraday_plpc: float = 0.0
    exchange:            str   = ""
    asset_class:         str   = ""
    raw:                 dict  = field(default_factory=dict)
    fetched_at:          str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_open(self) -> bool:
        return abs(self.qty) > 0

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def position_value(self) -> float:
        if self.market_value:
            return abs(self.market_value)
        return abs(self.qty * self.current_price)

    def to_dict(self) -> dict:
        return {
            "ticker":                    self.ticker,
            "qty":                       self.qty,
            "side":                      self.side,
            "avg_entry_price":           round(self.avg_entry_price, 4),
            "current_price":             round(self.current_price, 4),
            "market_value":              round(self.market_value, 2),
            "cost_basis":                round(self.cost_basis, 2),
            "unrealized_pl":             round(self.unrealized_pl, 2),
            "unrealized_plpc":           round(self.unrealized_plpc, 4),
            "unrealized_intraday_pl":    round(self.unrealized_intraday_pl, 2),
            "unrealized_intraday_plpc":  round(self.unrealized_intraday_plpc, 4),
            "exchange":                  self.exchange,
            "asset_class":               self.asset_class,
            "is_open":                   self.is_open,
            "position_value":            round(self.position_value, 2),
            "fetched_at":                self.fetched_at,
        }


# ── Position tracker result ──────────────────────────────────────────────────

@dataclass
class PositionTrackerResult:
    """
    Aggregated position tracking result.
    """
    positions:              list[PositionSnapshot] = field(default_factory=list)
    open_position_count:    int   = 0
    total_market_value:     float = 0.0
    total_unrealized_pl:    float = 0.0
    tickers:                list[str] = field(default_factory=list)
    has_error:              bool  = False
    error:                  str   = ""
    fetched_at:             str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def has_position(self, ticker: str) -> bool:
        ticker = ticker.upper()
        return ticker in self.tickers

    def get_position(self, ticker: str) -> Optional[PositionSnapshot]:
        ticker = ticker.upper()
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None

    def to_dict(self) -> dict:
        return {
            "open_position_count": self.open_position_count,
            "total_market_value":  round(self.total_market_value, 2),
            "total_unrealized_pl": round(self.total_unrealized_pl, 2),
            "tickers":             self.tickers,
            "has_error":           self.has_error,
            "error":               self.error,
            "positions":           [p.to_dict() for p in self.positions],
            "fetched_at":          self.fetched_at,
        }


# ── Tracker ──────────────────────────────────────────────────────────────────

class PositionTracker:
    """
    Reads and normalizes Alpaca paper positions.

    Usage:
        tracker = PositionTracker(alpaca_client)
        result = tracker.get_all_positions()

        if tracker.has_open_position("ABCD"):
            # block new buy
    """

    def __init__(self, alpaca_client):
        self._client = alpaca_client
        self._last_result: Optional[PositionTrackerResult] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_all_positions(self, force_refresh: bool = True) -> PositionTrackerResult:
        """
        Fetch and normalize all open positions.

        Args:
            force_refresh: If False and cached result exists, return cached result.

        Returns:
            PositionTrackerResult.
        """
        if not force_refresh and self._last_result is not None:
            return self._last_result

        result = PositionTrackerResult()

        try:
            raw_positions = self._client.get_positions()
        except Exception as exc:
            msg = f"failed to fetch positions: {exc}"
            log.error("[position_tracker] %s", msg)
            result.has_error = True
            result.error = msg
            self._last_result = result
            return result

        if raw_positions is None:
            raw_positions = []

        positions = [normalize_position(p) for p in raw_positions]
        positions = [p for p in positions if p.is_open]

        result.positions = positions
        result.open_position_count = len(positions)
        result.total_market_value = round(sum(p.position_value for p in positions), 2)
        result.total_unrealized_pl = round(sum(p.unrealized_pl for p in positions), 2)
        result.tickers = [p.ticker for p in positions]

        self._last_result = result

        log.debug(
            "[position_tracker] open_positions=%d exposure=%.2f unrealized=%.2f",
            result.open_position_count,
            result.total_market_value,
            result.total_unrealized_pl,
        )
        return result

    def get_position_for_ticker(
        self,
        ticker: str,
        force_refresh: bool = True,
    ) -> Optional[PositionSnapshot]:
        """
        Return a normalized PositionSnapshot for one ticker, or None.
        """
        ticker = ticker.upper()

        # Prefer direct single-position endpoint when refreshing.
        if force_refresh:
            try:
                raw = self._client.get_position(ticker)
                if raw:
                    pos = normalize_position(raw)
                    if pos.is_open:
                        return pos
            except Exception as exc:
                log.debug("[position_tracker] direct position fetch failed for %s: %s", ticker, exc)

        result = self.get_all_positions(force_refresh=force_refresh)
        return result.get_position(ticker)

    def has_open_position(
        self,
        ticker: str,
        force_refresh: bool = True,
    ) -> tuple[bool, str]:
        """
        Check whether a ticker already has an open position.

        Returns:
            (True, reason) if position exists.
            (False, "no open position") if not.
        """
        pos = self.get_position_for_ticker(ticker, force_refresh=force_refresh)

        if pos and pos.is_open:
            return True, (
                f"open position already exists for {ticker.upper()} "
                f"qty={pos.qty:g} avg_entry={pos.avg_entry_price:.4f}"
            )

        return False, "no open position"

    def open_tickers(self, force_refresh: bool = True) -> list[str]:
        """
        Return all tickers currently open.
        """
        result = self.get_all_positions(force_refresh=force_refresh)
        return result.tickers

    def total_exposure(self, force_refresh: bool = True) -> float:
        """
        Return total absolute market value of all open positions.
        """
        result = self.get_all_positions(force_refresh=force_refresh)
        return result.total_market_value

    def last_result(self) -> Optional[PositionTrackerResult]:
        """Return most recent tracker result without refetching."""
        return self._last_result


# ── Convenience wrappers ──────────────────────────────────────────────────────

def get_position_for_ticker(alpaca_client, ticker: str) -> Optional[PositionSnapshot]:
    """
    Convenience function used by bot_runner.py or tests.
    """
    tracker = PositionTracker(alpaca_client)
    return tracker.get_position_for_ticker(ticker)


def check_open_position(alpaca_client, ticker: str) -> tuple[bool, str]:
    """
    Convenience function to block duplicate buys.

    Returns:
        (True, reason) if ticker already has an open position.
        (False, "no open position") otherwise.
    """
    tracker = PositionTracker(alpaca_client)
    return tracker.has_open_position(ticker)


def get_open_position_tickers(alpaca_client) -> list[str]:
    """
    Convenience function to return open ticker symbols.
    """
    tracker = PositionTracker(alpaca_client)
    return tracker.open_tickers()


# ── Normalization helpers ─────────────────────────────────────────────────────

def normalize_position(raw: dict) -> PositionSnapshot:
    """
    Normalize an Alpaca position dict into PositionSnapshot.
    """
    symbol = (
        raw.get("symbol")
        or raw.get("ticker")
        or raw.get("asset_symbol")
        or ""
    ).upper()

    qty = _safe_float(raw.get("qty", 0.0))
    side = str(raw.get("side", "long") or "long").lower()

    avg_entry = _safe_float(raw.get("avg_entry_price", 0.0))
    current = _safe_float(raw.get("current_price", 0.0))
    market_value = _safe_float(raw.get("market_value", 0.0))
    cost_basis = _safe_float(raw.get("cost_basis", 0.0))
    unrealized_pl = _safe_float(raw.get("unrealized_pl", 0.0))
    unrealized_plpc = _safe_float(raw.get("unrealized_plpc", 0.0))
    unrealized_intraday_pl = _safe_float(raw.get("unrealized_intraday_pl", 0.0))
    unrealized_intraday_plpc = _safe_float(raw.get("unrealized_intraday_plpc", 0.0))

    # Alpaca may return negative market value for short positions.
    if market_value == 0.0 and current > 0 and qty != 0:
        market_value = qty * current

    return PositionSnapshot(
        ticker                    = symbol,
        qty                       = qty,
        side                      = side,
        avg_entry_price           = avg_entry,
        current_price             = current,
        market_value              = market_value,
        cost_basis                = cost_basis,
        unrealized_pl             = unrealized_pl,
        unrealized_plpc           = unrealized_plpc,
        unrealized_intraday_pl    = unrealized_intraday_pl,
        unrealized_intraday_plpc  = unrealized_intraday_plpc,
        exchange                  = str(raw.get("exchange", "")),
        asset_class               = str(raw.get("asset_class", "")),
        raw                       = dict(raw),
    )


def positions_to_alpaca_like_dicts(
    result: PositionTrackerResult,
) -> list[dict]:
    """
    Convert normalized positions back into simple dicts for account_risk_guard.py.
    """
    return [
        {
            "symbol": p.ticker,
            "qty": p.qty,
            "side": p.side,
            "avg_entry_price": p.avg_entry_price,
            "current_price": p.current_price,
            "market_value": p.market_value,
            "unrealized_pl": p.unrealized_pl,
        }
        for p in result.positions
    ]


def _safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
