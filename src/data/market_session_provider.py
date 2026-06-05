"""
src/data/market_session_provider.py — Market session state provider
Combines time_utils session logic with live market context to give every
other module a single, consistent view of the current session.

Responsibilities:
  - Track session open/close times and the opening range window
  - Determine whether the bot is allowed to scan, analyze, or trade
  - Provide session-level context (minutes since open, phase label, etc.)
  - Cache the opening range bars so opening_range_analyzer can read them
  - Detect session resets (new day) and clear stale state

All heavy time logic lives in time_utils.  This module adds state and
context on top of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from time_utils import (
    ET,
    can_trade_now,
    is_market_open,
    is_pre_market,
    is_after_hours,
    is_extended_hours,
    is_high_risk_window,
    is_avoid_window,
    is_same_session,
    is_weekend,
    market_open_dt,
    market_close_dt,
    minutes_since_open,
    minutes_until_close,
    minutes_until_open,
    now_et,
    session_label,
    trading_day_start,
    trading_day_end,
)

log = logging.getLogger(__name__)


# ── Session Phase ─────────────────────────────────────────────────────────────

class SessionPhase:
    """
    Fine-grained session phase labels used by the scanner and analysis engines
    to weight their signals differently depending on where we are in the day.
    """
    PRE_MARKET          = "pre_market"
    OPENING             = "opening"          # first 30 minutes
    ACTIVE              = "active"           # 10:00–14:00
    MIDDAY              = "midday"           # 12:00–13:00 (low volume)
    AFTERNOON           = "afternoon"        # 13:00–15:00
    POWER_HOUR          = "power_hour"       # 15:00–15:55
    CLOSE_APPROACH      = "close_approach"   # 15:55–16:00
    AFTER_HOURS         = "after_hours"
    CLOSED              = "closed"


# Phase time boundaries (minutes since open)
_PHASE_BOUNDARIES = [
    (0,    30,  SessionPhase.OPENING),
    (30,   150, SessionPhase.ACTIVE),
    (150,  210, SessionPhase.MIDDAY),
    (210,  330, SessionPhase.AFTERNOON),
    (330,  385, SessionPhase.POWER_HOUR),
    (385,  390, SessionPhase.CLOSE_APPROACH),
]


def _phase_from_minutes(mins: float) -> str:
    """Map minutes-since-open to a SessionPhase label."""
    for lo, hi, label in _PHASE_BOUNDARIES:
        if lo <= mins < hi:
            return label
    return SessionPhase.ACTIVE   # fallback for any gap


# ── Session Context dataclass ─────────────────────────────────────────────────

@dataclass
class SessionContext:
    """
    Snapshot of the current session state.
    Produced by MarketSessionProvider.get_context() and consumed by the
    scanner, analysis, scoring, and risk modules.
    """
    # Basic session flags
    is_market_open:      bool   = False
    is_pre_market:       bool   = False
    is_after_hours:      bool   = False
    is_extended_hours:   bool   = False
    is_weekend:          bool   = False
    is_high_risk_window: bool   = False
    is_avoid_window:     bool   = False

    # Phase and labels
    session_label:       str    = SessionPhase.CLOSED
    session_phase:       str    = SessionPhase.CLOSED

    # Timing
    minutes_since_open:  float  = 0.0
    minutes_until_close: float  = 0.0
    minutes_until_open:  float  = 0.0

    # Trade gate
    can_trade:           bool   = False
    cannot_trade_reason: str    = ""

    # Opening range tracking
    opening_range_complete:     bool        = False
    opening_range_minutes:      int         = 15
    opening_range_bars:         list[dict]  = field(default_factory=list)

    # Session key datetimes (ISO strings for JSON serialization)
    market_open_time:    str    = ""
    market_close_time:   str    = ""
    session_date:        str    = ""
    snapshot_time:       str    = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("opening_range_bars", None)   # bars are large; omit from snapshots
        return d


# ── Provider ──────────────────────────────────────────────────────────────────

class MarketSessionProvider:
    """
    Stateful session manager.  One instance lives for the lifetime of the bot.

    Usage:
        provider = MarketSessionProvider(settings)
        ctx      = provider.get_context()

        if ctx.can_trade:
            # ... run analysis
    """

    def __init__(self, settings: dict):
        self._settings              = settings
        self._session_cfg           = settings.get("session", {})
        self._or_minutes: int       = settings.get("opening_range", {}).get(
                                          "preferred_range_minutes", 15)

        # State that persists across loop iterations
        self._session_date:  Optional[str]       = None
        self._or_bars:       list[dict]           = []
        self._or_complete:   bool                 = False
        self._last_ctx:      Optional[SessionContext] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(self, ref: Optional[datetime] = None) -> SessionContext:
        """
        Build and return a fresh SessionContext for the current moment.
        Also handles session-reset detection (new trading day).
        """
        now = ref or now_et()

        self._maybe_reset_session(now)

        mkt_open   = is_market_open(self._settings, ref=now)
        pre_mkt    = is_pre_market(self._settings,  ref=now)
        aft_hrs    = is_after_hours(self._settings,  ref=now)
        ext_hrs    = is_extended_hours(self._settings, ref=now)
        weekend    = is_weekend(ref=now)
        hi_risk    = is_high_risk_window(self._settings, ref=now)
        avoid_win  = is_avoid_window(self._settings,  ref=now)
        s_label    = session_label(self._settings,   ref=now)
        mins_open  = minutes_since_open(self._settings, ref=now)
        mins_close = minutes_until_close(self._settings, ref=now)
        mins_until = minutes_until_open(self._settings,  ref=now)

        # Session phase
        if weekend:
            phase = SessionPhase.CLOSED
        elif mkt_open:
            phase = _phase_from_minutes(mins_open)
        elif pre_mkt:
            phase = SessionPhase.PRE_MARKET
        elif aft_hrs:
            phase = SessionPhase.AFTER_HOURS
        else:
            phase = SessionPhase.CLOSED

        can_trade, ct_reason = can_trade_now(self._settings, ref=now)

        # Opening range completion
        or_complete = self._or_complete or (
            mkt_open and mins_open >= self._or_minutes
        )
        if or_complete and not self._or_complete:
            self._or_complete = True
            log.info(
                "[session] Opening range complete (%d min) at %.1f min since open",
                self._or_minutes, mins_open,
            )

        open_dt  = market_open_dt(self._settings, ref=now)
        close_dt = market_close_dt(self._settings, ref=now)

        ctx = SessionContext(
            is_market_open      = mkt_open,
            is_pre_market       = pre_mkt,
            is_after_hours      = aft_hrs,
            is_extended_hours   = ext_hrs,
            is_weekend          = weekend,
            is_high_risk_window = hi_risk,
            is_avoid_window     = avoid_win,
            session_label       = s_label,
            session_phase       = phase,
            minutes_since_open  = round(mins_open,  2),
            minutes_until_close = round(mins_close, 2),
            minutes_until_open  = round(mins_until, 2),
            can_trade           = can_trade,
            cannot_trade_reason = ct_reason,
            opening_range_complete  = or_complete,
            opening_range_minutes   = self._or_minutes,
            opening_range_bars      = list(self._or_bars),
            market_open_time  = open_dt.isoformat(),
            market_close_time = close_dt.isoformat(),
            session_date      = now.astimezone(ET).date().isoformat(),
            snapshot_time     = now.astimezone(ET).isoformat(),
        )

        self._last_ctx = ctx
        return ctx

    def add_opening_range_bar(self, bar: dict) -> None:
        """
        Register a bar as part of the opening range.
        Called by candle_builder or the data loop for every bar in the
        first `opening_range_minutes` of the session.
        """
        self._or_bars.append(bar)
        log.debug("[session] Opening range bar added (%d total)", len(self._or_bars))

    def opening_range_bars(self) -> list[dict]:
        """Return the accumulated opening range bars (read-only copy)."""
        return list(self._or_bars)

    def is_opening_range_complete(self) -> bool:
        return self._or_complete

    def last_context(self) -> Optional[SessionContext]:
        """Return the most recently built SessionContext without recomputing."""
        return self._last_ctx

    def session_strength(self, ctx: Optional[SessionContext] = None) -> str:
        """
        Rate overall session quality for scoring purposes.

        Returns:
            "strong"   — active or power hour, no high-risk window
            "moderate" — opening or afternoon phase
            "weak"     — midday, extended hours, or high-risk window active
            "closed"   — market is not open
        """
        c = ctx or self._last_ctx
        if c is None:
            return "closed"
        if not c.is_market_open:
            return "closed"
        if c.is_high_risk_window or c.is_avoid_window:
            return "weak"
        if c.session_phase in (SessionPhase.ACTIVE, SessionPhase.POWER_HOUR):
            return "strong"
        if c.session_phase in (SessionPhase.OPENING, SessionPhase.AFTERNOON):
            return "moderate"
        if c.session_phase == SessionPhase.MIDDAY:
            return "weak"
        return "moderate"

    def should_scan(self, ctx: Optional[SessionContext] = None) -> bool:
        """
        True when the scanner should run.
        Scanning is allowed during market hours and, if configured,
        during extended hours.  It is blocked on weekends and when the
        market is fully closed outside extended hours.
        """
        c = ctx or self.get_context()
        after_hours_scan_only = self._settings.get("mode", {}).get(
            "after_hours_scan_only", False)

        if c.is_weekend:
            return False
        if c.is_market_open:
            return True
        if after_hours_scan_only and c.is_extended_hours:
            return True
        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_reset_session(self, now: datetime) -> None:
        """
        Detect a new trading day and reset opening-range state.
        Called at the start of every get_context() call.
        """
        today = now.astimezone(ET).date().isoformat()
        if self._session_date != today:
            if self._session_date is not None:
                log.info(
                    "[session] New session detected — resetting opening range state "
                    "(prev: %s → now: %s)",
                    self._session_date, today,
                )
            self._session_date = today
            self._or_bars      = []
            self._or_complete  = False
