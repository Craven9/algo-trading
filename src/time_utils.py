"""
src/core/time_utils.py — Market session time utilities
All time logic is centralized here so every module agrees on what
"market is open", "pre-market", "high-risk window", etc. means.

All datetimes are timezone-aware.  The market timezone is America/New_York.
The bot reads session config from bot_settings.json → session.

Public API:
    now_et()                    → current ET datetime
    is_market_open()            → bool
    is_pre_market()             → bool
    is_after_hours()            → bool
    is_extended_hours()         → bool
    is_high_risk_window()       → bool
    is_avoid_window()           → bool
    session_label()             → str
    minutes_until_close()       → float
    minutes_since_open()        → float
    market_open_dt()            → datetime
    market_close_dt()           → datetime
    et_time_from_str(hh_mm)     → time object in ET
    is_same_session(dt1, dt2)   → bool
    trading_day_start()         → datetime  (open minus avoid_first_minutes)
    trading_day_end()           → datetime  (close minus avoid_last_minutes)
    can_trade_now()             → tuple[bool, str]
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# ── Market timezone ───────────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")


# ── Defaults (used when no settings dict is supplied) ─────────────────────────

_DEFAULTS = {
    "market_open":          "09:30",
    "market_close":         "16:00",
    "pre_market_start":     "04:00",
    "after_hours_end":      "20:00",
    "timezone":             "America/New_York",
    "avoid_first_minutes":  1,
    "avoid_last_minutes":   5,
    "high_risk_windows": [
        {"start": "09:30", "end": "09:31", "reason": "open_volatility"},
        {"start": "15:55", "end": "16:00", "reason": "close_volatility"},
    ],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_hhmm(s: str) -> time:
    """Parse 'HH:MM' string into a time object."""
    h, m = s.split(":")
    return time(int(h), int(m), tzinfo=ET)


def _cfg(settings: Optional[dict]) -> dict:
    """Extract the session sub-dict from a settings dict, falling back to defaults."""
    if settings is None:
        return _DEFAULTS
    return {**_DEFAULTS, **settings.get("session", {})}


def _today_at(t: time, ref: Optional[datetime] = None) -> datetime:
    """
    Return today's date (ET) combined with the given time (ET).
    Pass ref to anchor 'today' to a specific datetime (useful in tests).
    """
    base = ref.astimezone(ET) if ref else datetime.now(ET)
    return datetime(base.year, base.month, base.day,
                    t.hour, t.minute, t.second, tzinfo=ET)


# ── Core getters ──────────────────────────────────────────────────────────────

def now_et() -> datetime:
    """Current time in Eastern Time (timezone-aware)."""
    return datetime.now(ET)


def market_open_dt(settings: Optional[dict] = None,
                   ref: Optional[datetime] = None) -> datetime:
    """Today's market open as a timezone-aware ET datetime."""
    cfg = _cfg(settings)
    t   = _parse_hhmm(cfg["market_open"])
    return _today_at(t, ref)


def market_close_dt(settings: Optional[dict] = None,
                    ref: Optional[datetime] = None) -> datetime:
    """Today's market close as a timezone-aware ET datetime."""
    cfg = _cfg(settings)
    t   = _parse_hhmm(cfg["market_close"])
    return _today_at(t, ref)


def pre_market_start_dt(settings: Optional[dict] = None,
                         ref: Optional[datetime] = None) -> datetime:
    """Today's pre-market start as a timezone-aware ET datetime."""
    cfg = _cfg(settings)
    t   = _parse_hhmm(cfg["pre_market_start"])
    return _today_at(t, ref)


def after_hours_end_dt(settings: Optional[dict] = None,
                        ref: Optional[datetime] = None) -> datetime:
    """Today's after-hours end as a timezone-aware ET datetime."""
    cfg = _cfg(settings)
    t   = _parse_hhmm(cfg["after_hours_end"])
    return _today_at(t, ref)


def trading_day_start(settings: Optional[dict] = None,
                       ref: Optional[datetime] = None) -> datetime:
    """
    Earliest safe entry time = market open + avoid_first_minutes.
    Trades should not be placed before this time.
    """
    cfg    = _cfg(settings)
    offset = int(cfg.get("avoid_first_minutes", 1))
    return market_open_dt(settings, ref) + timedelta(minutes=offset)


def trading_day_end(settings: Optional[dict] = None,
                     ref: Optional[datetime] = None) -> datetime:
    """
    Latest safe entry time = market close - avoid_last_minutes.
    No new trades should be placed after this time.
    """
    cfg    = _cfg(settings)
    offset = int(cfg.get("avoid_last_minutes", 5))
    return market_close_dt(settings, ref) - timedelta(minutes=offset)


# ── Session state predicates ──────────────────────────────────────────────────

def is_market_open(settings: Optional[dict] = None,
                   ref: Optional[datetime] = None) -> bool:
    """True during regular market hours (09:30–16:00 ET, inclusive)."""
    now = ref.astimezone(ET) if ref else now_et()
    return market_open_dt(settings, ref) <= now < market_close_dt(settings, ref)


def is_pre_market(settings: Optional[dict] = None,
                  ref: Optional[datetime] = None) -> bool:
    """True during pre-market hours (04:00–09:30 ET)."""
    now = ref.astimezone(ET) if ref else now_et()
    return pre_market_start_dt(settings, ref) <= now < market_open_dt(settings, ref)


def is_after_hours(settings: Optional[dict] = None,
                   ref: Optional[datetime] = None) -> bool:
    """True during after-hours trading (16:00–20:00 ET)."""
    now = ref.astimezone(ET) if ref else now_et()
    return market_close_dt(settings, ref) <= now < after_hours_end_dt(settings, ref)


def is_extended_hours(settings: Optional[dict] = None,
                       ref: Optional[datetime] = None) -> bool:
    """True during either pre-market or after-hours."""
    return is_pre_market(settings, ref) or is_after_hours(settings, ref)


def is_weekend(ref: Optional[datetime] = None) -> bool:
    """True on Saturday (5) or Sunday (6) ET."""
    now = ref.astimezone(ET) if ref else now_et()
    return now.weekday() >= 5


def is_trading_day(ref: Optional[datetime] = None) -> bool:
    """True when today is a weekday (does not account for market holidays)."""
    return not is_weekend(ref)


def is_high_risk_window(settings: Optional[dict] = None,
                         ref: Optional[datetime] = None) -> bool:
    """
    True when the current time falls inside any configured high_risk_window.
    High-risk windows are times where volatility or spread is elevated and
    new entries should be avoided.
    """
    cfg     = _cfg(settings)
    now     = ref.astimezone(ET) if ref else now_et()
    windows = cfg.get("high_risk_windows", [])
    for w in windows:
        try:
            w_start = _today_at(_parse_hhmm(w["start"]), ref)
            w_end   = _today_at(_parse_hhmm(w["end"]),   ref)
            if w_start <= now < w_end:
                return True
        except (KeyError, ValueError):
            continue
    return False


def is_avoid_window(settings: Optional[dict] = None,
                     ref: Optional[datetime] = None) -> bool:
    """
    True during the avoid_first_minutes and avoid_last_minutes buffers
    around the open and close respectively.
    These are distinct from high_risk_windows and apply to all entries.
    """
    now    = ref.astimezone(ET) if ref else now_et()
    t_open = market_open_dt(settings, ref)
    t_safe = trading_day_start(settings, ref)
    t_end  = trading_day_end(settings, ref)
    t_cls  = market_close_dt(settings, ref)

    in_open_buffer  = t_open <= now < t_safe
    in_close_buffer = t_end  <= now < t_cls
    return in_open_buffer or in_close_buffer


# ── Session label ─────────────────────────────────────────────────────────────

def session_label(settings: Optional[dict] = None,
                  ref: Optional[datetime] = None) -> str:
    """
    Human-readable label for the current trading session.

    Returns one of:
        "pre_market"   — 04:00–09:30
        "market_open"  — 09:30–16:00
        "after_hours"  — 16:00–20:00
        "closed"       — outside all sessions or weekend
    """
    if is_weekend(ref):
        return "closed"
    if is_pre_market(settings, ref):
        return "pre_market"
    if is_market_open(settings, ref):
        return "market_open"
    if is_after_hours(settings, ref):
        return "after_hours"
    return "closed"


# ── Time-distance helpers ─────────────────────────────────────────────────────

def minutes_until_close(settings: Optional[dict] = None,
                          ref: Optional[datetime] = None) -> float:
    """
    Minutes remaining until market close.
    Returns 0.0 when the market is already closed.
    """
    now   = ref.astimezone(ET) if ref else now_et()
    close = market_close_dt(settings, ref)
    delta = (close - now).total_seconds() / 60
    return max(delta, 0.0)


def minutes_since_open(settings: Optional[dict] = None,
                        ref: Optional[datetime] = None) -> float:
    """
    Minutes elapsed since market open.
    Returns 0.0 before the open.
    """
    now  = ref.astimezone(ET) if ref else now_et()
    open_dt = market_open_dt(settings, ref)
    delta   = (now - open_dt).total_seconds() / 60
    return max(delta, 0.0)


def minutes_until_open(settings: Optional[dict] = None,
                        ref: Optional[datetime] = None) -> float:
    """
    Minutes until the next market open.
    Returns 0.0 when the market is already open or past open.
    """
    now     = ref.astimezone(ET) if ref else now_et()
    open_dt = market_open_dt(settings, ref)
    delta   = (open_dt - now).total_seconds() / 60
    return max(delta, 0.0)


# ── Session identity ──────────────────────────────────────────────────────────

def is_same_session(dt1: datetime, dt2: datetime) -> bool:
    """
    True when two datetimes fall on the same calendar date in ET.
    Used to decide whether VWAP should be reset (new session = new VWAP).
    """
    d1 = dt1.astimezone(ET).date()
    d2 = dt2.astimezone(ET).date()
    return d1 == d2


def et_time_from_str(hhmm: str) -> time:
    """
    Parse 'HH:MM' into a time object localized to ET.
    Convenience wrapper for callers that need a time object.
    """
    return _parse_hhmm(hhmm)


# ── Master trade-time gate ────────────────────────────────────────────────────

def can_trade_now(settings: Optional[dict] = None,
                  ref: Optional[datetime] = None) -> tuple[bool, str]:
    """
    Master check: should the bot attempt new entries right now?

    Returns:
        (True,  "ok")          — safe to evaluate entries
        (False, reason_string) — blocked; reason explains why

    Order of checks mirrors the design doc's Step 1 safety flow.
    """
    if is_weekend(ref):
        return False, "weekend — market closed"

    cfg = _cfg(settings)

    # After-hours scan mode: only scan, never enter
    after_hours_scan_only = settings.get("mode", {}).get(
        "after_hours_scan_only", False) if settings else False

    label = session_label(settings, ref)

    if label == "closed":
        return False, "outside all trading sessions"

    if label in ("pre_market", "after_hours") and not after_hours_scan_only:
        return False, f"extended hours ({label}) — after_hours_scan_only is off"

    if label in ("pre_market", "after_hours") and after_hours_scan_only:
        # Scan-only mode: analysis allowed, execution blocked
        return False, f"extended hours ({label}) — scan only, no new entries"

    if is_avoid_window(settings, ref):
        avoid_first = cfg.get("avoid_first_minutes", 1)
        avoid_last  = cfg.get("avoid_last_minutes",  5)
        return False, (
            f"avoid window — first {avoid_first} min or last {avoid_last} min "
            f"of session"
        )

    if is_high_risk_window(settings, ref):
        return False, "high-risk window — elevated volatility period"

    return True, "ok"
