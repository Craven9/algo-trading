"""
src/data/candle_builder.py — OHLCV candle construction and management
Builds and maintains candle bars from raw trade ticks or from bar data
returned by the Alpaca data API.

Responsibilities:
  - Aggregate raw ticks into OHLCV bars for any timeframe
  - Accept pre-built bars from the Alpaca bars API and normalize them
  - Track session bars separately from pre/after-market bars
  - Detect and flag stale or incomplete bars
  - Provide clean bar lists to indicator_calculator and setup detectors

Bar dict schema (canonical across the whole bot):
    {
        "t":  ISO-8601 timestamp string (bar open time, ET)
        "o":  float  — open
        "h":  float  — high
        "l":  float  — low
        "c":  float  — close
        "v":  int    — volume
        "vw": float  — VWAP for this bar (if available)
        "n":  int    — number of trades in bar (if available)
        "session": "regular" | "pre_market" | "after_hours"
        "complete": bool — True once the bar's time window has closed
    }
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

# Supported timeframes in minutes
SUPPORTED_TIMEFRAMES = {1, 2, 3, 5, 10, 15, 30, 60}


# ── Bar normalization ─────────────────────────────────────────────────────────

def normalize_bar(raw: dict) -> dict:
    """
    Convert an Alpaca API bar dict into the bot's canonical bar schema.

    Alpaca uses keys: t, o, h, l, c, v, vw, n
    The bot uses the same keys but ensures all types are correct and
    adds 'session' and 'complete' fields.

    Args:
        raw: Raw bar dict from Alpaca or another source.

    Returns:
        Normalized bar dict.
    """
    ts_raw = raw.get("t", "")
    ts     = _parse_timestamp(ts_raw)

    bar = {
        "t":        ts.isoformat() if ts else ts_raw,
        "o":        float(raw.get("o", 0)),
        "h":        float(raw.get("h", 0)),
        "l":        float(raw.get("l", 0)),
        "c":        float(raw.get("c", 0)),
        "v":        int(raw.get("v", 0)),
        "vw":       float(raw.get("vw", 0)) if raw.get("vw") else None,
        "n":        int(raw.get("n", 0))    if raw.get("n")  else None,
        "session":  _session_label(ts),
        "complete": True,   # bars from the API are always complete
    }

    # Sanity-check OHLC relationships
    if bar["h"] < bar["l"]:
        bar["h"], bar["l"] = bar["l"], bar["h"]
    if bar["h"] < bar["o"]:
        bar["h"] = bar["o"]
    if bar["h"] < bar["c"]:
        bar["h"] = bar["c"]
    if bar["l"] > bar["o"]:
        bar["l"] = bar["o"]
    if bar["l"] > bar["c"]:
        bar["l"] = bar["c"]

    return bar


def normalize_bars(raw_list: list[dict]) -> list[dict]:
    """Normalize a list of raw bars, filtering out any with zero price."""
    bars = [normalize_bar(b) for b in raw_list]
    return [b for b in bars if b["c"] > 0]


# ── Session filtering ─────────────────────────────────────────────────────────

def filter_session_bars(bars: list[dict],
                         session: str = "regular") -> list[dict]:
    """
    Filter bars to a specific session.

    Args:
        bars:    List of normalized bar dicts.
        session: "regular" | "pre_market" | "after_hours" | "all"

    Returns:
        Filtered list.
    """
    if session == "all":
        return bars
    return [b for b in bars if b.get("session") == session]


def get_session_bars(bars: list[dict]) -> list[dict]:
    """Convenience wrapper — returns only regular-session bars."""
    return filter_session_bars(bars, "regular")


# ── Timeframe resampling ──────────────────────────────────────────────────────

def resample(bars: list[dict], target_minutes: int) -> list[dict]:
    """
    Resample a list of 1-minute bars into a larger timeframe.

    Example: 1-min bars → 5-min bars.

    Args:
        bars:           List of normalized 1-min bar dicts, sorted oldest→newest.
        target_minutes: Target timeframe in minutes.

    Returns:
        List of resampled bars in the canonical schema.
    """
    if target_minutes <= 1:
        return bars
    if not bars:
        return []

    resampled: list[dict] = []
    bucket:    list[dict] = []

    def _flush(bucket: list[dict]) -> Optional[dict]:
        if not bucket:
            return None
        return {
            "t":        bucket[0]["t"],
            "o":        bucket[0]["o"],
            "h":        max(b["h"] for b in bucket),
            "l":        min(b["l"] for b in bucket),
            "c":        bucket[-1]["c"],
            "v":        sum(b["v"] for b in bucket),
            "vw":       None,
            "n":        sum(b["n"] or 0 for b in bucket) or None,
            "session":  bucket[0].get("session", "regular"),
            "complete": bucket[-1].get("complete", True),
        }

    def _bucket_id(ts_str: str) -> int:
        """Map a timestamp string to a bucket index (minutes // target)."""
        dt = _parse_timestamp(ts_str)
        if dt is None:
            return 0
        # Align to ET midnight so bucket boundaries are consistent
        dt_et   = dt.astimezone(ET)
        minutes = dt_et.hour * 60 + dt_et.minute
        return minutes // target_minutes

    prev_id = None
    for bar in bars:
        bid = _bucket_id(bar["t"])
        if prev_id is not None and bid != prev_id:
            merged = _flush(bucket)
            if merged:
                resampled.append(merged)
            bucket = []
        bucket.append(bar)
        prev_id = bid

    # Flush the final bucket
    merged = _flush(bucket)
    if merged:
        resampled.append(merged)

    return resampled


# ── CandleBuilder — stateful tick aggregator ──────────────────────────────────

class CandleBuilder:
    """
    Builds OHLCV bars from a stream of trade ticks or from periodic
    price-update snapshots.

    Typical use:
        builder = CandleBuilder(timeframe_minutes=1)
        for tick in tick_stream:
            completed_bar = builder.update(tick["price"], tick["size"], tick["timestamp"])
            if completed_bar:
                bars.append(completed_bar)
        current = builder.current_bar()

    Also supports ingesting pre-built API bars directly via ingest_bars().
    """

    def __init__(self, timeframe_minutes: int = 1, session: str = "regular"):
        """
        Args:
            timeframe_minutes: Bar size in minutes (default: 1).
            session:           Session label to stamp on bars ("regular",
                               "pre_market", "after_hours").
        """
        if timeframe_minutes not in SUPPORTED_TIMEFRAMES:
            log.warning(
                "[candle_builder] Timeframe %d not in supported set %s — proceeding anyway",
                timeframe_minutes, SUPPORTED_TIMEFRAMES,
            )
        self._tf      = timeframe_minutes
        self._session = session
        self._bars:   list[dict]      = []
        self._current: Optional[dict] = None
        self._current_bucket: Optional[int] = None

    # ── Tick ingestion ────────────────────────────────────────────────────────

    def update(self, price: float, size: int,
               timestamp: Optional[datetime] = None) -> Optional[dict]:
        """
        Feed a single trade tick into the builder.

        Args:
            price:     Trade price.
            size:      Number of shares traded.
            timestamp: UTC-aware datetime of the trade.  Uses now() if None.

        Returns:
            The completed bar dict if this tick closed a bar, else None.
        """
        if price <= 0 or size < 0:
            return None

        ts     = timestamp or datetime.now(timezone.utc)
        ts_et  = ts.astimezone(ET)
        bucket = (ts_et.hour * 60 + ts_et.minute) // self._tf

        completed = None

        if self._current_bucket is not None and bucket != self._current_bucket:
            # Close the current bar
            completed = self._close_current(ts_et)

        if self._current is None:
            self._current = _new_bar(price, ts_et, self._session, size)
        else:
            _update_bar(self._current, price, size)

        self._current_bucket = bucket
        return completed

    def current_bar(self) -> Optional[dict]:
        """Return a snapshot of the in-progress bar (not yet closed)."""
        if self._current is None:
            return None
        snap = dict(self._current)
        snap["complete"] = False
        return snap

    def flush(self) -> Optional[dict]:
        """
        Force-close the current in-progress bar and add it to the bar list.
        Call at session end or when you need the final partial bar.
        """
        if self._current is None:
            return None
        bar = dict(self._current)
        bar["complete"] = True
        self._bars.append(bar)
        self._current        = None
        self._current_bucket = None
        return bar

    # ── Bar ingestion (API bars) ──────────────────────────────────────────────

    def ingest_bars(self, raw_bars: list[dict],
                    session_filter: str = "regular") -> int:
        """
        Ingest a list of pre-built API bars, normalizing and optionally
        filtering by session.

        Args:
            raw_bars:       Raw bar dicts from the Alpaca API.
            session_filter: Which session to keep ("regular", "all", etc.).

        Returns:
            Number of bars ingested.
        """
        normalized = normalize_bars(raw_bars)
        if session_filter != "all":
            normalized = filter_session_bars(normalized, session_filter)
        normalized.sort(key=lambda b: b["t"])
        self._bars.extend(normalized)
        log.debug(
            "[candle_builder] Ingested %d bars (session=%s)",
            len(normalized), session_filter,
        )
        return len(normalized)

    # ── Bar access ────────────────────────────────────────────────────────────

    def bars(self, n: Optional[int] = None) -> list[dict]:
        """
        Return completed bars, oldest→newest.

        Args:
            n: If given, return only the last n bars.
        """
        if n is None:
            return list(self._bars)
        return list(self._bars[-n:])

    def latest_bar(self) -> Optional[dict]:
        """Return the most recently completed bar, or None."""
        return self._bars[-1] if self._bars else None

    def bar_count(self) -> int:
        return len(self._bars)

    def clear(self) -> None:
        """Reset all state — used on session reset."""
        self._bars            = []
        self._current         = None
        self._current_bucket  = None
        log.debug("[candle_builder] State cleared")

    def with_current(self) -> list[dict]:
        """
        Return all completed bars plus the current in-progress bar (if any).
        Useful for real-time indicator calculation.
        """
        result = list(self._bars)
        snap   = self.current_bar()
        if snap:
            result.append(snap)
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _close_current(self, ts_et: datetime) -> dict:
        """Mark the current bar complete, store it, and return it."""
        bar             = dict(self._current)
        bar["complete"] = True
        self._bars.append(bar)
        self._current   = None
        log.debug(
            "[candle_builder] Bar closed: t=%s c=%.4f v=%d",
            bar["t"], bar["c"], bar["v"],
        )
        return bar


# ── Bar helpers ────────────────────────────────────────────────────────────────

def _new_bar(price: float, ts_et: datetime, session: str,
             size: int = 0) -> dict:
    return {
        "t":        ts_et.isoformat(),
        "o":        price,
        "h":        price,
        "l":        price,
        "c":        price,
        "v":        size,
        "vw":       None,
        "n":        1 if size > 0 else 0,
        "session":  session,
        "complete": False,
    }


def _update_bar(bar: dict, price: float, size: int) -> None:
    bar["h"] = max(bar["h"], price)
    bar["l"] = min(bar["l"], price)
    bar["c"] = price
    bar["v"] += size
    bar["n"]  = (bar.get("n") or 0) + 1


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def _parse_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse an ISO-8601 timestamp string into a UTC-aware datetime.
    Handles strings with or without timezone info.
    Returns None on failure.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def _session_label(dt: Optional[datetime]) -> str:
    """Classify a datetime into a session label."""
    if dt is None:
        return "regular"
    dt_et = dt.astimezone(ET)
    t     = dt_et.time()
    from datetime import time as dtime
    pre_start  = dtime(4,  0)
    mkt_open   = dtime(9,  30)
    mkt_close  = dtime(16, 0)
    aft_end    = dtime(20, 0)
    if pre_start <= t < mkt_open:
        return "pre_market"
    if mkt_open <= t < mkt_close:
        return "regular"
    if mkt_close <= t < aft_end:
        return "after_hours"
    return "regular"


# ── Utility functions ─────────────────────────────────────────────────────────

def is_bar_stale(bar: dict, max_age_seconds: float = 120) -> bool:
    """
    True when the bar's timestamp is older than max_age_seconds.
    Used by the data provider to detect stale cache entries.
    """
    ts = _parse_timestamp(bar.get("t", ""))
    if ts is None:
        return True
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return age > max_age_seconds


def bars_to_closes(bars: list[dict]) -> list[float]:
    """Extract close prices from a bar list."""
    return [b["c"] for b in bars]


def bars_to_volumes(bars: list[dict]) -> list[int]:
    """Extract volumes from a bar list."""
    return [b["v"] for b in bars]


def day_high(bars: list[dict]) -> Optional[float]:
    """Highest high across all bars."""
    if not bars:
        return None
    return max(b["h"] for b in bars)


def day_low(bars: list[dict]) -> Optional[float]:
    """Lowest low across all bars."""
    if not bars:
        return None
    return min(b["l"] for b in bars)


def premarket_high(bars: list[dict]) -> Optional[float]:
    """Highest high across pre-market bars only."""
    pre = filter_session_bars(bars, "pre_market")
    return day_high(pre)


def premarket_low(bars: list[dict]) -> Optional[float]:
    """Lowest low across pre-market bars only."""
    pre = filter_session_bars(bars, "pre_market")
    return day_low(pre)
