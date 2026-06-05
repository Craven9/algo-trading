"""
src/data/indicator_engine.py — Indicator computation engine
Wraps indicator_calculator.py and produces IndicatorSnapshot objects
that every other module consumes.

Responsibilities:
  - Accept raw bar lists and settings, return typed IndicatorSnapshot
  - Cache the last snapshot per ticker to avoid redundant computation
  - Validate bar quality before computing (min bar count, price sanity)
  - Log indicator computation events for debugging
  - Provide a multi-ticker batch computation path for the scanner loop

This is the ONLY file that calls indicator_calculator.compute_all().
All other modules read from IndicatorSnapshot — they never call
indicator_calculator directly.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

# Internal imports
import indicator_calculator as ic
from models import (
    IndicatorSnapshot,
    MacdResult,
    MaTrend,
    PriceVsVwap,
    RsiZone,
    VolumeDirection,
)

log = logging.getLogger(__name__)

# Minimum number of bars required to compute a meaningful indicator set.
# Below this threshold the snapshot is returned with all None/default values.
MIN_BARS_REQUIRED = 30


# ── Main entry point ──────────────────────────────────────────────────────────

def compute(
    bars: list[dict],
    settings: dict,
    ticker: str = "",
) -> IndicatorSnapshot:
    """
    Compute all indicators for a ticker and return a typed IndicatorSnapshot.

    Args:
        bars:     List of OHLCV bar dicts (canonical schema from candle_builder),
                  ordered oldest → newest.  Caller must pre-filter to session
                  bars for correct VWAP.
        settings: Full bot_settings dict (indicator periods are read from
                  settings["indicators"]).
        ticker:   Optional ticker symbol for logging purposes.

    Returns:
        IndicatorSnapshot — all fields populated, or default/None values
        when there is insufficient data.
    """
    label = f"[{ticker}]" if ticker else "[unknown]"

    if not bars:
        log.debug("%s compute() called with empty bar list", label)
        return IndicatorSnapshot()

    if len(bars) < MIN_BARS_REQUIRED:
        log.debug(
            "%s insufficient bars for indicators: %d < %d",
            label, len(bars), MIN_BARS_REQUIRED,
        )
        return _snapshot_from_raw({}, bars)

    # Validate bar quality — reject batches with clearly bad prices
    if not _bars_are_valid(bars):
        log.warning("%s bar quality check failed — returning empty snapshot", label)
        return IndicatorSnapshot()

    try:
        raw = ic.compute_all(bars, settings)
    except Exception as exc:
        log.error("%s indicator_calculator.compute_all() raised: %s", label, exc)
        return IndicatorSnapshot()

    snapshot = _snapshot_from_raw(raw, bars)
    log.debug(
        "%s indicators computed: rsi=%.1f vwap=%.4f ma_trend=%s rvol=%.2f",
        label,
        snapshot.rsi or 0,
        snapshot.vwap or 0,
        snapshot.ma_trend,
        snapshot.relative_volume,
    )
    return snapshot


def compute_batch(
    ticker_bars: dict[str, list[dict]],
    settings: dict,
) -> dict[str, IndicatorSnapshot]:
    """
    Compute indicators for multiple tickers in one call.

    Args:
        ticker_bars: Dict mapping ticker symbol → bar list.
        settings:    Full bot_settings dict.

    Returns:
        Dict mapping ticker symbol → IndicatorSnapshot.
    """
    results: dict[str, IndicatorSnapshot] = {}
    for ticker, bars in ticker_bars.items():
        results[ticker] = compute(bars, settings, ticker=ticker)
    return results


# ── Snapshot construction ─────────────────────────────────────────────────────

def _snapshot_from_raw(raw: dict, bars: list[dict]) -> IndicatorSnapshot:
    """
    Convert the raw dict from indicator_calculator.compute_all() into a
    fully typed IndicatorSnapshot.

    Falls back gracefully when any field is missing or None.
    """
    macd_raw  = raw.get("macd")
    macd_obj  = _build_macd(macd_raw)

    # Derive price_vs_vwap and vwap_distance_pct safely
    vwap_val      = raw.get("vwap")
    latest_close  = raw.get("latest_close") or (bars[-1]["c"] if bars else 0.0)

    if vwap_val and vwap_val > 0:
        price_vs_vwap    = PriceVsVwap.ABOVE.value if latest_close > vwap_val else PriceVsVwap.BELOW.value
        vwap_dist        = raw.get("vwap_distance_pct", 0.0) or 0.0
        vwap_extended    = raw.get("vwap_extended", False)
    else:
        price_vs_vwap    = PriceVsVwap.BELOW.value
        vwap_dist        = 0.0
        vwap_extended    = False

    # RSI zone
    rsi_val  = raw.get("rsi")
    rsi_zone = _safe_str(raw.get("rsi_zone"), RsiZone.UNKNOWN.value)

    # MA trend
    ma_trend = _safe_str(raw.get("ma_trend"), MaTrend.FLAT.value)

    # Volume trend
    vol_trend = _safe_str(raw.get("volume_trend"), VolumeDirection.FLAT.value)

    return IndicatorSnapshot(
        # VWAP
        vwap              = vwap_val,
        price_vs_vwap     = price_vs_vwap,
        vwap_distance_pct = round(float(vwap_dist), 4),
        vwap_extended     = bool(vwap_extended),

        # RSI
        rsi               = _safe_float(rsi_val),
        rsi_zone          = rsi_zone,

        # MACD
        macd              = macd_obj,

        # ATR
        atr               = _safe_float(raw.get("atr")),

        # Moving averages
        ma_fast           = _safe_float(raw.get("ma_fast")),
        ma_slow           = _safe_float(raw.get("ma_slow")),
        ma_trend          = ma_trend,
        ma_spread_pct     = round(float(raw.get("ma_spread_pct") or 0.0), 4),

        # Volume
        relative_volume   = round(float(raw.get("relative_volume") or 0.0), 4),
        volume_trend      = vol_trend,

        # Candle strength
        candle_strength      = round(float(raw.get("candle_strength") or 0.0), 4),
        avg_candle_strength  = round(float(raw.get("avg_candle_strength") or 0.0), 4),

        # Market structure
        higher_lows       = bool(raw.get("higher_lows", False)),
        lower_highs       = bool(raw.get("lower_highs", False)),
        swing_highs       = list(raw.get("swing_highs") or []),
        swing_lows        = list(raw.get("swing_lows") or []),

        # Raw bar data
        latest_close      = float(latest_close),
        latest_bar        = dict(raw.get("latest_bar") or (bars[-1] if bars else {})),
    )


def _build_macd(raw: Optional[dict]) -> Optional[MacdResult]:
    """Convert the raw MACD dict from indicator_calculator into a MacdResult."""
    if not raw or not isinstance(raw, dict):
        return None
    return MacdResult(
        macd                = float(raw.get("macd")      or 0.0),
        signal              = float(raw.get("signal")    or 0.0),
        histogram           = float(raw.get("histogram") or 0.0),
        histogram_direction = raw.get("histogram_direction", "contracting"),
        bullish_crossover   = bool(raw.get("bullish_crossover", False)),
        bearish_crossover   = bool(raw.get("bearish_crossover", False)),
        bullish             = bool(raw.get("bullish", False)),
    )


# ── Bar validation ────────────────────────────────────────────────────────────

def _bars_are_valid(bars: list[dict]) -> bool:
    """
    Quick sanity check on a bar list before running indicators.

    Rejects if:
      - Any bar has a zero or negative close price
      - Any bar has a high below its low
      - More than 20% of bars have zero volume
      - The latest close is not a finite number
    """
    if not bars:
        return False

    zero_price_count  = 0
    bad_ohlc_count    = 0
    zero_volume_count = 0

    for b in bars:
        c = b.get("c", 0)
        h = b.get("h", 0)
        l = b.get("l", 0)
        v = b.get("v", 0)

        if not c or c <= 0 or not math.isfinite(c):
            zero_price_count += 1
        if h < l:
            bad_ohlc_count += 1
        if v == 0:
            zero_volume_count += 1

    n = len(bars)
    if zero_price_count > 0:
        log.warning("Bar validation: %d bars have zero/negative close", zero_price_count)
        return False
    if bad_ohlc_count > 0:
        log.warning("Bar validation: %d bars have high < low", bad_ohlc_count)
        return False
    if zero_volume_count / n > 0.20:
        log.warning(
            "Bar validation: %.0f%% of bars have zero volume", zero_volume_count / n * 100
        )
        return False

    latest_c = bars[-1].get("c", 0)
    if not latest_c or not math.isfinite(latest_c):
        return False

    return True


# ── Per-ticker snapshot cache ─────────────────────────────────────────────────

class IndicatorCache:
    """
    Lightweight in-memory cache of the most recent IndicatorSnapshot
    per ticker.  Avoids recomputing indicators when bars haven't changed.

    Usage:
        cache = IndicatorCache()
        snap  = cache.get_or_compute("ABCD", bars, settings)
    """

    def __init__(self):
        self._cache: dict[str, IndicatorSnapshot] = {}
        self._bar_counts: dict[str, int]           = {}

    def get_or_compute(
        self,
        ticker: str,
        bars: list[dict],
        settings: dict,
    ) -> IndicatorSnapshot:
        """
        Return the cached snapshot if the bar count hasn't changed,
        otherwise recompute and update the cache.
        """
        current_count = len(bars)
        cached_count  = self._bar_counts.get(ticker, -1)

        if ticker in self._cache and current_count == cached_count:
            log.debug("[%s] Returning cached indicator snapshot", ticker)
            return self._cache[ticker]

        snap = compute(bars, settings, ticker=ticker)
        self._cache[ticker]      = snap
        self._bar_counts[ticker] = current_count
        return snap

    def invalidate(self, ticker: str) -> None:
        """Remove a ticker from the cache."""
        self._cache.pop(ticker, None)
        self._bar_counts.pop(ticker, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._bar_counts.clear()

    def cached_tickers(self) -> list[str]:
        return list(self._cache.keys())

    def get(self, ticker: str) -> Optional[IndicatorSnapshot]:
        """Return cached snapshot or None if not cached."""
        return self._cache.get(ticker)


# ── Derived signal helpers ────────────────────────────────────────────────────
# Convenience functions that extract commonly checked signal combinations
# from a snapshot.  Setup detectors and scoring engines use these instead
# of reading raw fields directly.

def is_bullish_momentum(snap: IndicatorSnapshot) -> bool:
    """
    True when multiple momentum indicators align bullishly:
      - Price above VWAP
      - MA trend is bullish
      - MACD is bullish (above signal line)
      - RSI is not overbought
    """
    macd_bullish = snap.macd.bullish if snap.macd else False
    return (
        snap.price_vs_vwap == PriceVsVwap.ABOVE.value
        and snap.ma_trend   == MaTrend.BULLISH.value
        and macd_bullish
        and snap.rsi_zone   != RsiZone.OVERBOUGHT.value
    )


def is_bearish_momentum(snap: IndicatorSnapshot) -> bool:
    """
    True when multiple momentum indicators align bearishly:
      - Price below VWAP
      - MA trend is bearish
      - MACD is bearish (below signal line)
    """
    macd_bearish = not snap.macd.bullish if snap.macd else False
    return (
        snap.price_vs_vwap == PriceVsVwap.BELOW.value
        and snap.ma_trend   == MaTrend.BEARISH.value
        and macd_bearish
    )


def volume_is_confirming(snap: IndicatorSnapshot,
                          min_rvol: float = 2.0) -> bool:
    """True when relative volume meets the minimum threshold."""
    return snap.relative_volume >= min_rvol


def has_bullish_crossover(snap: IndicatorSnapshot) -> bool:
    """True on the bar where MACD crosses above its signal line."""
    return snap.macd.bullish_crossover if snap.macd else False


def has_bearish_crossover(snap: IndicatorSnapshot) -> bool:
    """True on the bar where MACD crosses below its signal line."""
    return snap.macd.bearish_crossover if snap.macd else False


def is_overextended(snap: IndicatorSnapshot,
                     max_vwap_pct: float = 8.0) -> bool:
    """True when price is too far above VWAP for a safe entry."""
    return snap.vwap_extended or snap.vwap_distance_pct > max_vwap_pct


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Return a float or None, guarding against NaN and inf."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_str(val, default: str) -> str:
    """Return a string value or the default if val is None/empty."""
    if val is None:
        return default
    return str(val)
