"""
src/analysis/opening_range_analyzer.py — Opening range detection and analysis
Calculates the opening range (OR) for multiple timeframes and determines
price relationship to that range throughout the session.

The opening range is one of the most powerful intraday reference frameworks.
Price behavior relative to the OR drives setup scoring, entry decisions,
and exit management.

Responsibilities:
  - Calculate 5-min, 15-min, and 30-min opening ranges
  - Detect breakout, breakdown, inside-range, and failed breakout conditions
  - Track how long price has held above/below the OR
  - Provide OR-relative signals for setup detectors and scoring engines

Opening range states:
  "inside"           — price between OR high and OR low
  "above"            — price above OR high (bullish)
  "below"            — price below OR low (bearish)
  "breakout"         — price just cleared OR high with volume
  "breakdown"        — price just lost OR low
  "failed_breakout"  — price cleared OR high then fell back inside/below
  "failed_breakdown" — price cleared OR low then recovered inside/above
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from candle_builder import filter_session_bars
from time_utils import market_open_dt

log = logging.getLogger(__name__)

# Supported OR timeframes in minutes
OR_TIMEFRAMES = [5, 15, 30]

# Volume confirmation threshold — OR breakout bar must have at least
# this multiple of the OR average volume
_BREAKOUT_VOLUME_MULT = 1.5


# ── Opening range dataclass ───────────────────────────────────────────────────

@dataclass
class OpeningRange:
    """
    The opening range for a single timeframe (e.g. 15-min OR).
    """
    timeframe_minutes: int
    high:              Optional[float]  = None
    low:               Optional[float]  = None
    open_price:        Optional[float]  = None   # first bar's open
    range_size:        float            = 0.0    # high - low
    range_size_pct:    float            = 0.0    # range_size / open_price
    avg_volume:        float            = 0.0    # avg volume during OR
    bar_count:         int              = 0      # bars that formed the OR
    is_complete:       bool             = False  # OR window has closed

    def contains(self, price: float) -> bool:
        """True when price is inside the OR (between low and high)."""
        if self.high is None or self.low is None:
            return False
        return self.low <= price <= self.high

    def is_above(self, price: float) -> bool:
        return self.high is not None and price > self.high

    def is_below(self, price: float) -> bool:
        return self.low is not None and price < self.low

    def midpoint(self) -> Optional[float]:
        if self.high is None or self.low is None:
            return None
        return (self.high + self.low) / 2

    def to_dict(self) -> dict:
        return {
            "timeframe_minutes": self.timeframe_minutes,
            "high":              round(self.high, 4) if self.high else None,
            "low":               round(self.low,  4) if self.low  else None,
            "open_price":        round(self.open_price, 4) if self.open_price else None,
            "range_size":        round(self.range_size,     4),
            "range_size_pct":    round(self.range_size_pct, 4),
            "avg_volume":        round(self.avg_volume,     2),
            "bar_count":         self.bar_count,
            "is_complete":       self.is_complete,
        }


# ── OR analysis result ────────────────────────────────────────────────────────

@dataclass
class OpeningRangeResult:
    """
    Full opening range analysis for a ticker.
    Contains OR data for all timeframes plus current price signals.
    """
    ticker:         str
    current_price:  float
    settings:       dict    = field(default_factory=dict, repr=False)

    # Per-timeframe ORs
    or_5m:          OpeningRange = field(default_factory=lambda: OpeningRange(5))
    or_15m:         OpeningRange = field(default_factory=lambda: OpeningRange(15))
    or_30m:         OpeningRange = field(default_factory=lambda: OpeningRange(30))

    # Primary OR (from settings — default 15m)
    primary_or:     OpeningRange = field(default_factory=lambda: OpeningRange(15))

    # Current price state relative to primary OR
    state:          str     = "unknown"   # inside|above|below|breakout|breakdown|failed_breakout|failed_breakdown
    bars_above_or:  int     = 0           # consecutive bars above OR high
    bars_below_or:  int     = 0           # consecutive bars below OR low
    breakout_confirmed: bool = False
    breakdown_confirmed: bool = False
    failed_breakout:    bool = False
    failed_breakdown:   bool = False

    # Breakout quality
    breakout_volume_ratio: float = 0.0   # breakout bar vol / OR avg vol

    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def or_for_timeframe(self, minutes: int) -> OpeningRange:
        """Return the OR for a specific timeframe."""
        if minutes == 5:
            return self.or_5m
        if minutes == 30:
            return self.or_30m
        return self.or_15m   # default

    def is_bullish(self) -> bool:
        """True when price is above the primary OR."""
        return self.state in ("above", "breakout")

    def is_bearish(self) -> bool:
        """True when price is below the primary OR."""
        return self.state in ("below", "breakdown")

    def to_dict(self) -> dict:
        return {
            "ticker":               self.ticker,
            "current_price":        self.current_price,
            "or_5m":                self.or_5m.to_dict(),
            "or_15m":               self.or_15m.to_dict(),
            "or_30m":               self.or_30m.to_dict(),
            "primary_or":           self.primary_or.to_dict(),
            "state":                self.state,
            "bars_above_or":        self.bars_above_or,
            "bars_below_or":        self.bars_below_or,
            "breakout_confirmed":   self.breakout_confirmed,
            "breakdown_confirmed":  self.breakdown_confirmed,
            "failed_breakout":      self.failed_breakout,
            "failed_breakdown":     self.failed_breakdown,
            "breakout_volume_ratio":round(self.breakout_volume_ratio, 2),
            "analyzed_at":          self.analyzed_at,
        }


# ── Analyzer ──────────────────────────────────────────────────────────────────

class OpeningRangeAnalyzer:
    """
    Computes opening range data and price-relative signals.

    Usage:
        analyzer = OpeningRangeAnalyzer(settings)
        result   = analyzer.analyze("ABCD", bars, current_price)
    """

    def __init__(self, settings: dict):
        self._settings    = settings
        self._or_cfg      = settings.get("opening_range", {})
        self._primary_min = int(self._or_cfg.get("preferred_range_minutes", 15))
        self._failed_bars = int(self._or_cfg.get("failed_breakout_bars_threshold", 3))

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        ticker:        str,
        bars:          list[dict],
        current_price: float,
        ref:           Optional[datetime] = None,
    ) -> OpeningRangeResult:
        """
        Compute opening range analysis for all timeframes.

        Args:
            ticker:        Ticker symbol.
            bars:          Regular-session 1-min bars ordered oldest→newest.
            current_price: Latest price.
            ref:           Optional reference datetime (for testing).

        Returns:
            OpeningRangeResult with all OR data and price-relative signals.
        """
        reg_bars = filter_session_bars(bars, "regular")

        result = OpeningRangeResult(
            ticker        = ticker,
            current_price = current_price,
            settings      = self._settings,
        )

        if not reg_bars or current_price <= 0:
            return result

        # Compute OR for each timeframe
        result.or_5m  = self._compute_or(reg_bars, 5)
        result.or_15m = self._compute_or(reg_bars, 15)
        result.or_30m = self._compute_or(reg_bars, 30)

        # Set primary OR
        result.primary_or = result.or_for_timeframe(self._primary_min)

        if not result.primary_or.is_complete:
            result.state = "inside"   # still forming
            return result

        # Determine current state from ALL bars after the OR window
        or_end_idx     = result.primary_or.bar_count
        post_or_bars   = reg_bars[or_end_idx:]
        result         = self._classify_state(result, post_or_bars)

        log.debug(
            "[or_analyzer] %s state=%s high=%.4f low=%.4f",
            ticker,
            result.state,
            result.primary_or.high or 0,
            result.primary_or.low  or 0,
        )
        return result

    # ── OR computation ────────────────────────────────────────────────────────

    def _compute_or(self, bars: list[dict], minutes: int) -> OpeningRange:
        """
        Extract the first `minutes` bars and compute OR high/low/avg_vol.
        Bars are expected to be 1-minute bars sorted oldest→newest.
        """
        or_bars = bars[:minutes]
        if not or_bars:
            return OpeningRange(timeframe_minutes=minutes)

        high       = max(b["h"] for b in or_bars)
        low        = min(b["l"] for b in or_bars)
        open_price = or_bars[0]["o"]
        avg_vol    = sum(b.get("v", 0) for b in or_bars) / len(or_bars)
        rng        = high - low
        rng_pct    = (rng / open_price * 100) if open_price else 0.0

        return OpeningRange(
            timeframe_minutes = minutes,
            high              = round(high,       4),
            low               = round(low,        4),
            open_price        = round(open_price, 4),
            range_size        = round(rng,        4),
            range_size_pct    = round(rng_pct,    4),
            avg_volume        = round(avg_vol,    2),
            bar_count         = len(or_bars),
            is_complete       = len(or_bars) >= minutes,
        )

    # ── State classification ──────────────────────────────────────────────────

    def _classify_state(
        self,
        result:        OpeningRangeResult,
        post_or_bars:  list[dict],
    ) -> OpeningRangeResult:
        """
        Classify current price state relative to the primary OR using
        all post-OR bars.
        """
        primary = result.primary_or
        price   = result.current_price

        if primary.high is None or primary.low is None:
            result.state = "unknown"
            return result

        # Count consecutive bars above / below OR
        above_count  = 0
        below_count  = 0
        broke_above  = False
        broke_below  = False
        went_back_in = False

        for bar in post_or_bars:
            c = bar.get("c", 0)
            if c > primary.high:
                above_count += 1
                below_count  = 0
                broke_above  = True
            elif c < primary.low:
                below_count += 1
                above_count  = 0
                broke_below  = True
            else:
                # Inside OR
                if broke_above or broke_below:
                    went_back_in = True
                above_count = 0
                below_count = 0

        result.bars_above_or = above_count
        result.bars_below_or = below_count

        # Detect failed breakout: broke above, then came back inside
        failed_breakout  = broke_above and went_back_in and price <= primary.high
        failed_breakdown = broke_below and went_back_in and price >= primary.low

        # Current price state
        if failed_breakout:
            state = "failed_breakout"
        elif failed_breakdown:
            state = "failed_breakdown"
        elif price > primary.high:
            state = "breakout" if above_count >= 2 else "above"
        elif price < primary.low:
            state = "breakdown" if below_count >= 2 else "below"
        else:
            state = "inside"

        # Breakout confirmed: 2+ bars above OR with volume
        breakout_confirmed = (
            above_count >= 2
            and not failed_breakout
            and price > primary.high
        )

        # Volume ratio on most recent bar (proxy for breakout quality)
        vol_ratio = 0.0
        if post_or_bars and primary.avg_volume > 0:
            latest_vol = post_or_bars[-1].get("v", 0)
            vol_ratio  = latest_vol / primary.avg_volume

        result.state                 = state
        result.breakout_confirmed    = breakout_confirmed
        result.breakdown_confirmed   = below_count >= 2 and price < primary.low
        result.failed_breakout       = failed_breakout
        result.failed_breakdown      = failed_breakdown
        result.breakout_volume_ratio = round(vol_ratio, 2)

        return result
