"""
src/setups/opening_range_breakout.py — Opening Range Breakout setup detector
Detects when price breaks above the opening range high with conviction,
then holds above the range with volume confirmation.

The opening range breakout is a clean intraday momentum setup:
  1. The opening range is complete
  2. Price breaks above the opening range high
  3. Price holds above the opening range high for confirmation
  4. Breakout volume expands versus opening range average volume
  5. VWAP and market structure support continuation

This setup is strongest when:
  - The opening range is not too wide
  - Breakout happens with strong RVOL
  - Price holds above the OR high for multiple bars
  - VWAP is below price / below the OR high
  - The breakout bar has strong candle body
  - Session structure is bullish or improving

Returns a SetupResult with:
  confirmed:     True when all conditions are met
  score:         0–100 based on breakout quality
  entry_trigger: Opening range high or current breakout area
  stop_area:     Below OR high / OR midpoint / recent hold low
  target_area:   Fibonacci extension or next resistance
  reasons:       Human-readable confirmation factors
  warnings:      Any concerns about the setup quality
"""

from __future__ import annotations

import logging
from typing import Optional

from models import ConfidenceLabel, IndicatorSnapshot, SetupResult, SetupName

log = logging.getLogger(__name__)

_SETUP_NAME = SetupName.OPENING_RANGE_BREAKOUT.value


def detect(
    bars:       list[dict],
    indicators: IndicatorSnapshot,
    context:    dict,
    settings:   dict,
) -> SetupResult:
    """
    Detect the Opening Range Breakout setup.

    Args:
        bars:       Regular-session OHLCV bars, oldest→newest.
        indicators: IndicatorSnapshot from indicator_engine.
        context:    Analysis context dict. Expected keys:
                      ticker, current_price, vwap,
                      or_result, key_levels, structure, fib_result
        settings:   Full bot_settings dict.

    Returns:
        SetupResult — confirmed=True when setup is valid.
    """
    cfg           = settings.get("setups", {}).get(_SETUP_NAME, {})
    ticker        = context.get("ticker", "")
    current_price = float(context.get("current_price", 0))
    vwap          = context.get("vwap") or indicators.vwap
    or_result     = context.get("or_result")

    reasons:  list[str] = []
    warnings: list[str] = []

    # ── Basic data validation ─────────────────────────────────────────────────
    if not bars or current_price <= 0:
        return _reject("Insufficient data", ticker)

    if not or_result:
        return _reject("Opening range result unavailable", ticker)

    primary_or = getattr(or_result, "primary_or", None)
    if not primary_or:
        return _reject("Primary opening range unavailable", ticker)

    if not primary_or.is_complete:
        return _reject("Opening range is not complete yet", ticker)

    if not primary_or.high or not primary_or.low:
        return _reject("Opening range high/low unavailable", ticker)

    or_high = primary_or.high
    or_low  = primary_or.low

    # ── Price must be above OR high ───────────────────────────────────────────
    if current_price <= or_high:
        return _reject(
            f"Price {current_price:.4f} is not above OR high {or_high:.4f}",
            ticker,
        )

    reasons.append(f"Price is above OR high {or_high:.4f}")

    # ── Breakout state confirmation ───────────────────────────────────────────
    state = getattr(or_result, "state", "unknown")
    if state in ("breakout", "above"):
        reasons.append(f"Opening range state is bullish: {state}")
    elif state == "failed_breakout":
        return _reject("Opening range breakout already failed", ticker)
    else:
        warnings.append(f"Opening range state is {state}, not clean breakout")

    # ── Hold above OR high ────────────────────────────────────────────────────
    require_hold = bool(cfg.get("require_hold_above_range", True))
    min_hold_bars = 2
    bars_held = getattr(or_result, "bars_above_or", 0) or _count_hold_bars(bars, or_high)

    if require_hold and bars_held < min_hold_bars:
        return _reject(
            f"Only {bars_held} bar(s) held above OR high "
            f"(need {min_hold_bars})",
            ticker,
        )

    if bars_held > 0:
        reasons.append(f"Price held above OR high for {bars_held} bar(s)")

    # ── Volume confirmation ───────────────────────────────────────────────────
    require_volume = bool(cfg.get("require_volume_confirmation", True))
    breakout_volume_ratio = getattr(or_result, "breakout_volume_ratio", 0.0) or 0.0

    volume_ok = (
        breakout_volume_ratio >= 1.5
        or indicators.relative_volume >= 3.0
        or _latest_volume_expanded(bars)
    )

    if require_volume and not volume_ok:
        warnings.append("Opening range breakout volume is not clearly confirming")
    elif volume_ok:
        if breakout_volume_ratio > 0:
            reasons.append(f"Breakout volume ratio {breakout_volume_ratio:.2f}x")
        else:
            reasons.append(f"Volume confirming: RVOL {indicators.relative_volume:.1f}x")

    # ── Opening range size quality ────────────────────────────────────────────
    range_pct = getattr(primary_or, "range_size_pct", 0.0) or 0.0
    if range_pct > 12.0:
        warnings.append(f"Opening range is wide ({range_pct:.2f}%) — stop may be large")
    elif range_pct > 0:
        reasons.append(f"Opening range size is manageable ({range_pct:.2f}%)")

    # ── VWAP context ──────────────────────────────────────────────────────────
    vwap_bullish = bool(vwap and current_price > vwap and or_high >= vwap)
    if vwap_bullish:
        reasons.append("VWAP supports breakout — price and OR high above VWAP")
    elif vwap and current_price > vwap:
        reasons.append("Price is above VWAP")
    elif vwap:
        warnings.append("Price is not above VWAP — breakout quality reduced")

    # ── Candle strength confirmation ──────────────────────────────────────────
    latest_strength = _latest_candle_strength(bars)
    if latest_strength >= 0.50:
        reasons.append("Latest candle shows strong bullish body")
    elif latest_strength < 0:
        warnings.append("Latest candle is bearish despite OR breakout")

    # ── Structure confirmation ────────────────────────────────────────────────
    structure = context.get("structure")
    structure_bullish = False
    if structure:
        try:
            structure_bullish = structure.is_bullish()
        except AttributeError:
            structure_bullish = False

    if structure_bullish:
        reasons.append("Session structure is bullish")
    elif structure and getattr(structure, "trend_direction", "") == "bearish":
        warnings.append("Session structure is bearish — breakout may fail")

    # ── Momentum confirmation ─────────────────────────────────────────────────
    macd_bullish = bool(indicators.macd and indicators.macd.bullish)
    if macd_bullish:
        reasons.append("MACD bullish — momentum supports OR breakout")
    else:
        warnings.append("MACD is not yet bullish")

    ma_bullish = indicators.ma_trend == "bullish"
    if ma_bullish:
        reasons.append("Fast MA above slow MA — trend aligned")
    elif indicators.ma_trend == "bearish":
        warnings.append("MA trend is bearish — breakout is lower quality")

    if indicators.rsi_zone == "overbought":
        warnings.append("RSI overbought — chase risk elevated")
    elif indicators.rsi_zone == "neutral":
        reasons.append("RSI neutral — breakout not overheated")

    # ── Entry / stop / target ─────────────────────────────────────────────────
    entry_trigger = or_high

    hold_low  = _find_hold_low(bars, or_high)
    midpoint  = primary_or.midpoint() if hasattr(primary_or, "midpoint") else None
    stop_area = _choose_stop_area(
        current_price = current_price,
        or_high       = or_high,
        or_low        = or_low,
        midpoint      = midpoint,
        hold_low      = hold_low,
    )

    fib_result = context.get("fib_result")
    key_levels = context.get("key_levels")

    target_area = None
    if fib_result and getattr(fib_result, "target_extensions", None):
        target_area = (
            fib_result.target_extensions.get("1.272")
            or fib_result.target_extensions.get("1.2720")
            or fib_result.target_extensions.get("1.618")
        )

    if not target_area and key_levels and getattr(key_levels, "nearest_resistance", None):
        target_area = key_levels.nearest_resistance.price

    if not target_area:
        target_area = entry_trigger * 1.06

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _score(
        bars_held       = bars_held,
        volume_ok       = volume_ok,
        breakout_volume_ratio = breakout_volume_ratio,
        rvol            = indicators.relative_volume,
        range_pct       = range_pct,
        vwap_bullish    = vwap_bullish,
        candle_strength = latest_strength,
        structure_bullish = structure_bullish,
        macd_bullish    = macd_bullish,
        ma_bullish      = ma_bullish,
        rsi_zone        = indicators.rsi_zone,
    )

    confidence = _confidence_label(score)

    log.debug(
        "[%s] %s score=%.1f state=%s held=%d vol_ok=%s",
        _SETUP_NAME, ticker, score, state, bars_held, volume_ok,
    )

    return SetupResult(
        setup_name    = _SETUP_NAME,
        confirmed     = True,
        confidence    = confidence,
        score         = round(score, 2),
        entry_trigger = round(entry_trigger, 4),
        stop_area     = round(stop_area,     4),
        target_area   = round(target_area,   4),
        reasons       = reasons,
        warnings      = warnings,
    )


# ── Setup-specific helpers ────────────────────────────────────────────────────

def _count_hold_bars(bars: list[dict], level: float) -> int:
    """
    Count consecutive recent bars that closed above `level`.
    Stops counting at the first close below the level.
    """
    count = 0
    for bar in reversed(bars):
        if bar.get("c", 0) > level:
            count += 1
        else:
            break
    return count


def _latest_volume_expanded(bars: list[dict]) -> bool:
    """
    True when the latest bar volume is at least 1.5x the average
    of the prior five bars.
    """
    if len(bars) < 6:
        return False

    latest_vol = bars[-1].get("v", 0)
    prior      = bars[-6:-1]
    avg_prior  = sum(b.get("v", 0) for b in prior) / len(prior)

    if avg_prior <= 0:
        return False

    return latest_vol >= avg_prior * 1.5


def _latest_candle_strength(bars: list[dict]) -> float:
    """
    Return the latest candle body-to-range ratio.
    Positive = bullish candle, negative = bearish candle.
    """
    if not bars:
        return 0.0

    bar = bars[-1]
    high = bar.get("h", 0)
    low  = bar.get("l", 0)
    open_ = bar.get("o", 0)
    close = bar.get("c", 0)

    candle_range = high - low
    if candle_range <= 0:
        return 0.0

    return (close - open_) / candle_range


def _find_hold_low(bars: list[dict], level: float) -> Optional[float]:
    """
    Find the lowest low among recent bars that closed above the OR high.
    This becomes a tighter stop area when available.
    """
    above = [
        b for b in bars[-10:]
        if b.get("c", 0) > level and b.get("l", 0) > 0
    ]
    if not above:
        return None
    return min(b.get("l", 0) for b in above)


def _choose_stop_area(
    current_price: float,
    or_high:       float,
    or_low:        float,
    midpoint:      Optional[float],
    hold_low:      Optional[float],
) -> float:
    """
    Choose a practical stop area for an OR breakout.

    Priority:
      1. Recent hold low above/near OR high
      2. OR midpoint
      3. Slightly below OR high
      4. OR low as fallback
    """
    if hold_low and hold_low > 0 and hold_low < current_price:
        return hold_low

    if midpoint and midpoint > 0 and midpoint < current_price:
        return midpoint

    if or_high > 0 and or_high < current_price:
        return or_high * 0.985

    return or_low if or_low > 0 else current_price * 0.95


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    bars_held:        int,
    volume_ok:        bool,
    breakout_volume_ratio: float,
    rvol:             float,
    range_pct:        float,
    vwap_bullish:     bool,
    candle_strength:  float,
    structure_bullish:bool,
    macd_bullish:     bool,
    ma_bullish:       bool,
    rsi_zone:         str,
) -> float:
    """
    Score the Opening Range Breakout setup quality 0–100.

    Weights:
      Hold above OR high:     20 pts
      Volume / RVOL:          20 pts
      OR range quality:       10 pts
      VWAP context:           15 pts
      Candle strength:        10 pts
      Structure:              10 pts
      MACD / MA trend:        10 pts
      RSI zone:                5 pts
    """
    score = 0.0

    # Hold above OR high (20 pts)
    if bars_held >= 3:
        score += 20
    elif bars_held == 2:
        score += 15
    elif bars_held == 1:
        score += 10

    # Volume / RVOL (20 pts)
    if volume_ok:
        score += 8

    if breakout_volume_ratio >= 2.0:
        score += 7
    elif breakout_volume_ratio >= 1.5:
        score += 5

    if rvol >= 5.0:
        score += 5
    elif rvol >= 3.0:
        score += 4
    elif rvol >= 2.0:
        score += 2

    # OR range quality (10 pts)
    if 0 < range_pct <= 6.0:
        score += 10
    elif 6.0 < range_pct <= 10.0:
        score += 6
    elif range_pct > 12.0:
        score -= 5

    # VWAP context (15 pts)
    if vwap_bullish:
        score += 15

    # Candle strength (10 pts)
    if candle_strength >= 0.60:
        score += 10
    elif candle_strength >= 0.30:
        score += 6
    elif candle_strength < 0:
        score -= 5

    # Structure (10 pts)
    if structure_bullish:
        score += 10

    # MACD / MA trend (10 pts)
    if macd_bullish:
        score += 5
    if ma_bullish:
        score += 5

    # RSI zone (5 pts)
    if rsi_zone == "neutral":
        score += 5
    elif rsi_zone == "oversold":
        score += 2
    elif rsi_zone == "overbought":
        score -= 5

    return max(0.0, min(score, 100.0))


def _confidence_label(score: float) -> str:
    if score >= 88:
        return ConfidenceLabel.ELITE.value
    if score >= 78:
        return ConfidenceLabel.STRONG.value
    if score >= 68:
        return ConfidenceLabel.DECENT.value
    if score >= 58:
        return ConfidenceLabel.WEAK.value
    return ConfidenceLabel.REJECT.value


# ── Rejection helper ──────────────────────────────────────────────────────────

def _reject(reason: str, ticker: str = "") -> SetupResult:
    log.debug("[%s] %s rejected: %s", _SETUP_NAME, ticker, reason)
    return SetupResult(
        setup_name = _SETUP_NAME,
        confirmed  = False,
        confidence = ConfidenceLabel.REJECT.value,
        score      = 0.0,
        reasons    = [reason],
    )
