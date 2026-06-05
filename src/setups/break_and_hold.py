"""
src/setups/break_and_hold.py — Break and Hold setup detector
Detects when price breaks above a key level and holds above it
for a minimum number of bars with volume confirmation.

The break and hold is one of the cleanest momentum setups:
  1. Price approaches a key level (OR high, swing high, VWAP, etc.)
  2. Price closes above the level (the break)
  3. Price holds above the level for N bars without closing back below
  4. Volume confirms the breakout (not a low-volume fake)

This setup is strongest when:
  - The level has been tested multiple times before breaking
  - Volume expands on the break bar
  - Price forms a higher low above the level after breaking
  - VWAP is below the breakout level (bullish context)

Returns a SetupResult with:
  confirmed:     True when all conditions are met
  score:         0–100 based on quality factors
  entry_trigger: The breakout level (ideal entry on a pullback to level)
  stop_area:     Below the hold low (the last higher low above the level)
  target_area:   Fibonacci 1.272 extension or next resistance
  reasons:       Human-readable confirmation factors
  warnings:      Any concerns about the setup quality
"""

from __future__ import annotations

import logging
from typing import Optional

from models import ConfidenceLabel, IndicatorSnapshot, SetupResult, SetupName

log = logging.getLogger(__name__)

_SETUP_NAME = SetupName.BREAK_AND_HOLD.value


def detect(
    bars:       list[dict],
    indicators: IndicatorSnapshot,
    context:    dict,
    settings:   dict,
) -> SetupResult:
    """
    Detect the Break and Hold setup.

    Args:
        bars:       Regular-session OHLCV bars, oldest→newest.
        indicators: IndicatorSnapshot from indicator_engine.
        context:    Analysis context dict. Expected keys:
                      ticker, current_price, vwap,
                      key_levels (KeyLevelResult),
                      or_result  (OpeningRangeResult),
                      fib_result (FibonacciResult)
        settings:   Full bot_settings dict.

    Returns:
        SetupResult — confirmed=True when setup is valid.
    """
    cfg          = settings.get("setups", {}).get(_SETUP_NAME, {})
    ticker       = context.get("ticker", "")
    current_price= float(context.get("current_price", 0))
    vwap         = context.get("vwap") or indicators.vwap

    reasons:  list[str] = []
    warnings: list[str] = []

    # ── Basic data validation ─────────────────────────────────────────────────
    if not bars or current_price <= 0:
        return _reject("Insufficient data", ticker)

    min_bars = int(cfg.get("min_hold_bars", 2))

    # ── Identify the breakout level ───────────────────────────────────────────
    breakout_level = _find_breakout_level(context, indicators, current_price)
    if breakout_level is None or breakout_level <= 0:
        return _reject("No clear breakout level identified", ticker)

    # ── Check price is above the level ────────────────────────────────────────
    if current_price <= breakout_level:
        return _reject(
            f"Price {current_price:.4f} not above breakout level {breakout_level:.4f}",
            ticker,
        )

    # ── Check hold bars — count consecutive closes above level ────────────────
    hold_bars   = _count_hold_bars(bars, breakout_level)
    if hold_bars < min_bars:
        return _reject(
            f"Only {hold_bars} bar(s) held above {breakout_level:.4f} "
            f"(need {min_bars})",
            ticker,
        )
    reasons.append(f"Price held above {breakout_level:.4f} for {hold_bars} bar(s)")

    # ── Volume confirmation ───────────────────────────────────────────────────
    req_vol  = bool(cfg.get("require_volume_confirmation", True))
    vol_ok   = indicators.relative_volume >= 2.0

    if req_vol and not vol_ok:
        warnings.append(
            f"Low RVOL ({indicators.relative_volume:.1f}x) — "
            f"breakout volume not confirming"
        )
    elif vol_ok:
        reasons.append(f"Volume confirming: RVOL {indicators.relative_volume:.1f}x")

    # ── VWAP context ──────────────────────────────────────────────────────────
    vwap_bullish = vwap is not None and breakout_level > vwap
    if vwap_bullish:
        reasons.append("Breakout level is above VWAP — bullish context")
    elif vwap and current_price < vwap:
        warnings.append("Price is below VWAP — breakout quality reduced")

    # ── Higher low above the level ────────────────────────────────────────────
    higher_low_above = _detect_higher_low_above_level(bars, breakout_level)
    if higher_low_above:
        reasons.append("Higher low formed above breakout level — strong hold")

    # ── MA trend confirmation ─────────────────────────────────────────────────
    if indicators.ma_trend == "bullish":
        reasons.append("Fast MA above slow MA — trend aligned")
    elif indicators.ma_trend == "bearish":
        warnings.append("MA trend is bearish — counter-trend breakout")

    # ── MACD confirmation ─────────────────────────────────────────────────────
    if indicators.macd and indicators.macd.bullish:
        reasons.append("MACD bullish — momentum supporting breakout")

    # ── Entry / stop / target ─────────────────────────────────────────────────
    entry_trigger = breakout_level          # ideal entry on retest of level
    hold_low      = _find_hold_low(bars, breakout_level)
    stop_area     = hold_low if hold_low else breakout_level * 0.97

    # Target: use Fibonacci 1.272 if available, else 5% above entry
    fib_result = context.get("fib_result")
    if fib_result and fib_result.fib_target_1:
        target_area = fib_result.fib_target_1
    else:
        target_area = entry_trigger * 1.05

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _score(
        hold_bars      = hold_bars,
        vol_ok         = vol_ok,
        vwap_bullish   = vwap_bullish,
        higher_low     = higher_low_above,
        ma_bullish     = indicators.ma_trend == "bullish",
        macd_bullish   = indicators.macd.bullish if indicators.macd else False,
        rsi_zone       = indicators.rsi_zone,
        rvol           = indicators.relative_volume,
    )

    confidence = _confidence_label(score)

    log.debug(
        "[%s] %s score=%.1f hold=%d vol_ok=%s vwap_bull=%s",
        _SETUP_NAME, ticker, score, hold_bars, vol_ok, vwap_bullish,
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

def _find_breakout_level(
    context:       dict,
    indicators:    IndicatorSnapshot,
    current_price: float,
) -> Optional[float]:
    """
    Find the most relevant key level that price has recently broken above.
    Priority: OR high > VWAP > nearest resistance from key_levels.
    """
    # Opening range high
    or_result = context.get("or_result")
    if or_result and or_result.primary_or.high:
        or_high = or_result.primary_or.high
        if current_price > or_high:
            return or_high

    # VWAP as a breakout level
    vwap = context.get("vwap") or indicators.vwap
    if vwap and current_price > vwap:
        return vwap

    # Nearest resistance that has been cleared
    key_levels = context.get("key_levels")
    if key_levels and key_levels.nearest_resistance:
        res_price = key_levels.nearest_resistance.price
        if current_price > res_price:
            return res_price

    return None


def _count_hold_bars(bars: list[dict], level: float) -> int:
    """
    Count how many of the most recent consecutive bars closed above `level`.
    Stops counting at the first bar that closes below.
    """
    count = 0
    for bar in reversed(bars):
        if bar.get("c", 0) > level:
            count += 1
        else:
            break
    return count


def _find_hold_low(bars: list[dict], level: float) -> Optional[float]:
    """
    Find the lowest low among bars that closed above the level.
    This becomes the stop area.
    """
    hold_lows = [
        bar.get("l", 0)
        for bar in bars
        if bar.get("c", 0) > level and bar.get("l", 0) > 0
    ]
    return min(hold_lows) if hold_lows else None


def _detect_higher_low_above_level(bars: list[dict], level: float) -> bool:
    """
    True when there are at least 2 bars above the level and
    the most recent bar's low is higher than the previous bar's low.
    Indicates a higher low is forming above the breakout level.
    """
    above = [b for b in bars if b.get("c", 0) > level]
    if len(above) < 2:
        return False
    return above[-1].get("l", 0) > above[-2].get("l", 0)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    hold_bars:   int,
    vol_ok:      bool,
    vwap_bullish:bool,
    higher_low:  bool,
    ma_bullish:  bool,
    macd_bullish:bool,
    rsi_zone:    str,
    rvol:        float,
) -> float:
    """
    Score the Break and Hold setup quality 0–100.

    Weights:
      Hold duration:   25 pts
      Volume confirm:  20 pts
      VWAP context:    15 pts
      Higher low:      15 pts
      MA trend:        10 pts
      MACD:            10 pts
      RSI zone:         5 pts
    """
    score = 0.0

    # Hold duration (25 pts): 2 bars=15, 3=20, 4+=25
    hold_score = min((hold_bars / 4) * 25, 25)
    score += hold_score

    # Volume (20 pts)
    if vol_ok:
        rvol_bonus = min((rvol / 5.0) * 20, 20)
        score += rvol_bonus
    else:
        score += 5   # some credit even without full volume

    # VWAP context (15 pts)
    if vwap_bullish:
        score += 15

    # Higher low (15 pts)
    if higher_low:
        score += 15

    # MA trend (10 pts)
    if ma_bullish:
        score += 10

    # MACD (10 pts)
    if macd_bullish:
        score += 10

    # RSI zone (5 pts): neutral is ideal, oversold is ok, overbought penalizes
    if rsi_zone == "neutral":
        score += 5
    elif rsi_zone == "oversold":
        score += 3
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
