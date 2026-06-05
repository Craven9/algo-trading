"""
src/setups/bottom_base.py — Bottom Base setup detector
Detects when price forms a tight consolidation base near support,
then begins to break out with volume expansion.

The bottom base is a reversal / continuation setup:
  1. Price stops making lower lows
  2. Price consolidates in a tight range for several bars
  3. Volume contracts during the base
  4. The base forms near support, VWAP, or a Fibonacci level
  5. Price breaks above the base high
  6. Volume expands on the breakout attempt

This setup is strongest when:
  - Base range is tight
  - Base forms above VWAP or reclaims VWAP
  - Volume dries up during the base
  - Breakout volume expands clearly
  - Structure shifts from sideways to bullish
  - There is room to nearest resistance

Returns a SetupResult with:
  confirmed:     True when all conditions are met
  score:         0–100 based on base quality
  entry_trigger: Base high breakout level
  stop_area:     Below base low
  target_area:   Fibonacci extension or next resistance
  reasons:       Human-readable confirmation factors
  warnings:      Any concerns about the setup quality
"""

from __future__ import annotations

import logging
from typing import Optional

from models import ConfidenceLabel, IndicatorSnapshot, SetupResult, SetupName

log = logging.getLogger(__name__)

_SETUP_NAME = SetupName.BOTTOM_BASE.value


def detect(
    bars:       list[dict],
    indicators: IndicatorSnapshot,
    context:    dict,
    settings:   dict,
) -> SetupResult:
    """
    Detect the Bottom Base setup.

    Args:
        bars:       Regular-session OHLCV bars, oldest→newest.
        indicators: IndicatorSnapshot from indicator_engine.
        context:    Analysis context dict. Expected keys:
                      ticker, current_price, vwap,
                      key_levels, structure, fib_result
        settings:   Full bot_settings dict.

    Returns:
        SetupResult — confirmed=True when setup is valid.
    """
    cfg           = settings.get("setups", {}).get(_SETUP_NAME, {})
    ticker        = context.get("ticker", "")
    current_price = float(context.get("current_price", 0))
    vwap          = context.get("vwap") or indicators.vwap

    reasons:  list[str] = []
    warnings: list[str] = []

    # ── Basic data validation ─────────────────────────────────────────────────
    if not bars or current_price <= 0:
        return _reject("Insufficient data", ticker)

    min_base_bars = int(cfg.get("min_base_bars", 3))
    max_range_pct = float(cfg.get("max_base_range_percent", 5.0))

    if len(bars) < min_base_bars + 2:
        return _reject("Not enough bars to detect bottom base", ticker)

    # ── Identify recent base ──────────────────────────────────────────────────
    base_bars = bars[-min_base_bars:]
    base_high = max(b.get("h", 0) for b in base_bars)
    base_low  = min(b.get("l", 0) for b in base_bars if b.get("l", 0) > 0)

    if base_high <= 0 or base_low <= 0 or base_high <= base_low:
        return _reject("Invalid base high/low", ticker)

    base_range_pct = (base_high - base_low) / base_low * 100

    if base_range_pct > max_range_pct:
        return _reject(
            f"Base range {base_range_pct:.2f}% is wider than max {max_range_pct:.2f}%",
            ticker,
        )

    reasons.append(f"Tight base range: {base_range_pct:.2f}%")

    # ── Base should not still be making lower lows ────────────────────────────
    lower_lows = _recent_lower_lows(base_bars)
    if lower_lows:
        return _reject("Base is still making lower lows", ticker)

    reasons.append("Base stopped making lower lows")

    # ── Price must be breaking or pressing above base high ────────────────────
    breakout_confirmed = current_price > base_high
    pressing_high      = current_price >= base_high * 0.995

    if not breakout_confirmed and not pressing_high:
        return _reject(
            f"Price {current_price:.4f} is not breaking base high {base_high:.4f}",
            ticker,
        )

    if breakout_confirmed:
        reasons.append(f"Price broke above base high {base_high:.4f}")
    else:
        reasons.append(f"Price is pressing base high {base_high:.4f}")

    # ── Volume contraction / expansion ────────────────────────────────────────
    require_expansion = bool(cfg.get("require_volume_expansion", True))
    volume_contracted = _volume_contracted_into_base(bars, min_base_bars)
    volume_expanded   = _latest_volume_expanded(bars)

    if volume_contracted:
        reasons.append("Volume contracted during base")
    else:
        warnings.append("Base volume did not clearly contract")

    if require_expansion and not volume_expanded:
        warnings.append("Breakout volume has not expanded yet")
    elif volume_expanded:
        reasons.append("Volume expanded on base breakout")

    if indicators.relative_volume >= 3.0:
        reasons.append(f"Strong RVOL: {indicators.relative_volume:.1f}x")
    elif indicators.relative_volume < 2.0:
        warnings.append(f"Low RVOL: {indicators.relative_volume:.1f}x")

    # ── Support / level context ───────────────────────────────────────────────
    key_levels = context.get("key_levels")
    near_support = _base_near_support(key_levels, base_low, current_price)

    if near_support:
        reasons.append("Base formed near support")
    else:
        warnings.append("No clear support directly under base")

    # ── VWAP context ──────────────────────────────────────────────────────────
    above_vwap = bool(vwap and current_price >= vwap)
    base_above_vwap = bool(vwap and base_low >= vwap * 0.995)

    if base_above_vwap:
        reasons.append("Base formed above VWAP — bullish context")
    elif above_vwap:
        reasons.append("Price is above VWAP after base")
    elif vwap:
        warnings.append("Base is below VWAP — lower quality")

    # ── Fibonacci support context ─────────────────────────────────────────────
    fib_result = context.get("fib_result")
    near_fib = _base_near_fib(fib_result, base_low)

    if near_fib:
        reasons.append("Base formed near Fibonacci support")
    elif fib_result and getattr(fib_result, "block_trade", False):
        return _reject("Fibonacci engine is blocking long entry", ticker)

    # ── Structure confirmation ────────────────────────────────────────────────
    structure = context.get("structure")
    structure_bullish = False
    structure_sideways = False

    if structure:
        try:
            structure_bullish = structure.is_bullish()
        except AttributeError:
            structure_bullish = False
        structure_sideways = getattr(structure, "structure", "") in ("sideways", "forming", "mixed")

    if structure_bullish:
        reasons.append("Session structure is bullish")
    elif structure_sideways:
        reasons.append("Session structure is basing/sideways")
    elif structure and getattr(structure, "trend_direction", "") == "bearish":
        warnings.append("Session structure is bearish — base may fail")

    # ── Momentum confirmation ─────────────────────────────────────────────────
    macd_bullish = bool(indicators.macd and indicators.macd.bullish)
    macd_cross   = bool(indicators.macd and indicators.macd.bullish_crossover)

    if macd_cross:
        reasons.append("MACD bullish crossover supports base breakout")
    elif macd_bullish:
        reasons.append("MACD bullish — momentum improving")
    else:
        warnings.append("MACD is not yet bullish")

    ma_bullish = indicators.ma_trend == "bullish"
    if ma_bullish:
        reasons.append("MA trend is bullish")
    elif indicators.ma_trend == "bearish":
        warnings.append("MA trend is still bearish")

    # ── RSI zone ──────────────────────────────────────────────────────────────
    if indicators.rsi_zone == "neutral":
        reasons.append("RSI neutral — base is not overheated")
    elif indicators.rsi_zone == "oversold":
        reasons.append("RSI oversold/recovering — possible reversal base")
    elif indicators.rsi_zone == "overbought":
        warnings.append("RSI overbought — base breakout may be extended")

    # ── Entry / stop / target ─────────────────────────────────────────────────
    entry_trigger = base_high
    stop_area     = base_low * 0.995

    target_area = _choose_target_area(
        fib_result     = fib_result,
        key_levels     = key_levels,
        current_price  = current_price,
        entry_trigger  = entry_trigger,
    )

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _score(
        base_range_pct     = base_range_pct,
        max_range_pct      = max_range_pct,
        breakout_confirmed = breakout_confirmed,
        volume_contracted  = volume_contracted,
        volume_expanded    = volume_expanded,
        rvol               = indicators.relative_volume,
        near_support       = near_support,
        above_vwap         = above_vwap,
        base_above_vwap    = base_above_vwap,
        near_fib           = near_fib,
        structure_bullish  = structure_bullish,
        structure_sideways = structure_sideways,
        macd_bullish       = macd_bullish,
        macd_cross         = macd_cross,
        ma_bullish         = ma_bullish,
        rsi_zone           = indicators.rsi_zone,
    )

    confidence = _confidence_label(score)

    log.debug(
        "[%s] %s score=%.1f base_range=%.2f breakout=%s vol_exp=%s",
        _SETUP_NAME, ticker, score, base_range_pct, breakout_confirmed, volume_expanded,
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

def _recent_lower_lows(base_bars: list[dict]) -> bool:
    """
    True when the base is still making lower lows.
    A bottom base should stop pushing lower.
    """
    lows = [b.get("l", 0) for b in base_bars if b.get("l", 0) > 0]
    if len(lows) < 2:
        return False
    return all(lows[i] < lows[i - 1] for i in range(1, len(lows)))


def _volume_contracted_into_base(bars: list[dict], base_len: int) -> bool:
    """
    True when average volume during the base is lower than prior volume.
    """
    if len(bars) < base_len * 2:
        return False

    base = bars[-base_len:]
    prior = bars[-base_len * 2:-base_len]

    base_avg = sum(b.get("v", 0) for b in base) / len(base)
    prior_avg = sum(b.get("v", 0) for b in prior) / len(prior)

    if prior_avg <= 0:
        return False

    return base_avg <= prior_avg * 0.85


def _latest_volume_expanded(bars: list[dict]) -> bool:
    """
    True when the latest bar volume is stronger than recent average.
    """
    if len(bars) < 6:
        return False

    latest = bars[-1].get("v", 0)
    prior  = bars[-6:-1]
    avg    = sum(b.get("v", 0) for b in prior) / len(prior)

    if avg <= 0:
        return False

    return latest >= avg * 1.25


def _base_near_support(
    key_levels: object,
    base_low: float,
    current_price: float,
) -> bool:
    """
    True when the base low is near nearest support or any support level.
    """
    if not key_levels or base_low <= 0:
        return False

    nearest_support = getattr(key_levels, "nearest_support", None)
    if nearest_support and getattr(nearest_support, "price", 0):
        support = nearest_support.price
        return abs(base_low - support) / support * 100 <= 2.0

    try:
        supports = key_levels.support_levels()
    except AttributeError:
        supports = []

    for level in supports:
        support = getattr(level, "price", 0)
        if support and abs(base_low - support) / support * 100 <= 2.0:
            return True

    return False


def _base_near_fib(fib_result: object, base_low: float) -> bool:
    """
    True when the base formed near the nearest Fib retracement.
    """
    if not fib_result or base_low <= 0:
        return False

    nearest = getattr(fib_result, "nearest_retracement", None)
    if not nearest:
        return False

    fib_price = getattr(nearest, "price", 0)
    if not fib_price:
        return False

    return abs(base_low - fib_price) / fib_price * 100 <= 2.0


def _choose_target_area(
    fib_result: object,
    key_levels: object,
    current_price: float,
    entry_trigger: float,
) -> float:
    """
    Choose target from Fibonacci extensions, nearest resistance,
    then fallback percent target.
    """
    target = None

    if fib_result and getattr(fib_result, "target_extensions", None):
        target = (
            fib_result.target_extensions.get("1.272")
            or fib_result.target_extensions.get("1.2720")
            or fib_result.target_extensions.get("1.618")
            or fib_result.target_extensions.get("2.0")
            or fib_result.target_extensions.get("2.000")
        )

    if not target and key_levels and getattr(key_levels, "nearest_resistance", None):
        target = key_levels.nearest_resistance.price

    if target and target > current_price:
        return target

    return entry_trigger * 1.06


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    base_range_pct:     float,
    max_range_pct:      float,
    breakout_confirmed: bool,
    volume_contracted:  bool,
    volume_expanded:    bool,
    rvol:               float,
    near_support:       bool,
    above_vwap:         bool,
    base_above_vwap:    bool,
    near_fib:           bool,
    structure_bullish:  bool,
    structure_sideways: bool,
    macd_bullish:       bool,
    macd_cross:         bool,
    ma_bullish:         bool,
    rsi_zone:           str,
) -> float:
    """
    Score the Bottom Base setup quality 0–100.

    Weights:
      Base tightness:          20 pts
      Breakout confirmation:   15 pts
      Volume behavior:         20 pts
      Support / Fib context:   15 pts
      VWAP context:            10 pts
      Structure:                8 pts
      Momentum:                 7 pts
      RSI zone:                 5 pts
    """
    score = 0.0

    # Base tightness (20 pts)
    if base_range_pct <= max_range_pct * 0.5:
        score += 20
    elif base_range_pct <= max_range_pct:
        score += 14

    # Breakout confirmation (15 pts)
    if breakout_confirmed:
        score += 15
    else:
        score += 8

    # Volume behavior (20 pts)
    if volume_contracted:
        score += 8
    if volume_expanded:
        score += 7
    if rvol >= 5.0:
        score += 5
    elif rvol >= 3.0:
        score += 4
    elif rvol >= 2.0:
        score += 2

    # Support / Fib context (15 pts)
    if near_support:
        score += 9
    if near_fib:
        score += 6

    # VWAP context (10 pts)
    if base_above_vwap:
        score += 10
    elif above_vwap:
        score += 7

    # Structure (8 pts)
    if structure_bullish:
        score += 8
    elif structure_sideways:
        score += 5

    # Momentum (7 pts)
    if macd_cross:
        score += 3
    elif macd_bullish:
        score += 2

    if ma_bullish:
        score += 4

    # RSI zone (5 pts)
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
