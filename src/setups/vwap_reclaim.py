"""
src/setups/vwap_reclaim.py — VWAP Reclaim setup detector
Detects when price crosses back above VWAP after being below it,
then holds above VWAP with confirmation.

The VWAP reclaim is one of the most reliable intraday reversal setups:
  1. Price trades below VWAP
  2. Price crosses back above VWAP
  3. Price closes above VWAP, not just wicks above it
  4. Price holds above VWAP for confirmation
  5. A higher low forms above or near VWAP
  6. Volume and momentum confirm the reclaim

This setup is strongest when:
  - Reclaim happens with strong RVOL
  - Price holds above VWAP for multiple bars
  - A higher low forms after reclaim
  - MACD turns bullish or crosses up
  - RSI is neutral, not extremely overbought
  - Market structure is bullish or improving

Returns a SetupResult with:
  confirmed:     True when all conditions are met
  score:         0–100 based on reclaim quality
  entry_trigger: VWAP reclaim / current price area
  stop_area:     Below VWAP or below the reclaim higher low
  target_area:   Fibonacci 1.272 extension or next resistance
  reasons:       Human-readable confirmation factors
  warnings:      Any concerns about the setup quality
"""

from __future__ import annotations

import logging
from typing import Optional

from models import ConfidenceLabel, IndicatorSnapshot, SetupResult, SetupName

log = logging.getLogger(__name__)

_SETUP_NAME = SetupName.VWAP_RECLAIM.value


def detect(
    bars:       list[dict],
    indicators: IndicatorSnapshot,
    context:    dict,
    settings:   dict,
) -> SetupResult:
    """
    Detect the VWAP Reclaim setup.

    Args:
        bars:       Regular-session OHLCV bars, oldest→newest.
        indicators: IndicatorSnapshot from indicator_engine.
        context:    Analysis context dict. Expected keys:
                      ticker, current_price, vwap,
                      key_levels, or_result, structure, fib_result
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

    if not vwap or vwap <= 0:
        return _reject("VWAP unavailable — cannot detect VWAP reclaim", ticker)

    if len(bars) < 5:
        return _reject("Not enough bars for VWAP reclaim detection", ticker)

    # ── Price must be above VWAP now ──────────────────────────────────────────
    if current_price <= vwap:
        return _reject(
            f"Price {current_price:.4f} is not above VWAP {vwap:.4f}",
            ticker,
        )

    reasons.append(f"Price is above VWAP {vwap:.4f}")

    # ── Detect actual reclaim event ───────────────────────────────────────────
    reclaim_idx = _find_reclaim_bar(bars, vwap)
    if reclaim_idx is None:
        return _reject("No clear VWAP reclaim bar found", ticker)

    reclaim_bar = bars[reclaim_idx]
    if reclaim_bar.get("c", 0) <= vwap:
        return _reject("VWAP was wicked above but not closed above", ticker)

    reasons.append("Price closed back above VWAP — true reclaim")

    # ── Hold above VWAP ───────────────────────────────────────────────────────
    bars_held = _count_hold_bars_above_vwap(bars, vwap)
    if bars_held < 1:
        return _reject("Price has not held above VWAP after reclaim", ticker)

    reasons.append(f"Price held above VWAP for {bars_held} bar(s)")

    # ── Higher low requirement ────────────────────────────────────────────────
    require_higher_low = bool(cfg.get("require_higher_low", True))
    higher_low = _detect_higher_low_after_reclaim(bars, reclaim_idx, vwap)

    if require_higher_low and not higher_low:
        return _reject("No higher low formed after VWAP reclaim", ticker)

    if higher_low:
        reasons.append("Higher low formed after VWAP reclaim")

    # ── Volume confirmation ───────────────────────────────────────────────────
    require_volume = bool(cfg.get("require_volume_on_reclaim", True))
    volume_ok      = _reclaim_volume_confirming(bars, reclaim_idx)

    if require_volume and not volume_ok:
        warnings.append("VWAP reclaim volume is not clearly confirming")
    elif volume_ok:
        reasons.append("Volume expanded on/after VWAP reclaim")

    if indicators.relative_volume >= 3.0:
        reasons.append(f"Strong RVOL: {indicators.relative_volume:.1f}x")
    elif indicators.relative_volume < 2.0:
        warnings.append(f"Low RVOL: {indicators.relative_volume:.1f}x")

    # ── Distance from VWAP check ──────────────────────────────────────────────
    max_distance = float(cfg.get("max_distance_from_vwap_percent", 3.0))
    distance_pct = abs((current_price - vwap) / vwap * 100)

    if distance_pct > max_distance:
        warnings.append(
            f"Price is {distance_pct:.2f}% above VWAP — may be extended"
        )
    else:
        reasons.append(f"Price is close to VWAP ({distance_pct:.2f}% away)")

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
        warnings.append("Session structure is still bearish")

    # ── MACD confirmation ─────────────────────────────────────────────────────
    macd_bullish = bool(indicators.macd and indicators.macd.bullish)
    macd_cross   = bool(indicators.macd and indicators.macd.bullish_crossover)

    if macd_cross:
        reasons.append("MACD bullish crossover after reclaim")
    elif macd_bullish:
        reasons.append("MACD bullish — momentum supports reclaim")
    else:
        warnings.append("MACD is not yet bullish")

    # ── MA trend confirmation ─────────────────────────────────────────────────
    ma_bullish = indicators.ma_trend == "bullish"
    if ma_bullish:
        reasons.append("Fast MA above slow MA — trend improving")
    elif indicators.ma_trend == "bearish":
        warnings.append("MA trend still bearish")

    # ── RSI zone ──────────────────────────────────────────────────────────────
    if indicators.rsi_zone == "neutral":
        reasons.append("RSI neutral — reclaim not overheated")
    elif indicators.rsi_zone == "overbought":
        warnings.append("RSI overbought — chase risk elevated")
    elif indicators.rsi_zone == "oversold":
        reasons.append("RSI recovering from oversold")

    # ── Entry / stop / target ─────────────────────────────────────────────────
    entry_trigger = current_price

    reclaim_low = _find_reclaim_low(bars, reclaim_idx)
    stop_area   = reclaim_low if reclaim_low and reclaim_low < current_price else vwap * 0.98

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
        target_area = entry_trigger * 1.05

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _score(
        bars_held       = bars_held,
        higher_low      = higher_low,
        volume_ok       = volume_ok,
        rvol            = indicators.relative_volume,
        distance_pct    = distance_pct,
        max_distance    = max_distance,
        macd_bullish    = macd_bullish,
        macd_cross      = macd_cross,
        ma_bullish      = ma_bullish,
        structure_bullish = structure_bullish,
        rsi_zone        = indicators.rsi_zone,
    )

    confidence = _confidence_label(score)

    log.debug(
        "[%s] %s score=%.1f held=%d higher_low=%s vol_ok=%s dist=%.2f%%",
        _SETUP_NAME, ticker, score, bars_held, higher_low, volume_ok, distance_pct,
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

def _find_reclaim_bar(bars: list[dict], vwap: float) -> Optional[int]:
    """
    Find the most recent bar where price reclaimed VWAP.

    Reclaim means:
      - Previous close was below VWAP
      - Current close is above VWAP
    """
    if len(bars) < 2:
        return None

    for i in range(len(bars) - 1, 0, -1):
        prev_close = bars[i - 1].get("c", 0)
        close      = bars[i].get("c", 0)

        if prev_close < vwap and close > vwap:
            return i

    return None


def _count_hold_bars_above_vwap(bars: list[dict], vwap: float) -> int:
    """
    Count consecutive recent bars that closed above VWAP.
    Stops counting at the first close below VWAP.
    """
    count = 0
    for bar in reversed(bars):
        if bar.get("c", 0) > vwap:
            count += 1
        else:
            break
    return count


def _detect_higher_low_after_reclaim(
    bars: list[dict],
    reclaim_idx: int,
    vwap: float,
) -> bool:
    """
    True when price forms a higher low after reclaiming VWAP.

    A valid higher low means:
      - At least two bars exist after/including reclaim
      - Most recent low is higher than prior low
      - Most recent low stays near or above VWAP
    """
    post = bars[reclaim_idx:]
    if len(post) < 2:
        return False

    prev_low   = post[-2].get("l", 0)
    latest_low = post[-1].get("l", 0)

    if latest_low <= 0 or prev_low <= 0:
        return False

    near_vwap = latest_low >= vwap * 0.995
    return latest_low > prev_low and near_vwap


def _reclaim_volume_confirming(bars: list[dict], reclaim_idx: int) -> bool:
    """
    True when reclaim volume is stronger than recent average volume.
    Uses the reclaim bar and the bar after it if available.
    """
    if reclaim_idx < 0 or reclaim_idx >= len(bars):
        return False

    lookback_start = max(0, reclaim_idx - 5)
    prior = bars[lookback_start:reclaim_idx]

    if not prior:
        return False

    avg_prior_vol = sum(b.get("v", 0) for b in prior) / len(prior)
    if avg_prior_vol <= 0:
        return False

    reclaim_vol = bars[reclaim_idx].get("v", 0)

    # If there is a follow-through bar, include it as confirmation too
    follow_vol = 0
    if reclaim_idx + 1 < len(bars):
        follow_vol = bars[reclaim_idx + 1].get("v", 0)

    best_vol = max(reclaim_vol, follow_vol)
    return best_vol >= avg_prior_vol * 1.2


def _find_reclaim_low(bars: list[dict], reclaim_idx: int) -> Optional[float]:
    """
    Find the lowest low from the reclaim bar onward.
    This becomes the stop area.
    """
    post = bars[reclaim_idx:]
    lows = [b.get("l", 0) for b in post if b.get("l", 0) > 0]
    return min(lows) if lows else None


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    bars_held:        int,
    higher_low:       bool,
    volume_ok:        bool,
    rvol:             float,
    distance_pct:     float,
    max_distance:     float,
    macd_bullish:     bool,
    macd_cross:       bool,
    ma_bullish:       bool,
    structure_bullish:bool,
    rsi_zone:         str,
) -> float:
    """
    Score the VWAP Reclaim setup quality 0–100.

    Weights:
      Hold above VWAP:       20 pts
      Higher low:            20 pts
      Volume / RVOL:         20 pts
      VWAP distance:         10 pts
      MACD:                  10 pts
      MA trend:               8 pts
      Structure:              7 pts
      RSI zone:               5 pts
    """
    score = 0.0

    # Hold above VWAP (20 pts): 1 bar=10, 2=15, 3+=20
    if bars_held >= 3:
        score += 20
    elif bars_held == 2:
        score += 15
    elif bars_held == 1:
        score += 10

    # Higher low (20 pts)
    if higher_low:
        score += 20

    # Volume / RVOL (20 pts)
    if volume_ok:
        score += 10
    if rvol >= 5.0:
        score += 10
    elif rvol >= 3.0:
        score += 8
    elif rvol >= 2.0:
        score += 5

    # VWAP distance (10 pts)
    if distance_pct <= max_distance * 0.5:
        score += 10
    elif distance_pct <= max_distance:
        score += 6
    else:
        score -= 5

    # MACD (10 pts)
    if macd_cross:
        score += 10
    elif macd_bullish:
        score += 7

    # MA trend (8 pts)
    if ma_bullish:
        score += 8

    # Structure (7 pts)
    if structure_bullish:
        score += 7

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
