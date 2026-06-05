"""
src/setups/liquidity_sweep_reclaim.py — Liquidity Sweep Reclaim setup detector
Detects the specific combination of a liquidity sweep below a key level
followed by a clean reclaim and continuation confirmation.

The liquidity sweep reclaim is one of the strongest reversal setups:
  1. Price briefly pierces below a key support level
  2. Stops/liquidity are swept below that level
  3. Price quickly reclaims the level
  4. Reclaim holds with a higher low
  5. Volume confirms buyers stepped in
  6. VWAP, structure, and momentum support continuation

This setup is strongest when:
  - The swept level is VWAP, OR low, premarket low, or a strong swing low
  - Reclaim happens within a few bars
  - Reclaim volume is strong versus sweep volume
  - A higher low forms after reclaim
  - Price is back above VWAP or reclaiming VWAP
  - Session structure shifts from bearish/mixed to bullish

Returns a SetupResult with:
  confirmed:     True when sweep + reclaim conditions are met
  score:         0–100 based on sweep/reclaim quality
  entry_trigger: Reclaimed level or current price area
  stop_area:     Below sweep low
  target_area:   Fibonacci extension or next resistance
  reasons:       Human-readable confirmation factors
  warnings:      Any concerns about the setup quality
"""

from __future__ import annotations

import logging
from typing import Optional

from models import ConfidenceLabel, IndicatorSnapshot, SetupResult, SetupName

log = logging.getLogger(__name__)

_SETUP_NAME = SetupName.LIQUIDITY_SWEEP_RECLAIM.value


def detect(
    bars:       list[dict],
    indicators: IndicatorSnapshot,
    context:    dict,
    settings:   dict,
) -> SetupResult:
    """
    Detect the Liquidity Sweep Reclaim setup.

    Args:
        bars:       Regular-session OHLCV bars, oldest→newest.
        indicators: IndicatorSnapshot from indicator_engine.
        context:    Analysis context dict. Expected keys:
                      ticker, current_price, vwap,
                      sweep_result, key_levels, structure, fib_result
        settings:   Full bot_settings dict.

    Returns:
        SetupResult — confirmed=True when setup is valid.
    """
    cfg           = settings.get("setups", {}).get(_SETUP_NAME, {})
    ticker        = context.get("ticker", "")
    current_price = float(context.get("current_price", 0))
    vwap          = context.get("vwap") or indicators.vwap
    sweep_result  = context.get("sweep_result")

    reasons:  list[str] = []
    warnings: list[str] = []

    # ── Basic data validation ─────────────────────────────────────────────────
    if not bars or current_price <= 0:
        return _reject("Insufficient data", ticker)

    if not sweep_result:
        return _reject("Liquidity sweep result unavailable", ticker)

    if not getattr(sweep_result, "any_confirmed", False):
        return _reject("No confirmed liquidity sweep found", ticker)

    best_sweep = getattr(sweep_result, "best_sweep", None)
    if not best_sweep:
        return _reject("No best confirmed sweep available", ticker)

    if not getattr(best_sweep, "confirmed", False):
        return _reject("Best sweep is not confirmed", ticker)

    level_price = float(getattr(best_sweep, "level_price", 0) or 0)
    sweep_low   = float(getattr(best_sweep, "sweep_low", 0) or 0)
    level_type  = getattr(best_sweep, "level_type", "")

    if level_price <= 0:
        return _reject("Swept level price unavailable", ticker)

    # ── Price must be back above swept level ──────────────────────────────────
    if current_price <= level_price:
        return _reject(
            f"Price {current_price:.4f} has not reclaimed swept level {level_price:.4f}",
            ticker,
        )

    reasons.append(f"Price reclaimed swept {level_type or 'key'} level {level_price:.4f}")

    # ── Reclaim speed check ───────────────────────────────────────────────────
    max_reclaim_bars = int(cfg.get("require_reclaim_within_bars", 3))
    bars_to_reclaim  = int(getattr(best_sweep, "bars_to_reclaim", 0) or 0)

    if bars_to_reclaim > max_reclaim_bars:
        warnings.append(
            f"Reclaim took {bars_to_reclaim} bars "
            f"(preferred <= {max_reclaim_bars})"
        )
    else:
        reasons.append(f"Fast reclaim: {bars_to_reclaim} bar(s)")

    # ── Higher low requirement ────────────────────────────────────────────────
    require_higher_low = bool(cfg.get("require_higher_low_after_reclaim", True))
    higher_low = bool(getattr(best_sweep, "higher_low", False))

    if require_higher_low and not higher_low:
        return _reject("No higher low after liquidity sweep reclaim", ticker)

    if higher_low:
        reasons.append("Higher low formed after reclaim")

    # ── Sweep quality score ───────────────────────────────────────────────────
    sweep_quality = float(getattr(best_sweep, "quality_score", 0.0) or 0.0)
    quality_label = getattr(best_sweep, "quality_label", "none")

    if sweep_quality >= 80:
        reasons.append(f"High quality sweep score: {sweep_quality:.1f}")
    elif sweep_quality >= 60:
        reasons.append(f"Moderate quality sweep score: {sweep_quality:.1f}")
    else:
        warnings.append(f"Sweep quality is low: {sweep_quality:.1f}")

    # ── Reclaim volume confirmation ───────────────────────────────────────────
    reclaim_volume_ratio = float(getattr(best_sweep, "reclaim_volume_ratio", 0.0) or 0.0)
    volume_ok = reclaim_volume_ratio >= 0.8 or indicators.relative_volume >= 3.0

    if volume_ok:
        if reclaim_volume_ratio > 0:
            reasons.append(f"Reclaim volume ratio {reclaim_volume_ratio:.2f}x")
        else:
            reasons.append(f"Strong RVOL: {indicators.relative_volume:.1f}x")
    else:
        warnings.append("Reclaim volume is not clearly confirming")

    # ── VWAP context ──────────────────────────────────────────────────────────
    above_vwap = bool(vwap and current_price > vwap)
    vwap_swept = bool(getattr(sweep_result, "vwap_swept", False))

    if vwap_swept:
        reasons.append("VWAP was swept and reclaimed — strong reversal signal")
    elif above_vwap:
        reasons.append("Price is above VWAP after reclaim")
    elif vwap:
        warnings.append("Price is still below VWAP after reclaim")

    # ── Level importance ──────────────────────────────────────────────────────
    important_level = _is_important_level(level_type)
    if important_level:
        reasons.append(f"Sweep occurred at important level: {level_type}")
    else:
        warnings.append(f"Sweep level type is less important: {level_type or 'unknown'}")

    # ── Momentum confirmation ─────────────────────────────────────────────────
    macd_bullish = bool(indicators.macd and indicators.macd.bullish)
    macd_cross   = bool(indicators.macd and indicators.macd.bullish_crossover)

    if macd_cross:
        reasons.append("MACD bullish crossover confirms reclaim")
    elif macd_bullish:
        reasons.append("MACD bullish — momentum supports reclaim")
    else:
        warnings.append("MACD is not yet bullish")

    ma_bullish = indicators.ma_trend == "bullish"
    if ma_bullish:
        reasons.append("MA trend is bullish")
    elif indicators.ma_trend == "bearish":
        warnings.append("MA trend still bearish")

    # ── Structure confirmation ────────────────────────────────────────────────
    structure = context.get("structure")
    structure_bullish = False
    if structure:
        try:
            structure_bullish = structure.is_bullish()
        except AttributeError:
            structure_bullish = False

    if structure_bullish:
        reasons.append("Session structure is bullish after reclaim")
    elif structure and getattr(structure, "trend_direction", "") == "bearish":
        warnings.append("Session structure is still bearish")

    # ── RSI zone ──────────────────────────────────────────────────────────────
    if indicators.rsi_zone == "neutral":
        reasons.append("RSI neutral — reversal not overheated")
    elif indicators.rsi_zone == "oversold":
        reasons.append("RSI recovering from oversold")
    elif indicators.rsi_zone == "overbought":
        warnings.append("RSI overbought — chase risk elevated")

    # ── Entry / stop / target ─────────────────────────────────────────────────
    entry_trigger = current_price

    if sweep_low and sweep_low > 0:
        stop_area = sweep_low * 0.995
    else:
        stop_area = level_price * 0.97

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
        sweep_quality      = sweep_quality,
        bars_to_reclaim    = bars_to_reclaim,
        max_reclaim_bars   = max_reclaim_bars,
        higher_low         = higher_low,
        volume_ok          = volume_ok,
        reclaim_volume_ratio = reclaim_volume_ratio,
        rvol               = indicators.relative_volume,
        vwap_swept         = vwap_swept,
        above_vwap         = above_vwap,
        important_level    = important_level,
        macd_bullish       = macd_bullish,
        macd_cross         = macd_cross,
        ma_bullish         = ma_bullish,
        structure_bullish  = structure_bullish,
        rsi_zone           = indicators.rsi_zone,
    )

    confidence = _confidence_label(score)

    log.debug(
        "[%s] %s score=%.1f level=%s quality=%.1f reclaim_bars=%d",
        _SETUP_NAME, ticker, score, level_type, sweep_quality, bars_to_reclaim,
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

def _is_important_level(level_type: str) -> bool:
    """
    True when the swept level is one of the strongest reversal levels.
    """
    lt = (level_type or "").lower()
    return (
        lt == "vwap"
        or lt == "or_low"
        or lt == "pm_low"
        or lt == "premarket_low"
        or "swing_low" in lt
        or "prior_day_low" in lt
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    sweep_quality:       float,
    bars_to_reclaim:     int,
    max_reclaim_bars:    int,
    higher_low:          bool,
    volume_ok:           bool,
    reclaim_volume_ratio:float,
    rvol:                float,
    vwap_swept:          bool,
    above_vwap:          bool,
    important_level:     bool,
    macd_bullish:        bool,
    macd_cross:          bool,
    ma_bullish:          bool,
    structure_bullish:   bool,
    rsi_zone:            str,
) -> float:
    """
    Score the Liquidity Sweep Reclaim setup quality 0–100.

    Weights:
      Sweep quality:          25 pts
      Reclaim speed:          15 pts
      Higher low:             15 pts
      Volume / RVOL:          15 pts
      VWAP context:           10 pts
      Level importance:        8 pts
      Momentum / structure:    7 pts
      RSI zone:                5 pts
    """
    score = 0.0

    # Sweep quality (25 pts)
    score += min(sweep_quality / 100.0 * 25, 25)

    # Reclaim speed (15 pts)
    if bars_to_reclaim <= 1:
        score += 15
    elif bars_to_reclaim <= max_reclaim_bars:
        score += 12
    elif bars_to_reclaim <= max_reclaim_bars + 2:
        score += 6

    # Higher low (15 pts)
    if higher_low:
        score += 15

    # Volume / RVOL (15 pts)
    if volume_ok:
        score += 6

    if reclaim_volume_ratio >= 1.2:
        score += 5
    elif reclaim_volume_ratio >= 0.8:
        score += 3

    if rvol >= 5.0:
        score += 4
    elif rvol >= 3.0:
        score += 3
    elif rvol >= 2.0:
        score += 1

    # VWAP context (10 pts)
    if vwap_swept:
        score += 10
    elif above_vwap:
        score += 7

    # Level importance (8 pts)
    if important_level:
        score += 8

    # Momentum / structure (7 pts)
    if macd_cross:
        score += 3
    elif macd_bullish:
        score += 2

    if ma_bullish:
        score += 2

    if structure_bullish:
        score += 2

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
