"""
src/setups/fibonacci_pullback.py — Fibonacci Pullback setup detector
Detects when price pulls back to a preferred Fibonacci retracement level
and shows signs of continuation.

The Fibonacci pullback setup is a continuation setup:
  1. A valid swing high and swing low define the move
  2. Price pulls back near a preferred retracement level
  3. The preferred level is usually 0.382, 0.500, or 0.618
  4. Price holds the level with VWAP support
  5. A higher low forms near the retracement
  6. Volume contracts on the pullback, then expands on continuation

This setup is strongest when:
  - Price is near 0.382, 0.500, or 0.618
  - Price is above or reclaiming VWAP
  - A higher low confirms buyers defended the pullback
  - MACD and MA trend support continuation
  - Fibonacci extension levels provide clean targets

Returns a SetupResult with:
  confirmed:     True when all conditions are met
  score:         0–100 based on pullback quality
  entry_trigger: Current price or reclaim above Fib level
  stop_area:     Below the Fib level / higher low
  target_area:   Fibonacci extension target
  reasons:       Human-readable confirmation factors
  warnings:      Any concerns about the setup quality
"""

from __future__ import annotations

import logging
from typing import Optional

from models import ConfidenceLabel, IndicatorSnapshot, SetupResult, SetupName

log = logging.getLogger(__name__)

_SETUP_NAME = SetupName.FIBONACCI_PULLBACK.value


def detect(
    bars:       list[dict],
    indicators: IndicatorSnapshot,
    context:    dict,
    settings:   dict,
) -> SetupResult:
    """
    Detect the Fibonacci Pullback setup.

    Args:
        bars:       Regular-session OHLCV bars, oldest→newest.
        indicators: IndicatorSnapshot from indicator_engine.
        context:    Analysis context dict. Expected keys:
                      ticker, current_price, vwap,
                      fib_result, key_levels, structure
        settings:   Full bot_settings dict.

    Returns:
        SetupResult — confirmed=True when setup is valid.
    """
    cfg           = settings.get("setups", {}).get(_SETUP_NAME, {})
    ticker        = context.get("ticker", "")
    current_price = float(context.get("current_price", 0))
    vwap          = context.get("vwap") or indicators.vwap
    fib_result    = context.get("fib_result")

    reasons:  list[str] = []
    warnings: list[str] = []

    # ── Basic data validation ─────────────────────────────────────────────────
    if not bars or current_price <= 0:
        return _reject("Insufficient data", ticker)

    if not fib_result:
        return _reject("Fibonacci result unavailable", ticker)

    if getattr(fib_result, "block_trade", False):
        return _reject("Fibonacci engine is blocking long entry", ticker)

    if not getattr(fib_result, "fib_trend_valid", False):
        return _reject("Fibonacci trend is not valid", ticker)

    nearest = getattr(fib_result, "nearest_retracement", None)
    if not nearest:
        return _reject("No nearest Fibonacci retracement found", ticker)

    fib_price = float(getattr(nearest, "price", 0) or 0)
    fib_ratio = float(getattr(nearest, "ratio", 0) or 0)

    if fib_price <= 0:
        return _reject("Nearest Fibonacci retracement price unavailable", ticker)

    # ── Preferred retracement check ───────────────────────────────────────────
    preferred_levels = list(cfg.get("preferred_retracement_levels", [0.382, 0.5, 0.618]))
    at_preferred = bool(getattr(fib_result, "at_preferred_level", False)) or _ratio_is_preferred(
        fib_ratio, preferred_levels
    )

    if not at_preferred:
        return _reject(
            f"Nearest Fib level {fib_ratio:.3f} is not a preferred retracement",
            ticker,
        )

    reasons.append(f"Price is near preferred Fib retracement {fib_ratio:.3f}")

    # ── Distance from Fib level ───────────────────────────────────────────────
    max_distance = float(cfg.get("max_distance_from_fib_percent", 2.0))
    distance_pct = float(getattr(fib_result, "distance_from_fib_pct", 0.0) or 0.0)

    if distance_pct > max_distance:
        return _reject(
            f"Price is {distance_pct:.2f}% from Fib level "
            f"(max {max_distance:.2f}%)",
            ticker,
        )

    reasons.append(f"Price is close to Fib level ({distance_pct:.2f}% away)")

    # ── VWAP confirmation ─────────────────────────────────────────────────────
    require_vwap = bool(cfg.get("require_vwap_confirmation", True))
    vwap_ok = bool(vwap and current_price >= vwap)

    if require_vwap and not vwap_ok:
        return _reject("VWAP confirmation missing for Fibonacci pullback", ticker)

    if vwap_ok:
        reasons.append("Price is above VWAP — pullback has VWAP support")
    elif vwap:
        warnings.append("Price is below VWAP — Fib pullback quality reduced")

    # ── Higher low confirmation ───────────────────────────────────────────────
    require_higher_low = bool(cfg.get("require_higher_low", True))
    higher_low = bool(getattr(indicators, "higher_lows", False)) or _detect_higher_low_near_fib(
        bars, fib_price
    )

    if require_higher_low and not higher_low:
        return _reject("Higher low confirmation missing near Fibonacci level", ticker)

    if higher_low:
        reasons.append("Higher low confirmed near Fibonacci pullback area")

    # ── Pullback volume behavior ──────────────────────────────────────────────
    volume_contracting = _volume_contracting_on_pullback(bars)
    volume_expanding   = _latest_volume_expanding(bars)

    if volume_contracting:
        reasons.append("Volume contracted during pullback")
    else:
        warnings.append("Pullback volume did not clearly contract")

    if volume_expanding:
        reasons.append("Latest volume is expanding for continuation")
    elif indicators.relative_volume < 2.0:
        warnings.append(f"Low RVOL: {indicators.relative_volume:.1f}x")

    # ── Entry confirmation by Fibonacci engine ────────────────────────────────
    fib_entry_confirmed = bool(getattr(fib_result, "entry_confirmed_by_fib", False))
    if fib_entry_confirmed:
        reasons.append("Fibonacci engine confirms entry conditions")
    else:
        warnings.append("Fibonacci entry confirmation is not fully complete")

    # ── Momentum confirmation ─────────────────────────────────────────────────
    macd_bullish = bool(indicators.macd and indicators.macd.bullish)
    macd_cross   = bool(indicators.macd and indicators.macd.bullish_crossover)

    if macd_cross:
        reasons.append("MACD bullish crossover supports continuation")
    elif macd_bullish:
        reasons.append("MACD bullish — momentum supports Fib pullback")
    else:
        warnings.append("MACD is not yet bullish")

    ma_bullish = indicators.ma_trend == "bullish"
    if ma_bullish:
        reasons.append("MA trend is bullish")
    elif indicators.ma_trend == "bearish":
        warnings.append("MA trend is bearish — continuation less reliable")

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
        warnings.append("Session structure is bearish")

    # ── RSI zone ──────────────────────────────────────────────────────────────
    if indicators.rsi_zone == "neutral":
        reasons.append("RSI neutral — pullback not overheated")
    elif indicators.rsi_zone == "oversold":
        warnings.append("RSI oversold — pullback may still be weak")
    elif indicators.rsi_zone == "overbought":
        warnings.append("RSI overbought — continuation may be extended")

    # ── Entry / stop / target ─────────────────────────────────────────────────
    entry_trigger = current_price

    stop_area = _choose_stop_area(
        current_price = current_price,
        fib_price     = fib_price,
        bars          = bars,
    )

    target_area = _choose_target_area(fib_result, current_price, context)

    # ── Score ─────────────────────────────────────────────────────────────────
    score = _score(
        distance_pct       = distance_pct,
        max_distance       = max_distance,
        fib_ratio          = fib_ratio,
        fib_entry_confirmed= fib_entry_confirmed,
        vwap_ok            = vwap_ok,
        higher_low         = higher_low,
        volume_contracting = volume_contracting,
        volume_expanding   = volume_expanding,
        rvol               = indicators.relative_volume,
        macd_bullish       = macd_bullish,
        macd_cross         = macd_cross,
        ma_bullish         = ma_bullish,
        structure_bullish  = structure_bullish,
        rsi_zone           = indicators.rsi_zone,
    )

    confidence = _confidence_label(score)

    log.debug(
        "[%s] %s score=%.1f fib=%.3f dist=%.2f%% vwap_ok=%s higher_low=%s",
        _SETUP_NAME, ticker, score, fib_ratio, distance_pct, vwap_ok, higher_low,
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

def _ratio_is_preferred(ratio: float, preferred: list[float]) -> bool:
    """
    True when a Fibonacci ratio matches a preferred retracement.
    Uses small tolerance to handle 0.500 vs 0.5 formatting.
    """
    return any(abs(ratio - p) <= 0.001 for p in preferred)


def _detect_higher_low_near_fib(bars: list[dict], fib_price: float) -> bool:
    """
    True when a recent higher low formed near the Fibonacci level.
    """
    if len(bars) < 3 or fib_price <= 0:
        return False

    recent = bars[-3:]
    lows = [b.get("l", 0) for b in recent if b.get("l", 0) > 0]

    if len(lows) < 2:
        return False

    latest_low = lows[-1]
    prev_low   = lows[-2]

    near_fib = abs(latest_low - fib_price) / fib_price * 100 <= 2.0
    return latest_low > prev_low and near_fib


def _volume_contracting_on_pullback(bars: list[dict]) -> bool:
    """
    True when recent pullback volume is lower than earlier volume.
    """
    if len(bars) < 8:
        return False

    earlier = bars[-8:-4]
    recent  = bars[-4:]

    earlier_avg = sum(b.get("v", 0) for b in earlier) / len(earlier)
    recent_avg  = sum(b.get("v", 0) for b in recent) / len(recent)

    if earlier_avg <= 0:
        return False

    return recent_avg <= earlier_avg * 0.85


def _latest_volume_expanding(bars: list[dict]) -> bool:
    """
    True when the latest bar volume is stronger than the prior average.
    """
    if len(bars) < 6:
        return False

    latest = bars[-1].get("v", 0)
    prior  = bars[-6:-1]
    avg    = sum(b.get("v", 0) for b in prior) / len(prior)

    if avg <= 0:
        return False

    return latest >= avg * 1.2


def _choose_stop_area(
    current_price: float,
    fib_price:     float,
    bars:          list[dict],
) -> float:
    """
    Choose a practical stop area below the Fibonacci pullback.
    """
    recent_lows = [
        b.get("l", 0)
        for b in bars[-5:]
        if b.get("l", 0) > 0 and b.get("l", 0) < current_price
    ]

    if recent_lows:
        return min(recent_lows) * 0.995

    if fib_price > 0:
        return fib_price * 0.98

    return current_price * 0.95


def _choose_target_area(
    fib_result: object,
    current_price: float,
    context: dict,
) -> float:
    """
    Choose target from Fibonacci extensions, then nearest resistance,
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

    key_levels = context.get("key_levels")
    if not target and key_levels and getattr(key_levels, "nearest_resistance", None):
        target = key_levels.nearest_resistance.price

    return target if target else current_price * 1.06


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    distance_pct:        float,
    max_distance:        float,
    fib_ratio:           float,
    fib_entry_confirmed: bool,
    vwap_ok:             bool,
    higher_low:          bool,
    volume_contracting:  bool,
    volume_expanding:    bool,
    rvol:                float,
    macd_bullish:        bool,
    macd_cross:          bool,
    ma_bullish:          bool,
    structure_bullish:   bool,
    rsi_zone:            str,
) -> float:
    """
    Score the Fibonacci Pullback setup quality 0–100.

    Weights:
      Fib distance:           15 pts
      Preferred Fib level:    10 pts
      Fib engine confirm:     10 pts
      VWAP support:           15 pts
      Higher low:             15 pts
      Volume behavior:        15 pts
      Momentum / structure:   15 pts
      RSI zone:                5 pts
    """
    score = 0.0

    # Fib distance (15 pts)
    if distance_pct <= max_distance * 0.5:
        score += 15
    elif distance_pct <= max_distance:
        score += 10

    # Preferred Fib level (10 pts)
    if abs(fib_ratio - 0.382) <= 0.001:
        score += 10
    elif abs(fib_ratio - 0.500) <= 0.001:
        score += 9
    elif abs(fib_ratio - 0.618) <= 0.001:
        score += 8

    # Fib engine confirmation (10 pts)
    if fib_entry_confirmed:
        score += 10

    # VWAP support (15 pts)
    if vwap_ok:
        score += 15

    # Higher low (15 pts)
    if higher_low:
        score += 15

    # Volume behavior (15 pts)
    if volume_contracting:
        score += 7
    if volume_expanding:
        score += 5
    if rvol >= 3.0:
        score += 3
    elif rvol >= 2.0:
        score += 2

    # Momentum / structure (15 pts)
    if macd_cross:
        score += 5
    elif macd_bullish:
        score += 3

    if ma_bullish:
        score += 5

    if structure_bullish:
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
