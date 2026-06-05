"""
data/indicator_calculator.py — Technical indicator calculations
Computes RSI, MACD, VWAP, MAs, ATR, relative volume, candle strength,
trend strength, and all derived signal fields required by the scoring,
setup, and exit engines.

All functions are pure — they take bar data and return values. No side effects.

Bar dict keys expected:
    o  = open
    h  = high
    l  = low
    c  = close
    v  = volume (int or float)
"""

import logging
import math
from typing import Optional

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_series(bars: list[dict], field: str) -> list[float]:
    """Pull a single numeric field from every bar that contains it."""
    return [b[field] for b in bars if field in b]


# ── Moving Averages ───────────────────────────────────────────────────────────

def sma(values: list[float], period: int) -> Optional[float]:
    """Simple moving average over the last `period` values."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: list[float], period: int) -> Optional[float]:
    """
    Exponential moving average seeded with SMA of the first `period` values.
    Returns the single latest EMA value.
    """
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    result = sum(values[:period]) / period
    for v in values[period:]:
        result = v * k + result * (1 - k)
    return result


def ema_series(values: list[float], period: int) -> list[float]:
    """
    Return a full EMA series the same length as `values`.
    Early bars (before the seed window) are filled with NaN.
    """
    if len(values) < period:
        return [float("nan")] * len(values)
    k = 2 / (period + 1)
    emas: list[float] = [float("nan")] * (period - 1)
    seed = sum(values[:period]) / period
    emas.append(seed)
    for v in values[period:]:
        emas.append(v * k + emas[-1] * (1 - k))
    return emas


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """
    Wilder-smoothed RSI.
    Returns None when there is insufficient data.
    Returns 100.0 when average loss is zero (pure up-trend).
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def rsi_zone(rsi_val: Optional[float]) -> str:
    """
    Classify RSI into a human-readable zone used by the scoring engines.

    Returns:
        "oversold"   — RSI < 30
        "neutral"    — 30 <= RSI <= 70
        "overbought" — RSI > 70
        "unknown"    — RSI is None
    """
    if rsi_val is None:
        return "unknown"
    if rsi_val < 30:
        return "oversold"
    if rsi_val > 70:
        return "overbought"
    return "neutral"


# ── MACD ──────────────────────────────────────────────────────────────────────

def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[dict]:
    """
    Standard MACD with histogram, direction, bullish/bearish crossover flags.

    Returns a dict with keys:
        macd                 — MACD line value
        signal               — Signal line value
        histogram            — MACD minus signal
        histogram_direction  — "expanding" or "contracting"
        bullish_crossover    — True on the bar the MACD crosses above signal
        bearish_crossover    — True on the bar the MACD crosses below signal
        bullish              — True while MACD is above signal

    Returns None when there is insufficient data.
    """
    if len(closes) < slow + signal:
        return None

    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)

    macd_line = [
        f - s if not (math.isnan(f) or math.isnan(s)) else float("nan")
        for f, s in zip(ema_fast, ema_slow)
    ]

    valid_macd = [v for v in macd_line if not math.isnan(v)]
    if len(valid_macd) < signal:
        return None

    signal_line   = ema_series(valid_macd, signal)
    latest_macd   = valid_macd[-1]
    latest_signal = signal_line[-1]
    histogram     = latest_macd - latest_signal

    if len(valid_macd) >= 2 and len(signal_line) >= 2:
        prev_macd      = valid_macd[-2]
        prev_signal    = signal_line[-2]
        prev_histogram = prev_macd - prev_signal

        # Bullish crossover: MACD was at or below signal, now above
        bullish_crossover = (latest_macd > latest_signal) and (prev_macd <= prev_signal)
        # Bearish crossover: MACD was at or above signal, now below
        bearish_crossover = (latest_macd < latest_signal) and (prev_macd >= prev_signal)
    else:
        prev_histogram    = 0.0
        bullish_crossover = False
        bearish_crossover = False

    return {
        "macd":                latest_macd,
        "signal":              latest_signal,
        "histogram":           histogram,
        "histogram_direction": "expanding" if abs(histogram) > abs(prev_histogram) else "contracting",
        "bullish_crossover":   bullish_crossover,
        "bearish_crossover":   bearish_crossover,
        "bullish":             latest_macd > latest_signal,
    }


# ── VWAP ──────────────────────────────────────────────────────────────────────

def vwap(bars: list[dict]) -> Optional[float]:
    """
    Compute VWAP using typical price (H+L+C)/3 weighted by volume.

    IMPORTANT: The caller is responsible for passing only session bars
    (i.e., bars from the current regular-hours session).  This function
    treats every bar it receives as part of the same session.

    Returns None when bars is empty or total volume is zero.
    """
    if not bars:
        return None

    total_pv = 0.0
    total_v  = 0.0
    for b in bars:
        typical   = (b["h"] + b["l"] + b["c"]) / 3
        vol       = b.get("v", 0)
        total_pv += typical * vol
        total_v  += vol

    return total_pv / total_v if total_v > 0 else None


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr(bars: list[dict], period: int = 14) -> Optional[float]:
    """
    Average True Range using Wilder smoothing.
    Requires at least `period + 1` bars (one extra to compute the first TR).
    """
    if len(bars) < period + 1:
        return None

    trs: list[float] = []
    for i in range(1, len(bars)):
        h      = bars[i]["h"]
        l      = bars[i]["l"]
        prev_c = bars[i - 1]["c"]
        tr     = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)

    if len(trs) < period:
        return None

    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


# ── Volume ────────────────────────────────────────────────────────────────────

def relative_volume(bars: list[dict], lookback: int = 20) -> float:
    """
    Current bar volume divided by the average volume of the prior `lookback` bars.
    Returns 0.0 when there is insufficient data.
    """
    if len(bars) < 2:
        return 0.0
    current = bars[-1].get("v", 0)
    prior   = bars[-(lookback + 1):-1]
    if not prior:
        return 0.0
    avg = sum(b.get("v", 0) for b in prior) / len(prior)
    return current / avg if avg > 0 else 0.0


def volume_trend(bars: list[dict], lookback: int = 5) -> str:
    """
    Detect volume direction over the last `lookback` bars by comparing
    the average of the first half vs. the second half.

    Returns 'increasing', 'decreasing', or 'flat'.
    """
    if len(bars) < lookback:
        return "flat"
    vols             = [b.get("v", 0) for b in bars[-lookback:]]
    mid              = len(vols) // 2
    first_half_avg   = sum(vols[:mid]) / mid if mid else 0
    second_half_len  = len(vols) - mid
    second_half_avg  = sum(vols[mid:]) / second_half_len if second_half_len else 0

    if second_half_avg > first_half_avg * 1.1:
        return "increasing"
    if second_half_avg < first_half_avg * 0.9:
        return "decreasing"
    return "flat"


# ── Candle Strength ───────────────────────────────────────────────────────────

def candle_strength(bar: dict) -> float:
    """
    Body-to-range ratio for a single bar.

    Returns a value in [-1.0, 1.0]:
        +1.0  = full bullish engulfing candle (open == low, close == high)
        -1.0  = full bearish engulfing candle (open == high, close == low)
         0.0  = doji or zero-range bar

    Used by setup detectors and the probability engine to confirm momentum.
    """
    bar_range = bar["h"] - bar["l"]
    if bar_range == 0:
        return 0.0
    body = bar.get("c", 0) - bar.get("o", 0)
    return body / bar_range


def average_candle_strength(bars: list[dict], lookback: int = 5) -> float:
    """
    Average candle strength over the last `lookback` bars.
    Positive = net bullish momentum; negative = net bearish.
    """
    if not bars:
        return 0.0
    recent = bars[-lookback:]
    if not recent:
        return 0.0
    return sum(candle_strength(b) for b in recent) / len(recent)


# ── Trend Strength ────────────────────────────────────────────────────────────

def trend_strength(closes: list[float], fast: int = 9, slow: int = 20) -> dict:
    """
    Measure trend strength from the fast/slow EMA relationship and their spread.

    Returns a dict with:
        direction   — "bullish", "bearish", or "flat"
        spread_pct  — percentage spread between fast and slow EMA
                      (positive = fast above slow)
        ma_trend    — same as direction; exposed for convenience
    """
    fast_val = ema(closes, fast)
    slow_val = ema(closes, slow)

    if fast_val is None or slow_val is None or slow_val == 0:
        return {"direction": "flat", "spread_pct": 0.0, "ma_trend": "flat"}

    spread_pct = (fast_val - slow_val) / slow_val * 100

    if spread_pct > 0.1:
        direction = "bullish"
    elif spread_pct < -0.1:
        direction = "bearish"
    else:
        direction = "flat"

    return {
        "direction":  direction,
        "spread_pct": round(spread_pct, 4),
        "ma_trend":   direction,
    }


# ── Market Structure ──────────────────────────────────────────────────────────

def find_swing_highs(bars: list[dict], lookback: int = 3) -> list[float]:
    """
    Return a list of swing-high prices: bars whose high is greater than
    every bar within `lookback` bars on either side.
    """
    highs: list[float] = []
    for i in range(lookback, len(bars) - lookback):
        if all(
            bars[i]["h"] > bars[j]["h"]
            for j in range(i - lookback, i + lookback + 1)
            if j != i
        ):
            highs.append(bars[i]["h"])
    return highs


def find_swing_lows(bars: list[dict], lookback: int = 3) -> list[float]:
    """
    Return a list of swing-low prices: bars whose low is less than
    every bar within `lookback` bars on either side.
    """
    lows: list[float] = []
    for i in range(lookback, len(bars) - lookback):
        if all(
            bars[i]["l"] < bars[j]["l"]
            for j in range(i - lookback, i + lookback + 1)
            if j != i
        ):
            lows.append(bars[i]["l"])
    return lows


def detect_higher_lows(bars: list[dict], count: int = 3) -> bool:
    """True when the most recent `count` swing lows are each higher than the previous."""
    lows = find_swing_lows(bars)
    if len(lows) < count:
        return False
    recent = lows[-count:]
    return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))


def detect_lower_highs(bars: list[dict], count: int = 3) -> bool:
    """True when the most recent `count` swing highs are each lower than the previous."""
    highs = find_swing_highs(bars)
    if len(highs) < count:
        return False
    recent = highs[-count:]
    return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))


# ── Master Indicator Bundle ───────────────────────────────────────────────────

def compute_all(bars: list[dict], settings: dict) -> dict:
    """
    Compute every indicator the system needs and return one standardized dict.

    This is the single entry point consumed by:
        - setup detectors  (setups/)
        - scoring engines  (scoring/)
        - exit engine      (exits/)
        - move potential   (analysis/)
        - risk manager     (risk/)

    Config is read from settings["indicators"] with safe defaults so the
    function works even when that section is absent from bot_settings.json.

    Args:
        bars:     List of OHLCV bar dicts ordered oldest → newest.
                  Caller must pre-filter to session bars for correct VWAP.
        settings: Full bot_settings dict (or any dict containing an
                  optional "indicators" sub-dict with period overrides).

    Returns:
        A flat dict of indicator values and derived signal fields.
        Empty dict if `bars` is empty.
    """
    if not bars:
        return {}

    cfg          = settings.get("indicators", {})
    closes       = extract_series(bars, "c")
    latest_close = closes[-1] if closes else 0.0
    latest_bar   = bars[-1]

    # ── Core indicators ───────────────────────────────────────────────────────
    vwap_val     = vwap(bars)
    rsi_val      = rsi(closes, cfg.get("rsi_period", 14))
    macd_val     = macd(
                       closes,
                       cfg.get("macd_fast",   12),
                       cfg.get("macd_slow",   26),
                       cfg.get("macd_signal",  9),
                   )
    atr_val      = atr(bars, cfg.get("atr_period", 14))
    ma_fast_val  = ema(closes, cfg.get("ma_fast",  9))
    ma_slow_val  = ema(closes, cfg.get("ma_slow", 20))
    rvol         = relative_volume(bars, cfg.get("volume_ma_period", 20))

    # ── Derived VWAP fields ───────────────────────────────────────────────────
    vwap_distance_pct = (
        (latest_close - vwap_val) / vwap_val * 100 if vwap_val else 0.0
    )
    # Flag when price is too far above VWAP for safe entry.
    # Threshold is configurable; defaults to 8% per the entry_rules spec.
    max_extension   = cfg.get("max_vwap_extension_pct", 8.0)
    vwap_extended   = vwap_distance_pct > max_extension

    # ── Trend / MA relationship ───────────────────────────────────────────────
    ts = trend_strength(closes, cfg.get("ma_fast", 9), cfg.get("ma_slow", 20))

    # ── Candle quality ────────────────────────────────────────────────────────
    latest_candle_strength  = candle_strength(latest_bar)
    avg_candle_str          = average_candle_strength(bars, cfg.get("candle_lookback", 5))

    # ── RSI zone ──────────────────────────────────────────────────────────────
    rsi_zone_val = rsi_zone(rsi_val)

    return {
        # ── VWAP ─────────────────────────────────────────────────────────────
        "vwap":               vwap_val,
        "price_vs_vwap":      "above" if (vwap_val and latest_close > vwap_val) else "below",
        "vwap_distance_pct":  round(vwap_distance_pct, 4),
        "vwap_extended":      vwap_extended,          # bool — overextension guard

        # ── RSI ──────────────────────────────────────────────────────────────
        "rsi":                rsi_val,
        "rsi_zone":           rsi_zone_val,           # "oversold" | "neutral" | "overbought" | "unknown"

        # ── MACD ─────────────────────────────────────────────────────────────
        "macd":               macd_val,               # full dict or None

        # ── ATR ──────────────────────────────────────────────────────────────
        "atr":                atr_val,

        # ── Moving averages ───────────────────────────────────────────────────
        "ma_fast":            ma_fast_val,
        "ma_slow":            ma_slow_val,
        "ma_trend":           ts["direction"],         # "bullish" | "bearish" | "flat"
        "ma_spread_pct":      ts["spread_pct"],        # fast-slow spread as % of slow

        # ── Volume ───────────────────────────────────────────────────────────
        "relative_volume":    rvol,
        "volume_trend":       volume_trend(bars),      # "increasing" | "decreasing" | "flat"

        # ── Candle strength ───────────────────────────────────────────────────
        "candle_strength":    round(latest_candle_strength, 4),   # -1.0 to +1.0
        "avg_candle_strength": round(avg_candle_str, 4),          # rolling average

        # ── Market structure ─────────────────────────────────────────────────
        "higher_lows":        detect_higher_lows(bars),
        "lower_highs":        detect_lower_highs(bars),
        "swing_highs":        find_swing_highs(bars),
        "swing_lows":         find_swing_lows(bars),

        # ── Convenience ──────────────────────────────────────────────────────
        "latest_close":       latest_close,
        "latest_bar":         latest_bar,
    }
