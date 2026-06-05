"""
src/core/utils.py — General-purpose utility functions
Pure helpers used across every subsystem.  No side effects, no imports
from other bot modules (keeps the dependency graph clean).

Covers:
  - Number / price formatting
  - Percent and P&L calculations
  - Spread and liquidity checks
  - Risk / reward math
  - JSON file I/O (trade files, watchlists, snapshots)
  - Dict merging and safe field access
  - Validation helpers
"""

import json
import os
import math
from datetime import datetime, timezone
from typing import Any, Optional, Union


# ── Number Formatting ─────────────────────────────────────────────────────────

def round_price(price: float, decimals: int = 2) -> float:
    """Round a price to the given number of decimal places."""
    return round(price, decimals)


def fmt_price(price: Optional[float], decimals: int = 2) -> str:
    """Format a price as a dollar string, e.g. '$3.42'. Returns 'N/A' if None."""
    if price is None:
        return "N/A"
    return f"${price:,.{decimals}f}"


def fmt_pct(value: Optional[float], decimals: int = 2) -> str:
    """Format a float as a percentage string, e.g. '42.00%'. Returns 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}%"


def fmt_dollars(value: Optional[float], decimals: int = 2) -> str:
    """Format a dollar P&L value with sign, e.g. '+$38.40' or '-$12.00'."""
    if value is None:
        return "N/A"
    if value >= 0:
        return f"+${value:,.{decimals}f}"
    return f"-${abs(value):,.{decimals}f}"


def fmt_shares(shares: int) -> str:
    """Format a share count with comma separator."""
    return f"{shares:,}"


# ── Percent / P&L Calculations ────────────────────────────────────────────────

def pct_change(entry: float, current: float) -> float:
    """
    Percentage change from entry to current price.
    Returns 0.0 if entry is zero to avoid division errors.
    """
    if entry == 0:
        return 0.0
    return (current - entry) / entry * 100


def dollar_pnl(entry: float, current: float, shares: int) -> float:
    """Unrealized or realized P&L in dollars."""
    return (current - entry) * shares


def risk_reward_ratio(entry: float, stop: float, target: float) -> float:
    """
    Reward-to-risk ratio.
    Returns 0.0 when risk is zero (stop == entry) to avoid division errors.
    """
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


def risk_per_share(entry: float, stop: float) -> float:
    """Dollar risk per share (always positive)."""
    return abs(entry - stop)


def position_value(price: float, shares: int) -> float:
    """Total dollar value of a position."""
    return price * shares


def spread_percent(bid: float, ask: float) -> float:
    """
    Bid-ask spread as a percentage of the ask price.
    Returns 0.0 if ask is zero.
    """
    if ask == 0:
        return 0.0
    return (ask - bid) / ask * 100


def vwap_distance_pct(price: float, vwap: float) -> float:
    """
    Percentage distance of price from VWAP.
    Positive = above VWAP, negative = below VWAP.
    """
    if vwap == 0:
        return 0.0
    return (price - vwap) / vwap * 100


def r_multiple(entry: float, stop: float, current: float) -> float:
    """
    Current gain expressed as a multiple of initial risk (R).
    Example: entry=3.00, stop=2.85, current=3.45 → R = 3.0
    """
    r = risk_per_share(entry, stop)
    if r == 0:
        return 0.0
    return (current - entry) / r


# ── Validation Helpers ────────────────────────────────────────────────────────

def is_valid_price(price: Any) -> bool:
    """True if price is a positive finite number."""
    try:
        f = float(price)
        return f > 0 and math.isfinite(f)
    except (TypeError, ValueError):
        return False


def is_valid_volume(volume: Any) -> bool:
    """True if volume is a non-negative integer or float."""
    try:
        f = float(volume)
        return f >= 0 and math.isfinite(f)
    except (TypeError, ValueError):
        return False


def is_spread_acceptable(bid: float, ask: float, max_spread_pct: float) -> bool:
    """True when the bid-ask spread is within the allowed maximum."""
    if ask <= 0 or bid <= 0:
        return False
    return spread_percent(bid, ask) <= max_spread_pct


def is_quote_fresh(quote_timestamp: Union[str, datetime, None],
                   max_age_seconds: float = 20) -> bool:
    """
    True when the quote is recent enough to trade on.

    Args:
        quote_timestamp: ISO-8601 string or datetime object (UTC assumed).
        max_age_seconds: Maximum acceptable age in seconds.
    """
    if quote_timestamp is None:
        return False
    if isinstance(quote_timestamp, str):
        try:
            quote_timestamp = datetime.fromisoformat(quote_timestamp)
        except ValueError:
            return False
    if quote_timestamp.tzinfo is None:
        quote_timestamp = quote_timestamp.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - quote_timestamp).total_seconds()
    return 0 <= age <= max_age_seconds


def passes_risk_reward(entry: float, stop: float, target: float,
                        minimum_rr: float = 3.0) -> bool:
    """True when the trade's reward-to-risk meets the minimum threshold."""
    return risk_reward_ratio(entry, stop, target) >= minimum_rr


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# ── Safe Dict Access ──────────────────────────────────────────────────────────

def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely traverse a nested dict with a chain of keys.

    Example:
        safe_get(cfg, "risk", "max_open_positions", default=5)
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


def merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.  Values in override win.
    Neither input dict is mutated.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


# ── JSON File I/O ─────────────────────────────────────────────────────────────

def read_json(path: str, default: Any = None) -> Any:
    """
    Read and parse a JSON file.

    Returns `default` if the file does not exist or cannot be parsed,
    rather than raising — callers should always check for None/empty returns.
    """
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: str, data: Any, indent: int = 2) -> bool:
    """
    Write data to a JSON file, creating parent directories if needed.

    Returns True on success, False on failure (so callers can log/handle).
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except OSError:
        return False


def append_json_line(path: str, data: dict) -> bool:
    """
    Append a single JSON object as a new line to a file (JSON-Lines format).
    Used for rejection_log.json and exit_log.json.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
        return True
    except OSError:
        return False


def read_json_lines(path: str) -> list[dict]:
    """
    Read a JSON-Lines file and return a list of parsed dicts.
    Skips lines that fail to parse rather than crashing.
    """
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ── Trade File Helpers ────────────────────────────────────────────────────────

def trade_file_path(base_dir: str, ticker: str, trade_id: str) -> str:
    """
    Build a canonical trade JSON file path.

    Example:
        trade_file_path("trades/open", "ABCD", "abc-123") →
            "trades/open/ABCD_abc-123.json"
    """
    filename = f"{ticker}_{trade_id}.json"
    return os.path.join(base_dir, filename)


def list_trade_files(directory: str) -> list[str]:
    """Return all .json file paths in a trade directory, sorted by name."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json")
    )


# ── Timestamp Helpers (thin wrappers — time_utils has the full session logic) ─

def now_utc() -> datetime:
    """Current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def iso_to_dt(iso_str: str) -> Optional[datetime]:
    """
    Parse an ISO-8601 string to a timezone-aware datetime.
    Returns None on parse failure.
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def seconds_since(iso_str: str) -> float:
    """
    Seconds elapsed since a UTC ISO-8601 timestamp.
    Returns float('inf') if the string cannot be parsed.
    """
    dt = iso_to_dt(iso_str)
    if dt is None:
        return float("inf")
    return (now_utc() - dt).total_seconds()


# ── Scoring Helpers ───────────────────────────────────────────────────────────

def weighted_score(components: list[tuple[float, float]]) -> float:
    """
    Compute a weighted score from a list of (value, weight) pairs.
    Weights do not need to sum to 1 — they are normalized automatically.

    Example:
        weighted_score([(80, 0.35), (75, 0.35), (90, 0.30)]) → 80.5
    """
    if not components:
        return 0.0
    total_weight = sum(w for _, w in components)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in components) / total_weight


def confidence_label(score: float) -> str:
    """
    Map a numeric score to the design doc's confidence label.

    90-100 → elite
    80-89  → strong
    70-79  → decent
    60-69  → weak
    <60    → reject
    """
    if score >= 90:
        return "elite"
    if score >= 80:
        return "strong"
    if score >= 70:
        return "decent"
    if score >= 60:
        return "weak"
    return "reject"


def size_reduction_pct(final_score: float, cfg: dict) -> float:
    """
    Return the position size reduction percentage based on trade quality score
    and the confidence_sizing block from risk settings.

    Returns a float in [0, 100] representing the allowed size percentage.
    Example: 75.0 means use 75% of the calculated position size.
    """
    cs = cfg.get("confidence_sizing", {})
    if not cs.get("enabled", True):
        return 100.0
    if final_score >= 90:
        return float(cs.get("score_90_plus_size_pct", 100))
    if final_score >= 85:
        return float(cs.get("score_85_to_89_size_pct", 75))
    if final_score >= 80:
        return float(cs.get("score_80_to_84_size_pct", 50))
    return 0.0   # below 80 → no trade
