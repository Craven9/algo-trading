from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.config_loader import load_settings
from src.data.massive_rest_client import MassiveRestClient


PROJECT_ROOT = Path("C:/Projects/Algo-Bot-Trader")
WATCHLIST_DIR = PROJECT_ROOT / "watchlist"

SCANNER_CANDIDATES_PATH = WATCHLIST_DIR / "scanner_candidates.json"
ACTIVE_CANDIDATES_PATH = WATCHLIST_DIR / "active_candidates.json"

MAX_TICKERS = 100


def main() -> None:
    settings = load_settings()
    scanner_cfg = settings.get("scanner", {})

    min_price = float(scanner_cfg.get("min_price", 0.5))
    max_price = float(scanner_cfg.get("max_price", 1000.0))
    min_dollar_volume = float(scanner_cfg.get("min_dollar_volume", 500000))
    exclude_etfs = bool(scanner_cfg.get("exclude_etfs", False))
    exclude_otc = bool(scanner_cfg.get("exclude_otc", True))

    client = MassiveRestClient(settings)

    if not client.available():
        raise RuntimeError("Massive/Polygon client is not available. Check MASSIVE_API_KEY or POLYGON_API_KEY in .env")

    market_date = get_recent_market_date()

    print(f"Building market universe from Massive/Polygon grouped daily data: {market_date}")

    rows = safe_get_grouped_daily(client, market_date)

    if not rows:
        raise RuntimeError("No grouped daily rows returned from Massive/Polygon.")

    candidates = []

    for row in rows:
        ticker = str(row.get("ticker") or row.get("T") or row.get("symbol") or "").upper().strip()

        if not ticker:
            continue

        if is_bad_ticker_type(ticker):
            continue

        if exclude_otc and "." in ticker:
            continue

        if exclude_etfs and is_probable_etf(ticker):
            continue

        close_price = to_float(row.get("c") or row.get("close") or row.get("price"))
        open_price = to_float(row.get("o") or row.get("open"))
        volume = to_float(row.get("v") or row.get("volume"))
        vw = to_float(row.get("vw") or row.get("vwap"))

        if close_price <= 0:
            continue

        dollar_volume = close_price * volume

        if close_price < min_price:
            continue

        if close_price > max_price:
            continue

        if dollar_volume < min_dollar_volume:
            continue

        if open_price > 0:
            day_change_pct = ((close_price - open_price) / open_price) * 100.0
        else:
            day_change_pct = 0.0

        score = dollar_volume_score(dollar_volume) + abs(day_change_pct)

        candidates.append(
            {
                "ticker": ticker,
                "price": round(close_price, 4),
                "open": round(open_price, 4),
                "volume": round(volume, 2),
                "vwap": round(vw, 4),
                "dollar_volume": round(dollar_volume, 2),
                "day_change_pct": round(day_change_pct, 4),
                "score": round(score, 4),
                "source": "massive_grouped_daily",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = dedupe_by_ticker(candidates)[:MAX_TICKERS]

    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "massive_grouped_daily",
        "market_date": market_date,
        "count": len(candidates),
        "tickers": [c["ticker"] for c in candidates],
        "candidates": candidates,
    }

    SCANNER_CANDIDATES_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    ACTIVE_CANDIDATES_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved {len(candidates)} tickers.")
    print(f"Wrote: {SCANNER_CANDIDATES_PATH}")
    print(f"Wrote: {ACTIVE_CANDIDATES_PATH}")
    print()
    print("Top 20:")
    for c in candidates[:20]:
        print(
            f"{c['ticker']:6} price=${c['price']:<8} "
            f"change={c['day_change_pct']:<8}% "
            f"dollar_vol=${round(c['dollar_volume'] / 1_000_000, 2)}M"
        )


def safe_get_grouped_daily(client: MassiveRestClient, market_date: str) -> list[dict]:
    try:
        rows = client.get_grouped_daily(market_date)
        return normalize_rows(rows)
    except TypeError:
        pass

    try:
        rows = client.get_grouped_daily(date=market_date)
        return normalize_rows(rows)
    except TypeError:
        pass

    try:
        rows = client.get_grouped_daily(market_date, adjusted=True)
        return normalize_rows(rows)
    except TypeError:
        pass

    rows = client.get_grouped_daily()
    return normalize_rows(rows)


def normalize_rows(rows) -> list[dict]:
    if rows is None:
        return []

    if isinstance(rows, list):
        return rows

    if isinstance(rows, dict):
        if isinstance(rows.get("results"), list):
            return rows["results"]

        if isinstance(rows.get("tickers"), list):
            return rows["tickers"]

        if isinstance(rows.get("data"), list):
            return rows["data"]

    return []


def get_recent_market_date() -> str:
    """
    Return the most recent completed market day.

    Polygon/Massive free/lower plans often block today's grouped daily
    data until after market close, so we intentionally use the previous
    weekday.
    """
    day = date.today() - timedelta(days=1)

    # If weekend, back up to Friday.
    while day.weekday() >= 5:
        day -= timedelta(days=1)

    return day.isoformat()


def to_float(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def dollar_volume_score(dollar_volume: float) -> float:
    if dollar_volume >= 1_000_000_000:
        return 20
    if dollar_volume >= 500_000_000:
        return 15
    if dollar_volume >= 100_000_000:
        return 10
    if dollar_volume >= 50_000_000:
        return 7
    if dollar_volume >= 10_000_000:
        return 5
    return 1


def is_bad_ticker_type(ticker: str) -> bool:
    """
    Skip warrants, rights, units, preferreds, and weird ticker classes.
    These often show huge % moves but are poor candidates for this bot.
    """
    bad_suffixes = (
        "W", "WS", "WT", "WWS",
        "R", "RT", "U",
        "P", "PR",
        "Z"
    )

    # Keep normal 1-5 letter common stocks.
    # Reject longer weird symbols unless manually added elsewhere.
    if len(ticker) > 5:
        return True

    for suffix in bad_suffixes:
        if ticker.endswith(suffix) and len(ticker) >= 4:
            return True

    return False


def is_probable_etf(ticker: str) -> bool:
    common_etfs = {
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "XLK", "XLF", "XLE", "XLV",
        "XLY", "XLP", "XLI", "XLB", "XLU", "ARKK", "TQQQ", "SQQQ", "SOXL",
        "SOXS", "SPXL", "SPXS"
    }
    return ticker in common_etfs


def dedupe_by_ticker(candidates: list[dict]) -> list[dict]:
    seen = set()
    cleaned = []

    for c in candidates:
        ticker = c["ticker"]
        if ticker in seen:
            continue

        seen.add(ticker)
        cleaned.append(c)

    return cleaned


if __name__ == "__main__":
    main()
