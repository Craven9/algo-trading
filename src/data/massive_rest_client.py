"""
src/data/massive_rest_client.py — Massive/Polygon-style REST market data client
Handles on-demand REST market data requests for candles, snapshots,
ticker details, and reference data.

The bot overview separates data access into two lanes:
  - WebSocket API for live streaming updates
  - REST API for on-demand/background data

This client is for the REST lane.  It should be used when the bot needs:
  - Recent candles
  - Historical candles
  - Ticker/reference data
  - One-off snapshot requests
  - Backup/fallback data when WebSocket data is unavailable

Responsibilities:
  - Build safe REST requests
  - Handle API key loading from environment/settings
  - Fetch aggregates/candles
  - Fetch ticker details/reference info
  - Fetch grouped/snapshot-style market data when available
  - Normalize responses into plain dicts/lists
  - Retry temporary failures
  - Fail safely and log errors

Design rules:
  - This file does not scan by itself
  - This file does not score trades
  - This file does not place orders
  - This file does not approve trades
  - WebSocket remains the preferred source for live streaming
  - REST is used for on-demand, fallback, and historical data
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Optional
from urllib.parse import urlencode

try:
    import requests
except Exception:
    requests = None

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "https://api.polygon.io"
_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_SLEEP_SECONDS = 1.0


# ── Client ────────────────────────────────────────────────────────────────────

class MassiveRestClient:
    """
    REST client for Massive/Polygon-style market data.

    Usage:
        client = MassiveRestClient(settings)
        bars = client.get_aggregate_bars("AAPL", "2026-06-01", "2026-06-05")
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._cfg = (
            settings.get("massive", {})
            or settings.get("polygon", {})
            or settings.get("market_data", {}).get("massive", {})
            or {}
        )

        self._base_url = str(self._cfg.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self._api_key = (
            self._cfg.get("api_key")
            or os.environ.get("MASSIVE_API_KEY")
            or os.environ.get("POLYGON_API_KEY")
            or ""
        )

        self._timeout = float(self._cfg.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        self._max_retries = int(self._cfg.get("max_retries", _DEFAULT_MAX_RETRIES))
        self._retry_sleep = float(
            self._cfg.get("retry_sleep_seconds", _DEFAULT_RETRY_SLEEP_SECONDS)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def available(self) -> bool:
        """True when requests and API key are available."""
        return requests is not None and bool(self._api_key)

    def get_aggregate_bars(
        self,
        ticker: str,
        from_date: str | date,
        to_date: str | date,
        multiplier: int = 1,
        timespan: str = "minute",
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
    ) -> list[dict]:
        """
        Fetch aggregate/candle bars.

        Args:
            ticker:     Symbol.
            from_date:  YYYY-MM-DD or date.
            to_date:    YYYY-MM-DD or date.
            multiplier: Bar multiplier.
            timespan:   minute|hour|day|week|month.
            adjusted:   Adjusted prices.
            sort:       asc|desc.
            limit:      Max bars.

        Returns:
            List of normalized OHLCV bars.
        """
        ticker = ticker.upper()
        from_s = _date_str(from_date)
        to_s = _date_str(to_date)

        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_s}/{to_s}"
        params = {
            "adjusted": str(bool(adjusted)).lower(),
            "sort": sort,
            "limit": limit,
        }

        data = self._get(path, params=params)
        results = data.get("results", []) if data else []
        return [normalize_aggregate_bar(bar) for bar in results]

    def get_previous_close(self, ticker: str, adjusted: bool = True) -> Optional[dict]:
        """
        Fetch previous close bar for ticker.
        """
        ticker = ticker.upper()
        path = f"/v2/aggs/ticker/{ticker}/prev"
        params = {"adjusted": str(bool(adjusted)).lower()}

        data = self._get(path, params=params)
        results = data.get("results", []) if data else []
        if not results:
            return None
        return normalize_aggregate_bar(results[0])

    def get_ticker_details(self, ticker: str) -> dict:
        """
        Fetch ticker reference/details.
        """
        ticker = ticker.upper()
        path = f"/v3/reference/tickers/{ticker}"

        data = self._get(path)
        if not data:
            return {}
        return data.get("results", data)

    def get_ticker_news(
        self,
        ticker: str,
        limit: int = 10,
        order: str = "desc",
    ) -> list[dict]:
        """
        Fetch recent ticker news if endpoint is available.
        """
        ticker = ticker.upper()
        path = "/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": order,
        }

        data = self._get(path, params=params)
        results = data.get("results", []) if data else []
        return [normalize_news_item(item) for item in results]

    def get_snapshot(self, ticker: str) -> dict:
        """
        Fetch ticker snapshot if endpoint is available.
        """
        ticker = ticker.upper()
        path = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"

        data = self._get(path)
        if not data:
            return {}
        return data.get("ticker", data)

    def get_grouped_daily(
        self,
        day: str | date,
        adjusted: bool = True,
    ) -> list[dict]:
        """
        Fetch grouped daily market bars for a date.
        Useful for broad scanner universe fallback.
        """
        day_s = _date_str(day)
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{day_s}"
        params = {"adjusted": str(bool(adjusted)).lower()}

        data = self._get(path, params=params)
        results = data.get("results", []) if data else []
        return [normalize_grouped_bar(bar) for bar in results]

    def get_market_status(self) -> dict:
        """
        Fetch market status/clock when supported.
        """
        path = "/v1/marketstatus/now"
        data = self._get(path)
        return data or {}

    # ── Core HTTP helper ──────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """
        Execute a GET request with retries.

        Returns:
            Parsed JSON dict, or {} on failure.
        """
        if requests is None:
            log.error("[massive_rest] requests package unavailable")
            return {}

        if not self._api_key:
            log.warning("[massive_rest] API key missing")
            return {}

        params = dict(params or {})
        params["apiKey"] = self._api_key

        url = self._url(path, params)

        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.get(url, timeout=self._timeout)

                if resp.status_code == 200:
                    return resp.json()

                last_error = f"HTTP {resp.status_code}: {resp.text[:250]}"

                # Retry rate limits and temporary server issues.
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self._retry_sleep * attempt)
                    continue

                log.warning("[massive_rest] GET failed %s: %s", path, last_error)
                return {}

            except Exception as exc:
                last_error = str(exc)
                time.sleep(self._retry_sleep * attempt)

        log.error("[massive_rest] GET failed after retries %s: %s", path, last_error)
        return {}

    def _url(self, path: str, params: Optional[dict] = None) -> str:
        """Build full URL."""
        path = path if path.startswith("/") else f"/{path}"
        query = urlencode(params or {})
        return f"{self._base_url}{path}?{query}" if query else f"{self._base_url}{path}"


# ── Normalizers ───────────────────────────────────────────────────────────────

def normalize_aggregate_bar(bar: dict) -> dict:
    """
    Normalize aggregate bar into the bot's OHLCV style.

    Polygon aggregate keys:
      o, h, l, c, v, vw, t, n
    """
    ts = bar.get("t")
    return {
        "t": ts,
        "timestamp": _timestamp_to_iso(ts),
        "o": _safe_float(bar.get("o")),
        "h": _safe_float(bar.get("h")),
        "l": _safe_float(bar.get("l")),
        "c": _safe_float(bar.get("c")),
        "v": _safe_float(bar.get("v")),
        "vw": _safe_float(bar.get("vw")),
        "n": int(_safe_float(bar.get("n"))),
        "source": "massive_rest",
        "raw": dict(bar),
    }


def normalize_grouped_bar(bar: dict) -> dict:
    """
    Normalize grouped daily bar.
    """
    normalized = normalize_aggregate_bar(bar)
    normalized["ticker"] = str(bar.get("T", "")).upper()
    return normalized


def normalize_news_item(item: dict) -> dict:
    """
    Normalize news item into a stable dashboard/scoring shape.
    """
    return {
        "id": item.get("id", ""),
        "publisher": item.get("publisher", {}),
        "title": item.get("title", ""),
        "author": item.get("author", ""),
        "published_utc": item.get("published_utc", ""),
        "article_url": item.get("article_url", ""),
        "tickers": item.get("tickers", []),
        "description": item.get("description", ""),
        "source": "massive_rest",
        "raw": dict(item),
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def get_recent_bars(
    settings: dict,
    ticker: str,
    from_date: str | date,
    to_date: str | date,
    multiplier: int = 1,
    timespan: str = "minute",
) -> list[dict]:
    """
    Convenience function for market_data_provider.py or tests.
    """
    client = MassiveRestClient(settings)
    return client.get_aggregate_bars(
        ticker     = ticker,
        from_date  = from_date,
        to_date    = to_date,
        multiplier = multiplier,
        timespan   = timespan,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_str(value: str | date) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _timestamp_to_iso(ts: object) -> str:
    """
    Convert millisecond timestamp to ISO string.
    """
    try:
        if ts is None:
            return ""
        dt = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
