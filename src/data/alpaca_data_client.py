"""
src/data/alpaca_data_client.py — Alpaca Markets data API client
Handles all HTTP communication with the Alpaca Data API v2.
All other modules get market data through this client — never directly
via requests or the Alpaca SDK.

Responsibilities:
  - Fetch historical bars (1-min, 5-min, daily)
  - Fetch latest quote (bid/ask/spread)
  - Fetch latest trade (last price, size)
  - Fetch snapshot (bars + quote + trade in one call)
  - Fetch account information (for paper trading confirmation)
  - Rate-limit awareness and retry logic
  - Paper vs live endpoint routing
  - Environment variable loading for API credentials

Environment variables required (set in .env):
    ALPACA_API_KEY      — API key ID
    ALPACA_SECRET_KEY   — API secret key
    ALPACA_PAPER        — "true" to use paper endpoints (default: true)

All methods return plain dicts or lists of dicts matching the Alpaca
API response schema.  candle_builder.normalize_bar() converts them to
the bot's canonical bar schema.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ── API endpoints ─────────────────────────────────────────────────────────────

_DATA_BASE_URL   = "https://data.alpaca.markets/v2"
_BROKER_BASE_URL_PAPER = "https://paper-api.alpaca.markets/v2"
_BROKER_BASE_URL_LIVE  = "https://api.alpaca.markets/v2"

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT      = 10          # seconds per request
_DEFAULT_RETRIES      = 3
_DEFAULT_BACKOFF      = 0.5         # seconds between retries
_RATE_LIMIT_PAUSE     = 1.0         # pause after 429 response
_MAX_BARS_PER_REQUEST = 1000        # Alpaca API limit


# ── Client ────────────────────────────────────────────────────────────────────

class AlpacaDataClient:
    """
    Thin, authenticated HTTP client for the Alpaca Data API.

    Usage:
        client = AlpacaDataClient.from_env()
        bars   = client.get_bars("ABCD", timeframe="1Min", limit=60)
        quote  = client.get_latest_quote("ABCD")
    """

    def __init__(
        self,
        api_key:    str,
        secret_key: str,
        paper:      bool = True,
        timeout:    int  = _DEFAULT_TIMEOUT,
    ):
        if not api_key or not secret_key:
            raise ValueError(
                "[alpaca_data_client] API key and secret key are required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
            )

        self._api_key    = api_key
        self._secret_key = secret_key
        self._paper      = paper
        self._timeout    = timeout
        self._session    = self._build_session()

        broker = "paper" if paper else "live"
        log.info("[alpaca_data_client] Initialized — account mode: %s", broker)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, timeout: int = _DEFAULT_TIMEOUT) -> AlpacaDataClient:
        """
        Create a client from environment variables.
        Loads .env automatically if python-dotenv is installed.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass   # dotenv optional — env vars may already be set

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        paper_str  = os.environ.get("ALPACA_PAPER", "true").lower()
        paper      = paper_str in ("true", "1", "yes")

        if not api_key:
            log.error("[alpaca_data_client] ALPACA_API_KEY not set in environment")
        if not secret_key:
            log.error("[alpaca_data_client] ALPACA_SECRET_KEY not set in environment")

        return cls(api_key=api_key, secret_key=secret_key,
                   paper=paper, timeout=timeout)

    # ── Bar data ──────────────────────────────────────────────────────────────

    def get_bars(
        self,
        ticker:    str,
        timeframe: str = "1Min",
        limit:     int = 200,
        start:     Optional[datetime] = None,
        end:       Optional[datetime] = None,
        feed:      str = "iex",
    ) -> list[dict]:
        """
        Fetch historical OHLCV bars for a single ticker.

        Args:
            ticker:    Ticker symbol (e.g. "ABCD").
            timeframe: Alpaca timeframe string: "1Min", "5Min", "15Min",
                       "30Min", "1Hour", "1Day".
            limit:     Maximum number of bars to return (max 1000).
            start:     Start datetime (UTC-aware).  Defaults to today's open.
            end:       End datetime (UTC-aware).  Defaults to now.
            feed:      Data feed: "iex" (free) or "sip" (requires paid plan).

        Returns:
            List of raw Alpaca bar dicts with keys: t, o, h, l, c, v, vw, n.
            Empty list on error.
        """
        if not start:
            start = _today_open_utc()

        params = {
            "timeframe": timeframe,
            "limit":     min(limit, _MAX_BARS_PER_REQUEST),
            "start":     start.isoformat(),
            "feed":      feed,
            "sort":      "asc",
        }
        if end:
            params["end"] = end.isoformat()

        url  = f"{_DATA_BASE_URL}/stocks/{ticker}/bars"
        data = self._get(url, params)
        if not data:
            return []

        bars = data.get("bars", [])
        log.debug("[alpaca] %s bars fetched: %d (%s)", ticker, len(bars), timeframe)
        return bars

    def get_bars_multi(
        self,
        tickers:   list[str],
        timeframe: str = "1Min",
        limit:     int = 200,
        start:     Optional[datetime] = None,
        feed:      str = "iex",
    ) -> dict[str, list[dict]]:
        """
        Fetch bars for multiple tickers in a single API call.

        Returns:
            Dict mapping ticker → list of bar dicts.
        """
        if not tickers:
            return {}
        if not start:
            start = _today_open_utc()

        params = {
            "symbols":   ",".join(tickers),
            "timeframe": timeframe,
            "limit":     min(limit, _MAX_BARS_PER_REQUEST),
            "start":     start.isoformat(),
            "feed":      feed,
            "sort":      "asc",
        }

        url  = f"{_DATA_BASE_URL}/stocks/bars"
        data = self._get(url, params)
        if not data:
            return {t: [] for t in tickers}

        raw = data.get("bars", {})
        # Fill in missing tickers with empty lists
        return {t: raw.get(t, []) for t in tickers}

    # ── Latest quote ──────────────────────────────────────────────────────────

    def get_latest_quote(self, ticker: str, feed: str = "iex") -> Optional[dict]:
        """
        Fetch the latest bid/ask quote for a ticker.

        Returns a dict with keys:
            ap  — ask price
            as  — ask size
            bp  — bid price
            bs  — bid size
            t   — timestamp (ISO-8601)
            ax  — ask exchange
            bx  — bid exchange

        Returns None on error.
        """
        url  = f"{_DATA_BASE_URL}/stocks/{ticker}/quotes/latest"
        data = self._get(url, {"feed": feed})
        if not data:
            return None
        quote = data.get("quote")
        if quote:
            log.debug(
                "[alpaca] %s latest quote: bid=%.4f ask=%.4f",
                ticker, quote.get("bp", 0), quote.get("ap", 0),
            )
        return quote

    def get_latest_quotes_multi(
        self,
        tickers: list[str],
        feed:    str = "iex",
    ) -> dict[str, Optional[dict]]:
        """Fetch latest quotes for multiple tickers in one call."""
        if not tickers:
            return {}
        url  = f"{_DATA_BASE_URL}/stocks/quotes/latest"
        data = self._get(url, {"symbols": ",".join(tickers), "feed": feed})
        if not data:
            return {t: None for t in tickers}
        raw = data.get("quotes", {})
        return {t: raw.get(t) for t in tickers}

    # ── Latest trade ──────────────────────────────────────────────────────────

    def get_latest_trade(self, ticker: str, feed: str = "iex") -> Optional[dict]:
        """
        Fetch the most recent trade for a ticker.

        Returns a dict with keys:
            p  — price
            s  — size
            t  — timestamp
            x  — exchange

        Returns None on error.
        """
        url  = f"{_DATA_BASE_URL}/stocks/{ticker}/trades/latest"
        data = self._get(url, {"feed": feed})
        if not data:
            return None
        return data.get("trade")

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def get_snapshot(self, ticker: str, feed: str = "iex") -> Optional[dict]:
        """
        Fetch a full snapshot for a ticker: latest bar, quote, trade,
        minute bar, daily bar, and previous daily bar.

        This is the most efficient single call for getting current price
        context.  Returns None on error.
        """
        url  = f"{_DATA_BASE_URL}/stocks/{ticker}/snapshot"
        data = self._get(url, {"feed": feed})
        if not data:
            return None
        log.debug("[alpaca] %s snapshot fetched", ticker)
        return data

    def get_snapshots_multi(
        self,
        tickers: list[str],
        feed:    str = "iex",
    ) -> dict[str, Optional[dict]]:
        """Fetch snapshots for multiple tickers in one call."""
        if not tickers:
            return {}
        url  = f"{_DATA_BASE_URL}/stocks/snapshots"
        data = self._get(url, {"symbols": ",".join(tickers), "feed": feed})
        if not data:
            return {t: None for t in tickers}
        return {t: data.get(t) for t in tickers}

    # ── Account (broker API) ──────────────────────────────────────────────────

    def get_account(self) -> Optional[dict]:
        """
        Fetch Alpaca account information.

        Key fields in the returned dict:
            id               — account UUID
            account_number   — account number string
            status           — "ACTIVE" when ready to trade
            buying_power     — available buying power
            cash             — cash balance
            portfolio_value  — total portfolio value
            paper_trading    — True for paper accounts (Alpaca adds this field)

        Returns None on error.
        """
        base = _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE
        url  = f"{base}/account"
        data = self._get(url, {})
        if data:
            log.info(
                "[alpaca] Account fetched: status=%s buying_power=%s",
                data.get("status"), data.get("buying_power"),
            )
        return data

    def get_positions(self) -> list[dict]:
        """
        Fetch all open positions.

        Returns a list of position dicts with keys:
            symbol        — ticker
            qty           — shares held (string)
            avg_entry_price
            current_price
            unrealized_pl
            side          — "long" or "short"
        """
        base = _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE
        url  = f"{base}/positions"
        data = self._get(url, {})
        if data is None:
            return []
        # Alpaca returns a list directly for positions
        return data if isinstance(data, list) else []

    def get_position(self, ticker: str) -> Optional[dict]:
        """Fetch a single open position by ticker.  Returns None if no position."""
        base = _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE
        url  = f"{base}/positions/{ticker}"
        return self._get(url, {})

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_orders(self, status: str = "open") -> list[dict]:
        """
        Fetch orders filtered by status: "open", "closed", "all".
        Returns a list of order dicts.
        """
        base = _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE
        url  = f"{base}/orders"
        data = self._get(url, {"status": status})
        if data is None:
            return []
        return data if isinstance(data, list) else []

    def get_order(self, order_id: str) -> Optional[dict]:
        """Fetch a single order by ID."""
        base = _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE
        url  = f"{base}/orders/{order_id}"
        return self._get(url, {})

    # ── Market clock ─────────────────────────────────────────────────────────

    def get_clock(self) -> Optional[dict]:
        """
        Fetch the Alpaca market clock.

        Returns a dict with:
            is_open    — True when market is open
            next_open  — ISO-8601 string of next open time
            next_close — ISO-8601 string of next close time
            timestamp  — current server time
        """
        base = _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE
        url  = f"{base}/clock"
        return self._get(url, {})

    def is_market_open(self) -> bool:
        """True when Alpaca reports the market is currently open."""
        clock = self.get_clock()
        return bool(clock and clock.get("is_open", False))

    # ── HTTP layer ────────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict) -> Optional[dict | list]:
        """
        Execute an authenticated GET request with retry logic.

        Returns the parsed JSON body on success, or None on failure.
        Handles 429 rate limit responses with an automatic pause.
        """
        headers = {
            "APCA-API-KEY-ID":     self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept":              "application/json",
        }

        for attempt in range(1, _DEFAULT_RETRIES + 1):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    log.warning(
                        "[alpaca] Rate limited (429) on %s — pausing %.1fs",
                        url, _RATE_LIMIT_PAUSE,
                    )
                    time.sleep(_RATE_LIMIT_PAUSE)
                    continue

                if resp.status_code == 404:
                    log.debug("[alpaca] 404 Not Found: %s", url)
                    return None

                if resp.status_code in (401, 403):
                    log.error(
                        "[alpaca] Auth error %d on %s — check API credentials",
                        resp.status_code, url,
                    )
                    return None

                log.warning(
                    "[alpaca] HTTP %d on %s (attempt %d/%d): %s",
                    resp.status_code, url, attempt, _DEFAULT_RETRIES,
                    resp.text[:200],
                )

            except requests.exceptions.Timeout:
                log.warning(
                    "[alpaca] Timeout on %s (attempt %d/%d)",
                    url, attempt, _DEFAULT_RETRIES,
                )
            except requests.exceptions.ConnectionError as exc:
                log.warning(
                    "[alpaca] Connection error on %s (attempt %d/%d): %s",
                    url, attempt, _DEFAULT_RETRIES, exc,
                )
            except Exception as exc:
                log.error("[alpaca] Unexpected error on %s: %s", url, exc)
                return None

            if attempt < _DEFAULT_RETRIES:
                pause = _DEFAULT_BACKOFF * attempt
                log.debug("[alpaca] Retrying in %.1fs...", pause)
                time.sleep(pause)

        log.error("[alpaca] All %d attempts failed for %s", _DEFAULT_RETRIES, url)
        return None

    def _build_session(self) -> requests.Session:
        """Build a requests Session with connection pooling and retry adapter."""
        session = requests.Session()
        retry = Retry(
            total=0,           # we handle retries manually in _get()
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://",  adapter)
        return session

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_paper(self) -> bool:
        return self._paper

    @property
    def data_base_url(self) -> str:
        return _DATA_BASE_URL

    @property
    def broker_base_url(self) -> str:
        return _BROKER_BASE_URL_PAPER if self._paper else _BROKER_BASE_URL_LIVE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today_open_utc() -> datetime:
    """Return today's 09:30 ET as a UTC-aware datetime."""
    from zoneinfo import ZoneInfo
    ET    = ZoneInfo("America/New_York")
    now   = datetime.now(ET)
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return open_.astimezone(timezone.utc)


def extract_price_from_snapshot(snapshot: dict) -> Optional[float]:
    """
    Pull the best available price from a snapshot dict.
    Priority: latest trade → latest quote mid → minute bar close.
    """
    if not snapshot:
        return None

    trade = snapshot.get("latestTrade") or snapshot.get("latest_trade")
    if trade and trade.get("p"):
        return float(trade["p"])

    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote")
    if quote:
        bid = quote.get("bp", 0)
        ask = quote.get("ap", 0)
        if bid and ask:
            return (float(bid) + float(ask)) / 2

    minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar")
    if minute_bar and minute_bar.get("c"):
        return float(minute_bar["c"])

    return None


def extract_spread_from_quote(quote: dict) -> tuple[float, float, float]:
    """
    Extract bid, ask, and spread percentage from a quote dict.

    Returns:
        (bid, ask, spread_pct)  — all 0.0 on failure
    """
    if not quote:
        return 0.0, 0.0, 0.0
    bid = float(quote.get("bp", 0) or 0)
    ask = float(quote.get("ap", 0) or 0)
    if ask > 0:
        spread_pct = (ask - bid) / ask * 100
    else:
        spread_pct = 0.0
    return bid, ask, round(spread_pct, 4)


def build_day_change_pct(snapshot: dict) -> float:
    """
    Compute today's percentage price change using snapshot data.
    Uses: (latest price - prev_daily_close) / prev_daily_close * 100
    Returns 0.0 when data is unavailable.
    """
    if not snapshot:
        return 0.0

    current = extract_price_from_snapshot(snapshot)
    prev    = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar")

    if not current or not prev:
        return 0.0

    prev_close = float(prev.get("c", 0) or 0)
    if prev_close == 0:
        return 0.0

    return round((current - prev_close) / prev_close * 100, 4)
