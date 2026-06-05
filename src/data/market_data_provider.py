"""
src/data/market_data_provider.py — Central market data coordinator
The single source of truth for price and bar data inside the bot.
All scanners, analyzers, and scoring engines request data through this
provider — they never call AlpacaDataClient directly.

Responsibilities:
  - Per-ticker bar cache (avoids redundant API calls each loop)
  - Session-aware bar management (clear on new day)
  - Unified ticker data bundle (bars + quote + indicators + session context)
  - Staleness detection and forced refresh
  - Quote freshness validation before trade decisions
  - Day-level stats (day high/low, premarket high/low, day change %)

Data bundle schema (TickerData dataclass):
    ticker          str
    bars            list[dict]          — normalized session bars
    quote           dict | None         — latest bid/ask
    latest_trade    dict | None         — latest trade
    snapshot        dict | None         — full Alpaca snapshot
    indicators      IndicatorSnapshot   — computed indicators
    day_high        float | None
    day_low         float | None
    premarket_high  float | None
    premarket_low   float | None
    day_change_pct  float
    relative_volume float
    last_updated    str                 — ISO-8601 UTC
    is_fresh        bool
    errors          list[str]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from alpaca_data_client import (
    AlpacaDataClient,
    build_day_change_pct,
    extract_price_from_snapshot,
    extract_spread_from_quote,
)
from candle_builder import (
    CandleBuilder,
    day_high,
    day_low,
    normalize_bars,
    premarket_high,
    premarket_low,
)
from indicator_engine import IndicatorCache, compute
from models import IndicatorSnapshot
from time_utils import now_et, is_same_session

log = logging.getLogger(__name__)

# How old a quote can be before we consider it stale (seconds)
_DEFAULT_QUOTE_MAX_AGE = 20
# How many 1-min bars to fetch per ticker on refresh
_DEFAULT_BAR_LIMIT = 390   # full regular session


# ── Ticker Data Bundle ────────────────────────────────────────────────────────

@dataclass
class TickerData:
    """
    All market data the bot needs for one ticker, in one place.
    Produced by MarketDataProvider.get() and consumed by every module
    that needs current price context.
    """
    ticker:         str

    # Raw data
    bars:           list[dict]          = field(default_factory=list)
    quote:          Optional[dict]      = None
    latest_trade:   Optional[dict]      = None
    snapshot:       Optional[dict]      = None

    # Derived
    indicators:     IndicatorSnapshot   = field(default_factory=IndicatorSnapshot)
    day_high:       Optional[float]     = None
    day_low:        Optional[float]     = None
    premarket_high: Optional[float]     = None
    premarket_low:  Optional[float]     = None
    day_change_pct: float               = 0.0
    relative_volume: float              = 0.0

    # Quote fields (flattened for easy access)
    bid:            float               = 0.0
    ask:            float               = 0.0
    spread_pct:     float               = 0.0
    latest_price:   float               = 0.0

    # Status
    last_updated:   str                 = ""
    is_fresh:       bool                = False
    errors:         list[str]           = field(default_factory=list)

    def has_error(self) -> bool:
        return len(self.errors) > 0

    def is_tradable(self, max_spread_pct: float = 3.0) -> tuple[bool, str]:
        """
        Quick tradability check.
        Returns (True, "ok") or (False, reason).
        """
        if not self.is_fresh:
            return False, "data is stale"
        if self.latest_price <= 0:
            return False, "no valid price"
        if self.spread_pct > max_spread_pct:
            return False, f"spread too wide ({self.spread_pct:.2f}%)"
        if len(self.bars) < 30:
            return False, f"insufficient bars ({len(self.bars)})"
        return True, "ok"

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "latest_price":    self.latest_price,
            "bid":             self.bid,
            "ask":             self.ask,
            "spread_pct":      self.spread_pct,
            "day_change_pct":  self.day_change_pct,
            "relative_volume": self.relative_volume,
            "day_high":        self.day_high,
            "day_low":         self.day_low,
            "premarket_high":  self.premarket_high,
            "premarket_low":   self.premarket_low,
            "bar_count":       len(self.bars),
            "is_fresh":        self.is_fresh,
            "last_updated":    self.last_updated,
            "errors":          self.errors,
        }


# ── Provider ──────────────────────────────────────────────────────────────────

class MarketDataProvider:
    """
    Central data coordinator.  One instance lives for the lifetime of the bot.

    Usage:
        provider = MarketDataProvider(client, settings)
        data     = provider.get("ABCD")

        if data.is_fresh and not data.has_error():
            # use data.bars, data.indicators, data.quote, etc.
    """

    def __init__(self, client: AlpacaDataClient, settings: dict):
        self._client   = client
        self._settings = settings
        self._cfg      = settings.get("execution", {})
        self._ind_cfg  = settings.get("indicators", {})

        # Per-ticker bar builders (keyed by ticker)
        self._builders: dict[str, CandleBuilder]   = {}
        # Per-ticker last fetch timestamp
        self._last_fetch: dict[str, datetime]       = {}
        # Per-ticker last known session date (for reset detection)
        self._session_dates: dict[str, str]         = {}
        # Indicator cache
        self._indicator_cache = IndicatorCache()

        # Quote max age from config
        self._quote_max_age = int(
            self._cfg.get("max_quote_age_seconds", _DEFAULT_QUOTE_MAX_AGE)
        )
        self._bar_limit = _DEFAULT_BAR_LIMIT

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, ticker: str, force_refresh: bool = False) -> TickerData:
        """
        Return a fully populated TickerData bundle for a ticker.

        Fetches from the API when:
          - The ticker has never been fetched
          - force_refresh is True
          - The session has changed (new day)

        Otherwise returns from cache.

        Args:
            ticker:        Ticker symbol.
            force_refresh: Skip cache and always hit the API.

        Returns:
            TickerData bundle.
        """
        ticker = ticker.upper()
        errors: list[str] = []

        self._maybe_reset_session(ticker)

        needs_fetch = (
            force_refresh
            or ticker not in self._builders
            or not self._builders[ticker].bars()
        )

        if needs_fetch:
            self._fetch_bars(ticker, errors)

        # Always fetch a fresh quote and snapshot
        quote    = self._fetch_quote(ticker, errors)
        snapshot = self._fetch_snapshot(ticker, errors)
        trade    = snapshot.get("latestTrade") if snapshot else None

        builder = self._builders.get(ticker, CandleBuilder())
        bars    = builder.bars()

        # Compute indicators (uses cache when bar count unchanged)
        indicators = self._indicator_cache.get_or_compute(
            ticker, bars, self._settings
        )

        # Flatten quote fields
        bid, ask, spread_pct = extract_spread_from_quote(quote or {})
        latest_price = (
            extract_price_from_snapshot(snapshot)
            or (ask + bid) / 2 if (ask and bid) else 0.0
        )

        # Day stats
        d_high  = day_high(bars)
        d_low   = day_low(bars)
        pm_high = premarket_high(self._all_bars(ticker))
        pm_low  = premarket_low(self._all_bars(ticker))
        chg_pct = build_day_change_pct(snapshot or {})
        rvol    = indicators.relative_volume

        # Freshness: quote must be recent
        is_fresh = self._is_quote_fresh(quote)

        data = TickerData(
            ticker          = ticker,
            bars            = bars,
            quote           = quote,
            latest_trade    = trade,
            snapshot        = snapshot,
            indicators      = indicators,
            day_high        = d_high,
            day_low         = d_low,
            premarket_high  = pm_high,
            premarket_low   = pm_low,
            day_change_pct  = chg_pct,
            relative_volume = rvol,
            bid             = bid,
            ask             = ask,
            spread_pct      = spread_pct,
            latest_price    = float(latest_price) if latest_price else 0.0,
            last_updated    = _now_iso(),
            is_fresh        = is_fresh,
            errors          = errors,
        )

        log.debug(
            "[data] %s price=%.4f spread=%.2f%% bars=%d fresh=%s",
            ticker, data.latest_price, data.spread_pct,
            len(bars), data.is_fresh,
        )
        return data

    def get_batch(
        self,
        tickers: list[str],
        force_refresh: bool = False,
    ) -> dict[str, TickerData]:
        """
        Fetch data for multiple tickers.  Uses multi-bar fetch for efficiency
        when all tickers need a refresh, then builds individual bundles.

        Returns:
            Dict mapping ticker → TickerData.
        """
        results: dict[str, TickerData] = {}

        # Identify tickers that need a bar refresh
        needs_refresh = [
            t for t in tickers
            if force_refresh
            or t not in self._builders
            or not self._builders[t].bars()
        ]

        # Bulk bar fetch for efficiency
        if needs_refresh:
            self._fetch_bars_multi(needs_refresh)

        for ticker in tickers:
            results[ticker] = self.get(ticker, force_refresh=False)

        return results

    def invalidate(self, ticker: str) -> None:
        """Clear cached data for a ticker so the next get() hits the API."""
        self._builders.pop(ticker, None)
        self._last_fetch.pop(ticker, None)
        self._session_dates.pop(ticker, None)
        self._indicator_cache.invalidate(ticker)
        log.debug("[data] Cache invalidated for %s", ticker)

    def invalidate_all(self) -> None:
        """Clear all cached data."""
        self._builders.clear()
        self._last_fetch.clear()
        self._session_dates.clear()
        self._indicator_cache.invalidate_all()
        log.info("[data] All ticker caches cleared")

    def cached_tickers(self) -> list[str]:
        return list(self._builders.keys())

    # ── Internal — bar fetching ───────────────────────────────────────────────

    def _fetch_bars(self, ticker: str, errors: list[str]) -> None:
        """Fetch 1-min bars for a single ticker and store in the builder."""
        try:
            raw = self._client.get_bars(
                ticker,
                timeframe="1Min",
                limit=self._bar_limit,
            )
            self._ingest_bars(ticker, raw)
            self._last_fetch[ticker] = datetime.now(timezone.utc)
        except Exception as exc:
            msg = f"bar fetch failed for {ticker}: {exc}"
            errors.append(msg)
            log.error("[data] %s", msg)

    def _fetch_bars_multi(self, tickers: list[str]) -> None:
        """Bulk bar fetch for multiple tickers."""
        try:
            raw_map = self._client.get_bars_multi(
                tickers,
                timeframe="1Min",
                limit=self._bar_limit,
            )
            for ticker, raw in raw_map.items():
                self._ingest_bars(ticker, raw)
                self._last_fetch[ticker] = datetime.now(timezone.utc)
        except Exception as exc:
            log.error("[data] bulk bar fetch failed: %s", exc)

    def _ingest_bars(self, ticker: str, raw: list[dict]) -> None:
        """Normalize and store raw Alpaca bars for a ticker."""
        if ticker not in self._builders:
            self._builders[ticker] = CandleBuilder()

        builder = self._builders[ticker]
        builder.clear()

        if raw:
            count = builder.ingest_bars(raw, session_filter="all")
            log.debug("[data] %s: ingested %d bars", ticker, count)
        else:
            log.warning("[data] %s: no bars returned from API", ticker)

    # ── Internal — quote / snapshot fetching ──────────────────────────────────

    def _fetch_quote(self, ticker: str, errors: list[str]) -> Optional[dict]:
        try:
            return self._client.get_latest_quote(ticker)
        except Exception as exc:
            msg = f"quote fetch failed for {ticker}: {exc}"
            errors.append(msg)
            log.warning("[data] %s", msg)
            return None

    def _fetch_snapshot(self, ticker: str, errors: list[str]) -> Optional[dict]:
        try:
            return self._client.get_snapshot(ticker)
        except Exception as exc:
            msg = f"snapshot fetch failed for {ticker}: {exc}"
            errors.append(msg)
            log.warning("[data] %s", msg)
            return None

    # ── Internal — session management ─────────────────────────────────────────

    def _maybe_reset_session(self, ticker: str) -> None:
        """
        Detect a new trading day for a ticker and clear its bar cache.
        VWAP is session-anchored so we must clear bars on each new day.
        """
        today = now_et().date().isoformat()
        if self._session_dates.get(ticker) != today:
            if ticker in self._session_dates:
                log.info(
                    "[data] New session detected for %s — clearing bar cache", ticker
                )
                self.invalidate(ticker)
            self._session_dates[ticker] = today

    # ── Internal — all bars (including pre-market) ────────────────────────────

    def _all_bars(self, ticker: str) -> list[dict]:
        """Return all bars (all sessions) for a ticker."""
        builder = self._builders.get(ticker)
        return builder.bars() if builder else []

    # ── Internal — freshness ──────────────────────────────────────────────────

    def _is_quote_fresh(self, quote: Optional[dict]) -> bool:
        """
        True when the quote timestamp is within the configured max age.
        Falls back to False when the quote is None or has no timestamp.
        """
        if not quote:
            return False
        ts_str = quote.get("t") or quote.get("timestamp") or ""
        if not ts_str:
            return False
        try:
            from datetime import datetime as dt
            ts = dt.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return 0 <= age <= self._quote_max_age
        except (ValueError, TypeError):
            return False


# ── Module-level helper ───────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
