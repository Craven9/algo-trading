"""
src/data/news_catalyst_provider.py — News and catalyst detection
Fetches and evaluates news headlines for scanner candidates.
The bot uses catalyst presence as a scoring signal — not as a buy trigger.

Responsibilities:
  - Fetch recent news headlines for a ticker via Alpaca News API
  - Classify catalyst strength (earnings, FDA, halt, merger, etc.)
  - Cache results to avoid hammering the API each loop
  - Provide a CatalystResult that scoring engines can weight

Design rules:
  - News is a BONUS signal, not a requirement
  - A ticker with no news is still tradable if setup/probability pass
  - A ticker with strong catalyst gets a small score bonus
  - Catalyst data is treated as soft evidence — never hard-blocks a trade
  - Results expire after a configurable TTL (default: 5 minutes)

Catalyst strength labels:
  "strong"   — earnings, FDA approval/rejection, halt, merger/acquisition,
                major partnership, short squeeze news
  "moderate" — analyst upgrade/downgrade, secondary offering,
                contract win, product launch
  "weak"     — general mentions, social media buzz, sector news
  "none"     — no relevant news found in the lookback window
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Catalyst classification keywords ─────────────────────────────────────────

_STRONG_KEYWORDS = [
    "earnings", "beat", "miss", "revenue", "guidance", "fda", "approval",
    "approved", "rejected", "halt", "halted", "merger", "acquisition",
    "acquired", "buyout", "short squeeze", "squeeze", "partnership",
    "contract", "awarded", "breakthrough", "phase 3", "phase iii",
    "clinical trial", "sec", "subpoena", "investigation", "bankruptcy",
    "chapter 11", "delisted", "nasdaq notice",
]

_MODERATE_KEYWORDS = [
    "upgrade", "downgrade", "raised", "lowered", "price target",
    "offering", "secondary", "dilution", "launch", "product",
    "deal", "agreement", "signed", "expanded", "strategic",
    "insider", "bought", "sold",
]

_WEAK_KEYWORDS = [
    "mentions", "trending", "unusual", "momentum", "social",
    "retail", "reddit", "wallstreetbets", "options", "volume",
]

# ── Cache TTL ─────────────────────────────────────────────────────────────────

_DEFAULT_CACHE_TTL_SECONDS = 300    # 5 minutes
_DEFAULT_LOOKBACK_HOURS    = 24     # how far back to search for news
_DEFAULT_MAX_HEADLINES     = 10     # max headlines to fetch per ticker

# Alpaca news API endpoint
_NEWS_BASE_URL = "https://data.alpaca.markets/v1beta1/news"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CatalystResult:
    """
    Catalyst assessment for a single ticker.
    Produced by NewsCatalystProvider.get() and consumed by:
      - move_potential_engine.py  (score bonus)
      - probability_engine.py     (score bonus)
      - scanner candidate ranker  (soft ranking boost)
    """
    ticker:           str
    has_catalyst:     bool          = False
    strength:         str           = "none"    # "strong" | "moderate" | "weak" | "none"
    score_bonus:      float         = 0.0       # points added to scoring engines
    headline_count:   int           = 0
    top_headline:     str           = ""
    keywords_matched: list[str]     = field(default_factory=list)
    headlines:        list[dict]    = field(default_factory=list)
    source:           str           = "alpaca_news"
    fetched_at:       str           = ""
    error:            str           = ""

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "has_catalyst":     self.has_catalyst,
            "strength":         self.strength,
            "score_bonus":      self.score_bonus,
            "headline_count":   self.headline_count,
            "top_headline":     self.top_headline,
            "keywords_matched": self.keywords_matched,
            "fetched_at":       self.fetched_at,
            "error":            self.error,
        }

    @classmethod
    def empty(cls, ticker: str, reason: str = "") -> CatalystResult:
        """Return a no-catalyst result for a ticker."""
        return cls(
            ticker     = ticker,
            has_catalyst = False,
            strength   = "none",
            score_bonus = 0.0,
            fetched_at = _now_iso(),
            error      = reason,
        )


# ── Provider ──────────────────────────────────────────────────────────────────

class NewsCatalystProvider:
    """
    Fetches and classifies news catalysts for scanner candidates.

    Usage:
        provider = NewsCatalystProvider.from_env()
        result   = provider.get("ABCD")

        if result.has_catalyst:
            print(result.strength, result.top_headline)
    """

    def __init__(
        self,
        api_key:          str,
        secret_key:       str,
        cache_ttl:        int   = _DEFAULT_CACHE_TTL_SECONDS,
        lookback_hours:   int   = _DEFAULT_LOOKBACK_HOURS,
        max_headlines:    int   = _DEFAULT_MAX_HEADLINES,
        enabled:          bool  = True,
    ):
        self._api_key       = api_key
        self._secret_key    = secret_key
        self._cache_ttl     = cache_ttl
        self._lookback_hrs  = lookback_hours
        self._max_headlines = max_headlines
        self._enabled       = enabled

        # ticker → (CatalystResult, fetched_timestamp)
        self._cache: dict[str, tuple[CatalystResult, float]] = {}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, settings: Optional[dict] = None) -> NewsCatalystProvider:
        """
        Create from environment variables and optional settings dict.
        Falls back to no-op mode when credentials are missing.
        """
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        enabled    = bool(api_key and secret_key)

        cfg = (settings or {}).get("learning", {})   # no dedicated news section
        ttl = int((settings or {}).get("scanner", {}).get(
            "news_cache_ttl_seconds", _DEFAULT_CACHE_TTL_SECONDS))

        if not enabled:
            log.warning(
                "[news] API credentials missing — catalyst provider disabled"
            )

        return cls(
            api_key      = api_key,
            secret_key   = secret_key,
            cache_ttl    = ttl,
            enabled      = enabled,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, ticker: str) -> CatalystResult:
        """
        Return a CatalystResult for a ticker.
        Uses the cache when the result is still fresh.

        Args:
            ticker: Ticker symbol (e.g. "ABCD").

        Returns:
            CatalystResult — never raises.
        """
        ticker = ticker.upper()

        if not self._enabled:
            return CatalystResult.empty(ticker, "news provider disabled")

        # Check cache
        cached = self._cache.get(ticker)
        if cached:
            result, fetched_ts = cached
            if time.monotonic() - fetched_ts < self._cache_ttl:
                log.debug("[news] %s — cache hit (strength=%s)", ticker, result.strength)
                return result

        # Fetch fresh
        result = self._fetch(ticker)
        self._cache[ticker] = (result, time.monotonic())
        return result

    def get_batch(self, tickers: list[str]) -> dict[str, CatalystResult]:
        """
        Fetch catalyst results for multiple tickers.
        Each ticker is fetched individually (Alpaca news API is per-symbol).
        """
        return {t: self.get(t) for t in tickers}

    def invalidate(self, ticker: str) -> None:
        """Remove a ticker from the cache."""
        self._cache.pop(ticker.upper(), None)

    def invalidate_all(self) -> None:
        """Clear all cached catalyst results."""
        self._cache.clear()

    # ── Fetch and classify ────────────────────────────────────────────────────

    def _fetch(self, ticker: str) -> CatalystResult:
        """Fetch news from Alpaca and classify catalyst strength."""
        start = (
            datetime.now(timezone.utc) - timedelta(hours=self._lookback_hrs)
        ).isoformat()

        params = {
            "symbols":    ticker,
            "start":      start,
            "limit":      self._max_headlines,
            "sort":       "desc",
            "include_content": False,
        }
        headers = {
            "APCA-API-KEY-ID":     self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept":              "application/json",
        }

        try:
            resp = requests.get(
                _NEWS_BASE_URL,
                params=params,
                headers=headers,
                timeout=8,
            )
        except requests.exceptions.RequestException as exc:
            log.warning("[news] %s fetch failed: %s", ticker, exc)
            return CatalystResult.empty(ticker, str(exc))

        if resp.status_code == 401:
            log.error("[news] Auth error — check ALPACA_API_KEY / ALPACA_SECRET_KEY")
            return CatalystResult.empty(ticker, "auth error")

        if resp.status_code == 429:
            log.warning("[news] Rate limited — returning empty catalyst")
            return CatalystResult.empty(ticker, "rate limited")

        if resp.status_code != 200:
            log.warning("[news] %s HTTP %d", ticker, resp.status_code)
            return CatalystResult.empty(ticker, f"HTTP {resp.status_code}")

        try:
            data = resp.json()
        except Exception:
            return CatalystResult.empty(ticker, "JSON parse error")

        headlines = data.get("news", [])
        return self._classify(ticker, headlines)

    def _classify(self, ticker: str, headlines: list[dict]) -> CatalystResult:
        """
        Classify a list of raw Alpaca news items into a CatalystResult.

        Scoring logic:
          - Any strong keyword match → "strong"
          - Only moderate keyword matches → "moderate"
          - Only weak keyword matches → "weak"
          - No matches → "none"

        Score bonus:
          strong   → +5.0 points
          moderate → +3.0 points
          weak     → +1.0 points
          none     → +0.0 points
        """
        if not headlines:
            return CatalystResult(
                ticker       = ticker,
                has_catalyst = False,
                strength     = "none",
                score_bonus  = 0.0,
                headline_count = 0,
                fetched_at   = _now_iso(),
            )

        matched_strong   : list[str] = []
        matched_moderate : list[str] = []
        matched_weak     : list[str] = []
        top_headline = ""

        for item in headlines:
            title = (item.get("headline") or item.get("title") or "").lower()
            summary = (item.get("summary") or "").lower()
            text = f"{title} {summary}"

            if not top_headline and (item.get("headline") or item.get("title")):
                top_headline = item.get("headline") or item.get("title") or ""

            for kw in _STRONG_KEYWORDS:
                if kw in text and kw not in matched_strong:
                    matched_strong.append(kw)

            for kw in _MODERATE_KEYWORDS:
                if kw in text and kw not in matched_moderate:
                    matched_moderate.append(kw)

            for kw in _WEAK_KEYWORDS:
                if kw in text and kw not in matched_weak:
                    matched_weak.append(kw)

        # Determine strength
        if matched_strong:
            strength    = "strong"
            score_bonus = 5.0
            keywords    = matched_strong
        elif matched_moderate:
            strength    = "moderate"
            score_bonus = 3.0
            keywords    = matched_moderate
        elif matched_weak:
            strength    = "weak"
            score_bonus = 1.0
            keywords    = matched_weak
        else:
            strength    = "none"
            score_bonus = 0.0
            keywords    = []

        has_catalyst = strength != "none"

        log.info(
            "[news] %s — strength=%s bonus=%.1f keywords=%s headlines=%d",
            ticker, strength, score_bonus, keywords[:3], len(headlines),
        )

        return CatalystResult(
            ticker           = ticker,
            has_catalyst     = has_catalyst,
            strength         = strength,
            score_bonus      = score_bonus,
            headline_count   = len(headlines),
            top_headline     = top_headline,
            keywords_matched = keywords,
            headlines        = headlines,
            fetched_at       = _now_iso(),
        )


# ── Standalone classifier (no API) ───────────────────────────────────────────

def classify_headlines(ticker: str, headlines: list[dict]) -> CatalystResult:
    """
    Classify pre-fetched headlines without making any API calls.
    Useful when news is sourced from a different provider or cached externally.

    Args:
        ticker:    Ticker symbol.
        headlines: List of dicts with at least a "headline" or "title" key.

    Returns:
        CatalystResult.
    """
    provider = NewsCatalystProvider(
        api_key="", secret_key="", enabled=False
    )
    return provider._classify(ticker, headlines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
