"""
src/scanner/momentum_scanner.py — Momentum stock scanner
Finds high-momentum candidates from the market universe.
Outputs ONLY candidates — never buy signals.

Responsibilities:
  - Screen stocks for high relative volume, price move, and liquidity
  - Apply configurable filters (min RVOL, min change %, max spread, etc.)
  - Rank candidates by momentum quality
  - Return ScannerCandidate objects for the analysis pipeline

Design rules:
  - Scanner output is a candidate list, NEVER a buy signal
  - Every candidate must pass ALL hard filters before being returned
  - Soft filters reduce score but don't block
  - The scanner has no knowledge of setups, scoring, or risk
  - Candidates are re-evaluated every scan loop — nothing is assumed

Scanner flow:
  1. Fetch snapshots for the scan universe (Alpaca snapshot API)
  2. Apply hard filters (price, volume, spread, change %)
  3. Score each passing ticker by momentum quality
  4. Rank and return top N candidates

Output schema (ScannerCandidate):
  ticker, source, price, bid, ask, spread_percent, relative_volume,
  day_change_pct, dollar_volume, candidate_reason, scanned_at
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
from models import CandidateSource, ScannerCandidate

log = logging.getLogger(__name__)

# ── Default filter thresholds (overridden by bot_settings.json → scanner) ─────

_DEFAULTS = {
    "min_relative_volume":   3.0,
    "min_day_change_percent":10.0,
    "max_spread_percent":    3.0,
    "min_dollar_volume":     500_000,
    "max_price":             50.0,
    "min_price":             0.50,
    "max_candidates_per_loop": 20,
    "exclude_otc":           True,
    "exclude_etfs":          False,
}

# Minimum average volume to be considered liquid
_MIN_AVG_VOLUME = 100_000


# ── Candidate scoring weights ─────────────────────────────────────────────────
# Used to rank candidates after hard filters pass.
# Higher score = better candidate (more likely to be analyzed further).

_SCORE_WEIGHTS = {
    "rvol":       40,   # relative volume is the strongest signal
    "change_pct": 30,   # price move magnitude
    "spread":     15,   # tight spread = better execution quality
    "dollar_vol": 15,   # higher dollar volume = more liquid
}


# ── Scanner ───────────────────────────────────────────────────────────────────

class MomentumScanner:
    """
    Scans a universe of tickers for high-momentum candidates.

    Usage:
        scanner    = MomentumScanner(client, settings)
        candidates = scanner.scan(universe)
    """

    def __init__(self, client: AlpacaDataClient, settings: dict):
        self._client   = client
        self._settings = settings
        self._cfg      = {**_DEFAULTS, **settings.get("scanner", {})}
        self._last_scan_count = 0
        self._last_scan_time  = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, universe: list[str]) -> list[ScannerCandidate]:
        """
        Scan a list of tickers and return qualified momentum candidates.

        Args:
            universe: List of ticker symbols to evaluate.

        Returns:
            Ranked list of ScannerCandidate objects (best first).
            Empty list when scanner is disabled or no candidates qualify.
        """
        if not self._cfg.get("enabled", True):
            log.info("[scanner] Scanner is disabled in settings")
            return []

        if not universe:
            log.debug("[scanner] Empty universe — nothing to scan")
            return []

        log.info("[scanner] Scanning %d tickers...", len(universe))

        # Fetch snapshots in batches
        snapshots = self._fetch_snapshots(universe)
        if not snapshots:
            log.warning("[scanner] No snapshot data returned")
            return []

        # Evaluate each ticker
        scored: list[tuple[float, ScannerCandidate]] = []
        for ticker, snapshot in snapshots.items():
            if not snapshot:
                continue
            result = self._evaluate(ticker, snapshot)
            if result is not None:
                score, candidate = result
                scored.append((score, candidate))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        max_candidates = int(self._cfg.get("max_candidates_per_loop", 20))
        candidates = [c for _, c in scored[:max_candidates]]

        self._last_scan_count = len(candidates)
        self._last_scan_time  = _now_iso()

        log.info(
            "[scanner] Scan complete: %d/%d tickers qualified",
            len(candidates), len(universe),
        )
        return candidates

    def scan_and_merge(
        self,
        universe: list[str],
        manual_tickers: list[str],
    ) -> list[ScannerCandidate]:
        """
        Combine scanner candidates with manual watchlist tickers.
        Manual tickers are always included (as MANUAL source) if they
        pass basic price/volume checks.  Scanner candidates are ranked
        on top.

        Args:
            universe:       Tickers to scan for momentum.
            manual_tickers: Tickers from manual_watchlist.json.

        Returns:
            Merged, de-duplicated candidate list.
        """
        scanner_candidates = self.scan(universe) if self._cfg.get(
            "allow_scanner_candidates", True) else []

        manual_candidates: list[ScannerCandidate] = []
        if self._cfg.get("allow_manual_candidates", True) and manual_tickers:
            manual_candidates = self._evaluate_manual(manual_tickers)

        # Merge: manual first (always watched), scanner appended
        seen: set[str] = set()
        merged: list[ScannerCandidate] = []

        for c in manual_candidates + scanner_candidates:
            if c.ticker not in seen:
                seen.add(c.ticker)
                merged.append(c)

        log.info(
            "[scanner] Merged candidates: %d manual + %d scanner = %d total",
            len(manual_candidates), len(scanner_candidates), len(merged),
        )
        return merged

    @property
    def last_scan_count(self) -> int:
        return self._last_scan_count

    @property
    def last_scan_time(self) -> str:
        return self._last_scan_time

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        ticker: str,
        snapshot: dict,
    ) -> Optional[tuple[float, ScannerCandidate]]:
        """
        Apply hard filters to a snapshot and return (score, candidate)
        if it passes, or None if it is rejected.
        """
        # ── Extract fields ────────────────────────────────────────────────────
        price = extract_price_from_snapshot(snapshot) or 0.0

        quote = (
            snapshot.get("latestQuote")
            or snapshot.get("latest_quote")
            or {}
        )
        bid, ask, spread_pct = extract_spread_from_quote(quote)

        # Day bar for volume
        day_bar = (
            snapshot.get("dailyBar")
            or snapshot.get("daily_bar")
            or {}
        )
        minute_bar = (
            snapshot.get("minuteBar")
            or snapshot.get("minute_bar")
            or {}
        )
        prev_bar = (
            snapshot.get("prevDailyBar")
            or snapshot.get("prev_daily_bar")
            or {}
        )

        day_volume    = float(day_bar.get("v", 0) or 0)
        dollar_volume = price * day_volume if price and day_volume else 0.0
        day_change    = build_day_change_pct(snapshot)

        # Relative volume: today's volume vs prev day volume
        prev_volume   = float(prev_bar.get("v", 0) or 0)
        rvol = (day_volume / prev_volume) if prev_volume > 0 else 0.0

        # ── Hard filters ──────────────────────────────────────────────────────
        rejection = self._hard_filter(
            ticker, price, spread_pct, dollar_volume, rvol, day_change
        )
        if rejection:
            log.debug("[scanner] %s rejected: %s", ticker, rejection)
            return None

        # ── Soft score ────────────────────────────────────────────────────────
        score = self._momentum_score(rvol, day_change, spread_pct, dollar_volume)

        # ── Build reason string ───────────────────────────────────────────────
        reasons = []
        if rvol >= 5.0:
            reasons.append(f"RVOL {rvol:.1f}x")
        if day_change >= 20.0:
            reasons.append(f"+{day_change:.1f}% move")
        if spread_pct <= 1.0:
            reasons.append("tight spread")
        reason_str = ", ".join(reasons) if reasons else f"RVOL {rvol:.1f}x +{day_change:.1f}%"

        candidate = ScannerCandidate(
            ticker           = ticker,
            source           = CandidateSource.SCANNER,
            price            = round(price, 4),
            bid              = round(bid, 4),
            ask              = round(ask, 4),
            spread_percent   = round(spread_pct, 4),
            relative_volume  = round(rvol, 2),
            day_change_pct   = round(day_change, 4),
            dollar_volume    = round(dollar_volume, 0),
            candidate_reason = reason_str,
            scanned_at       = _now_iso(),
        )
        return score, candidate

    def _hard_filter(
        self,
        ticker:       str,
        price:        float,
        spread_pct:   float,
        dollar_volume:float,
        rvol:         float,
        day_change:   float,
    ) -> Optional[str]:
        """
        Apply hard filter rules.
        Returns a rejection reason string, or None if the ticker passes.
        """
        cfg = self._cfg

        if price <= 0:
            return "no valid price"
        if price < float(cfg.get("min_price", 0.50)):
            return f"price ${price:.2f} below minimum"
        if price > float(cfg.get("max_price", 50.0)):
            return f"price ${price:.2f} above maximum"
        if spread_pct > float(cfg.get("max_spread_percent", 3.0)):
            return f"spread {spread_pct:.2f}% too wide"
        if dollar_volume < float(cfg.get("min_dollar_volume", 500_000)):
            return f"dollar volume ${dollar_volume:,.0f} too low"
        if rvol < float(cfg.get("min_relative_volume", 3.0)):
            return f"RVOL {rvol:.2f} below minimum"
        if day_change < float(cfg.get("min_day_change_percent", 10.0)):
            return f"change {day_change:.2f}% below minimum"

        return None   # passed all filters

    def _momentum_score(
        self,
        rvol:         float,
        day_change:   float,
        spread_pct:   float,
        dollar_volume:float,
    ) -> float:
        """
        Score a ticker's momentum quality on a 0–100 scale.
        Used only for ranking — not exposed to the trade quality gate.
        """
        # RVOL: 3x=40pts, 5x=60pts, 10x+=100pts (capped)
        rvol_score = min(rvol / 10.0 * 100, 100) * (_SCORE_WEIGHTS["rvol"] / 100)

        # Change %: 10%=40pts, 30%=80pts, 50%+=100pts
        change_score = min(day_change / 50.0 * 100, 100) * (_SCORE_WEIGHTS["change_pct"] / 100)

        # Spread: 0%=100pts, 3%=0pts (inverse)
        spread_score = max(0, (3.0 - spread_pct) / 3.0 * 100) * (_SCORE_WEIGHTS["spread"] / 100)

        # Dollar volume: $500k=40pts, $2M=80pts, $5M+=100pts
        dvol_score = min(dollar_volume / 5_000_000 * 100, 100) * (_SCORE_WEIGHTS["dollar_vol"] / 100)

        return round(rvol_score + change_score + spread_score + dvol_score, 2)

    def _evaluate_manual(self, tickers: list[str]) -> list[ScannerCandidate]:
        """
        Evaluate manual watchlist tickers — always included as candidates
        regardless of momentum score, as long as price data is available.
        """
        if not tickers:
            return []

        try:
            snapshots = self._client.get_snapshots_multi(tickers)
        except Exception as exc:
            log.warning("[scanner] Manual watchlist snapshot fetch failed: %s", exc)
            return []

        candidates: list[ScannerCandidate] = []
        for ticker, snapshot in snapshots.items():
            if not snapshot:
                log.debug("[scanner] Manual %s: no snapshot data", ticker)
                continue

            price      = extract_price_from_snapshot(snapshot) or 0.0
            quote      = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
            bid, ask, spread_pct = extract_spread_from_quote(quote)
            day_change = build_day_change_pct(snapshot)
            day_bar    = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
            prev_bar   = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
            day_vol    = float(day_bar.get("v", 0) or 0)
            prev_vol   = float(prev_bar.get("v", 0) or 0)
            rvol       = (day_vol / prev_vol) if prev_vol > 0 else 0.0

            if price <= 0:
                log.debug("[scanner] Manual %s: no valid price", ticker)
                continue

            candidates.append(ScannerCandidate(
                ticker           = ticker,
                source           = CandidateSource.MANUAL,
                price            = round(price, 4),
                bid              = round(bid, 4),
                ask              = round(ask, 4),
                spread_percent   = round(spread_pct, 4),
                relative_volume  = round(rvol, 2),
                day_change_pct   = round(day_change, 4),
                dollar_volume    = round(price * day_vol, 0),
                candidate_reason = "manual watchlist",
                scanned_at       = _now_iso(),
            ))

        log.info(
            "[scanner] Manual watchlist: %d/%d tickers added as candidates",
            len(candidates), len(tickers),
        )
        return candidates

    # ── Snapshot fetching ─────────────────────────────────────────────────────

    def _fetch_snapshots(self, tickers: list[str]) -> dict[str, Optional[dict]]:
        """Fetch snapshots for a list of tickers, chunked to stay under API limits."""
        chunk_size = 100   # Alpaca supports up to 100 symbols per call
        results: dict[str, Optional[dict]] = {}

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                batch = self._client.get_snapshots_multi(chunk)
                results.update(batch)
            except Exception as exc:
                log.warning(
                    "[scanner] Snapshot fetch failed for chunk %d-%d: %s",
                    i, i + len(chunk), exc,
                )
                for t in chunk:
                    results[t] = None

        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
