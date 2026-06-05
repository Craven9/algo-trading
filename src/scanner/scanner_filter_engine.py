"""
src/scanner/scanner_filter_engine.py — Secondary filter pass for candidates
After the momentum scanner produces raw candidates, this engine applies
a deeper set of filters before candidates are passed to the analysis pipeline.

The scanner produces candidates fast with broad filters.
This engine applies precise, configurable rules to keep only the highest
quality candidates for full analysis — saving compute on the scoring engines.

Responsibilities:
  - Validate quote freshness and data integrity
  - Filter by price range, spread, volume thresholds
  - Remove already-open positions
  - Remove tickers that hit daily loss guard
  - Apply time-of-day filters (avoid opening minutes, closing minutes)
  - Apply session filters (no new entries after hours unless configured)
  - Flag candidates with warnings without hard-blocking them
  - Return FilterResult objects with pass/fail reason for every candidate

Design rule:
  A candidate that fails a HARD filter is REJECTED — removed from the list.
  A candidate that fails a SOFT filter gets a WARNING but stays in the list.
  The scoring engine reads warnings and may reduce the score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from models import CandidateSource, ScannerCandidate
from time_utils import (
    is_avoid_window,
    is_high_risk_window,
    is_market_open,
    minutes_since_open,
    now_et,
)

log = logging.getLogger(__name__)

# ── Filter result ─────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """
    Result of applying the filter engine to a single candidate.
    Passed candidates proceed to full analysis.
    Rejected candidates are logged and discarded.
    """
    candidate:   ScannerCandidate
    passed:      bool          = True
    reject_reason: str         = ""
    warnings:    list[str]     = field(default_factory=list)
    filtered_at: str           = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "ticker":        self.candidate.ticker,
            "passed":        self.passed,
            "reject_reason": self.reject_reason,
            "warnings":      self.warnings,
            "filtered_at":   self.filtered_at,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class ScannerFilterEngine:
    """
    Applies a layered set of hard and soft filters to scanner candidates.

    Usage:
        engine  = ScannerFilterEngine(settings)
        results = engine.filter(candidates, open_positions=["ABCD"])
        passing = [r.candidate for r in results if r.passed]
    """

    def __init__(self, settings: dict):
        self._settings   = settings
        self._scanner    = settings.get("scanner", {})
        self._entry      = settings.get("entry_rules", {})
        self._exec       = settings.get("execution", {})
        self._session    = settings.get("session", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(
        self,
        candidates:      list[ScannerCandidate],
        open_positions:  list[str]  = None,
        daily_loss_hit:  bool       = False,
        ref:             Optional[datetime] = None,
    ) -> list[FilterResult]:
        """
        Apply all filters to a list of candidates.

        Args:
            candidates:     Raw ScannerCandidate list from the scanner.
            open_positions: List of ticker symbols with open positions.
                            Candidates in this list are hard-rejected.
            daily_loss_hit: If True, all new entries are hard-rejected.
            ref:            Optional reference datetime (for testing).

        Returns:
            List of FilterResult objects — one per candidate.
            Order matches the input candidate list.
        """
        open_positions = [t.upper() for t in (open_positions or [])]
        results: list[FilterResult] = []

        for candidate in candidates:
            result = self._evaluate(candidate, open_positions, daily_loss_hit, ref)
            results.append(result)

            if result.passed:
                log.debug(
                    "[filter] %s PASSED%s",
                    candidate.ticker,
                    f" (warnings: {result.warnings})" if result.warnings else "",
                )
            else:
                log.debug(
                    "[filter] %s REJECTED: %s",
                    candidate.ticker, result.reject_reason,
                )

        passed  = sum(1 for r in results if r.passed)
        total   = len(results)
        log.info("[filter] %d/%d candidates passed filters", passed, total)
        return results

    def filter_passing(
        self,
        candidates:     list[ScannerCandidate],
        open_positions: list[str]  = None,
        daily_loss_hit: bool       = False,
        ref:            Optional[datetime] = None,
    ) -> list[ScannerCandidate]:
        """
        Convenience wrapper — returns only candidates that passed.
        Warnings are discarded; use filter() if you need them.
        """
        results = self.filter(candidates, open_positions, daily_loss_hit, ref)
        return [r.candidate for r in results if r.passed]

    # ── Per-candidate evaluation ──────────────────────────────────────────────

    def _evaluate(
        self,
        candidate:      ScannerCandidate,
        open_positions: list[str],
        daily_loss_hit: bool,
        ref:            Optional[datetime],
    ) -> FilterResult:
        """Apply all hard and soft filters to a single candidate."""
        warnings: list[str] = []
        now = ref or now_et()

        # ── HARD FILTERS — reject immediately ────────────────────────────────

        # 1. Daily loss limit
        if daily_loss_hit:
            return self._reject(candidate, "daily loss limit reached — no new entries")

        # 2. Position already open
        if candidate.ticker in open_positions:
            return self._reject(candidate, "position already open")

        # 3. Price sanity
        if candidate.price <= 0:
            return self._reject(candidate, "no valid price")

        min_price = float(self._scanner.get("min_price", 0.50))
        max_price = float(self._scanner.get("max_price", 50.0))
        if candidate.price < min_price:
            return self._reject(candidate, f"price ${candidate.price:.2f} below minimum ${min_price:.2f}")
        if candidate.price > max_price:
            return self._reject(candidate, f"price ${candidate.price:.2f} above maximum ${max_price:.2f}")

        # 4. Spread too wide (hard block)
        max_spread = float(self._entry.get("max_spread_percent_at_execution",
                           self._scanner.get("max_spread_percent", 3.0)))
        if candidate.spread_percent > max_spread:
            return self._reject(
                candidate,
                f"spread {candidate.spread_percent:.2f}% exceeds maximum {max_spread:.2f}%",
            )

        # 5. Market session — no new entries outside regular hours
        # unless after_hours_scan_only is explicitly enabled
        after_hours_scan_only = self._settings.get("mode", {}).get(
            "after_hours_scan_only", False
        )
        if not is_market_open(self._settings, ref=now) and not after_hours_scan_only:
            return self._reject(candidate, "market is not open for regular trading")

        # 6. Avoid window (first N / last N minutes of session)
        if is_avoid_window(self._settings, ref=now):
            avoid_first = self._session.get("avoid_first_minutes", 1)
            avoid_last  = self._session.get("avoid_last_minutes", 5)
            return self._reject(
                candidate,
                f"avoid window — first {avoid_first} min or last {avoid_last} min of session",
            )

        # 7. Volume sanity
        min_dollar_vol = float(self._scanner.get("min_dollar_volume", 500_000))
        if candidate.dollar_volume < min_dollar_vol:
            return self._reject(
                candidate,
                f"dollar volume ${candidate.dollar_volume:,.0f} below minimum",
            )

        # 8. Minimum RVOL
        min_rvol = float(self._scanner.get("min_relative_volume", 3.0))
        if candidate.relative_volume < min_rvol:
            return self._reject(
                candidate,
                f"RVOL {candidate.relative_volume:.2f}x below minimum {min_rvol:.1f}x",
            )

        # 9. Minimum day change
        min_change = float(self._scanner.get("min_day_change_percent", 10.0))
        if candidate.day_change_pct < min_change:
            return self._reject(
                candidate,
                f"day change {candidate.day_change_pct:.2f}% below minimum {min_change:.1f}%",
            )

        # ── SOFT FILTERS — add warnings but allow through ─────────────────────

        # High-risk window warning
        if is_high_risk_window(self._settings, ref=now):
            warnings.append("high-risk time window — elevated spread/volatility likely")

        # Overextended warning
        max_ext = float(self._entry.get("max_entry_extension_percent", 8.0))
        if candidate.day_change_pct > max_ext * 2:
            warnings.append(
                f"ticker up {candidate.day_change_pct:.1f}% today — may be overextended"
            )

        # Spread warning (still below hard limit but elevated)
        spread_warn_threshold = max_spread * 0.6
        if candidate.spread_percent > spread_warn_threshold:
            warnings.append(
                f"spread {candidate.spread_percent:.2f}% is elevated — watch execution"
            )

        # Very early session warning (first 5 minutes but past the hard avoid block)
        mins_in = minutes_since_open(self._settings, ref=now)
        if 1 < mins_in < 5:
            warnings.append(
                f"only {mins_in:.1f} minutes since open — volatility still high"
            )

        # Manual watchlist note
        if candidate.source == CandidateSource.MANUAL:
            warnings.append("manual watchlist ticker — confirm setup before entry")

        return FilterResult(
            candidate    = candidate,
            passed       = True,
            reject_reason= "",
            warnings     = warnings,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reject(candidate: ScannerCandidate, reason: str) -> FilterResult:
        return FilterResult(
            candidate     = candidate,
            passed        = False,
            reject_reason = reason,
            warnings      = [],
        )

    # ── Summary helpers ───────────────────────────────────────────────────────

    @staticmethod
    def passed_count(results: list[FilterResult]) -> int:
        return sum(1 for r in results if r.passed)

    @staticmethod
    def rejected_count(results: list[FilterResult]) -> int:
        return sum(1 for r in results if not r.passed)

    @staticmethod
    def rejection_summary(results: list[FilterResult]) -> list[dict]:
        """Return a list of rejection dicts for the frontend / logs."""
        return [
            {"ticker": r.candidate.ticker, "reason": r.reject_reason}
            for r in results if not r.passed
        ]

    @staticmethod
    def warning_summary(results: list[FilterResult]) -> list[dict]:
        """Return a list of warning dicts for the frontend / logs."""
        return [
            {"ticker": r.candidate.ticker, "warnings": r.warnings}
            for r in results if r.passed and r.warnings
        ]
