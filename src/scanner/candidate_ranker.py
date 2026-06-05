"""
src/scanner/candidate_ranker.py — Candidate ranking and prioritization
Takes the filtered candidate list and ranks them by overall trade quality
potential before passing to the analysis pipeline.

Responsibilities:
  - Score each candidate on momentum quality, catalyst presence, and
    session context
  - Apply priority boosts for manual watchlist tickers
  - Apply time-of-day weighting (early session momentum trades score higher)
  - Return a ranked CandidateRank list with scores and reasons
  - Cap the final list at max_candidates_per_loop

Design rules:
  - Ranker scores are for ORDERING only — not trade quality scores
  - The ranker has no knowledge of setups or technical analysis
  - All candidates that pass filters are ranked — none are rejected here
  - The final trade quality gate makes the actual buy/no-buy decision
  - Ranking is transparent — every score has a reason string

Ranking factors (total 100 points):
  Relative volume:   30 pts — strongest momentum signal
  Day change %:      25 pts — price move magnitude
  Catalyst:          20 pts — news/catalyst strength bonus
  Dollar volume:     15 pts — liquidity quality
  Spread quality:    10 pts — execution quality proxy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from models import CandidateSource, ScannerCandidate
from news_catalyst_provider import CatalystResult
from time_utils import minutes_since_open, now_et

log = logging.getLogger(__name__)

# ── Ranking weights ───────────────────────────────────────────────────────────

_WEIGHTS = {
    "rvol":        30,
    "change_pct":  25,
    "catalyst":    20,
    "dollar_vol":  15,
    "spread":      10,
}

# Caps for normalization
_RVOL_CAP        = 15.0     # 15x RVOL = max rvol score
_CHANGE_CAP      = 60.0     # 60% day change = max change score
_DOLLAR_VOL_CAP  = 5_000_000.0
_SPREAD_MAX      = 3.0      # 0% spread = max, 3% = zero


# ── Ranked candidate ──────────────────────────────────────────────────────────

@dataclass
class CandidateRank:
    """
    A scored and ranked scanner candidate.
    Produced by CandidateRanker.rank() and consumed by the analysis pipeline.
    """
    candidate:       ScannerCandidate
    rank_score:      float          = 0.0
    score_breakdown: dict           = field(default_factory=dict)
    catalyst:        Optional[CatalystResult] = None
    warnings:        list[str]      = field(default_factory=list)
    rank_reasons:    list[str]      = field(default_factory=list)
    ranked_at:       str            = field(
                         default_factory=lambda: datetime.now(timezone.utc).isoformat()
                     )

    @property
    def ticker(self) -> str:
        return self.candidate.ticker

    @property
    def price(self) -> float:
        return self.candidate.price

    @property
    def source(self) -> CandidateSource:
        return self.candidate.source

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "rank_score":      self.rank_score,
            "score_breakdown": self.score_breakdown,
            "rank_reasons":    self.rank_reasons,
            "warnings":        self.warnings,
            "price":           self.price,
            "relative_volume": self.candidate.relative_volume,
            "day_change_pct":  self.candidate.day_change_pct,
            "dollar_volume":   self.candidate.dollar_volume,
            "spread_percent":  self.candidate.spread_percent,
            "source":          self.source.value,
            "catalyst_strength": self.catalyst.strength if self.catalyst else "none",
            "ranked_at":       self.ranked_at,
        }


# ── Ranker ────────────────────────────────────────────────────────────────────

class CandidateRanker:
    """
    Ranks filtered scanner candidates by momentum quality potential.

    Usage:
        ranker  = CandidateRanker(settings)
        ranked  = ranker.rank(candidates, catalysts=catalyst_map)
        top     = ranker.top_n(ranked, n=10)
    """

    def __init__(self, settings: dict):
        self._settings      = settings
        self._scanner_cfg   = settings.get("scanner", {})
        self._max_candidates = int(
            self._scanner_cfg.get("max_candidates_per_loop", 20)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def rank(
        self,
        candidates: list[ScannerCandidate],
        catalysts:  Optional[dict[str, CatalystResult]] = None,
        warnings:   Optional[dict[str, list[str]]]      = None,
        ref:        Optional[datetime]                  = None,
    ) -> list[CandidateRank]:
        """
        Score and rank a list of filtered candidates.

        Args:
            candidates: Filtered ScannerCandidate list (all should have
                        passed scanner_filter_engine).
            catalysts:  Optional dict mapping ticker → CatalystResult.
                        Pass None to skip catalyst scoring.
            warnings:   Optional dict mapping ticker → list of warning strings
                        from scanner_filter_engine.
            ref:        Optional reference datetime for testing.

        Returns:
            List of CandidateRank objects sorted best-first.
            Capped at max_candidates_per_loop.
        """
        if not candidates:
            return []

        now = ref or now_et()
        catalysts = catalysts or {}
        warnings  = warnings  or {}

        # Session context for time-of-day weighting
        mins_open      = minutes_since_open(self._settings, ref=now)
        session_boost  = self._session_boost(mins_open)

        ranked: list[CandidateRank] = []

        for candidate in candidates:
            catalyst = catalysts.get(candidate.ticker)
            cand_warnings = warnings.get(candidate.ticker, [])

            rank = self._score_candidate(
                candidate, catalyst, cand_warnings, session_boost
            )
            ranked.append(rank)

        # Sort descending by rank score
        ranked.sort(key=lambda r: r.rank_score, reverse=True)

        # Apply manual priority boost — manual tickers float to top
        # within their score tier (within 5 points of next scanner candidate)
        ranked = self._apply_manual_boost(ranked)

        # Cap at max
        ranked = ranked[: self._max_candidates]

        log.info(
            "[ranker] Ranked %d candidates — top: %s (%.1f pts)",
            len(ranked),
            ranked[0].ticker if ranked else "none",
            ranked[0].rank_score if ranked else 0,
        )
        return ranked

    def top_n(self, ranked: list[CandidateRank], n: int) -> list[CandidateRank]:
        """Return the top N ranked candidates."""
        return ranked[:n]

    def tickers(self, ranked: list[CandidateRank]) -> list[str]:
        """Return the ticker symbols in rank order."""
        return [r.ticker for r in ranked]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_candidate(
        self,
        candidate:     ScannerCandidate,
        catalyst:      Optional[CatalystResult],
        warnings:      list[str],
        session_boost: float,
    ) -> CandidateRank:
        """Compute the rank score for a single candidate."""
        breakdown: dict[str, float] = {}
        reasons:   list[str]        = []

        # ── RVOL score (30 pts) ───────────────────────────────────────────────
        rvol_norm  = min(candidate.relative_volume / _RVOL_CAP, 1.0)
        rvol_score = round(rvol_norm * _WEIGHTS["rvol"], 2)
        breakdown["rvol"] = rvol_score
        if candidate.relative_volume >= 5.0:
            reasons.append(f"RVOL {candidate.relative_volume:.1f}x")

        # ── Day change score (25 pts) ─────────────────────────────────────────
        change_norm  = min(candidate.day_change_pct / _CHANGE_CAP, 1.0)
        change_score = round(change_norm * _WEIGHTS["change_pct"], 2)
        breakdown["change_pct"] = change_score
        if candidate.day_change_pct >= 20.0:
            reasons.append(f"+{candidate.day_change_pct:.1f}% move")

        # ── Catalyst score (20 pts) ───────────────────────────────────────────
        cat_score = 0.0
        if catalyst and catalyst.has_catalyst:
            if catalyst.strength == "strong":
                cat_score = _WEIGHTS["catalyst"]       # 20 pts
                reasons.append(f"strong catalyst: {catalyst.top_headline[:50]}")
            elif catalyst.strength == "moderate":
                cat_score = _WEIGHTS["catalyst"] * 0.6  # 12 pts
                reasons.append("moderate catalyst")
            elif catalyst.strength == "weak":
                cat_score = _WEIGHTS["catalyst"] * 0.2  # 4 pts
        breakdown["catalyst"] = round(cat_score, 2)

        # ── Dollar volume score (15 pts) ──────────────────────────────────────
        dvol_norm  = min(candidate.dollar_volume / _DOLLAR_VOL_CAP, 1.0)
        dvol_score = round(dvol_norm * _WEIGHTS["dollar_vol"], 2)
        breakdown["dollar_vol"] = dvol_score
        if candidate.dollar_volume >= 1_000_000:
            reasons.append(f"${candidate.dollar_volume/1e6:.1f}M vol")

        # ── Spread score (10 pts — inverse) ──────────────────────────────────
        spread_norm  = max(0.0, (_SPREAD_MAX - candidate.spread_percent) / _SPREAD_MAX)
        spread_score = round(spread_norm * _WEIGHTS["spread"], 2)
        breakdown["spread"] = spread_score
        if candidate.spread_percent <= 0.5:
            reasons.append("tight spread")

        # ── Session boost (multiplicative, max +10%) ──────────────────────────
        base_score   = sum(breakdown.values())
        boosted      = round(base_score * (1.0 + session_boost), 2)
        breakdown["session_boost"] = round(session_boost * 100, 1)

        if session_boost > 0:
            reasons.append(f"session boost +{session_boost*100:.0f}%")

        if not reasons:
            reasons.append(
                f"RVOL {candidate.relative_volume:.1f}x "
                f"+{candidate.day_change_pct:.1f}%"
            )

        return CandidateRank(
            candidate       = candidate,
            rank_score      = boosted,
            score_breakdown = breakdown,
            catalyst        = catalyst,
            warnings        = warnings,
            rank_reasons    = reasons,
        )

    def _session_boost(self, mins_open: float) -> float:
        """
        Time-of-day multiplier applied to the base rank score.

        First 30 min (opening):    +10% — highest momentum window
        30–90 min (early active):  +5%  — strong continuation window
        90–210 min (mid active):   +0%  — neutral
        210–330 min (afternoon):   -5%  — lower quality setups
        330+ min (power hour):     +5%  — late day momentum window
        """
        if mins_open < 0:
            return 0.0
        if mins_open < 30:
            return 0.10
        if mins_open < 90:
            return 0.05
        if mins_open < 210:
            return 0.0
        if mins_open < 330:
            return -0.05
        return 0.05

    def _apply_manual_boost(
        self, ranked: list[CandidateRank]
    ) -> list[CandidateRank]:
        """
        Ensure manual watchlist tickers are not buried below scanner
        tickers with a similar score.  Manual tickers within 5 rank points
        of the next scanner ticker are floated above it.
        This preserves ranking integrity while respecting user intent.
        """
        if not ranked:
            return ranked

        # Stable partition: manual first (within their score band), then scanner
        manual  = [r for r in ranked if r.source == CandidateSource.MANUAL]
        scanner = [r for r in ranked if r.source == CandidateSource.SCANNER]

        if not manual:
            return scanner

        # Interleave: manual ticker floats above scanner ticker if within 5 pts
        result:   list[CandidateRank] = []
        s_idx = 0
        for m in manual:
            # Insert any scanner tickers that are clearly higher scored (>5 pts)
            while s_idx < len(scanner) and scanner[s_idx].rank_score > m.rank_score + 5:
                result.append(scanner[s_idx])
                s_idx += 1
            result.append(m)

        # Append remaining scanner tickers
        result.extend(scanner[s_idx:])
        return result
