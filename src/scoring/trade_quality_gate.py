"""
src/scoring/trade_quality_gate.py — Final trade approval gate
The only module allowed to authorize a paper buy.

This file combines all scoring and risk inputs into one final decision.
Every scanner candidate, watchlist ticker, and setup must pass through this
gate before order_executor.py is allowed to place a paper order.

Final score formula from the bot overview:
  final_trade_quality_score =
      setup_score           * 0.35
    + probability_score     * 0.35
    + risk_reward_score     * 0.15
    + move_potential_score  * 0.10
    + historical_edge_score * 0.05

Possible decisions:
  APPROVED_FOR_PAPER_BUY
  REJECTED
  WATCH
  WAIT_FOR_PULLBACK
  WAIT_FOR_RECLAIM
  MANAGE_EXISTING_ONLY

Hard rejection rules:
  - Paper trading is not confirmed
  - Safety lock is enabled
  - Bot is disabled
  - Setup score below minimum
  - Probability score below minimum
  - Final quality score below minimum
  - Risk/reward hard block
  - Failed reclaim block
  - Fibonacci block
  - Existing position already open
  - Execution quality hard block
  - Account/risk guard rejection
  - Missing entry, stop, or target when required

Design rules:
  - This file does not place orders
  - This file only returns TradeDecisionResult
  - order_executor.py may only run after this returns APPROVED_FOR_PAPER_BUY
  - Safety checks always override scores
  - Rejections explain what would make the trade valid
"""

from __future__ import annotations

import logging
from typing import Optional

from confidence_labeler import confidence_result
from models import (
    ConfidenceLabel,
    PositionSize,
    SetupName,
    SetupResult,
    TradeDecision,
    TradeDecisionResult,
    TradeScores,
)

log = logging.getLogger(__name__)


# ── Score weights ─────────────────────────────────────────────────────────────

_FINAL_WEIGHTS = {
    "setup_score":           0.35,
    "probability_score":     0.35,
    "risk_reward_score":     0.15,
    "move_potential_score":  0.10,
    "historical_edge_score": 0.05,
}


# ── Gate ──────────────────────────────────────────────────────────────────────

class TradeQualityGate:
    """
    Final trade approval gate.

    Usage:
        gate = TradeQualityGate(settings)
        decision = gate.evaluate(
            ticker="ABCD",
            setup_result=best_setup,
            setup_score_result=setup_score,
            probability_result=probability,
            risk_reward_result=rr_result,
            move_potential_result=move_result,
            execution_quality_result=exec_result,
            risk_result=risk_result,
            context=context,
        )

        if decision.approved:
            # order_executor.py may place paper order
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._mode     = settings.get("mode", {})
        self._entry    = settings.get("entry_rules", {})
        self._risk     = settings.get("risk", {})

        self._min_setup = float(self._entry.get("minimum_setup_score", 78))
        self._min_probability = float(self._entry.get("minimum_probability_score", 75))
        self._min_final = float(self._entry.get("minimum_final_trade_quality_score", 80))

        self._require_clear_stop = bool(self._entry.get("require_clear_stop", True))
        self._require_clear_target = bool(self._entry.get("require_clear_target", True))

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        ticker:                    str,
        setup_result:              Optional[SetupResult],
        setup_score_result:        Optional[object],
        probability_result:        Optional[object],
        risk_reward_result:        Optional[object],
        move_potential_result:     Optional[object],
        execution_quality_result:  Optional[object] = None,
        risk_result:               Optional[object] = None,
        historical_edge_result:    Optional[object] = None,
        context:                   Optional[dict] = None,
        position_size:             Optional[PositionSize] = None,
    ) -> TradeDecisionResult:
        """
        Evaluate all inputs and return the final TradeDecisionResult.

        Args:
            ticker:                   Ticker symbol.
            setup_result:             Best confirmed setup result.
            setup_score_result:       SetupScoreResult.
            probability_result:       ProbabilityResult.
            risk_reward_result:       RiskRewardResult.
            move_potential_result:    MovePotentialResult.
            execution_quality_result: ExecutionQualityResult.
            risk_result:              Risk manager/account guard result.
            historical_edge_result:   Historical edge result.
            context:                  Analysis context dict.
            position_size:            Optional PositionSize object.

        Returns:
            TradeDecisionResult.
        """
        context = context or {}

        scores = self._build_scores(
            setup_score_result        = setup_score_result,
            probability_result        = probability_result,
            risk_reward_result        = risk_reward_result,
            move_potential_result     = move_potential_result,
            execution_quality_result  = execution_quality_result,
            historical_edge_result    = historical_edge_result,
        )

        confidence = confidence_result(scores.final_trade_quality_score, self._settings)
        scores.confidence_label = confidence.label

        entry_price, stop_loss, target_1, target_2, runner_target = self._extract_trade_plan(
            setup_result          = setup_result,
            risk_reward_result    = risk_reward_result,
            move_potential_result = move_potential_result,
            context               = context,
        )

        decision = TradeDecisionResult(
            ticker                = ticker,
            decision              = TradeDecision.REJECTED.value,
            setup                 = setup_result.setup_name if setup_result else SetupName.NONE.value,
            scores                = scores,
            entry_price           = entry_price,
            stop_loss             = stop_loss,
            target_1              = target_1,
            target_2              = target_2,
            runner_target         = runner_target,
            position_size         = position_size,
            reasons               = [],
            warnings              = [],
            what_would_make_valid = [],
        )

        # ── Hard blocks ───────────────────────────────────────────────────────
        hard_blocks = self._hard_blocks(
            decision                 = decision,
            setup_result             = setup_result,
            setup_score_result       = setup_score_result,
            probability_result       = probability_result,
            risk_reward_result       = risk_reward_result,
            move_potential_result    = move_potential_result,
            execution_quality_result = execution_quality_result,
            risk_result              = risk_result,
            context                  = context,
            confidence_no_trade      = confidence.no_trade,
        )

        if hard_blocks:
            decision.decision = self._reject_or_wait_decision(hard_blocks, context)
            decision.reasons.extend(hard_blocks)
            decision.what_would_make_valid.extend(
                self._what_would_make_valid(hard_blocks)
            )
            self._merge_optional_warnings(decision, [
                setup_score_result,
                probability_result,
                risk_reward_result,
                move_potential_result,
                execution_quality_result,
                risk_result,
                historical_edge_result,
            ])
            log.info(
                "[trade_gate] %s %s: %s",
                ticker, decision.decision, "; ".join(hard_blocks),
            )
            return decision

        # ── Soft watch / wait logic ───────────────────────────────────────────
        watch_reasons = self._watch_reasons(
            setup_result          = setup_result,
            setup_score_result    = setup_score_result,
            probability_result    = probability_result,
            move_potential_result = move_potential_result,
            context               = context,
        )

        if watch_reasons:
            decision.decision = self._watch_decision(watch_reasons, context)
            decision.reasons.extend(watch_reasons)
            decision.what_would_make_valid.extend(
                self._what_would_make_valid(watch_reasons)
            )
            self._merge_optional_warnings(decision, [
                setup_score_result,
                probability_result,
                risk_reward_result,
                move_potential_result,
                execution_quality_result,
                risk_result,
                historical_edge_result,
            ])
            log.info(
                "[trade_gate] %s %s: %s",
                ticker, decision.decision, "; ".join(watch_reasons),
            )
            return decision

        # ── Approved ──────────────────────────────────────────────────────────
        decision.decision = TradeDecision.APPROVED_FOR_PAPER_BUY.value
        decision.reasons.append(
            f"Approved for paper buy — final quality score "
            f"{scores.final_trade_quality_score:.1f}"
        )
        decision.reasons.append(f"Confidence label: {scores.confidence_label}")
        decision.reasons.append(confidence.reason)

        if setup_result:
            decision.reasons.extend(setup_result.reasons)
            decision.warnings.extend(setup_result.warnings)

        self._merge_optional_warnings(decision, [
            setup_score_result,
            probability_result,
            risk_reward_result,
            move_potential_result,
            execution_quality_result,
            risk_result,
            historical_edge_result,
        ])

        log.info(
            "[trade_gate] %s APPROVED final=%.1f setup=%.1f prob=%.1f rr=%.1f",
            ticker,
            scores.final_trade_quality_score,
            scores.setup_score,
            scores.probability_score,
            scores.risk_reward_score,
        )
        return decision

    # ── Score construction ────────────────────────────────────────────────────

    def _build_scores(
        self,
        setup_score_result:        Optional[object],
        probability_result:        Optional[object],
        risk_reward_result:        Optional[object],
        move_potential_result:     Optional[object],
        execution_quality_result:  Optional[object],
        historical_edge_result:    Optional[object],
    ) -> TradeScores:
        """Build TradeScores from all scoring inputs."""
        setup_score = _safe_float(
            getattr(setup_score_result, "setup_score", 0.0)
            if setup_score_result else 0.0
        )
        probability_score = _safe_float(
            getattr(probability_result, "probability_score", 0.0)
            if probability_result else 0.0
        )
        risk_reward_score = _safe_float(
            getattr(risk_reward_result, "risk_reward_score", 0.0)
            if risk_reward_result else 0.0
        )
        move_potential_score = _safe_float(
            getattr(move_potential_result, "move_potential_score", 0.0)
            if move_potential_result else 0.0
        )
        execution_quality_score = _safe_float(
            getattr(execution_quality_result, "execution_quality_score", 0.0)
            if execution_quality_result else 0.0
        )
        historical_edge_score = _safe_float(
            getattr(historical_edge_result, "historical_edge_score", 50.0)
            if historical_edge_result else 50.0
        )

        final_score = (
            setup_score           * _FINAL_WEIGHTS["setup_score"]
            + probability_score   * _FINAL_WEIGHTS["probability_score"]
            + risk_reward_score   * _FINAL_WEIGHTS["risk_reward_score"]
            + move_potential_score * _FINAL_WEIGHTS["move_potential_score"]
            + historical_edge_score * _FINAL_WEIGHTS["historical_edge_score"]
        )

        return TradeScores(
            setup_score               = round(setup_score, 2),
            probability_score         = round(probability_score, 2),
            risk_reward_score         = round(risk_reward_score, 2),
            move_potential_score      = round(move_potential_score, 2),
            execution_quality_score   = round(execution_quality_score, 2),
            historical_edge_score     = round(historical_edge_score, 2),
            final_trade_quality_score = round(max(0.0, min(final_score, 100.0)), 2),
            confidence_label          = ConfidenceLabel.REJECT.value,
        )

    # ── Trade plan extraction ─────────────────────────────────────────────────

    def _extract_trade_plan(
        self,
        setup_result: Optional[SetupResult],
        risk_reward_result: Optional[object],
        move_potential_result: Optional[object],
        context: dict,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Extract entry, stop, and targets from setup/risk/move context.
        """
        entry = None
        stop = None
        target_1 = None
        target_2 = None
        runner_target = None

        if setup_result:
            entry = setup_result.entry_trigger
            stop = setup_result.stop_area
            target_1 = setup_result.target_area

        if risk_reward_result:
            entry = entry or _optional_float(getattr(risk_reward_result, "entry_price", None))
            stop = stop or _optional_float(getattr(risk_reward_result, "stop_price", None))
            target_1 = target_1 or _optional_float(getattr(risk_reward_result, "target_price", None))

        if move_potential_result:
            target_1 = target_1 or _optional_float(getattr(move_potential_result, "fib_target_1", None))
            target_2 = _optional_float(getattr(move_potential_result, "fib_target_2", None))
            runner_target = _optional_float(getattr(move_potential_result, "runner_target", None))

        current_price = _optional_float(context.get("current_price"))
        if entry is None and current_price:
            entry = current_price

        return entry, stop, target_1, target_2, runner_target

    # ── Hard blocks ───────────────────────────────────────────────────────────

    def _hard_blocks(
        self,
        decision: TradeDecisionResult,
        setup_result: Optional[SetupResult],
        setup_score_result: Optional[object],
        probability_result: Optional[object],
        risk_reward_result: Optional[object],
        move_potential_result: Optional[object],
        execution_quality_result: Optional[object],
        risk_result: Optional[object],
        context: dict,
        confidence_no_trade: bool,
    ) -> list[str]:
        """
        Return hard-block reasons.
        """
        blocks: list[str] = []

        # Mode / safety
        if not bool(self._mode.get("bot_enabled", True)):
            blocks.append("bot_enabled is false")
        if bool(self._mode.get("safety_lock", False)):
            blocks.append("safety_lock is enabled")
        if bool(self._mode.get("allow_live_money", False)):
            blocks.append("allow_live_money is true — refusing to trade")
        if not bool(self._mode.get("paper_trading_only", True)):
            blocks.append("paper_trading_only is not active")

        # Setup
        if setup_result is None:
            blocks.append("no setup detected")
        elif not setup_result.confirmed:
            blocks.append("setup is not confirmed")

        setup_score = decision.scores.setup_score
        if setup_score < self._min_setup:
            blocks.append(
                f"setup score {setup_score:.1f} below minimum {self._min_setup:.1f}"
            )

        # Probability
        probability_score = decision.scores.probability_score
        if probability_score < self._min_probability:
            blocks.append(
                f"probability score {probability_score:.1f} below minimum "
                f"{self._min_probability:.1f}"
            )

        # Final score
        final_score = decision.scores.final_trade_quality_score
        if final_score < self._min_final:
            blocks.append(
                f"final trade quality score {final_score:.1f} below minimum "
                f"{self._min_final:.1f}"
            )

        if confidence_no_trade:
            blocks.append("confidence sizing says no trade")

        # Required trade plan
        if decision.entry_price is None or decision.entry_price <= 0:
            blocks.append("missing valid entry price")
        if self._require_clear_stop and (decision.stop_loss is None or decision.stop_loss <= 0):
            blocks.append("missing clear stop loss")
        if self._require_clear_target and (decision.target_1 is None or decision.target_1 <= 0):
            blocks.append("missing clear target")

        if (
            decision.entry_price
            and decision.stop_loss
            and decision.stop_loss >= decision.entry_price
        ):
            blocks.append("stop loss is not below entry for long trade")

        if (
            decision.entry_price
            and decision.target_1
            and decision.target_1 <= decision.entry_price
        ):
            blocks.append("target is not above entry for long trade")

        # Risk / reward block
        if risk_reward_result and getattr(risk_reward_result, "hard_block", False):
            blocks.append("risk/reward hard block")

        # Execution block
        if execution_quality_result and getattr(execution_quality_result, "hard_block", False):
            blocks.append("execution quality hard block")

        if execution_quality_result and getattr(execution_quality_result, "position_already_open", False):
            blocks.append("position already open")

        # Account/risk guard block
        if risk_result:
            if getattr(risk_result, "hard_block", False):
                blocks.append("account risk guard hard block")
            elif getattr(risk_result, "approved", True) is False:
                blocks.append("account risk guard rejected trade")

        # Failed reclaim block
        failed_reclaim = context.get("failed_reclaim")
        if failed_reclaim and getattr(failed_reclaim, "block_long_entry", False):
            blocks.append("failed reclaim blocks long entry")

        # Fibonacci block
        fib_result = context.get("fib_result")
        if fib_result and getattr(fib_result, "block_trade", False):
            blocks.append("Fibonacci engine blocks trade")

        # Entry rules
        if self._entry.get("block_failed_reclaims", True):
            if failed_reclaim and getattr(failed_reclaim, "any_detected", False):
                if getattr(failed_reclaim, "worst_severity", "") in ("critical", "moderate"):
                    blocks.append("failed reclaim detected")

        return _dedupe(blocks)

    # ── Watch / wait logic ────────────────────────────────────────────────────

    def _watch_reasons(
        self,
        setup_result: Optional[SetupResult],
        setup_score_result: Optional[object],
        probability_result: Optional[object],
        move_potential_result: Optional[object],
        context: dict,
    ) -> list[str]:
        """
        Return non-hard reasons to watch instead of immediately trade.
        """
        reasons: list[str] = []

        if setup_result and setup_result.confirmed:
            if setup_result.confidence in (
                ConfidenceLabel.WEAK.value,
                ConfidenceLabel.DECENT.value,
            ):
                reasons.append("setup confirmed but confidence is not strong")

        move_label = getattr(move_potential_result, "score_label", "") if move_potential_result else ""
        if move_label in ("limited", "weak"):
            reasons.append(f"move potential is {move_label}")

        or_result = context.get("or_result")
        if or_result and getattr(or_result, "state", "") == "inside":
            reasons.append("price is still inside opening range")

        fib_result = context.get("fib_result")
        if fib_result:
            nearest = getattr(fib_result, "nearest_retracement", None)
            dist = _safe_float(getattr(fib_result, "distance_from_fib_pct", 0.0))
            if nearest and dist > 1.5 and not getattr(fib_result, "entry_confirmed_by_fib", False):
                reasons.append("waiting for cleaner pullback to Fibonacci level")

        return _dedupe(reasons)

    @staticmethod
    def _reject_or_wait_decision(blocks: list[str], context: dict) -> str:
        """
        Convert hard-block reasons into the best decision label.
        """
        joined = " ".join(blocks).lower()

        if "position already open" in joined:
            return TradeDecision.MANAGE_EXISTING_ONLY.value
        if "pullback" in joined or "fibonacci" in joined:
            return TradeDecision.WAIT_FOR_PULLBACK.value
        if "reclaim" in joined or "vwap" in joined:
            return TradeDecision.WAIT_FOR_RECLAIM.value
        return TradeDecision.REJECTED.value

    @staticmethod
    def _watch_decision(reasons: list[str], context: dict) -> str:
        """
        Convert watch reasons into the best decision label.
        """
        joined = " ".join(reasons).lower()

        if "pullback" in joined or "fibonacci" in joined:
            return TradeDecision.WAIT_FOR_PULLBACK.value
        if "reclaim" in joined:
            return TradeDecision.WAIT_FOR_RECLAIM.value
        return TradeDecision.WATCH.value

    # ── Explanation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _what_would_make_valid(reasons: list[str]) -> list[str]:
        """
        Translate rejection/watch reasons into actionable fixes.
        """
        fixes: list[str] = []

        for reason in reasons:
            r = reason.lower()

            if "setup score" in r or "setup is not confirmed" in r or "no setup" in r:
                fixes.append("Wait for a confirmed setup with stronger price action")
            elif "probability score" in r:
                fixes.append("Wait for probability to improve through volume, VWAP, structure, or stronger confirmation")
            elif "final trade quality" in r:
                fixes.append("Wait for overall setup/probability/risk-reward quality to improve")
            elif "risk/reward" in r:
                fixes.append("Wait for a better entry, tighter stop, or larger target")
            elif "stop" in r:
                fixes.append("Define a clear stop below the setup invalidation area")
            elif "target" in r:
                fixes.append("Define a clear target above entry with enough reward")
            elif "reclaim" in r:
                fixes.append("Wait for price to reclaim the failed level and hold it with volume")
            elif "fibonacci" in r or "pullback" in r:
                fixes.append("Wait for a cleaner pullback near a preferred Fibonacci level")
            elif "execution" in r or "spread" in r or "quote" in r:
                fixes.append("Wait for spread and quote quality to improve")
            elif "position already open" in r:
                fixes.append("Manage the existing open position instead of adding another")
            elif "safety" in r or "paper" in r or "live money" in r:
                fixes.append("Fix bot safety settings before allowing any trade")
            elif "risk guard" in r:
                fixes.append("Reduce account risk or wait until risk guard allows new trades")

        return _dedupe(fixes)

    @staticmethod
    def _merge_optional_warnings(decision: TradeDecisionResult, objects: list[object]) -> None:
        """
        Merge warnings/reasons from result objects into decision warnings.
        """
        for obj in objects:
            if not obj:
                continue

            warnings = getattr(obj, "warnings", None)
            if isinstance(warnings, list):
                decision.warnings.extend(warnings)

        decision.warnings = _dedupe(decision.warnings)
        decision.reasons = _dedupe(decision.reasons)
        decision.what_would_make_valid = _dedupe(decision.what_would_make_valid)


# ── Standalone convenience ────────────────────────────────────────────────────

def evaluate_trade_quality(
    settings: dict,
    ticker: str,
    setup_result: Optional[SetupResult],
    setup_score_result: Optional[object],
    probability_result: Optional[object],
    risk_reward_result: Optional[object],
    move_potential_result: Optional[object],
    execution_quality_result: Optional[object] = None,
    risk_result: Optional[object] = None,
    historical_edge_result: Optional[object] = None,
    context: Optional[dict] = None,
    position_size: Optional[PositionSize] = None,
) -> TradeDecisionResult:
    """
    Convenience wrapper so bot_runner.py can call a simple function.
    """
    gate = TradeQualityGate(settings)
    return gate.evaluate(
        ticker                   = ticker,
        setup_result             = setup_result,
        setup_score_result       = setup_score_result,
        probability_result       = probability_result,
        risk_reward_result       = risk_reward_result,
        move_potential_result    = move_potential_result,
        execution_quality_result = execution_quality_result,
        risk_result              = risk_result,
        historical_edge_result   = historical_edge_result,
        context                  = context,
        position_size            = position_size,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: object) -> Optional[float]:
    """Safely convert a value to float or None."""
    try:
        if value is None:
            return None
        f = float(value)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    """
    Deduplicate strings while preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
