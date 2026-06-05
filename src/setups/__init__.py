"""
src/setups/__init__.py - Setup detector registry
Runs all setup detectors and returns the best confirmed setup.

Your setup files use:
    detect(...)
"""

from __future__ import annotations

import logging
from typing import Optional

from models import SetupResult

log = logging.getLogger(__name__)


def run_all_setups(
    bars: list[dict],
    indicators: object,
    context: dict,
    settings: dict,
) -> dict[str, SetupResult]:
    results: dict[str, SetupResult] = {}

    detectors = [
        ("break_and_hold", "break_and_hold"),
        ("bottom_base", "bottom_base"),
        ("vwap_reclaim", "vwap_reclaim"),
        ("opening_range_breakout", "opening_range_breakout"),
        ("liquidity_sweep_reclaim", "liquidity_sweep_reclaim"),
        ("fibonacci_pullback", "fibonacci_pullback"),
    ]

    for result_key, module_name in detectors:
        try:
            module = __import__(f"setups.{module_name}", fromlist=["detect"])
            detector = getattr(module, "detect")

            result = _call_detector(
                detector=detector,
                bars=bars,
                indicators=indicators,
                context=context,
                settings=settings,
            )

            if result is not None:
                results[result_key] = result
            else:
                results[result_key] = _empty_result(result_key, "detector returned None")

        except Exception as exc:
            log.warning("[setups] %s failed: %s", result_key, exc)
            results[result_key] = _empty_result(result_key, f"detector failed: {exc}")

    return results


def best_setup(results: dict[str, SetupResult]) -> Optional[SetupResult]:
    if not results:
        return None

    confirmed = [
        result for result in results.values()
        if result is not None and getattr(result, "confirmed", False)
    ]

    if not confirmed:
        return None

    return max(confirmed, key=lambda r: float(getattr(r, "score", 0.0) or 0.0))


def run_setup_detection(
    bars: list[dict],
    indicators: object,
    context: dict,
    settings: dict,
) -> Optional[SetupResult]:
    results = run_all_setups(
        bars=bars,
        indicators=indicators,
        context=context,
        settings=settings,
    )
    return best_setup(results)


def _call_detector(
    detector,
    bars: list[dict],
    indicators: object,
    context: dict,
    settings: dict,
):
    attempts = [
        {"bars": bars, "indicators": indicators, "context": context, "settings": settings},
        {"bars": bars, "indicators": indicators, "context": context},
        {"bars": bars, "indicators": indicators, "settings": settings},
        {"bars": bars, "indicators": indicators},
        {"bars": bars, "context": context, "settings": settings},
        {"bars": bars, "context": context},
    ]

    for kwargs in attempts:
        try:
            return detector(**kwargs)
        except TypeError:
            continue

    positional_attempts = [
        (bars, indicators, context, settings),
        (bars, indicators, context),
        (bars, indicators),
        (bars, context),
        (bars,),
    ]

    for args in positional_attempts:
        try:
            return detector(*args)
        except TypeError:
            continue

    return detector()


def _empty_result(setup_name: str, reason: str) -> SetupResult:
    return SetupResult(
        setup_name=setup_name,
        confirmed=False,
        score=0.0,
        confidence="reject",
        entry_trigger=None,
        stop_area=None,
        target_area=None,
        reasons=[],
        warnings=[reason],
    )
