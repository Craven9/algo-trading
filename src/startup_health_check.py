"""
src/startup_health_check.py — Project startup health checker
Runs a safe pre-flight check before starting the AI Trading Assistant.

This file helps confirm the bot is ready to run before bot_runner.py starts.
It checks:
  - Required folders exist
  - Required config sections exist
  - Safety settings are paper-only
  - Live-money trading is disabled
  - Alpaca environment variables are present when needed
  - Important modules/files exist
  - Dashboard paths are writable
  - Trade folders are writable

Responsibilities:
  - Detect missing folders/files early
  - Detect unsafe config settings early
  - Confirm dry-run / paper-only safety
  - Confirm dashboard/trade paths are writable
  - Return a structured HealthCheckResult
  - Never place trades or call broker order endpoints

Design rules:
  - This file does not scan
  - This file does not approve trades
  - This file does not place orders
  - This file is safe to run anytime
  - Warnings should explain exactly what to fix
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Required config sections ──────────────────────────────────────────────────

_REQUIRED_CONFIG_SECTIONS = [
    "mode",
    "scanner",
    "entry_rules",
    "risk",
    "exits",
    "paths",
]

_REQUIRED_MODULES = [
    "config_loader",
    "logger",
    "models",
    "utils",
    "time_utils",
    "data.market_data_provider",
    "data.indicator_engine",
    "scanner.momentum_scanner",
    "scanner.scanner_filter_engine",
    "scanner.candidate_ranker",
    "analysis.key_level_engine",
    "analysis.opening_range_analyzer",
    "analysis.liquidity_sweep_detector",
    "analysis.session_structure_analyzer",
    "analysis.fibonacci_strategy_engine",
    "analysis.move_potential_engine",
    "analysis.failed_reclaim_detector",
    "analysis.market_influence_filter",
    "setups",
    "scoring.confidence_labeler",
    "scoring.risk_reward_engine",
    "scoring.setup_score_engine",
    "scoring.probability_engine",
    "scoring.execution_quality_guard",
    "scoring.trade_quality_gate",
    "risk.position_sizer",
    "risk.position_tracker",
    "risk.account_risk_guard",
    "execution.order_executor",
    "execution.exit_manager",
    "execution.trade_logger",
    "execution.trade_status_updater",
    "learning.performance_tracker",
    "learning.backtest_reviewer",
    "learning.historical_edge_engine",
    "learning.learning_engine",
    "frontend.dashboard_state_writer",
    "frontend.dashboard_api",
    "frontend.bot_control_api",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class HealthCheckResult:
    """
    Startup health check result.
    """
    passed:          bool = True
    safe_to_run:     bool = True
    checks_run:      int  = 0
    errors:          list[str] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)
    passed_checks:   list[str] = field(default_factory=list)
    checked_at:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_pass(self, message: str) -> None:
        self.checks_run += 1
        self.passed_checks.append(message)

    def add_warning(self, message: str) -> None:
        self.checks_run += 1
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.checks_run += 1
        self.errors.append(message)
        self.passed = False
        self.safe_to_run = False

    def to_dict(self) -> dict:
        return {
            "passed":        self.passed,
            "safe_to_run":   self.safe_to_run,
            "checks_run":    self.checks_run,
            "errors":        self.errors,
            "warnings":      self.warnings,
            "passed_checks": self.passed_checks,
            "checked_at":    self.checked_at,
        }


# ── Health checker ────────────────────────────────────────────────────────────

class StartupHealthCheck:
    """
    Performs project startup health checks.

    Usage:
        checker = StartupHealthCheck(settings)
        result = checker.run()
    """

    def __init__(self, settings: dict, project_root: str | Path = "."):
        self._settings = settings
        self._root = Path(project_root)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, check_imports: bool = True) -> HealthCheckResult:
        """
        Run all health checks.

        Args:
            check_imports: Whether to import required modules.

        Returns:
            HealthCheckResult.
        """
        result = HealthCheckResult()

        self._check_config_sections(result)
        self._check_safety_settings(result)
        self._check_required_paths(result)
        self._check_writable_paths(result)
        self._check_environment_keys(result)

        if check_imports:
            self._check_required_imports(result)

        if result.errors:
            result.passed = False
            result.safe_to_run = False

        log.info(
            "[startup_health] passed=%s safe=%s errors=%d warnings=%d",
            result.passed,
            result.safe_to_run,
            len(result.errors),
            len(result.warnings),
        )
        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_config_sections(self, result: HealthCheckResult) -> None:
        """Check required bot_settings.json sections."""
        for section in _REQUIRED_CONFIG_SECTIONS:
            if section not in self._settings:
                result.add_error(f"Missing config section: {section}")
            else:
                result.add_pass(f"Config section present: {section}")

    def _check_safety_settings(self, result: HealthCheckResult) -> None:
        """Check paper-only safety settings."""
        mode = self._settings.get("mode", {})

        if bool(mode.get("allow_live_money", False)):
            result.add_error("allow_live_money is true — unsafe")
        else:
            result.add_pass("allow_live_money is false")

        if bool(mode.get("paper_trading_only", True)):
            result.add_pass("paper_trading_only is true")
        else:
            result.add_error("paper_trading_only is false — unsafe")

        if bool(mode.get("safety_lock", False)):
            result.add_warning("safety_lock is enabled — bot will not trade")
            result.safe_to_run = False
        else:
            result.add_pass("safety_lock is disabled")

        if bool(mode.get("dry_run", True)):
            result.add_warning("dry_run is enabled — orders will be simulated")
        else:
            result.add_pass("dry_run is disabled — Alpaca paper orders may submit")

        if bool(mode.get("bot_enabled", True)):
            result.add_pass("bot_enabled is true")
        else:
            result.add_warning("bot_enabled is false — bot loop will not run")
            result.safe_to_run = False

    def _check_required_paths(self, result: HealthCheckResult) -> None:
        """Create and check required data folders."""
        paths = self._settings.get("paths", {})

        required_dirs = [
            paths.get("trades_dir", "data/trades"),
            paths.get("open_trades_dir", "data/trades/open"),
            paths.get("closed_trades_dir", "data/trades/closed"),
            paths.get("dashboard_dir", "data/dashboard"),
            paths.get("learning_dir", "data/learning"),
        ]

        for raw in required_dirs:
            path = self._root / raw
            try:
                path.mkdir(parents=True, exist_ok=True)
                result.add_pass(f"Folder ready: {path}")
            except Exception as exc:
                result.add_error(f"Cannot create folder {path}: {exc}")

    def _check_writable_paths(self, result: HealthCheckResult) -> None:
        """Check dashboard/trade folders are writable."""
        paths = self._settings.get("paths", {})

        writable_dirs = [
            paths.get("open_trades_dir", "data/trades/open"),
            paths.get("closed_trades_dir", "data/trades/closed"),
            paths.get("dashboard_dir", "data/dashboard"),
            paths.get("learning_dir", "data/learning"),
        ]

        for raw in writable_dirs:
            folder = self._root / raw
            test_file = folder / ".write_test"

            try:
                folder.mkdir(parents=True, exist_ok=True)
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink(missing_ok=True)
                result.add_pass(f"Writable folder: {folder}")
            except Exception as exc:
                result.add_error(f"Folder not writable {folder}: {exc}")

    def _check_environment_keys(self, result: HealthCheckResult) -> None:
        """Check API environment variables."""
        alpaca_key = os.environ.get("ALPACA_API_KEY", "")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")

        mode = self._settings.get("mode", {})
        dry_run = bool(mode.get("dry_run", True))

        if alpaca_key and alpaca_secret:
            result.add_pass("Alpaca API environment variables found")
        elif dry_run:
            result.add_warning("Alpaca API keys missing, but dry_run is enabled")
        else:
            result.add_error("Alpaca API keys missing and dry_run is disabled")

        massive_key = (
            os.environ.get("MASSIVE_API_KEY", "")
            or os.environ.get("POLYGON_API_KEY", "")
        )
        if massive_key:
            result.add_pass("Massive/Polygon API key found")
        else:
            result.add_warning("Massive/Polygon API key missing — REST/WebSocket fallback may be unavailable")

    def _check_required_imports(self, result: HealthCheckResult) -> None:
        """
        Import key modules to catch missing files early.
        """
        for module_name in _REQUIRED_MODULES:
            try:
                importlib.import_module(module_name)
                result.add_pass(f"Import ok: {module_name}")
            except Exception as exc:
                result.add_error(f"Import failed: {module_name} — {exc}")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_startup_health_check(
    settings: dict,
    project_root: str | Path = ".",
    check_imports: bool = True,
) -> HealthCheckResult:
    """
    Convenience function for bot_runner.py or command-line use.
    """
    checker = StartupHealthCheck(settings, project_root=project_root)
    return checker.run(check_imports=check_imports)


# ── CLI helper ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run health check from command line:

        python -m src.startup_health_check
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        from config_loader import load_settings
        settings = load_settings()
    except Exception as exc:
        print(f"FAILED: could not load settings: {exc}")
        return

    result = run_startup_health_check(settings)

    print("AI Trading Assistant Startup Health Check")
    print("----------------------------------------")
    print(f"Passed: {result.passed}")
    print(f"Safe to run: {result.safe_to_run}")
    print(f"Checks run: {result.checks_run}")

    if result.errors:
        print()
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print()
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.passed_checks:
        print()
        print("Passed checks:")
        for check in result.passed_checks:
            print(f"  - {check}")


if __name__ == "__main__":
    main()
