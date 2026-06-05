"""
src/frontend/dashboard_api.py — Frontend dashboard API
Provides a simple FastAPI backend for the AI Trading Assistant dashboard.

The frontend should be able to read bot state, open trades, closed trades,
performance, learning summaries, and mode settings without reaching into
internal bot modules directly.

Responsibilities:
  - Serve dashboard_state.json
  - Serve open trades
  - Serve closed trades
  - Serve performance summary
  - Serve learning summary
  - Serve current bot mode/safety settings
  - Provide simple refresh/status endpoints
  - Never place trades directly from dashboard read endpoints

Design rules:
  - This file is for dashboard read/control API only
  - This file does not scan by itself
  - This file does not approve trades
  - This file does not place orders
  - Trading commands should be routed through safe bot control functions
  - Missing files should return empty objects/lists, not crash
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except Exception:
    FastAPI = None
    CORSMiddleware = None

from dashboard_state_writer import DashboardStateWriter
from performance_tracker import PerformanceTracker
from learning_engine import LearningEngine

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_DASHBOARD_STATE = Path("data") / "dashboard" / "dashboard_state.json"


# ── API factory ───────────────────────────────────────────────────────────────

def create_dashboard_app(settings: dict) -> object:
    """
    Create and configure the dashboard FastAPI app.

    Args:
        settings: Full bot_settings dict.

    Returns:
        FastAPI app instance.

    Raises:
        RuntimeError if fastapi is not installed.
    """
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="AI Trading Assistant Dashboard API",
        version=str(settings.get("version", "1.0.0")),
    )

    _add_cors(app, settings)

    writer = DashboardStateWriter(settings)
    performance_tracker = PerformanceTracker(settings)
    learning_engine = LearningEngine(settings)

    # ── Health / status ───────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "service": "AI Trading Assistant Dashboard API",
        }

    @app.get("/api/status")
    def status() -> dict:
        state = writer.read_state()
        bot = state.get("bot", {}) if state else {}
        return {
            "status": bot.get("status", "unknown"),
            "mode": _mode_state(settings),
            "updated_at": bot.get("updated_at", ""),
        }

    # ── Dashboard state ───────────────────────────────────────────────────────

    @app.get("/api/dashboard/state")
    def dashboard_state() -> dict:
        state = writer.read_state()
        if state:
            return state

        # Return a generated empty state if file does not exist yet.
        return writer.build_state(bot_status="idle")

    @app.get("/api/dashboard/refresh")
    def dashboard_refresh() -> dict:
        """
        Rebuild dashboard state from current local files.
        This does not scan or trade; it only refreshes frontend data.
        """
        try:
            performance = performance_tracker.calculate()
        except Exception as exc:
            log.warning("[dashboard_api] performance refresh failed: %s", exc)
            performance = None

        try:
            learning = learning_engine.load_latest_summary()
        except Exception as exc:
            log.warning("[dashboard_api] learning refresh failed: %s", exc)
            learning = None

        path = writer.write_state(
            bot_status="idle",
            performance_summary=performance,
            learning_summary=learning,
        )

        return {
            "ok": True,
            "message": "dashboard state refreshed",
            "path": path,
        }

    # ── Trades ────────────────────────────────────────────────────────────────

    @app.get("/api/trades/open")
    def open_trades() -> dict:
        state = writer.build_state(bot_status="idle")
        return {
            "count": len(state["trades"]["open"]),
            "trades": state["trades"]["open"],
        }

    @app.get("/api/trades/closed")
    def closed_trades(limit: int = 50) -> dict:
        trades = writer._load_closed_trades(limit=limit)
        return {
            "count": len(trades),
            "trades": trades,
        }

    # ── Performance / learning ────────────────────────────────────────────────

    @app.get("/api/performance")
    def performance(limit: Optional[int] = None) -> dict:
        summary = performance_tracker.calculate(limit=limit)
        return summary.to_dict()

    @app.get("/api/learning")
    def learning() -> dict:
        summary = learning_engine.load_latest_summary()
        return summary.to_dict()

    @app.post("/api/learning/generate")
    def generate_learning(limit: Optional[int] = None) -> dict:
        summary = learning_engine.generate_and_save(limit=limit)
        return summary.to_dict()

    # ── Mode / settings ───────────────────────────────────────────────────────

    @app.get("/api/mode")
    def mode() -> dict:
        return _mode_state(settings)

    @app.get("/api/settings/safe")
    def safe_settings() -> dict:
        """
        Return safe dashboard settings without secrets.
        """
        return {
            "bot_name": settings.get("bot_name", "AI Trading Assistant"),
            "version": settings.get("version", ""),
            "mode": _mode_state(settings),
            "scanner": settings.get("scanner", {}),
            "entry_rules": settings.get("entry_rules", {}),
            "risk": _risk_state(settings),
            "exits": settings.get("exits", {}),
        }

    return app


# ── Optional module-level app helper ──────────────────────────────────────────

def build_app_from_settings_file(settings_path: str = "config/bot_settings.json") -> object:
    """
    Build FastAPI app directly from a settings file.
    Useful for uvicorn:

        uvicorn dashboard_api:app --reload

    If using this helper manually, call:
        app = build_app_from_settings_file()
    """
    settings = _read_json(Path(settings_path))
    return create_dashboard_app(settings)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_cors(app: object, settings: dict) -> None:
    """
    Add CORS middleware for local frontend development.
    """
    if CORSMiddleware is None:
        return

    frontend = settings.get("frontend", {})
    origins = frontend.get("cors_origins", [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _mode_state(settings: dict) -> dict:
    mode = settings.get("mode", {})
    return {
        "bot_enabled": mode.get("bot_enabled", True),
        "dry_run": mode.get("dry_run", True),
        "paper_trading_only": mode.get("paper_trading_only", True),
        "allow_live_money": mode.get("allow_live_money", False),
        "safety_lock": mode.get("safety_lock", False),
        "after_hours_enabled": mode.get("after_hours_enabled", False),
    }


def _risk_state(settings: dict) -> dict:
    """
    Return risk settings safe for dashboard display.
    """
    risk = dict(settings.get("risk", {}))
    hidden_keys = {
        "api_key",
        "secret_key",
        "alpaca_api_key",
        "alpaca_secret_key",
    }

    for key in list(risk.keys()):
        if key.lower() in hidden_keys:
            risk[key] = "***"

    return risk


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[dashboard_api] Failed to read %s: %s", path, exc)
        return {}
