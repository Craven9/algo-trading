"""
src/frontend/api_server.py — Combined frontend API server
Creates one FastAPI app for the AI Trading Assistant frontend.

This server combines:
  - dashboard_api.py for dashboard state, trades, performance, and learning
  - bot_control_api.py for safe start/pause/resume/stop/refresh controls

The frontend should connect to this one API instead of reaching into
internal bot modules directly.

Responsibilities:
  - Load bot_settings.json
  - Create the main FastAPI app
  - Add CORS for local frontend development
  - Mount dashboard read endpoints
  - Mount safe bot control endpoints
  - Provide health/status routes
  - Never place orders directly

Design rules:
  - This file does not scan
  - This file does not approve trades
  - This file does not place orders
  - This file only exposes safe API endpoints
  - Trading execution still must go through bot_runner.py,
    trade_quality_gate.py, and order_executor.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except Exception:
    FastAPI = None
    CORSMiddleware = None

from bot_control_api import create_bot_control_router
from dashboard_api import create_dashboard_app

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_SETTINGS_PATH = Path("config") / "bot_settings.json"


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(settings_path: str | Path = _DEFAULT_SETTINGS_PATH) -> object:
    """
    Create the combined FastAPI app.

    Args:
        settings_path: Path to config/bot_settings.json.

    Returns:
        FastAPI app.
    """
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

    settings = _load_settings(settings_path)

    # dashboard_api.py already creates a FastAPI app with dashboard routes.
    app = create_dashboard_app(settings)

    # Add control routes.
    control_router = create_bot_control_router(settings)
    app.include_router(control_router)

    # Add root route.
    @app.get("/")
    def root() -> dict:
        return {
            "ok": True,
            "service": "AI Trading Assistant API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/api")
    def api_root() -> dict:
        return {
            "ok": True,
            "endpoints": {
                "dashboard_state": "/api/dashboard/state",
                "dashboard_refresh": "/api/dashboard/refresh",
                "open_trades": "/api/trades/open",
                "closed_trades": "/api/trades/closed",
                "performance": "/api/performance",
                "learning": "/api/learning",
                "bot_control_state": "/api/bot/control-state",
                "bot_start": "/api/bot/start",
                "bot_pause": "/api/bot/pause",
                "bot_resume": "/api/bot/resume",
                "bot_stop": "/api/bot/stop",
            },
        }

    log.info("[api_server] FastAPI app created")
    return app


# ── Settings loader ───────────────────────────────────────────────────────────

def _load_settings(settings_path: str | Path) -> dict:
    """
    Load bot settings from JSON file.
    Returns safe defaults if file is missing/unreadable.
    """
    path = Path(settings_path)

    try:
        if not path.exists():
            log.warning("[api_server] Settings file missing: %s", path)
            return _default_settings()

        with path.open("r", encoding="utf-8") as f:
            settings = json.load(f)

        if not isinstance(settings, dict):
            log.warning("[api_server] Settings file did not contain a dict")
            return _default_settings()

        return settings

    except Exception as exc:
        log.error("[api_server] Failed to load settings %s: %s", path, exc)
        return _default_settings()


def _default_settings() -> dict:
    """
    Minimal safe settings so the API can still start.
    """
    return {
        "bot_name": "AI Trading Assistant",
        "version": "unknown",
        "mode": {
            "bot_enabled": False,
            "dry_run": True,
            "paper_trading_only": True,
            "allow_live_money": False,
            "safety_lock": True,
            "after_hours_enabled": False,
        },
        "frontend": {
            "cors_origins": [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ],
        },
        "paths": {
            "trades_dir": "data/trades",
            "open_trades_dir": "data/trades/open",
            "closed_trades_dir": "data/trades/closed",
            "dashboard_dir": "data/dashboard",
            "dashboard_state_path": "data/dashboard/dashboard_state.json",
            "bot_control_state_path": "data/dashboard/bot_control_state.json",
            "learning_dir": "data/learning",
            "learning_summary_path": "data/learning/learning_summary.json",
        },
    }


# ── Uvicorn entrypoint ────────────────────────────────────────────────────────

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
