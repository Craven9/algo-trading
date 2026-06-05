"""
src/frontend/bot_control_api.py — Safe bot control API helpers
Provides safe backend control endpoints for the frontend dashboard.

The dashboard needs buttons like:
  - Start bot process
  - Pause bot
  - Resume bot
  - Stop bot
  - Refresh live dashboard state
  - View/toggle dry-run intent safely

This module stores control state in a local JSON file that bot_runner.py
can read during its loop.  It does not directly place trades or bypass the
trade quality gate.

Responsibilities:
  - Store bot control state in JSON
  - Expose safe FastAPI routes for dashboard buttons
  - Allow pause/resume/stop/start intent flags
  - Record requested dry-run state
  - Keep safety lock and paper-only rules visible
  - Provide a clean state object for bot_runner.py
  - Never submit orders

Design rules:
  - This file does not trade
  - This file does not scan by itself
  - This file does not approve trades
  - This file only writes control intent
  - bot_runner.py is still responsible for acting on control state
  - Live-money trading cannot be enabled from this API
  - Safety lock cannot be disabled from this API
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from fastapi import APIRouter
except Exception:
    APIRouter = None

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_CONTROL_PATH = Path("data") / "dashboard" / "bot_control_state.json"


# ── Control state manager ─────────────────────────────────────────────────────

class BotControlState:
    """
    Reads and writes safe bot control state.

    Usage:
        control = BotControlState(settings)
        control.request_pause()
        state = control.read()
    """

    def __init__(self, settings: dict):
        self._settings = settings
        paths = settings.get("paths", {})

        self._state_path = Path(
            paths.get("bot_control_state_path", str(_DEFAULT_CONTROL_PATH))
        )
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self) -> dict:
        """
        Read control state.  If missing, create a default state.
        """
        state = _read_json(self._state_path)
        if not state:
            state = self._default_state()
            self.write(state)
        return state

    def write(self, state: dict) -> str:
        """
        Write control state to disk.
        """
        state["updated_at"] = _now_iso()
        _write_json(self._state_path, state)
        return str(self._state_path)

    def request_start(self) -> dict:
        """
        Request bot loop to start/resume.
        """
        state = self.read()
        state["requested_action"] = "start"
        state["bot_should_run"] = True
        state["paused"] = False
        state["stop_requested"] = False
        state["message"] = "Start requested from dashboard"
        self.write(state)
        return state

    def request_pause(self) -> dict:
        """
        Request bot loop to pause.
        """
        state = self.read()
        state["requested_action"] = "pause"
        state["bot_should_run"] = False
        state["paused"] = True
        state["stop_requested"] = False
        state["message"] = "Pause requested from dashboard"
        self.write(state)
        return state

    def request_resume(self) -> dict:
        """
        Request bot loop to resume.
        """
        state = self.read()
        state["requested_action"] = "resume"
        state["bot_should_run"] = True
        state["paused"] = False
        state["stop_requested"] = False
        state["message"] = "Resume requested from dashboard"
        self.write(state)
        return state

    def request_stop(self) -> dict:
        """
        Request bot loop to stop gracefully.
        """
        state = self.read()
        state["requested_action"] = "stop"
        state["bot_should_run"] = False
        state["paused"] = False
        state["stop_requested"] = True
        state["message"] = "Stop requested from dashboard"
        self.write(state)
        return state

    def request_refresh(self) -> dict:
        """
        Request dashboard/live state refresh.
        """
        state = self.read()
        state["refresh_requested"] = True
        state["requested_action"] = "refresh"
        state["message"] = "Refresh requested from dashboard"
        self.write(state)
        return state

    def clear_refresh_request(self) -> dict:
        """
        Clear refresh_requested after bot_runner.py handles it.
        """
        state = self.read()
        state["refresh_requested"] = False
        self.write(state)
        return state

    def set_dry_run_requested(self, dry_run: bool) -> dict:
        """
        Store requested dry-run state.

        This does not directly rewrite bot_settings.json.  bot_runner.py
        may read this and choose whether to apply it safely.
        """
        state = self.read()
        state["requested_dry_run"] = bool(dry_run)
        state["requested_action"] = "set_dry_run"
        state["message"] = f"Dry-run requested: {bool(dry_run)}"
        self.write(state)
        return state

    def runtime_allows_loop(self) -> tuple[bool, str]:
        """
        Bot loop helper.  Returns whether bot_runner.py should continue running.
        """
        state = self.read()

        if state.get("stop_requested"):
            return False, "stop requested"
        if state.get("paused"):
            return False, "bot paused"
        if not state.get("bot_should_run", False):
            return False, "bot_should_run is false"

        mode = self._settings.get("mode", {})
        if not bool(mode.get("bot_enabled", True)):
            return False, "bot_enabled is false"
        if bool(mode.get("safety_lock", False)):
            return False, "safety_lock is enabled"

        return True, "bot loop allowed"

    # ── Internal state ────────────────────────────────────────────────────────

    def _default_state(self) -> dict:
        """
        Build default control state.
        """
        mode = self._settings.get("mode", {})
        return {
            "bot_should_run": False,
            "paused": False,
            "stop_requested": False,
            "refresh_requested": False,
            "requested_action": "none",
            "requested_dry_run": bool(mode.get("dry_run", True)),
            "paper_trading_only": bool(mode.get("paper_trading_only", True)),
            "allow_live_money": bool(mode.get("allow_live_money", False)),
            "safety_lock": bool(mode.get("safety_lock", False)),
            "message": "Default control state created",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }


# ── FastAPI router factory ────────────────────────────────────────────────────

def create_bot_control_router(settings: dict) -> object:
    """
    Create FastAPI router for bot control endpoints.

    These endpoints only write control intent.  They do not place trades.
    """
    if APIRouter is None:
        raise RuntimeError("FastAPI is not installed. Install with: pip install fastapi uvicorn")

    router = APIRouter(prefix="/api/bot", tags=["bot-control"])
    control = BotControlState(settings)

    @router.get("/control-state")
    def control_state() -> dict:
        return control.read()

    @router.post("/start")
    def start() -> dict:
        return {
            "ok": True,
            "state": control.request_start(),
        }

    @router.post("/pause")
    def pause() -> dict:
        return {
            "ok": True,
            "state": control.request_pause(),
        }

    @router.post("/resume")
    def resume() -> dict:
        return {
            "ok": True,
            "state": control.request_resume(),
        }

    @router.post("/stop")
    def stop() -> dict:
        return {
            "ok": True,
            "state": control.request_stop(),
        }

    @router.post("/refresh")
    def refresh() -> dict:
        return {
            "ok": True,
            "state": control.request_refresh(),
        }

    @router.post("/dry-run/on")
    def dry_run_on() -> dict:
        return {
            "ok": True,
            "state": control.set_dry_run_requested(True),
        }

    @router.post("/dry-run/off")
    def dry_run_off() -> dict:
        """
        Request dry-run off.

        This does not enable live money.  It only requests that bot_runner.py
        move from dry-run simulation to Alpaca paper-order submission if the
        rest of the safety settings allow it.
        """
        mode = settings.get("mode", {})
        if bool(mode.get("allow_live_money", False)):
            return {
                "ok": False,
                "error": "allow_live_money is true — refusing to modify dry-run state",
                "state": control.read(),
            }

        if not bool(mode.get("paper_trading_only", True)):
            return {
                "ok": False,
                "error": "paper_trading_only is not active — refusing to turn dry-run off",
                "state": control.read(),
            }

        return {
            "ok": True,
            "state": control.set_dry_run_requested(False),
        }

    return router


# ── Convenience helpers for bot_runner.py ─────────────────────────────────────

def read_bot_control_state(settings: dict) -> dict:
    """
    Convenience function for bot_runner.py.
    """
    control = BotControlState(settings)
    return control.read()


def bot_loop_allowed(settings: dict) -> tuple[bool, str]:
    """
    Convenience function for bot_runner.py loop checks.
    """
    control = BotControlState(settings)
    return control.runtime_allows_loop()


def clear_dashboard_refresh_request(settings: dict) -> dict:
    """
    Convenience function after refresh is handled.
    """
    control = BotControlState(settings)
    return control.clear_refresh_request()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[bot_control] Failed to read %s: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
