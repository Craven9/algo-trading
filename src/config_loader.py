"""
src/core/config_loader.py — Centralized configuration loader
Loads, validates, and provides access to bot_settings.json.
All modules should import settings through this module — never read
the JSON file directly in other source files.
"""

import json
import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

# Default path — can be overridden via the BOT_SETTINGS_PATH env variable
DEFAULT_SETTINGS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "config", "bot_settings.json"
))

# Module-level cache so the file is only read once per process
_settings: Optional[dict] = None
_settings_path: Optional[str] = None


def _project_root() -> str:
    """
    Return the project root based on the loaded settings path.

    For this project layout:
        C:\Projects\Algo-Bot-Trader\src\config_loader.py
        C:\Projects\Algo-Bot-Trader\config\bot_settings.json

    The project root is:
        C:\Projects\Algo-Bot-Trader
    """
    settings_path = _settings_path or DEFAULT_SETTINGS_PATH
    config_dir = os.path.dirname(os.path.abspath(settings_path))
    return os.path.abspath(os.path.join(config_dir, ".."))


# ── Required top-level sections ───────────────────────────────────────────────
# Every section listed here must exist in bot_settings.json.
# A missing section raises a hard error at startup so problems are caught
# before any market data is fetched or any order is attempted.

REQUIRED_SECTIONS = [
    "mode",
    "scanner",
    "indicators",
    "entry_rules",
    "risk",
    "execution",
    "setups",
    "fibonacci_strategy",
    "exits",
    "opening_range",
    "learning",
    "frontend",
    "logging",
    "session",
    "watchlist",
    "paths",
]


# ── Loader ────────────────────────────────────────────────────────────────────

def load(path: Optional[str] = None, force_reload: bool = False) -> dict:
    """
    Load bot_settings.json and return the full settings dict.

    Results are cached after the first successful load.  Pass
    force_reload=True to re-read from disk (useful in tests or when
    the file has been edited while the bot is running).

    Args:
        path:         Explicit path to bot_settings.json.  Falls back to
                      the BOT_SETTINGS_PATH environment variable, then to
                      DEFAULT_SETTINGS_PATH.
        force_reload: If True, discard the cached copy and reload from disk.

    Returns:
        The full settings dict.

    Raises:
        FileNotFoundError: If the settings file cannot be found.
        ValueError:        If the JSON is malformed or required sections
                           are missing.
    """
    global _settings, _settings_path

    if _settings is not None and not force_reload:
        return _settings

    resolved_path = (
        path
        or os.environ.get("BOT_SETTINGS_PATH")
        or DEFAULT_SETTINGS_PATH
    )
    resolved_path = os.path.abspath(resolved_path)

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            f"[config_loader] bot_settings.json not found at: {resolved_path}"
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[config_loader] bot_settings.json is not valid JSON: {exc}"
            ) from exc

    _validate(data, resolved_path)

    _settings = data
    _settings_path = resolved_path
    log.info("[config_loader] Settings loaded from %s", resolved_path)
    return _settings


def _validate(data: dict, path: str) -> None:
    """
    Raise ValueError if any required section is missing.
    Logs a warning for unrecognised top-level keys (future-proofing).
    """
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ValueError(
            f"[config_loader] bot_settings.json at '{path}' is missing "
            f"required sections: {missing}"
        )

    # Safety-critical hard checks
    mode = data.get("mode", {})
    if mode.get("allow_live_money", False) and mode.get("paper_trading_only", True):
        raise ValueError(
            "[config_loader] Conflicting mode settings: "
            "'allow_live_money' is true but 'paper_trading_only' is also true."
        )

    known = set(REQUIRED_SECTIONS) | {"_comment"}
    unknown = set(data.keys()) - known
    if unknown:
        log.warning(
            "[config_loader] Unrecognised top-level keys in bot_settings.json "
            "(they will be ignored): %s", sorted(unknown)
        )


# ── Section accessors ─────────────────────────────────────────────────────────
# Each accessor loads settings if not already cached, then returns the
# requested section.  Callers never need to know the file structure.

def get(section: str, key: Optional[str] = None, default: Any = None) -> Any:
    """
    General-purpose getter.

    Usage:
        get("risk")                          → full risk dict
        get("risk", "max_open_positions")    → 5
        get("risk", "nonexistent", 99)       → 99
    """
    cfg = load()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data
    return section_data.get(key, default)


def mode() -> dict:
    return load().get("mode", {})


def scanner() -> dict:
    return load().get("scanner", {})


def indicators() -> dict:
    return load().get("indicators", {})


def entry_rules() -> dict:
    return load().get("entry_rules", {})


def risk() -> dict:
    return load().get("risk", {})


def execution() -> dict:
    return load().get("execution", {})


def setups() -> dict:
    return load().get("setups", {})


def fibonacci_strategy() -> dict:
    return load().get("fibonacci_strategy", {})


def exits() -> dict:
    return load().get("exits", {})


def opening_range() -> dict:
    return load().get("opening_range", {})


def learning() -> dict:
    return load().get("learning", {})


def frontend() -> dict:
    return load().get("frontend", {})


def logging_cfg() -> dict:
    return load().get("logging", {})


def session() -> dict:
    return load().get("session", {})


def watchlist() -> dict:
    return load().get("watchlist", {})


def paths() -> dict:
    return load().get("paths", {})


# ── Safety helpers ────────────────────────────────────────────────────────────

def is_paper_trading() -> bool:
    """True when paper_trading_only is set and allow_live_money is false."""
    m = mode()
    return m.get("paper_trading_only", True) and not m.get("allow_live_money", False)


def is_dry_run() -> bool:
    return mode().get("dry_run", True)


def is_safety_locked() -> bool:
    return mode().get("safety_lock", False)


def is_bot_enabled() -> bool:
    return mode().get("bot_enabled", True)


def trading_is_safe() -> tuple[bool, list[str]]:
    """
    Master safety check called at startup and before every order.

    Returns:
        (True, [])                  — safe to proceed
        (False, [reason, ...])      — blocked; list explains why
    """
    reasons: list[str] = []

    if not is_paper_trading():
        reasons.append("paper_trading_only is not active — live money risk detected")

    if is_safety_locked():
        reasons.append("safety_lock is enabled in mode config")

    if not is_bot_enabled():
        reasons.append("bot_enabled is false in mode config")

    if mode().get("allow_live_money", False):
        reasons.append("allow_live_money is true — refusing to proceed")

    return (len(reasons) == 0, reasons)


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_path(key: str) -> str:
    """
    Return the absolute path for a named path key from the [paths] section.
    Creates the directory if it does not exist.

    Example:
        resolve_path("open_trades_dir") → "/abs/path/to/trades/open/"
    """
    cfg   = load()
    root  = _project_root()
    rel   = cfg.get("paths", {}).get(key, "")
    if not rel:
        raise KeyError(f"[config_loader] Path key '{key}' not found in settings[paths]")
    full = os.path.join(root, rel)
    os.makedirs(full, exist_ok=True)
    return full


def resolve_log_path(key: str) -> str:
    """
    Return the absolute path for a named log file from the [logging] section.
    Creates the parent directory if it does not exist.
    """
    cfg  = load()
    root = _project_root()
    rel  = cfg.get("logging", {}).get(key, "")
    if not rel:
        raise KeyError(f"[config_loader] Log key '{key}' not found in settings[logging]")
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


# ── Compatibility helpers ─────────────────────────────────────────────────────

def load_settings(config_path: Optional[str] = None, force_reload: bool = False) -> dict:
    """
    Compatibility wrapper used by newer modules.

    Existing code can keep using:
        config_loader.load()

    Newer files can use:
        from config_loader import load_settings

    Args:
        config_path: Optional explicit path to bot_settings.json.
        force_reload: If True, reload settings from disk.

    Returns:
        Full bot settings dict.
    """
    return load(path=config_path, force_reload=force_reload)


# ── Reload / introspection ────────────────────────────────────────────────────

def reload() -> dict:
    """Force a fresh read from disk and return updated settings."""
    return load(force_reload=True)


def current_path() -> Optional[str]:
    """Return the path the settings were loaded from, or None if not yet loaded."""
    return _settings_path


def dump() -> str:
    """Return a pretty-printed JSON string of the current settings (for logging/debug)."""
    return json.dumps(load(), indent=2)
