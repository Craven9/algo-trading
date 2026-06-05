"""
src/core/logger.py — Centralized logging setup
Configures the root logger and named child loggers for every subsystem.
All modules should call get_logger(__name__) instead of logging.getLogger()
directly so formatting, handlers, and log levels stay consistent.

Log destinations (controlled by config/bot_settings.json → logging):
    - Console (StreamHandler)
    - Rotating file: logs/bot_activity.log
    - Rotating file: logs/error_log.log      (ERROR and above only)
    - Rotating file: logs/order_activity.log (order events only)
    - JSON file:     logs/rejection_log.json (trade rejections)
    - JSON file:     logs/exit_log.json      (trade exits)
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LOG_LEVEL   = "INFO"
DEFAULT_MAX_BYTES   = 50 * 1024 * 1024   # 50 MB
DEFAULT_BACKUP_COUNT = 5

# Named logger used internally by this module
_MODULE_LOG = "core.logger"

# Track whether setup has already run so calling setup() twice is safe
_initialized = False


# ── Formatters ────────────────────────────────────────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    """
    Human-readable console output with color coding by level.
    Falls back to plain text when the terminal does not support ANSI codes.
    """
    _RESET  = "\033[0m"
    _COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    _FMT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    _DATE = "%H:%M:%S"

    def __init__(self, use_color: bool = True):
        super().__init__(fmt=self._FMT, datefmt=self._DATE)
        self._use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._use_color:
            color = self._COLORS.get(record.levelno, "")
            return f"{color}{msg}{self._RESET}"
        return msg


class _FileFormatter(logging.Formatter):
    """Plain-text formatter for rotating log files."""
    _FMT  = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATE)


# ── JSON event log (rejections / exits) ───────────────────────────────────────

class _JsonFileHandler(logging.Handler):
    """
    Appends one JSON object per line to a file.
    Used for structured rejection and exit logs that the frontend and
    learning system can parse without regex.
    """

    def __init__(self, filepath: str):
        super().__init__()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._path = filepath

    # Standard LogRecord attributes we do NOT want to copy into the JSON output
    _SKIP_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs",
        "msg", "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName",
        # filter flags — internal use only
        "order_event", "rejection_event", "exit_event",
    })

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level":     record.levelname,
                "logger":    record.name,
                "message":   record.getMessage(),
            }
            # Merge any extra fields that were passed via extra={...}
            # logging copies extra keys directly onto the LogRecord instance.
            for key, val in record.__dict__.items():
                if key not in self._SKIP_ATTRS:
                    entry[key] = val
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            self.handleError(record)


# ── Named subsystem filters ───────────────────────────────────────────────────

class _SubsystemFilter(logging.Filter):
    """Only pass records whose logger name starts with a given prefix."""

    def __init__(self, prefix: str):
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefix)


class _OrderFilter(logging.Filter):
    """Pass records tagged with extra={'order_event': True}."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "order_event", False)


class _RejectionFilter(logging.Filter):
    """Pass records tagged with extra={'rejection_event': True}."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "rejection_event", False)


class _ExitFilter(logging.Filter):
    """Pass records tagged with extra={'exit_event': True}."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "exit_event", False)


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup(settings: Optional[dict] = None, force: bool = False) -> None:
    """
    Configure all log handlers.  Safe to call multiple times — subsequent
    calls are no-ops unless force=True.

    Args:
        settings: The full bot_settings dict (or just the "logging" sub-dict).
                  If None, safe defaults are used so the bot can log even
                  before config is fully loaded.
        force:    Tear down existing handlers and reconfigure from scratch.
    """
    global _initialized
    if _initialized and not force:
        return

    # Accept either the full settings dict or just the logging sub-dict
    if settings is None:
        cfg = {}
    elif "logging" in settings:
        cfg = settings["logging"]
    else:
        cfg = settings

    level_name  = cfg.get("log_level", DEFAULT_LOG_LEVEL).upper()
    level       = getattr(logging, level_name, logging.INFO)
    to_console  = cfg.get("log_to_console", True)
    to_file     = cfg.get("log_to_file", True)
    max_bytes   = int(cfg.get("max_log_size_mb", 50)) * 1024 * 1024
    backup_count = int(cfg.get("backup_count", DEFAULT_BACKUP_COUNT))

    # Resolved log file paths
    activity_log  = _resolve(cfg.get("bot_activity_log",  "logs/bot_activity.log"))
    error_log     = _resolve(cfg.get("error_log",         "logs/error_log.log"))
    order_log     = _resolve(cfg.get("order_activity_log","logs/order_activity.log"))
    rejection_log = _resolve(cfg.get("rejection_log",     "logs/rejection_log.json"))
    exit_log      = _resolve(cfg.get("exit_log",          "logs/exit_log.json"))

    root = logging.getLogger()
    # Remove any handlers from a previous setup() call
    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

    root.setLevel(level)

    # ── Console handler ───────────────────────────────────────────────────────
    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_ConsoleFormatter())
        root.addHandler(ch)

    if to_file:
        # ── Main rotating activity log ────────────────────────────────────────
        fh = logging.handlers.RotatingFileHandler(
            activity_log, maxBytes=max_bytes, backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(_FileFormatter())
        root.addHandler(fh)

        # ── Error-only rotating log ───────────────────────────────────────────
        eh = logging.handlers.RotatingFileHandler(
            error_log, maxBytes=max_bytes, backupCount=backup_count,
            encoding="utf-8",
        )
        eh.setLevel(logging.ERROR)
        eh.setFormatter(_FileFormatter())
        root.addHandler(eh)

        # ── Order event rotating log ──────────────────────────────────────────
        oh = logging.handlers.RotatingFileHandler(
            order_log, maxBytes=max_bytes, backupCount=backup_count,
            encoding="utf-8",
        )
        oh.setLevel(logging.DEBUG)
        oh.addFilter(_OrderFilter())
        oh.setFormatter(_FileFormatter())
        root.addHandler(oh)

        # ── Rejection JSON log ────────────────────────────────────────────────
        rh = _JsonFileHandler(rejection_log)
        rh.setLevel(logging.INFO)
        rh.addFilter(_RejectionFilter())
        root.addHandler(rh)

        # ── Exit JSON log ─────────────────────────────────────────────────────
        xh = _JsonFileHandler(exit_log)
        xh.setLevel(logging.INFO)
        xh.addFilter(_ExitFilter())
        root.addHandler(xh)

    # Quiet noisy third-party libraries
    for noisy in ("urllib3", "requests", "alpaca", "websocket", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True
    logging.getLogger(_MODULE_LOG).info(
        "[logger] Logging initialized — level=%s console=%s file=%s",
        level_name, to_console, to_file,
    )


def _resolve(rel_path: str) -> str:
    """
    Convert a relative log path (from bot_settings.json) to an absolute
    path anchored at the project root and ensure the parent dir exists.
    The project root is assumed to be two directories above this file:
        src/core/logger.py  →  ../../  →  project root
    """
    if os.path.isabs(rel_path):
        full = rel_path
    else:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        full = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Calls setup() with defaults if logging has not
    been initialized yet — ensures every module can log safely even if
    setup() was never called explicitly.

    Usage (in every other module):
        from src.core.logger import get_logger
        log = get_logger(__name__)
    """
    if not _initialized:
        setup()
    return logging.getLogger(name)


# ── Structured event helpers ──────────────────────────────────────────────────
# These helpers tag log records so the correct JSON handler picks them up.

def log_order_event(logger: logging.Logger, message: str, data: Optional[dict] = None) -> None:
    """
    Emit a structured order event to the order activity log.

    Args:
        logger:  The calling module's logger (get_logger(__name__)).
        message: Human-readable description of the order event.
        data:    Optional dict of structured fields merged into the log record.
    """
    extra = {"order_event": True}
    if data:
        extra.update(data)
    logger.info(message, extra=extra)


def log_rejection(logger: logging.Logger, ticker: str, reasons: list[str],
                  scores: Optional[dict] = None,
                  what_would_make_valid: Optional[list[str]] = None) -> None:
    """
    Emit a structured trade rejection to the rejection JSON log.

    Args:
        logger:                Logger from the calling module.
        ticker:                The ticker being rejected.
        reasons:               List of rejection reason strings.
        scores:                Optional dict of score values at rejection time.
        what_would_make_valid: Optional list of conditions that would flip to APPROVE.
    """
    data = {
        "ticker":                ticker,
        "reasons":               reasons,
        "scores":                scores or {},
        "what_would_make_valid": what_would_make_valid or [],
    }
    logger.info(
        "[REJECTION] %s — %s",
        ticker, "; ".join(reasons),
        extra={"rejection_event": True, **data},
    )


def log_exit(logger: logging.Logger, ticker: str, reason: str,
             pnl: Optional[float] = None, data: Optional[dict] = None) -> None:
    """
    Emit a structured trade exit to the exit JSON log.

    Args:
        logger: Logger from the calling module.
        ticker: The ticker being exited.
        reason: Exit reason string (e.g. "stop_loss_hit", "partial_profit").
        pnl:    Realized P&L in dollars at exit time.
        data:   Any additional structured fields.
    """
    payload = {
        "ticker": ticker,
        "reason": reason,
        "pnl":    pnl,
    }
    if data:
        payload.update(data)
    logger.info(
        "[EXIT] %s — %s — P&L: $%.2f",
        ticker, reason, pnl or 0.0,
        extra={"exit_event": True, **payload},
    )


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """
    Log an exception with full traceback at ERROR level.
    Convenience wrapper so callers don't have to import traceback.
    """
    tb = traceback.format_exc()
    logger.error("%s\n%s: %s\nTraceback:\n%s", message, type(exc).__name__, exc, tb)

# ---------------------------------------------------------------------------
# Compatibility helper used by bot_runner.py
# ---------------------------------------------------------------------------

def setup_logging(settings: dict | None = None) -> None:
    """
    Configure basic project logging.

    This compatibility helper lets newer modules call:
        from logger import setup_logging

    It is safe to call more than once.
    """
    import logging
    import os

    settings = settings or {}
    logging_cfg = settings.get("logging", {})

    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = (
        logging_cfg.get("bot_log_file")
        or logging_cfg.get("log_file")
        or os.path.join("logs", "bot.log")
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging initialized")
