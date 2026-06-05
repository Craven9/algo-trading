"""
src/scanner/manual_watchlist_loader.py — Manual watchlist loader
Loads and validates the user-curated ticker list from
watchlist/manual_watchlist.json.

Design rules:
  - A ticker being on the watchlist means ONLY: "this ticker is allowed
    to be analyzed."  It is NOT a buy signal.
  - The loader validates ticker format but does NOT filter by price,
    volume, or any market condition — that is the scanner's job.
  - Tickers can be annotated with notes and priority levels for
    dashboard display.
  - The file is re-read on every call to get() so live edits are
    picked up without restarting the bot.

Watchlist file schema (watchlist/manual_watchlist.json):
    {
      "tickers": [
        {
          "symbol":   "ABCD",
          "notes":    "earnings play — watching for VWAP reclaim",
          "priority": "high",       // "high" | "normal" | "low"
          "enabled":  true
        },
        ...
      ],
      "last_updated": "2026-06-02T10:00:00Z"
    }

    Shorthand also supported (list of plain strings):
    {
      "tickers": ["ABCD", "EFGH", "WXYZ"]
    }
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils import read_json, write_json

log = logging.getLogger(__name__)

# Valid ticker format: 1–5 uppercase letters (standard US equities)
_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")

# Default path — resolved relative to project root
_DEFAULT_PATH = os.path.join("watchlist", "manual_watchlist.json")


# ── Watchlist entry ───────────────────────────────────────────────────────────

@dataclass
class WatchlistEntry:
    """A single entry in the manual watchlist."""
    symbol:   str
    notes:    str  = ""
    priority: str  = "normal"   # "high" | "normal" | "low"
    enabled:  bool = True

    def to_dict(self) -> dict:
        return {
            "symbol":   self.symbol,
            "notes":    self.notes,
            "priority": self.priority,
            "enabled":  self.enabled,
        }


# ── Loader ────────────────────────────────────────────────────────────────────

class ManualWatchlistLoader:
    """
    Loads, validates, and manages the manual watchlist file.

    Usage:
        loader  = ManualWatchlistLoader(settings)
        tickers = loader.get_tickers()         # ["ABCD", "EFGH"]
        entries = loader.get_entries()         # [WatchlistEntry, ...]
    """

    def __init__(self, settings: dict):
        self._settings = settings
        watchlist_cfg  = settings.get("watchlist", {})
        self._path     = watchlist_cfg.get(
            "manual_watchlist_path", _DEFAULT_PATH
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_tickers(self) -> list[str]:
        """
        Return a list of enabled ticker symbols from the watchlist.
        Re-reads the file on every call so live edits are picked up.

        Returns:
            List of uppercase ticker strings.  Empty list when the file
            is missing, empty, or contains no valid enabled tickers.
        """
        entries = self.get_entries()
        return [e.symbol for e in entries if e.enabled]

    def get_entries(self) -> list[WatchlistEntry]:
        """
        Return all WatchlistEntry objects from the file (enabled and disabled).
        Re-reads the file on every call.
        """
        raw = read_json(self._path, default=None)

        if raw is None:
            log.debug(
                "[watchlist] %s not found — returning empty watchlist", self._path
            )
            return []

        return self._parse(raw)

    def get_high_priority(self) -> list[str]:
        """Return only tickers marked as high priority."""
        return [
            e.symbol for e in self.get_entries()
            if e.enabled and e.priority == "high"
        ]

    def add(self, symbol: str, notes: str = "",
             priority: str = "normal") -> bool:
        """
        Add a ticker to the watchlist file.

        Args:
            symbol:   Ticker symbol (will be uppercased and validated).
            notes:    Optional notes string.
            priority: "high" | "normal" | "low"

        Returns:
            True on success, False on validation failure or write error.
        """
        symbol = symbol.strip().upper()
        if not _is_valid_ticker(symbol):
            log.warning("[watchlist] Invalid ticker format: %s", symbol)
            return False

        raw = read_json(self._path, default={"tickers": []})
        entries = self._parse(raw)

        # Avoid duplicates
        existing = {e.symbol for e in entries}
        if symbol in existing:
            log.info("[watchlist] %s already in watchlist", symbol)
            return True

        entries.append(WatchlistEntry(
            symbol=symbol, notes=notes, priority=priority, enabled=True
        ))

        return self._save(entries)

    def remove(self, symbol: str) -> bool:
        """
        Remove a ticker from the watchlist file.

        Returns:
            True on success, False on write error.
        """
        symbol = symbol.strip().upper()
        raw    = read_json(self._path, default={"tickers": []})
        entries = [
            e for e in self._parse(raw) if e.symbol != symbol
        ]
        log.info("[watchlist] Removed %s from watchlist", symbol)
        return self._save(entries)

    def enable(self, symbol: str) -> bool:
        """Enable a ticker that was previously disabled."""
        return self._set_enabled(symbol, True)

    def disable(self, symbol: str) -> bool:
        """Disable a ticker without removing it from the file."""
        return self._set_enabled(symbol, False)

    def count(self) -> int:
        """Return the number of enabled tickers."""
        return len(self.get_tickers())

    def is_empty(self) -> bool:
        return self.count() == 0

    def path(self) -> str:
        """Return the resolved file path."""
        return self._path

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw: dict | list) -> list[WatchlistEntry]:
        """
        Parse the raw JSON (dict or list) into WatchlistEntry objects.
        Handles both the full schema and the shorthand list-of-strings format.
        """
        # Support plain list: ["ABCD", "EFGH"]
        if isinstance(raw, list):
            return self._parse_list(raw)

        # Support dict with "tickers" key
        if not isinstance(raw, dict):
            log.warning("[watchlist] Unexpected file format — expected dict or list")
            return []

        tickers_raw = raw.get("tickers", [])

        if not tickers_raw:
            return []

        # Shorthand: tickers is a list of strings
        if tickers_raw and isinstance(tickers_raw[0], str):
            return self._parse_list(tickers_raw)

        # Full schema: tickers is a list of dicts
        entries: list[WatchlistEntry] = []
        for item in tickers_raw:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol", "").strip().upper()
            if not _is_valid_ticker(symbol):
                log.warning("[watchlist] Skipping invalid ticker: %r", symbol)
                continue
            entries.append(WatchlistEntry(
                symbol   = symbol,
                notes    = str(item.get("notes", "")),
                priority = str(item.get("priority", "normal")),
                enabled  = bool(item.get("enabled", True)),
            ))

        log.debug("[watchlist] Loaded %d entries from %s", len(entries), self._path)
        return entries

    def _parse_list(self, raw_list: list) -> list[WatchlistEntry]:
        """Parse a plain list of ticker strings into WatchlistEntry objects."""
        entries: list[WatchlistEntry] = []
        for item in raw_list:
            symbol = str(item).strip().upper()
            if not _is_valid_ticker(symbol):
                log.warning("[watchlist] Skipping invalid ticker: %r", symbol)
                continue
            entries.append(WatchlistEntry(symbol=symbol))
        return entries

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save(self, entries: list[WatchlistEntry]) -> bool:
        """Write the current entry list back to the watchlist file."""
        data = {
            "tickers":      [e.to_dict() for e in entries],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        ok = write_json(self._path, data)
        if ok:
            log.info(
                "[watchlist] Saved %d entries to %s", len(entries), self._path
            )
        else:
            log.error("[watchlist] Failed to save watchlist to %s", self._path)
        return ok

    def _set_enabled(self, symbol: str, enabled: bool) -> bool:
        symbol = symbol.strip().upper()
        raw    = read_json(self._path, default={"tickers": []})
        entries = self._parse(raw)
        changed = False
        for e in entries:
            if e.symbol == symbol:
                e.enabled = enabled
                changed = True
        if not changed:
            log.warning("[watchlist] %s not found in watchlist", symbol)
            return False
        return self._save(entries)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_valid_ticker(symbol: str) -> bool:
    """True for 1–5 uppercase ASCII letters (standard US equity format)."""
    return bool(symbol and _TICKER_PATTERN.match(symbol))


def load_tickers_from_file(path: str) -> list[str]:
    """
    Convenience function — load enabled ticker symbols from any watchlist
    file path without instantiating ManualWatchlistLoader.
    """
    raw = read_json(path, default=None)
    if raw is None:
        return []

    loader = ManualWatchlistLoader.__new__(ManualWatchlistLoader)
    loader._path = path
    return [e.symbol for e in loader._parse(raw) if e.enabled]
