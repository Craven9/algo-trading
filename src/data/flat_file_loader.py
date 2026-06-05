"""
src/data/flat_file_loader.py — Local flat-file market data loader
Loads market data from local CSV/JSON files for fallback, testing,
backtesting, and offline development.

The bot should be able to run tests and backtest reviews without always
depending on live APIs.  This module provides a safe local data path for:
  - Historical candle files
  - Scanner universe files
  - Watchlist-style ticker lists
  - Cached snapshot files
  - Backtest datasets

Responsibilities:
  - Load OHLCV bars from CSV or JSON
  - Normalize bars into the bot's standard schema
  - Load ticker universes from CSV, JSON, or TXT
  - Load cached snapshot-style data
  - Fail safely with empty lists/dicts
  - Never place trades or approve trades

Design rules:
  - This file only reads local files
  - This file does not scan by itself
  - This file does not score trades
  - This file does not place orders
  - Missing files return empty results
  - Output should match the same bar shape used by candle_builder.py:
      {"t","timestamp","o","h","l","c","v"}
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Loader ────────────────────────────────────────────────────────────────────

class FlatFileLoader:
    """
    Loads local market data files.

    Usage:
        loader = FlatFileLoader(settings)
        bars = loader.load_bars("data/backtests/AAPL_1m.csv")
        tickers = loader.load_ticker_universe("data/universe/small_caps.txt")
    """

    def __init__(self, settings: dict):
        self._settings = settings
        paths = settings.get("paths", {})

        self._data_dir = Path(paths.get("data_dir", "data"))
        self._flat_file_dir = Path(
            paths.get("flat_file_dir", str(self._data_dir / "flat_files"))
        )

        self._flat_file_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_bars(self, file_path: str | Path) -> list[dict]:
        """
        Load OHLCV bars from CSV or JSON.

        Args:
            file_path: Path to CSV or JSON file.

        Returns:
            List of normalized bars.
        """
        path = self._resolve(file_path)

        if not path.exists():
            log.warning("[flat_file_loader] bar file missing: %s", path)
            return []

        if path.suffix.lower() == ".csv":
            return self._load_bars_csv(path)

        if path.suffix.lower() == ".json":
            return self._load_bars_json(path)

        log.warning("[flat_file_loader] unsupported bar file type: %s", path)
        return []

    def load_ticker_universe(self, file_path: str | Path) -> list[str]:
        """
        Load ticker symbols from TXT, CSV, or JSON.

        Supported:
          TXT: one ticker per line
          CSV: column named ticker/symbol, or first column
          JSON: list[str] or list[dict] with ticker/symbol
        """
        path = self._resolve(file_path)

        if not path.exists():
            log.warning("[flat_file_loader] ticker universe file missing: %s", path)
            return []

        suffix = path.suffix.lower()

        if suffix == ".txt":
            return self._load_tickers_txt(path)

        if suffix == ".csv":
            return self._load_tickers_csv(path)

        if suffix == ".json":
            return self._load_tickers_json(path)

        log.warning("[flat_file_loader] unsupported ticker file type: %s", path)
        return []

    def load_snapshot_file(self, file_path: str | Path) -> dict:
        """
        Load cached snapshot-style data from JSON.
        """
        path = self._resolve(file_path)

        if not path.exists():
            log.warning("[flat_file_loader] snapshot file missing: %s", path)
            return {}

        if path.suffix.lower() != ".json":
            log.warning("[flat_file_loader] snapshots must be JSON: %s", path)
            return {}

        data = _read_json(path)
        return data if isinstance(data, dict) else {}

    def save_bars(self, file_path: str | Path, bars: list[dict]) -> str:
        """
        Save bars to JSON file for later testing/backtesting.
        """
        path = self._resolve(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        normalized = [normalize_bar(b) for b in bars]
        _write_json(path, normalized)

        log.info("[flat_file_loader] saved %d bars to %s", len(normalized), path)
        return str(path)

    # ── Bar loaders ───────────────────────────────────────────────────────────

    def _load_bars_csv(self, path: Path) -> list[dict]:
        """Load bars from CSV."""
        bars: list[dict] = []

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bar = normalize_bar(row)
                    if _valid_bar(bar):
                        bars.append(bar)
        except Exception as exc:
            log.warning("[flat_file_loader] failed to read CSV bars %s: %s", path, exc)
            return []

        return bars

    def _load_bars_json(self, path: Path) -> list[dict]:
        """Load bars from JSON."""
        data = _read_json(path)

        if isinstance(data, dict):
            raw_bars = (
                data.get("bars")
                or data.get("results")
                or data.get("data")
                or []
            )
        elif isinstance(data, list):
            raw_bars = data
        else:
            raw_bars = []

        bars = [normalize_bar(b) for b in raw_bars if isinstance(b, dict)]
        return [b for b in bars if _valid_bar(b)]

    # ── Ticker loaders ────────────────────────────────────────────────────────

    def _load_tickers_txt(self, path: Path) -> list[str]:
        """Load tickers from a text file."""
        try:
            tickers = []
            for line in path.read_text(encoding="utf-8").splitlines():
                ticker = line.strip().upper()
                if ticker and not ticker.startswith("#"):
                    tickers.append(ticker)
            return _dedupe(tickers)
        except Exception as exc:
            log.warning("[flat_file_loader] failed to read TXT tickers %s: %s", path, exc)
            return []

    def _load_tickers_csv(self, path: Path) -> list[str]:
        """Load tickers from CSV."""
        tickers: list[str] = []

        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = [name.lower() for name in (reader.fieldnames or [])]

                preferred = None
                for key in ("ticker", "symbol", "ticker_symbol"):
                    if key in fieldnames:
                        preferred = reader.fieldnames[fieldnames.index(key)]
                        break

                for row in reader:
                    if preferred:
                        ticker = str(row.get(preferred, "")).upper().strip()
                    else:
                        first_key = reader.fieldnames[0] if reader.fieldnames else ""
                        ticker = str(row.get(first_key, "")).upper().strip()

                    if ticker:
                        tickers.append(ticker)

        except Exception as exc:
            log.warning("[flat_file_loader] failed to read CSV tickers %s: %s", path, exc)
            return []

        return _dedupe(tickers)

    def _load_tickers_json(self, path: Path) -> list[str]:
        """Load tickers from JSON."""
        data = _read_json(path)
        tickers: list[str] = []

        if isinstance(data, dict):
            raw = (
                data.get("tickers")
                or data.get("symbols")
                or data.get("universe")
                or data.get("watchlist")
                or []
            )
        elif isinstance(data, list):
            raw = data
        else:
            raw = []

        for item in raw:
            if isinstance(item, str):
                ticker = item.upper().strip()
            elif isinstance(item, dict):
                ticker = str(
                    item.get("ticker")
                    or item.get("symbol")
                    or item.get("ticker_symbol")
                    or ""
                ).upper().strip()
            else:
                ticker = ""

            if ticker:
                tickers.append(ticker)

        return _dedupe(tickers)

    # ── Path helper ───────────────────────────────────────────────────────────

    def _resolve(self, file_path: str | Path) -> Path:
        """
        Resolve a path. Relative paths are resolved from project root/current cwd.
        """
        path = Path(file_path)
        if path.is_absolute():
            return path
        return Path(path)


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_bar(raw: dict) -> dict:
    """
    Normalize one raw bar into bot OHLCV schema.

    Accepts common aliases:
      open/o, high/h, low/l, close/c, volume/v, timestamp/t/time/date
    """
    timestamp = (
        raw.get("timestamp")
        or raw.get("time")
        or raw.get("datetime")
        or raw.get("date")
        or raw.get("t")
        or ""
    )

    ts_ms = raw.get("t")
    if not ts_ms and timestamp:
        ts_ms = _timestamp_to_ms(timestamp)

    return {
        "t": ts_ms,
        "timestamp": _timestamp_to_iso(timestamp, ts_ms),
        "o": _safe_float(raw.get("o", raw.get("open", 0.0))),
        "h": _safe_float(raw.get("h", raw.get("high", 0.0))),
        "l": _safe_float(raw.get("l", raw.get("low", 0.0))),
        "c": _safe_float(raw.get("c", raw.get("close", 0.0))),
        "v": _safe_float(raw.get("v", raw.get("volume", 0.0))),
        "vw": _safe_float(raw.get("vw", raw.get("vwap", 0.0))),
        "source": raw.get("source", "flat_file"),
        "raw": dict(raw),
    }


def _valid_bar(bar: dict) -> bool:
    """True when a normalized bar has usable OHLC values."""
    return (
        bar.get("o", 0) > 0
        and bar.get("h", 0) > 0
        and bar.get("l", 0) > 0
        and bar.get("c", 0) > 0
        and bar.get("h", 0) >= bar.get("l", 0)
    )


# ── Convenience wrappers ──────────────────────────────────────────────────────

def load_bars(settings: dict, file_path: str | Path) -> list[dict]:
    """
    Convenience function for tests/backtests.
    """
    loader = FlatFileLoader(settings)
    return loader.load_bars(file_path)


def load_ticker_universe(settings: dict, file_path: str | Path) -> list[str]:
    """
    Convenience function for scanner universe loading.
    """
    loader = FlatFileLoader(settings)
    return loader.load_ticker_universe(file_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[flat_file_loader] failed to read JSON %s: %s", path, exc)
        return {} if path.suffix.lower() == ".json" else []


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp_to_ms(value: object) -> Optional[int]:
    """
    Convert ISO/date timestamp to milliseconds.
    """
    try:
        if value is None or value == "":
            return None

        if isinstance(value, (int, float)):
            return int(value)

        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _timestamp_to_iso(timestamp: object, ts_ms: object = None) -> str:
    """
    Normalize timestamp to ISO string when possible.
    """
    if timestamp:
        try:
            if isinstance(timestamp, str):
                # Return normalized ISO if parseable, otherwise original.
                text = timestamp.replace("Z", "+00:00")
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
        except Exception:
            return str(timestamp)

    try:
        if ts_ms:
            return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        pass

    return ""


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = item.upper().strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
