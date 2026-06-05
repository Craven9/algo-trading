"""
src/data/massive_websocket_stream.py — Massive/Polygon-style WebSocket stream
Handles live market data streaming for price, trades, quotes, and minute bars.

The bot overview separates data access into two lanes:
  - WebSocket API for live streaming updates
  - REST API for on-demand/background data

This stream is for the live lane.  It should be used for:
  - Live price updates
  - Live scanner updates
  - Live trade monitoring
  - Live exit monitoring
  - Live VWAP / price / volume / breakout confirmation

Responsibilities:
  - Connect to Massive/Polygon-style WebSocket endpoint
  - Authenticate with API key
  - Subscribe/unsubscribe to ticker channels
  - Normalize live messages into plain dict events
  - Call user-provided callbacks for each event
  - Reconnect safely after connection loss
  - Keep the stream separate from trading logic

Design rules:
  - This file does not scan by itself
  - This file does not score trades
  - This file does not place orders
  - This file does not approve trades
  - WebSocket is preferred for live updates
  - REST should be used for fallback/historical/on-demand data
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

try:
    import websocket
except Exception:
    websocket = None

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_WS_URL = "wss://socket.polygon.io/stocks"
_DEFAULT_RECONNECT_SECONDS = 5.0
_DEFAULT_MAX_RECONNECTS = 10


# ── Event dataclass ───────────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    """
    Normalized WebSocket event.
    """
    event_type:      str
    ticker:          str = ""
    price:           float = 0.0
    size:            float = 0.0
    bid:             float = 0.0
    ask:             float = 0.0
    volume:          float = 0.0
    timestamp_ms:    Optional[int] = None
    timestamp:       str = ""
    raw:             dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":   self.event_type,
            "ticker":       self.ticker,
            "price":        self.price,
            "size":         self.size,
            "bid":          self.bid,
            "ask":          self.ask,
            "volume":       self.volume,
            "timestamp_ms": self.timestamp_ms,
            "timestamp":    self.timestamp,
            "raw":          self.raw,
        }


# ── Stream client ─────────────────────────────────────────────────────────────

class MassiveWebSocketStream:
    """
    WebSocket stream client for Massive/Polygon-style live stock data.

    Usage:
        stream = MassiveWebSocketStream(settings)
        stream.subscribe_trades(["AAPL", "TSLA"])
        stream.run_forever(on_event=my_callback)
    """

    def __init__(self, settings: dict):
        self._settings = settings
        self._cfg = (
            settings.get("massive", {})
            or settings.get("polygon", {})
            or settings.get("market_data", {}).get("massive", {})
            or {}
        )

        self._ws_url = str(self._cfg.get("websocket_url", _DEFAULT_WS_URL))
        self._api_key = (
            self._cfg.get("api_key")
            or os.environ.get("MASSIVE_API_KEY")
            or os.environ.get("POLYGON_API_KEY")
            or ""
        )

        self._reconnect_seconds = float(
            self._cfg.get("websocket_reconnect_seconds", _DEFAULT_RECONNECT_SECONDS)
        )
        self._max_reconnects = int(
            self._cfg.get("websocket_max_reconnects", _DEFAULT_MAX_RECONNECTS)
        )

        self._ws = None
        self._connected = False
        self._stop_requested = False

        self._subscriptions: set[str] = set()
        self._on_event: Optional[Callable[[StreamEvent], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def available(self) -> bool:
        """
        True when websocket-client and API key are available.
        """
        return websocket is not None and bool(self._api_key)

    def subscribe_trades(self, tickers: list[str]) -> None:
        """
        Subscribe to live trade events.
        Channel format: T.SYMBOL
        """
        self._add_subscriptions([f"T.{t.upper()}" for t in tickers])

    def subscribe_quotes(self, tickers: list[str]) -> None:
        """
        Subscribe to live quote events.
        Channel format: Q.SYMBOL
        """
        self._add_subscriptions([f"Q.{t.upper()}" for t in tickers])

    def subscribe_minute_bars(self, tickers: list[str]) -> None:
        """
        Subscribe to live aggregate minute bars.
        Channel format: AM.SYMBOL
        """
        self._add_subscriptions([f"AM.{t.upper()}" for t in tickers])

    def subscribe_second_bars(self, tickers: list[str]) -> None:
        """
        Subscribe to live aggregate second bars.
        Channel format: A.SYMBOL
        """
        self._add_subscriptions([f"A.{t.upper()}" for t in tickers])

    def unsubscribe(self, channels: list[str]) -> None:
        """
        Unsubscribe from channels.
        """
        channels = [c.upper() for c in channels if c]
        for channel in channels:
            self._subscriptions.discard(channel)

        if self._connected:
            self._send({
                "action": "unsubscribe",
                "params": ",".join(channels),
            })

    def run_forever(
        self,
        on_event: Callable[[StreamEvent], None],
        on_error: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Connect and run stream forever until stop() is called.

        Args:
            on_event:  Callback for each normalized StreamEvent.
            on_error:  Optional error callback.
            on_status: Optional status callback.
        """
        self._on_event = on_event
        self._on_error = on_error
        self._on_status = on_status
        self._stop_requested = False

        if not self.available():
            self._emit_error("WebSocket stream unavailable: missing package or API key")
            return

        reconnect_count = 0

        while not self._stop_requested:
            try:
                self._emit_status("connecting")
                self._ws = websocket.WebSocketApp(
                    self._ws_url,
                    on_open=self._handle_open,
                    on_message=self._handle_message,
                    on_error=self._handle_error,
                    on_close=self._handle_close,
                )
                self._ws.run_forever()

            except Exception as exc:
                self._emit_error(f"stream crashed: {exc}")

            if self._stop_requested:
                break

            reconnect_count += 1
            if reconnect_count > self._max_reconnects:
                self._emit_error("max reconnect attempts reached")
                break

            self._emit_status(
                f"reconnecting in {self._reconnect_seconds:.1f}s "
                f"({reconnect_count}/{self._max_reconnects})"
            )
            time.sleep(self._reconnect_seconds)

    def stop(self) -> None:
        """
        Request stream stop.
        """
        self._stop_requested = True
        self._connected = False

        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def is_connected(self) -> bool:
        """Return current connection flag."""
        return self._connected

    def subscriptions(self) -> list[str]:
        """Return sorted active subscription channels."""
        return sorted(self._subscriptions)

    # ── Internal subscription helpers ─────────────────────────────────────────

    def _add_subscriptions(self, channels: list[str]) -> None:
        """
        Add subscriptions and send to socket if already connected.
        """
        channels = [c.upper() for c in channels if c]
        for channel in channels:
            self._subscriptions.add(channel)

        if self._connected:
            self._send({
                "action": "subscribe",
                "params": ",".join(channels),
            })

    def _subscribe_current_channels(self) -> None:
        """
        Send all current subscriptions to WebSocket.
        """
        if not self._subscriptions:
            return

        self._send({
            "action": "subscribe",
            "params": ",".join(sorted(self._subscriptions)),
        })

    # ── WebSocket callbacks ───────────────────────────────────────────────────

    def _handle_open(self, ws) -> None:
        """
        Called when socket opens.
        """
        self._connected = True
        self._emit_status("connected")
        self._send({
            "action": "auth",
            "params": self._api_key,
        })
        self._subscribe_current_channels()

    def _handle_message(self, ws, message: str) -> None:
        """
        Called for every raw socket message.
        """
        try:
            payload = json.loads(message)
        except Exception:
            self._emit_error(f"invalid websocket message: {message[:200]}")
            return

        if isinstance(payload, dict):
            payload = [payload]

        if not isinstance(payload, list):
            return

        for raw in payload:
            event = normalize_stream_message(raw)
            if event and self._on_event:
                self._on_event(event)

    def _handle_error(self, ws, error) -> None:
        """
        Called when socket has an error.
        """
        self._connected = False
        self._emit_error(str(error))

    def _handle_close(self, ws, close_status_code, close_msg) -> None:
        """
        Called when socket closes.
        """
        self._connected = False
        self._emit_status(f"closed: {close_status_code} {close_msg}")

    # ── Send / emit helpers ───────────────────────────────────────────────────

    def _send(self, payload: dict) -> None:
        """
        Send a JSON payload to the WebSocket.
        """
        if not self._ws:
            return

        try:
            self._ws.send(json.dumps(payload))
        except Exception as exc:
            self._emit_error(f"send failed: {exc}")

    def _emit_error(self, message: str) -> None:
        log.warning("[massive_ws] %s", message)
        if self._on_error:
            self._on_error(message)

    def _emit_status(self, message: str) -> None:
        log.info("[massive_ws] %s", message)
        if self._on_status:
            self._on_status(message)


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_stream_message(raw: dict) -> Optional[StreamEvent]:
    """
    Normalize Massive/Polygon WebSocket message into StreamEvent.

    Common event codes:
      T  = trade
      Q  = quote
      A  = second aggregate
      AM = minute aggregate
      status = auth/subscription status
    """
    if not raw:
        return None

    ev = str(raw.get("ev", raw.get("event", "")))

    if ev in ("status", "subscription_status"):
        return StreamEvent(
            event_type="status",
            raw=dict(raw),
            timestamp=_now_iso(),
        )

    if ev == "T":
        ts = raw.get("t")
        return StreamEvent(
            event_type   = "trade",
            ticker       = str(raw.get("sym", raw.get("T", ""))).upper(),
            price        = _safe_float(raw.get("p")),
            size         = _safe_float(raw.get("s")),
            timestamp_ms = _safe_int(ts),
            timestamp    = _timestamp_to_iso(ts),
            raw          = dict(raw),
        )

    if ev == "Q":
        ts = raw.get("t")
        return StreamEvent(
            event_type   = "quote",
            ticker       = str(raw.get("sym", raw.get("T", ""))).upper(),
            bid          = _safe_float(raw.get("bp")),
            ask          = _safe_float(raw.get("ap")),
            size         = _safe_float(raw.get("bs")),
            timestamp_ms = _safe_int(ts),
            timestamp    = _timestamp_to_iso(ts),
            raw          = dict(raw),
        )

    if ev in ("A", "AM"):
        ts = raw.get("s") or raw.get("t")
        close_price = _safe_float(raw.get("c"))
        return StreamEvent(
            event_type   = "minute_bar" if ev == "AM" else "second_bar",
            ticker       = str(raw.get("sym", raw.get("T", ""))).upper(),
            price        = close_price,
            volume       = _safe_float(raw.get("v")),
            timestamp_ms = _safe_int(ts),
            timestamp    = _timestamp_to_iso(ts),
            raw          = normalize_bar_event(raw),
        )

    return StreamEvent(
        event_type=ev or "unknown",
        raw=dict(raw),
        timestamp=_now_iso(),
    )


def normalize_bar_event(raw: dict) -> dict:
    """
    Normalize aggregate bar event to bot OHLCV style.
    """
    ts = raw.get("s") or raw.get("t")
    return {
        "t": ts,
        "timestamp": _timestamp_to_iso(ts),
        "o": _safe_float(raw.get("o")),
        "h": _safe_float(raw.get("h")),
        "l": _safe_float(raw.get("l")),
        "c": _safe_float(raw.get("c")),
        "v": _safe_float(raw.get("v")),
        "vw": _safe_float(raw.get("vw")),
        "source": "massive_websocket",
        "raw": dict(raw),
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def create_stream(settings: dict) -> MassiveWebSocketStream:
    """
    Convenience function for bot_runner.py.
    """
    return MassiveWebSocketStream(settings)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _timestamp_to_iso(ts: object) -> str:
    try:
        if ts is None:
            return ""
        return datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
