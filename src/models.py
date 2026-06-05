"""
src/core/models.py — Canonical data models for the trading bot
All data that flows between modules uses these dataclasses.
No module should invent its own dict schema — import from here instead.

Design rules:
  - Dataclasses only (no ORM, no Pydantic dependency)
  - Every field has a type annotation and a default where sensible
  - to_dict() / from_dict() on every class for JSON serialization
  - Enums for all fixed-value string fields so typos are caught at import time
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class CandidateSource(str, Enum):
    SCANNER  = "scanner"
    MANUAL   = "manual"
    WATCHLIST= "watchlist"


class SetupName(str, Enum):
    BREAK_AND_HOLD          = "break_and_hold"
    BOTTOM_BASE             = "bottom_base"
    VWAP_RECLAIM            = "vwap_reclaim"
    OPENING_RANGE_BREAKOUT  = "opening_range_breakout"
    LIQUIDITY_SWEEP_RECLAIM = "liquidity_sweep_reclaim"
    FIBONACCI_PULLBACK      = "fibonacci_pullback"
    NONE                    = "none"


class TradeDecision(str, Enum):
    APPROVED_FOR_PAPER_BUY = "APPROVED_FOR_PAPER_BUY"
    REJECTED               = "REJECTED"
    WATCH                  = "WATCH"
    WAIT_FOR_PULLBACK      = "WAIT_FOR_PULLBACK"
    WAIT_FOR_RECLAIM       = "WAIT_FOR_RECLAIM"
    MANAGE_EXISTING_ONLY   = "MANAGE_EXISTING_ONLY"


class TradeStatus(str, Enum):
    OPEN      = "open"
    PARTIAL   = "partial"
    CLOSED    = "closed"
    REJECTED  = "rejected"
    CANCELLED = "cancelled"


class TradeResult(str, Enum):
    WIN       = "win"
    LOSS      = "loss"
    BREAKEVEN = "breakeven"
    OPEN      = "open"


class ExitReason(str, Enum):
    STOP_LOSS_HIT        = "stop_loss_hit"
    MAX_LOSS_HIT         = "max_loss_hit"
    BREAK_EVEN_STOP      = "break_even_stop"
    PARTIAL_PROFIT       = "partial_profit"
    TRAILING_STOP        = "trailing_stop"
    FAILED_BREAKOUT      = "failed_breakout"
    FAILED_VWAP_HOLD     = "failed_vwap_hold"
    VOLUME_FADE          = "volume_fade"
    FIB_TARGET_HIT       = "fib_target_hit"
    RUNNER_TARGET        = "runner_target"
    MANUAL               = "manual"
    RISK_OFF_MARKET      = "risk_off_market"
    END_OF_DAY           = "end_of_day"


class ExitDecision(str, Enum):
    HOLD                    = "HOLD"
    MOVE_STOP_TO_BREAK_EVEN = "MOVE_STOP_TO_BREAK_EVEN"
    TAKE_PARTIAL            = "TAKE_PARTIAL"
    TRAIL_STOP              = "TRAIL_STOP"
    EXIT_FULL               = "EXIT_FULL"
    EXIT_FAILED_BREAKOUT    = "EXIT_FAILED_BREAKOUT"
    EXIT_MAX_LOSS           = "EXIT_MAX_LOSS"


class ConfidenceLabel(str, Enum):
    ELITE    = "elite"       # 90-100
    STRONG   = "strong"      # 80-89
    DECENT   = "decent"      # 70-79
    WEAK     = "weak"        # 60-69
    REJECT   = "reject"      # below 60


class VolumeDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    FLAT       = "flat"


class MaTrend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    FLAT    = "flat"


class PriceVsVwap(str, Enum):
    ABOVE = "above"
    BELOW = "below"


class RsiZone(str, Enum):
    OVERSOLD   = "oversold"
    NEUTRAL    = "neutral"
    OVERBOUGHT = "overbought"
    UNKNOWN    = "unknown"


class OrderType(str, Enum):
    MARKET           = "market"
    LIMIT            = "limit"
    MARKETABLE_LIMIT = "marketable_limit"


class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


# ── Helper ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Scanner / Candidate ───────────────────────────────────────────────────────

@dataclass
class ScannerCandidate:
    """
    Output of the scanner system.
    Represents a ticker worth analyzing — NOT a buy signal.
    """
    ticker:            str
    source:            CandidateSource         = CandidateSource.SCANNER
    price:             float                   = 0.0
    bid:               float                   = 0.0
    ask:               float                   = 0.0
    spread_percent:    float                   = 0.0
    relative_volume:   float                   = 0.0
    day_change_pct:    float                   = 0.0
    dollar_volume:     float                   = 0.0
    candidate_reason:  str                     = ""
    scanned_at:        str                     = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ScannerCandidate:
        d = d.copy()
        d["source"] = CandidateSource(d.get("source", "scanner"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Indicators ────────────────────────────────────────────────────────────────

@dataclass
class MacdResult:
    macd:                float = 0.0
    signal:              float = 0.0
    histogram:           float = 0.0
    histogram_direction: str   = "flat"
    bullish_crossover:   bool  = False
    bearish_crossover:   bool  = False
    bullish:             bool  = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MacdResult:
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class IndicatorSnapshot:
    """
    Standardized indicator output from indicator_calculator.compute_all().
    Every setup detector and scoring engine reads from this model.
    """
    # VWAP
    vwap:               Optional[float]  = None
    price_vs_vwap:      str              = PriceVsVwap.BELOW.value
    vwap_distance_pct:  float            = 0.0
    vwap_extended:      bool             = False

    # RSI
    rsi:                Optional[float]  = None
    rsi_zone:           str              = RsiZone.UNKNOWN.value

    # MACD
    macd:               Optional[MacdResult] = None

    # ATR
    atr:                Optional[float]  = None

    # Moving averages
    ma_fast:            Optional[float]  = None
    ma_slow:            Optional[float]  = None
    ma_trend:           str              = MaTrend.FLAT.value
    ma_spread_pct:      float            = 0.0

    # Volume
    relative_volume:    float            = 0.0
    volume_trend:       str              = VolumeDirection.FLAT.value

    # Candle strength
    candle_strength:     float           = 0.0
    avg_candle_strength: float           = 0.0

    # Market structure
    higher_lows:        bool             = False
    lower_highs:        bool             = False
    swing_highs:        list[float]      = field(default_factory=list)
    swing_lows:         list[float]      = field(default_factory=list)

    # Raw bar data
    latest_close:       float            = 0.0
    latest_bar:         dict             = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.macd is not None:
            d["macd"] = self.macd.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IndicatorSnapshot:
        d = d.copy()
        macd_raw = d.pop("macd", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if isinstance(macd_raw, dict):
            obj.macd = MacdResult.from_dict(macd_raw)
        return obj


# ── Setup Detection ───────────────────────────────────────────────────────────

@dataclass
class SetupResult:
    """
    Output of a single setup detector (e.g. vwap_reclaim.py).
    Setup detectors ONLY detect — they never place trades.
    """
    setup_name:     str              = SetupName.NONE.value
    confirmed:      bool             = False
    confidence:     str              = ConfidenceLabel.REJECT.value
    score:          float            = 0.0
    entry_trigger:  Optional[float]  = None
    stop_area:      Optional[float]  = None
    target_area:    Optional[float]  = None
    reasons:        list[str]        = field(default_factory=list)
    warnings:       list[str]        = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SetupResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Fibonacci ─────────────────────────────────────────────────────────────────

@dataclass
class FibonacciResult:
    """Output of fibonacci_strategy_engine.py"""
    fib_trend_valid:          bool             = False
    swing_high:               Optional[float]  = None
    swing_low:                Optional[float]  = None
    nearest_retracement:      Optional[float]  = None
    distance_from_fib_pct:    float            = 0.0
    entry_confirmed_by_fib:   bool             = False
    block_trade:              bool             = False
    retracement_levels:       dict             = field(default_factory=dict)
    target_extensions:        dict             = field(default_factory=dict)
    reasons:                  list[str]        = field(default_factory=list)
    warnings:                 list[str]        = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FibonacciResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Scores ────────────────────────────────────────────────────────────────────

@dataclass
class TradeScores:
    """
    All numeric scores produced by the scoring engines.
    trade_quality_gate.py uses this as its sole input.
    """
    setup_score:              float = 0.0
    probability_score:        float = 0.0
    risk_reward_score:        float = 0.0
    move_potential_score:     float = 0.0
    execution_quality_score:  float = 0.0
    historical_edge_score:    float = 0.0
    final_trade_quality_score:float = 0.0
    confidence_label:         str   = ConfidenceLabel.REJECT.value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TradeScores:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Risk / Sizing ─────────────────────────────────────────────────────────────

@dataclass
class PositionSize:
    """Output of position_sizer.py"""
    shares:              int    = 0
    entry_price:         float  = 0.0
    stop_price:          float  = 0.0
    risk_per_share:      float  = 0.0
    max_risk_dollars:    float  = 0.0
    position_value:      float  = 0.0
    size_reduction_pct:  float  = 0.0   # applied when confidence or streak reduces size
    reasons:             list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PositionSize:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Trade Decision ────────────────────────────────────────────────────────────

@dataclass
class TradeDecisionResult:
    """
    Final output of trade_quality_gate.py.
    This is the ONLY object that can authorize a paper buy.
    """
    ticker:                   str
    decision:                 str              = TradeDecision.REJECTED.value
    setup:                    str              = SetupName.NONE.value
    scores:                   TradeScores      = field(default_factory=TradeScores)
    entry_price:              Optional[float]  = None
    stop_loss:                Optional[float]  = None
    target_1:                 Optional[float]  = None
    target_2:                 Optional[float]  = None
    runner_target:            Optional[float]  = None
    position_size:            Optional[PositionSize] = None
    reasons:                  list[str]        = field(default_factory=list)
    warnings:                 list[str]        = field(default_factory=list)
    what_would_make_valid:    list[str]        = field(default_factory=list)
    decided_at:               str              = field(default_factory=_now_iso)

    @property
    def approved(self) -> bool:
        return self.decision == TradeDecision.APPROVED_FOR_PAPER_BUY.value

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.scores:
            d["scores"] = self.scores.to_dict()
        if self.position_size:
            d["position_size"] = self.position_size.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TradeDecisionResult:
        d = d.copy()
        scores_raw = d.pop("scores", None)
        size_raw   = d.pop("position_size", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if isinstance(scores_raw, dict):
            obj.scores = TradeScores.from_dict(scores_raw)
        if isinstance(size_raw, dict):
            obj.position_size = PositionSize.from_dict(size_raw)
        return obj


# ── Trade (open / closed) ─────────────────────────────────────────────────────

@dataclass
class TradeEntry:
    price:         float           = 0.0
    shares:        int             = 0
    order_type:    str             = OrderType.MARKETABLE_LIMIT.value
    order_id:      str             = ""
    timestamp:     str             = field(default_factory=_now_iso)
    setup:         str             = SetupName.NONE.value
    stop_loss:     Optional[float] = None
    target_1:      Optional[float] = None
    target_2:      Optional[float] = None
    runner_target: Optional[float] = None
    reasons:       list[str]       = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeExit:
    price:         float           = 0.0
    shares:        int             = 0
    reason:        str             = ExitReason.MANUAL.value
    timestamp:     str             = field(default_factory=_now_iso)
    pnl_dollars:   float           = 0.0
    pnl_percent:   float           = 0.0
    order_id:      str             = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeManagement:
    """Live management state — updated every loop while a trade is open."""
    current_price:          float           = 0.0
    unrealized_pnl_dollars: float           = 0.0
    unrealized_pnl_pct:     float           = 0.0
    current_stop:           Optional[float] = None
    break_even_triggered:   bool            = False
    partials_taken:         int             = 0
    shares_remaining:       int             = 0
    max_favorable_move_pct: float           = 0.0
    max_adverse_move_pct:   float           = 0.0
    exit_warnings:          list[str]       = field(default_factory=list)
    next_action:            str             = ExitDecision.HOLD.value
    last_updated:           str             = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeReview:
    """Post-close review fields populated by the learning system."""
    backtest_reviewed:    bool       = False
    mistakes_found:       list[str]  = field(default_factory=list)
    risk_reward_achieved: float      = 0.0
    planned_vs_actual:    str        = ""
    notes:                str        = ""
    reviewed_at:          str        = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trade:
    """
    Master trade record.  Written to trades/open/, trades/closed/, or
    trades/rejected/ as JSON.  All subsystems read and update this object.

    Schema mirrors the design doc's recommended trade JSON structure:
        trade_id, ticker, status, entry, exit, setup, scores,
        risk, management, reasons, broker, review
    """
    trade_id:   str              = field(default_factory=_new_id)
    ticker:     str              = ""
    status:     str              = TradeStatus.OPEN.value
    result:     str              = TradeResult.OPEN.value
    paper:      bool             = True
    dry_run:    bool             = True

    entry:      TradeEntry       = field(default_factory=TradeEntry)
    exits:      list[TradeExit]  = field(default_factory=list)
    management: TradeManagement  = field(default_factory=TradeManagement)
    scores:     TradeScores      = field(default_factory=TradeScores)
    review:     TradeReview      = field(default_factory=TradeReview)

    # Broker-level fields
    alpaca_order_id:  str  = ""
    alpaca_account:   str  = ""

    created_at:  str = field(default_factory=_now_iso)
    closed_at:   str = ""

    def total_pnl(self) -> float:
        return sum(e.pnl_dollars for e in self.exits)

    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN.value

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry"]      = self.entry.to_dict()
        d["exits"]      = [e.to_dict() for e in self.exits]
        d["management"] = self.management.to_dict()
        d["scores"]     = self.scores.to_dict()
        d["review"]     = self.review.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Trade:
        d = d.copy()

        entry_raw  = d.pop("entry",      {})
        exits_raw  = d.pop("exits",      [])
        mgmt_raw   = d.pop("management", {})
        scores_raw = d.pop("scores",     {})
        review_raw = d.pop("review",     {})

        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.entry      = TradeEntry(**{k: v for k, v in entry_raw.items()
                                       if k in TradeEntry.__dataclass_fields__})
        obj.exits      = [TradeExit(**{k: v for k, v in e.items()
                                       if k in TradeExit.__dataclass_fields__})
                          for e in exits_raw]
        obj.management = TradeManagement(**{k: v for k, v in mgmt_raw.items()
                                            if k in TradeManagement.__dataclass_fields__})
        obj.scores     = TradeScores.from_dict(scores_raw)
        obj.review     = TradeReview(**{k: v for k, v in review_raw.items()
                                        if k in TradeReview.__dataclass_fields__})
        return obj


# ── Opening Range ─────────────────────────────────────────────────────────────

@dataclass
class OpeningRange:
    """Output of opening_range_analyzer.py"""
    range_minutes:     int            = 15
    high:              Optional[float]= None
    low:               Optional[float]= None
    range_size:        float          = 0.0
    range_size_pct:    float          = 0.0
    inside_range:      bool           = False
    breakout_above:    bool           = False
    breakdown_below:   bool           = False
    failed_breakout:   bool           = False
    calculated_at:     str            = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OpeningRange:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Exit Score ────────────────────────────────────────────────────────────────

@dataclass
class ExitScoreResult:
    """Output of exit_score_engine.py — evaluated continuously while trade is open."""
    ticker:       str            = ""
    exit_score:   float          = 0.0
    decision:     str            = ExitDecision.HOLD.value
    stop_update:  Optional[float]= None
    next_target:  Optional[float]= None
    warnings:     list[str]      = field(default_factory=list)
    reasons:      list[str]      = field(default_factory=list)
    evaluated_at: str            = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExitScoreResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Performance ───────────────────────────────────────────────────────────────

@dataclass
class SetupPerformanceRecord:
    """
    Tracks historical win/loss statistics for a single setup type.
    Used by the learning system to apply score bonuses/penalties.
    """
    setup_name:        str   = SetupName.NONE.value
    total_trades:      int   = 0
    wins:              int   = 0
    losses:            int   = 0
    breakevens:        int   = 0
    total_pnl:         float = 0.0
    avg_win:           float = 0.0
    avg_loss:          float = 0.0
    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    score_adjustment:  float = 0.0   # bonus (+) or penalty (-) applied to scoring
    last_updated:      str   = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SetupPerformanceRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
