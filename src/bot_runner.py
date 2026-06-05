"""
src/bot_runner.py — Main AI Trading Assistant orchestrator
Runs the paper-trading bot loop by calling each module in the correct order.

bot_runner.py should coordinate the system, not contain the trading strategy.
The trading logic belongs in the scanner, analysis, setup, scoring, risk,
execution, exit, logging, and learning modules.

Main flow:
  1. Load settings
  2. Validate safety / control state
  3. Update open trade statuses
  4. Manage exits for open trades
  5. Scan for candidates
  6. Filter and rank candidates
  7. Analyze each candidate
  8. Run setup detectors
  9. Score setup, probability, risk/reward, move potential, historical edge
 10. Size position
 11. Run account risk guard
 12. Run execution quality guard
 13. Run trade quality gate
 14. Execute approved paper buy
 15. Log trade
 16. Refresh dashboard state
 17. Repeat

Design rules:
  - This file does not contain setup logic
  - This file does not contain scoring logic
  - This file does not place orders unless trade_quality_gate approved
  - This file must respect dry-run and paper-only safety settings
  - Watchlist candidates are opportunities only, never automatic buys
  - Every buy must pass through trade_quality_gate.py
"""

from __future__ import annotations

import logging
import json
import os
from dotenv import load_dotenv
import time
from pathlib import Path
from typing import Optional

from config_loader import load_settings
from logger import setup_logging

from data.market_data_provider import MarketDataProvider
from data.massive_rest_client import MassiveRestClient
from data.alpaca_data_client import AlpacaDataClient
from data.news_catalyst_provider import NewsCatalystProvider

from scanner.momentum_scanner import MomentumScanner
from scanner.manual_watchlist_loader import ManualWatchlistLoader
from scanner.scanner_filter_engine import ScannerFilterEngine
from scanner.candidate_ranker import CandidateRanker

from analysis.key_level_engine import KeyLevelEngine
from analysis.opening_range_analyzer import OpeningRangeAnalyzer
from analysis.liquidity_sweep_detector import LiquiditySweepDetector
from analysis.session_structure_analyzer import SessionStructureAnalyzer
from analysis.fibonacci_strategy_engine import FibonacciStrategyEngine
from analysis.move_potential_engine import MovePotentialEngine
from analysis.failed_reclaim_detector import FailedReclaimDetector
from analysis.market_influence_filter import MarketInfluenceFilter

from setups import run_all_setups, best_setup

from scoring.risk_reward_engine import RiskRewardEngine
from scoring.setup_score_engine import SetupScoreEngine
from scoring.probability_engine import ProbabilityEngine
from scoring.execution_quality_guard import ExecutionQualityGuard
from scoring.trade_quality_gate import TradeQualityGate

from risk.position_sizer import PositionSizer
from risk.position_tracker import PositionTracker, positions_to_alpaca_like_dicts
from risk.account_risk_guard import AccountRiskGuard

from execution.order_executor import OrderExecutor
from execution.exit_manager import ExitManager
from execution.trade_logger import TradeLogger
from execution.trade_status_updater import TradeStatusUpdater

from learning.performance_tracker import PerformanceTracker
from learning.backtest_reviewer import BacktestReviewer
from learning.historical_edge_engine import HistoricalEdgeEngine
from learning.learning_engine import LearningEngine

from frontend.dashboard_state_writer import DashboardStateWriter
from frontend.bot_control_api import BotControlState

load_dotenv()

log = logging.getLogger(__name__)


# ── Runner ────────────────────────────────────────────────────────────────────


class NullAlpacaDataClient:
    """
    Safe fallback client used when API keys are missing in dry-run mode.
    It lets BotRunner initialize, but returns empty market/account data.
    """

    def get_bars(self, *args, **kwargs):
        return []

    def get_bars_multi(self, *args, **kwargs):
        return {}

    def get_latest_quote(self, *args, **kwargs):
        return {}

    def get_latest_trade(self, *args, **kwargs):
        return {}

    def get_snapshot(self, *args, **kwargs):
        return {}

    def get_snapshots(self, *args, **kwargs):
        return {}

    def get_account(self, *args, **kwargs):
        return {}

    def get_positions(self, *args, **kwargs):
        return []

    def get_position(self, *args, **kwargs):
        return None



class NullNewsCatalystProvider:
    """
    Safe fallback news provider used when news API keys are missing.
    """

    def get_news(self, *args, **kwargs):
        return []

    def get_ticker_news(self, *args, **kwargs):
        return []

    def get_catalysts(self, *args, **kwargs):
        return []

    def analyze(self, *args, **kwargs):
        return None



class HybridMassiveAlpacaMarketDataClient:
    """
    Hybrid market data client.

    Massive/Polygon:
      - aggregate bars / candles

    Alpaca:
      - latest quote
      - latest trade
      - snapshot fallback
      - account / positions / paper trading support stays on Alpaca
    """

    def __init__(self, massive_client, alpaca_client, settings: dict):
        self.massive_client = massive_client
        self.alpaca_client = alpaca_client
        self.settings = settings

    def get_bars(self, ticker: str, timeframe: str = "1Min", limit: int = 200, *args, **kwargs):
        today = date.today().isoformat()

        try:
            bars = self.massive_client.get_aggregate_bars(ticker, today, today)
            if bars:
                return bars[-limit:]
        except Exception as exc:
            log.warning("[hybrid_data] Massive bars failed for %s, using Alpaca fallback: %s", ticker, exc)

        try:
            bars = self.alpaca_client.get_bars(ticker, timeframe=timeframe, limit=limit, *args, **kwargs)
            return bars or []
        except TypeError:
            bars = self.alpaca_client.get_bars(ticker, timeframe, limit)
            return bars or []

    def get_latest_quote(self, ticker: str, *args, **kwargs):
        return self.alpaca_client.get_latest_quote(ticker, *args, **kwargs)

    def get_latest_trade(self, ticker: str, *args, **kwargs):
        if hasattr(self.alpaca_client, "get_latest_trade"):
            return self.alpaca_client.get_latest_trade(ticker, *args, **kwargs)
        return {}

    def get_snapshot(self, ticker: str, *args, **kwargs):
        return self.alpaca_client.get_snapshot(ticker, *args, **kwargs)

    def get_snapshots(self, tickers: list[str], *args, **kwargs):
        if hasattr(self.alpaca_client, "get_snapshots"):
            return self.alpaca_client.get_snapshots(tickers, *args, **kwargs)

        if hasattr(self.alpaca_client, "get_snapshots_multi"):
            return self.alpaca_client.get_snapshots_multi(tickers, *args, **kwargs)

        return {ticker: self.get_snapshot(ticker) for ticker in tickers}

    def get_snapshots_multi(self, tickers: list[str], *args, **kwargs):
        """
        MomentumScanner expects get_snapshots_multi().
        Use Alpaca for latest quote/snapshot data because Massive snapshot
        endpoint is blocked by the current Massive plan.
        """
        if hasattr(self.alpaca_client, "get_snapshots_multi"):
            return self.alpaca_client.get_snapshots_multi(tickers, *args, **kwargs)

        if hasattr(self.alpaca_client, "get_snapshots"):
            return self.alpaca_client.get_snapshots(tickers, *args, **kwargs)

        return {ticker: self.get_snapshot(ticker) for ticker in tickers}

    def get_previous_close(self, ticker: str, *args, **kwargs):
        try:
            previous = self.massive_client.get_previous_close(ticker)
            if previous:
                return previous
        except Exception:
            pass

        if hasattr(self.alpaca_client, "get_previous_close"):
            return self.alpaca_client.get_previous_close(ticker, *args, **kwargs)

        return {}


class BotRunner:
    """
    Main bot orchestrator.

    Usage:
        runner = BotRunner()
        runner.run_once()

        # or live loop:
        runner.run_loop()
    """

    def __init__(self, settings: Optional[dict] = None):
        self.settings = settings or load_settings()
        setup_logging(self.settings)

        # ── Core helpers ──────────────────────────────────────────────────────
        self.control = BotControlState(self.settings)
        self.dashboard = DashboardStateWriter(self.settings)

        # ── Data providers ────────────────────────────────────────────────────
        self.alpaca_client = self._create_alpaca_data_client()
        self.massive_client = MassiveRestClient(self.settings)
        self.market_data_client = HybridMassiveAlpacaMarketDataClient(
            self.massive_client,
            self.alpaca_client,
            self.settings,
        )
        self.market_data = MarketDataProvider(self.market_data_client, self.settings)
        self.news_provider = self._create_news_provider()

        # ── Scanner system ────────────────────────────────────────────────────
        self.momentum_scanner = MomentumScanner(self.market_data_client, self.settings)
        self.watchlist_loader = ManualWatchlistLoader(self.settings)
        self.scanner_filter = ScannerFilterEngine(self.settings)
        self.candidate_ranker = CandidateRanker(self.settings)

        # ── Analysis system ───────────────────────────────────────────────────
        self.key_levels = KeyLevelEngine(self.settings)
        self.opening_range = OpeningRangeAnalyzer(self.settings)
        self.liquidity_sweep = LiquiditySweepDetector(self.settings)
        self.session_structure = SessionStructureAnalyzer(self.settings)
        self.fibonacci = FibonacciStrategyEngine(self.settings)
        self.move_potential = MovePotentialEngine(self.settings)
        self.failed_reclaim = FailedReclaimDetector(self.settings)
        self.market_influence = MarketInfluenceFilter(self.settings)

        # ── Scoring system ────────────────────────────────────────────────────
        self.risk_reward = RiskRewardEngine(self.settings)
        self.setup_score = SetupScoreEngine(self.settings)
        self.probability = ProbabilityEngine(self.settings)
        self.execution_quality = ExecutionQualityGuard(self.settings)
        self.trade_gate = TradeQualityGate(self.settings)

        # ── Risk / execution / logging ────────────────────────────────────────
        self.position_sizer = PositionSizer(self.settings)
        self.account_risk = AccountRiskGuard(self.settings)
        self.order_executor = OrderExecutor(self.settings)
        self.exit_manager = ExitManager(self.settings)
        self.trade_logger = TradeLogger(self.settings)

        # ── Broker / account helpers ──────────────────────────────────────────
        # self.alpaca_client is created above as the market data client
        self.position_tracker = PositionTracker(self.alpaca_client) if self.alpaca_client else None
        self.trade_status = TradeStatusUpdater(self.settings, self.alpaca_client)

        # ── Learning system ───────────────────────────────────────────────────
        self.performance = PerformanceTracker(self.settings)
        self.backtest_reviewer = BacktestReviewer(self.settings)
        self.historical_edge = HistoricalEdgeEngine(self.settings)
        self.learning = LearningEngine(self.settings)

    # ── Public run methods ────────────────────────────────────────────────────

    def run_once(self) -> dict:
        """
        Run one full bot cycle.

        Returns:
            Dict summary of the cycle.
        """
        cycle = {
            "status": "started",
            "scanner_candidates": [],
            "ranked_candidates": [],
            "decisions": [],
            "orders": [],
            "errors": [],
            "warnings": [],
        }

        allowed, reason = self._runtime_allowed()
        if not allowed:
            cycle["status"] = "blocked"
            cycle["warnings"].append(reason)
            self._write_dashboard(cycle)
            return cycle

        try:
            # 1. Sync existing trades first.
            self._sync_trade_statuses(cycle)

            # 2. Manage exits before looking for new entries.
            self._manage_open_trades(cycle)

            # 3. Scan/filter/rank candidates.
            candidates = self._get_candidates(cycle)
            cycle["scanner_candidates"] = candidates

            filtered = self._filter_candidates(candidates, cycle)
            ranked = self._rank_candidates(filtered, cycle)
            cycle["ranked_candidates"] = ranked

            # 4. Evaluate entries.
            for ranked_item in ranked:
                decision, order = self._evaluate_candidate(ranked_item, cycle)
                if decision:
                    cycle["decisions"].append(decision)
                if order:
                    cycle["orders"].append(order)

            # 5. Learning/dashboard refresh.
            self._review_and_learn(cycle)
            self._write_dashboard(cycle)

            cycle["status"] = "complete"
            return cycle

        except Exception as exc:
            log.exception("[bot_runner] run_once failed")
            cycle["status"] = "error"
            cycle["errors"].append(str(exc))
            self._write_dashboard(cycle)
            return cycle

    def run_loop(self) -> None:
        """
        Run the bot loop until stopped by control state or config.
        """
        loop_cfg = self.settings.get("live_loop", {})
        interval_seconds = float(loop_cfg.get("interval_seconds", 30))

        log.info("[bot_runner] Starting bot loop interval=%.1fs", interval_seconds)

        while True:
            allowed, reason = self._runtime_allowed()
            if not allowed:
                log.info("[bot_runner] Loop stopped/paused: %s", reason)
                self.dashboard.write_state(
                    bot_status="paused",
                    warnings=[reason],
                )
                break

            self.run_once()
            time.sleep(interval_seconds)

    # ── Candidate flow ────────────────────────────────────────────────────────

    def _get_candidates(self, cycle: dict) -> list:
        """
        Get scanner + manual watchlist candidates.

        This is intentionally flexible because older project files used
        different method names/signatures.
        """
        candidates = []
        universe = self._load_scan_universe()

        # Momentum scanner
        try:
            if hasattr(self.momentum_scanner, "scan"):
                try:
                    scanner_candidates = self.momentum_scanner.scan(universe)
                except TypeError:
                    scanner_candidates = self.momentum_scanner.scan()
                candidates.extend(scanner_candidates or [])
            else:
                cycle["warnings"].append("momentum scanner has no scan method")
        except Exception as exc:
            cycle["errors"].append(f"momentum scanner failed: {exc}")
            log.exception("[bot_runner] momentum scanner failed")

        # Manual watchlist loader
        try:
            watchlist_candidates = []

            for method_name in [
                "load",
                "load_watchlist",
                "load_candidates",
                "get_watchlist",
                "get_candidates",
                "get_tickers",
                "get_entries",
                "symbols",
                "tickers",
            ]:
                method = getattr(self.watchlist_loader, method_name, None)
                if callable(method):
                    watchlist_candidates = method()
                    break

            candidates.extend(watchlist_candidates or [])

        except Exception as exc:
            cycle["errors"].append(f"manual watchlist loader failed: {exc}")
            log.exception("[bot_runner] manual watchlist loader failed")

        return self._normalize_candidates(candidates)

    def _normalize_candidates(self, candidates: list) -> list:
        """
        Remove duplicate plain-string watchlist tickers when scanner already
        returned proper ScannerCandidate objects.
        """
        if not candidates:
            return []

        object_tickers = set()
        for item in candidates:
            ticker = getattr(item, "ticker", None)
            if ticker:
                object_tickers.add(str(ticker).upper())

        cleaned = []
        seen = set()

        for item in candidates:
            ticker = getattr(item, "ticker", None)

            # Plain string ticker from watchlist.
            # Do not pass raw strings into filter/ranker because those expect
            # ScannerCandidate objects with .ticker.
            if isinstance(item, str):
                symbol = item.upper().strip()
                if symbol in object_tickers:
                    continue
                continue

            if ticker:
                symbol = str(ticker).upper()
                if symbol in seen:
                    continue
                seen.add(symbol)

            cleaned.append(item)

        return cleaned

    def _candidate_objects_only(self, candidates: list) -> list:
        """
        Keep only scanner/rank candidate objects.

        Watchlist files can add plain strings like "AAPL".
        ScannerFilterEngine and CandidateRanker require objects with .ticker.
        """
        cleaned = []
        dropped = []

        for item in candidates or []:
            if hasattr(item, "ticker"):
                cleaned.append(item)
            else:
                dropped.append(str(item))

        if dropped:
            log.warning(
                "[bot_runner] dropped raw string candidates before filter/ranker: %s",
                dropped[:20],
            )

        return cleaned


    def _filter_candidates(self, candidates: list, cycle: dict) -> list:

        candidates = self._candidate_objects_only(candidates)

        """
        Apply scanner filter engine.

        Supports multiple possible existing method names.
        """
        if not candidates:
            return []

        try:
            open_tickers = self._open_position_tickers()

            for method_name in [
                "filter_candidates",
                "filter",
                "apply_filters",
                "apply",
                "run",
            ]:
                method = getattr(self.scanner_filter, method_name, None)
                if not callable(method):
                    continue

                try:
                    filtered = method(candidates, open_positions=open_tickers)
                except TypeError:
                    try:
                        filtered = method(candidates, open_tickers)
                    except TypeError:
                        filtered = method(candidates)

                # ScannerFilterEngine returns FilterResult objects.
                # CandidateRanker expects ScannerCandidate objects.
                cleaned = []
                for item in filtered or []:
                    if hasattr(item, "candidate"):
                        if getattr(item, "passed", True):
                            cleaned.append(item.candidate)
                    else:
                        cleaned.append(item)

                return cleaned

            cycle["warnings"].append("scanner filter has no supported filter method ? using unfiltered candidates")
            return candidates

        except Exception as exc:
            cycle["errors"].append(f"scanner filter failed: {exc}")
            log.exception("[bot_runner] scanner filter failed")
            return candidates

    def _load_scan_universe(self) -> list[str]:
        """
        Load ticker universe from settings or watchlist JSON files.
        """
        scanner_cfg = self.settings.get("scanner", {})
        watchlist_cfg = self.settings.get("watchlist", {})

        tickers = []

        # Inline config tickers
        universe = (
            scanner_cfg.get("universe")
            or scanner_cfg.get("tickers")
            or scanner_cfg.get("symbols")
            or watchlist_cfg.get("symbols")
            or watchlist_cfg.get("tickers")
            or watchlist_cfg.get("universe")
            or []
        )

        if isinstance(universe, str):
            tickers.extend([t.strip().upper() for t in universe.split(",") if t.strip()])
        elif isinstance(universe, list):
            tickers.extend([str(t).strip().upper() for t in universe if str(t).strip()])

        # Watchlist JSON files from config paths
        for key in ["manual_watchlist_path", "scanner_candidates_path", "active_candidates_path"]:
            rel_path = watchlist_cfg.get(key)
            if not rel_path:
                continue

            file_path = Path(rel_path)
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path

            if not file_path.exists():
                continue

            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))

                if isinstance(data, dict):
                    raw = data.get("tickers") or data.get("symbols") or data.get("candidates") or []
                elif isinstance(data, list):
                    raw = data
                else:
                    raw = []

                for item in raw:
                    if isinstance(item, str):
                        tickers.append(item.strip().upper())
                    elif isinstance(item, dict):
                        symbol = item.get("ticker") or item.get("symbol")
                        if symbol:
                            tickers.append(str(symbol).strip().upper())

            except Exception as exc:
                log.warning("[bot_runner] failed to read watchlist file %s: %s", file_path, exc)

        # Fallback to manual loader if it has get_tickers()
        try:
            if hasattr(self.watchlist_loader, "get_tickers"):
                tickers.extend(self.watchlist_loader.get_tickers() or [])
        except Exception:
            pass

        cleaned = []
        seen = set()
        for ticker in tickers:
            ticker = str(ticker).strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                cleaned.append(ticker)

        return cleaned

    def _rank_candidates(self, candidates: list, cycle: dict) -> list:

        candidates = self._candidate_objects_only(candidates)

        """
        Rank candidates.
        """
        try:
            return self.candidate_ranker.rank(candidates)
        except Exception as exc:
            cycle["errors"].append(f"candidate ranker failed: {exc}")
            log.exception("[bot_runner] ranker failed")
            return candidates

    # ── Entry evaluation ──────────────────────────────────────────────────────

    def _evaluate_candidate(self, ranked_item: object, cycle: dict):
        """
        Analyze, score, gate, size, and possibly execute one candidate.
        """
        candidate = getattr(ranked_item, "candidate", ranked_item)
        ticker = getattr(candidate, "ticker", "") or getattr(candidate, "symbol", "")
        ticker = str(ticker).upper()

        if not ticker:
            return None, None

        try:
            data = self._get_market_data_for_ticker(ticker)
            bars = getattr(data, "bars", []) or []
            indicators = getattr(data, "indicators", None)
            current_price = getattr(data, "current_price", None) or getattr(data, "latest_price", 0.0)

            if not bars or not indicators or not current_price:
                cycle["warnings"].append(f"{ticker}: missing market data")
                return None, None

            context = self._build_analysis_context(
                ticker=ticker,
                candidate=candidate,
                data=data,
                bars=bars,
                indicators=indicators,
                current_price=current_price,
            )

            setup_results = run_all_setups(
                bars       = bars,
                indicators = indicators,
                context    = context,
                settings   = self.settings,
            )
            best = best_setup(setup_results)

            if not best:
                cycle["warnings"].append(f"{ticker}: no confirmed setup")
                return None, None

            rr_result = self.risk_reward.score(
                ticker       = ticker,
                entry_price  = best.entry_trigger or current_price,
                stop_price   = best.stop_area,
                target_price = best.target_area,
            )

            setup_score = self.setup_score.score(
                ticker             = ticker,
                setup_result       = best,
                indicators         = indicators,
                context            = context,
                risk_reward_result = rr_result,
            )

            historical_edge = self.historical_edge.score(
                setup_type = best.setup_name,
                ticker     = ticker,
            )

            probability = self.probability.score(
                ticker                 = ticker,
                setup_result           = best,
                indicators             = indicators,
                context                = context,
                setup_score_result     = setup_score,
                risk_reward_result     = rr_result,
                move_potential_result  = context.get("move_potential"),
                historical_edge_result = historical_edge,
            )

            preliminary_final = (
                setup_score.setup_score * 0.35
                + probability.probability_score * 0.35
                + rr_result.risk_reward_score * 0.15
                + getattr(context.get("move_potential"), "move_potential_score", 0.0) * 0.10
                + historical_edge.historical_edge_score * 0.05
            )

            account = self._get_account()
            position_size = self.position_sizer.calculate(
                ticker                    = ticker,
                entry_price               = best.entry_trigger or current_price,
                stop_price                = best.stop_area,
                account                   = account,
                final_trade_quality_score = preliminary_final,
            )

            account_risk = self.account_risk.check(
                account            = account,
                positions          = self._positions_as_dicts(),
                new_position_value = position_size.position_value,
                daily_pnl_dollars  = self._daily_pnl(),
                paper_trading      = True,
            )

            quote = getattr(data, "quote", {}) or {}
            exec_quality = self.execution_quality.check(
                ticker                = ticker,
                planned_entry         = best.entry_trigger or current_price,
                current_price         = current_price,
                bid                   = _safe_float(quote.get("bid_price", quote.get("bid", getattr(data, "bid", 0.0)))),
                ask                   = _safe_float(quote.get("ask_price", quote.get("ask", getattr(data, "ask", 0.0)))),
                quote_timestamp       = quote.get("timestamp") or quote.get("t"),
                open_positions        = self._open_position_tickers(),
                account               = account,
                paper_trading         = True,
                required_buying_power = position_size.position_value,
            )

            decision = self.trade_gate.evaluate(
                ticker                   = ticker,
                setup_result             = best,
                setup_score_result       = setup_score,
                probability_result       = probability,
                risk_reward_result       = rr_result,
                move_potential_result    = context.get("move_potential"),
                execution_quality_result = exec_quality,
                risk_result              = account_risk,
                historical_edge_result   = historical_edge,
                context                  = context,
                position_size            = position_size,
            )

            if decision.decision != "approved_for_paper_buy":
                return decision, None

            order = self.order_executor.execute_buy(
                decision    = decision,
                current_ask = _safe_float(quote.get("ask_price", quote.get("ask", getattr(data, "ask", 0.0)))),
            )

            self.trade_logger.log_new_trade(
                decision      = decision,
                order_result  = order,
                extra_context = {
                    "candidate": _to_dict(candidate),
                    "setup_results": {k: v.to_dict() for k, v in setup_results.items()},
                    "analysis": _context_summary(context),
                },
            )

            return decision, order

        except Exception as exc:
            log.exception("[bot_runner] candidate evaluation failed for %s", ticker)
            cycle["errors"].append(f"{ticker}: evaluation failed: {exc}")
            return None, None

    def _get_market_data_for_ticker(self, ticker: str):
        """
        Compatibility helper for MarketDataProvider.

        Your existing provider uses:
            get(ticker)

        Some newer code expected:
            get_ticker_data(ticker)
        """
        if hasattr(self.market_data, "get_ticker_data"):
            return self.market_data.get_ticker_data(ticker)

        if hasattr(self.market_data, "get"):
            return self.market_data.get(ticker)

        raise AttributeError("MarketDataProvider has no get_ticker_data() or get() method")

    def _build_analysis_context(
        self,
        ticker: str,
        candidate: object,
        data: object,
        bars: list[dict],
        indicators: object,
        current_price: float,
    ) -> dict:
        """
        Build analysis context shared by setups/scoring/gates.
        """
        try:
            key_levels = self.key_levels.analyze(
                ticker=ticker,
                bars=bars,
                current_price=current_price,
            )
        except TypeError:
            try:
                key_levels = self.key_levels.analyze(
                    ticker=ticker,
                    bars=bars,
                )
            except TypeError:
                key_levels = self.key_levels.analyze(bars)

        or_result = self.opening_range.analyze(
            ticker=ticker,
            bars=bars,
            current_price=current_price,
        )

        sweep_result = self.liquidity_sweep.analyze(
            ticker=ticker,
            bars=bars,
            current_price=current_price,
            key_levels=_key_level_dict(key_levels, indicators, or_result),
        )

        structure = self.session_structure.analyze(
            ticker=ticker,
            bars=bars,
            current_price=current_price,
        )

        try:
            fib_result = self.fibonacci.analyze(
                ticker=ticker,
                current_price=current_price,
                indicators=indicators,
            )
        except TypeError:
            try:
                fib_result = self.fibonacci.analyze(
                    ticker=ticker,
                    current_price=current_price,
                )
            except TypeError:
                try:
                    fib_result = self.fibonacci.analyze(
                        current_price=current_price,
                        indicators=indicators,
                    )
                except TypeError:
                    fib_result = self.fibonacci.analyze()

        failed_reclaim = self.failed_reclaim.analyze(
            ticker=ticker,
            bars=bars,
            current_price=current_price,
            key_levels=_key_level_dict(key_levels, indicators, or_result),
        )

        move_potential = None

        for method_name in ["analyze", "score", "calculate", "evaluate", "check"]:
            method = getattr(self.move_potential, method_name, None)
            if not callable(method):
                continue

            try:
                move_potential = method(
                    ticker=ticker,
                    current_price=current_price,
                    key_levels=key_levels,
                    fib_result=fib_result,
                    indicators=indicators,
                    candidate=candidate,
                )
                break
            except TypeError:
                try:
                    move_potential = method(
                        ticker=ticker,
                        current_price=current_price,
                        key_levels=key_levels,
                    )
                    break
                except TypeError:
                    try:
                        move_potential = method(
                            current_price=current_price,
                            key_levels=key_levels,
                        )
                        break
                    except TypeError:
                        try:
                            move_potential = method()
                            break
                        except TypeError:
                            continue

        return {
            "ticker": ticker,
            "candidate": candidate,
            "data": data,
            "bars": bars,
            "indicators": indicators,
            "current_price": current_price,
            "vwap": getattr(indicators, "vwap", None),
            "key_levels": key_levels,
            "or_result": or_result,
            "sweep_result": sweep_result,
            "structure": structure,
            "fib_result": fib_result,
            "failed_reclaim": failed_reclaim,
            "move_potential": move_potential,
        }

    # ── Open trade management ─────────────────────────────────────────────────

    def _sync_trade_statuses(self, cycle: dict) -> None:
        """
        Update local trade JSON files from broker status.
        """
        try:
            results = self.trade_status.update_all_open_trades()
            cycle["trade_status_updates"] = [r.to_dict() for r in results]
        except Exception as exc:
            cycle["errors"].append(f"trade status update failed: {exc}")
            log.exception("[bot_runner] trade status update failed")

    def _manage_open_trades(self, cycle: dict) -> None:
        """
        Evaluate exits for open trades.
        """
        open_trades = self.trade_logger.list_open_trades()
        exit_decisions = []
        exit_orders = []

        for trade in open_trades:
            try:
                ticker = str(trade.get("ticker", "")).upper()
                if not ticker:
                    continue

                pos = self._position_for_ticker(ticker)
                if not pos:
                    continue

                data = self._get_market_data_for_ticker(ticker)
                current_price = getattr(data, "current_price", None) or getattr(data, "latest_price", 0.0)
                indicators = getattr(data, "indicators", None)
                bars = getattr(data, "bars", []) or []

                context = {}
                if bars and indicators and current_price:
                    context = self._build_analysis_context(
                        ticker=ticker,
                        candidate={},
                        data=data,
                        bars=bars,
                        indicators=indicators,
                        current_price=current_price,
                    )

                exit_decision = self.exit_manager.evaluate(
                    trade                = trade,
                    current_price        = current_price,
                    current_position_qty = int(abs(getattr(pos, "qty", 0) or 0)),
                    indicators           = indicators,
                    context              = context,
                )
                exit_decisions.append(exit_decision)

                file_path = trade.get("_file_path", "")
                if exit_decision.action in ("move_stop", "trail_stop") and exit_decision.new_stop_price:
                    self.trade_logger.update_stop(
                        file_path,
                        exit_decision.new_stop_price,
                        reason=exit_decision.exit_reason,
                    )

                if exit_decision.should_exit and exit_decision.quantity_to_sell > 0:
                    order = self.order_executor.execute_sell(
                        ticker      = ticker,
                        quantity    = exit_decision.quantity_to_sell,
                        reason      = exit_decision.exit_reason,
                        current_bid = self._current_bid(data),
                    )
                    exit_orders.append(order)

                    if exit_decision.partial_exit:
                        self.trade_logger.mark_partial_taken(file_path, exit_decision, order)
                    elif exit_decision.full_exit:
                        self.trade_logger.close_trade(
                            trade_path     = file_path,
                            exit_price     = current_price,
                            exit_quantity  = exit_decision.quantity_to_sell,
                            close_reason   = exit_decision.exit_reason,
                            order_result   = order,
                            exit_result    = exit_decision,
                        )

            except Exception as exc:
                cycle["errors"].append(f"exit management failed: {exc}")
                log.exception("[bot_runner] exit management failed")

        cycle["exit_decisions"] = [e.to_dict() for e in exit_decisions]
        cycle["exit_orders"] = [o.to_dict() for o in exit_orders]

    # ── Learning / dashboard ──────────────────────────────────────────────────

    def _review_and_learn(self, cycle: dict) -> None:
        """
        Review closed trades and update learning summary.
        """
        try:
            reviews = self.backtest_reviewer.review_ready_trades(limit=10)
            cycle["backtest_reviews"] = [r.to_dict() for r in reviews]
        except Exception as exc:
            cycle["warnings"].append(f"backtest review failed: {exc}")

        try:
            cycle["performance"] = self.performance.calculate().to_dict()
        except Exception as exc:
            cycle["warnings"].append(f"performance calculation failed: {exc}")

        try:
            cycle["learning_summary"] = self.learning.generate_and_save().to_dict()
        except Exception as exc:
            cycle["warnings"].append(f"learning summary failed: {exc}")

    def _write_dashboard(self, cycle: dict) -> None:
        """
        Refresh dashboard state.
        """
        try:
            self.dashboard.write_state(
                bot_status          = cycle.get("status", "idle"),
                scanner_results     = cycle.get("scanner_candidates", []),
                ranked_candidates   = cycle.get("ranked_candidates", []),
                recent_decisions    = cycle.get("decisions", []),
                performance_summary = cycle.get("performance", {}),
                learning_summary    = cycle.get("learning_summary", {}),
                errors              = cycle.get("errors", []),
                warnings            = cycle.get("warnings", []),
                extra               = {
                    "orders": cycle.get("orders", []),
                    "exit_orders": cycle.get("exit_orders", []),
                    "exit_decisions": cycle.get("exit_decisions", []),
                },
            )
        except Exception:
            log.exception("[bot_runner] failed to write dashboard state")

    # ── Runtime / account helpers ─────────────────────────────────────────────

    def _runtime_allowed(self) -> tuple[bool, str]:
        """
        Check control state and safety settings.
        """
        mode = self.settings.get("mode", {})

        if not bool(mode.get("bot_enabled", True)):
            return False, "bot_enabled is false"
        if bool(mode.get("safety_lock", False)):
            return False, "safety_lock is enabled"
        if bool(mode.get("allow_live_money", False)):
            return False, "allow_live_money is true — refusing to run"
        if not bool(mode.get("paper_trading_only", True)):
            return False, "paper_trading_only is false"

        return self.control.runtime_allows_loop()

    def _create_news_provider(self):
        """
        Create NewsCatalystProvider when credentials are available.
        Falls back to a null provider during dry-run/import testing.
        """
        news_cfg = self.settings.get("news", {}) or self.settings.get("market_data", {}).get("news", {})

        api_key = (
            news_cfg.get("api_key")
            or os.environ.get("NEWS_API_KEY")
            or os.environ.get("BENZINGA_API_KEY")
            or os.environ.get("ALPACA_API_KEY")
            or ""
        )

        secret_key = (
            news_cfg.get("secret_key")
            or os.environ.get("NEWS_SECRET_KEY")
            or os.environ.get("ALPACA_SECRET_KEY")
            or ""
        )

        try:
            return NewsCatalystProvider(api_key, secret_key)
        except TypeError:
            try:
                return NewsCatalystProvider(self.settings)
            except Exception as exc:
                log.warning("[bot_runner] News provider unavailable, using null provider: %s", exc)
                return NullNewsCatalystProvider()
        except Exception as exc:
            log.warning("[bot_runner] News provider unavailable, using null provider: %s", exc)
            return NullNewsCatalystProvider()


    def _create_alpaca_data_client(self):
        """
        Create Alpaca data client when API keys exist.
        In dry_run without keys, return a null client so BotRunner can still initialize.
        """
        try:
            return AlpacaDataClient.from_env()
        except Exception as exc:
            log.warning("[bot_runner] Alpaca client unavailable, using null data client: %s", exc)
            return NullAlpacaDataClient()

    def _get_alpaca_client(self):
        """
        Use market_data provider's Alpaca client when available.
        """
        return getattr(self.market_data, "alpaca_client", None) or getattr(self.market_data, "_alpaca", None)

    def _get_account(self) -> dict:
        if not self.alpaca_client:
            return {}
        try:
            return self.alpaca_client.get_account()
        except Exception:
            return {}

    def _open_position_tickers(self) -> list[str]:
        if not self.position_tracker:
            return []
        try:
            return self.position_tracker.open_tickers(force_refresh=True)
        except Exception:
            return []

    def _positions_as_dicts(self) -> list[dict]:
        if not self.position_tracker:
            return []
        try:
            result = self.position_tracker.get_all_positions(force_refresh=True)
            return positions_to_alpaca_like_dicts(result)
        except Exception:
            return []

    def _position_for_ticker(self, ticker: str):
        if not self.position_tracker:
            return None
        try:
            return self.position_tracker.get_position_for_ticker(ticker, force_refresh=True)
        except Exception:
            return None

    def _daily_pnl(self) -> float:
        try:
            summary = self.performance.calculate(limit=100)
            today = time.strftime("%Y-%m-%d")
            return float(summary.daily_pl.get(today, 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _current_bid(data: object) -> float:
        quote = getattr(data, "quote", {}) or {}
        return _safe_float(quote.get("bid_price", quote.get("bid", getattr(data, "bid", 0.0))))


# ── Convenience wrappers ──────────────────────────────────────────────────────

def run_once(settings: Optional[dict] = None) -> dict:
    """
    Convenience function to run one cycle.
    """
    runner = BotRunner(settings=settings)
    return runner.run_once()


def run_loop(settings: Optional[dict] = None) -> None:
    """
    Convenience function to run the live loop.
    """
    runner = BotRunner(settings=settings)
    runner.run_loop()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_dict(obj: object) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _key_level_dict(key_levels: object, indicators: object, or_result: object) -> dict[str, float]:
    """
    Convert analysis objects into failed_reclaim_detector key-level dict.
    """
    levels: dict[str, float] = {}

    vwap = getattr(indicators, "vwap", None)
    if vwap:
        levels["vwap"] = vwap

    if or_result and getattr(or_result, "primary_or", None):
        primary = or_result.primary_or
        if getattr(primary, "high", None):
            levels["or_high"] = primary.high
        if getattr(primary, "low", None):
            levels["or_low"] = primary.low

    if key_levels:
        support = getattr(key_levels, "nearest_support", None)
        resistance = getattr(key_levels, "nearest_resistance", None)
        if support and getattr(support, "price", None):
            levels["key_support"] = support.price
        if resistance and getattr(resistance, "price", None):
            levels["key_resistance"] = resistance.price

    return levels


def _context_summary(context: dict) -> dict:
    """
    Return dashboard/log-safe context summary.
    """
    return {
        "ticker": context.get("ticker"),
        "current_price": context.get("current_price"),
        "vwap": context.get("vwap"),
        "has_key_levels": context.get("key_levels") is not None,
        "has_or_result": context.get("or_result") is not None,
        "has_sweep_result": context.get("sweep_result") is not None,
        "has_fib_result": context.get("fib_result") is not None,
        "has_failed_reclaim": context.get("failed_reclaim") is not None,
    }


if __name__ == "__main__":
    run_once()
