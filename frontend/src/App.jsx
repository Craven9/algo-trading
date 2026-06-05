import React, { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [state, setState] = useState(null);
  const [health, setHealth] = useState(null);
  const [control, setControl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastLoaded, setLastLoaded] = useState("");
  const [error, setError] = useState("");

  async function apiGet(path) {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) throw new Error(`${path} failed with ${response.status}`);
    return await response.json();
  }

  async function apiPost(path) {
    const response = await fetch(`${API_BASE}${path}`, { method: "POST" });
    if (!response.ok) throw new Error(`${path} failed with ${response.status}`);
    return await response.json();
  }

  async function loadDashboard() {
    try {
      setLoading(true);
      setError("");

      const [healthJson, stateJson, controlJson] = await Promise.all([
        apiGet("/health"),
        apiGet("/api/dashboard/state"),
        apiGet("/api/bot/control-state")
      ]);

      setHealth(healthJson);
      setState(stateJson);
      setControl(controlJson);
      setLastLoaded(new Date().toLocaleTimeString());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function postAndReload(path) {
    try {
      setLoading(true);
      setError("");
      await apiPost(path);
      await loadDashboard();
    } catch (err) {
      setError(String(err));
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
    const id = setInterval(loadDashboard, 10000);
    return () => clearInterval(id);
  }, []);

  const decisions = useMemo(() => {
    const raw = state?.recent_decisions;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === "object") return Object.values(raw);
    return [];
  }, [state]);

  const candidates = state?.scanner?.ranked_candidates?.length
    ? state.scanner.ranked_candidates
    : state?.scanner?.candidates || [];

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">AI Trading Assistant</p>
          <h1>Algo Bot Trader Dashboard</h1>
          <p className="sub">Scanner, decisions, risk, trades, and learning in one control panel.</p>
        </div>

        <div className="heroRight">
          <span className={`status ${state?.bot?.status === "started" ? "green" : ""}`}>
            {state?.bot?.status || "loading"}
          </span>
          <p>Updated: {lastLoaded || "loading..."}</p>
        </div>
      </header>

      {error && <div className="banner error">Error: {error}</div>}

      <section className="controls">
        <button onClick={loadDashboard} disabled={loading}>{loading ? "Loading..." : "Reload"}</button>
        <button onClick={() => postAndReload("/api/dashboard/refresh")} disabled={loading}>Refresh Dashboard</button>
        <button className="good" onClick={() => postAndReload("/api/bot/start")} disabled={loading}>Start Bot</button>
        <button className="warn" onClick={() => postAndReload("/api/bot/pause")} disabled={loading}>Pause Bot</button>
        <button className="bad" onClick={() => postAndReload("/api/bot/stop")} disabled={loading}>Stop Bot</button>
      </section>

      <section className="metrics">
        <Metric title="Backend" value={health?.ok ? "Online" : "Offline"} detail={health?.service || "API not loaded"} tone={health?.ok ? "good" : "bad"} />
        <Metric title="Bot" value={control?.bot_should_run ? "Running" : "Stopped"} detail={`Paused: ${yesNo(control?.paused)}`} tone={control?.bot_should_run ? "good" : "neutral"} />
        <Metric title="Candidates" value={state?.scanner?.candidate_count ?? 0} detail={`Ranked: ${state?.scanner?.ranked_count ?? 0}`} tone="good" />
        <Metric title="Open Trades" value={state?.trades?.open_count ?? 0} detail={`Closed: ${state?.trades?.closed_count ?? 0}`} tone="neutral" />
        <Metric title="Total P/L" value={`$${fmt(state?.performance?.total_realized_pl ?? 0)}`} detail={`Win rate: ${fmt(state?.performance?.win_rate ?? 0)}%`} tone={(state?.performance?.total_realized_pl ?? 0) >= 0 ? "good" : "bad"} />
        <Metric title="Dry Run" value={yesNo(state?.mode?.dry_run)} detail={`Paper only: ${yesNo(state?.mode?.paper_trading_only)}`} tone={state?.mode?.dry_run ? "warn" : "good"} />
      </section>

      <section className="layout">
        <Card title="Safety">
          <Rows rows={[
            ["Bot should run", yesNo(control?.bot_should_run)],
            ["Paused", yesNo(control?.paused)],
            ["Dry run", yesNo(state?.mode?.dry_run)],
            ["Paper trading only", yesNo(state?.mode?.paper_trading_only)],
            ["Allow live money", yesNo(state?.mode?.allow_live_money)],
            ["Safety lock", yesNo(state?.mode?.safety_lock)]
          ]} />
        </Card>

        <Card title="Trade Decisions">
          <DecisionList decisions={decisions} />
        </Card>
      </section>

      <section className="layout">
        <Card title="Ranked Candidates">
          <CandidateTable candidates={candidates} />
        </Card>

        <Card title="Warnings / Errors">
          <h3>Warnings</h3>
          <MessageList items={toList(state?.warnings)} empty="No warnings." />
          <h3>Errors</h3>
          <MessageList items={toList(state?.errors)} empty="No errors." error />
        </Card>
      </section>

      <section className="layout">
        <Card title="Performance">
          <Rows rows={[
            ["Trades", state?.performance?.trade_count ?? 0],
            ["Wins", state?.performance?.wins ?? 0],
            ["Losses", state?.performance?.losses ?? 0],
            ["Breakevens", state?.performance?.breakevens ?? 0],
            ["Average trade P/L", `$${fmt(state?.performance?.average_trade_pl ?? 0)}`],
            ["Profit factor", fmt(state?.performance?.profit_factor ?? 0)]
          ]} />
        </Card>

        <Card title="Learning">
          <Rows rows={[
            ["Reviewed trades", state?.learning?.reviewed_trade_count ?? 0],
            ["Strongest setup", state?.learning?.strongest_setup || "-"],
            ["Weakest setup", state?.learning?.weakest_setup || "-"]
          ]} />
          <h3>Notes</h3>
          <MessageList items={toList(state?.learning?.learning_notes)} empty="No learning notes yet." />
        </Card>
      </section>

      <details className="raw">
        <summary>Raw dashboard JSON</summary>
        <pre>{JSON.stringify(state, null, 2)}</pre>
      </details>
    </div>
  );
}

function Metric({ title, value, detail, tone }) {
  return (
    <div className={`metric ${tone || "neutral"}`}>
      <p>{title}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {children}
    </div>
  );
}

function Rows({ rows }) {
  return (
    <div className="rows">
      {rows.map(([label, value]) => (
        <div className="row" key={label}>
          <span>{label}</span>
          <strong>{String(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function CandidateTable({ candidates }) {
  if (!candidates || candidates.length === 0) return <div className="empty">No candidates yet.</div>;

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Rank</th>
            <th>Price</th>
            <th>RVOL</th>
            <th>Change</th>
            <th>Spread</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((item, index) => {
            const c = item?.candidate || item;
            return (
              <tr key={`${c?.ticker || index}-${index}`}>
                <td className="ticker">{c?.ticker || "-"}</td>
                <td>{fmt(item?.rank_score ?? c?.rank_score ?? "-")}</td>
                <td>${fmt(c?.price)}</td>
                <td>{fmt(c?.relative_volume)}x</td>
                <td>{fmt(c?.day_change_pct)}%</td>
                <td>{fmt(c?.spread_percent)}%</td>
                <td>{c?.candidate_reason || item?.rank_reasons?.join(", ") || "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DecisionList({ decisions }) {
  if (!decisions || decisions.length === 0) return <div className="empty">No decisions yet.</div>;

  return (
    <div className="decisions">
      {decisions.map((d, i) => {
        const rejected = String(d?.decision || "").toLowerCase().includes("reject");
        const approved = String(d?.decision || "").toLowerCase().includes("approved");

        return (
          <div className={`decision ${approved ? "approved" : rejected ? "rejected" : ""}`} key={i}>
            <div className="decisionTop">
              <strong>{d?.ticker || "-"}</strong>
              <span className={`pill ${approved ? "goodPill" : rejected ? "badPill" : ""}`}>{d?.decision || "-"}</span>
            </div>
            <p className="setup">{d?.setup || "No setup"}</p>
            <div className="scoreChips">
              <span>Final {fmt(d?.scores?.final_trade_quality_score)}</span>
              <span>Setup {fmt(d?.scores?.setup_score)}</span>
              <span>Prob {fmt(d?.scores?.probability_score)}</span>
              <span>R/R {fmt(d?.scores?.risk_reward_score)}</span>
            </div>
            <ul>
              {(d?.reasons || []).slice(0, 5).map((r, index) => <li key={index}>{r}</li>)}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

function MessageList({ items, empty, error }) {
  if (!items || items.length === 0) return <div className="empty">{empty}</div>;

  return (
    <ul className={error ? "messages errors" : "messages"}>
      {items.map((item, i) => <li key={i}>{String(item)}</li>)}
    </ul>
  );
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function yesNo(value) {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "-";
}

function toList(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "object") return Object.values(value);
  return [value];
}
