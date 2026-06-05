import React, { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [state, setState] = useState(null);
  const [health, setHealth] = useState(null);
  const [control, setControl] = useState(null);
  const [error, setError] = useState("");

  async function loadDashboard() {
    try {
      setError("");

      const healthRes = await fetch(`${API_BASE}/health`);
      setHealth(await healthRes.json());

      const stateRes = await fetch(`${API_BASE}/api/dashboard/state`);
      setState(await stateRes.json());

      const controlRes = await fetch(`${API_BASE}/api/bot/control-state`);
      setControl(await controlRes.json());
    } catch (err) {
      setError(String(err));
    }
  }

  async function post(path) {
    await fetch(`${API_BASE}${path}`, { method: "POST" });
    await loadDashboard();
  }

  useEffect(() => {
    loadDashboard();
  }, []);

  return (
    <div style={{ fontFamily: "Arial", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>Algo Bot Trader Dashboard</h1>

      {error && (
        <div style={{ background: "#ffe0e0", padding: 12, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
        <button onClick={loadDashboard}>Reload</button>
        <button onClick={() => post("/api/dashboard/refresh")}>Refresh Dashboard</button>
        <button onClick={() => post("/api/bot/start")}>Start Bot</button>
        <button onClick={() => post("/api/bot/pause")}>Pause Bot</button>
        <button onClick={() => post("/api/bot/stop")}>Stop Bot</button>
      </div>

      <section style={box}>
        <h2>Backend Health</h2>
        <pre>{JSON.stringify(health, null, 2)}</pre>
      </section>

      <section style={box}>
        <h2>Bot Status</h2>
        <p>Status: {state?.bot?.status || "loading..."}</p>
        <p>Bot Should Run: {String(control?.bot_should_run)}</p>
        <p>Paused: {String(control?.paused)}</p>
        <p>Dry Run: {String(state?.mode?.dry_run)}</p>
        <p>Paper Trading Only: {String(state?.mode?.paper_trading_only)}</p>
        <p>Allow Live Money: {String(state?.mode?.allow_live_money)}</p>
      </section>

      <section style={box}>
        <h2>Scanner</h2>
        <p>Candidates: {state?.scanner?.candidate_count ?? 0}</p>
        <p>Ranked: {state?.scanner?.ranked_count ?? 0}</p>
      </section>

      <section style={box}>
        <h2>Trades</h2>
        <p>Open Trades: {state?.trades?.open_count ?? 0}</p>
        <p>Closed Trades: {state?.trades?.closed_count ?? 0}</p>
      </section>

      <section style={box}>
        <h2>Performance</h2>
        <p>Total P/L: ${state?.performance?.total_realized_pl ?? 0}</p>
        <p>Win Rate: {state?.performance?.win_rate ?? 0}%</p>
        <p>Trades: {state?.performance?.trade_count ?? 0}</p>
      </section>

      <section style={box}>
        <h2>Warnings / Errors</h2>
        <h3>Warnings</h3>
        <pre>{JSON.stringify(state?.warnings || [], null, 2)}</pre>
        <h3>Errors</h3>
        <pre>{JSON.stringify(state?.errors || [], null, 2)}</pre>
      </section>

      <section style={box}>
        <h2>Raw Dashboard State</h2>
        <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(state, null, 2)}</pre>
      </section>
    </div>
  );
}

const box = {
  border: "1px solid #ccc",
  padding: 16,
  marginBottom: 16,
  borderRadius: 8
};
