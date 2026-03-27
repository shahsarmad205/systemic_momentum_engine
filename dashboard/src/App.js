import React, { useCallback, useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

/**
 * Valid absolute URL for fetch(). Fixes bad .env values and avoids WebKit
 * "The string did not match the expected pattern" when fetch() gets an invalid URL.
 */
function normalizeApiBase(raw) {
  let s = String(raw ?? "").trim();
  if (!s) s = "http://localhost:5001";
  if (!/^https?:\/\//i.test(s)) s = `http://${s.replace(/^\/+/, "")}`;
  s = s.replace(/\/$/, "");
  try {
    const u = new URL(s);
    if (!/^https?:$/i.test(u.protocol.replace(/:$/, ""))) {
      s = "http://localhost:5001";
    }
  } catch {
    s = "http://localhost:5001";
  }
  return s;
}

const API_BASE = normalizeApiBase(process.env.REACT_APP_API_URL);

const REFRESH_MS = 30000;

function formatMoney(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(v));
}

function formatPct(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${(Number(v) * 100).toFixed(2)}%`;
}

export default function App() {
  const [account, setAccount] = useState({});
  const [positions, setPositions] = useState([]);
  const [posMeta, setPosMeta] = useState({});
  const [signals, setSignals] = useState([]);
  const [signalsFile, setSignalsFile] = useState("");
  const [equitySeries, setEquitySeries] = useState([]);
  const [icSeries, setIcSeries] = useState([]);
  const [rolling, setRolling] = useState({});
  const [error, setError] = useState("");
  const [lastFetch, setLastFetch] = useState(null);

  const loadAll = useCallback(async () => {
    setError("");
    const safeJson = async (path) => {
      const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
      let r;
      try {
        r = await fetch(url);
      } catch (err) {
        console.error("fetch failed:", url, err);
        throw new Error(`Network error calling ${url} (${err.message || err})`);
      }
      const text = await r.text();
      if (!r.ok) {
        throw new Error(`${url} → HTTP ${r.status}: ${text.slice(0, 120)}`);
      }
      try {
        return JSON.parse(text);
      } catch (err) {
        console.error("Non-JSON response from", url, text.slice(0, 300));
        throw new SyntaxError(
          `Expected JSON from ${url}; got ${(text || "").trim().slice(0, 40) || "(empty)"}… Is the Flask API running on ${API_BASE}?`
        );
      }
    };
    try {
      const [acc, posRes, sigRes, eq, ic, roll] = await Promise.all([
        safeJson("/api/account"),
        safeJson("/api/positions"),
        safeJson("/api/signals"),
        safeJson("/api/equity"),
        safeJson("/api/ic"),
        safeJson("/api/rolling"),
      ]);
      setAccount(acc || {});
      setPositions(Array.isArray(posRes?.positions) ? posRes.positions : []);
      setPosMeta({
        source: posRes?.source,
        timestamp: posRes?.timestamp,
      });
      setSignals(Array.isArray(sigRes?.signals) ? sigRes.signals : []);
      setSignalsFile(sigRes?.file || "");
      setEquitySeries(Array.isArray(eq) ? eq : []);
      setIcSeries(Array.isArray(ic) ? ic : []);
      setRolling(roll && typeof roll === "object" ? roll : {});
      setLastFetch(new Date());
    } catch (e) {
      console.error(e);
      setError(e.message || String(e));
    }
  }, []);

  useEffect(() => {
    loadAll();
    const id = setInterval(loadAll, REFRESH_MS);
    return () => clearInterval(id);
  }, [loadAll]);

  const icChartData = icSeries.map((row) => ({
    date: row.date,
    ic_daily: row.ic_daily,
    rolling_ic: row.rolling_ic,
  }));

  return (
    <div className="app">
      <header>
        <h1>Strategy monitoring</h1>
        <div className="meta">
          API: <code>{API_BASE}</code>
          <br />
          {lastFetch
            ? `Last updated: ${lastFetch.toLocaleTimeString()} · refresh ${REFRESH_MS / 1000}s`
            : "Loading…"}
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="cards">
        <div className="card">
          <div className="label">Equity</div>
          <div className="value">{formatMoney(account.equity)}</div>
        </div>
        <div className="card">
          <div className="label">Cash</div>
          <div className="value">{formatMoney(account.cash)}</div>
        </div>
        <div className="card">
          <div className="label">Buying power</div>
          <div className="value">
            {account.buying_power != null
              ? formatMoney(account.buying_power)
              : "—"}
          </div>
        </div>
        <div className="card">
          <div className="label">Total return (PnL series)</div>
          <div className="value">{formatPct(account.total_return)}</div>
        </div>
        <div className="card">
          <div className="label">Rolling IC (latest)</div>
          <div className="value">
            {rolling.rolling_ic != null ? rolling.rolling_ic.toFixed(4) : "—"}
          </div>
        </div>
        <div className="card">
          <div className="label">Rolling Sharpe (latest)</div>
          <div className="value">
            {rolling.rolling_sharpe != null
              ? rolling.rolling_sharpe.toFixed(3)
              : "—"}
          </div>
        </div>
      </div>

      <section>
        <h2>Equity curve</h2>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={equitySeries} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="#8b949e" />
              <YAxis
                tick={{ fontSize: 11 }}
                stroke="#8b949e"
                tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${v / 1000}k`)}
              />
              <Tooltip
                contentStyle={{ background: "#21262d", border: "1px solid #30363d" }}
                labelStyle={{ color: "#e6edf3" }}
              />
              <Legend />
              <Line type="monotone" dataKey="equity" stroke="#58a6ff" dot={false} name="Equity" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section>
        <h2>IC & rolling IC</h2>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={icChartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="#8b949e" />
              <YAxis tick={{ fontSize: 11 }} stroke="#8b949e" />
              <Tooltip
                contentStyle={{ background: "#21262d", border: "1px solid #30363d" }}
                labelStyle={{ color: "#e6edf3" }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="ic_daily"
                stroke="#3fb950"
                dot={false}
                name="Daily IC"
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="rolling_ic"
                stroke="#d29922"
                dot={false}
                name="Rolling IC"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section>
        <h2>
          Top signals {signalsFile ? <span className="meta">({signalsFile})</span> : null}
        </h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {(signals[0] ? Object.keys(signals[0]) : ["ticker", "score"]).map((k) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {signals.length === 0 ? (
                <tr>
                  <td colSpan={99}>No signal data</td>
                </tr>
              ) : (
                signals.map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).map(([k, v]) => (
                      <td key={k}>{v == null ? "" : String(v)}</td>
                    ))}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2>
          Positions{" "}
          {posMeta.source ? (
            <span className="meta">
              ({posMeta.source}
              {posMeta.timestamp ? ` · ${posMeta.timestamp}` : ""})
            </span>
          ) : null}
        </h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {(positions[0]
                  ? Object.keys(positions[0])
                  : ["ticker", "notional", "side"]
                ).map((k) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr>
                  <td colSpan={99}>No positions</td>
                </tr>
              ) : (
                positions.map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).map(([k, v]) => (
                      <td key={k}>{v == null ? "" : String(v)}</td>
                    ))}
                  </tr>
                ))
                )}
              </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
