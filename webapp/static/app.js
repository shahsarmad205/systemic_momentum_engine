const API = "";

async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function post(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function el(id) { return document.getElementById(id); }

function renderTable(containerId, rows, columns) {
  const c = el(containerId);
  if (!rows.length) {
    c.innerHTML = "<p class='muted'>No data. Run backtest or use mock signals.</p>";
    return;
  }
  const cols = columns || Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  cols.forEach((col) => { html += `<th>${col}</th>`; });
  html += "</tr></thead><tbody>";
  rows.forEach((row) => {
    html += "<tr>";
    cols.forEach((col) => {
      let v = row[col];
      if (v === undefined || v === null) v = "";
      if (col === "rank_tag") {
        if (v === "long") v = "<span class='tag-long'>LONG</span>";
        else if (v === "short") v = "<span class='tag-short'>SHORT</span>";
        else v = "";
      }
      html += `<td>${v}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  c.innerHTML = html;
}

async function loadUniverse() {
  const data = await get("/api/universe");
  el("ticker-input").value = data.tickers.join(" ");
  renderTable("universe-table", data.details, ["ticker", "sector", "price"]);
}

async function saveUniverse() {
  const raw = el("ticker-input").value;
  const tickers = raw.split(/[\s,]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
  await post("/api/universe", { tickers });
  await loadUniverse();
}

async function uploadTickers(e) {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(API + "/api/universe/upload", { method: "POST", body: fd });
  const data = await r.json();
  if (data.tickers) {
    el("ticker-input").value = data.tickers.join(" ");
    await loadUniverse();
  }
}

async function loadConfig() {
  const cfg = await get("/api/config");
  el("top-longs").value = cfg.top_longs;
  el("top-shorts").value = cfg.top_shorts;
  el("market-neutral").checked = cfg.market_neutral;
  el("rebalance-daily").checked = cfg.cross_sectional_rebalance_daily;
  el("min-strength").value = cfg.min_signal_strength;
  el("initial-capital").value = cfg.initial_capital;
}

async function computeRanking() {
  const body = {
    top_longs: parseInt(el("top-longs").value, 10) || 5,
    top_shorts: parseInt(el("top-shorts").value, 10) || 5,
    market_neutral: el("market-neutral").checked,
    min_signal_strength: parseFloat(el("min-strength").value) || 0,
    initial_capital: parseFloat(el("initial-capital").value) || 100000,
  };
  await post("/api/config", body);
  const result = await post("/api/ranking/compute", body);
  el("equal-weight-out").textContent = "Equal-weight size (per name): $" + (result.equal_weight_size || 0).toLocaleString();
  el("long-targets").textContent = (result.long_targets || []).join(", ") || "—";
  el("short-targets").textContent = (result.short_targets || []).join(", ") || "—";
  renderTable("ranking-table", result.sorted_rows || [], ["ticker", "adjusted_score", "signal", "confidence", "rank_tag"]);
}

async function loadDailyPositions() {
  const data = await get("/api/positions/daily");
  renderTable("positions-table", data.rows || [], null);
}

async function loadTradesEquity() {
  const [trades, equity, metrics] = await Promise.all([
    get("/api/trades"),
    get("/api/equity"),
    get("/api/metrics"),
  ]);
  renderTable("trades-table", (trades.rows || []).slice(0, 50), null);
  el("metrics-pre").textContent = JSON.stringify(metrics, null, 2);
  drawEquityChart(equity.rows || []);
}

function drawEquityChart(rows) {
  const canvas = el("equity-chart");
  if (!rows.length || typeof Chart === "undefined") return;
  const dates = rows.map((r) => r.date || r.Date);
  const eq = rows.map((r) => parseFloat(r.equity || r.Equity || 0));
  if (window._equityChart) window._equityChart.destroy();
  window._equityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels: dates,
      datasets: [{ label: "Equity", data: eq, borderColor: "#3b82f6", tension: 0.1 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8b9cb3" } } },
      scales: {
        x: { ticks: { color: "#8b9cb3", maxTicksLimit: 12 }, grid: { color: "#2d3a4d" } },
        y: { ticks: { color: "#8b9cb3" }, grid: { color: "#2d3a4d" } },
      },
    },
  });
}

async function runBacktest() {
  el("backtest-status").textContent = "Starting…";
  await post("/api/backtest/run", {});
  const poll = setInterval(async () => {
    const s = await get("/api/backtest/status");
    el("backtest-status").textContent = s.running ? "Running…" : "Done: " + (s.last || "");
    if (!s.running) {
      clearInterval(poll);
      await loadDailyPositions();
      await loadTradesEquity();
    }
  }, 2000);
}

function exportDailyPositions() {
  window.open(API + "/api/export/daily_positions.csv", "_blank");
}

document.addEventListener("DOMContentLoaded", () => {
  loadUniverse().catch(console.error);
  loadConfig().catch(console.error);
  computeRanking().catch(console.error);
  loadDailyPositions().catch(() => {});
  loadTradesEquity().catch(() => {});

  el("btn-save-universe").addEventListener("click", () => saveUniverse().catch(alert));
  el("file-upload").addEventListener("change", uploadTickers);
  el("btn-ranking").addEventListener("click", () => computeRanking().catch(alert));
  el("btn-backtest").addEventListener("click", () => runBacktest().catch(alert));
  el("btn-refresh-positions").addEventListener("click", () => loadDailyPositions().catch(alert));
  el("btn-export").addEventListener("click", exportDailyPositions);
});
