const $ = (id) => document.getElementById(id);
const API = "";

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c7d4ec", family: "-apple-system,Segoe UI,Roboto,sans-serif", size: 12 },
  margin: { l: 54, r: 20, t: 40, b: 46 },
  xaxis: { gridcolor: "#1c2947", zerolinecolor: "#22304f" },
  yaxis: { gridcolor: "#1c2947", zerolinecolor: "#22304f" },
  legend: { orientation: "h", y: -0.22 },
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

async function loadPresets() {
  try {
    const r = await fetch(`${API}/api/presets`);
    const d = await r.json();
    const dl = $("preset-list");
    Object.values(d.groups).flat().forEach((name) => {
      const o = document.createElement("option"); o.value = name; dl.appendChild(o);
    });
    window.EXAMPLES = d.example_conditions || {};
  } catch (e) { /* non-fatal */ }
}

async function loadHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    const d = await r.json();
    const b = $("source-badge");
    if (d.data_source === "bloomberg") { b.textContent = "● Bloomberg live"; b.className = "badge live"; }
    else { b.textContent = "● Synthetic surface (no Terminal)"; b.className = "badge mock"; }
  } catch (e) {
    $("source-badge").textContent = "● backend offline"; $("source-badge").className = "badge mock";
  }
}

async function refreshChain() {
  const und = $("underlying").value.trim();
  if (!und) return;
  try {
    const r = await fetch(`${API}/api/chain?underlying=${encodeURIComponent(und)}`);
    if (!r.ok) return;
    const d = await r.json();
    const sel = $("expiry");
    sel.innerHTML = '<option value="">Nearest to target date</option>';
    d.expiries.forEach((e) => {
      const o = document.createElement("option");
      o.value = e.expiry;
      o.textContent = `${e.expiry}  (T=${e.T.toFixed(2)}y, F=${fmtNum(e.forward)}, ${e.n_strikes} strikes)`;
      sel.appendChild(o);
    });
    // Auto-tick the percent box for rates
    $("force_pct").checked = d.asset_class === "RATES";
    const ex = (window.EXAMPLES || {})[und];
    $("hint").innerHTML = `Asset class <b>${d.asset_class}</b> · ${d.expiries.length} expiries loaded`
      + (ex ? ` · try condition: <b>${ex}</b>` : "");
  } catch (e) { /* ignore */ }
}

function fmtNum(x) {
  if (x == null || isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 10) return x.toFixed(2);
  return x.toFixed(4);
}

async function run() {
  const btn = $("run"); btn.disabled = true; btn.textContent = "Computing…";
  $("error").classList.add("hidden");
  const body = {
    underlying: $("underlying").value.trim(),
    condition: $("condition").value.trim(),
    r: parseFloat($("rate").value || "0"),
    force_percent: $("force_pct").checked,
  };
  const betaV = $("beta").value.trim();
  if (betaV !== "") body.beta = parseFloat(betaV);
  const expV = $("expiry").value;
  if (expV) body.expiry = expV;

  try {
    const r = await fetch(`${API}/api/distribution`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Request failed");
    render(d);
    loadPositioning(body.underlying, body.condition, d);
  } catch (e) {
    $("error").textContent = "Error: " + e.message;
    $("error").classList.remove("hidden");
  } finally {
    btn.disabled = false; btn.textContent = "Compute probability";
  }
}

function render(d) {
  $("result").classList.remove("hidden");
  const pct = (d.probability * 100).toFixed(1) + "%";
  $("prob").textContent = pct;
  $("prob-cond").textContent = `P( ${d.underlying} ${d.condition} )`;
  $("odds").textContent = `Complement ${(d.complement*100).toFixed(1)}% · implied odds ${d.odds}`;

  $("m-und").textContent = d.underlying;
  $("m-ac").textContent = d.asset_class;
  $("m-exp").textContent = d.expiry;
  $("m-t").textContent = d.T.toFixed(3);
  $("m-fwd").textContent = fmtNum(d.forward);
  $("m-src").textContent = d.source === "bloomberg" ? "Bloomberg" : "Synthetic";

  const unit = d.is_percent ? "%" : "";
  const thrLine = (v, name, color) => ({
    type: "line", x0: v, x1: v, yref: "paper", y0: 0, y1: 1,
    line: { color, width: 1.6, dash: "dash" },
  });
  const shapes = [];
  const annos = [];
  if (d.direction === "between") {
    shapes.push(thrLine(d.threshold, "lo", "#ffb020"), thrLine(d.threshold_hi, "hi", "#ffb020"));
  } else {
    shapes.push(thrLine(d.threshold, "thr", "#ffb020"));
  }
  shapes.push({ type: "line", x0: d.forward, x1: d.forward, yref: "paper", y0: 0, y1: 1,
    line: { color: "#4da3ff", width: 1.2, dash: "dot" } });

  // --- PDF with shaded event region ---
  const g = d.grid, pdf = d.pdf;
  let mask;
  if (d.direction === "above") mask = g.map((x) => x >= d.threshold);
  else if (d.direction === "below") mask = g.map((x) => x <= d.threshold);
  else mask = g.map((x) => x >= d.threshold && x <= d.threshold_hi);
  const fx = [], fy = [];
  g.forEach((x, i) => { if (mask[i]) { fx.push(x); fy.push(pdf[i]); } });

  Plotly.newPlot("chart-dist", [
    { x: g, y: pdf, type: "scatter", mode: "lines", name: "Risk-neutral PDF",
      line: { color: "#4da3ff", width: 2 } },
    { x: fx, y: fy, type: "scatter", mode: "lines", name: "Event region",
      fill: "tozeroy", line: { color: "#38d39f", width: 0 },
      fillcolor: "rgba(56,211,159,.35)" },
  ], { ...PLOT_LAYOUT, title: "Risk-neutral density (PDF)", shapes,
       xaxis: { ...PLOT_LAYOUT.xaxis, title: `Level${unit ? " ("+unit+")" : ""}` },
       yaxis: { ...PLOT_LAYOUT.yaxis, title: "Density" } }, PLOT_CONFIG);

  // --- CDF ---
  Plotly.newPlot("chart-cdf", [
    { x: g, y: d.cdf, type: "scatter", mode: "lines", name: "CDF",
      line: { color: "#38d39f", width: 2 } },
  ], { ...PLOT_LAYOUT, title: "Cumulative distribution (CDF)", shapes,
       xaxis: { ...PLOT_LAYOUT.xaxis, title: `Level${unit ? " ("+unit+")" : ""}` },
       yaxis: { ...PLOT_LAYOUT.yaxis, title: "P(S ≤ x)", range: [0, 1] } }, PLOT_CONFIG);

  // --- Smile: market vs fitted ---
  const sk = d.smile.map((s) => s.strike);
  Plotly.newPlot("chart-smile", [
    { x: sk, y: d.smile.map((s) => s.market_vol * 100), mode: "markers", name: "Market IV",
      marker: { color: "#ffb020", size: 7 } },
    { x: sk, y: d.smile.map((s) => s.fitted_vol * 100), mode: "lines", name: "SABR fit",
      line: { color: "#4da3ff", width: 2 } },
  ], { ...PLOT_LAYOUT, title: "Volatility smile — market vs SABR fit",
       xaxis: { ...PLOT_LAYOUT.xaxis, title: "Strike" },
       yaxis: { ...PLOT_LAYOUT.yaxis, title: "Implied vol (%)" } }, PLOT_CONFIG);

  // --- Stats tables ---
  const s = d.stats;
  const rows = [
    ["Forward", fmtNum(s.forward)], ["Mean", fmtNum(s.mean)],
    ["Mode", fmtNum(s.mode)], ["Median", fmtNum(s.median)],
    ["Std dev", fmtNum(s.std)],
    ["5th pctile", fmtNum(s.p05)], ["25th pctile", fmtNum(s.p25)],
    ["75th pctile", fmtNum(s.p75)], ["95th pctile", fmtNum(s.p95)],
  ];
  $("stats-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  const sb = d.sabr;
  $("sabr-table").innerHTML = [
    ["α (alpha)", sb.alpha.toFixed(4)], ["β (beta)", sb.beta.toFixed(2)],
    ["ρ (rho)", sb.rho.toFixed(3)], ["ν (nu / vol-of-vol)", sb.nu.toFixed(3)],
    ["Displacement", sb.shift.toFixed(2)], ["Fit RMSE (vol pts)", (sb.rmse*100).toFixed(3)],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

// ================= Positioning / conviction =================
const CONV_COLOR = { high: "#38d39f", moderate: "#4da3ff", conflicting: "#ff5c7a",
                     low: "#5b6577", na: "#3a4255" };
const CONV_LABEL = { high: "High", moderate: "Moderate", conflicting: "Conflicting",
                     low: "Low", na: "n/a" };

async function loadPositioning(underlying, condition, dist) {
  const sec = $("pos-section");
  try {
    const url = `${API}/api/positioning?underlying=${encodeURIComponent(underlying)}`
              + `&condition=${encodeURIComponent(condition)}`;
    const r = await fetch(url);
    const p = await r.json();
    if (!r.ok) throw new Error(p.detail || "positioning failed");
    renderPositioning(p, dist);
    sec.classList.remove("hidden");
  } catch (e) {
    // positioning is supplementary; never block the main result
    sec.classList.add("hidden");
    console.warn("positioning unavailable:", e.message);
  }
}

function renderPositioning(p, dist) {
  const strikes = (p.strikes || []).slice().sort((a, b) => a.strike - b.strike);
  const daysMsg = p.deltas_available
    ? `${p.history_days} day(s) of snapshots · z-scored context active`
    : `${p.history_days || 0} day(s) of snapshots · building history — run /api/backfill to seed`;
  $("pos-status").textContent = `Expiry ${p.expiry} · ${daysMsg}`;

  // ---- Chart 1: OI by strike, colored by conviction, PDF overlaid ----
  const calls = strikes.filter((s) => s.call_put === "C");
  const puts  = strikes.filter((s) => s.call_put === "P");
  const barTrace = (arr, name) => ({
    x: arr.map((s) => s.strike), y: arr.map((s) => s.oi || 0),
    type: "bar", name,
    marker: { color: arr.map((s) => CONV_COLOR[s.composite] || CONV_COLOR.na) },
    customdata: arr.map((s) => [CONV_LABEL[s.composite] || "n/a", (s.magnitude ?? 0).toFixed(2),
                               s.n_agree ?? 0]),
    hovertemplate: "Strike %{x}<br>OI %{y:,.0f}<br>Conviction %{customdata[0]}"
                 + "<br>Magnitude %{customdata[1]} · agree %{customdata[2]}/3<extra>" + name + "</extra>",
  });
  const oiTraces = [];
  if (calls.length) oiTraces.push(barTrace(calls, "Calls OI"));
  if (puts.length)  oiTraces.push(barTrace(puts, "Puts OI"));
  if (dist && dist.grid && dist.pdf) {
    oiTraces.push({ x: dist.grid, y: dist.pdf, type: "scatter", mode: "lines",
      name: "RND (PDF)", yaxis: "y2", line: { color: "#c9d3e3", width: 1.6, dash: "dot" } });
  }
  const shapes = [];
  if (dist) shapes.push({ type: "line", x0: dist.threshold, x1: dist.threshold,
    yref: "paper", y0: 0, y1: 1, line: { color: "#ffb020", width: 1.6, dash: "dash" } });
  Plotly.newPlot("chart-oi", oiTraces, {
    ...PLOT_LAYOUT, title: "Open interest by strike (colored by conviction)",
    barmode: "group", shapes,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Strike" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Open interest" },
    yaxis2: { title: "Density", overlaying: "y", side: "right", showgrid: false,
              zeroline: false, tickfont: { color: "#8b95a7" }, titlefont: { color: "#8b95a7" } },
  }, PLOT_CONFIG);

  // ---- Chart 2: signed magnitude bars (direction × magnitude) ----
  const scored = strikes.filter((s) => s.composite !== "na");
  const signed = scored.map((s) => (s.direction || 0) * (s.magnitude || 0));
  Plotly.newPlot("chart-conv", [{
    x: scored.map((s) => `${s.strike}${s.call_put}`), y: signed, type: "bar",
    marker: { color: scored.map((s) => CONV_COLOR[s.composite] || CONV_COLOR.na) },
    customdata: scored.map((s) => [CONV_LABEL[s.composite], s.z_oi, s.z_voloi, s.z_iv]
      .map((v) => typeof v === "number" ? v.toFixed(2) : v)),
    hovertemplate: "%{x}<br>Signed magnitude %{y:.2f}<br>Read %{customdata[0]}"
      + "<br>zΔOI %{customdata[1]} · z vol/OI %{customdata[2]} · zΔIV %{customdata[3]}<extra></extra>",
  }], {
    ...PLOT_LAYOUT, title: "Conviction magnitude (↑ bullish build · ↓ bearish build)",
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "Strike" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "Direction × magnitude" },
  }, PLOT_CONFIG);

  // ---- Summary table ----
  const sm = p.summary || {};
  const pc = sm.put_call_oi_ratio;
  const rows = [
    ["Put / call OI ratio", pc == null ? "—" : pc.toFixed(2)],
    ["OI center of gravity", sm.oi_center_of_gravity == null ? "—" : fmtNum(sm.oi_center_of_gravity)],
    ["Max pain", sm.max_pain == null ? "—" : fmtNum(sm.max_pain)],
    ["Total premium notional", sm.total_premium_notional == null ? "—"
        : "$" + (sm.total_premium_notional / 1e9).toFixed(2) + "bn"],
    ["Call OI / Put OI", (sm.total_call_oi == null && sm.total_put_oi == null) ? "—"
        : `${fmtNum(sm.total_call_oi || 0)} / ${fmtNum(sm.total_put_oi || 0)}`],
  ];
  $("pos-summary").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  // ---- Top conviction table ----
  const top = (sm.top_conviction || []);
  if (top.length) {
    const head = `<tr><th>Strike</th><th>Read</th><th>Mag</th><th>Agree</th><th>Dir</th></tr>`;
    const body = top.map((t) => {
      const dir = t.direction > 0 ? "▲" : t.direction < 0 ? "▼" : "·";
      const chip = `<span class="chip" style="background:${CONV_COLOR[t.composite]}22;`
                 + `color:${CONV_COLOR[t.composite]}">${CONV_LABEL[t.composite]}</span>`;
      return `<tr><td>${t.strike}${t.call_put}</td><td>${chip}</td>`
           + `<td>${(t.magnitude ?? 0).toFixed(2)}</td><td>${t.n_agree ?? 0}/3</td><td>${dir}</td></tr>`;
    }).join("");
    $("pos-top").innerHTML = head + body;
  } else {
    $("pos-top").innerHTML = `<tr><td>No scored strikes yet — accumulate history or run backfill.</td></tr>`;
  }
}

$("run").addEventListener("click", run);
$("underlying").addEventListener("change", refreshChain);
$("condition").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

(async function init() {
  await loadPresets();
  await loadHealth();
  await refreshChain();
})();
