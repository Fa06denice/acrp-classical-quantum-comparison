"use strict";

// ---- tiny API helpers -----------------------------------------------------
async function _parse(r) {
  let detail = r.statusText;
  let body = null;
  try { body = await r.json(); detail = body.detail ?? detail; } catch { /* non-JSON */ }
  if (!r.ok) {
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = r.status;
    throw err;
  }
  return body;
}
const api = {
  async get(path, signal) { return _parse(await fetch(path, { signal })); },
  async post(path, body, signal) {
    return _parse(await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }));
  },
  async del(path, signal) { return _parse(await fetch(path, { method: "DELETE", signal })); },
};

// Abortable delay used while polling a job (rejects with AbortError on abort).
function _sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal && signal.aborted) { reject(new DOMException("aborted", "AbortError")); return; }
    const t = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener("abort", () => {
        clearTimeout(t);
        reject(new DOMException("aborted", "AbortError"));
      }, { once: true });
    }
  });
}

// Collapse a burst of calls (e.g. dragging a slider) into a single trailing run.
function _debounce(fn, ms) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => { t = null; fn(...args); }, ms);
  };
}
// Slider drags fire many input events; only the settled value needs a round-trip.
const debouncedPreview = _debounce(() => refreshPreview().catch(showError), 180);

// classify an API error into (kind, advice) for a helpful, non-blocking banner
function classifyError(e) {
  const s = e.status || 0;
  const m = (e.message || "").toLowerCase();
  if (s === 503 || m.includes("pip install")) return ["dependency", "Install the required extra, or pick another solver."];
  if (m.includes("qubit") || m.includes("discrete variables") || m.includes("memory") || m.includes("budget"))
    return ["capacity", "Reduce heading levels / aircraft, or raise the budget."];
  if (s === 422) return ["validation", "Check the parameters and try again."];
  if (m.includes("account") || m.includes("token") || m.includes("qpu") || m.includes("backend"))
    return ["backend", "Real-hardware backend unavailable — use the local simulator."];
  if (s === 0 || m.includes("failed to fetch")) return ["offline", "Backend unreachable — is the dashboard server running?"];
  return ["error", "Try again, or return to a safe configuration."];
}

const $ = (id) => document.getElementById(id);
let SCENARIOS = Object.create(null);
let CURRENT = null; // last radar payload
let LAST_QUBO = null;
let LAST_QUBO_CID = -1; // CONFIG_ID the cached QUBO belongs to
let LAST_PREFLIGHT = null; // last /api/preflight applicability report
let LAST_PREFLIGHT_CID = -1; // CONFIG_ID the preflight belongs to
// Monotonic id of the *current* configuration. Every mutation bumps it; any
// derived datum (preflight, QUBO, comparison) is tagged with the id it was
// computed for, so a result from an older configuration can never be shown or
// exported as if it belonged to the new one.
let CONFIG_ID = 0;
let COMPARE_SNAPSHOT = null; // immutable config the current AUTO_COMPARE ran on
// Immutable export snapshot: frozen (deep-copied) at the moment the comparison
// completes, tagged with its config_id. Export always writes THIS object, so the
// file can never mix a new configuration with old results.
let EXPORT_SNAPSHOT = null;

// ---- tabs -----------------------------------------------------------------
document.querySelectorAll(".tab[data-tab]").forEach((btn) => {
  btn.onclick = () => switchTab(btn.dataset.tab);
});

// One implementation, two information depths. Demo mode is deliberately the
// default on every load so a defense starts with the scientific conclusion,
// while Expert mode reveals all controls, QUBO internals and audit evidence.
function setExperienceMode(expert) {
  document.body.classList.toggle("expert-mode", expert);
  const toggle = $("experience-toggle");
  toggle.setAttribute("aria-pressed", String(expert));
  toggle.textContent = expert ? "VUE DÉMO ▴" : "DÉTAILS SCIENTIFIQUES ▾";
  toggle.title = expert
    ? "revenir à la vue démo minimale"
    : "révéler Explain, les détails scientifiques et le pipeline IBM QPU";
  if (!expert) {
    // every disclosed tab is expert-only; leaving expert mode returns to radar
    const active = document.querySelector(".tab-panel.active");
    if (active && active.id !== "tab-radar") switchTab("radar");
  }
}

// ---- radar simulator ------------------------------------------------------
// The radar animates the model's LINEAR kinematics: position(t) = p0 + v·t.
// It is not a physical/ATC simulation. All geometry (CPA, conflicts) comes from
// the API payload; JavaScript only renders it.
const SIM = {
  payload: null,
  t: 0,
  tMax: 1,
  playing: false,
  speed: 1,
  lastTs: null,
  rafId: null,
  highlightPair: null, // "i-j" to emphasise, or null
  highlightAircraft: null, // aircraft id to ring, or null
  toggles: { motion: "discrete", initial: true, continuous: true, discrete: true,
    labels: true, vectors: true, sep: true, cpa: true, trails: true },
};

function drawRadar(payload) {
  CURRENT = payload;
  SIM.payload = payload;
  SIM.t = 0;
  SIM.tMax = payload.suggested_horizon && payload.suggested_horizon > 0 ? payload.suggested_horizon : 1;
  SIM.highlightPair = null;
  pauseSim();
  syncTimelineUI();
  renderRadarAt(0);
}

function _hidpi(cv) {
  // crisp canvas on retina: back the logical size with devicePixelRatio pixels
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || cv.width;
  const cssH = cv.clientHeight || cv.height;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: cssW, H: cssH };
}

// position of an aircraft at model time t, for initial (vx0/vy0) or resolved (vx/vy)
function _posAt(a, t, resolved) {
  const vx = resolved ? a.vx : a.vx0;
  const vy = resolved ? a.vy : a.vy0;
  return [a.x + vx * t, a.y + vy * t];
}

function _comparisonTrajectoryLayers(p) {
  const ref = AUTO_COMPARE.find((r) => r.solver === "reference" && r.state === "ok");
  const discreteById = Object.fromEntries(
    (ref?.payload?.aircraft || []).map((a) => [a.id, a])
  );
  const controls = _gurobiLeg()?.continuous_controls;
  const q = controls?.q, theta = controls?.theta;
  return p.aircraft.map((a, idx) => {
    let continuous = null;
    if (Array.isArray(q) && Array.isArray(theta) && q.length === p.aircraft.length
        && theta.length === p.aircraft.length) {
      const heading = a.heading_initial_deg * Math.PI / 180 + Number(theta[idx]);
      const speed = a.speed_initial * Number(q[idx]);
      continuous = {...a, vx: speed * Math.cos(heading), vy: speed * Math.sin(heading)};
    }
    return {initial: a, continuous, discrete: discreteById[a.id] || null};
  });
}

function renderRadarAt(t) {
  const p = SIM.payload;
  if (!p) return;
  const cv = $("radar");
  const { ctx, W, H } = _hidpi(cv);
  const cx = W / 2, cy = H / 2;
  ctx.clearRect(0, 0, W, H);

  // world extent: cover start + end-of-horizon positions of all aircraft
  let ext = p.radius || 2;
  const tog = SIM.toggles;
  const trajectoryLayers = _comparisonTrajectoryLayers(p);
  const movingLayer = (layers) => {
    const requested = layers[tog.motion];
    if (requested) return {aircraft: requested, resolved: tog.motion !== "initial"};
    // Before comparison results arrive, keep the animation usable and honest.
    return {aircraft: layers.initial, resolved: false};
  };
  const movingById = Object.fromEntries(
    trajectoryLayers.map((layers) => [layers.initial.id, movingLayer(layers)])
  );
  trajectoryLayers.forEach((layers) => {
    for (const a of [layers.initial, layers.continuous, layers.discrete].filter(Boolean)) {
      const [ex, ey] = _posAt(a, SIM.tMax, a !== layers.initial);
      ext = Math.max(ext, Math.abs(a.x), Math.abs(a.y), Math.abs(ex), Math.abs(ey));
    }
  });
  ext *= 1.15;
  const scale = (Math.min(W, H) / 2 - 40) / ext;
  const X = (x) => cx + x * scale;
  const Y = (y) => cy - y * scale;
  const showRes = true; // blips/CPA keep using the currently selected solver payload

  // range rings + scale + crosshair
  ctx.strokeStyle = "#15463a"; ctx.lineWidth = 1;
  ctx.font = "10px monospace"; ctx.fillStyle = "#1d8f63";
  for (let r = 1; r <= Math.ceil(ext); r++) {
    ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(cx, 14); ctx.lineTo(cx, H - 14);
  ctx.moveTo(14, cy); ctx.lineTo(W - 14, cy); ctx.stroke();
  // scale bar (1 world unit) + time readout
  ctx.fillStyle = "#6fa892";
  ctx.fillText("1 unit", cx + scale + 4, cy - 4);
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + scale, cy);
  ctx.strokeStyle = "#6fa892"; ctx.stroke();
  ctx.fillStyle = "#36f2a0";
  ctx.fillText(`t = ${t.toFixed(3)} / ${SIM.tMax.toFixed(3)} ${p.time_unit || ""}`, 14, 18);

  const byId = Object.fromEntries(p.aircraft.map((a) => [a.id, a]));

  // trajectories (initial dashed, resolved solid; traveled vs predicted split at t)
  function trajectory(a, resolved, faint, strong, dashed = false) {
    const [sx, sy] = _posAt(a, 0, resolved);
    const [nx, ny] = _posAt(a, t, resolved);
    const [ex, ey] = _posAt(a, SIM.tMax, resolved);
    // predicted (t..tMax)
    if (tog.trails) {
      ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.strokeStyle = faint;
      ctx.beginPath(); ctx.moveTo(X(nx), Y(ny)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
      // traveled (0..t) brighter
      ctx.strokeStyle = strong;
      ctx.beginPath(); ctx.moveTo(X(sx), Y(sy)); ctx.lineTo(X(nx), Y(ny)); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  trajectoryLayers.forEach((layers) => {
    if (tog.initial)
      trajectory(layers.initial, false, "rgba(255,77,94,0.35)", "rgba(255,77,94,0.9)", true);
    if (tog.continuous && layers.continuous)
      trajectory(layers.continuous, true, "rgba(54,242,160,0.35)", "rgba(54,242,160,0.95)");
    if (tog.discrete && layers.discrete)
      trajectory(layers.discrete, true, "rgba(255,204,77,0.35)", "rgba(255,204,77,0.95)");
  });

  // which aircraft are involved in a conflict that is ACTIVE at time t
  // (min-sep over the pair happens near t_cpa; we flag "active now" if the two
  // are within d at the current t, and "predicted" if resolved_conflict holds).
  const activeNow = new Set(), predicted = new Set();
  (p.pairs || []).forEach((pr) => {
    if (pr.excluded) return;
    if (pr.resolved_conflict) { predicted.add(pr.i); predicted.add(pr.j); }
    const ma = movingById[pr.i], mb = movingById[pr.j];
    if (!ma || !mb) return;
    const [ax, ay] = _posAt(ma.aircraft, t, ma.resolved);
    const [bx, by] = _posAt(mb.aircraft, t, mb.resolved);
    const sep = Math.hypot(ax - bx, ay - by);
    if (sep < p.d) { activeNow.add(pr.i); activeNow.add(pr.j); }
  });

  // CPA markers + links (resolved trajectories)
  if (tog.cpa) {
    (p.pairs || []).forEach((pr) => {
      if (pr.excluded || !pr.closing) return;
      const hl = SIM.highlightPair === `${pr.i}-${pr.j}`;
      const margin = pr.min_separation - pr.required;
      const col = pr.resolved_conflict ? "#ff4d5e" : (margin < pr.required ? "#ffcc4d" : "#36f2a0");
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = hl ? 2.5 : 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(X(pr.xi_cpa), Y(pr.yi_cpa)); ctx.lineTo(X(pr.xj_cpa), Y(pr.yj_cpa)); ctx.stroke();
      ctx.setLineDash([]);
      for (const [mx, my] of [[pr.xi_cpa, pr.yi_cpa], [pr.xj_cpa, pr.yj_cpa]]) {
        ctx.beginPath(); ctx.arc(X(mx), Y(my), hl ? 4 : 3, 0, 2 * Math.PI); ctx.fill();
      }
      if (hl) {
        ctx.fillStyle = "#cdeee0";
        ctx.fillText(`CPA ${pr.min_separation.toFixed(3)} (t=${pr.t_cpa.toFixed(2)})`,
          X((pr.xi_cpa + pr.xj_cpa) / 2) + 6, Y((pr.yi_cpa + pr.yj_cpa) / 2));
      }
    });
    ctx.lineWidth = 1;
  }

  // separation zones (radius d/2 around current position) + velocity vectors + blips
  p.aircraft.forEach((a) => {
    const moving = movingById[a.id] || {aircraft: a, resolved: showRes};
    const shown = moving.aircraft;
    const [px, py] = _posAt(shown, t, moving.resolved);
    if (tog.sep) {
      ctx.strokeStyle = "rgba(111,168,146,0.45)";
      ctx.beginPath(); ctx.arc(X(px), Y(py), Math.max((p.d / 2) * scale, 2), 0, 2 * Math.PI); ctx.stroke();
    }
    if (tog.vectors) {
      const vx = moving.resolved ? shown.vx : shown.vx0;
      const vy = moving.resolved ? shown.vy : shown.vy0;
      const L = 0.12;
      const ex = px + vx * L, ey = py + vy * L;
      ctx.strokeStyle = "#1d8f63";
      ctx.beginPath(); ctx.moveTo(X(px), Y(py)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
      const ang = Math.atan2(Y(ey) - Y(py), X(ex) - X(px));
      ctx.beginPath(); ctx.moveTo(X(ex), Y(ey));
      ctx.lineTo(X(ex) - 7 * Math.cos(ang - 0.4), Y(ey) - 7 * Math.sin(ang - 0.4));
      ctx.lineTo(X(ex) - 7 * Math.cos(ang + 0.4), Y(ey) - 7 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fillStyle = "#1d8f63"; ctx.fill();
    }
    const hot = activeNow.has(a.id);
    const warn = !hot && predicted.has(a.id);
    ctx.fillStyle = hot ? "#ff4d5e" : warn ? "#ffcc4d" : "#36f2a0";
    ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = hot ? 12 : 8;
    ctx.beginPath(); ctx.arc(X(px), Y(py), 5, 0, 2 * Math.PI); ctx.fill();
    ctx.shadowBlur = 0;
    if (SIM.highlightAircraft === a.id) {
      ctx.strokeStyle = "#4dd2ff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(px), Y(py), 11, 0, 2 * Math.PI); ctx.stroke();
      ctx.lineWidth = 1;
    }
    if (tog.labels) {
      ctx.fillStyle = "#cdeee0"; ctx.font = "11px monospace";
      const lvl = a.level != null ? ` FL${a.level}` : "";
      ctx.fillText(`#${a.id}${lvl}`, X(px) + 8, Y(py) - 8);
    }
  });
}

// ---- timeline engine ------------------------------------------------------
function _tick(ts) {
  if (!SIM.playing) return;
  if (SIM.lastTs == null) SIM.lastTs = ts;
  const dt = (ts - SIM.lastTs) / 1000; // seconds, framerate-independent
  SIM.lastTs = ts;
  // 1 real second == (tMax/4) model units at ×1, so a full sweep ≈ 4 s at ×1
  SIM.t += dt * SIM.speed * (SIM.tMax / 4);
  if (SIM.t >= SIM.tMax) { SIM.t = SIM.tMax; pauseSim(); }
  syncTimelineUI();
  renderRadarAt(SIM.t);
  if (SIM.playing) SIM.rafId = requestAnimationFrame(_tick);
}
function playSim() {
  if (!SIM.payload || SIM.playing) return;
  if (SIM.t >= SIM.tMax) SIM.t = 0;
  SIM.playing = true; SIM.lastTs = null;
  const b = $("sim-play"); if (b) b.textContent = "⏸";
  SIM.rafId = requestAnimationFrame(_tick);
}
function pauseSim() {
  SIM.playing = false;
  if (SIM.rafId) cancelAnimationFrame(SIM.rafId);
  SIM.rafId = null;
  const b = $("sim-play"); if (b) b.textContent = "▶";
}
function resetSim() { pauseSim(); SIM.t = 0; syncTimelineUI(); renderRadarAt(0); }
function seekSim(t) {
  SIM.t = Math.max(0, Math.min(SIM.tMax, t));
  syncTimelineUI(); renderRadarAt(SIM.t);
}
function stepSim(dir) { pauseSim(); seekSim(SIM.t + dir * SIM.tMax / 40); }
function syncTimelineUI() {
  const sl = $("sim-slider");
  if (sl) { sl.max = "1000"; sl.value = String(Math.round((SIM.t / SIM.tMax) * 1000)); }
  const tt = $("sim-time");
  if (tt) tt.textContent = `t=${SIM.t.toFixed(3)} / ${SIM.tMax.toFixed(3)}`;
}

// ---- metrics panel --------------------------------------------------------
function setMetrics(p) {
  $("m-name").textContent = p.name;
  $("m-n").textContent = p.n;
  const confBox = $("m-conf-box");
  $("m-conf").textContent = p.n_conflicts;
  confBox.classList.toggle("ok", p.n_conflicts === 0);
  confBox.classList.toggle("bad", p.n_conflicts > 0);

  const s = p.solution;
  $("explain-run-btn").classList.toggle("hidden", !s?.explanation);
  if (s) {
    $("m-resolved").textContent = `${s.n_resolved} / ${s.n_initial_conflicts}`;
    $("m-obj").textContent = s.objective != null ? s.objective.toFixed(5) : "—";
    $("m-solver").textContent = s.solver;
    $("m-time").textContent = s.wall_time_s != null ? `${(s.wall_time_s * 1000).toFixed(0)} ms` : "—";
    const q = $("q-metrics");
    if (s.n_qubits != null) {
      q.classList.remove("hidden");
      $("m-backend").textContent = s.backend || (s.qubo_energy != null ? "classical QUBO" : "—");
      $("m-qubits").textContent = s.n_qubits;
      // circuit metrics only exist for QAOA; QUBO solvers show energy instead
      $("m-depth").textContent = s.circuit_depth != null ? s.circuit_depth : "—";
      $("m-2q").textContent =
        s.qubo_energy != null ? `H=${s.qubo_energy.toFixed(4)}` : (s.two_qubit_gates ?? "—");
    } else {
      q.classList.add("hidden");
    }
  } else {
    ["m-resolved", "m-obj", "m-solver", "m-profile", "m-time"].forEach((id) => ($(id).textContent = "—"));
    $("q-metrics").classList.add("hidden");
    const cb = $("compare-box"); if (cb) cb.classList.add("hidden");
    const rb = $("repro-box"); if (rb) rb.classList.add("hidden");
    SIM.highlightAircraft = null;
  }
}

// ---- load / solve ---------------------------------------------------------
// ---- single source of truth for the configured problem -------------------
const currentConfiguration = {
  instance: null, scenario: "", subset: null, n_theta: 3, n_q: 1,
};
let INSTANCE_CATALOG_REQUEST = 0;
let GRID_LIMIT_REQUEST = 0;

async function refreshGridLimit() {
  const requestId = ++GRID_LIMIT_REQUEST;
  const instance = currentConfiguration.instance;
  const nQ = currentConfiguration.n_q;
  if (!instance) return;
  const limit = await api.get(
    `/api/instances/${encodeURIComponent(instance)}/grid-limit?n_q=${encodeURIComponent(nQ)}`
  );
  if (requestId !== GRID_LIMIT_REQUEST || instance !== currentConfiguration.instance
      || nQ !== currentConfiguration.n_q) return;
  const maxK = limit.slider_max_n_theta;
  for (const id of ["ntheta", "qb-ntheta"]) {
    const slider = $(id);
    if (slider) slider.max = String(maxK);
  }
  const selectedQubits = limit.n_aircraft * currentConfiguration.n_theta * nQ;
  const longRunWarning = selectedQubits > limit.demo_qaoa_max_qubits
    ? ` · warning: ${selectedQubits}q QAOA may exceed the 120 s demo timeout`
    : "";
  $("k-limit-note").textContent = limit.all_methods_at_slider_max
    ? `recommended: K3 · technical max: K${maxK}${longRunWarning}`
    : `recommended: K3 · no grid runs every method (${limit.blocked_at_k3.join(", ")} skipped)`;
  if (currentConfiguration.n_theta > maxK) {
    currentConfiguration.n_theta = maxK;
    invalidateDerivedState();
    _syncControlsFromConfig();
    debouncedInstanceCatalog();
    refreshPreview().catch(showError);
  }
}

function _catalogOption(record) {
  const option = new Option(`${record.name} · ${record.n_qubits}q`, record.name);
  option.title = record.reason || "all automatic local algorithms applicable";
  return option;
}

async function refreshInstanceCatalog() {
  const requestId = ++INSTANCE_CATALOG_REQUEST;
  const { n_theta, n_q, instance } = _configSnapshot();
  const catalog = await api.get(
    `/api/instances/catalog?n_theta=${encodeURIComponent(n_theta)}&n_q=${encodeURIComponent(n_q)}`
  );
  if (requestId !== INSTANCE_CATALOG_REQUEST || n_theta !== currentConfiguration.n_theta
      || n_q !== currentConfiguration.n_q) return;

  const runnable = document.createElement("optgroup");
  runnable.label = `ALL ALGORITHMS (${catalog.counts.all_applicable})`;
  runnable.append(...catalog.all_applicable.map(_catalogOption));
  const limited = document.createElement("optgroup");
  limited.label = `LIMITED / SOME SKIPPED (${catalog.counts.limited})`;
  limited.append(...catalog.limited.map(_catalogOption));
  $("instance").replaceChildren(runnable, limited);
  if ([...$("instance").options].some((o) => o.value === instance)) $("instance").value = instance;
  $("instance-classification-note").textContent =
    `Full instances at K=${catalog.grid_k}: ${catalog.counts.all_applicable} run every local route; ` +
    `${catalog.counts.limited} skip at least one. Hover an instance for the reason.`;
}
const debouncedInstanceCatalog = _debounce(
  () => refreshInstanceCatalog().catch((e) => {
    $("instance-classification-note").textContent = `Classification indisponible : ${e.message}`;
  }),
  180,
);

// Immutable snapshot of the current configuration, tagged with CONFIG_ID.
function _configSnapshot() {
  const c = currentConfiguration;
  return {
    id: CONFIG_ID,
    instance: c.instance,
    subset: c.subset ? [...c.subset] : null,
    n_theta: c.n_theta, n_q: c.n_q, w: c.w ?? null,
  };
}

// Call at the moment a configuration mutates (before any network round-trip):
// bump the id, cancel the previous comparison immediately (real backend DELETE),
// drop every derived datum so nothing stale can be shown or exported, and tell
// the user results are recomputing rather than leaving the old numbers on screen.
function invalidateDerivedState() {
  CONFIG_ID++;
  _supersedeAutoRun();
  AUTO_COMPARE = [];
  COMPARE_SNAPSHOT = null;
  EXPORT_SNAPSHOT = null;          // a stale export snapshot must never survive
  updateExportButtons();           // export disabled until the new runs settle
  renderDemoExplanation();         // the explanation panel clears immediately
  const idl0 = $("demo-identity");
  if (idl0) idl0.textContent = "";   // stale identity must not survive a config change
  const obj = $("demo-objective");
  if (obj) obj.textContent = "";
  renderAlgorithmComparison();
  const st = $("ex-auto-status");
  if (st) { st.className = "ex-verdict"; st.textContent = "Configuration changed — results recomputing…"; }
  renderExplain();
  updateManualApplicability();
  if (typeof qvBump === "function") qvBump();  // invalidate QPU-validation derived state
}

function _syncControlsFromConfig() {
  const c = currentConfiguration;
  if ($("instance").value !== c.instance) $("instance").value = c.instance;
  if ($("scenario").value !== c.scenario) $("scenario").value = c.scenario;
  const nt = $("ntheta");
  if (nt && parseInt(nt.value, 10) !== c.n_theta) { nt.value = c.n_theta; $("ntheta-val").textContent = `K${c.n_theta}`; }
  const qbnt = $("qb-ntheta");
  if (qbnt) { qbnt.value = c.n_theta; $("qb-ntheta-val").textContent = c.n_theta; }
}

// Radar + QUBO tab + controls always describe the SAME problem.
async function refreshPreview() {
  const snap = _configSnapshot(); // pin the config this preview is for
  _syncControlsFromConfig();
  clearError();
  const p = await api.post("/api/preview", {
    instance: snap.instance, subset: snap.subset, n_theta: snap.n_theta, n_q: snap.n_q,
  });
  if (snap.id !== CONFIG_ID) return; // a newer configuration superseded this preview
  LAST_SOLUTION = null;
  drawRadar(p);
  setMetrics(p);
  renderExplain();
  renderAmplGurobi();  // per-instance AMPL/Gurobi baseline (read-only, no stale)
  if ($("tab-qubo").classList.contains("active")) loadQubo();
  scheduleAutoComparison();
}

function selectScenario(scId) {
  const sc = SCENARIOS[scId];
  if (!sc) return;
  currentConfiguration.scenario = scId;
  currentConfiguration.instance = sc.instance_name;
  currentConfiguration.subset = sc.subset || null;
  currentConfiguration.n_theta = sc.n_theta;
  currentConfiguration.n_q = sc.n_q || 1;
  invalidateDerivedState();
  refreshGridLimit().catch(showError);
  debouncedInstanceCatalog();
  refreshPreview().catch(showError);
}

function selectCustomInstance(name) {
  currentConfiguration.scenario = "";
  currentConfiguration.instance = name;
  currentConfiguration.subset = null;
  currentConfiguration.n_theta = parseInt($("ntheta").value, 10);
  invalidateDerivedState();
  refreshGridLimit().catch(showError);
  debouncedInstanceCatalog();
  refreshPreview().catch(showError);
}

// backwards-compatible entry used by init/reset
async function showInstance(name) {
  selectCustomInstance(name);
}

// Manual-solver preflight guard. The caps live in the backend preflight; this
// only reads LAST_PREFLIGHT — it never re-derives a cap in JavaScript. Real/
// remote hardware and the always-applicable routes are never blocked.
function _manualBlockReason() {
  const solver = $("solver").value, mode = $("mode").value;
  const capExempt = ["reference", "qubo-annealing", "dwave", "dwave-qpu"].includes(solver)
    || (solver === "qaoa" && mode !== "aer_sim");
  // only trust a preflight computed for the *current* configuration
  if (capExempt || !LAST_PREFLIGHT || LAST_PREFLIGHT_CID !== CONFIG_ID) return null;
  const m = LAST_PREFLIGHT.methods.find((x) => x.method === solver);
  return m && !m.applicable ? m.reason : null;
}

function updateManualApplicability() {
  const btn = $("solve-btn");
  const reason = _manualBlockReason();
  btn.disabled = !!reason;
  btn.title = reason || "";
  if (reason) $("status").textContent = `✗ ${$("solver").value} not applicable — ${reason}`;
  else if (btn.dataset.blocked === "1") $("status").textContent = "";
  btn.dataset.blocked = reason ? "1" : "";
}

async function solve() {
  const solver = $("solver").value;
  const mode = $("mode").value;
  // Preflight guard: never submit a method the backend reports as non-applicable.
  if (_manualBlockReason()) { updateManualApplicability(); return; }
  // QPU safety: never launch real hardware on a bare click
  const isQpu = (solver === "qaoa" && mode === "ibm_real") || solver === "dwave-qpu";
  if (isQpu) {
    const ok = await confirmQpu(solver, mode);
    if (!ok) return;
  }
  const btn = $("solve-btn");
  btn.disabled = true;
  clearError();
  $("status").textContent = "computing…";
  try {
    const c = currentConfiguration;
    const profile = $("profile").value;
    const body = {
      instance: c.instance, subset: c.subset, n_theta: c.n_theta, n_q: c.n_q,
      solver, mode,
      reps: parseInt($("reps").value, 10),
      profile,
    };
    if (profile === "scientific") {
      await runScientific(body); // seed-repeated measurement + dispersion
      return;
    }
    if ($("async-job").checked) {
      await runAsyncJob(body);  // manages its own status + button
      return;
    }
    const p = await api.post("/api/solve", body);
    applySolution(p);
  } catch (e) {
    showError(e);
  } finally {
    btn.disabled = false;
  }
}

// SCIENTIFIC profile: run the solver across several seeds and report dispersion
// (feasibility rate, objective mean ± sd, seed spread) instead of one number.
async function runScientific(body, save = false) {
  const seeds = ($("seeds").value || "")
    .split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isInteger(n));
  if (!seeds.length) { showError(new Error("Enter at least one integer seed")); return; }
  $("status").textContent = save
    ? `scientific run over ${seeds.length} seed(s) + saving…`
    : `scientific run over ${seeds.length} seed(s)…`;
  const out = await api.post("/api/scientific", { ...body, seeds, save });
  const a = out.aggregate;
  applySolution(out.runs[0]); // show the first seed's resolved radar/explanation
  $("sci-box").classList.remove("hidden");
  $("sci-runs").textContent = `${a.n_runs} seeds · ${a.solver}`;
  $("sci-feas").textContent = `${a.feasible_count}/${a.n_runs} (${(100 * a.feasible_rate).toFixed(0)}%)`;
  $("sci-obj").textContent = a.objective_mean != null
    ? `${a.objective_mean.toFixed(5)} ± ${(a.objective_stdev ?? 0).toFixed(5)}`
    : "—";
  $("sci-spread").textContent = a.objective_spread != null
    ? `${a.objective_spread.toFixed(5)}${a.deterministic ? " (deterministic)" : ""}`
    : "—";
  $("status").textContent = out.saved
    ? `✓ scientific run SAVED — ${out.saved.run_id} (${a.feasible_count}/${a.n_runs} feasible)`
    : `✓ scientific run complete — ${a.feasible_count}/${a.n_runs} feasible across seeds`;
  $("save-run-note").textContent = out.saved ? `saved: ${out.saved.path}` : "";
}

function _solveBody() {
  const c = currentConfiguration;
  return {
    instance: c.instance, subset: c.subset, n_theta: c.n_theta, n_q: c.n_q,
    solver: $("solver").value, mode: $("mode").value,
    reps: parseInt($("reps").value, 10), profile: $("profile").value,
  };
}

function applySolution(p) {
  drawRadar(p);
  setMetrics(p);
  LAST_SOLUTION = p;
  if (p.profile) {
    $("m-profile").textContent = p.profile.citable ? "scientific ✓" : "demo (not citable)";
    $("m-profile").title = p.profile.note || "";
    if (!p.profile.citable) $("sci-box").classList.add("hidden");
  }
  renderExplain();
  updateProvenancePanel(p);
  renderComparison(p);
  $("status").textContent = p.solution.feasible
    ? `✓ resolved (${p.solution.n_resolved}/${p.solution.n_initial_conflicts})`
    : `⚠ ${p.solution.n_residual_conflicts} conflict(s) remain`;
}

let LAST_SOLUTION = null;

// ---- bounded async jobs (Phase 6): submit, poll, cancel, reconnect --------
const JOB_KEY = "acrpq_job", HIST_KEY = "acrpq_job_hist";
let JOB_POLL = null;

async function runAsyncJob(body) {
  clearError();
  let sub;
  try { sub = await api.post("/api/jobs", body); }
  catch (e) { showError(e); $("solve-btn").disabled = false; return; }
  try { localStorage.setItem(JOB_KEY, sub.job_id); } catch { /* ignore */ }
  $("job-box").classList.remove("hidden");
  pollJob(sub.job_id);
}

function pollJob(jobId) {
  if (JOB_POLL) clearInterval(JOB_POLL);
  const cancelBtn = $("job-cancel");
  cancelBtn.disabled = false;
  const tick = async () => {
    let s;
    try { s = await api.get(`/api/jobs/${jobId}`); }
    catch { stopJob(); $("job-state").textContent = "expired"; return; }
    $("job-state").textContent = s.state + (s.cancel_requested ? " (cancel…)" : "");
    $("job-elapsed").textContent = `${s.elapsed_s.toFixed(1)} s`;
    if (["completed", "failed", "cancelled"].includes(s.state)) {
      stopJob();
      pushHistory(jobId, s.state);
      try { localStorage.removeItem(JOB_KEY); } catch { /* ignore */ }
      $("solve-btn").disabled = false;
      if (s.state === "completed") {
        try { applySolution(await api.get(`/api/jobs/${jobId}/result`)); }
        catch (e) { showError(e); }
      } else if (s.state === "failed") {
        showError(Object.assign(new Error(s.error || "job failed"), { status: 400 }));
      } else {
        $("status").textContent = "job cancelled";
      }
    }
  };
  JOB_POLL = setInterval(tick, 400);
  tick();
}

function stopJob() { if (JOB_POLL) { clearInterval(JOB_POLL); JOB_POLL = null; } $("job-cancel").disabled = true; }

async function cancelJob() {
  let jid = null;
  try { jid = localStorage.getItem(JOB_KEY); } catch { /* ignore */ }
  if (!jid) return;
  try {
    const s = await api.del(`/api/jobs/${jid}`);
    $("job-state").textContent = `${s.state} — ${s.cancel_note || ""}`;
  } catch { /* already gone */ }
}

function pushHistory(jobId, state) {
  let h = [];
  try { h = JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); } catch { h = []; }
  h.unshift(`${jobId.slice(0, 6)}:${state}`);
  h = h.slice(0, 5);
  try { localStorage.setItem(HIST_KEY, JSON.stringify(h)); } catch { /* ignore */ }
  $("job-hist").textContent = "recent: " + h.join(", ");
}

function reconnectJob() {
  let jid = null;
  try { jid = localStorage.getItem(JOB_KEY); } catch { /* ignore */ }
  if (!jid) return;
  $("job-box").classList.remove("hidden");
  $("job-state").textContent = "reconnecting…";
  pollJob(jid);
}

// ---- non-blocking error banner -------------------------------------------
function showError(e) {
  const [kind, advice] = classifyError(e);
  const box = $("err-box");
  if (!box) { $("status").textContent = "✗ " + e.message; return; }
  $("err-kind").textContent = kind.toUpperCase();
  $("err-msg").textContent = e.message;
  $("err-advice").textContent = advice;
  box.classList.remove("hidden");
  $("status").textContent = "✗ " + kind;
}
function clearError() { const b = $("err-box"); if (b) b.classList.add("hidden"); }

// ---- QPU confirmation modal ----------------------------------------------
function confirmQpu(solver, mode) {
  return new Promise((resolve) => {
    const m = $("qpu-modal");
    $("qpu-detail").textContent =
      `Solver: ${solver} · backend: ${solver === "dwave-qpu" ? "D-Wave QPU" : mode} · ` +
      `shots: ${$("shots") ? $("shots").value : "n/a"} · reps: ${$("reps").value}. ` +
      `Real quantum hardware: may queue, may incur cost/quota, and gives no guaranteed result.`;
    m.classList.remove("hidden");
    const done = (v) => { m.classList.add("hidden"); $("qpu-yes").onclick = null; $("qpu-no").onclick = null; resolve(v); };
    $("qpu-yes").onclick = () => done(true);
    $("qpu-no").onclick = () => done(false);
  });
}

// ---- reproducibility panel + exports (Phase 8) ---------------------------
function updateProvenancePanel(p) {
  const box = $("repro-box");
  if (!p || !p.provenance) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const pv = p.provenance, s = p.solution || {};
  const set = (id, v) => ($(id).textContent = v == null ? "—" : String(v));
  set("r-commit", pv.git_commit);
  set("r-dirty", pv.git_dirty ? "dirty" : "clean");
  set("r-hash", pv.source_hash ? pv.source_hash.slice(0, 12) : "—");
  set("r-py", pv.python_version);
  set("r-qiskit", (pv.deps && pv.deps.qiskit) || "absent");
  set("r-dwave", (pv.deps && pv.deps.dimod) || "absent");
  set("r-seed", pv.seed);
  set("r-backend", s.backend || s.solver_executed);
  set("r-grid", s.grid_k);
  set("r-lambda", s.lambda_pen != null ? `${s.lambda_pen.toFixed(1)} / ${s.lambda_oh.toFixed(1)}` : "—");
  set("r-ts", new Date().toISOString().replace("T", " ").slice(0, 19) + " (client)");
}

function _conflictsCSV(p) {
  const h = ["i", "j", "excluded", "initial_conflict", "resolved_conflict",
    "min_separation", "required", "violation_depth", "t_cpa", "closing"];
  const lines = [h.join(",")];
  (p.pairs || []).forEach((pr) => lines.push(h.map((k) => pr[k]).join(",")));
  return lines.join("\n");
}

function cliCommand(p) {
  const c = currentConfiguration, s = p.solution || {};
  const solver = s.solver_requested || "reference";
  if (["reference", "qubo-exact", "qubo-annealing", "dwave-sa"].includes(solver)) {
    let cmd = `acrpq classical ${c.instance} --solver ${solver} --n-theta ${c.n_theta}`;
    if (c.n_q > 1) cmd += ` --n-q ${c.n_q}`;
    return cmd;
  }
  // quantum route
  return `acrpq quantum ${c.scenario || c.instance} --mode ${$("mode").value} ` +
    `--reps ${$("reps").value} --seed ${p.provenance?.seed ?? 42}`;
}

function standaloneHTML(p) {
  const s = p.solution || {}, pv = p.provenance || {};
  const png = $("radar").toDataURL("image/png");
  const acRows = p.aircraft.map((a) =>
    `<tr><td>${a.id}</td><td>${a.heading_initial_deg?.toFixed(1)}</td>` +
    `<td>${a.heading_deg?.toFixed(1)}</td><td>${((a.theta || 0) * 180 / Math.PI).toFixed(1)}</td>` +
    `<td>${(a.q ?? 1).toFixed(3)}</td><td>${a.option_label ?? ""}</td>` +
    `<td>${(a.maneuver_cost ?? 0).toFixed(4)}</td></tr>`).join("");
  return `<!doctype html><html><head><meta charset="utf-8"><title>ACRP report ${p.name}</title>
<style>body{font-family:monospace;background:#04110d;color:#cdeee0;padding:24px}
h1{color:#36f2a0}table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #16463a;padding:4px 8px;text-align:right}
img{border:1px solid #16463a;max-width:640px}.warn{color:#ffcc4d}code{color:#ffcc4d}</style></head><body>
<h1>ACRP resolution report — ${p.name}</h1>
<p>solver requested <b>${s.solver_requested}</b> · executed <b>${s.solver_executed}</b> ·
backend <b>${s.backend || "—"}</b></p>
<h3>Result</h3>
<p>feasible <b>${s.feasible}</b> · conflicts ${s.n_residual_conflicts}/${s.n_initial_conflicts} resolved
${s.n_resolved} · objective ${s.objective?.toFixed(5)}</p>
<img src="${png}" alt="radar"/>
<h3>Maneuvers</h3>
<table><tr><th>#</th><th>init°</th><th>new°</th><th>Δθ°</th><th>q</th><th>option</th><th>cost</th></tr>${acRows}</table>
<h3>Provenance</h3>
<p>commit <code>${pv.git_commit}</code> (${pv.git_dirty ? "dirty" : "clean"}) ·
source_hash <code>${(pv.source_hash || "").slice(0, 16)}</code> · python ${pv.python_version} ·
qiskit ${pv.deps?.qiskit || "absent"} · seed ${pv.seed}</p>
<p>CLI: <code>${cliCommand(p)}</code></p>
<h3 class="warn">Scientific note</h3>
<p class="warn">This report animates the model's linear kinematics; it is not an operational
ATC simulation, not a physical high-fidelity simulation, and makes no quantum-advantage claim.
Simulator vs real-QPU results are distinct; times are host/queue-dependent.</p>
</body></html>`;
}

function setupExports() {
  const need = () => { if (!LAST_SOLUTION) { $("status").textContent = "solve first to export"; return false; } return true; };
  $("ex-json").onclick = () => need() && downloadText(`result_${LAST_SOLUTION.name}.json`, JSON.stringify(LAST_SOLUTION, null, 2));
  $("ex-aircraft").onclick = () => need() && exportManeuverCSV();
  $("ex-conflicts").onclick = () => need() && downloadText(`conflicts_${LAST_SOLUTION.name}.csv`, _conflictsCSV(LAST_SOLUTION));
  $("ex-png").onclick = () => {
    if (!SIM.payload) return;
    const a = document.createElement("a");
    a.href = $("radar").toDataURL("image/png");
    a.download = `radar_${SIM.payload.name}.png`; a.click();
  };
  $("ex-html").onclick = () => need() && downloadText(`report_${LAST_SOLUTION.name}.html`, standaloneHTML(LAST_SOLUTION));
  $("ex-cli").onclick = () => {
    if (!need()) return;
    navigator.clipboard?.writeText(cliCommand(LAST_SOLUTION));
    $("status").textContent = "CLI command copied";
  };
}

// ---- presentation mode (Phase 9) -----------------------------------------
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  const btn = document.querySelector(`.tab[data-tab="${name}"]`);
  if (btn) btn.classList.add("active");
  $("tab-" + name).classList.add("active");
  if (name === "benchmark") loadBenchmark();
  if (name === "qubo") loadQubo();
  if (name === "explain") loadQubo();
  if (name === "qpuval") { qvRefreshFlags().catch(showError); qvRender(); }
}

// each step drives the existing functions; nothing needs the network or a QPU
const PRESENT_STEPS = [
  { title: "1 · The conflict", desc: "CP_4: four aircraft converge on the centre — six pairwise conflicts.",
    run: async () => { switchTab("radar"); $("solver").value = "reference"; await selectCustomInstance("CP_4"); $("ba-before").click(); } },
  { title: "2 · Classical resolution", desc: "Exact in-grid solver finds minimal-deviation, conflict-free maneuvers.",
    run: async () => { await solve(); $("ba-after").click(); } },
  { title: "3 · Watch it play out", desc: "Linear kinematics animate to the closest points of approach.",
    run: async () => { $("ba-both").click(); resetSim(); playSim(); } },
  { title: "4 · The maneuvers", desc: "Per-aircraft heading/speed changes and their individual cost.",
    run: async () => { pauseSim(); seekSim(SIM.tMax); } },
  { title: "5 · QUBO reformulation", desc: "The same problem as a binary quadratic energy H(x).",
    run: async () => { switchTab("qubo"); } },
  { title: "6 · Quantum (QAOA, simulator)", desc: "Solve the identical QUBO with gate-model QAOA on the Aer simulator.",
    run: async () => {
      switchTab("radar");
      const qa = $("solver").querySelector('option[value="qaoa"]');
      if (qa && !qa.disabled) { $("solver").value = "qaoa"; $("mode").value = "aer_sim"; $("qaoa-opts").classList.remove("hidden"); await solve(); }
      else { $("status").textContent = "quantum extra not installed — showing classical"; }
    } },
  { title: "7 · Reproducibility", desc: "Every figure carries commit, source hash and versions; IBM hardware results are preserved.",
    run: async () => { updateProvenancePanel(LAST_SOLUTION); } },
];
let PS_I = 0;

function enterPresent() {
  document.body.classList.add("presenting");
  $("present-bar").classList.remove("hidden");
  PS_I = 0; runPresentStep();
  const el = document.documentElement;
  if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
}
function exitPresent() {
  document.body.classList.remove("presenting");
  $("present-bar").classList.add("hidden");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}
async function runPresentStep() {
  const s = PRESENT_STEPS[PS_I];
  $("ps-title").textContent = s.title;
  $("ps-desc").textContent = " — " + s.desc;
  try { await s.run(); } catch (e) { showError(e); }
}
function presentNext() { if (PS_I < PRESENT_STEPS.length - 1) { PS_I++; runPresentStep(); } }
function presentPrev() { if (PS_I > 0) { PS_I--; runPresentStep(); } }

// ---- automatic multi-solver comparison -----------------------------------
// Each route runs as a bounded JobManager job (POST /api/jobs → poll → result),
// so cancellation is a real backend DELETE, not just an abandoned browser fetch.
let AUTO_COMPARE = [];
let AUTO_RUN_ID = 0;            // generation id; results from an older id are ignored
let AUTO_TIMER = null;          // debounce timer
let AUTO_CONTROLLER = null;     // aborts the in-flight poll/submit fetch
let AUTO_JOB_ID = null;         // the backend job currently tracked (for real cancel)
const AUTO_POLL_MS = 400;
const AUTO_POLL_TIMEOUT_MS = 120000; // bounded polling per route

// Best-effort real cancellation of a backend job (fire-and-forget).
function _cancelJob(jobId) {
  if (jobId) api.del(`/api/jobs/${jobId}`).catch(() => { /* already gone */ });
}

// Supersede any running/pending comparison: stop the loop (fresh generation id),
// abort the in-flight poll, and DELETE the tracked backend job so the server is
// actually asked to cancel it (a queued job is dropped; a running one is asked).
function _supersedeAutoRun() {
  if (AUTO_TIMER) { clearTimeout(AUTO_TIMER); AUTO_TIMER = null; }
  if (AUTO_CONTROLLER) AUTO_CONTROLLER.abort();
  if (AUTO_JOB_ID) { _cancelJob(AUTO_JOB_ID); AUTO_JOB_ID = null; }
  AUTO_CONTROLLER = new AbortController();
  return { runId: ++AUTO_RUN_ID, signal: AUTO_CONTROLLER.signal };
}

function scheduleAutoComparison() {
  const { runId, signal } = _supersedeAutoRun();
  AUTO_TIMER = setTimeout(() => runAllAlgorithms(runId, signal), 350);
}

async function runCompare() {
  switchTab("radar");  // the unified comparison panel lives on the radar screen
  const { runId, signal } = _supersedeAutoRun();
  await runAllAlgorithms(runId, signal);
}

// Submit one solve as a bounded job and poll to a terminal state. Returns a
// tagged outcome; never throws for a normal failure/timeout (only AbortError).
async function _runSolveJob(body, runId, signal) {
  const sub = await api.post("/api/jobs", body, signal); // 429 if the backlog is full
  const jobId = sub.job_id;
  AUTO_JOB_ID = jobId;
  const t0 = Date.now();
  try {
    for (;;) {
      if (runId !== AUTO_RUN_ID) return { superseded: true };
      const snap = await api.get(`/api/jobs/${jobId}`, signal);
      if (snap.state === "completed") {
        return { ok: true, result: await api.get(`/api/jobs/${jobId}/result`, signal) };
      }
      if (snap.state === "failed") return { ok: false, error: snap.error || "failed" };
      if (snap.state === "cancelled") return { cancelled: true };
      if (Date.now() - t0 > AUTO_POLL_TIMEOUT_MS) {
        _cancelJob(jobId);
        return { ok: false, error: "polling timed out; cancellation requested" };
      }
      await _sleep(AUTO_POLL_MS, signal);
    }
  } finally {
    if (AUTO_JOB_ID === jobId) AUTO_JOB_ID = null;
  }
}

async function runAllAlgorithms(runId, signal) {
  const snap = _configSnapshot(); // the exact config this comparison runs on
  const c = {
    instance: snap.instance, subset: snap.subset, n_theta: snap.n_theta, n_q: snap.n_q,
  };
  if (!c.instance) return;
  const status = $("ex-auto-status");
  status.className = "ex-verdict";
  status.textContent = `Préparation de ${c.instance}…`;

  // The backend preflight is the single source of truth for applicability.
  let pf;
  try {
    pf = await api.post("/api/preflight", c, signal);
  } catch (e) {
    if (e.name === "AbortError" || runId !== AUTO_RUN_ID) return;
    status.className = "ex-verdict bad";
    status.textContent = `Échec de l\u2019analyse d\u2019applicabilité : ${e.message}`;
    return;
  }
  if (runId !== AUTO_RUN_ID || snap.id !== CONFIG_ID) return; // config moved on
  LAST_PREFLIGHT = pf; LAST_PREFLIGHT_CID = snap.id;
  COMPARE_SNAPSHOT = snap; // export/render bind to this snapshot, not live config
  renderExplain(); // show the pre-run capacity verdict immediately
  updateManualApplicability(); // keep the manual RUN button in sync with preflight
  renderBusinessContext(pf); // what the instance optimises + K semantics (French)
  const routes = pf.methods.map((m) => ({
    solver: m.method, label: m.label, family: m.family, mode: m.mode,
    labelFr: m.label_fr || m.label,
    business: m.business_reason_fr || null,
    skip: m.applicable ? null : m.reason,
  }));

  AUTO_COMPARE = routes.map((route) => ({
    ...route,
    state: route.skip ? "skipped" : "pending",
    reason: route.skip || "",
  }));
  renderAlgorithmComparison();
  $("ex-cancel-all").classList.remove("hidden"); // a run is now in progress

  // progress: "3/4 methods completed · QAOA running" + Δt between successive ends
  const total = AUTO_COMPARE.filter((r) => r.state !== "skipped").length;
  let done = 0;
  let lastEnd = null;
  for (const row of AUTO_COMPARE) {
    if (runId !== AUTO_RUN_ID) return;
    if (row.state === "skipped") continue;
    row.state = "running";
    status.className = "ex-verdict";
    status.dataset.state = "running";
  status.textContent = `${done}/${total} méthodes terminées · ${row.label} en cours…`;
    renderAlgorithmComparison();
    try {
      const out = await _runSolveJob({
        instance: c.instance, subset: c.subset, n_theta: c.n_theta, n_q: c.n_q,
        solver: row.solver, mode: row.mode || "aer_sim",
        reps: parseInt($("reps").value, 10), maxiter: 30, shots: 256, profile: "demo",
      }, runId, signal);
      if (runId !== AUTO_RUN_ID || out.superseded) return; // stale generation: drop
      if (out.ok) {
        row.state = "ok";
        row.payload = out.result;
        if (row.solver === "reference") applySolution(out.result);
      } else if (out.cancelled) {
        row.state = "cancelled";
      } else {
        row.state = "error";
        row.reason = out.error;
      }
    } catch (e) {
      if (e.name === "AbortError" || runId !== AUTO_RUN_ID) return;
      row.state = "error";
      row.reason = e.message;
    }
    done += 1;
    const now = Date.now();
    const dt = lastEnd == null ? null : (now - lastEnd) / 1000;
    lastEnd = now;
    status.textContent = `${done}/${total} méthodes terminées`
      + (dt != null ? ` · Δ +${dt.toFixed(1)}s since previous` : "");
    renderAlgorithmComparison();
  }

  if (runId !== AUTO_RUN_ID) return;
  $("ex-cancel-all").classList.add("hidden"); // run finished on its own
  const ok = AUTO_COMPARE.filter((row) => row.state === "ok").length;
  const skipped = AUTO_COMPARE.filter((row) => row.state === "skipped").length;
  const errors = AUTO_COMPARE.filter((row) => row.state === "error").length;
  status.className = `ex-verdict ${errors ? "bad" : "ok"}`;
  status.textContent = `Comparaison terminée : ${ok} exécutée(s), ${skipped} non applicable(s), ${errors} en échec.`
    + (skipped
      ? " Suggestion : réduire K ou choisir une autre méthode applicable — les méthodes applicables ont bien été exécutées."
      : "");
  status.dataset.state = "complete";   // language-independent hook for tests
  $("status").textContent = `✓ comparaison de tous les algorithmes prête (${ok} exécutée(s), ${skipped} ignorée(s))`;
  renderAlgorithmComparison();
  renderDemoExplanation();
  // Freeze the export snapshot NOW (deep copy): later exports write exactly what
  // is displayed for this settled epoch, never a mix with a newer configuration.
  if (snap.id === CONFIG_ID) {
    EXPORT_SNAPSHOT = JSON.parse(JSON.stringify(_comparisonExport()));
  }
  updateExportButtons();
}

// Cancel a running comparison: stop the loop, abort the in-flight poll, and
// DELETE the tracked job. A queued job is dropped; a running one is *asked* to
// cancel — we never claim its computation has stopped.
async function cancelAutoComparison() {
  const jobId = AUTO_JOB_ID;
  AUTO_JOB_ID = null;
  if (AUTO_TIMER) { clearTimeout(AUTO_TIMER); AUTO_TIMER = null; }
  if (AUTO_CONTROLLER) AUTO_CONTROLLER.abort();
  AUTO_CONTROLLER = new AbortController();
  ++AUTO_RUN_ID; // supersede: the loop stops and any late result is ignored

  let note = "queued routes dropped";
  if (jobId) {
    try {
      const snap = await api.del(`/api/jobs/${jobId}`);
      note = snap && snap.state === "running"
        ? "cancellation requested; running computation may finish server-side"
        : (snap && snap.cancel_note) || "job cancelled before start";
    } catch { /* job already terminal/expired */ }
  }
  for (const row of AUTO_COMPARE) {
    if (row.state === "pending" || row.state === "running") row.state = "cancelled";
  }
  renderAlgorithmComparison();
  updateExportButtons(); // a cancelled run never becomes exportable
  $("ex-cancel-all").classList.add("hidden");
  const status = $("ex-auto-status");
  status.className = "ex-verdict";
  status.textContent = `Comparaison annulée (cancelled) — ${note}.`;
  $("status").textContent = "✕ comparison cancelled";
}

// ---- full-comparison export (AUTO_COMPARE, all methods) -------------------
function _escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}
function _csvCell(value) {
  const s = value == null ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// Structured snapshot of the whole comparison — every method's status, outcome
// and provenance, plus honest caveats. The interactive comparison is DEMO, so
// it is explicitly not citable, and Δ is only an "optimality gap" against a
// proven-optimum anchor.
function _comparisonExport() {
  const anchorRow = AUTO_COMPARE.find((r) => r.solver === "reference" && r.state === "ok");
  const anchor = anchorRow?.payload?.solution?.objective;
  const referenceTime = anchorRow?.payload?.solution?.wall_time_s;
  const gapAllowed = !!anchorRow?.payload?.solution?.proof_status?.gap_allowed;
  // Bind to the snapshot the comparison actually ran on — never the live config
  // (which may already have moved on to another instance).
  const snap = COMPARE_SNAPSHOT;
  return {
    schema: "acrpq-comparison/1",
    kind: "solver-comparison",
    created_utc: new Date().toISOString(),
    citable: false,
    profile: "demo",
    config_id: snap ? snap.id : null,
    configuration: snap
      ? { instance: snap.instance, subset: snap.subset, n_theta: snap.n_theta, n_q: snap.n_q, w: snap.w }
      : null,
    preflight: LAST_PREFLIGHT
      ? { methods: LAST_PREFLIGHT.methods, resources: LAST_PREFLIGHT.resources }
      : null,
    anchor_is_proven_optimum: gapAllowed,
    caveats: [
      "Interactive DEMO comparison — not a citable measurement.",
      gapAllowed
        ? "Δ vs reference is a genuine optimality gap (proven-optimum anchor)."
        : "Δ vs reference is a route-to-route difference, NOT an optimality gap (heuristic anchor).",
      "Wall times are host-dependent and are not evidence of quantum advantage.",
    ],
    methods: AUTO_COMPARE.map((row, idx) => {
      const s = row.payload?.solution;
      const prev = AUTO_COMPARE.slice(0, idx).reverse()
        .find((r) => r.payload?.solution?.wall_time_s != null);
      const prevTime = prev?.payload?.solution?.wall_time_s ?? null;
      return {
        method: row.solver, label: row.label, family: row.family,
        label_fr: row.labelFr ?? null,
        state: row.state, reason: row.reason || null,
        business_reason_fr: row.business ?? null,
        feasible: s?.feasible ?? null,
        objective_id: LAST_PREFLIGHT?.objective_id ?? null,   // two objectives exist: never ambiguous
        residual_conflicts: s?.n_residual_conflicts ?? null,
        initial_conflicts: s?.n_initial_conflicts ?? null,
        objective: s?.objective ?? null,
        // No official delta/match for an INFEASIBLE solution: an infeasible route can
        // score below the feasible optimum and would export a negative gap (and even
        // matches_reference:true). The on-screen table already guards on feasibility;
        // the downloadable artifact must too.
        delta_vs_reference: s && s.feasible && anchor != null && s.objective != null
          ? s.objective - anchor : null,
        matches_reference: s && s.feasible && anchor != null && s.objective != null
          ? Math.abs(s.objective - anchor) <= 1e-9 : null,
        delta_vs_reference_suppressed_reason:
          s && !s.feasible ? "infeasible_solution_no_official_gap" : null,
        n_maneuvered: s?.n_maneuvered ?? null,
        n_unchanged: s?.n_unchanged ?? null,
        proof_status: s?.proof_status?.status ?? null,
        gap_allowed: s?.proof_status?.gap_allowed ?? null,
        energy: s?.qubo_energy ?? s?.qaoa_energy ?? s?.explanation?.energy?.total ?? null,
        wall_time_s: s?.wall_time_s ?? null,
        wall_time_delta_vs_reference_s: s?.wall_time_s != null && referenceTime != null
          ? s.wall_time_s - referenceTime : null,
        wall_time_delta_vs_previous_s: s?.wall_time_s != null && prevTime != null
          ? s.wall_time_s - prevTime : null,
        profile: row.payload?.profile?.name ?? null,
        provenance: row.payload?.provenance ?? null,
      };
    }),
  };
}

function _comparisonCSV(data) {
  const cols = ["method", "state", "reason", "feasible", "residual_conflicts",
    "objective", "delta_vs_reference", "proof_status", "gap_allowed", "energy",
    "wall_time_s", "wall_time_delta_vs_reference_s", "profile"];
  const lines = [cols.join(",")];
  for (const m of data.methods) lines.push(cols.map((k) => _csvCell(m[k])).join(","));
  return lines.join("\n") + "\n";
}

function _comparisonHTML(data) {
  const cfg = data.configuration;
  const rows = data.methods.map((m) => `<tr>
    <td>${_escapeHtml(m.label)}</td><td>${_escapeHtml(m.state)}</td>
    <td>${_escapeHtml(m.feasible)}</td>
    <td>${m.objective != null ? _escapeHtml(m.objective.toFixed(5)) : "—"}</td>
    <td>${m.delta_vs_reference != null ? _escapeHtml(m.delta_vs_reference.toFixed(5)) : "—"}</td>
    <td>${_escapeHtml(m.proof_status ?? "—")}</td>
    <td>${m.wall_time_s != null ? _escapeHtml(m.wall_time_s.toFixed(3)) : "—"}</td>
    <td>${m.wall_time_delta_vs_reference_s != null ? _escapeHtml(m.wall_time_delta_vs_reference_s.toFixed(3)) : "—"}</td>
    <td>${_escapeHtml(m.reason ?? "")}</td></tr>`).join("");
  const caveats = data.caveats.map((t) => `<li>${_escapeHtml(t)}</li>`).join("");
  return `<!doctype html><html><head><meta charset="utf-8">
<title>ACRP comparison ${_escapeHtml(cfg.instance)}</title>
<style>body{font-family:monospace;background:#04110d;color:#cdeee0;padding:24px}
h1{color:#36f2a0}table{border-collapse:collapse;margin:10px 0}
td,th{border:1px solid #16463a;padding:4px 8px;text-align:left}.warn{color:#ffcc4d}</style></head><body>
<h1>ACRP solver comparison — ${_escapeHtml(cfg.instance)}</h1>
<p>K=${_escapeHtml(cfg.n_theta)} · generated ${_escapeHtml(data.created_utc)} ·
profile <b>${_escapeHtml(data.profile)}</b> · citable <b>${_escapeHtml(data.citable)}</b></p>
<table><tr><th>Algorithm</th><th>State</th><th>Feasible</th><th>Objective</th>
<th>Δ vs ref</th><th>Proof</th><th>Time (s)</th><th>Δ time vs ref (s)</th><th>Why / reason</th></tr>${rows}</table>
<h3>Caveats</h3><ul class="warn">${caveats}</ul></body></html>`;
}

// Export buttons are usable only once the current epoch's runs have settled and
// the frozen snapshot matches the live configuration (no mixing, ever).
function _exportReady() {
  return !!(EXPORT_SNAPSHOT && EXPORT_SNAPSHOT.config_id === CONFIG_ID
    && COMPARE_SNAPSHOT && COMPARE_SNAPSHOT.id === CONFIG_ID);
}

function updateExportButtons() {
  const ready = _exportReady();
  for (const id of ["cmp-json", "cmp-csv", "cmp-html"]) {
    const b = $(id);
    if (b) b.disabled = !ready;
  }
  const note = $("cmp-export-note");
  if (note) {
    note.textContent = ready
      ? "" : "export désactivé tant que les calculs de la configuration courante ne sont pas terminés";
  }
}

function exportComparison(fmt) {
  // Export only the frozen snapshot of a completed comparison belonging to the
  // *current* configuration — otherwise the file would pair the new instance's
  // config with old results. A missing/stale snapshot covers both "nothing run
  // yet" and "config changed"; mixing epochs is refused outright.
  if (!_exportReady()) {
    $("status").textContent =
      "comparison results are recomputing for the current configuration — export again once ready";
    return;
  }
  const data = EXPORT_SNAPSHOT; // immutable: frozen when the comparison settled
  const base = `comparison_${data.configuration.instance || "instance"}`;
  if (fmt === "json") downloadText(`${base}.json`, JSON.stringify(data, null, 2));
  else if (fmt === "csv") downloadText(`${base}.csv`, _comparisonCSV(data));
  else downloadText(`${base}.html`, _comparisonHTML(data));
}

function _algorithmReason(row, anchor) {
  if (row.state === "skipped") return row.reason;
  if (row.state === "error") return `failed independently: ${row.reason}`;
  if (row.state === "cancelled") return "cancelled by user";
  if (row.state !== "ok") return row.state === "running" ? "running now…" : "queued";
  const s = row.payload.solution;
  const outcome = s.feasible
    ? `feasible, objective ${s.objective.toFixed(5)}`
    : `${s.n_residual_conflicts} residual conflict(s)`;
  if (row.solver === "reference") {
    return s.exact_method
      ? `exact in-grid anchor; ${outcome}`
      : `greedy/local-search fallback because the exact grid search is too large; ${outcome}`;
  }
  if (row.solver === "qubo-exact") {
    const domain = s.exact_domain === "all_bitstrings" ? "all bitstrings" : "the one-hot subspace";
    return `exact minimum of the encoded H(x) over ${domain}; ${outcome}`;
  }
  if (row.solver === "qubo-annealing") return `stochastic local search on the same H(x); ${outcome}`;
  if (row.solver === "dwave") return `Recuit simulé Ocean (neal) sur CPU local ; ${outcome}`;
  if (row.solver === "qaoa") {
    const strategy = s.exact_component_decomposition
      ? `exact model split into ${s.n_components} independent circuits (largest ${s.largest_executed_circuit_qubits}q)`
      : `one ${s.n_qubits}q variational circuit`;
    return `${strategy}; sampled approximate optimizer; ${outcome}`;
  }
  return anchor ? outcome : `${outcome}; no exact anchor available`;
}

// Cell-class-aware table renderer (values may be {t, cls, title} or a scalar).
function _renderRichTable(id, headers, rows) {
  const table = $(id);
  if (!table) return;
  table.replaceChildren();
  const trh = document.createElement("tr");
  headers.forEach((h) => { const th = document.createElement("th"); th.textContent = h; trh.appendChild(th); });
  table.appendChild(trh);
  rows.forEach((cells) => {
    const tr = document.createElement("tr");
    cells.forEach((c) => {
      const td = document.createElement("td");
      if (c && typeof c === "object") {
        td.textContent = c.t == null ? "—" : String(c.t);
        if (c.cls) td.className = c.cls;
        if (c.title) td.title = c.title;
      } else { td.textContent = c == null ? "—" : String(c); }
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
}

const _cf = (x, d = 5) => (typeof x === "number" && isFinite(x)) ? x.toFixed(d) : "—";
// Wall-time cell with honest Δ annotations (vs reference and vs previous method).
const _TIME_HONESTY =
  "Temps mural (résolution seule, pas de répartition par phases disponible) — "
  + "non reproductible bit-à-bit ; un temps de file d'attente QPU ne serait pas comparable.";
function _timeCell(t, refT, prevT) {
  if (!(typeof t === "number" && isFinite(t))) return { t: "—" };
  const parts = [];
  if (typeof refT === "number" && isFinite(refT))
    parts.push(`Δréf ${_cs((t - refT) * 1000, 1)} ms`);
  // Δpréc removed: it differenced against the PREVIOUS DISPLAYED ROW, so its sign and
  // magnitude depended on table ordering and read as a speed-up (MoE Expert G, C3).
  // Only Δ vs the reference is well-defined.
  void prevT;
  return {
    t: _ct(t) + (parts.length ? ` (${parts.join(" · ")})` : ""),
    title: _TIME_HONESTY,
  };
}
const _cs = (x, d = 5) => (typeof x === "number" && isFinite(x)) ? `${x >= 0 ? "+" : ""}${x.toFixed(d)}` : "—";
const _ct = (x) => {
  if (!(typeof x === "number" && isFinite(x))) return "—";
  return x < 1 ? `${(x * 1000).toFixed(x < 0.01 ? 2 : 1)} ms` : `${x.toFixed(3)} s`;
};

// The current-instance Gurobi continuous leg (read-only payload) or null. The campaign
// ran WHOLE instances, so the leg never applies to a subset scenario (a different,
// smaller problem) — matching one would be scientifically wrong.
function _gurobiLeg() {
  if (AMPL_GUROBI_STATE !== "ready" || !AMPL_GUROBI || !AMPL_GUROBI.instances) return null;
  if (currentConfiguration.subset) return null;  // whole-instance only
  const e = AMPL_GUROBI.instances.find((x) => x.instance === currentConfiguration.instance);
  return (e && !e.artifact_error) ? e : null;
}

// Unified RESULTS chain: continuous Gurobi -> discretisation cost -> discrete K ->
// algorithmic gap -> executed algorithms (all on the SAME grid; never K3 vs K5/K7).
function renderAlgorithmComparison() {
  const table = $("ex-algo-table");
  if (!table) return;
  const nt = currentConfiguration.n_theta;
  const ref = AUTO_COMPARE.find((r) => r.solver === "reference" && r.state === "ok");
  const rs = ref?.payload?.solution;
  const discObj = rs?.objective ?? null;
  const discProven = !!rs?.proof_status?.gap_allowed;   // proven in-grid optimum
  const g = _gurobiLeg();
  const gObj = g ? g.original_ampl_gurobi_objective : null;
  const gOfficial = !!(g && g.is_official_anchor);
  const discCost = (gObj != null && discObj != null) ? discObj - gObj : null;
  const discCostOfficial = gOfficial && discProven && discCost != null;
  // wording: "gap" only when the compared-against leg is a proven optimum
  const algLabel = discProven ? "algorithmic gap" : "difference vs discrete incumbent";

  const header = ["Méthode", "Domaine", "Objectif", "Faisable", "Garantie", "Temps",
                  `Δ vs discrete K${nt}`, "Δ vs continuous Gurobi", "Statut / applicabilité"];
  const rows = [];

  if (g) {
    rows.push([
      "Original AMPL / Gurobi", "continuous", { t: _cf(gObj, 6) },
      { t: g.feasible_common_numeric === true ? "yes" : (g.feasible_common_numeric === false ? "no" : "unknown"),
        cls: g.feasible_common_numeric === true ? "cell-ok" : (g.feasible_common_numeric === false ? "cell-bad" : "cell-warn") },
      { t: gOfficial ? "certified: gap≤1e-6, MP≤1e-8" : "provisional incumbent (not certified)",
        cls: gOfficial ? "cell-ok" : "cell-warn" },
      { t: _ct(g.wall_time_s) }, { t: "N/A" }, { t: "0", cls: "cell-mut" },
      { t: gOfficial ? "Certified numerical anchor" : "Provisional feasible incumbent",
        cls: gOfficial ? "cell-ok" : "cell-warn" },
    ]);
  }
  if (ref) {
    rows.push([
      `Référence discrète K${nt}`, `grid K${nt}`, { t: _cf(discObj) },
      { t: rs.feasible ? "yes" : "no", cls: rs.feasible ? "cell-ok" : "cell-bad" },
      { t: discProven ? "Proven in-grid optimum" : "Discrete incumbent (heuristic/timeout)",
        cls: discProven ? "cell-ok" : "cell-warn" },
      _timeCell(rs.wall_time_s, rs.wall_time_s, null), { t: "0", cls: "cell-mut" },
      { t: discCost == null ? "N/A" : _cs(discCost) + (discCostOfficial ? "" : " (prov.)"),
        cls: discCostOfficial ? "" : "cell-warn",
        title: discCostOfficial ? "official discretisation cost = discrete − continuous"
          : "not an official discretisation cost (Gurobi not an anchor or K not proven)" },
      { t: discProven ? "exhaustive in-grid optimum" : "not exhaustively proven" },
    ]);
  }

  // executed algorithms on this grid, feasible-first (an infeasible low objective is
  // never ranked above a feasible one)
  const algo = AUTO_COMPARE.filter((r) => r.solver !== "reference");
  const rank = (r) => {
    const s = r.payload?.solution;
    if (r.state === "ok" && s?.feasible) return 0;
    if (r.state === "ok") return 1;
    if (r.state === "skipped") return 3;
    return 2;
  };
  const scientificOrder = {"qubo-exact": 0, "qubo-annealing": 1, "dwave": 2, "qaoa": 3};
  algo.sort((a, b) => rank(a) - rank(b)
    || (scientificOrder[a.solver] ?? 99) - (scientificOrder[b.solver] ?? 99)
    || (a.label || "").localeCompare(b.label || ""));
  let prevTime = rs?.wall_time_s ?? null; // Δpréc: previous displayed method's wall time
  for (const r of algo) {
    const s = r.payload?.solution;
    const obj = s?.objective ?? null;
    const feas = s ? !!s.feasible : null;
    const alg = (feas && obj != null && discObj != null) ? obj - discObj : null; // no ratio for infeasible
    const matchesRef = alg != null && Math.abs(alg) <= 1e-9;
    const total = (feas && obj != null && gObj != null && gOfficial) ? obj - gObj : null;
    let statusCell;
    if (r.state === "skipped") {
      // BUSINESS language first (never an error); the technical string stays in
      // the tooltip and in the collapsible "Détails techniques" block below.
      statusCell = { t: `non applicable — ${r.business || r.reason || "capacité"}`,
                     cls: "cell-mut", title: r.reason || "" };
    } else if (r.state === "ok") statusCell = { t: feas ? "feasible" : `INFEASIBLE (${s.n_residual_conflicts} conflicts)`,
                                              cls: feas ? "cell-ok" : "cell-bad" };
    else statusCell = { t: r.state + (r.reason ? `: ${r.reason}` : ""), cls: "cell-bad" };
    const timeCell = _timeCell(s?.wall_time_s, rs?.wall_time_s ?? null, prevTime);
    if (typeof s?.wall_time_s === "number" && isFinite(s.wall_time_s)) prevTime = s.wall_time_s;
    rows.push([
      { t: r.label, title: r.labelFr || "" }, `grid K${nt}`,
      { t: _cf(obj) },
      { t: s ? (feas ? "yes" : "no") : "—", cls: s ? (feas ? "cell-ok" : "cell-bad") : "" },
      { t: r.state === "ok"
          ? (s?.proof_status?.label || "approximate (no optimality guarantee)")
          : "—" },
      timeCell,
      { t: alg == null ? (r.state === "ok" && !feas ? "N/A (infeasible)" : "—")
             : _cs(alg) + (matchesRef ? " · = réf ✓" : "") + (discProven ? "" : " (diff.)"),
        cls: matchesRef ? "cell-ok" : "",
        title: feas ? (discProven ? `${algLabel} = method − discrete K${nt}` : algLabel) : "no official ratio for an infeasible solution" },
      { t: total == null ? (gObj != null && !gOfficial && feas ? "provisional" : "N/A")
             : _cs(total),
        cls: total == null ? "cell-mut" : "",
        title: gOfficial ? "total continuous gap = method − Gurobi continuous"
          : "Gurobi is not an official anchor here → provisional / N/A" },
      statusCell,
    ]);
  }

  _renderRichTable("ex-algo-table", header, rows);
  _renderSkippedTechnicalDetails();
  _renderControlDifference(g, ref, nt);
  _renderChainDecomposition({ nt, g, gObj, gOfficial, discObj, discProven, discCost, discCostOfficial, algo });
  if (SIM.payload) renderRadarAt(SIM.t);
}

// Collapsible technical detail for the skipped methods: the table shows the
// business-language reason; the exact backend formula/cap string lives here.
function _renderSkippedTechnicalDetails() {
  const box = $("na-tech"), list = $("na-tech-list");
  if (!box || !list) return;
  const skipped = AUTO_COMPARE.filter((r) => r.state === "skipped");
  box.classList.toggle("hidden", skipped.length === 0);
  list.replaceChildren(...skipped.map((r) => {
    const li = document.createElement("li");
    li.textContent = `${r.label} — ${r.reason || "no technical detail"}`;
    return li;
  }));
}

// What the selected instance optimises + K semantics, in business language.
// Both texts come from the backend preflight (single source of truth).
function renderBusinessContext(pf) {
  const obj = $("demo-objective");
  if (obj) obj.textContent = pf.objective_fr || "";
  const k = $("k-info-text");
  if (k && pf.k_explanation_fr) k.textContent = pf.k_explanation_fr;
  // ALWAYS-VISIBLE scientific identity: which objective produced the numbers, on which
  // grid, with how many qubits. Previously objective_id appeared nowhere in the UI and
  // the qubit count was hidden inside an expert-only box (MoE Expert G, BLOCKER 1 & 3).
  const idl = $("demo-identity");
  if (idl) {
    const oid = pf.objective_id || "quadratic_control_cost_v1";
    const oname = oid === "maneuver_count_v1"
      ? "minimum d'avions déviés (maneuver_count_v1)"
      : "objectif quadratique historique (quadratic_control_cost_v1)";
    const n = pf.n_aircraft ?? pf.n ?? null;
    const K = pf.grid_k ?? pf.n_theta ?? null;
    const q = pf.n_qubits ?? (n != null && K != null ? n * K : null);
    const bits = [`Objectif optimisé : ${oname}`];
    if (K != null) bits.push(`grille K${K}`);
    if (q != null) bits.push(`${q} variables binaires (= qubits${n != null && K != null
      ? ` = ${n} avions × K${K}` : ""})`);
    idl.textContent = bits.join(" · ");
  }
  // K=3 degeneracy: displayed ONLY when the backend computed that every non-NOOP
  // option shares one amplitude (true at K3/q1, false at K5/K7/q>1). Never generalised.
  const deg = pf.objective_degeneracy || null;
  const dbox = $("objective-degeneracy");
  if (dbox) {
    if (deg && deg.proportional) {
      dbox.textContent = deg.note_fr || "";
      dbox.hidden = false;
    } else {
      dbox.textContent = "";
      dbox.hidden = true;
    }
  }
}

// ---- business-language explanation of a completed run ----------------------
const _FR_LABELS_FALLBACK = {
  reference: "référence classique",
  "qubo-exact": "recherche exacte",
  "qubo-annealing": "heuristique classique (recuit simulé)",
  dwave: "heuristique classique (recuit simulé)",
  qaoa: "QAOA simulé localement (Aer)",
};
const _frLabel = (row) => row.labelFr || _FR_LABELS_FALLBACK[row.solver] || row.label;

function _bestClassicalReference() {
  const classical = ["reference", "qubo-exact", "qubo-annealing", "dwave"];
  const done = AUTO_COMPARE.filter((r) => classical.includes(r.solver)
    && r.state === "ok" && r.payload?.solution?.feasible
    && r.payload.solution.objective != null);
  if (!done.length) return null;
  return done.reduce((a, b) =>
    a.payload.solution.objective <= b.payload.solution.objective ? a : b);
}

const _isDeviated = (a) =>
  a.deviated === true || (a.deviated == null && !!a.option_label && a.option_label !== "NOOP");

function renderDemoExplanation() {
  const sel = $("expl-method"), sum = $("expl-summary");
  const dev = $("expl-deviations"), note = $("expl-note");
  if (!sel || !sum) return;
  const done = AUTO_COMPARE.filter((r) => r.state === "ok" && r.payload?.solution);
  const prevChoice = sel.value;
  sel.replaceChildren(...done.map((r) => new Option(_frLabel(r), r.solver)));
  if (!done.length) {
    sum.className = "ex-verdict";
    sum.textContent = "Lancez la comparaison pour obtenir l'explication du résultat.";
    if (dev) dev.textContent = "";
    if (note) note.textContent = "";
    _renderRichTable("expl-aircraft-table",
      ["Avion", "Option initiale", "Option finale", "q", "θ (°)", "Dévié ?"], []);
    return;
  }
  sel.value = done.some((r) => r.solver === prevChoice)
    ? prevChoice
    : (done.find((r) => r.solver === "reference") || done[0]).solver;
  const row = done.find((r) => r.solver === sel.value) || done[0];
  const p = row.payload, s = p.solution;
  const best = _bestClassicalReference();
  let deltaTxt = "aucune référence classique disponible";
  if (!s.feasible) {
    // an infeasible route gets no official gap (it can score below the feasible optimum)
    deltaTxt = "aucun écart officiel : la solution est infaisable";
  } else if (best && s.objective != null) {
    deltaTxt = best.solver === row.solver
      ? "c'est la meilleure référence classique disponible"
      : `écart vs meilleure référence classique (${_frLabel(best)}) : `
        + _cs(s.objective - best.payload.solution.objective);
  }
  const devs = (p.aircraft || []).filter(_isDeviated);
  const nDev = s.n_maneuvered != null ? s.n_maneuvered : devs.length;
  sum.className = `ex-verdict ${s.feasible ? "ok" : "bad"}`;
  sum.textContent =
    `Méthode : ${_frLabel(row)}. `
    + (s.feasible
      ? `Solution sans conflit (${s.n_resolved}/${s.n_initial_conflicts} conflits initiaux résolus)`
      : `Solution INFAISABLE : ${s.n_residual_conflicts} conflit(s) restant(s)`)
    + ` · objectif (déviation totale) ${_cf(s.objective)} · ${deltaTxt}.`;
  if (dev) {
    dev.textContent =
      `Avions déviés / réaffectations vs planning initial : ${nDev} / ${p.n} — `
      + `${p.n - nDev} avion(s) inchangé(s) (option no-op).`;
  }
  _renderRichTable("expl-aircraft-table",
    ["Avion", "Option initiale", "Option finale", "q", "θ (°)", "Dévié ?"],
    (p.aircraft || []).map((a) => {
      const deviated = _isDeviated(a);
      return [
        `A${a.id}`, "no-op (q=1, θ=0°)", a.option_label ?? "—",
        (a.q ?? 1).toFixed(3), ((a.theta || 0) * 180 / Math.PI).toFixed(1),
        { t: deviated ? "oui — dévié" : "non", cls: deviated ? "cell-warn" : "cell-ok" },
      ];
    }));
  if (note) {
    note.textContent =
      "Objectif optimisé : minimiser la déviation totale par rapport au planning "
      + "initial, sous contrainte de séparation. Une simulation n'est jamais du "
      + "« quantique réel » ; tout circuit destiné au QPU reste « circuit préparé "
      + "pour hardware, non soumis ».";
  }
}

function _renderControlDifference(g, ref, nt) {
  const table = $("control-diff-table");
  const note = $("control-diff-note");
  if (!table || !note) return;
  const q = g?.continuous_controls?.q;
  const theta = g?.continuous_controls?.theta;
  const aircraft = ref?.payload?.aircraft;
  if (!Array.isArray(q) || !Array.isArray(theta) || !Array.isArray(aircraft)
      || q.length !== aircraft.length || theta.length !== aircraft.length) {
    _renderRichTable("control-diff-table", ["Avion", "Continu", `Discrète K${nt}`], []);
    note.textContent = g
      ? "Control vectors are unavailable or have incompatible cardinality; no comparison is fabricated."
      : "No continuous AMPL/Gurobi leg applies to this configuration.";
    return;
  }
  const deg = (x) => x * 180 / Math.PI;
  const rows = aircraft.map((a, idx) => {
    const cq = Number(q[idx]), ct = Number(theta[idx]);
    const dq = Number(a.q), dt = Number(a.theta);
    return [
      `A${a.id}`,
      `q=${_cf(cq, 6)} · θ=${_cf(deg(ct), 3)}°`,
      `q=${_cf(dq, 6)} · θ=${_cf(deg(dt), 3)}° · ${a.option_label || "option"}`,
      `${_cs(dq - cq, 6)} · ${_cs(deg(dt - ct), 3)}°`,
    ];
  });
  _renderRichTable("control-diff-table",
    ["Avion", "Gurobi continu", `Discrète K${nt}`, "Δq · Δθ"], rows);
  note.textContent = `The K${nt} reference can only select its configured levels; these control jumps explain the discretisation cost shown above.`;
}

// "Continuous optimum + discretisation cost + algorithmic gap = method objective"
function _renderChainDecomposition(x) {
  const box = $("ex-chain-decomp");
  if (!box) return;
  box.replaceChildren();
  if (!x.g || x.discObj == null) return;  // need both continuous + discrete legs
  const word = x.discProven && x.gOfficial ? "gap" : "difference";
  const head = document.createElement("div");
  head.className = "chain-head";
  head.textContent = "Continuous optimum + discretisation cost + algorithmic " + word + " = method objective";
  box.appendChild(head);
  const feasibleAlgo = x.algo.filter((r) => r.state === "ok" && r.payload?.solution?.feasible);
  for (const r of feasibleAlgo) {
    const obj = r.payload.solution.objective;
    const alg = (obj != null && x.discObj != null) ? obj - x.discObj : null;
    const line = document.createElement("div");
    line.className = "chain-line";
    line.textContent =
      `Gurobi ${_cf(x.gObj, 6)}  +  K${x.nt} discretisation ${_cf(x.discCost, 6)}` +
      `  +  ${r.label} ${word} ${_cf(alg, 6)}  =  ${r.label} objective ${_cf(obj, 6)}`;
    box.appendChild(line);
  }
  if (!x.gOfficial) {
    const w = document.createElement("div");
    w.className = "chain-line warn";
    w.textContent = "Gurobi is a provisional incumbent here — the continuous differences are provisional, not official.";
    box.appendChild(w);
  }
}

function _renderResultStory(x) {
  const story = $("ex-result-story");
  if (!story) return;  // result-story block removed from the UI
  story.replaceChildren();
  const completed = AUTO_COMPARE.filter((row) => row.state === "ok" && row.solver !== "reference");
  if (!completed.length && !x.ref) return;
  const add = (text, warn = false) => {
    const p = document.createElement("p");
    p.textContent = text; if (warn) p.className = "warn"; story.appendChild(p);
  };
  const feasible = completed.filter((row) => row.payload.solution.feasible);
  add(`${feasible.length}/${completed.length} executed algorithms return a conflict-free decoded solution; feasible solutions are ranked before infeasible ones. All objectives are recomputed by the shared geometry scorer on the SAME K${x.nt} grid.`);
  if (x.discProven) {
    add(`The discrete reference K${x.nt} is a proven in-grid optimum, so the "Δ vs discrete" column is a genuine algorithmic gap (extra in-grid cost above the optimum).`);
  } else {
    add(`The discrete reference K${x.nt} is a heuristic/timeout incumbent, NOT a proven optimum — "Δ vs discrete" is only a difference against that incumbent, not an optimality gap.`, true);
  }
  const skipped = AUTO_COMPARE.filter((row) => row.state === "skipped");
  if (skipped.length) add(`${skipped.length} route(s) were not applicable and were not faked: read their per-row reason.`, true);
  add("Discretisation sensitivity K5/K7 lives in its own panel and is never used as the reference for K3-grid algorithms. Wall times are shown for transparency and are not evidence of quantum advantage.");
}

// ---- before/after comparison + maneuver table (Phase 4) -------------------
let MAN_SORT = { key: "id", dir: 1 };

function renderComparison(p) {
  const box = $("compare-box");
  if (!p || !p.solution) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const s = p.solution;
  $("m-dth").textContent = s.total_heading_change != null
    ? `${(s.total_heading_change * 180 / Math.PI).toFixed(1)}°` : "—";
  $("m-dq").textContent = s.total_speed_change != null ? s.total_speed_change.toFixed(4) : "—";
  // minimum safety margin over non-excluded pairs (min_separation - required)
  let margin = Infinity;
  (p.pairs || []).forEach((pr) => {
    if (!pr.excluded) margin = Math.min(margin, pr.min_separation - pr.required);
  });
  const mBox = $("m-margin");
  mBox.textContent = isFinite(margin) ? margin.toFixed(4) : "—";
  mBox.style.color = !isFinite(margin) ? "" : margin < 0 ? "var(--red)" : margin < p.d ? "var(--amber)" : "var(--green)";
  renderManeuverTable(p);
}

function renderManeuverTable(p) {
  const rows = p.aircraft.map((a) => ({
    id: a.id,
    h0: a.heading_initial_deg,
    h1: a.heading_deg,
    dth: (a.theta || 0) * 180 / Math.PI,
    q: a.q != null ? a.q : 1,
    opt: a.option_label ?? "—",
    cost: a.maneuver_cost ?? 0,
  }));
  const { key, dir } = MAN_SORT;
  rows.sort((x, y) => (x[key] > y[key] ? 1 : x[key] < y[key] ? -1 : 0) * dir);
  const cols = [["id", "#"], ["h0", "init°"], ["h1", "new°"], ["dth", "Δθ°"],
    ["q", "q"], ["opt", "opt"], ["cost", "cost"]];
  const th = cols.map(([k, lbl]) => `<th data-k="${k}">${lbl}</th>`).join("");
  const trs = rows.map((r) => {
    const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v);
    return `<tr data-id="${r.id}">` + cols.map(([k]) => `<td>${fmt(r[k])}</td>`).join("") + "</tr>";
  }).join("");
  const t = $("man-table");
  t.innerHTML = `<tr>${th}</tr>${trs}`;
  t.querySelectorAll("th").forEach((h) => {
    h.onclick = () => {
      const k = h.dataset.k;
      MAN_SORT = { key: k, dir: MAN_SORT.key === k ? -MAN_SORT.dir : 1 };
      renderManeuverTable(p);
    };
  });
  t.querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.onclick = () => {
      SIM.highlightAircraft = SIM.highlightAircraft === +tr.dataset.id ? null : +tr.dataset.id;
      t.querySelectorAll("tr[data-id]").forEach((x) => x.classList.remove("sel"));
      if (SIM.highlightAircraft != null) tr.classList.add("sel");
      renderRadarAt(SIM.t);
    };
  });
}

function exportManeuverCSV() {
  const p = LAST_SOLUTION;
  if (!p) return;
  const header = ["aircraft", "initial_heading_deg", "new_heading_deg", "delta_theta_deg", "q", "option", "cost"];
  const lines = [header.join(",")];
  p.aircraft.forEach((a) => {
    lines.push([a.id, a.heading_initial_deg, a.heading_deg, (a.theta || 0) * 180 / Math.PI,
      a.q, a.option_label ?? "", a.maneuver_cost ?? ""].join(","));
  });
  downloadText(`maneuvers_${p.name}.csv`, lines.join("\n"));
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- benchmark tab --------------------------------------------------------
let charts = {};
async function loadBenchmark() {
  const data = await api.get("/api/campaign");
  const empty = $("bench-empty");
  if (!data.available || !data.points.length) {
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  const pts = data.points;

  // group by scenario for line charts (x = reps)
  const scs = [...new Set(pts.map((p) => p.scenario_id))];
  const palette = ["#36f2a0", "#ffcc4d", "#ff4d5e", "#4dd2ff", "#c08bff", "#ff9e4d", "#9effa0"];
  const mkSets = (yKey, opts = {}) =>
    scs.map((sc, k) => {
      const rows = pts.filter((p) => p.scenario_id === sc).sort((a, b) => a.reps - b.reps);
      return {
        label: opts.suffix ? `${sc} ${opts.suffix}` : sc,
        // null-safe: drop missing points rather than plotting them as 0
        data: rows
          .filter((r) => r[yKey] != null && isFinite(r[yKey]))
          .map((r) => ({ x: r.reps, y: opts.scale ? r[yKey] * opts.scale : r[yKey] })),
        borderColor: palette[k % palette.length],
        backgroundColor: palette[k % palette.length],
        borderDash: opts.dash || [],
        tension: 0.2,
        spanGaps: false,
      };
    });

  const opts = (yTitle) => ({
    responsive: true,
    scales: {
      x: { title: { display: true, text: "QAOA depth p" }, ticks: { color: "#6fa892" }, grid: { color: "#15463a" } },
      y: { title: { display: true, text: yTitle }, ticks: { color: "#6fa892" }, grid: { color: "#15463a" } },
    },
    plugins: { legend: { labels: { color: "#cdeee0", font: { family: "monospace" } } } },
  });

  charts.feas?.destroy();
  charts.feas = new Chart($("chart-feas"), {
    type: "line",
    data: { datasets: mkSets("feasible_rate", { scale: 100 }) },
    options: opts("feasibility %"),
  });
  charts.approx?.destroy();
  // dispersion: mean (solid) + best (dashed) approximation ratio per scenario
  charts.approx = new Chart($("chart-approx"), {
    type: "line",
    data: { datasets: [...mkSets("mean_approx_ratio", { suffix: "mean" }),
      ...mkSets("best_approx_ratio", { suffix: "best", dash: [4, 3] })] },
    options: opts("approx ratio (1=opt)"),
  });

  // table
  const cols = ["scenario_id", "n", "n_qubits", "reps", "feasible_rate", "mean_approx_ratio", "mean_circuit_depth", "classical_objective"];
  const head = ["Scenario", "n", "Qubits", "p", "Faisable %", "Approx", "Depth", "Classical obj"];
  const table = $("bench-table");
  table.replaceChildren();
  const headerRow = document.createElement("tr");
  head.forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    headerRow.appendChild(cell);
  });
  table.appendChild(headerRow);
  pts.forEach((p) => {
    const row = document.createElement("tr");
    cols.forEach((c) => {
      let v = p[c];
      let cls = "";
      if (c === "feasible_rate") { v = (v * 100).toFixed(0) + "%"; cls = p[c] >= 1 ? "good" : "warn"; }
      else if (c === "mean_approx_ratio") { cls = v != null && v <= 1.05 ? "good" : "warn"; v = v != null ? v.toFixed(2) : "—"; }
      else if (typeof v === "number") v = Number.isInteger(v) ? v : v.toFixed(2);
      const cell = document.createElement("td");
      cell.className = cls;
      cell.textContent = String(v);
      row.appendChild(cell);
    });
    table.appendChild(row);
  });
}

// ---- QUBO tab -------------------------------------------------------------
async function loadQubo() {
  const snap = _configSnapshot(); // pin the config this QUBO load is for
  const c = currentConfiguration;
  // keep the QUBO tab's K in sync with the current configuration
  const nt = c.n_theta;
  $("qb-ntheta").value = nt; $("qb-ntheta-val").textContent = nt;
  let q;
  try {
    q = await api.post("/api/qubo", {
      instance: snap.instance, subset: snap.subset, n_theta: snap.n_theta, n_q: snap.n_q,
    });
  } catch (e) {
    if (snap.id !== CONFIG_ID) return; // stale error for a superseded config
    LAST_QUBO = null; LAST_QUBO_CID = -1;
    $("qb-matrix-note").textContent = "✗ " + e.message;
    renderExplain();
    return;
  }
  if (snap.id !== CONFIG_ID) return; // a newer configuration superseded this QUBO
  LAST_QUBO = q; LAST_QUBO_CID = snap.id;
  $("qb-name").textContent = q.instance;
  $("qb-shape").textContent = `${q.n_aircraft} × ${q.grid_k}`;
  $("qb-qubits").textContent = q.n_qubits;
  $("qb-cterms").textContent = q.n_conflict_terms;
  $("qb-oterms").textContent = q.n_onehot_terms;
  $("qb-lpen").textContent = q.lambda_conflict.toFixed(2);
  $("qb-loh").textContent = q.lambda_onehot.toFixed(2);
  drawQMatrix(q);
  renderExplain();
}

function drawQMatrix(q) {
  const cv = $("qmatrix");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const note = $("qb-matrix-note");
  if (!q.matrix_included) {
    note.textContent = `matrix hidden (${q.n_qubits} qubits > display cap); metrics shown left.`;
    cv.onmousemove = null;
    cv.onmouseleave = null;
    return;
  }
  const defaultNote = `${q.n_qubits}×${q.n_qubits} Q matrix — hover a cell for its exact origin.`;
  note.textContent = defaultNote;
  const m = q.matrix, n = m.length, cell = Math.min(W, H) / n;
  // symmetric color scale around 0
  let amax = 1e-9;
  for (const row of m) for (const v of row) amax = Math.max(amax, Math.abs(v));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = m[i][j];
      if (v === 0) { ctx.fillStyle = "#06160f"; }
      else if (v > 0) { // penalties (conflict / one-hot) -> warm
        const t = Math.min(1, v / amax);
        ctx.fillStyle = `rgba(255,${Math.round(204 - t * 150)},${Math.round(77 - t * 60)},${0.25 + 0.75 * t})`;
      } else { // negative (one-hot linear) -> green
        const t = Math.min(1, -v / amax);
        ctx.fillStyle = `rgba(54,242,160,${0.2 + 0.7 * t})`;
      }
      ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }
  // grid lines every K to delimit aircraft blocks
  ctx.strokeStyle = "rgba(111,168,146,0.4)";
  for (let b = 0; b <= n; b += q.grid_k) {
    ctx.beginPath(); ctx.moveTo(b * cell, 0); ctx.lineTo(b * cell, n * cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, b * cell); ctx.lineTo(n * cell, b * cell); ctx.stroke();
  }
  const explanations = q.coefficient_explanations || { linear: [], quadratic: [] };
  const linear = new Map(explanations.linear.map((row) => [row.bit, row]));
  const quadratic = new Map(explanations.quadratic.map((row) => [`${row.a}:${row.b}`, row]));
  cv.onmousemove = (event) => {
    const rect = cv.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * n);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * n);
    if (row < 0 || col < 0 || row >= n || col >= n) return;
    if (row === col) {
      const term = linear.get(row);
      note.textContent = term
        ? `Q[${row},${col}] ${q.labels[row]} = ${term.coefficient.toFixed(4)} = cost ${term.maneuver_cost.toFixed(4)} − λ one-hot`
        : `Q[${row},${col}] = 0`;
    } else {
      const a = Math.min(row, col), b = Math.max(row, col);
      const term = quadratic.get(`${a}:${b}`);
      note.textContent = term
        ? `Q[${a},${b}] ${q.labels[a]} × ${q.labels[b]} = ${term.coefficient.toFixed(4)} (${term.kind} penalty)`
        : `Q[${a},${b}] ${q.labels[a]} × ${q.labels[b]} = 0 (no coupling)`;
    }
  };
  cv.onmouseleave = () => { note.textContent = defaultNote; };
}

// ---- Explain tab ----------------------------------------------------------
function _fmtGiB(value) {
  if (value == null || !isFinite(value)) return "—";
  if (value < 0.01) return `${(value * 1024).toFixed(2)} MiB`;
  return `${value.toFixed(value < 10 ? 2 : 1)} GiB`;
}

function _renderTable(id, headers, rows) {
  const table = $(id);
  table.replaceChildren();
  const trh = document.createElement("tr");
  headers.forEach((heading) => {
    const th = document.createElement("th");
    th.textContent = heading;
    trh.appendChild(th);
  });
  table.appendChild(trh);
  rows.forEach((values) => {
    const tr = document.createElement("tr");
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value == null ? "—" : String(value);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
}

function renderExplain() {
  // only surface resource estimates that belong to the current configuration
  const resources = (LAST_QUBO_CID === CONFIG_ID && LAST_QUBO?.resources)
    || (LAST_PREFLIGHT_CID === CONFIG_ID && LAST_PREFLIGHT?.resources)
    || LAST_SOLUTION?.solution?.explanation?.resources;
  if (resources) {
    $("ex-raw-q").textContent = `${resources.raw_qubits} qubits`;
    $("ex-raw-ram").textContent = _fmtGiB(resources.raw_statevector_gib);
    $("ex-safe-q").textContent = `${resources.max_safe_qubits} qubits / ${_fmtGiB(resources.safe_budget_gib)}`;
    $("ex-components").textContent = resources.n_exact_components;
    $("ex-peak-q").textContent = `${resources.largest_component_qubits} qubits`;
    $("ex-peak-ram").textContent = _fmtGiB(resources.decomposed_peak_gib);
    $("ex-reduction").textContent = resources.memory_reduction_factor != null
      ? `×${resources.memory_reduction_factor.toFixed(1)}`
      : `2^${resources.memory_reduction_log2}`;
    const solved = LAST_SOLUTION?.solution;
    $("ex-strategy").textContent = solved?.exact_component_decomposition
      ? `exact split · ${solved.n_components} circuits · max ${solved.largest_executed_circuit_qubits}q`
      : solved ? "monolithic" : "not run";
    const verdict = $("ex-capacity-verdict");
    const fits = resources.raw_fits_statevector || resources.decomposed_fits_statevector;
    verdict.className = `ex-verdict ${fits ? "ok" : "bad"}`;
    if (resources.raw_fits_statevector) {
      verdict.textContent = "✓ Raw statevector fits the conservative local RAM budget.";
    } else if (resources.exact_decomposition_available && resources.decomposed_fits_statevector) {
      verdict.textContent = "✓ Raw statevector is too large, but exact disconnected-component decomposition fits. This is a mathematical reduction, not a heuristic.";
    } else {
      verdict.textContent = "✗ Statevector exceeds the safe budget, including the largest exact component. Reduce K/aircraft, use hardware, or use MPS with explicit runtime caveats.";
    }
    _renderTable("ex-component-table", ["Composante", "Avions", "Qubits", "Vecteur d'état"],
      resources.components.map((c) => [
        `C${c.id}`, c.aircraft.map((a) => `A${a}`).join(", "), c.n_qubits, _fmtGiB(c.statevector_gib),
      ]));
  }

  const explanation = LAST_SOLUTION?.solution?.explanation;
  const energy = explanation?.energy;
  const encoding = explanation?.encoding;
  if (!energy || !encoding) {
    ["ex-e-cost", "ex-e-conflict", "ex-e-onehot", "ex-e-total", "ex-onehot",
      "ex-repair", "ex-feasible", "ex-dom-term", "ex-dom-ac", "ex-active-pairs"]
      .forEach((id) => { $(id).textContent = "—"; });
    $("ex-energy-check").className = "ex-verdict";
    $("ex-energy-check").textContent = "Run a solver to explain its result.";
    $("ex-proof").className = "ex-verdict";
    $("ex-proof").textContent = "Proof status — run a solver.";
    $("ex-proof-caveat").textContent = "";
    $("ex-bitstring").textContent = "No solved bitstring yet.";
    _renderTable("ex-bit-table", ["Avion", "Bits", "Actifs bruts", "Décodé", "q", "θ", "Coût", "Réparation"], []);
    _renderTable("ex-sample-table", ["Bitstring", "Probability", "Energy", "Raw valid", "Faisable", "Conflicts"], []);
    $("ex-samples-empty").classList.remove("hidden");
    return;
  }

  const num = (value) => Number(value).toFixed(6);
  $("ex-e-cost").textContent = num(energy.maneuver_cost);
  $("ex-e-conflict").textContent = `${num(energy.conflict_penalty)} (${energy.conflict_activations} active)`;
  $("ex-e-onehot").textContent = `${num(energy.onehot_penalty)} (residual ${energy.onehot_residual})`;
  $("ex-e-total").textContent = num(energy.total);
  const check = $("ex-energy-check");
  check.className = `ex-verdict ${energy.identity_verified ? "ok" : "bad"}`;
  check.textContent = energy.identity_verified
    ? "✓ Identity verified: displayed terms sum to the encoded QUBO energy."
    : `✗ Explanation mismatch: terms ${num(energy.total)} vs encoded ${num(energy.encoded_total)}.`;
  $("ex-onehot").textContent = encoding.raw_onehot_valid ? "yes ✓" : "no ⚠";
  $("ex-repair").textContent = encoding.repair_applied ? "yes — disclosed" : "no";
  $("ex-feasible").textContent = encoding.decoded_feasible ? "yes ✓" : `no (${encoding.decoded_conflicts} conflicts)`;

  // proof status + dominant contributors — strictly what the backend computed
  const proof = LAST_SOLUTION?.solution?.proof_status;
  const summary = explanation.summary;
  if (proof) {
    const strong = proof.status === "exact_optimum";
    $("ex-proof").className = `ex-verdict ${strong ? "ok" : "bad"}`;
    $("ex-proof").textContent = `${strong ? "✓" : "⚠"} Proof status: ${proof.label}`;
    $("ex-proof-caveat").textContent = proof.caveat || "";
  }
  if (summary) {
    $("ex-dom-term").textContent =
      `${summary.dominant_energy_term} (${Number(summary.dominant_energy_value).toFixed(5)})`;
    $("ex-dom-ac").textContent = summary.top_cost_aircraft != null
      ? `A${summary.top_cost_aircraft} (${Number(summary.top_cost_value).toFixed(5)})` : "—";
    $("ex-active-pairs").textContent = summary.active_conflict_pairs;
  }
  $("ex-bitstring").textContent = encoding.bitstring;
  _renderTable("ex-bit-table", ["Avion", "Bits", "Actifs bruts", "Décodé", "q", "θ", "Coût", "Réparation"],
    encoding.aircraft.map((row) => [
      `A${row.aircraft}`, `${row.bit_range[0]}…${row.bit_range[1]}`,
      row.active_options.length ? row.active_options.join(", ") : "none",
      `${row.decoded_option} · ${row.decoded_label}`, row.q.toFixed(4),
      row.theta.toFixed(4), row.cost.toFixed(6), row.repair || "none",
    ]));

  const samples = explanation.top_samples || [];
  _renderTable("ex-sample-table", ["Bitstring", "Probability", "Energy", "Raw valid", "Faisable", "Conflicts"],
    samples.map((row) => [
      row.bitstring, `${(100 * row.probability).toFixed(2)}%`, row.energy.toFixed(5),
      row.raw_onehot_valid ? "yes" : "no", row.decoded_feasible ? "yes" : "no",
      row.decoded_conflicts,
    ]));
  $("ex-samples-empty").classList.toggle("hidden", samples.length > 0);
}

// ---- init -----------------------------------------------------------------
// ===== Original AMPL / Gurobi baseline (read-only, per instance) =====
let AMPL_GUROBI = null;             // cached read-only payload (static campaign data)
let AMPL_GUROBI_STATE = "loading";  // loading | ready | error | absent

async function loadAmplGurobi() {
  AMPL_GUROBI_STATE = "loading";
  renderAmplGurobi();
  try {
    const p = await api.get("/api/ampl-gurobi");
    AMPL_GUROBI = p;
    AMPL_GUROBI_STATE = p && p.available ? "ready" : (p && p.error ? "error" : "absent");
  } catch (e) {
    AMPL_GUROBI = { error: String((e && e.message) || e) };
    AMPL_GUROBI_STATE = "error";
  }
  renderAmplGurobi();
  renderAlgorithmComparison();  // surface the Gurobi/discrete chain rows once loaded
  qvRenderComparison();          // AMPL may arrive after the QPU table's first render
}

function _agNum(x, kind) {
  if (typeof x !== "number" || !isFinite(x)) return "—";
  if (kind === "gap") return (x * 100).toPrecision(3) + " %";
  if (kind === "time") return x.toFixed(1) + " s";
  return x.toExponential(3);
}

// Renders the AMPL/Gurobi baseline for the CURRENT instance. Reads only the cached
// read-only payload; always shows the current instance (no stale result after a
// change), an honest empty/absent state, or an explicit error (never fake numbers).
function renderAmplGurobi() {
  const box = $("ampl-gurobi-box");
  if (!box) return;
  const badge = $("ag-badge"), qual = $("ag-qual"), metrics = $("ag-metrics"),
        table = $("ag-table"), explain = $("ag-explain");
  const reset = () => { qual.textContent = ""; metrics.innerHTML = ""; table.innerHTML = ""; explain.innerHTML = ""; };

  if (AMPL_GUROBI_STATE === "loading") {
    box.classList.remove("hidden"); reset();
    badge.className = "ag-badge ag-skeleton"; badge.textContent = "Loading AMPL/Gurobi baseline…";
    metrics.innerHTML = '<div class="ag-skel-line"></div><div class="ag-skel-line"></div>';
    return;
  }
  if (AMPL_GUROBI_STATE === "error") {
    box.classList.remove("hidden"); reset();
    badge.className = "ag-badge ag-error";
    badge.textContent = "AMPL/Gurobi baseline unavailable (artifact error)";
    explain.textContent = (AMPL_GUROBI && AMPL_GUROBI.error) ? String(AMPL_GUROBI.error) : "";
    return;
  }
  if (AMPL_GUROBI_STATE === "absent" || !AMPL_GUROBI || !AMPL_GUROBI.instances) {
    box.classList.add("hidden"); return;  // honest: nothing produced yet
  }
  const name = currentConfiguration.instance;
  // whole-instance only: the campaign never ran subset scenarios
  const e = currentConfiguration.subset ? null
    : AMPL_GUROBI.instances.find((x) => x.instance === name);
  if (!e) {
    box.classList.remove("hidden"); reset();
    badge.className = "ag-badge ag-skeleton";
    badge.textContent = currentConfiguration.subset
      ? "No AMPL/Gurobi baseline for a subset scenario (whole-instance only)"
      : "No AMPL/Gurobi baseline for this instance";
    return;
  }
  if (e.artifact_error) {  // corruption -> explicit error, never numbers
    box.classList.remove("hidden"); reset();
    badge.className = "ag-badge ag-error";
    badge.textContent = "Artifact error — " + e.artifact_error;
    return;
  }
  box.classList.remove("hidden"); reset();
  const anchor = e.is_official_anchor;
  badge.className = "ag-badge " + (anchor ? "ag-green" : "ag-orange");
  badge.textContent = anchor
    ? "Certified numerical anchor"
    : "Provisional feasible incumbent — not certified";
  qual.textContent = anchor
    ? (e.anchor_qualification || "")
    : (e.exclusion_reason || "not certified within the time budget");

  // This panel is PROVENANCE + sensitivity detail. The main comparison numbers
  // (objective, discretisation cost, algorithmic gaps) live in the unified RESULTS
  // table on the Explain tab.
  const mrow = (label, val, tip) =>
    `<div class="metric" title="${tip || ""}"><label>${label}</label><b>${val}</b></div>`;
  const m = e.measurement || {};
  metrics.innerHTML = [
    mrow("Protocol", e.protocol_id || "—", "frozen supervised AMPL/Gurobi protocol"),
    mrow("MP residual", _agNum(e.mp_residual),
      "measured max model-constraint residual; certified only when ≤ 1e-8"),
    mrow("Min safety margin", _agNum(e.minimum_safety_margin),
      "signed min(separation − d); a tiny NEGATIVE value is NOT a positive operational margin"),
    anchor
      ? mrow("Measurement Δq / Δθ",
          `${_agNum(m.max_abs_delta_q)} / ${_agNum(m.max_abs_delta_theta)}`,
          "warn-mode measurement reproduced the primary solution to this tolerance")
      : mrow("Optimality gap", _agNum(e.optimality_gap, "gap"),
          "relative MIP gap of the provisional incumbent"),
  ].join("");

  // K5/K7 discretisation SENSITIVITY only — never a reference for K3 algorithms.
  const sens = (e.discrete || []).filter((d) => d.K !== 3);
  if (sens.length) {
    const rows = sens.map((d) =>
      `<tr><td>${d.grid_key} <span class="ag-role">Sensitivity only</span></td>` +
      `<td>${_agNum(d.objective)}</td>` +
      `<td>${d.exhaustive_optimality_proven ? "proven" : "not proven"}</td></tr>`).join("");
    table.innerHTML =
      "<caption>Discretisation sensitivity (grid convergence toward the continuum — NOT algorithm references)</caption>" +
      "<tr><th>Grid</th><th>Objective</th><th>Optimality</th></tr>" + rows;
  } else {
    table.innerHTML = "";
  }

  explain.innerHTML =
    "<b>Provenance.</b> Gurobi solves the supervisor's <b>original continuous</b> NC_ACRP " +
    "model; the full comparison chain (continuous → discretisation cost → discrete K3 → " +
    "algorithmic gap → QUBO/SA/QAOA) is in the <b>Explain</b> tab's results table. K5/K7 " +
    "above are <b>sensitivity only</b> — they show the grid converging toward the continuum " +
    "and are never used as the reference for the K3-grid algorithms.";
}

async function init() {
  // capabilities banner + disable quantum if absent
  const health = await api.get("/api/health");
  const caps = [`classical: ✓`];
  caps.push(`quantum: ${health.quantum_available ? "✓" : "✗"}`);
  if (health.ibm_available) caps.push("IBM: ✓");
  if (health.dwave_available) caps.push(`D-Wave: ${health.dwave_qpu_available ? "QPU✓" : "SA✓"}`);
  $("caps").textContent = caps.join("  ·  ");
  const disable = (val, cond) => {
    const o = $("solver").querySelector(`option[value="${val}"]`);
    if (o) o.disabled = cond;
  };
  disable("qaoa", !health.quantum_available);
  disable("dwave", !health.dwave_available);
  disable("dwave-qpu", !health.dwave_qpu_available);

  // instances + scenarios
  const { instances } = await api.get("/api/instances");
  $("instance").replaceChildren(...instances.map((n) => new Option(n, n)));
  const { scenarios } = await api.get("/api/scenarios");
  scenarios.forEach((s) => (SCENARIOS[s.id] = s));
  $("scenario").replaceChildren(
    new Option("— custom —", ""),
    ...scenarios.map(
      (s) => new Option(`${s.id} · ${s.instance_name} (~${s.expected_qubits}q)`, s.id)
    )
  );

  // default to a small interesting case
  const defaultInstance = instances.includes("CP_4") ? "CP_4" : instances[0];
  $("instance").value = defaultInstance;
  currentConfiguration.instance = defaultInstance;
  currentConfiguration.n_theta = parseInt($("ntheta").value, 10);

  // wire controls
  $("solver").onchange = () => {
    $("qaoa-opts").classList.toggle("hidden", $("solver").value !== "qaoa");
    updateManualApplicability();
  };
  $("mode").onchange = () => updateManualApplicability();
  $("qb-ntheta").oninput = () => {
    $("qb-ntheta-val").textContent = $("qb-ntheta").value;
    currentConfiguration.scenario = "";
    currentConfiguration.n_theta = parseInt($("qb-ntheta").value, 10);
    invalidateDerivedState();
    refreshGridLimit().catch(showError);
    debouncedInstanceCatalog();
    debouncedPreview();
  };
  $("scenario").onchange = () => {
    const v = $("scenario").value;
    if (v) selectScenario(v);
  };
  $("instance").onchange = () => selectCustomInstance($("instance").value);
  $("reps").oninput = () => ($("reps-val").textContent = $("reps").value);
  $("ntheta").oninput = () => {
    $("ntheta-val").textContent = `K${$("ntheta").value}`;
    // editing K makes it a custom configuration; keep everything in sync
    currentConfiguration.scenario = "";
    currentConfiguration.n_theta = parseInt($("ntheta").value, 10);
    invalidateDerivedState();
    refreshGridLimit().catch(showError);
    debouncedInstanceCatalog();
    debouncedPreview();
  };
  $("solve-btn").onclick = solve;
  $("reset-btn").onclick = () => showInstance($("instance").value);
  $("explain-run-btn").onclick = () => switchTab("explain");
  $("profile").onchange = () => {
    const scientific = $("profile").value === "scientific";
    $("seeds-row").classList.toggle("hidden", !scientific);
    $("save-run-btn").classList.toggle("hidden", !scientific);
    $("solve-btn").textContent = scientific ? "▶ RUN SCIENTIFIC (seeds)" : "▶ RUN SELECTED ONLY";
    if (!scientific) { $("sci-box").classList.add("hidden"); $("save-run-note").textContent = ""; }
  };
  $("save-run-btn").onclick = async () => {
    const btn = $("save-run-btn");
    btn.disabled = true;
    clearError();
    try { await runScientific(_solveBody(), true); }
    catch (e) { showError(e); }
    finally { btn.disabled = false; }
  };

  // before/after + maneuver CSV
  const setTraj = (initial, continuous, discrete) => {
    Object.assign(SIM.toggles, {initial, continuous, discrete});
    $("tg-initial").checked = initial;
    $("tg-continuous").checked = continuous;
    $("tg-discrete").checked = discrete;
    renderRadarAt(SIM.t);
  };
  $("ba-before").onclick = () => setTraj(true, false, false);
  $("ba-both").onclick = () => setTraj(true, true, true);
  $("ba-after").onclick = () => setTraj(false, true, true);
  $("man-csv").onclick = exportManeuverCSV;
  setupExports();

  // error recovery
  $("err-retry").onclick = () => { clearError(); solve(); };
  $("err-safe").onclick = () => {
    clearError();
    $("solver").value = "reference";
    $("qaoa-opts").classList.add("hidden");
    selectCustomInstance(defaultInstance);
    $("status").textContent = "returned to safe configuration";
  };

  // timeline controls
  $("sim-play").onclick = () => (SIM.playing ? pauseSim() : playSim());
  $("sim-reset").onclick = resetSim;
  $("sim-back").onclick = () => stepSim(-1);
  $("sim-fwd").onclick = () => stepSim(1);
  $("sim-slider").oninput = (e) => { pauseSim(); seekSim((e.target.value / 1000) * SIM.tMax); };
  $("sim-speed").onchange = (e) => (SIM.speed = parseFloat(e.target.value));

  // layer toggles
  const rerender = () => renderRadarAt(SIM.t);
  $("tg-motion").onchange = (e) => { SIM.toggles.motion = e.target.value; rerender(); };
  for (const [id, key] of [["tg-initial", "initial"], ["tg-continuous", "continuous"],
    ["tg-discrete", "discrete"], ["tg-labels", "labels"], ["tg-vectors", "vectors"],
    ["tg-sep", "sep"], ["tg-cpa", "cpa"], ["tg-trails", "trails"]]) {
    $(id).onchange = (e) => { SIM.toggles[key] = e.target.checked; rerender(); };
  }

  // keyboard shortcuts (space/R/arrows) — active on the Radar tab
  // async jobs
  $("job-cancel").onclick = cancelJob;
  reconnectJob();  // resume a job that was running before a page reload

  // multi-solver comparison
  $("compare-btn").onclick = () => runCompare().catch(showError);
  $("ex-rerun-all").onclick = () => runCompare().catch(showError);
  $("ex-cancel-all").onclick = () => cancelAutoComparison().catch(showError);
  $("cmp-json").onclick = () => exportComparison("json");
  $("cmp-csv").onclick = () => exportComparison("csv");
  $("cmp-html").onclick = () => exportComparison("html");
  $("cmp-close").onclick = () => $("cmp-modal").classList.add("hidden");
  const explSel = $("expl-method");
  if (explSel) explSel.onchange = () => renderDemoExplanation();
  updateExportButtons();

  qvInit();  // wire the QPU validation tab (Phase 8)

  // presentation mode
  $("experience-toggle").onclick = () => {
    setExperienceMode(!document.body.classList.contains("expert-mode"));
  };
  $("demo-results-btn").onclick = () => runCompare().catch(showError);
  $("present-btn").onclick = enterPresent;
  $("ps-next").onclick = presentNext;
  $("ps-prev").onclick = presentPrev;
  $("ps-exit").onclick = exitPresent;

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || (e.target.tagName === "INPUT" && e.target.type !== "range")) return;
    const presenting = document.body.classList.contains("presenting");
    const k = e.key.toLowerCase();
    if (k === "f") { e.preventDefault(); if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.(); return; }
    if (presenting && (e.key === "ArrowRight" || k === "n")) { e.preventDefault(); presentNext(); return; }
    if (presenting && (e.key === "ArrowLeft" || k === "p")) { e.preventDefault(); presentPrev(); return; }
    if (k === "escape" && presenting) { exitPresent(); return; }
    // radar-tab shortcuts
    if (!$("tab-radar").classList.contains("active")) return;
    if (e.key === " ") { e.preventDefault(); SIM.playing ? pauseSim() : playSim(); }
    else if (k === "r") resetSim();
    else if (k === "b") {
      const onlyInitial = SIM.toggles.initial && !SIM.toggles.continuous && !SIM.toggles.discrete;
      setTraj(!onlyInitial, onlyInitial, onlyInitial);
    }
    else if (k === "c") { SIM.toggles.cpa = !SIM.toggles.cpa; $("tg-cpa").checked = SIM.toggles.cpa; renderRadarAt(SIM.t); }
    else if (e.key === "ArrowRight") stepSim(1);
    else if (e.key === "ArrowLeft") stepSim(-1);
  });

  // redraw on resize (HiDPI + responsive canvas)
  window.addEventListener("resize", () => { if (SIM.payload) renderRadarAt(SIM.t); });

  setExperienceMode(false);
  debouncedInstanceCatalog();
  loadAmplGurobi();  // fetch the read-only AMPL/Gurobi baseline once (non-blocking)
  await showInstance($("instance").value);
}

init().catch((e) => ($("status").textContent = "✗ init: " + e.message));

// ===== QPU Validation tab (Phase 8) =====
let QV_EPOCH = 0;
let QV_DECISION = null, QV_DECISION_E = -1;
let QV_DRYRUN = null, QV_DRYRUN_E = -1;
let QV_PREPARED = null, QV_PREPARED_E = -1;
let QV_CHALLENGE = null;
let QV_RUN_ID = null;

// Provider-job evidence for the currently displayed QPU result. A run WITHOUT an
// ibm_job_id is a local artefact and must never be labelled real hardware
// (MoE Expert D, CRITICAL-1).
function _qvHasJobEvidence() {
  const ids = QV_QPU_RESULT?.provenance?.ibm_job_ids;
  return Array.isArray(ids) && ids.length > 0;
}
function _qvBackendLabel() {
  return QV_QPU_RESULT?.provenance?.backend || "backend inconnu";
}
let QV_FLAGS = null;
let QV_LIVE_SNAPSHOT = null;
let QV_BASELINES = [], QV_BASELINES_E = -1;
let QV_LOCAL_JOB_ID = null;
let QV_QPU_RESULT = null;
let QV_MONITOR_TIMER = null;

function qvBackend() {
  return QV_FLAGS && QV_FLAGS.runtime_factory_enabled ? "ibm_marrakesh" : "fake_lagos";
}

function qvBump() {
  QV_EPOCH++;
  QV_DECISION = null; QV_DECISION_E = -1;
  QV_DRYRUN = null; QV_DRYRUN_E = -1;
  QV_PREPARED = null; QV_PREPARED_E = -1;
  QV_CHALLENGE = null;
  if (QV_LOCAL_JOB_ID) { _cancelJob(QV_LOCAL_JOB_ID); QV_LOCAL_JOB_ID = null; }
  QV_BASELINES = []; QV_BASELINES_E = -1; QV_QPU_RESULT = null;
  const stale = $("qv-stale-banner");
  if (stale && QV_RUN_ID) stale.classList.remove("hidden");
  qvRenderComparison();  // immediately bind the continuous row to the new instance
  qvRender();
}

function qvInstance() { return $("qv-instance").value; }
function qvSnapshots() {
  if (!QV_LIVE_SNAPSHOT) throw new Error("Load the backend snapshot first");
  return [QV_LIVE_SNAPSHOT];
}
function qvBody() {
  const real = qvBackend().startsWith("ibm_");
  return { instance: qvInstance(), n_theta: real ? 3 : 2, n_q: 1,
    shots: 512, transpiler_seed: 1,
    backend_requested: qvBackend(), gammas: [parseFloat($("qv-gamma").value)],
    betas: [parseFloat($("qv-beta").value)], max_jobs: 1, max_quantum_seconds: 20.0,
    optimization_level: 0, snapshots: qvSnapshots() };
}

async function qvLoadLiveSnapshot() {
  if (!QV_FLAGS || !QV_FLAGS.runtime_factory_enabled) {
    QV_LIVE_SNAPSHOT = { name: "fake_lagos", region: "us-east", family: "Falcon",
      physical_qubits: 7, programmable_qubits: 7, operational: true, maintenance: false,
      pending_jobs: 3, two_qubit_error: 0.006, readout_error: 0.012,
      clops: 2500.0, avg_coupling_degree: 2.0, data_ts: 0.0 };
    $("qv-snapshot-list").textContent = "fake_lagos · local test backend";
    return QV_LIVE_SNAPSHOT;
  }
  const r = await api.get(`/api/qpu/backends/live/${encodeURIComponent(qvBackend())}`);
  QV_LIVE_SNAPSHOT = r.backend;
  $("qv-snapshot-list").textContent =
    `${r.backend.name} · ${r.backend.programmable_qubits} qubits · ` +
    `${r.backend.pending_jobs ?? "?"} pending · live IBM snapshot`;
  return r.backend;
}

async function qvRefreshFlags() {
  QV_FLAGS = await api.get("/api/qpu/flags");
  const badge = $("qv-badge-mode");
  if (!QV_FLAGS.qpu_enabled) {
    badge.textContent = "QPU DISABLED · OFFLINE — NO SUBMISSION"; badge.dataset.kind = "dry-run";
    $("qv-disabled-note").classList.remove("hidden");
  } else if (QV_FLAGS.dry_run_only || !QV_FLAGS.submission_allowed) {
    badge.textContent = "DRY-RUN ONLY · OFFLINE — NO SUBMISSION"; badge.dataset.kind = "dry-run";
    $("qv-disabled-note").classList.add("hidden");
  } else {
    badge.textContent = "REAL QPU ENABLED"; badge.dataset.kind = "real-qpu";
    $("qv-disabled-note").classList.add("hidden");
  }
  $("qv-flags-note").textContent =
    `qpu_enabled=${QV_FLAGS.qpu_enabled} · dry_run_only=${QV_FLAGS.dry_run_only} · submission_allowed=${QV_FLAGS.submission_allowed}`;
  qvRender();
}

async function qvResumeLatest() {
  const listing = await api.get("/api/qpu/runs");
  const allRuns = listing.runs || [];
  const latest = allRuns.find((run) => run.has_remote_job) || allRuns[0];
  if (!latest) return;
  if (latest.instance && Array.from($("qv-instance").options)
      .some((option) => option.value === latest.instance)) {
    $("qv-instance").value = latest.instance;
  }
  QV_RUN_ID = latest.run_id;
  $("qv-run-id").textContent = latest.run_id;
  await qvRefreshStatus();
}

async function qvAssess() {
  const e = QV_EPOCH;
  await qvLoadLiveSnapshot();
  const decision = await api.post("/api/qpu/backends/assess",
    { logical_qubits: qvBackend().startsWith("ibm_") ? 9 : 6,
      interaction_density: 0.4, shots: 512, qaoa_reps: 1,
      snapshots: qvSnapshots() });
  if (e !== QV_EPOCH) return;
  QV_DECISION = decision; QV_DECISION_E = e;
  const rows = (decision.ranking || []).map((r) => [
    String(r.rank), r.name, String(r.applicable),
    r.total_score == null ? "—" : r.total_score.toFixed(3),
    r.explanation ? String(r.explanation.heuristic_warning || "").slice(0, 40) : ""]);
  _renderTable("qv-ranking-table", ["rank", "backend", "applicable", "score", "note"], rows);
  $("qv-selected-backend").textContent = decision.selected || ("withheld: " + (decision.selection_note || ""));
  const real = qvBackend().startsWith("ibm_");
  $("qv-badge-selected").textContent = real ? "real IBM backend" : "fake backend";
  $("qv-badge-selected").dataset.kind = real ? "real-qpu" : "fake-backend";
  await qvRunLocalBaselines(e);
  qvRender();
}

async function qvRunLocalBaselines(epoch) {
  const config = { instance: qvInstance(), subset: null, n_theta: 3, n_q: 1 };
  $("qv-decision-verdict").textContent = "Exécution des références locales sur le même problème CP/K3…";
  const pf = await api.post("/api/preflight", config);
  if (epoch !== QV_EPOCH) return;
  QV_BASELINES = pf.methods.map((m) => ({
    solver: m.method, mode: m.mode, label: m.label,
    state: m.applicable ? "pending" : "skipped", reason: m.applicable ? "" : m.reason,
  }));
  qvRenderComparison();
  for (const row of QV_BASELINES) {
    if (epoch !== QV_EPOCH) return;
    if (row.state === "skipped") continue;
    row.state = "running"; qvRenderComparison();
    try {
      const sub = await api.post("/api/jobs", {
        ...config, solver: row.solver, mode: row.mode || "aer_sim",
        reps: 1, maxiter: 30, shots: 512, profile: "demo",
      });
      QV_LOCAL_JOB_ID = sub.job_id;
      const started = Date.now();
      for (;;) {
        if (epoch !== QV_EPOCH) return;
        const status = await api.get(`/api/jobs/${sub.job_id}`);
        if (status.state === "completed") {
          row.payload = await api.get(`/api/jobs/${sub.job_id}/result`);
          row.state = "ok";
          break;
        }
        if (["failed", "cancelled"].includes(status.state)) {
          row.state = "error"; row.reason = status.error || status.state;
          break;
        }
        if (Date.now() - started > AUTO_POLL_TIMEOUT_MS) {
          _cancelJob(sub.job_id);
          row.state = "error"; row.reason = "local baseline exceeded 120 s";
          break;
        }
        await _sleep(AUTO_POLL_MS);
      }
    } catch (err) {
      row.state = "error"; row.reason = err.message;
    } finally {
      QV_LOCAL_JOB_ID = null;
    }
    qvRenderComparison();
  }
  if (epoch !== QV_EPOCH) return;
  QV_BASELINES_E = epoch;
  $("qv-decision-verdict").textContent =
    "Local baselines complete on CP/K3. Marrakesh preparation is now unlocked.";
  qvRenderComparison();
  qvRender();
}

function qvRenderComparison() {
  const ref = QV_BASELINES.find((r) => r.solver === "reference" && r.state === "ok");
  const refObj = ref?.payload?.solution?.objective ?? null;
  const rows = [];
  const continuous = AMPL_GUROBI?.instances?.find((r) => r.instance === qvInstance());
  if (continuous && continuous.original_ampl_gurobi_objective != null) {
    const certified = continuous.is_official_anchor === true;
    rows.push(["Original AMPL / Gurobi", "local CPU · continuous",
      _cf(continuous.original_ampl_gurobi_objective),
      continuous.feasible_common_numeric ? "yes" : "no",
      _ct(continuous.wall_time_s), "—",
      certified ? "certified numerical anchor" : (continuous.badge || "provisional")]);
  }
  rows.push(...QV_BASELINES.map((r) => {
    const s = r.payload?.solution;
    const delta = s?.feasible && refObj != null ? s.objective - refObj : null;
    return [r.label, "local CPU", s ? _cf(s.objective) : "—",
      s ? (s.feasible ? "yes" : "no") : "—", _ct(s?.wall_time_s),
      delta == null ? "—" : _cs(delta),
      r.state === "ok" ? (s.feasible ? "feasible" : "infeasible")
        : r.state + (r.reason ? `: ${r.reason}` : "")];
  }));
  if (QV_QPU_RESULT?.available
      && QV_QPU_RESULT.run_id === QV_RUN_ID
      && QV_QPU_RESULT.provenance?.instance === qvInstance()) {
    const s = QV_QPU_RESULT.summary;
    const best = s.best_feasible;
    const delta = best && refObj != null ? best.objective - refObj : null;
    rows.push([`QAOA IBM ${_qvBackendLabel()}`, _qvHasJobEvidence() ? "matériel réel enregistré" : "artefact local (non vérifié)", best ? _cf(best.objective) : "—",
      best ? "oui" : "non", "temps fournisseur", delta == null ? "—" : _cs(delta),
      best ? `terminé · faisabilité ${(100 * s.feasibility_rate).toFixed(1)} %`
        : "completed · no feasible sample"]);
  }
  _renderTable("qv-comparison-table",
    ["Méthode", "Compute", "Objectif", "Faisable", "Temps", "Δ vs discrete K3", "Statut"], rows);
}

async function qvDryRun() {
  const e = QV_EPOCH;
  const r = await api.post("/api/qpu/dry-run", qvBody());
  if (e !== QV_EPOCH) return;
  QV_DRYRUN = r; QV_DRYRUN_E = e;
  $("qv-dryrun-jobs").textContent = String((r.would_submit || []).length);
  $("qv-dryrun-submittable").textContent = String(r.submittable);
  const ul = $("qv-dryrun-warnings"); ul.textContent = "";
  (r.warnings || []).forEach((w) => { const li = document.createElement("li"); li.textContent = w; ul.appendChild(li); });
  qvRender();
}

async function qvPrepare() {
  const e = QV_EPOCH;
  const r = await api.post("/api/qpu/prepare", qvBody());
  if (e !== QV_EPOCH) return;
  QV_PREPARED = r; QV_PREPARED_E = e; QV_RUN_ID = r.run_id; QV_CHALLENGE = r.confirmation;
  // A result belongs to one immutable run, not merely to an instance. Starting
  // another run for the same CP instance must never leave the previous IBM row
  // looking like the outcome of the new queued/submitted run.
  QV_QPU_RESULT = null;
  qvRenderComparison();
  $("qv-stale-banner").classList.add("hidden");
  $("qv-run-id").textContent = r.run_id;
  $("qv-artefact-hash").textContent = r.artefact_sha256.slice(0, 22) + "…";
  $("qv-budget").textContent = `${r.budget.max_jobs}/${r.budget.total_shots}/${r.budget.max_quantum_seconds}`;
  $("qv-consent-phrase").textContent =
    "Prepared backend, hashes and budget are bound to this confirmation. No job is submitted yet.";
  await qvRefreshStatus();
  qvRender();
}

async function qvConfirm() {
  if (!QV_PREPARED || QV_PREPARED_E !== QV_EPOCH) { showError(new Error("configuration changed — re-prepare first")); return; }
  const p = QV_PREPARED;
  const body = { nonce: QV_CHALLENGE.nonce, config_hash: p.config_hash, artefact_sha256: p.artefact_sha256,
    decision_sha256: p.decision_sha256, backend: p.backend, total_shots: p.budget.total_shots,
    max_jobs: p.budget.max_jobs, max_quantum_seconds: p.budget.max_quantum_seconds,
    consent: QV_CHALLENGE.consent_phrase };
  const r = await api.post(`/api/qpu/${p.run_id}/confirm`, body);
  const badge = $("qv-badge-confirm"); badge.textContent = r.state; badge.dataset.kind = "dry-run";
  qvUpdateSubmitButton(r.submission_allowed);
  await qvRefreshStatus();
}

// The submit button is shown only when the server allows submission AND the
// prepared artefact is genuinely real-submittable. A fake/dry-run artefact keeps
// it hidden AND disabled.
function qvUpdateSubmitButton(submissionAllowed) {
  const sb = $("qv-submit-btn");
  const realSubmittable = !!(QV_PREPARED && QV_PREPARED.submittable === true);
  if (submissionAllowed && realSubmittable) {
    sb.classList.remove("hidden");
    sb.disabled = false;
    $("qv-consent-phrase").dataset.note = "";
  } else {
    sb.classList.add("hidden");
    sb.disabled = true;
    if (submissionAllowed) {
      $("qv-result-explain").textContent =
        "Submit stays disabled: this artefact is a fake/dry-run (not real-submittable).";
    }
  }
}

let QV_SUBMITTING = false;
async function qvSubmit() {
  if (QV_SUBMITTING) return;                      // double-click guard
  if (!QV_PREPARED || QV_PREPARED_E !== QV_EPOCH) { showError(new Error("stale — re-prepare")); return; }
  const p = QV_PREPARED;
  const ok = window.confirm(
    `REAL IBM QPU submission\nbackend: ${p.backend}\nshots: ${p.budget.total_shots}\n` +
    `jobs: ${p.budget.max_jobs}\nbudget: ${p.budget.max_quantum_seconds}s\n` +
    `artefact: ${p.artefact_sha256.slice(0, 18)}…\n\n` +
    `This may incur cost and queue time. No optimality is guaranteed. Proceed?`);
  if (!ok) return;
  QV_SUBMITTING = true;
  $("qv-submit-btn").disabled = true;
  try {
    const r = await api.post(`/api/qpu/${p.run_id}/submit`, {});
    $("qv-run-state").textContent = r.state;
    await qvRefreshStatus();
  } catch (e) {
    // a fake/guarded run is refused by the server (409/403/503) — show it plainly
    showError(e);
  } finally {
    QV_SUBMITTING = false;
  }
}

async function qvRefreshStatus() {
  if (!QV_RUN_ID) return;
  if (QV_MONITOR_TIMER) { clearTimeout(QV_MONITOR_TIMER); QV_MONITOR_TIMER = null; }
  let run = await api.get(`/api/qpu/${QV_RUN_ID}`);
  if (QV_QPU_RESULT && QV_QPU_RESULT.run_id !== QV_RUN_ID) {
    QV_QPU_RESULT = null;
    qvRenderComparison();
  }
  const terminal = ["completed", "failed", "cancelled"].includes(run.state);
  if (!terminal && (run.ibm_job_ids || []).length) {
    await api.post(`/api/qpu/${QV_RUN_ID}/reconcile`, {});
    run = await api.get(`/api/qpu/${QV_RUN_ID}`);
  }
  $("qv-run-state").textContent = run.state;
  if (!terminal && (run.ibm_job_ids || []).length) {
    $("qv-result-explain").textContent =
      `${run.state.toUpperCase()} on IBM · job ${run.ibm_job_ids[0]} · monitoring every 15 s`;
  }
  const ev = await api.get(`/api/qpu/${QV_RUN_ID}/events`);
  const rows = (ev.events || []).map((x) => [
    String(x.seq), x.event_kind, (x.payload && x.payload.state) || "", x.stamp_utc || ""]);
  _renderTable("qv-events-table", ["seq", "kind", "state", "stamp"], rows);
  if (run.state === "completed") {
    const result = await api.get(`/api/qpu/${QV_RUN_ID}/result`);
    if (result.available) {
      QV_QPU_RESULT = result;
      const s = result.summary;
      const best = s.best_feasible;
      // A result is "matériel quantique réel" ONLY with provider job-id evidence.
      // Without it this is a local artefact: badge and wording must say so
      // (MoE Expert D, CRITICAL-1 — the badge previously ignored the job ids).
      const prov = result.provenance || {};
      const hasJob = Array.isArray(prov.ibm_job_ids) && prov.ibm_job_ids.length > 0;
      const bk = prov.backend || "backend inconnu";
      $("qv-badge-provenance").textContent = hasJob
        ? `Matériel quantique réel enregistré · ${bk} · job ${prov.ibm_job_ids[0]}`
        : "Artefact local — aucune preuve de job IBM";
      $("qv-badge-provenance").dataset.kind = hasJob ? "real-qpu" : "unverified";
      const head = hasJob ? `${bk} (counts enregistrés)` : `${bk} — SANS preuve de job IBM`;
      $("qv-result-explain").textContent = best
        ? `${head} : ${s.n_shots} shots · faisabilité ${(100 * s.feasibility_rate).toFixed(1)} % · `
          + `meilleure solution faisable ${best.bitstring}, objectif ${best.objective.toFixed(6)}`
        : `${head} : ${s.n_shots} shots · aucun échantillon faisable`;
      qvRenderComparison();
    } else {
      $("qv-result-explain").textContent = result.note || "Completed; result not persisted yet.";
    }
  }
  if (!["completed", "failed", "cancelled"].includes(run.state)
      && (run.ibm_job_ids || []).length) {
    QV_MONITOR_TIMER = setTimeout(
      () => qvRefreshStatus().catch(showError), 15000);
  }
}

async function qvCancel() {
  if (!QV_RUN_ID) return;
  const r = await api.post(`/api/qpu/${QV_RUN_ID}/cancel`, {});
  $("qv-run-state").textContent = r.state;
}

async function qvExport() {
  if (!QV_RUN_ID) { showError(new Error("prepare a run first")); return; }
  const exp = await api.post(`/api/qpu/${QV_RUN_ID}/export`, {});
  downloadText(`qpu-export-${QV_RUN_ID}.json`, JSON.stringify(exp, null, 2));
  $("qv-result-explain").textContent = "Export written and verified (see downloaded JSON).";
}

function qvRender() {
  const enabled = QV_FLAGS ? QV_FLAGS.qpu_enabled : false;
  const set = (id, dis) => { const el = $(id); if (el) el.disabled = dis; };
  set("qv-assess-btn", false);
  set("qv-dryrun-btn", QV_DECISION_E !== QV_EPOCH || QV_BASELINES_E !== QV_EPOCH);
  set("qv-prepare-btn", !enabled || QV_DRYRUN_E !== QV_EPOCH
    || QV_BASELINES_E !== QV_EPOCH);
  set("qv-confirm-btn", !enabled || QV_PREPARED_E !== QV_EPOCH);
}

function qvInit() {
  const src = $("instance"), dst = $("qv-instance");
  if (src && dst && !dst.options.length) dst.innerHTML = src.innerHTML;
  const bind = (id, fn) => { const el = $(id); if (el) el.onclick = () => fn().catch(showError); };
  bind("qv-assess-btn", qvAssess);
  bind("qv-dryrun-btn", qvDryRun);
  bind("qv-prepare-btn", qvPrepare);
  bind("qv-confirm-btn", qvConfirm);
  bind("qv-submit-btn", qvSubmit);
  bind("qv-refresh-btn", qvRefreshStatus);
  bind("qv-cancel-btn", qvCancel);
  bind("qv-export-json", qvExport);
  ["qv-instance", "qv-gamma", "qv-beta"].forEach((id) => { const el = $(id); if (el) el.addEventListener("change", qvBump); });
  qvRefreshFlags().then(qvResumeLatest).catch(showError);
}
