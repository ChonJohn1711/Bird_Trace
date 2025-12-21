/* global L */
const $ = (id) => document.getElementById(id);

// --- UI Elements ---
const statusBadge = $("statusBadge");
const dtInput = $("dtInput");
const modelSelect = $("modelSelect");
const btnPredict = $("btnPredict");
const predictSpinner = $("predictSpinner");
const btnPlay = $("btnPlay");
const btnPause = $("btnPause");
const btnResetView = $("btnResetView");
const speedRange = $("speedRange");
const speedLabel = $("speedLabel");
const followToggle = $("followToggle");
const notesEl = $("notes");
const datasetInfoEl = $("datasetInfo");
const predTableBody = $("predTable").querySelector("tbody");
const btnDownload = $("btnDownload");
const btnDownloadXlsx = $("btnDownloadXlsx");
const tabNotes = $("tabNotes");
const tabTable = $("tabTable");
const panelNotes = $("panelNotes");
const panelTable = $("panelTable");
const nowLine = $("nowLine");
const btnTogglePanel = $("btnTogglePanel");
const sidePanel = $("sidePanel");
const toastHost = $("toastHost");
const cardKpi = $("cardKpi");
const btnToggleKpi = $("btnToggleKpi");

// KPIs
const kpiModel = $("kpiModel");
const kpiHistPts = $("kpiHistPts");
const kpiPredPts = $("kpiPredPts");
const kpiTotalDist = $("kpiTotalDist");
const kpiAvgStep = $("kpiAvgStep");

// --- Map / Track Colors ---
const HIST_COLOR = "#3b82f6"; // lịch sử
const PRED_COLOR = "#16a34a"; // dự đoán (xanh lá đậm)
let animationBoundary = 0;

// --- Playback ---
const BASE_TICK_MS = 40; // 1x
let tickMs = BASE_TICK_MS;
let lastPanMs = 0;

// --- Leaflet ---
let map;
let baseTile;
let histLayer;
let predLayer;
let marker;

let lastFitBounds = null;

let animationTimer = null;
let animationIndex = 0;

// Dùng danh sách điểm đã "làm dày" để marker di chuyển mượt theo đường nối.
// Mỗi phần tử: { latlng:[lat,lon], isPred:boolean, timestampMs?:number }
let animationPath = [];

function setActiveTab(which) {
  const isNotes = which === "notes";
  tabNotes.classList.toggle("active", isNotes);
  tabTable.classList.toggle("active", !isNotes);
  panelNotes.classList.toggle("hidden", !isNotes);
  panelTable.classList.toggle("hidden", isNotes);
}

function setStatus(text, kind = "info") {
  statusBadge.textContent = text;
  statusBadge.style.borderColor = kind === "ok" ? "rgba(110,231,183,0.4)" : (kind === "err" ? "rgba(251,191,36,0.45)" : "rgba(255,255,255,0.12)");
  statusBadge.style.color = kind === "ok" ? "#c6f6df" : (kind === "err" ? "#fde68a" : "#a8b0c6");
}

function showToast(title, body, kind = "ok", ttlMs = 2800) {
  if (!toastHost) return;
  const el = document.createElement("div");
  el.className = `toast ${kind === "err" ? "err" : "ok"}`;
  el.innerHTML = `
    <div class="toastTitle">${escapeHTML(title || "")}</div>
    <div class="toastBody">${escapeHTML(body || "")}</div>
  `;
  toastHost.appendChild(el);
  setTimeout(() => {
    try { el.remove(); } catch (_) { }
  }, ttlMs);
}

function escapeHTML(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function tickMsFromSpeed(x) {
  const v = Number(x);
  if (!Number.isFinite(v) || v <= 0) return BASE_TICK_MS;
  const ms = Math.round(BASE_TICK_MS / v);
  return Math.max(15, Math.min(120, ms));
}

function fmtTimestampMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return "";
  try { return new Date(ms).toISOString().replace("T", " ").slice(0, 19); } catch (_) { return ""; }
}

function updateNowBox(item) {
  if (!nowLine) return;
  if (!item) {
    nowLine.textContent = "—";
    return;
  }
  const phase = item.isPred ? "predict" : "history";
  const ts = (item.timestampMs != null) ? fmtTimestampMs(item.timestampMs) : "";
  const lat = (item.latlng && item.latlng.length === 2) ? fmtNum(item.latlng[0]) : "";
  const lon = (item.latlng && item.latlng.length === 2) ? fmtNum(item.latlng[1]) : "";
  const parts = [];
  if (ts) parts.push(ts);
  if (lat && lon) parts.push(`lat ${lat} | lon ${lon}`);
  parts.push(phase);
  nowLine.textContent = parts.join("  •  ");
}

function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  if (Math.abs(n) >= 1000) return n.toFixed(2);
  return n.toFixed(6);
}

function fmtDistance(m) {
  const n = Number(m);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(2)} km`;
  return `${n.toFixed(1)} m`;
}

function setKpis({ modelName, histCount, predCount, totalDist, avgStep } = {}) {
  if (kpiModel) kpiModel.textContent = modelName || "—";
  if (kpiHistPts) kpiHistPts.textContent = (histCount == null) ? "—" : String(histCount);
  if (kpiPredPts) kpiPredPts.textContent = (predCount == null) ? "—" : String(predCount);
  if (kpiTotalDist) kpiTotalDist.textContent = (totalDist == null) ? "—" : fmtDistance(totalDist);
  if (kpiAvgStep) kpiAvgStep.textContent = (avgStep == null) ? "—" : fmtDistance(avgStep);
}

function setKpiCollapsed(isCollapsed) {
  if (!cardKpi) return;
  cardKpi.classList.toggle("isCollapsed", Boolean(isCollapsed));
  try { localStorage.setItem("kpiCollapsed", Boolean(isCollapsed) ? "1" : "0"); } catch (_) { }
}

function clearTable() { predTableBody.innerHTML = ""; }

function fillTable(prediction) {
  clearTable();
  for (let i = 0; i < Math.min(prediction.length, 200); i++) {
    const r = prediction[i];
    const tr = document.createElement("tr");
    const cells = [
      r.timestamp || "",
      fmtNum(r.x_m),
      fmtNum(r.y_m),
      r.lat == null ? "" : fmtNum(r.lat),
      r.lon == null ? "" : fmtNum(r.lon),
    ];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    predTableBody.appendChild(tr);
  }
}

function tooltipHTML(p) {
  const ts = (p.timestamp != null) ? String(p.timestamp) : "";
  const lat = (p.lat == null || Number.isNaN(p.lat)) ? null : Number(p.lat);
  const lon = (p.lon == null || Number.isNaN(p.lon)) ? null : Number(p.lon);
  const latTxt = (lat == null || !Number.isFinite(lat)) ? "" : fmtNum(lat);
  const lonTxt = (lon == null || !Number.isFinite(lon)) ? "" : fmtNum(lon);
  const parts = [];
  if (ts) parts.push(`<div style="font-weight:600">${ts}</div>`);
  if (latTxt || lonTxt) {
    parts.push(`<div style="opacity:0.95">lat: ${latTxt || ""}</div>`);
    parts.push(`<div style="opacity:0.95">lon: ${lonTxt || ""}</div>`);
  }
  if (!parts.length) return "";
  return `<div style="padding:2px 4px">${parts.join("")}</div>`;
}

function renderTrack(points, layer, color) {
  // Track = datapoints connected by straight segments, with hover timestamp on each point.
  layer.clearLayers();

  const latlngs = [];
  for (const p of points) {
    if (p.lat == null || p.lon == null || Number.isNaN(p.lat) || Number.isNaN(p.lon)) continue;
    const lat = Number(p.lat);
    const lon = Number(p.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    latlngs.push([lat, lon]);
  }

  // Draw connecting line first (behind markers)
  if (latlngs.length >= 2) {
    L.polyline(latlngs, {
      color,
      weight: 3,
      opacity: 0.85,
      lineJoin: "round"
    }).addTo(layer);
  }

  // Draw points
  for (const p of points) {
    if (p.lat == null || p.lon == null || Number.isNaN(p.lat) || Number.isNaN(p.lon)) continue;
    const lat = Number(p.lat);
    const lon = Number(p.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const cm = L.circleMarker([lat, lon], {
      radius: 4,
      color,
      fillColor: color,
      weight: 2,
      opacity: 0.95,
      fillOpacity: 0.85
    });
    const html = tooltipHTML(p);
    if (html) cm.bindTooltip(html, { direction: "top", sticky: true, opacity: 0.95, offset: [0, -6] });
    cm.addTo(layer);
  }
}


function ensureMap() {
  if (map) return;

  // Defensive cleanup: avoid duplicate maps if the script is reloaded (e.g., dev reload)
  const container = document.getElementById("map");
  if (container && container._leaflet_id) {
    try { container._leaflet_id = null; } catch (_) { }
    container.innerHTML = "";
  }

  // Prevent infinite world repetition (single world only)
  const WORLD_BOUNDS = [[-85, -180], [85, 180]];
  const worldBounds = L.latLngBounds(WORLD_BOUNDS);
  map = L.map("map", {
    zoomControl: true,
    maxBounds: WORLD_BOUNDS,
    maxBoundsViscosity: 1.0,
  });

  // Base map layer (only once)
  try {
    baseTile = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
      crossOrigin: true,
      noWrap: true,
      continuousWorld: false,
      bounds: WORLD_BOUNDS
    }).addTo(map);
  } catch (_) { }

  map.setView([0, 0], 2);

  // Extra guard: keep the viewport inside the single-world bounds even during drag.
  const clampToWorld = () => {
    try { map.panInsideBounds(worldBounds, { animate: false }); } catch (_) { }
  };
  map.on("drag", clampToWorld);
  map.on("moveend", clampToWorld);
  map.on("zoomend", clampToWorld);

  // Datapoint layers (no polyline)
  histLayer = L.layerGroup().addTo(map);
  predLayer = L.layerGroup().addTo(map);

  // Animation marker
  marker = L.circleMarker([0, 0], {
    radius: 7,
    color: HIST_COLOR,
    fillColor: HIST_COLOR,
    weight: 2,
    opacity: 1,
    fillOpacity: 1
  }).addTo(map);
}

function stopAnimation() {
  if (animationTimer) {
    clearInterval(animationTimer);
    animationTimer = null;
  }
  btnPlay.disabled = animationPath.length === 0;
  btnPause.disabled = true;
}

function stepAnimation() {
  if (!animationPath.length) return;
  animationIndex = (animationIndex + 1) % animationPath.length;
  const cur = animationPath[animationIndex];
  marker.setLatLng(cur.latlng);
  marker.setStyle({
    color: cur.isPred ? PRED_COLOR : HIST_COLOR,
    fillColor: cur.isPred ? PRED_COLOR : HIST_COLOR
  });
  updateNowBox(cur);
  maybeFollow(cur);
}

function startAnimation() {
  stopAnimation();
  if (animationPath.length === 0) return;
  btnPlay.disabled = true;
  btnPause.disabled = false;
  animationIndex = 0;
  marker.setLatLng(animationPath[0].latlng);
  marker.setStyle({
    color: animationPath[0].isPred ? PRED_COLOR : HIST_COLOR,
    fillColor: animationPath[0].isPred ? PRED_COLOR : HIST_COLOR
  });
  updateNowBox(animationPath[0]);

  tickMs = tickMsFromSpeed(speedRange ? speedRange.value : 1);
  animationTimer = setInterval(stepAnimation, tickMs);
}

function maybeFollow(cur) {
  if (!followToggle || !followToggle.checked || !map || !cur || !cur.latlng) return;
  // Throttle to avoid jitter
  const now = Date.now();
  if (now - lastPanMs < 200) return;
  lastPanMs = now;
  try {
    const ll = L.latLng(cur.latlng[0], cur.latlng[1]);
    const inner = map.getBounds().pad(-0.25);
    if (!inner.contains(ll)) {
      map.panTo(ll, { animate: true, duration: 0.20 });
    }
  } catch (_) { }
}

function densifyAnimationPath(rawPoints, histLen) {
  // rawPoints: Array<{latlng:[lat,lon], isPred:boolean, timestampMs:number|null}>
  // histLen: số điểm thuộc history trong rawPoints (để xác định pha)
  const out = [];
  if (!rawPoints || rawPoints.length === 0) return out;

  const asLL = (p) => L.latLng(p.latlng[0], p.latlng[1]);

  out.push({
    latlng: rawPoints[0].latlng,
    isPred: (0 >= histLen),
    timestampMs: rawPoints[0].timestampMs
  });

  for (let i = 0; i < rawPoints.length - 1; i++) {
    const a = rawPoints[i];
    const b = rawPoints[i + 1];
    const segIsPred = (i + 1) >= histLen;

    let steps = 1;
    try {
      const distM = map && map.distance ? map.distance(asLL(a), asLL(b)) : null;
      if (distM != null && Number.isFinite(distM)) {
        // khoảng ~2km/điểm, giới hạn để không quá nặng
        steps = Math.min(60, Math.max(1, Math.ceil(distM / 2000)));
      } else {
        steps = 10;
      }
    } catch (_) {
      steps = 10;
    }

    for (let s = 1; s <= steps; s++) {
      const t = s / steps;
      const lat = a.latlng[0] + (b.latlng[0] - a.latlng[0]) * t;
      const lon = a.latlng[1] + (b.latlng[1] - a.latlng[1]) * t;

      let timestampMs = null;
      if (Number.isFinite(a.timestampMs) && Number.isFinite(b.timestampMs)) {
        timestampMs = a.timestampMs + (b.timestampMs - a.timestampMs) * t;
      } else if (Number.isFinite(b.timestampMs)) {
        timestampMs = b.timestampMs;
      } else if (Number.isFinite(a.timestampMs)) {
        timestampMs = a.timestampMs;
      }

      out.push({ latlng: [lat, lon], isPred: segIsPred, timestampMs });
    }
  }

  return out;
}

function setTracks(history, prediction) {
  ensureMap();

  const isValidLatLon = (r) =>
    r.lat != null && r.lon != null &&
    !Number.isNaN(r.lat) && !Number.isNaN(r.lon) &&
    Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon));

  const histPts = history.filter(isValidLatLon).map(r => ({
    lat: Number(r.lat), lon: Number(r.lon),
    timestamp: r.timestamp, x_m: r.x_m, y_m: r.y_m
  }));

  const predPts = prediction.filter(isValidLatLon).map(r => ({
    lat: Number(r.lat), lon: Number(r.lon),
    timestamp: r.timestamp, x_m: r.x_m, y_m: r.y_m
  }));

  if (histPts.length > 0 || predPts.length > 0) {
    renderTrack(histPts, histLayer, HIST_COLOR);
    renderTrack(predPts, predLayer, PRED_COLOR);
    // Connect last history point to first prediction point with prediction color
    if (histPts.length > 0 && predPts.length > 0) {
      const a = histPts[histPts.length - 1];
      const b = predPts[0];
      if (Number.isFinite(Number(a.lat)) && Number.isFinite(Number(a.lon)) && Number.isFinite(Number(b.lat)) && Number.isFinite(Number(b.lon))) {
        L.polyline([[Number(a.lat), Number(a.lon)], [Number(b.lat), Number(b.lon)]], {
          color: PRED_COLOR,
          weight: 3,
          opacity: 0.85,
          lineJoin: "round"
        }).addTo(predLayer);
      }
    }

    const parseTs = (ts) => {
      if (!ts) return null;
      try {
        const t = Date.parse(String(ts).replace(" ", "T") + "Z");
        return Number.isFinite(t) ? t : null;
      } catch (_) {
        return null;
      }
    };

    const histRaw = histPts.map(p => ({
      latlng: [p.lat, p.lon],
      isPred: false,
      timestampMs: parseTs(p.timestamp)
    }));
    const predRaw = predPts.map(p => ({
      latlng: [p.lat, p.lon],
      isPred: true,
      timestampMs: parseTs(p.timestamp)
    }));

    animationBoundary = histRaw.length;
    const rawPoints = histRaw.concat(predRaw);
    animationPath = densifyAnimationPath(rawPoints, animationBoundary);

    if (animationPath.length > 0) {
      marker.setLatLng(animationPath[0].latlng);
      marker.setStyle({
        color: animationPath[0].isPred ? PRED_COLOR : HIST_COLOR,
        fillColor: animationPath[0].isPred ? PRED_COLOR : HIST_COLOR
      });
    }

    const allPath = rawPoints.map(p => p.latlng);
    if (allPath.length > 0) {
      lastFitBounds = L.latLngBounds(allPath).pad(0.2);
      map.fitBounds(lastFitBounds);
    }
    return;
  }

  // fallback: normalize x/y to pseudo-latlon
  const all = history.concat(prediction);
  const xs = all.map(r => Number(r.x_m)).filter(Number.isFinite);
  const ys = all.map(r => Number(r.y_m)).filter(Number.isFinite);
  if (!xs.length || !ys.length) return;

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const norm = (x, a, b) => (b - a) === 0 ? 0 : ((x - a) / (b - a)) * 140 - 70;

  const make = (rows) => rows.map(r => ({
    lat: norm(Number(r.y_m), minY, maxY),
    lon: norm(Number(r.x_m), minX, maxX),
    timestamp: r.timestamp, x_m: r.x_m, y_m: r.y_m
  }));

  const histFallbackPts = make(history);
  const predFallbackPts = make(prediction);

  renderTrack(histFallbackPts, histLayer, HIST_COLOR);
  renderTrack(predFallbackPts, predLayer, PRED_COLOR);

  // Connect last history point to first prediction point with prediction color
  if (histFallbackPts.length > 0 && predFallbackPts.length > 0) {
    const a = histFallbackPts[histFallbackPts.length - 1];
    const b = predFallbackPts[0];
    if (Number.isFinite(Number(a.lat)) && Number.isFinite(Number(a.lon)) && Number.isFinite(Number(b.lat)) && Number.isFinite(Number(b.lon))) {
      L.polyline([[Number(a.lat), Number(a.lon)], [Number(b.lat), Number(b.lon)]], {
        color: PRED_COLOR,
        weight: 3,
        opacity: 0.85,
        lineJoin: "round"
      }).addTo(predLayer);
    }
  }

  const parseTs = (ts) => {
    if (!ts) return null;
    try {
      const t = Date.parse(String(ts).replace(" ", "T") + "Z");
      return Number.isFinite(t) ? t : null;
    } catch (_) {
      return null;
    }
  };

  const histRaw = histFallbackPts.map(p => ({ latlng: [p.lat, p.lon], isPred: false, timestampMs: parseTs(p.timestamp) }));
  const predRaw = predFallbackPts.map(p => ({ latlng: [p.lat, p.lon], isPred: true, timestampMs: parseTs(p.timestamp) }));

  animationBoundary = histRaw.length;
  const rawPoints = histRaw.concat(predRaw);
  animationPath = densifyAnimationPath(rawPoints, animationBoundary);

  if (animationPath.length > 0) {
    marker.setLatLng(animationPath[0].latlng);
    marker.setStyle({
      color: animationPath[0].isPred ? PRED_COLOR : HIST_COLOR,
      fillColor: animationPath[0].isPred ? PRED_COLOR : HIST_COLOR
    });
  }

  const allPath = rawPoints.map(p => p.latlng);
  if (allPath.length > 0) {
    lastFitBounds = L.latLngBounds(allPath).pad(0.2);
    map.fitBounds(lastFitBounds);
  }
}

function resetView() {
  if (!map || !lastFitBounds) return;
  try { map.fitBounds(lastFitBounds); } catch (_) { }
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const txt = await res.text();
  let data;
  try { data = JSON.parse(txt); } catch (_) { data = { ok: false, error: txt }; }
  if (!res.ok) {
    const detail = (data && data.detail) ? data.detail : (data && data.error) ? data.error : txt;
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return data;
}

async function loadDatasetInfo() {
  const info = await fetchJSON("/api/dataset_info");
  if (!info.ok) {
    setStatus("Dataset chưa sẵn sàng", "err");
    datasetInfoEl.textContent = info.error || "Dataset load failed.";
    btnPredict.disabled = true;
    return;
  }
  setStatus("Sẵn sàng", "ok");
  datasetInfoEl.textContent = `Dataset: ${info.path} | rows=${info.rows} | range=${info.start_ts} → ${info.end_ts}`;
  if (!dtInput.value) dtInput.value = info.default_datetime;
}

async function loadModels() {
  const m = await fetchJSON("/api/models");
  modelSelect.innerHTML = "";
  if (!m.models || m.models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Không tìm thấy model trong models/";
    modelSelect.appendChild(opt);
    btnPredict.disabled = true;
    setStatus("Thiếu model", "err");
    return;
  }
  for (const name of m.models) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    modelSelect.appendChild(opt);
  }
}

async function doPredict() {
  stopAnimation();
  notesEl.textContent = "";
  clearTable();
  btnDownload.style.display = "none";
  if (btnDownloadXlsx) btnDownloadXlsx.style.display = "none";
  setKpis({});

  setActiveTab("notes");

  const payload = { datetime: dtInput.value, model: modelSelect.value || null };

  btnPredict.disabled = true;
  if (predictSpinner) predictSpinner.style.display = "inline-block";
  setStatus("Đang dự đoán...", "info");

  try {
    const res = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    setStatus("Dự đoán xong", "ok");
    setTracks(res.history || [], res.prediction || []);
    fillTable(res.prediction || []);

    setKpis({
      modelName: (res.notes && res.notes.used_model) ? res.notes.used_model : (payload.model || "—"),
      histCount: (res.history || []).length,
      predCount: (res.prediction || []).length,
      totalDist: res.summary ? res.summary.pred_total_distance : null,
      avgStep: res.summary ? res.summary.pred_avg_step_distance : null,
    });

    if (res.notes && res.notes.messages) {
      notesEl.textContent = "Ghi chú\n" + res.notes.messages.join("\n");
    }

    const q = new URLSearchParams({ datetime: payload.datetime, model: payload.model || "" });
    btnDownload.href = "/api/prediction.csv?" + q.toString();
    btnDownload.style.display = "inline-block";

    if (btnDownloadXlsx) {
      btnDownloadXlsx.href = "/api/prediction.xlsx?" + q.toString();
      btnDownloadXlsx.style.display = "inline-block";
    }

    btnPlay.disabled = (animationPath.length === 0);
    btnPause.disabled = true;
    if (btnResetView) btnResetView.disabled = !lastFitBounds;

    showToast("Dự đoán xong", "Có thể bấm Play để xem marker chạy.", "ok");
  } catch (e) {
    setStatus("Lỗi dự đoán", "err");
    notesEl.textContent = "Lỗi\n" + (e && e.message ? e.message : String(e));
    showToast("Lỗi", (e && e.message) ? e.message : String(e), "err", 3800);
  } finally {
    btnPredict.disabled = false;
    if (predictSpinner) predictSpinner.style.display = "none";
  }
}

btnPredict.addEventListener("click", doPredict);
btnPlay.addEventListener("click", startAnimation);
btnPause.addEventListener("click", stopAnimation);
if (btnResetView) btnResetView.addEventListener("click", resetView);

tabNotes.addEventListener("click", () => setActiveTab("notes"));
tabTable.addEventListener("click", () => setActiveTab("table"));

speedRange.addEventListener("input", () => {
  const v = Number(speedRange.value);
  const speed = (Number.isFinite(v) && v > 0) ? v : 1;
  speedLabel.textContent = `${speed}x`;
  tickMs = tickMsFromSpeed(speed);
  if (animationTimer) {
    clearInterval(animationTimer);
    animationTimer = setInterval(stepAnimation, tickMs);
  }
});

(async function init() {
  ensureMap();
  setActiveTab("notes");
  speedLabel.textContent = `${Number(speedRange.value) || 1}x`;
  setStatus("Đang tải...", "info");
  try {
    // KPI collapse state (persisted)
    const syncKpiAria = () => {
      if (!btnToggleKpi) return;
      const collapsed = Boolean(cardKpi && cardKpi.classList.contains("isCollapsed"));
      btnToggleKpi.setAttribute("aria-expanded", collapsed ? "false" : "true");
    };
    const toggleKpi = () => {
      const next = !(cardKpi && cardKpi.classList.contains("isCollapsed"));
      setKpiCollapsed(next);
      syncKpiAria();
    };

    try {
      const saved = localStorage.getItem("kpiCollapsed");
      if (saved === "1") setKpiCollapsed(true);
    } catch (_) { }
    syncKpiAria();

    if (btnToggleKpi) {
      btnToggleKpi.addEventListener("click", (ev) => {
        ev.stopPropagation();
        toggleKpi();
      });
    }
    // Click anywhere on the KPI header to toggle (more forgiving UX)
    if (cardKpi) {
      const hdr = cardKpi.querySelector(".cardHeader");
      if (hdr) {
        hdr.addEventListener("click", (ev) => {
          // ignore clicks on interactive controls
          if (ev.target && ev.target.closest && ev.target.closest("button,a,input,select,textarea,label")) return;
          toggleKpi();
        });
      }
    }

    await loadDatasetInfo();

    await loadModels();
    setKpis({});
    // UX: Enter to predict
    dtInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") doPredict();
    });
    // Panel toggle (mobile)
    if (btnTogglePanel && sidePanel) {
      btnTogglePanel.addEventListener("click", () => {
        sidePanel.classList.toggle("isCollapsed");
      });
    }
  } catch (e) {
    setStatus("Lỗi khởi tạo", "err");
    notesEl.textContent = "Lỗi\n" + (e && e.message ? e.message : String(e));
    showToast("Lỗi khởi tạo", (e && e.message) ? e.message : String(e), "err", 4200);
  }
})();
