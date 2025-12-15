let map, histLayer, predLayer, marker;
let histPoints = [];
let predPoints = [];
let anim = { running: false, t: 0, speed: 1, raf: null };
let currentMapMode = null; // 'mercator' | 'simple' | 'latlon'
let lastUsedModel = "-";

function qs(sel){ return document.querySelector(sel); }
function setStatus(text, ok=true){
  const el = qs("#status");
  el.textContent = text;
  el.style.color = ok ? "var(--muted)" : "var(--danger)";
}

function addNote(text){
  const li = document.createElement("li");
  li.textContent = text;
  qs("#notes").appendChild(li);
}
function clearNotes(){ qs("#notes").innerHTML = ""; }

function initMap(){
  // Map is initialized lazily once we know the coordinate mode.
}

function detectMode(points){
  // Decide how to interpret x_m/y_m for visualization.
  // - latlon: values look like degrees (|x|<=180 && |y|<=90)
  // - mercator: values fit Web Mercator meters (|x|,|y| <= 20037508)
  // - simple: fallback, draw on a plain XY plane
  const sample = (points || []).slice(0, 10);
  if(!sample.length) return "mercator";

  let looksLatLon = true;
  let looksMercator = true;

  for(const p of sample){
    const x = Number(p.x_m);
    const y = Number(p.y_m);
    if(!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if(Math.abs(x) > 180 || Math.abs(y) > 90) looksLatLon = false;
    if(Math.abs(x) > 20037508.34 || Math.abs(y) > 20037508.34) looksMercator = false;
  }

  if(looksLatLon) return "latlon";
  if(looksMercator) return "mercator";
  return "simple";
}

function mercatorToLatLon(x, y){
  const R = 6378137;
  const lon = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - (Math.PI / 2)) * (180 / Math.PI);
  return [lat, lon];
}

function ensureMap(mode){
  if(currentMapMode === mode && map) return;

  if(map){
    try{ map.remove(); }catch{}
  }

  currentMapMode = mode;

  if(mode === "simple"){
    map = L.map("map", {
      crs: L.CRS.Simple,
      zoomControl: true,
      minZoom: -5
    });
    // default bounds
    map.fitBounds(L.latLngBounds([0,0], [10000,10000]));
  }else{
    map = L.map("map", { zoomControl: true });

    // OpenStreetMap tiles (requires internet)
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    L.control.scale({ imperial: false }).addTo(map);
    map.setView([0,0], 2);
  }

  histLayer = L.layerGroup().addTo(map);
  predLayer = L.layerGroup().addTo(map);

  marker = L.circleMarker([0,0], { radius: 6, weight: 2, fillOpacity: 0.95 }).addTo(map);
}

function toLatLng(pt){
  const x = Number(pt.x_m);
  const y = Number(pt.y_m);

  if(currentMapMode === "latlon"){
    return L.latLng(y, x);
  }
  if(currentMapMode === "mercator"){
    const ll = mercatorToLatLon(x, y);
    return L.latLng(ll[0], ll[1]);
  }
  // CRS.Simple uses [y, x]
  return L.latLng(y, x);
}

function computeBoundsFromLatLng(latlngs){
  if(!latlngs || !latlngs.length) return null;
  const b = L.latLngBounds(latlngs);
  return b.pad(0.15);
}

function fmtNum(v, digits=6){
  if(v == null || !Number.isFinite(Number(v))) return "-";
  return Number(v).toFixed(digits);
}

function fmtMeters(m){
  if(!Number.isFinite(m)) return "-";
  if(m >= 1000) return (m/1000).toFixed(2) + " km";
  return m.toFixed(0) + " m";
}

function getAny(obj, keys){
  for(const k of keys){
    if(obj && obj[k] != null && obj[k] !== "") return obj[k];
  }
  return null;
}

function haversineMeters(lat1, lon1, lat2, lon2){
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const p1 = toRad(lat1), p2 = toRad(lat2);
  const dphi = toRad(lat2 - lat1);
  const dl = toRad(lon2 - lon1);
  const a = Math.sin(dphi/2)**2 + Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function updateOverlay(){
  const histN = histPoints.length;
  const predN = predPoints.length;

  qs("#kvModel").textContent = lastUsedModel || "-";
  qs("#kvHorizon").textContent = predN ? (predN + " giờ") : "-";
  qs("#kvHist").textContent = histN ? (histN + " dòng") : "-";

  const t0 = histN ? (histPoints[0].timestamp || "") : "";
  const t1 = histN ? (histPoints[histN - 1].timestamp || "") : "";
  const t2 = predN ? (predPoints[predN - 1].timestamp || "") : "";
  const timeStr = (t1 && t2) ? (t1 + " → " + t2) : (t0 && t1 ? (t0 + " → " + t1) : "-");
  qs("#kvTime").textContent = timeStr || "-";

  // last-known features (from history)
  const last = histN ? histPoints[histN - 1] : null;
  const temp = getAny(last, ["external_temperature","external-temperature"]);
  const gs = getAny(last, ["ground_speed","ground-speed"]);
  const light = getAny(last, ["gls_light_level","gls:light-level"]);
  const alt = getAny(last, ["height_above_msl","height-above-msl"]);
  const tod = getAny(last, ["time_of_day","time-of-day","time_of_day_code"]);
  const season = getAny(last, ["season","season_code"]);

  qs("#kvTemp").textContent = (temp == null) ? "-" : (Number(temp).toFixed(2));
  qs("#kvGS").textContent = (gs == null) ? "-" : (Number(gs).toFixed(3));
  qs("#kvLight").textContent = (light == null) ? "-" : (Number(light).toFixed(2));
  qs("#kvAlt").textContent = (alt == null) ? "-" : (Number(alt).toFixed(2));
  qs("#kvLabel").textContent = (tod == null && season == null) ? "-" : (String(tod ?? "-") + " / " + String(season ?? "-"));

  // distance/speed (approx)
  let distMeters = 0;
  if(predN >= 2){
    if(currentMapMode === "latlon"){
      for(let i=1;i<predN;i++){
        const p0 = toLatLng(predPoints[i-1]);
        const p1 = toLatLng(predPoints[i]);
        distMeters += haversineMeters(p0.lat, p0.lng, p1.lat, p1.lng);
      }
    }else if(currentMapMode === "mercator"){
      for(let i=1;i<predN;i++){
        const dx = Number(predPoints[i].x_m) - Number(predPoints[i-1].x_m);
        const dy = Number(predPoints[i].y_m) - Number(predPoints[i-1].y_m);
        if(Number.isFinite(dx) && Number.isFinite(dy)) distMeters += Math.hypot(dx, dy);
      }
    }else{
      // unknown unit
      for(let i=1;i<predN;i++){
        const dx = Number(predPoints[i].x_m) - Number(predPoints[i-1].x_m);
        const dy = Number(predPoints[i].y_m) - Number(predPoints[i-1].y_m);
        if(Number.isFinite(dx) && Number.isFinite(dy)) distMeters += Math.hypot(dx, dy);
      }
    }
  }

  const speedMetersPerHour = (predN > 0) ? (distMeters / predN) : 0;
  const distText = predN ? (currentMapMode === "simple" ? (distMeters.toFixed(2) + " (đv)") : fmtMeters(distMeters)) : "-";
  const speedText = predN ? (currentMapMode === "simple" ? (speedMetersPerHour.toFixed(2) + " (đv)/h") : (fmtMeters(speedMetersPerHour) + "/h")) : "-";
  qs("#kvDist").textContent = distText;
  qs("#kvSpeed").textContent = speedText;

  const modeLabel = currentMapMode === "mercator" ? "Web Mercator → lat/lon" : (currentMapMode === "latlon" ? "lat/lon trực tiếp" : "Mặt phẳng XY");
  qs("#kvMode").textContent = modeLabel;

  const hint = qs("#overlayHint");
  if(currentMapMode === "mercator"){
    hint.textContent = "Trang đang hiển thị trên bản đồ vì x_m/y_m nằm trong khoảng Web Mercator. Nếu dataset dùng CRS khác, vị trí trên bản đồ có thể không đúng.";
  }else if(currentMapMode === "simple"){
    hint.textContent = "Không đủ dấu hiệu để chuyển x_m/y_m sang lat/lon. Đang hiển thị trên mặt phẳng XY.";
  }else{
    hint.textContent = "";
  }
}

function updatePredTable(){
  const tbody = qs("#predTable tbody");
  tbody.innerHTML = "";
  if(!predPoints.length) return;

  // Render all rows; table has its own scroll.
  for(let i=0;i<predPoints.length;i++){
    const p = predPoints[i];
    const ll = toLatLng(p);
    const lat = (currentMapMode === "simple") ? null : ll.lat;
    const lon = (currentMapMode === "simple") ? null : ll.lng;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i+1}</td>
      <td>${p.timestamp || ""}</td>
      <td>${fmtNum(p.x_m, 3)}</td>
      <td>${fmtNum(p.y_m, 3)}</td>
      <td>${lat == null ? "-" : fmtNum(lat, 6)}</td>
      <td>${lon == null ? "-" : fmtNum(lon, 6)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function render(){
  // Decide map mode from data
  const mode = detectMode(histPoints.length ? histPoints : predPoints);
  ensureMap(mode);

  histLayer.clearLayers();
  predLayer.clearLayers();

  if(histPoints.length){
    const latlngs = histPoints.map(toLatLng);
    L.polyline(latlngs, { weight: 4, opacity: 0.9, color: "#7aa2ff" }).addTo(histLayer);
    for(const ll of latlngs){
      L.circleMarker(ll, { radius: 2, weight: 0, fillOpacity: 0.9, color: "#7aa2ff", fillColor: "#7aa2ff" }).addTo(histLayer);
    }
    marker.setLatLng(latlngs[latlngs.length - 1]);
  }

  if(predPoints.length){
    const latlngs = predPoints.map(toLatLng);
    L.polyline(latlngs, { weight: 4, dashArray: "8 10", opacity: 0.95, color: "#5be49b" }).addTo(predLayer);
    for(const ll of latlngs){
      L.circleMarker(ll, { radius: 2, weight: 0, fillOpacity: 0.9, color: "#5be49b", fillColor: "#5be49b" }).addTo(predLayer);
    }
  }

  // fit bounds
  const allLatLngs = [...histPoints, ...predPoints].map(toLatLng);
  const b = computeBoundsFromLatLng(allLatLngs);
  if(b) map.fitBounds(b, { animate: true });

  updateOverlay();
  updatePredTable();
}

function stopAnim(){
  anim.running = false;
  if(anim.raf) cancelAnimationFrame(anim.raf);
  anim.raf = null;
}
// Note: startAnim was removed; animation is handled by animateAlongPath().

function animateAlongPath(){
  stopAnim();
  if(!predPoints.length) return;

  anim.running = true;
  anim.t = 0;

  const segCount = predPoints.length - 1;

  let lastNow = performance.now();

  function tick(now){
    if(!anim.running) return;
    const deltaSec = (now - lastNow) / 1000;
    lastNow = now;

    // how fast: 0.7 segments/sec at 1x
    const segPerSec = 0.7 * anim.speed;
    anim.t += deltaSec * segPerSec;

    // clamp & loop
    if(anim.t > segCount){
      anim.t = segCount;
      anim.running = false;
    }

    const i = Math.floor(anim.t);
    const f = Math.min(Math.max(anim.t - i, 0), 1);

    const p0 = predPoints[i];
    const p1 = predPoints[Math.min(i + 1, predPoints.length - 1)];

    const x = p0.x_m + (p1.x_m - p0.x_m) * f;
    const y = p0.y_m + (p1.y_m - p0.y_m) * f;

    marker.setLatLng(toLatLng({ x_m: x, y_m: y }));

    anim.raf = requestAnimationFrame(tick);
  }

  anim.raf = requestAnimationFrame(tick);
}

function downloadPredCSV(){
  if(!predPoints.length) return;
  const headers = ["timestamp","x_m","y_m","lat","lon"].join(",");
  const rows = predPoints.map(p => {
    const ll = toLatLng(p);
    const lat = (currentMapMode === "simple") ? "" : String(Number(ll.lat));
    const lon = (currentMapMode === "simple") ? "" : String(Number(ll.lng));
    const cols = [
      p.timestamp || "",
      Number(p.x_m),
      Number(p.y_m),
      lat,
      lon
    ];
    return cols.join(",");
  });
  const csv = [headers, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "predicted_path.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function refreshModels(){
  const res = await fetch("/api/models");
  const data = await res.json();
  const select = qs("#modelSelect");
  select.innerHTML = "";

  const optAuto = document.createElement("option");
  optAuto.value = "";
  optAuto.textContent = "(Tự chọn / fallback heuristic)";
  select.appendChild(optAuto);

  for(const m of data.models || []){
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    select.appendChild(opt);
  }
}

function parseCSV(text){
  // Minimal CSV parser (comma-separated, header on first line)
  // Supports the dataset headers you showed (dash/colon names).
  const lines = text.split(/\r?\n/).filter(l => l.trim().length);
  if(lines.length < 2) throw new Error("CSV rỗng hoặc thiếu dữ liệu.");

  const headersRaw = lines[0].split(",").map(h => h.trim());

  const normalizeHeader = (h) => {
    const m = {
      "external-temperature": "external_temperature",
      "ground-speed": "ground_speed",
      "height-above-msl": "height_above_msl",
      "gls:light-level": "gls_light_level",
    };
    if(m[h]) return m[h];
    // generic normalization
    return h.replace(/[:\-]/g, "_");
  };

  const headers = headersRaw.map(normalizeHeader);

  const numericCols = new Set([
    "external_temperature", "ground_speed", "height_above_msl", "gls_light_level",
    "x_m", "y_m",
    "sin_heading", "cos_heading",
    "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month",
    "distance"
  ]);

  const records = [];
  for(let i=1;i<lines.length;i++){
    const parts = lines[i].split(","); // basic; sufficient for demo
    if(parts.length !== headers.length) continue;

    const rec = {};
    for(let j=0;j<headers.length;j++){
      const key = headers[j];
      const raw = parts[j].trim();

      if(raw === "") continue;

      if(numericCols.has(key)){
        const v = Number(raw);
        if(!Number.isNaN(v)) rec[key] = v;
      }else{
        // keep as string (timestamp, time_of_day, season, ...)
        rec[key] = raw;
      }
    }

    // required
    if(rec.x_m == null || rec.y_m == null) continue;
    if(Number.isNaN(rec.x_m) || Number.isNaN(rec.y_m)) continue;

    records.push(rec);
  }

  // Keep last 48 records if longer (matches INPUT_WINDOW)
  return records.slice(Math.max(0, records.length - 48));
}

async function loadSample(){
  clearNotes();
  const res = await fetch("/api/sample");
  const data = await res.json();
  histPoints = data.records || [];
  predPoints = [];
  lastUsedModel = "-";
  addNote("Đã tải dữ liệu mẫu (48h).");
  render();
}

async function predict(){
  clearNotes();
  if(!histPoints.length){
    addNote("Chưa có dữ liệu lịch sử. Hãy tạo dữ liệu mẫu hoặc tải CSV.");
    return;
  }
  const horizon = Number(qs("#horizon").value) || 24;
  const modelName = qs("#modelSelect").value || null;

  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records: histPoints, horizon_hours: horizon, model_name: modelName })
  });

  if(!res.ok){
    const t = await res.text();
    addNote("Lỗi backend: " + t);
    return;
  }

  const data = await res.json();
  predPoints = (data.predicted || []).map(p => ({ x_m: p.x_m, y_m: p.y_m, timestamp: p.timestamp }));
  lastUsedModel = (data.used_model || "unknown");
  addNote("Đã chạy dự đoán. Used_model = " + lastUsedModel);

  for(const n of (data.notes || [])){
    addNote(n);
  }

  render();
  animateAlongPath();
}

function wireUI(){
  qs("#btnSample").addEventListener("click", loadSample);

  qs("#fileInput").addEventListener("change", async (e) => {
    clearNotes();
    const file = e.target.files?.[0];
    if(!file) return;

    const text = await file.text();
    try{
      histPoints = parseCSV(text);
      predPoints = [];
      addNote(`Đã tải CSV. Số dòng dùng làm lịch sử: ${histPoints.length}.`);
      if(histPoints.length < 2){
        addNote("CSV cần ít nhất 2 dòng hợp lệ.");
      }
      render();
    }catch(err){
      addNote("Không đọc được CSV: " + (err?.message || String(err)));
    }finally{
      e.target.value = "";
    }
  });

  qs("#btnPredict").addEventListener("click", predict);

  qs("#btnPlay").addEventListener("click", () => {
    if(!predPoints.length) return;
    anim.running = true;
    animateAlongPath();
  });
  qs("#btnPause").addEventListener("click", () => stopAnim());

  qs("#btnReset").addEventListener("click", () => {
    const allLatLngs = [...histPoints, ...predPoints].map(toLatLng);
    const b = computeBoundsFromLatLng(allLatLngs);
    if(b) map.fitBounds(b, { animate: true });
  });

  qs("#btnDownloadPred").addEventListener("click", downloadPredCSV);

  const speed = qs("#speed");
  const speedVal = qs("#speedVal");
  speed.addEventListener("input", () => {
    anim.speed = Number(speed.value) || 1;
    speedVal.textContent = `${anim.speed}x`;
  });
  anim.speed = Number(speed.value) || 1;
  speedVal.textContent = `${anim.speed}x`;
}

async function boot(){
  wireUI();

  try{
    const res = await fetch("/api/health");
    const data = await res.json();
    setStatus("Server OK • Models: " + (data.models?.length ?? 0), true);
  }catch{
    setStatus("Không kết nối được backend (/api/health). Hãy chạy server.", false);
  }

  await refreshModels();
  await loadSample();
}

boot();
