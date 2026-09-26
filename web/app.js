/* VitalGuard AI - browser client.
 * Talks to FastAPI at /api/* only (relative URLs, so it works through any proxy).
 * The camera path here is the same math the Android app uses: per-frame mean of
 * the ROI's green channel -> POST /api/ppg/process -> SciPy filtering, peak
 * detection, HistGB risk model and SHAP all run server-side. */
(() => {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (path, opts = {}) => {
  // Content-type must be set whenever a JSON body is present, not only when the
  // caller happened to pass headers: FastAPI rejects an untyped body with 422,
  // which silently broke every POST (simulate / record / seed / re-score).
  const headers = { ...(opts.body ? { "content-type": "application/json" } : {}), ...opts.headers };
  const res = await fetch(path, { ...opts, ...(Object.keys(headers).length ? { headers } : {}) });
  if (!res.ok) {
    let d; try { d = (await res.json()).detail; } catch { d = res.statusText; }
    const txt = typeof d === "string" ? d : JSON.stringify(d);
    throw new Error(res.status === 422
      ? `server rejected the request (422): ${txt.slice(0, 200)}`
      : `${res.status}: ${txt.slice(0, 200)}`);
  }
  return res.json();
};
const post = (p, body) => api(p, { method: "POST", body: JSON.stringify(body ?? {}) });
const getJSON = (p) => api(p);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const FIXED_SEED = 20260926;   // reproducible synthetic signal; not a measurement
const HARD_SEED = 4242;      // shaky-mode base seed (deterministic refusal)
const TIERCOL = { critical: "#ff4d5e", high: "#ff9f2e", moderate: "#ffd83d", low: "#5ad1a5", minimal: "#4aa3ff" };

/* ---------------------------------------------------------------- tabs --- */
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "ward") loadWard();
  if (b.dataset.tab === "alerts") loadAlerts();
  if (b.dataset.tab === "model") loadModel();
  if (b.dataset.tab === "capture") loadPatientsIntoSelect();
}));

/* --------------------------------------------------------------- camera --- */
const S = { stream: null, track: null, vfc: null, raf: null, useVfc: false, last: -1, frames: [], recording: false,
            timer: null, t0: 0, torch: false, roi: null, lastMean: 0, trace: [], sel: null, busy: false, requestId: 0 };
const video = $("#video"), trace = $("#trace"), tctx = trace.getContext("2d");
let off = document.createElement("canvas"), octx = off.getContext("2d", { willReadFrequently: true });

function captureBusy(value) {
  S.busy = value;
  $("#btnSim").disabled = value;
  $("#simHard").disabled = value;
  $("#btnCam").disabled = value;
  $("#btnRec").disabled = value || !S.stream;
  $("#btnFlash").disabled = value || !S.track?.getCapabilities?.().torch;
}
function beginResult(source, truth) {
  S.requestId += 1;
  $("#resultBody").hidden = true;
  $("#resultEmpty").hidden = false;
  $("#resultEmpty").textContent = "Processing new capture… previous result cleared.";
  $("#resultSource").textContent = source === "simulation"
    ? `DEMO ONLY · synthetic ${truth.toFixed(1)} BPM input · NOT your heart rate · never saved to a patient`
    : "CAMERA ESTIMATE · experimental, not medically validated · verify independently";
  return S.requestId;
}
async function startCamera() {
  if (S.busy) return;
  if (S.stream) return stopCamera();
  captureBusy(true);
  // A prior simulation hid the video and left its ROI behind. Restore both.
  if (S.simCanvas) { S.simCanvas.remove(); S.simCanvas = null; }
  video.style.display = "";
  S.roi = null; S.frames = []; S.trace = []; S.last = -1; S.warned = false;
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 },
               frameRate: { ideal: 30, max: 60 } }, audio: false });
  } catch (e) {
    $("#camMsg").innerHTML = `Camera unavailable (${esc(e.name)}). Use <b>Simulate fingertip</b> to exercise the exact same processing path.`;
    captureBusy(false);
    return;
  }
  try { video.srcObject = S.stream; await video.play(); }
  catch (e) { stopCamera(); captureBusy(false); diag(`Camera playback failed: ${esc(e.message)}`); return; }
  S.track = S.stream.getVideoTracks()[0];
  const caps = S.track.getCapabilities ? S.track.getCapabilities() : {};
  $("#btnFlash").disabled = !caps.torch;
  $("#btnRec").disabled = false;
  $("#btnCam").textContent = "Stop camera";
  $("#camMsg").innerHTML = "Press your fingertip gently but fully over the lens and flash. Keep the phone still.";
  $("#roiRing").classList.add("live");
  S.useVfc = typeof video.requestVideoFrameCallback === "function";
  sizeCanvas();
  schedule();
  diag(`sampling at ~${S.useVfc ? "camera frame rate (requestVideoFrameCallback)" : "display rate (rAF fallback)"} · luma will show here once frames flow`);
  captureBusy(false);
}
function stopCamera() {
  if (S.vfc != null && typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(S.vfc);
  if (S.raf) cancelAnimationFrame(S.raf);
  S.vfc = S.raf = null;
  S.stream?.getTracks().forEach((t) => t.stop()); S.stream = null; S.track = null;
  video.srcObject = null; $("#btnRec").disabled = true; $("#btnFlash").disabled = true;
  $("#btnCam").textContent = "Start camera"; $("#roiRing").classList.remove("live");
  $("#btnFlash").textContent = "Flash: off"; S.torch = false;
}
async function toggleFlash() {
  if (!S.track) return;
  S.torch = !S.torch;
  try { await S.track.applyConstraints({ advanced: [{ torch: S.torch }] }); $("#btnFlash").textContent = "Flash: " + (S.torch ? "on" : "off"); }
  catch { $("#btnFlash").textContent = "flash n/a"; }
}
function roiBox() {
  const vw = video.videoWidth, vh = video.videoHeight;
  // Until the stream has produced a frame, videoWidth is 0 and every derived value
  // is NaN/0 - returning a zero-size box used to throw inside getImageData and kill
  // the sampler for the rest of the session. Report "not ready" instead.
  if (!vw || !vh) return null;
  const r = $("#roiRing").getBoundingClientRect(), b = video.getBoundingClientRect();
  if (!(r.width > 8 && b.width > 8 && b.height > 8)) return null;
  const sx = vw / b.width, sy = vh / b.height;
  const cx = (r.left + r.width / 2 - b.left) * sx, cy = (r.top + r.height / 2 - b.top) * sy;
  const side = Math.max(12, Math.min(60, Math.round(r.width * 0.55 * sx)));
  const n = (v) => Math.round(v);
  if (!(side >= 8) || !Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  return { x: n(cx - side / 2), y: n(cy - side / 2), s: n(side) };
}
function pushFrame(img, t) {
  const s = S.roi.s | 0, rx = S.roi.x | 0, ry = S.roi.y | 0;   // getImageData needs ints
  octx.drawImage(img, rx, ry, s, s, 0, 0, s, s);
  const d = octx.getImageData(0, 0, s, s).data;
  let g = 0; for (let i = 1; i < d.length; i += 4) g += d[i];
  g /= d.length / 4;
  S.frames.push({ t: +t.toFixed(4), v: +g.toFixed(3), luma: +g.toFixed(3) });
  S.lastMean = g;
  const ac = g - (S.trace.length ? S.trace.reduce((a, b2) => a + b2, 0) / S.trace.length : g);
  S.trace.push(ac); if (S.trace.length > 260) S.trace.shift();
  drawTrace();
}
function schedule() {
  if (!S.stream) return;
  if (S.useVfc) S.vfc = video.requestVideoFrameCallback(loop);
  else S.raf = requestAnimationFrame(loop);
}
function loop(_, meta) {
  if (!S.stream) return;
  const now = performance.now() / 1000;
  try { sampleOnce(now, meta); }
  catch (e) {
    // One bad frame must not end the capture; surface it once and keep going.
    if (!S.warned) { S.warned = true; diag(`sampling problem: ${esc(e.message)} - retrying`); }
  }
  schedule();
}
function sampleOnce(now, meta) {
  // recompute the ROI only when the layout may have changed (rVFC gives us a new
  // mediaTime each frame; the rAF path would otherwise read geometry 60x/second)
  if (S.roi == null || (S.useVfc && meta?.mediaTime !== undefined)) S.roi = roiBox() ?? S.roi;
  // throttle the rAF path to ~20 fps: sampling at 60 Hz from a 30 fps camera just
  // duplicates frames and biases the effective sample rate the filter relies on
  if (S.roi && (!S.useVfc && now - (S.last ?? -1) >= 0.045 || S.useVfc)) {
    S.last = now; pushFrame(video, now);
    if (S.frames.length === 1) diag(`camera streaming · ${video.videoWidth}x${video.videoHeight} @ ~${S.useVfc ? "camera" : "display"} rate`);
  }
}
function sizeCanvas() {
  const r = trace.getBoundingClientRect();
  trace.width = Math.max(320, r.width * devicePixelRatio); trace.height = 120 * devicePixelRatio;
  const p = $("#ppgPlot"); const pr = p.getBoundingClientRect();
  p.width = Math.max(320, pr.width * devicePixelRatio); p.height = 140 * devicePixelRatio;
}
addEventListener("resize", () => { sizeCanvas(); S.roi = null; });
function drawTrace() {
  const w = trace.width, h = trace.height, c = tctx;
  c.clearRect(0, 0, w, h); c.lineWidth = 2 * devicePixelRatio;
  c.strokeStyle = "rgba(255,255,255,.08)"; c.beginPath(); c.moveTo(0, h / 2); c.lineTo(w, h / 2); c.stroke();
  const mx = Math.max(3, ...S.trace.map(Math.abs));
  c.strokeStyle = S.recording ? "#5ad1a5" : "#4aa3ff"; c.beginPath();
  S.trace.forEach((v, i) => {
    const x = (i / Math.max(60, S.trace.length)) * w, y = h / 2 - (v / mx) * (h / 2 - 6);
    i ? c.lineTo(x, y) : c.moveTo(x, y);
  });
  c.stroke();
}

/* ------------------------------------------------------------- simulate --- */
/* Render a synthetic fingertip clip on a canvas, then read it back through the
   identical pixel -> green mean path. Keeps the demo honest and camera-free. */
function simSurface() {
  if (S.simCanvas) return S.simCanvas;
  const c = document.createElement("canvas");
  c.width = 320; c.height = 240;
  c.style.cssText = "width:100%;display:block;aspect-ratio:4/3;background:#05080f";
  video.style.display = "none";
  video.parentNode.insertBefore(c, video);
  S.simCanvas = c;
  return c;
}
/* Render a synthetic fingertip clip on an offscreen canvas, then read it back
   through the identical pixel -> green-channel-mean path used by the camera.
   The server never learns which source produced the numbers. */
function simulateFrames(hr, secs = 12, fps = 20, seed = FIXED_SEED, opts = {}) {
  const noise = opts.noise ?? 1.1, motion = opts.motion ?? 0;
  const n = Math.floor(secs * fps), w = 320, h = 240;
  off.width = w; off.height = h;
  const surf = simSurface(), sctx = surf.getContext("2d");
  let s = seed >>> 0;
  const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const out = [], f = hr / 60, base = 132, amp = base * 0.030;
  for (let i = 0; i < n; i++) {
    const t = i / fps + gauss() * 0.004;   // timestamp jitter, ~4 ms
    const ph = 2 * Math.PI * f * t;
    let p = Math.sin(ph) - 0.42 * Math.sin(2 * ph + 0.5) + 0.2 * Math.sin(3 * ph + 1.1);
    p /= 1.16;
    // per-frame noise ~1.1 LSB on a 132-luma signal, i.e. ~13 dB SNR: what a phone
    // sensor with the torch on actually delivers. Earlier 2.4 (4 dB) made the demo
    // button produce a legitimately gated 'retake' reading.
    const shake = motion > 0 ? amp * 1.1 * Math.sin(2 * Math.PI * (motion / 60) * t) : 0;
    const lum = Math.max(24, Math.min(244, base + amp * p + gauss() * noise + 5 * Math.sin(2 * Math.PI * 0.25 * t) + shake));
    const g = Math.round(lum), r = Math.min(255, g + 30), b = Math.max(0, g - 14);
    const grad = octx.createRadialGradient(w / 2, h / 2, 10, w / 2, h / 2, w * 0.44);
    grad.addColorStop(0, `rgb(${r},${g},${b})`);
    grad.addColorStop(0.62, `rgb(${Math.round(r * 0.62)},${Math.round(g * 0.5)},${Math.round(b * 0.5)})`);
    grad.addColorStop(1, "rgb(12,8,10)");
    octx.fillStyle = grad; octx.fillRect(0, 0, w, h);
    for (let k = 0; k < 26; k++) {
      octx.fillStyle = `rgba(255,255,255,${0.015 + rnd() * 0.05})`;
      octx.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 2, 1 + rnd() * 2);
    }
    $("#roiRing").style.top = "50%";
    S.roi = { x: Math.round(w / 2 - 26), y: Math.round(h / 2 - 26), s: 52 };
    const gmean = meanGreen();
    out.push({ t: +t.toFixed(4), v: +gmean.toFixed(3), luma: +gmean.toFixed(3) });
    sctx.drawImage(off, 0, 0, surf.width, surf.height);
    S.trace.push(gmean - (S.trace.length ? S.trace.reduce((a, b2) => a + b2, 0) / S.trace.length : gmean));
    if (S.trace.length > 260) S.trace.shift();
    if (i % 2 === 0) drawTrace();
  }
  return out;
  function rnd() { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }
}
function meanGreen() {
  const s = S.roi.s | 0; const d = octx.getImageData(S.roi.x | 0, S.roi.y | 0, s, s).data;
  let g = 0; for (let i = 1; i < d.length; i += 4) g += d[i];
  return g / (d.length / 4);
}

/* --------------------------------------------------------------- record --- */
function collectVitals() {
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : +v; };
  return { spo2: num("#inSpo2"), rr: num("#inRr"), sbp: num("#inSbp"), dbp: num("#inDbp"), temp_c: num("#inTemp") };
}
async function record() {
  if (S.busy || S.recording || !S.stream) return;
  captureBusy(true);
  const requestId = beginResult("camera");
  S.recording = true; S.frames = []; S.trace = []; S.t0 = performance.now() / 1000;
  $("#btnRec").disabled = true; $("#btnRec").textContent = "Recording…";
  const DUR = 12000; const start = performance.now();
  $("#roiRing").classList.add("live");
  await new Promise((res) => {
    S.timer = setInterval(() => {
      const k = Math.min(1, (performance.now() - start) / DUR);
      $("#progBar").style.width = (k * 100).toFixed(0) + "%";
      $("#progTxt").textContent = `capturing ${((DUR - (performance.now() - start)) / 1000).toFixed(1)} s · ${S.frames.length} frames · luma ${S.lastMean.toFixed(0)}`
        + (S.frames.length === 0 ? "  ← NO FRAMES: camera not streaming" : "");
      if (k >= 1) { clearInterval(S.timer); res(); }
    }, 100);
  });
  S.recording = false; $("#btnRec").textContent = "Record 12 s";
  $("#progTxt").textContent = `sending ${S.frames.length} frames to /api/ppg/process …`;
  try { await submit({ frames: S.frames.slice(), requestId }); $("#progTxt").textContent = `done · ${S.frames.length} frames`; }
  catch (e) { fail(e.message); }
  finally { captureBusy(false); }
}
async function runSim() {
  if (S.busy || S.recording) return;
  const hard = !!$("#simHard").checked;
  const hr = hard ? 96 : 80;
  // Demo and physical capture must never share a live sampling loop.
  if (S.stream) stopCamera();
  captureBusy(true);
  const requestId = beginResult("simulation", hr);
  S.trace = [];
  $("#progTxt").textContent = hard
    ? "DEMO: synthetic 96 BPM plus motion/noise; rejection is expected"
    : "DEMO: generating an 80 BPM signal; independently estimating it…";
  try {
    const frames = simulateFrames(hr, 12, 20, hard ? HARD_SEED : FIXED_SEED,
                                  hard ? { noise: 1.6, motion: 64 } : {});
    const r = await submit({ frames, _truth: hr, requestId });
    if (r) $("#progTxt").textContent = r.__gated
      ? "DEMO rejected — no reliable measurement or risk score"
      : `DEMO ONLY · simulated truth ${hr.toFixed(1)} BPM · estimate ${r.heart_rate.hr_bpm} BPM`;
  } catch (e) { fail(e.message); }
  finally { captureBusy(false); }
}
function fail(m) { $("#resultEmpty").hidden = false; $("#resultBody").hidden = true;
  $("#resultEmpty").innerHTML = `<b style="color:#ff8b96">${esc(m)}</b>`; }
function diag(m) { const el = $("#camMsg"); if (el) el.innerHTML = m; }
function banner(m, col = "#ff4d5e") {
  let b = $("#diag");
  if (!b) { b = document.createElement("div"); b.id = "diag";
    b.style.cssText = "position:fixed;left:0;right:0;top:0;z-index:99;padding:9px 14px;font:12.5px/1.4 ui-monospace,monospace;color:#fff;cursor:pointer";
    document.body.prepend(b);
    b.onclick = () => { b.style.display = "none"; }; }
  b.style.background = col; b.style.display = "block"; b.textContent = m;
  b.title = "click to dismiss";
}
window.addEventListener("error", (e) => banner(`JS error: ${e.message} @ ${(e.filename || "").split("/").pop()}:${e.lineno}`));
window.addEventListener("unhandledrejection", (e) => banner(`async error: ${e.reason?.message || e.reason}`));
(function selfcheck() {
  const secure = window.isSecureContext;
  const md = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  if (!md) banner("No camera API in this browser: navigator.mediaDevices.getUserMedia is missing. Use Chrome/Safari, not an in-app browser (WhatsApp/Instagram links open a view without camera access) — tap the browser's open-in-browser menu.", "#ff9f2e");
  else if (!secure) banner(`Insecure context (isSecureContext=false, page is ${location.protocol}) — cameras are blocked. Use the https:// link.`, "#ff9f2e");
})();

async function submit({ frames, _truth, requestId = S.requestId }) {
  if (!frames || frames.length < 16) {
    fail(`Only ${frames ? frames.length : 0} frames were captured (16 needed). ` +
      `The camera never started or the page lost focus. Press Start camera first and confirm the preview is moving.`);
    banner(`capture produced ${frames ? frames.length : 0} frames - camera did not stream`, "#ff9f2e");
    return;
  }
  const pick = ensurePick();
  const simulated = Number.isFinite(_truth);
  const body = { frames, vitals: collectVitals(), source: simulated ? "simulation" : "camera",
                 expected_hr_bpm: simulated ? _truth : null,
                 save: !simulated && !!pick.value, patient_id: !simulated && pick.value ? +pick.value : null };
  const r = await post("/api/ppg/process", body);
  if (requestId !== S.requestId) return; // never render a stale response
  if (simulated) r.__truth = _truth;
  const h = r.heart_rate;
  // Independently check demo integrity in the client too, including older servers.
  const candidate = h.hr_bpm;
  const mismatch = simulated && (!Number.isFinite(candidate) || Math.abs(candidate - _truth) > 3);
  const rejected = mismatch || r.retry_required || r.accepted === false || !r.risk;
  if (rejected) {
    r.risk = null; r.__gated = true;
    if (mismatch) r.message = "Demo validation failed or the signal was refused. No measurement is shown; this is not your pulse.";
    h.hr_bpm = null; h.hrv_rmssd_ms = null; h.breathing_rate_bpm = null;
  }
  $("#resultEmpty").hidden = true; $("#resultBody").hidden = false;
  // one decimal always: the server rounds to 1 dp, so 89.0 would otherwise print as
  // "89" next to "quality 100.0" and read like a different kind of number
  $("#oHr").textContent = h.hr_bpm == null ? "—" : Number(h.hr_bpm).toFixed(1);
  $("#oQual").textContent = h.quality; $("#oQualU").textContent = "/ 100 · " + h.quality_label;
  $("#oHrv").textContent = h.hrv_rmssd_ms ?? "—";
  $("#oResp").textContent = h.breathing_rate_bpm ?? "not derived";
  drawPPG(r.waveform, h);
  const notes = $("#oNotes");
  notes.innerHTML = (h.guidance || []).map((g) => `<div>⚠ ${esc(g)}</div>`).join("")
    + (simulated ? `<div style="color:#8ea3c4">simulated ground truth ${_truth.toFixed(1)} BPM → ${rejected ? "REJECTED (no measurement)" : `estimated ${h.hr_bpm} BPM (error ${Math.abs(_truth - h.hr_bpm).toFixed(2)} BPM)`}</div>` : "")
    + `<details><summary>Technical diagnostics — not separate measurements</summary><div style="color:#8ea3c4">spectral candidate ${h.hr_spectral_bpm} · autocorrelation candidate ${h.hr_temporal_bpm} · peak-interval candidate ${h.hr_from_peaks_bpm ?? "n/a"} BPM (spread ${h.agree_bpm})</div></details>`;
  if (!r.risk) {
    r.__gated = true;
    $("#oRisk").textContent = "not scored"; $("#oTier").textContent = "retry required";
    $("#oTier").style.color = "#ff9f2e"; $("#oAdvice").innerHTML = `<b>${esc(r.message || "Signal too weak")}</b>Quality gate (step 5) blocked the reading before the AI stage.`;
    $("#oFactors").innerHTML = ""; $("#oFlags").innerHTML = ""; $("#oMeta").textContent = "";
    drawGauge(null);
    return r;              // report refusal; never silently generate another clip
  }
  const ex = r.risk;
  $("#oRisk").textContent = ex.risk_percent + "%";
  if (simulated) $("#resultSource").textContent += " · risk shown below is also demo output";
  $("#oTier").textContent = ex.tier; $("#oTier").style.color = TIERCOL[ex.tier];
  drawGauge(ex.risk_score);
  const a = ex.advice || {};
  $("#oAdvice").innerHTML = `<b>${esc(a.action || "")}</b>${esc(a.detail || "")}` +
    (a.conflict ? `<div style="color:#ffb3bb;margin-top:6px">⚠ ${esc(a.conflict)}</div>` : "") +
    `<div style="color:#8ea3c4;font-size:11.5px;margin-top:5px">recheck in ${a.recheck_min} min · NEWS2 ${ex.news2 ?? "—"} · ${ex.top_factors.length} explained factors</div>`;
  renderFactors(ex.top_factors, $("#oFactors"));
  $("#oFlags").innerHTML = (ex.red_flags || []).map((f) => `<span class="flag">${esc(f.text)}</span>`).join("");
  $("#oMeta").textContent = `explanation: ${ex.shap_method || "n/a"} · base logit ${ex.shap_base_value} · model HistGB · saved reading #${r.saved_reading_id ?? "—"}`;
  refreshSnapshot();
  return r;
}
function renderFactors(list, host) {
  if (!list?.length) { host.innerHTML = `<div class="empty">No factor exceeded the reporting threshold.</div>`; return; }
  const mx = Math.max(...list.map((f) => Math.abs(f.shap)), 0.001);
  host.innerHTML = list.map((f) => {
    const w = (Math.abs(f.shap) / mx) * 50, pos = f.shap > 0;
    return `<div class="factor" title="${esc(f.text)}">
      <span class="lab">${esc(f.label)} <em>${esc(f.value_text)}</em></span>
      <span class="track"><span class="fill" style="${pos ? "left:50%" : `right:50%`};width:${w.toFixed(1)}%;background:${pos ? "var(--crit)" : "var(--low)"}"></span></span>
      <span class="num" style="color:${pos ? "#ff8b96" : "#7fe0bd"}">${f.shap > 0 ? "+" : ""}${f.shap.toFixed(3)}</span></div>`;
  }).join("");
}
function drawPPG(wf, h) {
  const c = $("#ppgPlot"), x = c.getContext("2d"), w = c.width, ht = c.height;
  x.clearRect(0, 0, w, ht);
  if (!wf?.t?.length) return;
  const tmax = wf.t[wf.t.length - 1] || 1;
  x.strokeStyle = "rgba(255,255,255,.07)"; x.lineWidth = 1;
  for (let g = 0; g <= 4; g++) { const yy = (g / 4) * ht; x.beginPath(); x.moveTo(0, yy); x.lineTo(w, yy); x.stroke(); }
  if (h.hr_bpm) { // mark detected beats on the time axis
    x.strokeStyle = "rgba(255,77,94,.30)"; const per = 60 / h.hr_bpm;
    for (let tt = 0; tt < tmax; tt += per) { const px = (tt / tmax) * w; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, ht); x.stroke(); }
  }
  x.strokeStyle = "#5ad1a5"; x.lineWidth = 1.8 * devicePixelRatio; x.beginPath();
  wf.t.forEach((tt, i) => { const px = (tt / tmax) * w, py = ht / 2 - wf.y[i] * (ht / 2 - 8);
    i ? x.lineTo(px, py) : x.moveTo(px, py); });
  x.stroke();
  x.fillStyle = "rgba(142,163,196,.9)"; x.font = `${11 * devicePixelRatio}px ui-monospace,monospace`;
  x.fillText("band-passed PPG (0.7–3.6 Hz) · 12 s", 8 * devicePixelRatio, 14 * devicePixelRatio);
}
function drawGauge(p) {
  const c = $("#gauge"), x = c.getContext("2d"), w = c.width, h = c.height, cx = w / 2, cy = h * 0.86, r = Math.min(w, h * 1.15) * 0.44;
  x.clearRect(0, 0, w, h);
  x.lineWidth = 12 * devicePixelRatio; x.lineCap = "round";
  x.strokeStyle = "#152341"; x.beginPath(); x.arc(cx, cy, r, Math.PI, 2 * Math.PI); x.stroke();
  if (p == null) return;
  const segs = [[.10, "#4aa3ff"], [.25, "#5ad1a5"], [.45, "#ffd83d"], [.70, "#ff9f2e"], [1.01, "#ff4d5e"]];
  let from = 0;
  for (const [to, col] of segs) {
    const end = Math.min(p, to);
    if (end > from) { x.strokeStyle = col; x.beginPath();
      x.arc(cx, cy, r, Math.PI + (from / 1) * Math.PI, Math.PI + (end / 1) * Math.PI); x.stroke(); }
    from = to;
  }
  const a = Math.PI + p * Math.PI;
  x.strokeStyle = "#fff"; x.lineWidth = 2 * devicePixelRatio; x.beginPath();
  x.moveTo(cx, cy); x.lineTo(cx + Math.cos(a) * (r - 12), cy + Math.sin(a) * (r - 12)); x.stroke();
}

/* ---------------------------------------------------------------- ward --- */
function ensurePick() {
  // Created eagerly, never inside a try: submit() reads it, so a failed /api/patients
  // call used to leave the control absent and turn every button into a TypeError.
  if ($("#ptPick")) return $("#ptPick");
  const l = document.createElement("label");
  l.innerHTML = `Save reading to<select id="ptPick"><option value="">don't save</option></select>`;
  $(".fields").append(l);
  return $("#ptPick");
}
async function loadPatientsIntoSelect() {
  const sel = ensurePick(); const cur = sel.value;
  try {
    const ps = await api("/api/patients");
    sel.innerHTML = `<option value="">don't save</option>` +
      ps.map((p) => `<option value="${p.id}">${esc(p.name)} · ${esc(p.mrn)}</option>`).join("");
    if (cur) sel.value = cur;
  } catch (e) { sel.innerHTML = `<option value="">don't save</option>`; diag(`patient list unavailable: ${e.message}`); }
}
async function loadWard() {
  const d = await api("/api/snapshot");
  const s = d.stats;
  $("#wardStats").textContent = `${s.patients} patients · ${s.readings} readings · ${s.open_alerts} open alerts`;
  $("#ptBody").innerHTML = d.patients.map((p) => {
    const t = p.tier || "none";
    return `<tr data-id="${p.id}" class="${S.sel === p.id ? "sel" : ""}">
      <td><b>${esc(p.name)}</b> <span style="color:#8ea3c4">${p.age_years ?? "?"}</span></td>
      <td style="color:#8ea3c4;font-family:var(--mono)">${esc(p.mrn)}</td><td>${esc(p.ward || "")}</td>
      <td class="num">${p.hr ?? "—"}</td>
      <td class="num">${p.spo2 ?? "—"}%</td><td class="num">${p.rr ?? "—"}</td>
      <td class="num">${p.sbp ?? "—"}/${p.dbp ?? "—"}</td><td class="num">${p.temp_c ?? "—"}°</td>
      <td class="num">${p.news2 ?? "—"}</td>
      <td><span class="chip ${t}">${p.risk_score != null ? (p.risk_score * 100).toFixed(0) + "%" : "no data"}</span></td>
      <td>${p.open_alerts ? `<span class="chip critical">${p.open_alerts}</span>` : '<span class="chip none">0</span>'}</td>
      <td style="color:#4aa3ff;font-size:16px">›</td></tr>`;
  }).join("") || `<tr><td colspan="12" style="color:#8ea3c4">Empty database — click “Load demo ward”.</td></tr>`;
  $$("#ptBody tr").forEach((tr) => tr.addEventListener("click", () => openPatient(+tr.dataset.id, tr)));
  if (d.patients.length && !S.sel) openPatient(d.patients[0].id);
  $("#alertBadge").textContent = s.open_alerts;
}
async function openPatient(id, tr) {
  S.sel = id; $$("#ptBody tr").forEach((x) => x.classList.toggle("sel", x === tr));
  const [hist, pats] = await Promise.all([api(`/api/patients/${id}/history?limit=24`), api("/api/patients")]);
  const p = pats.find((x) => x.id === id);
  const h = hist[0];
  $("#detName").innerHTML = `${esc(p.name)} <small>${esc(p.mrn)} · ${esc(p.ward || "")} · ${p.age_years ?? "?"} yrs · ${hist.length} stored readings · last ${h ? h.ts.slice(0, 19).replace("T", " ") : "—"}</small>`;
  if (!h) { $("#detBody").innerHTML = `<div class="empty">No readings stored for this patient.</div>`; return; }
  const ex = h.explanation || {};
  $("#detBody").innerHTML = `
    <div class="metrics">
      <div class="metric"><span class="k">Risk</span><span class="v" style="color:${TIERCOL[ex.tier] || "#4aa3ff"}">${(h.risk_score * 100).toFixed(0)}%</span><span class="u">${esc(h.tier || "")}</span></div>
      <div class="metric"><span class="k">HR</span><span class="v">${h.hr ?? "—"}</span><span class="u">BPM · q ${h.signal_quality ?? "—"}</span></div>
      <div class="metric"><span class="k">NEWS2</span><span class="v">${h.news2 ?? "—"}</span><span class="u">reference</span></div>
      <div class="metric"><span class="k">SpO₂</span><span class="v">${h.spo2 ?? "—"}</span><span class="u">% · RR ${h.rr ?? "—"}</span></div>
    </div>
    <canvas id="trend" width="900" height="170" style="width:100%;height:170px;background:#0a1424;border:1px solid var(--line);border-radius:10px"></canvas>
    <div class="row" style="margin:10px 0 4px">
      <button class="btn primary" id="btnRescore">Re-score with new vitals</button>
      <button class="btn" id="btnPhone">Measure HR by camera now</button>
    </div>
    <h4>Latest explanation</h4>
    <div id="detFactors" class="factors"></div>
    <div class="flags">${(ex.red_flags || []).map((f) => `<span class="flag">${esc(f.text)}</span>`).join("")}</div>
    <h4>Trend table</h4>
    <div class="tablewrap"><table><thead><tr><th>time</th><th>src</th><th>HR</th><th>SpO₂</th><th>RR</th><th>BP</th><th>Temp</th><th>Q</th><th>NEWS2</th><th>risk</th></tr></thead>
    <tbody>${hist.map((x) => `<tr><td class="num">${x.ts.slice(5, 16).replace("T", " ")}</td><td>${esc(x.source || "")}</td>
      <td class="num">${x.hr ?? "—"}</td><td class="num">${x.spo2 ?? "—"}</td><td class="num">${x.rr ?? "—"}</td>
      <td class="num">${x.sbp ?? "—"}/${x.dbp ?? "—"}</td><td class="num">${x.temp_c ?? "—"}</td>
      <td class="num">${x.signal_quality ?? "—"}</td><td class="num">${x.news2 ?? "—"}</td>
      <td><span class="chip ${x.tier || "none"}">${(x.risk_score * 100).toFixed(0)}%</span></td></tr>`).join("")}</tbody></table></div>`;
  drawTrend(hist);
  renderFactors(ex.top_factors || [], $("#detFactors"));
  $("#btnRescore").onclick = async () => {
    const v = collectVitals();
    try { await post(`/api/patients/${id}/score`, { ...v, signal_quality: h.signal_quality, source: "manual" });
      $("#btnRescore").textContent = "Saved ✓"; setTimeout(() => ($("#btnRescore").textContent = "Re-score with new vitals"), 1200);
      loadWard(); } catch (e) { alert(e.message); }
  };
  $("#btnPhone").onclick = () => {
    $$(".tab")[0].click();                       // jump to Capture
    $("#ptPick").value = String(id);              // pre-select this patient
    toast(`Recording will be saved to ${p.name}. Press Start camera, then Record 12 s.`);
  };
}
function toast(m) { $("#progTxt").textContent = m; }
function drawTrend(hist) {
  const c = $("#trend"); if (!c) return; const x = c.getContext("2d");
  const W = c.width = 900 * devicePixelRatio, H = c.height = 170 * devicePixelRatio;
  const rows = hist.slice().reverse();
  const series = [{ k: "hr", col: "#ff4d5e", lo: 30, hi: 190 }, { k: "spo2", col: "#4aa3ff", lo: 80, hi: 100 }, { k: "rr", col: "#ffd83d", lo: 6, hi: 40 }];
  x.clearRect(0, 0, W, H); x.strokeStyle = "rgba(255,255,255,.06)";
  for (let i = 0; i <= 4; i++) { const yy = (i / 4) * H; x.beginPath(); x.moveTo(0, yy); x.lineTo(W, yy); x.stroke(); }
  const n = rows.length;
  series.forEach((s) => {
    const vals = s.k === "hr" ? rows.map((r) => r.hr) : s.k === "spo2" ? rows.map((r) => r.spo2) : rows.map((r) => r.rr);
    const mn = Math.min(...vals.filter((v) => v != null)), mx = Math.max(...vals.filter((v) => v != null));
    const lo = mn ?? s.lo, hi = (mx ?? s.hi) || (lo + 1);
    x.strokeStyle = s.col; x.lineWidth = 2 * devicePixelRatio; x.beginPath(); let started = false;
    vals.forEach((v, i) => { if (v == null) return; const px = (i / Math.max(1, n - 1)) * (W - 20) + 10,
      py = H - 14 - ((v - lo) / (hi - lo || 1)) * (H - 28);
      started ? x.lineTo(px, py) : (x.moveTo(px, py), started = true); });
    x.stroke();
    const li = vals.length - 1, last = vals[li];
    x.fillStyle = s.col; x.font = `${11 * devicePixelRatio}px ui-sans-serif`;
    x.fillText(`${s.k}${last != null ? " " + (hi === lo ? last : last.toFixed(0)) : ""}`, 12 * devicePixelRatio, (series.indexOf(s) + 1) * 15 * devicePixelRatio);
  });
  x.fillStyle = "rgba(142,163,196,.75)"; x.font = `${10.5 * devicePixelRatio}px ui-monospace`;
  x.fillText(`${n} readings · oldest → newest`, 12 * devicePixelRatio, H - 5);
}

/* -------------------------------------------------------------- alerts --- */
async function loadAlerts() {
  const a = await api("/api/alerts");
  $("#alertList").innerHTML = a.length ? a.map((x) => `<div class="alert ${esc(x.tier)} ${x.acknowledged_at ? "done" : ""}">
      <div style="min-width:0"><div class="who">${esc(x.name)} <span style="color:#8ea3c4;font-weight:400">${esc(x.mrn)} · ${esc(x.ward || "")}</span></div>
      <div class="msg">${esc(x.message)}</div><div class="when">${esc(x.ts.replace("T", " ").slice(0, 19))} ${x.acknowledged_at ? "· ack by " + esc(x.acknowledged_by) : ""}</div></div>
      <div class="sp"><span class="chip ${esc(x.tier)}">${esc(x.tier)}</span></div>
      ${x.acknowledged_at ? "" : `<button class="btn ghost" data-ack="${x.id}">Acknowledge</button>`}</div>`).join("")
    : `<div class="empty">No alerts yet. Capture a reading with a high-risk profile (or load the demo ward) and one appears here automatically.</div>`;
  $$("#alertList [data-ack]").forEach((b) => b.onclick = async () => { await post(`/api/alerts/${b.dataset.ack}/ack`); loadAlerts(); loadWard(); });
  const st = await api("/api/health"); $("#alertBadge").textContent = st.db?.open_alerts ?? 0;
}

/* --------------------------------------------------------------- model --- */
async function loadModel() {
  try {
    const d = await api("/api/metrics"), m = d.metrics, g = d.global_shap || {};
    $("#metricsBody").innerHTML = Object.entries(m).map(([k, v]) =>
      `<div class="box"><span>${esc(k.replace(/_/g, " "))}</span><b>${typeof v === "number" ? (v < 1.5 ? v.toFixed(3) : v) : esc(v)}</b></div>`).join("")
      + `<div class="box" style="grid-column:1/-1;border-color:#26385c"><span>honesty note</span><b style="font-size:13px;font-family:ui-sans-serif;font-weight:400;color:#c9daf7">Labels are NEWS2-derived synthetic data — ${m.news2_only_roc_auc ? `the plain NEWS2 rule alone scores ${m.news2_only_roc_auc} AUC on the same target` : ""}, which is the ceiling a rule-trained model can reach. Swap in MIMIC-IV/eICU via <code style="font-family:var(--mono);color:#8cc4ff">python train.py --dataset ward.csv --target deteriorated_24h</code> and these numbers become the real claim.</b></div>`;
    const imp = Object.entries(g.mean_abs || {});
    const mx = Math.max(...imp.map(([, v]) => v), 0.001);
    $("#impBody").innerHTML = imp.map(([k, v]) => {
      const dir = (g.direction || {})[k] || 0, pos = dir > 0, w = (v / mx) * 50;
      return `<div class="factor"><span class="lab">${esc(k)} <em>${pos ? "↑" : "↓"} risk</em></span>
        <span class="track"><span class="fill" style="${pos ? "left:50%" : "right:50%"};width:${w.toFixed(1)}%;background:${pos ? "var(--crit)" : "var(--low)"}"></span></span>
        <span class="num">${v.toFixed(3)}</span></div>`;
    }).join("") || `<div class="empty">${esc(g.error || "no global SHAP - retrain")}</div>`;
  } catch (e) { $("#metricsBody").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}

/* ------------------------------------------------------------- snapshot --- */
async function refreshSnapshot() {
  try { const d = await api("/api/snapshot"); $("#alertBadge").textContent = d.stats.open_alerts;
    $("#connDot").classList.add("on"); $("#connTxt").textContent = "API + SQLite online"; }
  catch { $("#connDot").classList.remove("on"); $("#connTxt").textContent = "API offline"; }
}
$("#btnCam").onclick = startCamera; $("#btnFlash").onclick = toggleFlash; $("#btnRec").onclick = record;
$("#btnSim").onclick = runSim; $("#btnRefresh").onclick = loadWard; $("#btnSeed").onclick = async (e) => {
  e.target.disabled = true; e.target.textContent = "seeding ward…";
  try { await post("/api/seed", {}); await loadWard(); } finally { e.target.disabled = false; e.target.textContent = "Load demo ward (14 patients)"; }
};
document.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "r" && e.shiftKey && !S.recording && $("#tab-capture").classList.contains("active")) record();
});
/* deep links: /#capture, /#ward, /#alerts, /#model - so a phone that opens the app
   from a QR code lands on the right tab, and back/forward work. */
function activate(tab) {
  const btn = $$(".tab").find((b) => b.dataset.tab === tab);
  if (btn) btn.click();
}
addEventListener("hashchange", () => activate(location.hash.replace("#", "")));
window.__vg = S;   // inspection handle for tests and console debugging
window.__vg.sim = simulateFrames;   // lets tests sweep the synthetic generator directly
sizeCanvas(); loadPatientsIntoSelect(); refreshSnapshot(); setInterval(refreshSnapshot, 5000);
if (location.hash) activate(location.hash.replace("#", ""));
})();
