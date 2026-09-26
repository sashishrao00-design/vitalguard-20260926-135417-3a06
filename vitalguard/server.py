"""
VitalGuard AI - FastAPI server (step 10: connect everything).

    uvicorn vitalguard.server:app --host 0.0.0.0 --port 8000

Three kinds of client hit the same endpoints:
  * the browser capture page   -> POST /api/ppg/process  (per-frame green means)
  * the Android CameraX app    -> POST /api/ppg/process  (same payload) or
                                  POST /api/ppg/video    (uploaded .mp4/.avi)
  * the dashboard / Streamlit  -> everything under /api/*
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .explain import explain, tier_for
from .heart_rate import analyze
from .ppg import clean_signal, series_from_video, synthetic_recording, write_demo_video
from .risk_model import FEATURES, MODEL_DIR, engineer, explain_row, load

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web"

app = FastAPI(
    title="VitalGuard AI",
    version="0.1.0",
    description="Camera PPG -> heart rate -> deterioration risk -> SHAP explanation -> alerts",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def no_stale_assets(request, call_next):
    """Always revalidate the frontend.

    Without a Cache-Control header browsers apply heuristic caching to /static/*,
    which meant a phone that had already loaded the old app.js kept running it -
    the classic "I fixed it but it still does not work" on a demo Wi-Fi. ETag
    revalidation costs one 304 and guarantees the newest file.
    """
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/") or path == "/phone":
        resp.headers["Cache-Control"] = "no-cache"
    return resp

_model = load()


@app.on_event("startup")
def _startup() -> None:
    """Initialise SQLite whether launched via `uvicorn vitalguard.server:app`
    or via `python -m vitalguard.server` - the schema must exist before the
    first request arrives."""
    db.init()


def get_model():
    global _model
    if _model is None:
        raise HTTPException(503, "model not trained yet - run: python train.py")
    return _model


def score_vitals(vitals: dict) -> tuple[float, dict]:
    m = get_model()
    feats = engineer(vitals)
    x = np.array([[feats[f] for f in FEATURES]], dtype=float)
    p = float(m.predict_proba(x)[0, 1])
    return p, explain(vitals, p, explain_row(m, x))


# ---------------------------------------------------------------------------
class Frame(BaseModel):
    t: float = Field(..., allow_inf_nan=False, description="seconds since capture start (performance.now()/1000)")
    v: float = Field(..., allow_inf_nan=False, description="mean intensity of the ROI for that frame")
    luma: float | None = Field(None, allow_inf_nan=False, description="mean luma, for exposure rejection")


class ProcessRequest(BaseModel):
    frames: list[Frame]
    patient_id: int | None = None
    vitals: dict[str, Any] = Field(default_factory=dict)
    save: bool = True
    source: Literal["camera", "simulation"] = "camera"
    expected_hr_bpm: float | None = Field(None, gt=0, le=300, allow_inf_nan=False)


@app.get("/api/health")
def health() -> dict:
    ready = _model is not None
    return {
        "status": "ok",
        "model_ready": ready,
        "app_version": "signal-safety-2",
        "model_artifact": str(MODEL_DIR / "risk_model.joblib"),
        "features": FEATURES,
        "db": db.stats() if Path(db.DB_PATH).exists() else {"patients": 0, "readings": 0},
    }


@app.get("/api/metrics")
def metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "no metrics - run python train.py first")
    return json.loads(p.read_text())


# --------------------------------------------------------- steps 2-5: PPG ---
@app.post("/api/ppg/process")
def ppg_process(req: ProcessRequest) -> dict:
    if len(req.frames) < 16:
        raise HTTPException(422, f"need at least 16 frames, received {len(req.frames)}")
    if req.source == "simulation" and req.expected_hr_bpm is None:
        raise HTTPException(422, "simulation requires its known expected_hr_bpm")
    if req.source == "camera" and req.expected_hr_bpm is not None:
        raise HTTPException(422, "expected_hr_bpm is only allowed for synthetic simulations")
    t = np.array([f.t for f in req.frames], dtype=float)
    v = np.array([f.v for f in req.frames], dtype=float)
    if req.frames[0].luma is not None:
        v = np.array([f.luma if f.luma is not None else f.v for f in req.frames], dtype=float)
    try:
        ppg = clean_signal(t, v)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    res = analyze(ppg)
    out: dict[str, Any] = {
        "heart_rate": res.to_dict(),
        "source": req.source,
        "signal": {
            "duration_s": round(ppg.duration, 2), "frames": ppg.n_frames_used,
            "raw_mean_luma": round(ppg.raw_mean, 1), "ac_dc_ratio": round(ppg.ac_dc_ratio, 5),
            "fs": round(ppg.fs, 2), "exposure": ppg.rejected,
        },
        "waveform": {
            "t": [round(float(x - ppg.t[0]), 4) for x in ppg.t],
            "y": [round(float(y), 5) for y in ppg.y],
        },
    }

    snr_db = res.flags.get("spectral_snr_db", -99)
    demo_ok = True
    if req.source == "simulation":
        error = abs(res.hr_bpm - req.expected_hr_bpm) if res.hr_bpm is not None else None
        demo_ok = error is not None and error <= 3.0
        out["simulation_check"] = {"expected_hr_bpm": req.expected_hr_bpm,
                                   "error_bpm": round(error, 3) if error is not None else None,
                                   "tolerance_bpm": 3.0, "passed": demo_ok}
    # Expected truth is an integrity CHECK, never substituted for the estimator.
    # It only exists for synthetic data. Real camera readings do not get clamped.
    out["accepted"] = bool(res.hr_bpm and res.quality >= 60 and snr_db >= 6.0 and demo_ok)
    if out["accepted"]:
        vitals = dict(req.vitals)
        vitals["hr"] = res.hr_bpm
        p, ex = score_vitals(vitals)
        out["vitals"] = {k: vitals.get(k) for k in ("hr", "spo2", "rr", "sbp", "dbp", "temp_c")}
        out["risk"] = ex
        out["risk"]["risk_score"] = round(p, 4)
        if req.source == "camera" and req.save and req.patient_id is not None:
            rec = db.add_reading(req.patient_id, vitals, ex, signal_quality=res.quality, source="phone")
            out["saved_reading_id"] = rec["id"]
    else:
        out["risk"] = None
        out["retry_required"] = True
        why = []
        # Keep rejected estimates only in diagnostics, never as a measured HR.
        out["heart_rate"]["flags"]["candidate_hr_bpm"] = res.hr_bpm
        for key in ("hr_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "breathing_rate_bpm"):
            out["heart_rate"][key] = None
        if not demo_ok:
            why.append("synthetic signal estimate disagrees with its known input by more than 3 BPM")
        if not res.hr_bpm:
            why.append("no pulse frequency was found")
        else:
            if res.quality < 60:
                why.append(f"quality {res.quality}/100 below the 60 required to trust a heart rate")
            if snr_db < 6.0:
                why.append(f"pulse peak only {snr_db} dB above the noise floor (needs 6 dB)")
        out["message"] = ("Not scored: " + "; ".join(why) +
                          ". The AI stage is skipped rather than fed an unreliable BPM - reposition and retake.")
    return out


@app.post("/api/ppg/video")
async def ppg_video(file: UploadFile = File(...), patient_id: int | None = Form(None),
                    vitals: str = Form("{}")) -> dict:
    """Upload path used by the Android app when it records instead of streaming."""
    tmp = Path(db.DB_PATH).parent / f"upload_{file.filename or 'clip.mp4'}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    try:
        t, v = series_from_video(tmp)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    ppg = clean_signal(t, v)
    res = analyze(ppg)
    payload = {"frames": [Frame(t=float(a), v=float(b)) for a, b in zip(t, v)],
               "patient_id": patient_id, "vitals": json.loads(vitals or "{}"), "save": True}
    return {"source_video": str(tmp), **ppg_process(ProcessRequest(**payload))}


@app.api_route("/api/demo/fingertip", methods=["GET", "POST"])
def demo_fingertip(n: int = 20, seed: int = 1) -> dict:
    """Synthesise a fingertip recording server-side and push it through the real
    OpenCV path. Lets the whole pipeline demo with zero camera permissions."""
    info = write_demo_video(Path(db.DB_PATH).parent / f"synthetic_{seed}.avi",
                            hr_bpm=float(n + 50), seconds=12.0, fps=20, seed=seed)
    t, v = series_from_video(info["path"])
    ppg = clean_signal(t, v)
    res = analyze(ppg)
    return {"video": info, "true_hr_bpm": info["true_hr_bpm"], "heart_rate": res.to_dict(),
            "error_bpm": round(abs((res.hr_bpm or 0) - info["true_hr_bpm"]), 2)}


@app.post("/api/simulate")
def simulate(hr_bpm: float = 112.0, seconds: float = 12.0, snr_db: float = 14.0,
             motion_bpm: float = 0.0) -> dict:
    """Fast synthetic PPG (no video file) - used by the 'simulate' button."""
    t, v = synthetic_recording(hr_bpm=hr_bpm, seconds=seconds, snr_db=snr_db,
                               motion_bpm=motion_bpm, seed=int(hr_bpm * 7 + motion_bpm))
    ppg = clean_signal(t, v)
    res = analyze(ppg)
    return {"true_hr_bpm": hr_bpm, "heart_rate": res.to_dict(),
            "error_bpm": round(abs((res.hr_bpm or 0) - hr_bpm), 2),
            "waveform": {"t": [round(float(x - ppg.t[0]), 4) for x in ppg.t],
                         "y": [round(float(y), 5) for y in ppg.y]}}


# ------------------------------------------------------------ step 6: data --
class PatientIn(BaseModel):
    mrn: str
    name: str
    age_years: float | None = None
    sex: str = "M"
    ward: str = "General"
    unit: str = "ward"
    chronic_score: float = 0


@app.get("/api/patients")
def patients() -> list[dict]:
    return db.list_patients()


@app.post("/api/patients")
def new_patient(p: PatientIn) -> dict:
    return db.add_patient(p.mrn, p.name, p.age_years, p.sex, p.ward, p.unit, p.chronic_score)


@app.get("/api/patients/{pid}/history")
def patient_history(pid: int, limit: int = 50) -> list[dict]:
    return db.history(pid, limit)


class VitalsIn(BaseModel):
    spo2: float | None = None
    rr: float | None = None
    sbp: float | None = None
    dbp: float | None = None
    temp_c: float | None = None
    hr: float | None = None
    conscience: float | None = 0
    signal_quality: float | None = None
    source: str = "manual"
    save: bool = True


@app.post("/api/patients/{pid}/score")
def score_patient(pid: int, body: VitalsIn = Body(...)) -> dict:
    pats = {p["id"]: p for p in db.list_patients()}
    if pid not in pats:
        raise HTTPException(404, "unknown patient_id")
    pat = pats[pid]
    vitals = body.model_dump(exclude={"save", "source", "signal_quality"})
    vitals.update({"age_years": pat.get("age_years"), "sex": pat.get("sex"),
                   "unit": pat.get("unit"), "chronic_score": pat.get("chronic_score") or 0})
    if vitals.get("hr") is None:
        last = db.history(pid, 1)
        if last:
            vitals["hr"] = last[0]["hr"]
    p, ex = score_vitals(vitals)
    out = {"risk": ex, "vitals": vitals}
    if body.save:
        rec = db.add_reading(pid, vitals, ex, signal_quality=body.signal_quality, source=body.source)
        out["saved_reading_id"] = rec["id"]
    return out


# ------------------------------------------------------------- step 9: alerts
@app.get("/api/alerts")
def alerts(open_only: bool = False, limit: int = 100) -> list[dict]:
    return db.alerts(limit=limit, open_only=open_only)


@app.post("/api/alerts/{alert_id}/ack")
def ack(alert_id: int, by: str = "clinician") -> dict:
    return {"acknowledged": db.ack_alert(alert_id, by)}


@app.post("/api/seed")
def seed(n: int = 14, seed: int = 3) -> dict:
    """Populate a believable ward so the dashboard is never empty on first open."""
    from vitalguard.risk_model import make_dataset, COHORTS

    rng = np.random.default_rng(seed)
    names = list(COHORTS)
    first = ["Aisha", "Ravi", "Meera", "John", "Fatima", "Chen", "Olga", "Samuel",
             "Priya", "Lena", "Marcus", "Zara", "Tom", "Ingrid", "Kwame", "Yuki",
             "Nadia", "Peter", "Sunita", "Diego"]
    last = ["Rahman", "Kumar", "Nair", "Boyle", "Sayed", "Wei", "Petrov", "Osei",
            "Sharma", "Berg", "Webb", "Haddad", "Novak", "Faure", "Mensah", "Tanaka"]
    created = []
    for i in range(n):
        name = f"{first[i % len(first)]} {last[(i * 7) % len(last)]}"
        mrn = f"MRN-{1000 + seed * 100 + i}"
        cohort = names[int(rng.integers(0, len(names)))]
        age = float(np.clip(rng.normal(70, 14), 22, 95))
        pat = db.add_patient(mrn, name, round(age, 0), "M" if rng.random() < 0.5 else "F",
                             f"Ward {rng.choice(['2A', '3B', '4C', 'ICU'])}",
                             "icu" if cohort in ("sepsis", "shock") and rng.random() < 0.5 else "ward",
                             float(rng.poisson(1.0 if cohort != "stable" else 0.3)))
        for k in range(int(rng.integers(2, 6))):
            hz, sv = rng.normal(0, 1), rng.normal(0, 1)
            (h_mu, h_s), (o_mu, o_s), (r_mu, r_s), (s_mu, s_s), (t_mu, t_s), base_p = {
                "stable": ((76, 10), (97, 1.3), (16, 2.4), (124, 12), (36.7, 0.35), 0.03),
                "sepsis": ((116, 18), (92, 2.6), (28, 4.6), (93, 12), (38.7, 0.9), 0.55),
                "respiratory": ((96, 14), (88, 3.8), (26, 5.0), (119, 13), (37.2, 0.5), 0.45),
                "shock": ((121, 16), (93, 2.6), (25, 4.6), (80, 9), (36.1, 0.7), 0.65),
                "hypertensive_renal": ((88, 11), (95, 1.9), (17, 2.8), (174, 16), (36.9, 0.4), 0.18),
            }[cohort]
            vitals = {
                "hr": round(float(h_mu + h_s * sv), 1),
                "spo2": None if rng.random() < 0.12 else round(float(np.clip(o_mu - o_s * sv, 74, 100)), 1),
                "rr": None if rng.random() < 0.12 else round(float(max(8, r_mu + r_s * sv * hz)), 1),
                "sbp": round(float(s_mu - s_s * sv), 1),
                "dbp": None if rng.random() < 0.3 else round(float(s_mu * 0.63 - s_s * 0.3 * sv), 1),
                "temp_c": round(float(t_mu + t_s * hz), 1),
                "age_years": age, "sex": pat["sex"], "unit": pat["unit"],
                "chronic_score": pat["chronic_score"], "conscience": 1.0 if rng.random() < base_p * 0.25 else 0.0,
            }
            # let a third of readings carry an HR measured by the camera instead
            if rng.random() < 0.34:
                t, vv = synthetic_recording(hr_bpm=vitals["hr"], seconds=12, fps=20,
                                            snr_db=float(rng.normal(15, 3)), seed=int(rng.integers(0, 10**6)))
                res = analyze(clean_signal(np.asarray(t), np.asarray(vv)))
                if res.hr_bpm and res.quality >= 45:
                    vitals["hr"] = res.hr_bpm
                    pat["_q"] = res.quality
            p_, ex_ = score_vitals(vitals)
            db.add_reading(pat["id"], vitals, ex_, signal_quality=pat.get("_q", round(float(rng.normal(82, 9)), 1)),
                           source="phone" if "_q" in pat else "manual")
            if k == 0:  # one line per patient (first reading), not per reading
                created.append({"mrn": mrn, "name": name, "risk": ex_["risk_percent"], "tier": ex_["tier"]})
    return {"patients": n, "detail": created, "stats": db.stats()}



def _qr_data_uri(url: str) -> str:
    """Render a QR code for `url` as an inline data URI (no network deps in the page)."""
    try:
        import base64
        import io

        import qrcode
        import qrcode.constants

        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        buf = io.BytesIO()
        qr.make_image(fill_color="#0b1220", back_color="white").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


@app.get("/phone", response_class=HTMLResponse)
def phone(request: Request) -> str:
    """Scan-this-to-open-on-your-phone page.

    The origin is taken from the request headers, so the QR always encodes the URL
    the browser actually used to reach us - the Cloudflare tunnel, the sandbox
    preview host, or localhost - without any hardcoded address.
    """
    fwd_host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    # Prefer the proxy's view of the scheme, then the ASGI-detected one. The known
    # preview/tunnel hosts always terminate TLS upstream, so the browser is on https
    # even though the app sees plain http - forcing it there is what keeps camera
    # access working, and falling back to request.url.scheme keeps localhost honest.
    scheme = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    if (fwd_host.split(":", 1)[0].endswith((".trycloudflare.com", ".e2b.app", ".lhr.life", ".onrender.com"))):
        scheme = "https"
    origin = f"{scheme}://{fwd_host}" if fwd_host else ""
    target = origin + "/#capture"
    qr = _qr_data_uri(target)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VitalGuard AI - open on your phone</title>
<style>
 body{{margin:0;min-height:100vh;background:radial-gradient(900px 600px at 20% -10%,#132340,#070c16 60%);
  color:#e6efff;font:15px/1.55 ui-sans-serif,system-ui,sans-serif;display:grid;place-items:center;padding:24px}}
 .c{{max-width:720px;text-align:center;background:#0d1524;border:1px solid #1e2d47;border-radius:18px;padding:26px 26px 20px}}
 h1{{margin:0 0 4px;font-size:22px}}h1 span{{color:#4aa3ff}}
 p{{color:#8ea3c4;margin:6px 0 18px;font-size:13.5px}}
 img{{width:min(340px,72vw);image-rendering:pixelated;background:#fff;border-radius:12px;padding:10px;box-sizing:border-box}}
 a.u{{display:inline-block;margin-top:14px;color:#8cc4ff;font-family:ui-monospace,monospace;font-size:13px;
   word-break:break-all;text-decoration:none;border-bottom:1px dashed #33507d}}
 ol{{text-align:left;color:#c9daf7;font-size:13.5px;max-width:520px;margin:18px auto 0;padding-left:20px}}
 li{{margin:5px 0}} code{{background:#0a1424;padding:1px 5px;border-radius:5px;font-size:12.5px}}
 .w{{margin-top:16px;font-size:12px;color:#ffd83d}}
</style></head><body><div class="c">
 <h1>VitalGuard <span>AI</span> &middot; phone capture</h1>
 <p>Point your camera at this code to open the live prototype on your phone.</p>
 {f'<img src="{qr}" alt="QR code for {target}">' if qr else '<p style="color:#ffd83d">pip install qrcode to render a code</p>'}
 <br><a class="u" href="{target}">{target}</a>
 <ol>
  <li>Tap <b>Start camera</b>, then allow camera access.</li>
  <li>Put your fingertip over the lens so it covers the whole dashed circle, torch on.</li>
  <li>Tap <b>Record 12 s</b> and hold still - heart rate, quality, risk and the SHAP
      explanation come back from the same server path the app uses.</li>
  <li>Enter SpO<sub>2</sub>/BP/temp if you have them, or just score from the camera alone.</li>
  <li>Tick <b>shaky finger</b> then <b>Simulate fingertip</b> to watch the quality gate refuse a bad read.</li>
 </ol>
 <p class="w">HTTPS is required for camera access &mdash; this link is https, so it works.
    Data entered here is written to the shared demo database.</p>
</div></body></html>"""


# ------------------------------------------------------------------- static -
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/snapshot")
def snapshot() -> dict:
    """One call with everything the dashboard needs, for 5 s polling."""
    return {"patients": db.list_patients(), "alerts": db.alerts(limit=40), "stats": db.stats()}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def init_db() -> None:
    db.init()


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
