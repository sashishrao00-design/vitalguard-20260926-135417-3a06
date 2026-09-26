#!/usr/bin/env python3
"""End-to-end proof of the whole chain, in one command, no browser:

    python demo.py

Video file -> OpenCV ROI -> green-channel series -> SciPy band-pass -> peak
detection -> BPM + signal-quality gate -> HistGB risk model -> SHAP factors ->
SQLite row -> alert. Prints a report you can put straight in the submission.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vitalguard import db  # noqa: E402
from vitalguard.explain import explain  # noqa: E402
from vitalguard.heart_rate import analyze  # noqa: E402
from vitalguard.ppg import (clean_signal, series_from_video, synthetic_recording,  # noqa: E402
                           write_demo_video)
from vitalguard.risk_model import FEATURES, MODEL_DIR, engineer, explain_row, load  # noqa: E402

LINE = "─" * 74


def rule(t=""):
    print(f"\n{LINE}\n {t}\n{LINE}")


def main() -> int:
    t_start = time.time()
    rule("VitalGuard AI · end-to-end demo")
    print(" chain: fingertip video → PPG → heart rate → quality gate → AI risk → SHAP → SQLite → alert")

    model = load()
    if model is None:
        print("\n model not found - training now ...")
        from vitalguard.risk_model import train
        tr = train(n=6000)
        print(f" trained: cv_roc_auc={tr.metrics['cv_roc_auc']} prevalence={tr.metrics['prevalence']:.3f}")
        model = load()
    else:
        print(f" model loaded: {MODEL_DIR / 'risk_model.joblib'}")

    # ---------------------------------------------------------------- step 1
    rule("step 1-2 · camera → OpenCV frames → mean green intensity")
    vid = Path("artifacts") / "demo_finger.avi"
    truth = 96.0
    info = write_demo_video(vid, hr_bpm=truth, seconds=12.0, fps=20, seed=21)
    t, v = series_from_video(vid)
    print(f" rendered {info['frames']} frames at {info['fps']} fps -> {vid.name}")
    print(f" decoded {len(t)} frames, ROI green intensity {v.min():.1f}-{v.max():.1f} (255 scale)")

    # ------------------------------------------------------------ step 3-5
    rule("step 3-5 · band-pass filter → peaks → BPM → quality gate")
    ppg = clean_signal(t, v)
    res = analyze(ppg)
    err = abs((res.hr_bpm or 0) - truth)
    print(f" fs={res.flags['fs']} Hz · duration {res.flags['duration_s']} s · {res.n_peaks} beats found")
    print(f" true HR      {truth:6.1f} BPM")
    print(f" estimated    {res.hr_bpm:6.1f} BPM   error {err:.2f} BPM")
    print(f"   ├─ spectral (Welch)      {res.hr_spectral_bpm:6.1f}")
    print(f"   ├─ autocorrelation       {res.hr_temporal_bpm:6.1f}")
    print(f"   └─ peak intervals        {(res.hr_from_peaks_bpm or 0):6.1f}   spread {res.agree_bpm} BPM")
    print(f" quality gate {res.quality}/100 → '{res.quality_label}' (score is skipped below 60)")
    print(f" HRV RMSSD {res.hrv_rmssd_ms} ms · breathing rate {res.breathing_rate_bpm} /min")
    print(f" perfusion proxy AC/DC {ppg.ac_dc_ratio:.4f} · spectral SNR {res.flags['spectral_snr_db']} dB")

    # ------------------------------------------------- step 5 negative test
    rule("step 5 · rejection path (finger lifted: no pulse present)")
    rng = np.random.default_rng(4)
    tt = np.arange(240) / 20.0
    vv = 140 + rng.normal(0, 0.6, 240)
    bad = analyze(clean_signal(tt, vv))
    print(f" quality {bad.quality}/100 → '{bad.quality_label}' → gated out")
    for g in bad.guidance[:2]:
        print(f"   guidance: {g}")

    # ------------------------------------------------------------ step 6-8
    rule("step 6-8 · vitals assembled → HistGB risk score → SHAP explanation")
    vitals = {"hr": res.hr_bpm, "spo2": 91.0, "rr": 26.0, "sbp": 99.0, "dbp": 61.0,
              "temp_c": 38.4, "age_years": 73.0, "sex": "M", "unit": "ward",
              "chronic_score": 2.0, "conscience": 0}
    feats = engineer(vitals)
    x = np.array([[feats[f] for f in FEATURES]], dtype=float)
    p = float(model.predict_proba(x)[0, 1])
    ex = explain(vitals, p, explain_row(model, x))
    print(f" input: HR {vitals['hr']} (from PPG) SpO2 {vitals['spo2']}% RR {vitals['rr']} "
          f"BP {vitals['sbp']}/{vitals['dbp']} T {vitals['temp_c']}C age {vitals['age_years']} +2 chronic")
    print(f"\n deterioration risk  {ex['risk_percent']}%   tier={ex['tier'].upper()}   NEWS2 reference={ex['news2']}")
    print(f"\n why (exact TreeSHAP, log-odds, base {ex['shap_base_value']}):")
    for c in ex["top_factors"]:
        bar = "█" * max(1, int(abs(c["shap"]) * 12))
        col = "↑" if c["shap"] > 0 else "↓"
        print(f"   {col} {c['text'][:52]:54} {c['shap']:+.3f} {bar}")
    print(f"\n red flags: {', '.join(f['text'] for f in ex['red_flags']) or 'none'}")
    print(f" advice: {ex['advice']['action']} — {ex['advice']['detail']}")

    # ----------------------------------------------------------- step 10
    rule("step 9-10 · persist to SQLite and raise the alert")
    db.init()
    pat = db.add_patient("MRN-DEMO1", "Demo Patient", 73, "M", "Ward 4B", "ward", 2)
    rec = db.add_reading(pat["id"], vitals, ex, signal_quality=res.quality, source="demo-video")
    print(f" patient id={pat['id']} reading id={rec['id']} tier={rec['tier']}")
    st = db.stats()
    print(f" db now: {st['patients']} patients, {st['readings']} readings, {st['open_alerts']} open alerts")
    for a in db.alerts(open_only=True):
        print(f"   ALERT [{a['tier']}] {a['name']} ({a['mrn']}): {a['message']}")

    rule("accuracy across a heart-rate sweep (synthetic clips, same code path)")
    errs, worst = [], 0.0
    for i, hr in enumerate([48, 58, 66, 74, 82, 90, 98, 108, 122, 138, 156, 178]):
        ts, vs = synthetic_recording(hr_bpm=hr, seconds=12.0, fps=20.0,
                                     snr_db=float(np.clip(16 - i * 0.6, 7, 16)),
                                     motion_bpm=float(18 * (i % 3)), seed=500 + i)
        r = analyze(clean_signal(ts, vs))
        e = abs((r.hr_bpm or 0) - hr)
        errs.append(e); worst = max(worst, e)
        flag = "" if r.quality >= 60 else "  (gated out by quality check)"
        print(f"   true {hr:3d} → {r.hr_bpm if r.hr_bpm else '—':>6} BPM  err {e:5.2f}  q {r.quality:5.1f}{flag}")
    print(f"\n MAE {np.mean(errs):.2f} BPM · worst {worst:.2f} BPM over {len(errs)} clips")
    print(f"\ndone in {time.time() - t_start:.1f} s · dashboard: http://localhost:8000/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
