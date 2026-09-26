"""End-to-end check through whatever base URL is given (defaults to the public tunnel).

Sends well-formed capture payloads - `synthetic_recording` already returns values in
luma units, so v and luma are the same number; do not add an offset or every frame
saturates and the gate correctly refuses.
"""

import json
import os
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitalguard.ppg import synthetic_recording

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

CASES = [
    # hr, spo2, rr, sbp, dbp, temp, age, chronic, label
    (89.0, 94.0, 19.0, 112.0, 70.0, 37.2, 54, 1, "stable patient"),
    (118.0, 90.0, 27.0, 96.0, 58.0, 38.6, 78, 3, "deteriorating"),
]


def main() -> int:
    bad = 0
    for hr, spo2, rr, sbp, dbp, temp, age, chronic, label in CASES:
        t, v = synthetic_recording(hr_bpm=hr, seconds=12.0, fps=20.0, snr_db=16.0, seed=int(hr * 13))
        frames = [{"t": float(a), "v": float(c), "luma": float(c)} for a, c in zip(t, v)]
        # same shape the browser sends: vitals nested, not top level
        body = {"frames": frames, "save": False, "patient_id": None,
                "vitals": {"spo2": spo2, "rr": rr, "sbp": sbp, "dbp": dbp, "temp_c": temp,
                           "age_years": age, "sex": "M", "chronic_score": chronic,
                           "in_icu": False}}
        req = urllib.request.Request(f"{BASE}/api/ppg/process", data=json.dumps(body).encode(),
                                      headers={"content-type": "application/json"})
        j = json.load(urllib.request.urlopen(req, timeout=120))
        h = j["heart_rate"]
        r = j.get("risk") or {}
        err = abs((h.get("hr_bpm") or 0) - hr)
        clipped = j["signal"]["exposure"]["frac_clipped"]
        print(f"  {label:18} true {hr:5.1f} -> HR {h['hr_bpm']:6.1f} (err {err:4.2f})  q {h['quality']:5.1f}"
              f"  fold {h['flags']['fold_gain']:4.2f}  clipped {clipped}")
        factors = r.get("top_factors") or []
        print(f"  {'':18} risk {r.get('risk_percent')}% {str(r.get('tier')).upper()}"
              f"  {len(factors)} SHAP factors  | {str(r.get('advice'))[:64]}")
        if factors:
            tops = ", ".join(f"{f['feature']} {f['shap']:+.2f}" for f in factors[:4])
            print(f"  {'':18} top contributions: {tops}")
        if err > 2 or not r:
            bad += 1
    print(f"\n  {'END-TO-END OK' if bad == 0 else str(bad) + ' PROBLEM(S)'} via {BASE}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
