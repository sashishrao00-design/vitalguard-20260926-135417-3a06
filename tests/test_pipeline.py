"""VitalGuard AI regression tests.

    pip install pytest && pytest -q

These are the checks that decide whether the prototype can be trusted in front of
judges: the heart-rate estimator must be accurate, the quality gate must refuse bad
signals, the risk model must rank correctly, and SHAP must sum to the model output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vitalguard import db  # noqa: E402
from vitalguard.explain import explain, tier_for  # noqa: E402
from vitalguard.heart_rate import analyze  # noqa: E402
from vitalguard.ppg import clean_signal, synthetic_recording  # noqa: E402
from vitalguard.risk_model import FEATURES, explain_row, engineer, load  # noqa: E402

HRS = [48, 58, 66, 74, 82, 90, 98, 108, 122, 138, 156]


def frames_payload(hr, seconds=12.0, fps=20.0, snr=15.0, motion=0.0, seed=1):
    t, v = synthetic_recording(hr_bpm=hr, seconds=seconds, fps=fps, snr_db=snr,
                               motion_bpm=motion, seed=seed)
    return t, v


# ------------------------------------------------------------- steps 2-5 ---
@pytest.mark.parametrize("hr", HRS)
def test_heart_rate_within_two_bpm(hr):
    t, v = frames_payload(hr, seed=hr)
    res = analyze(clean_signal(t, v))
    assert res.hr_bpm is not None
    assert abs(res.hr_bpm - hr) <= 2.0, f"true {hr} -> {res.hr_bpm}"


def test_quality_gate_refutes_pulseless_signal():
    """A finger lifted off the lens must NOT produce a scored reading."""
    rng = np.random.default_rng(0)
    t = np.arange(240) / 20.0
    v = 140 + rng.normal(0, 0.6, 240)
    res = analyze(clean_signal(t, v))
    assert res.quality < 60
    assert res.quality_label == "poor"
    assert res.guidance, "user must get retry instructions"


def test_quality_survives_motion_and_low_snr():
    t, v = frames_payload(74, fps=15.0, snr=8.0, motion=45.0, seed=4)
    res = analyze(clean_signal(t, v))
    assert abs(res.hr_bpm - 74) <= 8.0
    assert res.quality >= 45


def test_short_clip_is_rejected_not_silenced():
    t, v = frames_payload(80, seconds=0.5, seed=2)
    with pytest.raises(ValueError):
        clean_signal(t, v)


def test_three_estimators_agree_on_clean_signal():
    t, v = frames_payload(90, seed=6)
    res = analyze(clean_signal(t, v))
    assert res.agree_bpm <= 6.0
    assert res.n_peaks >= 8


def test_signal_is_finite_and_normalised():
    t, v = frames_payload(88, seed=7)
    ppg = clean_signal(t, v)
    assert np.all(np.isfinite(ppg.y))
    assert abs(float(np.std(ppg.y)) - 1.0) < 1e-6      # unit-variance AC
    assert ppg.duration > 11.0


# ----------------------------------------------------------- steps 6-8 ----
@pytest.fixture(scope="module")
def model():
    m = load()
    if m is None:
        pytest.skip("model not trained - run: python train.py")
    return m


def score(m, vitals):
    x = np.array([[engineer(vitals)[f] for f in FEATURES]], dtype=float)
    return float(m.predict_proba(x)[0, 1]), x


def test_healthy_scores_lower_than_deteriorating(model):
    p_ok, _ = score(model, {"hr": 76, "spo2": 98, "rr": 15, "sbp": 126, "dbp": 78,
                            "temp_c": 36.7, "age_years": 62, "chronic_score": 0})
    p_bad, _ = score(model, {"hr": 128, "spo2": 88, "rr": 30, "sbp": 86, "dbp": 48,
                             "temp_c": 39.1, "age_years": 78, "chronic_score": 3})
    assert p_bad > 0.6 and p_ok < 0.25
    assert tier_for(p_bad)[0] in ("critical", "high")
    assert tier_for(p_ok)[0] in ("minimal", "low")


def test_single_abnormal_vital_moves_the_score(model):
    base = {"hr": 78, "spo2": 97, "rr": 16, "sbp": 122, "dbp": 74, "temp_c": 36.8,
            "age_years": 65, "chronic_score": 1}
    p0, _ = score(model, base)
    for key, worse in [("spo2", 86), ("rr", 34), ("sbp", 80), ("hr", 140), ("temp_c", 40.2)]:
        p1, _ = score(model, {**base, key: worse})
        assert p1 > p0, f"{key}={worse} should raise risk ({p0:.3f} -> {p1:.3f})"


def test_missing_vitals_do_not_crash(model):
    p, x = score(model, {"hr": 91})
    assert np.isfinite(p)
    assert np.isnan(x).any()


def test_shap_sums_to_model_output(model):
    """The exact-attribution property. If this fails, the 'why' panel is a lie."""
    vitals = {"hr": 118, "spo2": 91, "rr": 26, "sbp": 95, "dbp": 57, "temp_c": 38.6,
              "age_years": 74, "chronic_score": 2}
    p, x = score(model, vitals)
    payload = explain_row(model, x)
    assert payload["contributions"], "SHAP returned nothing"
    logit = np.log(p / (1 - p))
    total = payload["base_value"] + sum(c["shap"] for c in payload["contributions"])
    assert abs(total - logit) < 0.05, f"base+Σφ={total:.4f} vs model logit={logit:.4f}"


def test_global_importances_exist(model):
    from vitalguard.risk_model import MODEL_DIR
    import json
    d = json.loads((MODEL_DIR / "metrics.json").read_text())
    imp = d["global_shap"]["mean_abs"]
    assert len(imp) == len(FEATURES)
    assert set(imp) == set(FEATURES)
    assert d["metrics"]["cv_roc_auc"] > 0.8


# -------------------------------------------------------- steps 9-10 ------
def test_persistence_and_alert_dedupe(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init(tmp_path / "t.db")
    pat = db.add_patient("MRN-1", "Test", 70, "M", "4B", "ward", 1)
    vit = {"hr": 120, "spo2": 90, "rr": 27, "sbp": 92, "dbp": 55, "temp_c": 38.8}
    ex = explain(vit, 0.93, {"method": "test", "base_value": 0.0,
                             "contributions": [{"feature": "spo2", "shap": 0.4, "value": 90}]})
    db.add_reading(pat["id"], vit, ex, 88, "test")
    db.add_reading(pat["id"], vit, ex, 85, "test")           # same tier -> no new alert
    assert len(db.alerts()) == 1
    milder = explain({**vit, "hr": 78, "spo2": 98, "rr": 15, "sbp": 126, "dbp": 78,
                      "temp_c": 36.7}, 0.05, None)
    db.add_reading(pat["id"], {**vit, "hr": 78}, milder, 90, "test")
    assert len(db.history(pat["id"])) == 3
    assert len(db.alerts()) == 1                              # improvement adds none
    a = db.alerts()[0]
    assert db.ack_alert(a["id"], "nurse") is True
    assert db.ack_alert(a["id"], "nurse") is False             # already acked


def test_patient_upsert(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init(tmp_path / "t.db")
    a = db.add_patient("MRN-9", "Before", 60, "F", "2A", "ward", 0)
    b = db.add_patient("MRN-9", "After", 61, "F", "2A", "ward", 0)
    assert a["id"] == b["id"] and b["name"] == "After"
    assert len(db.list_patients()) == 1


def test_news2_reference_matches_published_bands():
    """If we advertise NEWS2 we must get the bands right."""
    from vitalguard.risk_model import news2_score
    assert news2_score(78, 97, 36.8, 120, 16) == 0            # all green
    assert news2_score(135, 90, 39.5, 85, 28) >= 12           # multiple reds
    assert news2_score(45, 92, 35.0, 95, 7) >= 10             # brady + hypo + fever-ish
    assert news2_score(None, None, None, None, None) == 5     # all-missing -> 1 pt each


def test_explanation_is_plain_language(model):
    vit = {"hr": 121, "spo2": 89, "rr": 28, "sbp": 90, "dbp": 52, "temp_c": 38.9,
            "age_years": 76, "chronic_score": 2}
    p, x = score(model, vit)
    ex = explain(vit, p, explain_row(model, x))
    assert ex["top_factors"]
    for c in ex["top_factors"]:
        assert "→" in c["text"]
        if c["value"] is None:
            assert "not recorded" in c["text"]
            assert "low " not in c["text"].split("→")[1]       # never call missing "low"
    assert ex["red_flags"] and "action" in ex["advice"]
