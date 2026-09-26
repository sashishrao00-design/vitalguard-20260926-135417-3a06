"""
Step 8 - explainability, and the clinical text that wraps a probability.

The dashboard must answer two different questions, so this module does too:

  * "Why is the score high?"   -> SHAP contributions translated into plain sentences
  * "What do I do about it?"   -> tiered escalation advice + NEWS2 comparison
"""

from __future__ import annotations

import numpy as np

from .risk_model import FEATURES, engineer, news2_score

# Human phrasing per feature: (unit text, direction when shap > 0)
PLAIN = {
    "hr": ("heart rate", "elevated pulse", "bradycardic pulse"),
    "spo2": ("oxygen saturation", "low saturation", "high saturation"),
    "rr": ("respiratory rate", "fast breathing", "very slow breathing"),
    "sbp": ("systolic blood pressure", "low blood pressure", "high blood pressure"),
    "dbp": ("diastolic blood pressure", "low diastolic pressure", "high diastolic pressure"),
    "temp_c": ("temperature", "fever", "hypothermia"),
    "age_years": ("age", "older age band", "younger age band"),
    "sex_male": ("sex", "male sex association", "female sex association"),
    "chronic_score": ("chronic disease load", "more comorbidities", "fewer comorbidities"),
    "in_icu": ("care setting", "already in critical care", "general ward"),
    "shock_index": ("shock index (HR/SBP)", "high shock index", "low shock index"),
    "map_mmhg": ("mean arterial pressure", "low perfusion pressure", "high perfusion pressure"),
    "pulse_pressure": ("pulse pressure", "narrow pulse pressure", "wide pulse pressure"),
    "spo2_floor": ("lowest recorded saturation", "desaturation trend", "stable saturation"),
    "news2": ("NEWS2 aggregate", "high NEWS2", "low NEWS2"),
    "hr_respr": ("HR-to-RR ratio", "disproportionate tachycardia", "matched HR/RR"),
    "bun_like": ("renal proxy", "implied renal strain", "no renal strain"),
}

TIERS = [
    (0.70, "critical", "#ff4d5e"),
    (0.45, "high", "#ff9f2e"),
    (0.25, "moderate", "#ffd83d"),
    (0.10, "low", "#5ad1a5"),
]


def tier_for(p: float) -> tuple[str, str]:
    for cut, name, colour in TIERS:
        if p >= cut:
            return name, colour
    return "minimal", "#4aa3ff"


def _fmt(name: str, value: float) -> str:
    if name == "temp_c":
        return f"{value:.1f} °C"
    if name == "spo2" or name == "spo2_floor":
        return f"{value:.0f} %"
    if name in ("map_mmhg", "sbp", "dbp", "pulse_pressure"):
        return f"{value:.0f} mmHg"
    if name in ("age_years", "chronic_score", "in_icu", "sex_male"):
        return f"{value:.0f}"
    if name == "shock_index":
        return f"{value:.2f}"
    return f"{value:.0f}"


def advise(p: float, tier: str, news2: float | None) -> dict:
    """Escalation advice keyed to risk band. Deliberately conservative: a prototype
    must never read as an autonomous order set."""
    advice = {
        "critical": {
            "action": "Escalate now",
            "detail": "Alert the on-call team, continuous monitoring, recheck vitals every 15 min.",
            "recheck_min": 15,
        },
        "high": {
            "action": "Senior review within 30 min",
            "detail": "Ward doctor review, repeat PPG + vitals, consider fluid/obs bundle.",
            "recheck_min": 30,
        },
        "moderate": {
            "action": "Increase monitoring frequency",
            "detail": "Hourly obs, re-screen in 2 h, flag to nurse in charge.",
            "recheck_min": 60,
        },
        "low": {
            "action": "Routine observations",
            "detail": "Continue standard schedule; re-measure on the next round.",
            "recheck_min": 240,
        },
        "minimal": {
            "action": "No action",
            "detail": "Stable. Re-measure at the next scheduled round.",
            "recheck_min": 480,
        },
    }[tier]
    n = None if news2 is None else float(news2)
    if n is not None and n >= 7 and p < 0.45:
        advice["conflict"] = ("NEWS2 is already in the emergency range while the model score is lower - "
                              "trust the NEWS2 and escalate. Divergence usually means sparse inputs.")
    return advice


def explain(vitals: dict, p: float, shap_payload: dict | None = None) -> dict:
    """Full explanation object for one reading."""
    feats = engineer(vitals)
    x = np.array([[feats[f] for f in FEATURES]], dtype=float)
    tier, colour = tier_for(p)

    contribs = []
    if shap_payload and shap_payload.get("contributions"):
        for c in shap_payload["contributions"][:8]:
            name = c["feature"]
            val = c["value"]
            phrase = PLAIN.get(name, (name, "higher", "lower"))
            missing = not np.isfinite(val)
            direction = ("model fell back to its missing-value branch" if missing
                         else (phrase[1] if c["shap"] > 0 else phrase[2]))
            contribs.append({
                "feature": name,
                "label": phrase[0],
                "value": None if not np.isfinite(val) else round(float(val), 2),
                "value_text": "not recorded" if missing else _fmt(name, float(val)),
                "shap": round(float(c["shap"]), 4),
                "effect": "raises risk" if c["shap"] > 0 else "lowers risk",
                "text": (f"{phrase[0]} not recorded → {direction}" if missing
                         else f"{phrase[0]} = {_fmt(name, float(val))} → {direction}"),
            })

    # transparent reference: NEWS2 alone, computed without the model
    news = feats["news2"]

    # which single vital, on its own, is the loudest red flag
    red_flags = []
    checks = [
        ("spo2", lambda v: v < 92, "Saturation below 92%"),
        ("rr", lambda v: v > 24, "Respiratory rate above 24/min"),
        ("sbp", lambda v: v < 100, "Systolic BP below 100 mmHg"),
        ("hr", lambda v: v > 120, "Pulse above 120/min"),
        ("hr", lambda v: v < 45, "Pulse below 45/min"),
        ("temp_c", lambda v: v > 39.0, "Temperature above 39 °C"),
        ("temp_c", lambda v: v < 35.5, "Temperature below 35.5 °C"),
        ("shock_index", lambda v: v > 1.0, "Shock index above 1.0"),
        ("map_mmhg", lambda v: v < 65, "MAP below 65 mmHg"),
    ]
    for key, fn, text in checks:
        v = feats.get(key)
        if v is not None and np.isfinite(v) and fn(float(v)):
            red_flags.append({"field": key, "value": round(float(v), 2), "text": text})

    return {
        "risk_score": round(float(p), 4),
        "risk_percent": round(float(p) * 100, 1),
        "tier": tier,
        "colour": colour,
        "news2": None if not np.isfinite(news) else round(float(news), 0),
        "features": {k: (None if not np.isfinite(v) else round(float(v), 3)) for k, v in feats.items()},
        "top_factors": contribs,
        "red_flags": red_flags,
        "advice": advise(p, tier, news),
        "shap_method": (shap_payload or {}).get("method"),
        "shap_base_value": (shap_payload or {}).get("base_value"),
    }
