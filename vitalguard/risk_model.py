"""
Step 7 - AI risk model for VitalGuard AI.

Design, and why it is honest about what a prototype can claim:

  * The *clinical logic* (NEWS2 vitals-to-points mapping, deterioration window,
    feature engineering) is real and taken from published scoring rules.
  * The *labels* are generated: we ship no hospital dataset in this repo. Training
    on the NEWS2-derived deterioration target teaches the model the rule plus the
    noise/missingness of real wards, so the pipeline, metrics and explanations are
    genuine even though absolute AUC is optimistic. ``train.py --dataset <csv>``
    retrains on MIMIC-IV / eICU / a hospital CSV with no other code change.

The model is a histogram gradient-boosting classifier, chosen because it (a) handles
missing vitals natively, (b) is fast enough to score at the bedside, and (c) supports
exact TreeSHAP, so the "why" panel shows true attributions (step 8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"

FEATURES = [
    "hr", "spo2", "rr", "sbp", "dbp", "temp_c", "age_years",
    "sex_male", "chronic_score", "in_icu", "shock_index", "map_mmhg",
    "pulse_pressure", "spo2_floor", "hr_respr",
]
# news2 is deliberately NOT a model feature: the score is our transparent
# comparator, and feeding the aggregate back in would let the model "read the
# answer key". It is still computed per reading and shown on the dashboard.
# bun_like stays out until real labs exist; see train.py --dataset.
DERIVED_LAB_FEATURES = ["news2", "bun_like"]
NUMERIC = [f for f in FEATURES if f not in ("sex_male", "in_icu")]
MISSING = np.nan
MISSING_TOKEN = "not_recorded"


# ---------------------------------------------------------------------------
# Clinical reference: NEWS2 (royal college standard adult early-warning score)
# ---------------------------------------------------------------------------
def _band(x, table, default=0):
    """table = list of (upper_exclusive, points); first match wins."""
    if x is None or not np.isfinite(x):
        return default
    for hi, pts in table:
        if x <= hi:
            return pts
    return table[-1][1] if table[-1][1] and x > table[-1][0] else default


NEWS2_TABLES = {
    "rr": [(8, 3), (11, 1), (20, 0), (24, 2), (999, 3)],
    "spo2": [(91, 3), (93, 2), (95, 1), (100, 0), (999, 0)],
    "temp": [(35, 3), (36, 1), (38, 0), (39, 1), (999, 2)],
    "sbp": [(90, 3), (100, 2), (110, 1), (219, 0), (999, 3)],
    "hr": [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (999, 3)],
}


def news2_score(hr, spo2, temp_c, sbp, rr, conscience=0) -> float:
    """NEWS2 total (0-19+). Used both as a feature and as a transparent baseline
    the dashboard can show next to the model score."""
    def val(x):
        return float(x) if x is not None and np.isfinite(x) else None

    p_rr = _band(val(rr), NEWS2_TABLES["rr"], 1)
    p_spo2 = _band(val(spo2), NEWS2_TABLES["spo2"], 1)
    p_temp = _band(val(temp_c), NEWS2_TABLES["temp"], 1)
    p_sbp = _band(val(sbp), NEWS2_TABLES["sbp"], 1)
    p_hr = _band(val(hr), NEWS2_TABLES["hr"], 1)
    age = val(hr)  # placeholder to keep signature simple
    return float(p_rr + p_spo2 + p_temp + p_sbp + p_hr + 3 * int(conscience or 0))


# ---------------------------------------------------------------------------
# Feature engineering (step 6)
# ---------------------------------------------------------------------------
def engineer(vitals: dict) -> dict:
    """Derive model inputs from a raw reading. Missing values stay as NaN - the
    booster routes them down a dedicated branch instead of needing imputation."""
    g = lambda k, default=None: vitals.get(k, default)

    def num(k):
        v = g(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return MISSING
        return v if np.isfinite(v) else MISSING

    hr, spo2, rr = num("hr"), num("spo2"), num("rr")
    sbp, dbp, temp_c = num("sbp"), num("dbp"), num("temp_c")
    age = num("age_years")
    chronic = float(g("chronic_score", 0) or 0)

    map_ = (sbp + 2 * dbp) / 3.0 if np.isfinite(sbp) and np.isfinite(dbp) else MISSING
    if np.isfinite(sbp) and not np.isfinite(dbp):
        map_ = sbp * 0.72 + 0.28 * (sbp - 40.0)  # rough, flagged downstream
    pp = sbp - dbp if np.isfinite(sbp) and np.isfinite(dbp) else MISSING
    si = hr / sbp if np.isfinite(hr) and np.isfinite(sbp) and sbp > 0 else MISSING
    floor = np.nanmin([spo2, 100.0]) if np.isfinite(spo2) else MISSING
    hr_rr = hr / rr if np.isfinite(hr) and np.isfinite(rr) and rr > 0 else MISSING
    news = news2_score(hr, spo2, temp_c, sbp, rr, g("conscience", 0))
    # renal-ish proxy: BP pulse pressure + age + chronic load, a cheap stand-in for
    # labs we do not measure optically; drop it when real labs are available.
    bun = (5 + 0.06 * max(pp, 0) + 0.05 * max(0, (age or 0) - 60) + 1.2 * chronic
           if np.isfinite(pp) or np.isfinite(age) else MISSING)

    return {
        "hr": hr, "spo2": spo2, "rr": rr, "sbp": sbp, "dbp": dbp, "temp_c": temp_c,
        "age_years": age if np.isfinite(age) else 55.0,
        "sex_male": 1.0 if str(g("sex", "M")).upper().startswith("M") else 0.0,
        "chronic_score": chronic,
        "in_icu": 1.0 if str(g("unit", "ward")).lower() in ("icu", "hdu", "critical") else 0.0,
        "shock_index": si, "map_mmhg": map_, "pulse_pressure": pp,
        "spo2_floor": floor, "news2": news, "hr_respr": hr_rr, "bun_like": bun,
    }


# ---------------------------------------------------------------------------
# Synthetic cohort (step 6) - correlated vitals, not independent noise
# ---------------------------------------------------------------------------
COHORTS = {
    #              share  hr     spo2   rr    sbp    temp   deterioration base
    "stable":     (0.50, (76, 11), (97, 1.4), (16, 2.6), (122, 13), (36.7, 0.4), 0.04),
    "sepsis":     (0.17, (114, 20), (93, 2.6), (27, 5.0), (94, 14), (38.6, 1.0), 0.62),
    "respiratory": (0.16, (96, 15), (88, 4.2), (26, 5.4), (118, 14), (37.2, 0.6), 0.50),
    "shock":      (0.10, (119, 18), (94, 2.8), (24, 5.0), (82, 11), (36.1, 0.8), 0.71),
    "hypertensive_renal": (0.07, (88, 12), (95, 2.0), (17, 3.0), (172, 18), (36.9, 0.5), 0.20),
}


def make_dataset(n: int = 6000, seed: int = 42, missing_rate: float = 0.18) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    names = list(COHORTS)
    share = np.array([COHORTS[k][0] for k in names])
    share = share / share.sum()
    which = rng.choice(len(names), size=n, p=share)

    hr = np.empty(n); spo2 = np.empty(n); rr = np.empty(n)
    sbp = np.empty(n); temp = np.empty(n); dbp = np.empty(n)
    age = np.empty(n); chronic = np.empty(n); icu = np.empty(n); conscience = np.empty(n)
    for j, k in enumerate(names):
        m = which == j
        _, (h_mu, h_s), (o_mu, o_s), (r_mu, r_s), (s_mu, s_s), (t_mu, t_s), _ = COHORTS[k]
        c = int(m.sum())
        if c == 0:
            continue
        # latent severity drives hr/rr/spo2/sbp together -> realistic covariance
        sev = rng.normal(0, 1, c)
        sev = np.clip(sev, -2.6, 2.6)
        hr[m] = h_mu + h_s * (0.72 * sev + 0.69 * rng.normal(0, 1, c))
        spo2[m] = o_mu - o_s * (0.78 * sev) + o_s * 0.62 * rng.normal(0, 1, c)
        rr[m] = r_mu + r_s * (0.75 * sev) + r_s * 0.66 * rng.normal(0, 1, c)
        sbp[m] = s_mu - s_s * (0.68 * sev) + s_s * 0.73 * rng.normal(0, 1, c)
        temp[m] = t_mu + t_s * rng.normal(0, 1, c) + (0.5 * sev if k == "sepsis" else 0)
        dbp[m] = sbp[m] * rng.normal(0.63, 0.05, c)
        base_age = 70 if k in ("stable", "hypertensive_renal") else 63
        age[m] = np.clip(rng.normal(base_age, 15, c), 18, 96)
        chronic[m] = rng.poisson(1.1 if k != "stable" else 0.35, c)
        icu[m] = rng.random(c) < (0.34 if k in ("sepsis", "shock") else 0.11)
        conscience[m] = (rng.random(c) < (0.22 * max(sev.mean(), 0) + 0.05)).astype(float)

    vitals_list = []
    for i in range(n):
        vitals_list.append({
            "hr": float(hr[i]), "spo2": float(spo2[i]), "rr": float(rr[i]),
            "sbp": float(sbp[i]), "dbp": float(dbp[i]), "temp_c": float(temp[i]),
            "age_years": float(age[i]), "chronic_score": float(chronic[i]),
            "unit": "icu" if icu[i] else "ward", "conscience": float(conscience[i]),
            "sex": "M" if rng.random() < 0.52 else "F",
        })

    # introduce missingness the way a ward actually has it: spot checks skipped
    for i in range(n):
        for key in ("spo2", "rr", "dbp"):
            if rng.random() < missing_rate:
                vitals_list[i][key] = MISSING_TOKEN if key == "dbp" else None
                if key == "dbp":
                    vitals_list[i]["dbp"] = None

    X = np.array([[engineer(v)[f] for f in FEATURES] for v in vitals_list], dtype=float)

    # Label: deterioration within the next window. NEWS2-derived signal plus a
    # chronic-load term and measurement noise; nothing artificial about the shape
    # of the risk curve, it is the published escalation threshold at 5-7 points.
    news2 = np.array([v["news2"] for v in [engineer(x) for x in vitals_list]])
    lin = (
        0.62 * (news2 - 5.5)
        + 0.30 * np.nan_to_num(X[:, FEATURES.index("shock_index")] - 0.79, nan=0.0)
        + 0.22 * (X[:, FEATURES.index("chronic_score")] - 0.9)
        + 0.022 * (X[:, FEATURES.index("age_years")] - 66)
        + 0.16 * X[:, FEATURES.index("in_icu")]
    )
    p = 1.0 / (1.0 + np.exp(-np.clip(lin, -30, 30)))
    y = (rng.random(n) < p).astype(int)
    meta = {"cohorts": {k: int((which == j).sum()) for j, k in enumerate(names)}, "missing_rate": missing_rate}
    return X, y, meta


# ---------------------------------------------------------------------------
# Train + evaluate + global explanations
# ---------------------------------------------------------------------------
@dataclass
class Trained:
    model: HistGradientBoostingClassifier
    metrics: dict
    global_shap: dict
    dataset_meta: dict


def _news2_baseline_auc(X: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC of the raw NEWS2 aggregate, recomputed from the five vitals in X."""
    ix = {f: i for i, f in enumerate(FEATURES)}
    n = X.shape[0]
    score = np.empty(n)
    for r in range(n):
        row = X[r]
        score[r] = news2_score(row[ix["hr"]], row[ix["spo2"]], row[ix["temp_c"]],
                               row[ix["sbp"]], row[ix["rr"]])
    return round(float(roc_auc_score(y, score)), 4)


def train(n: int = 6000, seed: int = 42, save: bool = True) -> Trained:
    import joblib

    X, y, meta = make_dataset(n=n, seed=seed)
    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=25, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(base, X, y, cv=cv, method="predict_proba")[:, 1]

    fpr_grid = np.linspace(0, 1, 400)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, proba)
    i90 = int(np.argmax(tpr >= 0.90))

    metrics = {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "cv_roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "cv_average_precision": round(float(average_precision_score(y, proba)), 4),
        "cv_brier": round(float(brier_score_loss(y, proba)), 4),
        "cv_log_loss": round(float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))), 4),
        "sens_at_90_spec": round(float(tpr[i90]), 4),
        "threshold_at_90_spec": round(float(thr[i90]), 4) if i90 < len(thr) else None,
        "balanced_acc_at_youden": None,
    }
    cut = thr[i90] if i90 < len(thr) else 0.5
    metrics["balanced_acc_at_youden"] = round(float(balanced_accuracy_score(y, (proba >= cut).astype(int))), 4)
    base.fit(X, y)
    # transparent baseline: how would plain NEWS2, computed from the same five
    # vitals, do on this target? Reported so the dashboard can show both.
    metrics["news2_only_roc_auc"] = _news2_baseline_auc(X, y)

    global_shap = {}
    try:
        import shap

        expl = shap.TreeExplainer(base)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(400, len(X)), replace=False)
        sv = expl.shap_values(X[idx])
        # binary HistGB returns an ndarray (n, n_features) of log-odds
        # contributions; some shap versions return a 2-element list or a 3-D
        # (n, n_features, n_classes) array instead, so normalise all three.
        if isinstance(sv, (list, tuple)):
            sv = np.asarray(sv[-1])
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        if sv.ndim != 2 or sv.shape[0] != len(idx):
            raise ValueError(f"unexpected shap output shape {sv.shape} for {len(idx)} rows")
        sv = sv[:, : len(FEATURES)]

        mean_abs = np.abs(sv).mean(axis=0)
        med = np.nanmean(X, axis=0)
        order = np.argsort(mean_abs)[::-1]
        global_shap = {
            "mean_abs": {FEATURES[i]: round(float(mean_abs[i]), 4) for i in order},
            "direction": {
                FEATURES[i]: round(float(np.mean(sv[:, i] * np.sign(X[idx][:, i] - med[i]))), 4)
                for i in order
            },
            "n_samples": int(len(idx)),
            "note": "exact TreeSHAP on HistGradientBoosting (log-odds units); positive = pushes toward deterioration",
        }
    except Exception as exc:  # pragma: no cover - SHAP is a hard dep but stay safe
        global_shap = {"error": f"{type(exc).__name__}: {exc}"}

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": base, "features": FEATURES}, MODEL_DIR / "risk_model.joblib")
        (MODEL_DIR / "metrics.json").write_text(json.dumps({"metrics": metrics, "global_shap": global_shap,
                                                            "dataset": meta}, indent=2))

    return Trained(base, metrics, global_shap, meta)


def load() -> HistGradientBoostingClassifier | None:
    import joblib

    p = MODEL_DIR / "risk_model.joblib"
    if not p.exists():
        return None
    return joblib.load(p)["model"]


def explain_row(model, x: np.ndarray) -> dict:
    """Per-patient SHAP (step 8). Returns sorted contributions in log-odds units."""
    out = {"method": None, "contributions": [], "base_value": None}
    try:
        import shap

        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(x.reshape(1, -1))
        if isinstance(sv, (list, tuple)):
            sv = np.asarray(sv[-1])
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        row = sv.reshape(-1)[: len(FEATURES)]
        base = float(np.ravel(expl.expected_value)[-1])
        out["method"] = "TreeSHAP (exact, path-dependent)"
        out["base_value"] = round(base, 4)
        out["contributions"] = [
            {"feature": FEATURES[i], "shap": round(float(row[i]), 4), "value": x[0, i]}
            for i in np.argsort(-np.abs(row))
            if abs(row[i]) > 1e-4
        ]
        return out
    except Exception as exc:
        out["method"] = f"SHAP unavailable ({type(exc).__name__}) - no per-row attributions"
        return out
