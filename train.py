#!/usr/bin/env python3
"""Train the VitalGuard deterioration-risk model (step 7) and dump metrics +
global SHAP importances.

    python train.py                      # 6000 synthetic patients
    python train.py --n 20000 --seed 7   # bigger cohort
    python train.py --dataset ward.csv --target deteriorated_24h

The --dataset path is what you swap in for MIMIC-IV / eICU / a hospital CSV; the
column names expected are hr, spo2, rr, sbp, dbp, temp_c, age_years, sex,
chronic_score, unit, conscience.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitalguard.risk_model import FEATURES, engineer, train, MODEL_DIR  # noqa: E402


def train_from_csv(path: str, target: str) -> dict:
    """Same pipeline, real labels. Kept tiny on purpose so the swap is one command."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"column '{target}' not found; have: {list(df.columns)[:20]}")
    rows = [engineer(r) for r in df.to_dict("records")]
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = df[target].astype(int).to_numpy()
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0,
    )
    proba = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    metrics = {"n": len(y), "prevalence": float(y.mean()),
               "cv_roc_auc": round(float(roc_auc_score(y, proba)), 4),
               "cv_average_precision": round(float(average_precision_score(y, proba)), 4),
               "source": path}
    m.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"model": m, "features": FEATURES}, MODEL_DIR / "risk_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"metrics": metrics, "dataset": {"source": path}}, indent=2))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--target", default="deteriorated_24h")
    a = ap.parse_args()

    if a.dataset:
        print(json.dumps(train_from_csv(a.dataset, a.target), indent=2))
        return 0

    print(f"training on {a.n} synthetic patients (seed {a.seed}) ...")
    t = train(n=a.n, seed=a.seed)
    print("\n--- validation (5-fold stratified CV) ---")
    for k, v in t.metrics.items():
        print(f"  {k:28} {v}")
    print("\n--- global feature importance (mean |SHAP|, log-odds) ---")
    for k, v in list(t.global_shap.get("mean_abs", {}).items())[:12]:
        d = t.global_shap.get("direction", {}).get(k, 0)
        arrow = "↑ risk" if d > 0 else "↓ risk"
        print(f"  {k:16} {v:7.4f}  {arrow}")
    print(f"\nwrote {MODEL_DIR/'risk_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
