#!/usr/bin/env python3
"""Recompute SHAP feature importance for the post-dose nephrotoxicity and
clinical-cure models and dump the rankings (Reviewer #5: Table 5 must match the
actual 55-feature models). Reuses the Phase 5 DoseOptimizer feature-matrix
reconstruction (which replays the Phase 4 pipeline) so features/scaling match
the saved models, then runs TreeSHAP.

Outputs results/phase4_ml_enhanced/shap_importance.json.
"""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from phase5_dose_optimization import DoseOptimizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "phase4_ml_enhanced" / "shap_importance.json"


def rank(model, X, feature_names, topn=20):
    # Use XGBoost's native TreeSHAP (pred_contribs) to avoid shap-library
    # base_score parsing incompatibilities. The last column is the bias term.
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    sv = contribs[:, :-1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    total = mean_abs.sum()
    out = []
    for i in order[:topn]:
        out.append({
            "feature": feature_names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 4),
            "pct": round(float(mean_abs[i] / total * 100), 1),
        })
    return out


def main():
    opt = DoseOptimizer()
    feats = opt.POSTDOSE_FEATURES
    X = opt._build_feature_matrix(feats)
    Xs = opt.scalers["postdose"].transform(X)
    print(f"Reconstructed post-dose matrix: {Xs.shape}, {len(feats)} features")

    res = {}
    for model_name in ["nephrotoxicity_postdose", "clinical_cure"]:
        m = opt.models.get(model_name)
        if m is None:
            print(f"  [WARN] model {model_name} not loaded; skipping")
            continue
        n_in = getattr(m, "n_features_in_", len(feats))
        if n_in != Xs.shape[1]:
            print(f"  [WARN] {model_name} expects {n_in} features but matrix has {Xs.shape[1]}; skipping")
            continue
        res[model_name] = rank(m, Xs, feats, topn=20)
        print(f"\n=== {model_name} top 10 (SHAP) ===")
        for r in res[model_name][:10]:
            print(f"  {r['feature']:28s} {r['pct']:5.1f}%")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n[OK] Wrote {OUT}")


if __name__ == "__main__":
    main()
