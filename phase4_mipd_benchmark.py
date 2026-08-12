#!/usr/bin/env python3
"""Benchmark the machine-learning layer against a classical model-informed
precision-dosing (MIPD) baseline (Reviewer #7.6).

A standard MAP-Bayesian TDM workflow estimates each patient's individual PK from
population priors + TDM samples and then applies a simple exposure-toxicity
relationship. With dense TDM sampling the MAP exposure estimates converge to the
observed exposures; we therefore proxy the Bayesian-TDM baseline by a logistic
regression on the observed/estimated exposure indices (C_min, AUC24 [, baseline
SCr]) - the classical exposure-response a clinician would use - and compare its
discrimination against gradient-boosted ML on the fuller feature set, on the
SAME 70/30 stratified split (seed 42) so the comparison is apples-to-apples.

Outputs results/phase4_ml_enhanced/mipd_benchmark.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "phase4_ml_enhanced" / "mipd_benchmark.json"
RNG = 42


def main():
    ml = pd.read_csv(HERE / "data" / "processed" / "ml_dataset.csv")
    pk = pd.read_csv(HERE / "results" / "phase3_pkpd" / "pkpd_indices.csv")
    df = ml.merge(pk, on="patient_id", how="inner", suffixes=("", "_pk"))

    # Target
    ycol = "nephrotoxicity" if "nephrotoxicity" in df.columns else "aki_stage"
    y = (df[ycol].astype(float) > 0).astype(int).values

    # Bayesian-TDM exposure-response predictors (classical)
    exp_feats = [c for c in ["Cmin", "AUC24"] if c in df.columns]
    exp_scr = exp_feats + [c for c in ["baseline_scr"] if c in df.columns]

    # Full ML predictor pool: numeric exposure + individual PK + baseline covariates.
    # Exclude the target and all OUTCOME-DERIVED columns (which would leak): AKI
    # stage / peak SCr are deterministic functions of the nephrotoxicity label,
    # and mortality / LOS / microbiological eradication are downstream outcomes.
    leak = {"patient_id", ycol, "clinical_cure", "aki_stage", "peak_scr",
            "ototoxicity", "neurotoxicity", "icu_mortality", "day_28_mortality",
            "icu_los", "hospital_los", "microbiological_eradication",
            "time_to_clinical_improvement", "day_28_mortality_pk"}
    num = df.select_dtypes(include=[np.number]).columns
    ml_feats = [c for c in num if c not in leak
                and not any(k in c.lower() for k in ["mortality", "_los", "aki_",
                    "peak_scr", "nephrotox", "cure", "ototox", "neurotox"])]

    Xexp = df[exp_feats].fillna(df[exp_feats].median()).values
    Xexp_scr = df[exp_scr].fillna(df[exp_scr].median()).values
    Xml = df[ml_feats].fillna(df[ml_feats].median()).values

    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.30, random_state=RNG, stratify=y)

    def lr_auc(X):
        sc = StandardScaler().fit(X[itr])
        m = LogisticRegression(max_iter=1000).fit(sc.transform(X[itr]), y[itr])
        return roc_auc_score(y[ite], m.predict_proba(sc.transform(X[ite]))[:, 1])

    # Model A: MAP-Bayesian TDM proxy (exposure-response, logistic)
    auc_bayes = lr_auc(Xexp)
    auc_bayes_scr = lr_auc(Xexp_scr)

    # Model B: gradient-boosted ML on the full feature pool
    pos = y[itr].sum()
    spw = (len(y[itr]) - pos) / max(pos, 1)
    xgbm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, random_state=RNG,
                             eval_metric="logloss", tree_method="hist")
    xgbm.fit(Xml[itr], y[itr])
    auc_ml = roc_auc_score(y[ite], xgbm.predict_proba(Xml[ite])[:, 1])

    res = {
        "n_train": int(len(itr)), "n_test": int(len(ite)),
        "map_bayesian_tdm_exposure_response": {
            "features": exp_feats, "test_auroc": round(float(auc_bayes), 3)},
        "map_bayesian_tdm_plus_scr": {
            "features": exp_scr, "test_auroc": round(float(auc_bayes_scr), 3)},
        "ml_gradient_boosted_full": {
            "n_features": len(ml_feats), "test_auroc": round(float(auc_ml), 3)},
        "delta_ml_minus_bayesian": round(float(auc_ml - auc_bayes), 3),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n[OK] Wrote {OUT}")


if __name__ == "__main__":
    main()
