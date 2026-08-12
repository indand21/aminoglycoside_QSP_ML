#!/usr/bin/env python3
"""Multi-seed replicate-stability analysis (Reviewer #7.8). Regenerates the
synthetic cohort under several random seeds and reports the stability of the
two primary, simulator-independent summaries: the pre-dose (26-admission-feature)
nephrotoxicity AUROC (the clinically deployable model) and the combined
efficacy-safety target attainment. Post-dose AUROCs depend on a per-seed
Bayesian refit and are out of scope here; the pre-dose model needs no PK
posteriors, so it is the appropriate stability target.

Outputs results/phase4_ml_enhanced/multiseed_stability.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import generate_synthetic_data_python as gen

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "phase4_ml_enhanced" / "multiseed_stability.json"
SEEDS = [12345, 111, 222, 333, 444]
N = 1500

ADMIT_NUM = ["age", "weight", "height", "bmi", "apache_ii", "sofa_score",
             "baseline_crcl", "baseline_scr", "baseline_egfr", "baseline_albumin",
             "baseline_bilirubin", "diabetes", "mechanical_ventilation",
             "vasopressor_use"]
ADMIT_CAT = ["sex", "sepsis_type", "infection_site", "ckd_stage", "drug"]


def predose_auroc(pat, outcomes):
    df = pat.merge(outcomes[["patient_id", "nephrotoxicity"]], on="patient_id")
    y = df["nephrotoxicity"].astype(int).values
    X = df[ADMIT_NUM].copy()
    for c in ADMIT_CAT:
        if c in df:
            X[c] = pd.Categorical(df[c]).codes
    # drug-specific MIC as an admission (pathogen) feature
    X["mic"] = np.where(df["drug"] == "amikacin", df["mic_amikacin"], df["mic_gentamicin"])
    X = X.fillna(X.median()).values
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.30,
                                random_state=42, stratify=y)
    pos = y[itr].sum()
    spw = (len(y[itr]) - pos) / max(pos, 1)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          random_state=42, eval_metric="logloss", tree_method="hist")
    m.fit(X[itr], y[itr])
    return float(roc_auc_score(y[ite], m.predict_proba(X[ite])[:, 1]))


def combined_attainment(pat, conc):
    g = conc.groupby("patient_id")["concentration"]
    ex = pd.DataFrame({"Cmax": g.max(), "Cmin": g.min()}).reset_index()
    df = pat.merge(ex, on="patient_id")
    mic = np.where(df["drug"] == "amikacin", df["mic_amikacin"], df["mic_gentamicin"])
    thr = np.where(df["drug"] == "amikacin", 2.5, 2.0)
    eff = (df["Cmax"].values / mic) >= 8
    safe = df["Cmin"].values < thr
    return float((eff & safe).mean() * 100)


def main():
    rows = []
    for s in SEEDS:
        np.random.seed(s)
        pat = gen.generate_patient_demographics(N)
        dose = gen.generate_dosing_records(pat)
        conc = gen.generate_concentrations(pat, dose)
        out = gen.generate_outcomes(pat, conc, dose)
        auc = predose_auroc(pat, out)
        att = combined_attainment(pat, conc)
        neph = out["nephrotoxicity"].astype(int).mean() * 100
        ami = (pat["drug"] == "amikacin").mean() * 100
        rows.append({"seed": s, "predose_auroc": round(auc, 3),
                     "combined_attainment_pct": round(att, 1),
                     "nephrotox_prev_pct": round(neph, 1),
                     "amikacin_pct": round(ami, 1)})
        print(f"seed {s}: pre-dose AUROC {auc:.3f} | combined attainment {att:.1f}% | "
              f"nephrotox {neph:.1f}% | amikacin {ami:.1f}%")

    def ms(k):
        v = np.array([r[k] for r in rows])
        return {"mean": round(float(v.mean()), 3), "sd": round(float(v.std(ddof=1)), 3),
                "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}

    summary = {"seeds": SEEDS, "per_seed": rows,
               "predose_auroc": ms("predose_auroc"),
               "combined_attainment_pct": ms("combined_attainment_pct"),
               "nephrotox_prev_pct": ms("nephrotox_prev_pct")}
    OUT.write_text(json.dumps(summary, indent=2))
    print("\nPre-dose AUROC: %.3f ± %.3f (range %.3f-%.3f)" % (
        summary["predose_auroc"]["mean"], summary["predose_auroc"]["sd"],
        summary["predose_auroc"]["min"], summary["predose_auroc"]["max"]))
    print("Combined attainment: %.1f ± %.1f%%" % (
        summary["combined_attainment_pct"]["mean"], summary["combined_attainment_pct"]["sd"]))
    print(f"[OK] Wrote {OUT}")


if __name__ == "__main__":
    main()
