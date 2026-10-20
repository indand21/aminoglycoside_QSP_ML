#!/usr/bin/env python3
"""Revision 2: do the results depend on the author-chosen simulation values?

Supplementary Table S1 lists simulation assumptions that were set by the
authors rather than taken from a published source. Where a verified source
reports a different value, this script regenerates the cohort with the
published value ("source-aligned" parameters) and compares it, seed by seed,
with cohorts regenerated under the primary parameters:

    age              mean 58 / SD 15       -> 62.1 / 16.8   (SEPSIS INDIA)
    male sex         65%                   -> 61%           (SEPSIS INDIA)
    APACHE II        mean 22 / SD 8        -> 21.9 / 12.5   (SEPSIS INDIA)
    SOFA             mean 9 / SD 4         -> 6.7 / 3.6     (SEPSIS INDIA)
    septic shock     40%                   -> 43%           (SEPSIS INDIA; other
                                              sepsis categories rescaled)
    amikacin resistance, Pseudomonas 0.45 -> 0.303;
                         Acinetobacter 0.80 -> 0.70        (Kaur et al. 2024)

Everything else, including the outcome-generating model, is unchanged. The
gentamicin trough threshold is examined separately in the dose-optimisation
sensitivity analysis (r2_dose_optimization.py).

Each replicate (5 seeds x 2 parameter sets) is analysed with the production
pre-dose procedure, so the comparison does not depend on a PopPK refit:
26 admission features (19 covariates + 7 engineered), median imputation,
stratified 80/20 split (seed 42), standard scaling, Optuna TPE (100 trials,
inner 80/20 validation split), refit, held-out AUROC with bootstrap CI;
logistic regression on the same features; realised event rates and
combined target attainment (Phase 3 definitions: Cmax = highest and
Cmin = lowest observed concentration). The ten replicates run in parallel
worker processes, each single-threaded (CPU parallelism is faster than the
GPU for these 1,200-row models: 0.72 s vs 0.81 s per fit in a benchmark).

Output: results/revision2/sensitivity_cohorts.json
Usage:  python r2_sensitivity_cohorts.py [--workers 10]
"""

from __future__ import annotations

import os

# One thread per worker process: ten single-threaded workers on 12 cores, rather
# than ten processes each spawning 12 XGBoost threads
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import contextlib
import re
import inspect
import io
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / 'results' / 'revision2' / 'sensitivity_cohorts.json'
SEEDS = [12345, 111, 222, 333, 444]
N = 1500

SOURCE_ALIGNED = {
    'age_mean': 62.1, 'age_sd': 16.8, 'male_proportion': 0.61,
    'apache_ii_mean': 21.9, 'apache_ii_sd': 12.5, 'sofa_mean': 6.7, 'sofa_sd': 3.6,
}
SEPSIS_PROBS_SOURCE = [0.25 * 0.57 / 0.60, 0.35 * 0.57 / 0.60, 0.43]
AMI_RESISTANCE_SOURCE = {'pseudomonas': 0.303, 'acinetobacter': 0.70}

BASE = ['age', 'sex', 'weight', 'height', 'bmi', 'apache_ii', 'sofa_score', 'baseline_crcl',
        'baseline_scr', 'baseline_egfr', 'baseline_albumin', 'baseline_bilirubin', 'diabetes',
        'ckd_stage', 'mechanical_ventilation', 'vasopressor_use', 'sepsis_type', 'infection_site',
        'drug']


ALIGNED_SPEC = {
    'params': {**SOURCE_ALIGNED, 'ami_resistance': AMI_RESISTANCE_SOURCE},
    'patches': [('generate_patient_demographics', 'p=[0.25, 0.35, 0.40]', 'p=' + repr(SEPSIS_PROBS_SOURCE))],
}


def apply_spec(g, spec):
    """Modify the freshly imported simulator module: 'params' updates
    INDIAN_ICU_PARAMS (dict-valued entries are merged), 'patches' replace a
    literal in a simulator function's source (the function is recompiled)."""
    for k, v in spec.get('params', {}).items():
        if isinstance(v, dict) and isinstance(g.INDIAN_ICU_PARAMS.get(k), dict):
            g.INDIAN_ICU_PARAMS[k] = {**g.INDIAN_ICU_PARAMS[k], **v}
        else:
            g.INDIAN_ICU_PARAMS[k] = v
    for func, old, new in spec.get('patches', []):
        src = inspect.getsource(getattr(g, func))
        assert src.count(old) == 1, (func, old)
        exec(compile(src.replace(old, new), g.__file__, 'exec'), g.__dict__)


def simulate(seed, spec=None):
    """Generate one cohort in this (fresh) process; returns patient, dosing, concentration, outcome tables."""
    import generate_synthetic_data_python as g
    if spec:
        apply_spec(g, spec)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        pat = g.generate_patient_demographics(N)
        dose = g.generate_dosing_records(pat)
        conc = g.generate_concentrations(pat, dose)
        g.generate_time_varying(pat, 7)          # keeps the RNG sequence of the original pipeline
        out = g.generate_outcomes(pat, conc, dose)
    return pat, dose, conc, out


def predose_matrix(pat):
    import pandas as pd
    X = pat[BASE].copy()
    for c in ['sex', 'sepsis_type', 'infection_site', 'ckd_stage', 'drug']:
        X[c] = pd.Categorical(X[c]).codes
    X = X.astype(float)
    X = X.fillna(X.median())
    X['age_x_baseline_scr'] = X['age'] * X['baseline_scr']
    X['diabetes_x_baseline_crcl'] = X['diabetes'] * X['baseline_crcl']
    X['severity_composite'] = X['apache_ii'] + X['sofa_score']
    X['renal_score'] = X['baseline_crcl'] / (X['baseline_scr'] + 0.1)
    X['elderly'] = (X['age'] > 65).astype(float)
    X['impaired_renal'] = (X['baseline_crcl'] < 50).astype(float)
    X['obese'] = (X['bmi'] > 30).astype(float)
    return X.values


def run_one(job):
    seed, aligned = job
    return run_spec((seed, 'source-aligned' if aligned else 'primary', ALIGNED_SPEC if aligned else None))


def run_spec(job):
    """Simulate one cohort under a parameter specification and evaluate it."""
    seed, label, spec = job
    warnings.filterwarnings('ignore')
    t0 = time.time()
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from phase4_machine_learning_enhanced import EnhancedMLPipeline
    from r2_ml_analyses import boot_auc_ci

    pat, dose, conc, out = simulate(seed, spec)
    df = pat.merge(out[['patient_id', 'nephrotoxicity', 'clinical_cure', 'icu_mortality']], on='patient_id')
    y = df['nephrotoxicity'].astype(int).to_numpy()
    X = predose_matrix(df)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    a, b, ya, yb = train_test_split(Xtr, y[tr], test_size=0.2, random_state=42, stratify=y[tr])

    pipe = EnhancedMLPipeline.__new__(EnhancedMLPipeline)    # only results_dir is used
    pipe.results_dir = REPO / 'results' / 'revision2' / '_optuna_scratch' / 'sens_{}_{}'.format(seed, re.sub(r'[^A-Za-z0-9._-]+', '_', label))
    pipe.results_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model, params, _ = pipe.optimize_hyperparameters_optuna(a, ya, b, yb, scale_pos_weight=spw, n_trials=100)
    model.set_params(n_jobs=1)
    model.fit(Xtr, y[tr])
    p_xgb = model.predict_proba(Xte)[:, 1]
    p_lr = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

    # Combined target attainment with the Phase 3 definitions
    g = conc.groupby('patient_id')['concentration']
    ex = pd.DataFrame({'Cmax': g.max(), 'Cmin': g.min()}).reset_index().merge(pat, on='patient_id')
    ami = (ex['drug'] == 'amikacin').to_numpy()
    mic = np.where(ami, ex['mic_amikacin'], ex['mic_gentamicin'])
    eff = ex['Cmax'].to_numpy() / mic >= 8
    safe = ex['Cmin'].to_numpy() < np.where(ami, 2.5, 2.0)

    return {
        'seed': seed, 'parameters': label,
        'mean_age': float(pat['age'].mean()), 'mean_sofa': float(pat['sofa_score'].mean()),
        'mean_apache': float(pat['apache_ii'].mean()), 'mean_crcl': float(pat['baseline_crcl'].mean()),
        'pct_septic_shock': float(100 * (pat['sepsis_type'] == 'septic shock').mean()),
        'nephrotoxicity_pct': float(100 * y.mean()), 'cure_pct': float(100 * df['clinical_cure'].mean()),
        'icu_mortality_pct': float(100 * df['icu_mortality'].mean()),
        'efficacy_pct': float(100 * eff.mean()), 'safety_pct': float(100 * safe.mean()),
        'combined_attainment_pct': float(100 * (eff & safe).mean()),
        'predose_auroc_xgb': float(roc_auc_score(y[te], p_xgb)),
        'predose_auroc_xgb_ci': boot_auc_ci(y[te], p_xgb),
        'predose_auroc_lr': float(roc_auc_score(y[te], p_lr)),
        'seconds': time.time() - t0,
    }


def summarise(rows):
    keys = ['mean_age', 'mean_sofa', 'mean_apache', 'mean_crcl', 'pct_septic_shock', 'nephrotoxicity_pct',
            'cure_pct', 'icu_mortality_pct', 'efficacy_pct', 'safety_pct', 'combined_attainment_pct',
            'predose_auroc_xgb', 'predose_auroc_lr']
    out = {}
    for par in ['primary', 'source-aligned']:
        r = [x for x in rows if x['parameters'] == par]
        out[par] = {k: {'mean': float(np.mean([x[k] for x in r])), 'sd': float(np.std([x[k] for x in r], ddof=1)),
                        'min': float(np.min([x[k] for x in r])), 'max': float(np.max([x[k] for x in r]))}
                    for k in keys}
    paired = {}
    for k in ['nephrotoxicity_pct', 'cure_pct', 'combined_attainment_pct', 'predose_auroc_xgb', 'predose_auroc_lr']:
        d = [next(x[k] for x in rows if x['seed'] == s and x['parameters'] == 'source-aligned')
             - next(x[k] for x in rows if x['seed'] == s and x['parameters'] == 'primary') for s in SEEDS]
        paired[k] = {'mean_difference': float(np.mean(d)), 'sd': float(np.std(d, ddof=1)),
                     'min': float(np.min(d)), 'max': float(np.max(d))}
    out['paired_difference_source_minus_primary'] = paired
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    jobs = [(s, a) for a in (False, True) for s in SEEDS]
    t0 = time.time()
    # max_tasks_per_child=1: every cohort runs in a fresh process, because the
    # simulator module is modified in place for each parameter specification
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
        rows = list(ex.map(run_one, jobs))
    for r in rows:
        print(f"{r['parameters']:<15} seed {r['seed']:>5}: AUROC {r['predose_auroc_xgb']:.3f} (LR {r['predose_auroc_lr']:.3f}) | "
              f"neph {r['nephrotoxicity_pct']:.1f}% | cure {r['cure_pct']:.1f}% | combined {r['combined_attainment_pct']:.1f}% | {r['seconds']:.0f}s")
    res = {'seeds': SEEDS, 'source_aligned_changes': {**SOURCE_ALIGNED, 'sepsis_probs': SEPSIS_PROBS_SOURCE,
                                                      'ami_resistance': AMI_RESISTANCE_SOURCE},
           'per_replicate': rows, 'summary': summarise(rows), 'wall_seconds': time.time() - t0}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res['summary'], indent=1, default=float)[:3000])
    print(f'Wall time {res["wall_seconds"] / 60:.1f} min')


if __name__ == '__main__':
    main()
