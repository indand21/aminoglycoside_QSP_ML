#!/usr/bin/env python3
"""Revision 2: ML analyses requested in the second review round.

1. Out-of-fold (OOF) predictions for all 1,500 patients: nested 5-fold
   stratified CV in which the complete production procedure (scaler, Optuna
   tuning with 100 trials, refit) is repeated within each training fold. These
   replace the R1 Table 7 / Figure 6 values, which were not computed from model
   predictions.
2. Risk stratification (Table 7): pre-specified probability cutoffs (tertiles
   as sensitivity analysis), event rates with Wilson 95% CIs, relative risks with log-scale (Katz) CIs,
   subgroups, and the action-zone cutoffs used by Figure 6D. No NNT/NNH.
3. Feature-engineering ablation: each production model re-tuned with and without
   the engineered features under the identical procedure (Optuna TPE, 100
   trials, seed 42, inner 80/20 validation split), compared on the held-out set
   with a paired bootstrap CI and DeLong test.
4. Like-for-like comparisons with logistic regression on the primary 80/20
   split: the same two exposure predictors in both arms, and the full pre-dose
   and post-dose feature sets in both arms.
5. Sensitivity of post-dose discrimination to the 50 patients whose individual
   PK features were median-imputed (no PopPK estimate in the R1 fit).

Outputs: results/revision2/{oof_predictions.csv, risk_stratification.json,
feature_ablation.json, lr_comparisons.json, imputed_pk_sensitivity.json}

Usage:
    python r2_ml_analyses.py              # everything
    python r2_ml_analyses.py --skip-ablation
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from r2_ml_common import (ENGINEERED, OUT_DIR, engineer, production_setup, quiet,
                          raw_features)

warnings.filterwarnings('ignore')
SEED = 42
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def rr_ci(k1, n1, k0, n0, z=1.96):
    """Relative risk group1 vs group0 with log-scale (Katz) CI; 0.5 correction if a cell is 0."""
    if k1 == 0 or k0 == 0:
        k1, k0, n1, n0 = k1 + 0.5, k0 + 0.5, n1 + 1, n0 + 1
    rr = (k1 / n1) / (k0 / n0)
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return rr, rr * np.exp(-z * se), rr * np.exp(z * se)


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, p1, p2):
    """Paired DeLong test for two correlated AUCs. Returns (auc1, auc2, p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        p = np.asarray(p)
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(p)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[pos] - tx) / n)
        v01.append(1 - (tz[neg] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    zstat = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(zstat)))


def paired_boot_diff(y, p1, p2, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_auc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    a = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def compare(name_a, pa, name_b, pb, y):
    a1, a2, p = delong(y, pa, pb)
    lo, hi = paired_boot_diff(y, pa, pb)
    # full precision is stored; rounding happens only when tables are rendered
    return {name_a: float(a1), name_b: float(a2),
            f'{name_a}_ci': [round(v, 3) for v in boot_auc_ci(y, pa)],
            f'{name_b}_ci': [round(v, 3) for v in boot_auc_ci(y, pb)],
            'difference': float(a1 - a2), 'difference_95ci': [float(lo), float(hi)],
            'delong_p': float(p), 'n_test': int(len(y)), 'n_events': int(np.sum(y))}


# --------------------------------------------------------------------------- #
# 1-2. Out-of-fold predictions and risk stratification
# --------------------------------------------------------------------------- #
def oof_predictions(setup, pipe):
    """Nested 5-fold cross-validated predictions for all patients.

    The saved production models contain the fitted trees but not the tuned
    hyperparameters, so the complete production procedure (scaler, Optuna TPE
    with 100 trials on an inner 80/20 split, refit) is re-run inside each
    training fold; each patient's prediction comes from a model whose tuning and
    fitting never saw that patient."""
    pid = next(iter(setup.values()))['patient_id']
    out = pd.DataFrame({'patient_id': pid})
    for name, e in setup.items():
        X, y = e['X'].values, e['y']
        oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
            oof[te], _ = tune_and_fit(pipe, X, y, tr, te)
            print(f'    {name}: fold {k + 1}/5 done', flush=True)
        out[f'oof_{name}'] = oof
        out[f'y_{name}'] = y
        print(f'  OOF {name:<26} AUROC {roc_auc_score(y, oof):.3f}', flush=True)
    return out


def stratify(p, y, cuts):
    groups = np.digitize(p, cuts)  # 0 low, 1 medium, 2 high
    rows = []
    k0, n0 = int(y[groups == 0].sum()), int((groups == 0).sum())
    for g, label in enumerate(['Low', 'Medium', 'High']):
        mask = groups == g
        k, n = int(y[mask].sum()), int(mask.sum())
        lo, hi = wilson(k, n)
        row = {'group': label, 'n': n, 'events': k, 'rate': k / n,
               'rate_ci': [lo, hi], 'pred_mean': float(p[mask].mean())}
        if g > 0:
            rr, rlo, rhi = rr_ci(k, n, k0, n0)
            row.update({'rr_vs_low': rr, 'rr_ci': [rlo, rhi]})
        rows.append(row)
    return rows, groups


# Pre-specified probability cutoffs (R1 Table 7D), used for Table 7 and Figure 6D
NEPH_CUTS = (0.15, 0.35)
CURE_CUTS = (0.55, 0.75)


def risk_stratification(oof, pipe):
    res = {'method': ('Nested 5-fold stratified cross-validated (out-of-fold) predictions for all '
                      '1,500 patients; the production tuning procedure (Optuna, 100 trials) and '
                      'fit were repeated within each training fold. '
                      'Risk groups use the pre-specified probability cutoffs '
                      f'{NEPH_CUTS} for nephrotoxicity and {CURE_CUTS} for cure; '
                      'tertile groupings are reported as a sensitivity analysis.')}
    data = pipe.data
    for name in ['nephrotoxicity_postdose', 'nephrotoxicity_predose', 'clinical_cure']:
        p, y = oof[f'oof_{name}'].to_numpy(), oof[f'y_{name}'].to_numpy()
        cuts = CURE_CUTS if name == 'clinical_cure' else NEPH_CUTS
        rows, groups = stratify(p, y, cuts)
        tcuts = np.percentile(p, [100 / 3, 200 / 3])
        trows, _ = stratify(p, y, tcuts)
        res[name] = {'cutoffs': list(cuts), 'groups': rows, 'oof_auroc': roc_auc_score(y, p),
                     'tertile_cutoffs': tcuts.tolist(), 'tertile_groups': trows}
        oof[f'riskgroup_{name}'] = groups

    # Subgroups (post-dose nephrotoxicity, same cutoffs)
    p = oof['oof_nephrotoxicity_postdose'].to_numpy()
    y = oof['y_nephrotoxicity_postdose'].to_numpy()
    g = oof['riskgroup_nephrotoxicity_postdose'].to_numpy()
    sub_defs = {
        'Age <65 years': data['age'] < 65, 'Age >=65 years': data['age'] >= 65,
        'CrCL >80 mL/min': data['baseline_crcl'] > 80, 'CrCL <=80 mL/min': data['baseline_crcl'] <= 80,
        'Diabetes present': data['diabetes'].astype(bool), 'No diabetes': ~data['diabetes'].astype(bool),
        'APACHE II <20': data['apache_ii'] < 20, 'APACHE II >=20': data['apache_ii'] >= 20,
        'Amikacin': data['drug'] == 'amikacin', 'Gentamicin': data['drug'] == 'gentamicin',
    }
    subs = []
    for label, mask in sub_defs.items():
        mask = np.asarray(mask)
        hi, lo = mask & (g == 2), mask & (g == 0)
        kh, nh, kl, nl = int(y[hi].sum()), int(hi.sum()), int(y[lo].sum()), int(lo.sum())
        rr, rlo, rhi = rr_ci(kh, nh, kl, nl)
        subs.append({'subgroup': label, 'n': int(mask.sum()), 'events': int(y[mask].sum()),
                     'auroc': roc_auc_score(y[mask], p[mask]),
                     'high_n': nh, 'high_events': kh, 'high_rate': kh / nh if nh else np.nan,
                     'high_rate_ci': list(wilson(kh, nh)),
                     'low_n': nl, 'low_events': kl, 'low_rate': kl / nl if nl else np.nan,
                     'low_rate_ci': list(wilson(kl, nl)),
                     'rr': rr, 'rr_ci': [rlo, rhi]})
    res['subgroups_postdose'] = subs

    # Action zones (Figure 6D) using the same cutoffs
    pn, pc = oof['oof_nephrotoxicity_postdose'].to_numpy(), oof['oof_clinical_cure'].to_numpy()
    zone = np.where((pn < NEPH_CUTS[0]) & (pc > CURE_CUTS[1]), 'favourable',
                    np.where((pn > NEPH_CUTS[1]) | (pc < CURE_CUTS[0]), 'unfavourable',
                             'intermediate'))
    oof['action_zone'] = zone
    zones = ['favourable', 'intermediate', 'unfavourable']
    res['action_zones'] = {
        'definition': (f'favourable: P(nephrotoxicity) < {NEPH_CUTS[0]} and P(cure) > '
                       f'{CURE_CUTS[1]}; unfavourable: P(nephrotoxicity) > {NEPH_CUTS[1]} or '
                       f'P(cure) < {CURE_CUTS[0]}; intermediate: all others'),
        'counts': {z: int((zone == z).sum()) for z in zones},
        'nephrotoxicity_rate': {z: float(oof.loc[zone == z, 'y_nephrotoxicity_postdose'].mean())
                                for z in zones},
        'cure_rate': {z: float(oof.loc[zone == z, 'y_clinical_cure'].mean()) for z in zones},
    }
    return res


# --------------------------------------------------------------------------- #
# 3. Feature-engineering ablation
# --------------------------------------------------------------------------- #
def tune_and_fit(pipe, X, y, tr, te):
    """Replicates Phase 4: scaler on training split, Optuna (100 trials) on an
    inner 80/20 split of the training data, refit on the full training split."""
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte, ytr = sc.transform(X[tr]), sc.transform(X[te]), y[tr]
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
    # Phase 4 writes Optuna plots to pipe.results_dir; send them to a scratch
    # folder so the R1 artifacts in results/phase4_ml_enhanced are not overwritten
    saved_dir, pipe.results_dir = pipe.results_dir, OUT_DIR / '_optuna_scratch'
    pipe.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, params, val = quiet(pipe.optimize_hyperparameters_optuna, a, ya, b, yb,
                                   scale_pos_weight=spw, n_trials=100)
    finally:
        pipe.results_dir = saved_dir
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1], val


def ablation(pipe, setup):
    out = {}
    for name, e in setup.items():
        X_with = e['X']
        X_without = X_with[[c for c in X_with.columns if c not in ENGINEERED]]
        y, tr, te = e['y'], e['train'], e['test']
        p_with, v_with = tune_and_fit(pipe, X_with.values, y, tr, te)
        p_without, v_without = tune_and_fit(pipe, X_without.values, y, tr, te)
        r = compare('with_engineered', p_with, 'without_engineered', p_without, y[te])
        r.update({'n_features_with': X_with.shape[1], 'n_features_without': X_without.shape[1],
                  'engineered_removed': [c for c in X_with.columns if c in ENGINEERED],
                  'inner_val_auc_with': v_with, 'inner_val_auc_without': v_without})
        out[name] = r
        print(f'  ablation {name:<26} with {r["with_engineered"]:.3f}  without '
              f'{r["without_engineered"]:.3f}  diff {r["difference"]:+.3f} '
              f'{r["difference_95ci"]}  p={r["delong_p"]:.3f}')
    return out


# --------------------------------------------------------------------------- #
# 4. Like-for-like logistic regression comparisons
# --------------------------------------------------------------------------- #
def lr_fit_predict(X, y, tr, te, penalty_C=1.0):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=penalty_C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return m.predict_proba(sc.transform(X[te]))[:, 1]


def xgb_fixed_fit_predict(X, y, tr, te):
    """Same untuned configuration as the R1 MIPD benchmark (phase4_mipd_benchmark.py)."""
    spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, random_state=SEED,
                          eval_metric='logloss', tree_method='hist')
    return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def lr_comparisons(setup):
    out = {}
    post, pre = setup['nephrotoxicity_postdose'], setup['nephrotoxicity_predose']
    y, tr, te = post['y'], post['train'], post['test']
    expo = post['X'][['Cmin', 'AUC24']].values
    out['two_exposure_predictors'] = compare(
        'xgboost', xgb_fixed_fit_predict(expo, y, tr, te),
        'logistic', lr_fit_predict(expo, y, tr, te), y[te])
    out['two_exposure_predictors']['predictors'] = ['Cmin', 'AUC24']
    p_post = post['model'].predict_proba(post['scaler'].transform(post['X'].values[te]))[:, 1]
    out['postdose_55_features'] = compare(
        'xgboost_production', p_post,
        'logistic_l2', lr_fit_predict(post['X'].values, y, tr, te), y[te])
    p_pre = pre['model'].predict_proba(pre['scaler'].transform(pre['X'].values[te]))[:, 1]
    out['predose_26_features'] = compare(
        'xgboost_production', p_pre,
        'logistic_l2', lr_fit_predict(pre['X'].values, y, tr, te), y[te])
    out['note'] = ('All comparisons use the primary 80/20 split (N = 300 held-out, 99 events) '
                   'and give both arms identical predictors. Logistic regression: standardised '
                   'predictors, L2 penalty C = 1. Two-predictor XGBoost arm: fixed configuration '
                   'of the R1 benchmark. DeLong test for paired AUCs; 95% CI of the difference by '
                   f'paired bootstrap ({N_BOOT} resamples).')
    for k, v in out.items():
        if isinstance(v, dict):
            print(f'  LR comparison {k:<26} {v}')
    return out


# --------------------------------------------------------------------------- #
# 5. Imputed-PK sensitivity
# --------------------------------------------------------------------------- #
def imputed_pk_sensitivity(setup):
    pk = pd.read_csv('results/phase3_pkpd/pkpd_indices.csv')[['patient_id', 'CL_bayes']]
    out = {}
    for name in ['nephrotoxicity_postdose', 'clinical_cure']:
        e = setup[name]
        pid = e['patient_id']
        imputed = ~pd.Series(pid).isin(pk.dropna()['patient_id']).to_numpy()
        te = e['test']
        p = e['model'].predict_proba(e['scaler'].transform(e['X'].values[te]))[:, 1]
        keep = ~imputed[te]
        out[name] = {'n_imputed_total': int(imputed.sum()),
                     'n_imputed_in_test': int(imputed[te].sum()),
                     'n_imputed_in_train': int(imputed[e['train']].sum()),
                     'auroc_all_test': roc_auc_score(e['y'][te], p),
                     'auroc_test_excluding_imputed': roc_auc_score(e['y'][te][keep], p[keep])}
        print(f'  imputed-PK sensitivity {name}: {out[name]}')
    return out


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-ablation', action='store_true')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe, setup = production_setup()

    print('Out-of-fold predictions ...')
    oof = oof_predictions(setup, pipe)
    rs = risk_stratification(oof, pipe)
    oof.to_csv(OUT_DIR / 'oof_predictions.csv', index=False)
    (OUT_DIR / 'risk_stratification.json').write_text(json.dumps(to_jsonable(rs), indent=2))

    print('Imputed-PK sensitivity ...')
    (OUT_DIR / 'imputed_pk_sensitivity.json').write_text(
        json.dumps(to_jsonable(imputed_pk_sensitivity(setup)), indent=2))

    print('Logistic-regression comparisons ...')
    (OUT_DIR / 'lr_comparisons.json').write_text(
        json.dumps(to_jsonable(lr_comparisons(setup)), indent=2))

    if not args.skip_ablation:
        print('Feature-engineering ablation (6 Optuna studies x 100 trials) ...')
        (OUT_DIR / 'feature_ablation.json').write_text(
            json.dumps(to_jsonable(ablation(pipe, setup)), indent=2))


if __name__ == '__main__':
    main()
