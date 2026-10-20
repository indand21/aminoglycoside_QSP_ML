#!/usr/bin/env python3
"""Revision 2: vectorised full-cohort dose optimisation with a ground-truth check.

Changes from phase5_dose_optimization.py (R1), in response to both reviewers:

* Full cohort (N = 1,500) instead of a 50-patient subsample: candidate doses are
  scored in one batched XGBoost call per dose level (no per-patient loop).
* Drug-specific dose grids that always contain the patient's baseline dose:
  gentamicin 50-800 mg in 10-mg steps, amikacin 250-3,000 mg in 25-mg steps.
  The patient's own dosing interval (24, 36 or 48 h) is kept unchanged.
* Drug-specific trough threshold in the objective (2.0 mg/L gentamicin,
  2.5 mg/L amikacin); R1 applied 2.0 mg/L to both drugs.
* Exposure features for a candidate dose d. Aminoglycoside PK is linear, so
  every concentration in the patient's record scales by r = d / d0, where d0 is
  the administered dose. Each dose-dependent feature is recomputed with the
  Phase 3 definitions applied to the scaled record:
    Cmax (maximum observed), Cmin (minimum observed), AUC24 (trapezoid over the
    last 24 h of sampling, normalised to 24 h), Cmax/MIC and AUC24/MIC all scale
    by r; time above MIC and AUC above MIC are recomputed from the scaled,
    interpolated record; fluctuation index, peak-trough ratio and time to peak
    are scale-invariant. Covariates and individual Bayesian PK parameters are
    dose-independent. Engineered features are then rebuilt with Phase 4's own
    `engineer_features`. At d = d0 the features equal the training features
    exactly (checked at run time).
* Ground-truth ("oracle") check. The simulator is reproduced from its seed
  (r2_simulator_truth.py), so each patient's true CL, Vc, Q and Vp are known.
  For the baseline and recommended doses, noise-free concentrations at the
  patient's actual sampling times are simulated from the true parameters and
  passed through the true outcome equations. This tests whether the ML-based
  recommendations improve the outcome probabilities that actually generated
  the data, and compares them with the dose that maximises the same objective
  under the true model ("oracle-optimal" dose).
* Uncertainty: 95% CIs are percentile bootstrap intervals over patients (2,000
  resamples). They reflect patient-sampling variability only, conditional on
  the fitted PopPK and ML models; PopPK posterior and ML-model uncertainty are
  not propagated.

Outputs (results/revision2/):
    dose_optimization_patients.csv   per-patient doses, predicted and true outcomes
    dose_optimization_summary.json   cohort summaries with bootstrap CIs
    dose_optimization_weights.json   objective-weight sensitivity (full cohort)
    dose_optimization_trough_sensitivity.json  guideline gentamicin trough limit (1.0 mg/L)
    dose_optimization_timing.json    run time

Usage:
    python r2_dose_optimization.py
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from r2_ml_common import OUT_DIR, REPO_ROOT, engineer, production_setup
from r2_simulator_truth import load_truth, true_p_cure, true_p_nephrotoxicity

warnings.filterwarnings('ignore')

GRIDS = {'gentamicin': np.arange(50, 800 + 1, 10.0),
         'amikacin': np.arange(250, 3000 + 1, 25.0)}
TROUGH_LIMIT = {'gentamicin': 2.0, 'amikacin': 2.5}
# Sensitivity analysis: guideline gentamicin trough threshold in the objective
# (Abdul-Aziz et al. 2020: toxicity above 1 mg/L for extended-interval dosing).
# Only the optimiser's objective changes; the ML models and the simulator's true
# outcome model (which uses 2.0 mg/L for gentamicin) are unchanged.
TROUGH_SCENARIOS = {'Gentamicin trough limit 1.0 mg/L': {'gentamicin': 1.0, 'amikacin': 2.5}}
WEIGHTS = {'cure': 0.40, 'safety': 0.30, 'cmax_target': 0.20, 'trough_safety': 0.10}
WEIGHT_PROFILES = {
    'Baseline (0.40/0.30/0.20/0.10)': WEIGHTS,
    'Safety-weighted': {'cure': 0.25, 'safety': 0.45, 'cmax_target': 0.15, 'trough_safety': 0.15},
    'Efficacy-weighted': {'cure': 0.55, 'safety': 0.20, 'cmax_target': 0.20, 'trough_safety': 0.05},
    'Equal weights': {'cure': 0.25, 'safety': 0.25, 'cmax_target': 0.25, 'trough_safety': 0.25},
    'Cmax-weighted': {'cure': 0.30, 'safety': 0.25, 'cmax_target': 0.35, 'trough_safety': 0.10},
}
SCALE_LINEAR = ['Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC']
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# Phase 3 time-based features on a scaled concentration record
# --------------------------------------------------------------------------- #
def phase3_window(times, concs):
    """Last-24-h window and fine interpolation grid, as in phase3 (lines ~420-436)."""
    max_time = times.max()
    interval = 24.0
    if max_time >= interval:
        m = times >= (max_time - interval)
        t, c = times[m] - (max_time - interval), concs[m]
    else:
        t, c = times, concs
        interval = max_time if max_time > 0 else 24.0
    tf = np.linspace(t[0], t[-1], 1000)
    cf = np.interp(tf, t, c)
    return tf, cf, interval


class TimeFeatureCache:
    """Phase 3 interpolation grids for every patient, stacked for vectorised use."""

    def __init__(self, windows, raw):
        n = len(windows)
        self.has = np.array([w[0] is not None for w in windows])
        self.TF = np.zeros((n, 1000))
        self.CF = np.zeros((n, 1000))
        self.interval = np.full(n, 24.0)
        for i, w in enumerate(windows):
            if w[0] is not None:
                self.TF[i], self.CF[i], self.interval[i] = w
        self.span = self.TF[:, -1] - self.TF[:, 0]
        self.raw = raw[['time_above_MIC', 'time_above_MIC_percent', 'AUC_above_MIC']].to_numpy()

    def subset(self, idx):
        c = object.__new__(TimeFeatureCache)
        for k in ['has', 'TF', 'CF', 'interval', 'span', 'raw']:
            setattr(c, k, getattr(self, k)[idx])
        return c

    def features(self, mic, r):
        """time_above_MIC, time_above_MIC_percent, AUC_above_MIC at per-patient scale r."""
        conc = r[:, None] * self.CF
        t_above = (conc > mic[:, None]).sum(axis=1) / self.CF.shape[1] * self.span
        auc_above = np.trapezoid(np.maximum(conc - mic[:, None], 0.0), self.TF, axis=1)
        out = np.column_stack([t_above, t_above / self.interval * 100.0, auc_above])
        out[~self.has] = self.raw[~self.has]
        return out


# --------------------------------------------------------------------------- #
# True PK (oracle)
# --------------------------------------------------------------------------- #
def true_concentrations(sample_t, dose_t, dose_amt, CL, Vc, Q, Vp):
    """Noise-free 2-compartment, 1-h infusion (identical to the simulator)."""
    k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
    s = k10 + k12 + k21
    sq = np.sqrt(s ** 2 - 4 * k10 * k21)
    a, b = 0.5 * (s + sq), 0.5 * (s - sq)
    A, B = (a - k21) / (a - b) / Vc, (b - k21) / (b - a) / Vc
    dt = sample_t[:, None] - dose_t[None, :]
    on = dt >= 0
    inf = dt <= 1
    c_inf = dose_amt * (A * (1 - np.exp(-a * dt)) / a + B * (1 - np.exp(-b * dt)) / b)
    c_post = dose_amt * (A * (1 - np.exp(-a)) / a * np.exp(-a * (dt - 1))
                         + B * (1 - np.exp(-b)) / b * np.exp(-b * (dt - 1)))
    return np.where(on, np.where(inf, c_inf, c_post), 0.0).sum(axis=1)


def oracle_inputs(pdata, truth, conc, dosing):
    """Per patient: noise-free Cmax (max peak sample), mean trough, AUC=dose/CL at d0."""
    t = truth.set_index('patient_id')
    rows = []
    for pid, g in conc.groupby('patient_id'):
        dz = dosing[dosing['patient_id'] == pid]
        tp = t.loc[pid]
        pred = true_concentrations(g['time'].to_numpy(), dz['time'].to_numpy(),
                                   dz['dose'].to_numpy(), tp['CL_individual'],
                                   tp['Vc_individual'], tp['Q_individual'], tp['Vp_individual'])
        st = g['sample_type'].to_numpy()
        rows.append({'patient_id': pid,
                     'true_cmax0': pred[st == 'peak'].max() if (st == 'peak').any() else np.nan,
                     'true_trough0': pred[st == 'trough'].mean() if (st == 'trough').any() else np.nan,
                     'true_auc0': dz['dose'].iloc[0] / tp['CL_individual']})
    return pdata[['patient_id']].merge(pd.DataFrame(rows), on='patient_id', how='left')


def true_outcomes(r, o, pat):
    """True P(AKI), P(cure) and noise-free exposures at scale factor(s) r."""
    cmax = r * o['true_cmax0']
    trough = r * o['true_trough0']
    auc = r * o['true_auc0']
    mic = pat['MIC']
    cmax_mic = np.where(np.isnan(cmax), 1.0, cmax / mic)          # simulator: fillna(1)
    tr_neph = np.where(np.isnan(trough), 1.0, trough)               # simulator: fillna(1)
    tr_cure = np.where(np.isnan(trough), 0.0, trough)               # simulator: fillna(0)
    ami = pat['drug_name'] == 'amikacin'
    pn = true_p_nephrotoxicity(tr_neph, ami, pat['age'], pat['diabetes_i'], pat['baseline_scr'],
                               pat['mv_i'], pat['vp_i'], pat['baseline_albumin'], pat['sofa_score'])
    pc = true_p_cure(cmax_mic, auc / mic, tr_cure, pat['apache_ii'], pat['sofa_score'],
                     pat['acinetobacter'], pat['baseline_albumin'], pat['baseline_crcl'])
    return pn, pc, cmax_mic, tr_neph


def objective(p_cure, p_aki, cmax_mic, cmin, limit, w):
    return (w['cure'] * p_cure + w['safety'] * (1 - p_aki)
            + w['cmax_target'] / (1 + np.exp(-0.5 * (cmax_mic - 8.0)))
            + w['trough_safety'] / (1 + np.exp(2.0 * (cmin - limit))))


# --------------------------------------------------------------------------- #
# Main optimisation
# --------------------------------------------------------------------------- #
def build_patient_table(pipe):
    d = pipe.data
    pk = pd.read_csv(REPO_ROOT / 'results' / 'phase3_pkpd' / 'pkpd_indices.csv')[['patient_id', 'MIC']]
    dosing = pd.read_csv(REPO_ROOT / 'data' / 'dosing.csv')
    first = dosing.groupby('patient_id').agg(d0=('dose', 'first'), n_doses=('dose', 'size'))
    interval = dosing.groupby('patient_id')['time'].apply(lambda t: float(np.diff(t).min()))
    pat = d[['patient_id', 'age', 'weight', 'baseline_crcl', 'baseline_scr', 'baseline_albumin',
             'apache_ii', 'sofa_score', 'drug']].copy()
    pat['drug_name'] = d['drug'].astype(str).values
    pat['diabetes_i'] = d['diabetes'].astype(int).values
    pat['mv_i'] = d['mechanical_ventilation'].astype(int).values
    pat['vp_i'] = d['vasopressor_use'].astype(int).values
    pdat = pd.read_csv(REPO_ROOT / 'data' / 'patient_data.csv')[['patient_id', 'pathogen']]
    pat = pat.merge(pdat, on='patient_id', how='left').merge(pk, on='patient_id', how='left')
    pat['acinetobacter'] = (pat['pathogen'] == 'acinetobacter').astype(int)
    pat = pat.merge(first, left_on='patient_id', right_index=True)
    pat['interval_h'] = pat['patient_id'].map(interval)
    # Same categories as Tables 2 and 4: <50 impaired, 50-130 normal, >130 augmented
    pat['renal_group'] = pd.Categorical(
        np.select([pat['baseline_crcl'] < 50, pat['baseline_crcl'] > 130],
                  ['Impaired (<50)', 'Augmented (>130)'], 'Normal (50-130)'),
        categories=['Impaired (<50)', 'Normal (50-130)', 'Augmented (>130)'])
    return pat.reset_index(drop=True), dosing


def dose_features(pipe, raw, nan_mask, tcache, pat, r):
    """Raw (pre-engineering) post-dose feature table for per-patient scale factors r."""
    out = raw.copy()
    for c in SCALE_LINEAR:
        out[c] = np.where(nan_mask[c], raw[c], raw[c] * r)
    tf = tcache.features(pat['MIC'].to_numpy(float), r)
    out['time_above_MIC'], out['time_above_MIC_percent'], out['AUC_above_MIC'] = tf.T
    return out


def predict_both(setup, X_eng):
    neph, cure = setup['nephrotoxicity_postdose'], setup['clinical_cure']
    p_aki = neph['model'].predict_proba(neph['scaler'].transform(X_eng[neph['features']].values))[:, 1]
    p_cure = cure['model'].predict_proba(cure['scaler'].transform(X_eng[cure['features']].values))[:, 1]
    return p_aki, p_cure


def optimise_profile(grid_scores, pat, w, limits):
    """Recommended dose per patient for objective weights w; limits overrides the
    drug-specific trough thresholds (used for both the ML and the true objective)."""
    rows = []
    for drug, gs in grid_scores.items():
        limit = gs['limit'] if limits is None else limits[drug]
        S = objective(gs['P_cure'], gs['P_aki'], gs['CMAX_MIC'], gs['CMIN'], limit, w)
        S_true = objective(gs['T_cure'], gs['T_aki'], gs['T_cmic'], gs['T_tr'], limit, w)
        j_opt = S.argmax(axis=1)
        j_base = np.searchsorted(gs['grid'], gs['d0'])
        j_orc = S_true.argmax(axis=1)
        for k, i in enumerate(gs['idx']):
            jo, jb, jr = j_opt[k], j_base[k], j_orc[k]
            rows.append({
                'patient_id': pat.at[i, 'patient_id'], 'drug': drug,
                'weight_kg': pat.at[i, 'weight'], 'baseline_crcl': pat.at[i, 'baseline_crcl'],
                'renal_group': pat.at[i, 'renal_group'], 'interval_h': pat.at[i, 'interval_h'],
                'mic': pat.at[i, 'MIC'],
                'baseline_dose': gs['d0'][k], 'optimal_dose': gs['grid'][jo],
                'oracle_dose': gs['grid'][jr],
                'baseline_mgkg': gs['d0'][k] / pat.at[i, 'weight'],
                'optimal_mgkg': gs['grid'][jo] / pat.at[i, 'weight'],
                'oracle_mgkg': gs['grid'][jr] / pat.at[i, 'weight'],
                'pred_aki_base': gs['P_aki'][k, jb], 'pred_aki_opt': gs['P_aki'][k, jo],
                'pred_cure_base': gs['P_cure'][k, jb], 'pred_cure_opt': gs['P_cure'][k, jo],
                'cmax_mic_base': gs['CMAX_MIC'][k, jb], 'cmax_mic_opt': gs['CMAX_MIC'][k, jo],
                'cmin_base': gs['CMIN'][k, jb], 'cmin_opt': gs['CMIN'][k, jo],
                'score_base': S[k, jb], 'score_opt': S[k, jo],
                'true_aki_base': gs['T_aki'][k, jb], 'true_aki_opt': gs['T_aki'][k, jo],
                'true_aki_oracle': gs['T_aki'][k, jr],
                'true_cure_base': gs['T_cure'][k, jb], 'true_cure_opt': gs['T_cure'][k, jo],
                'true_cure_oracle': gs['T_cure'][k, jr],
                'true_cmax_mic_base': gs['T_cmic'][k, jb], 'true_cmax_mic_opt': gs['T_cmic'][k, jo],
                'true_trough_base': gs['T_tr'][k, jb], 'true_trough_opt': gs['T_tr'][k, jo],
                'true_score_base': S_true[k, jb], 'true_score_opt': S_true[k, jo],
                'true_score_oracle': S_true[k, jr],
            })
    return pd.DataFrame(rows).set_index('patient_id').loc[pat['patient_id']].reset_index()


def run():
    t_start = time.time()
    pipe, setup = production_setup()
    neph = setup['nephrotoxicity_postdose']
    raw = neph['raw'].copy()                      # 42 raw post-dose columns, imputed
    nan_mask = {c: pipe.data[c].isna().to_numpy() for c in SCALE_LINEAR}
    pat, dosing = build_patient_table(pipe)
    assert (pat['patient_id'].to_numpy() == neph['patient_id']).all()

    # Concentration-record windows for time-based features
    conc = pd.read_csv(REPO_ROOT / 'data' / 'concentrations.csv')
    groups = {pid: g.sort_values('time') for pid, g in conc.groupby('patient_id')}
    time_cache = []
    for pid in pat['patient_id']:
        g = groups.get(pid)
        time_cache.append(None if g is None or len(g) == 0 else
                          phase3_window(g['time'].to_numpy(), g['concentration'].to_numpy()))
    time_cache = TimeFeatureCache([(None, None, None) if tc is None else tc
                                   for tc in time_cache], raw)

    # Check: r = 1 reproduces the training features and predictions exactly
    X1 = engineer(pipe, dose_features(pipe, raw, nan_mask, time_cache, pat, np.ones(len(pat))))
    p_aki_0, p_cure_0 = predict_both(setup, X1)
    ref_aki = neph['model'].predict_proba(neph['scaler'].transform(neph['X'].values))[:, 1]
    max_dev = float(np.abs(p_aki_0 - ref_aki).max())
    print(f'r = 1 check: max |P(AKI) - production prediction| = {max_dev:.2e}')

    # Oracle inputs
    truth = load_truth()
    o = oracle_inputs(pat, truth, conc, dosing)

    results = []
    grid_scores = {}
    t_opt = time.time()
    for drug, grid in GRIDS.items():
        idx = np.where(pat['drug_name'] == drug)[0]
        d0 = pat.loc[idx, 'd0'].to_numpy()
        assert np.isin(d0, grid).all(), f'baseline dose not on {drug} grid'
        sub_raw = raw.iloc[idx].reset_index(drop=True)
        sub_nan = {c: nan_mask[c][idx] for c in SCALE_LINEAR}
        sub_tc = time_cache.subset(idx)
        sub_pat = pat.iloc[idx].reset_index(drop=True)
        limit = TROUGH_LIMIT[drug]
        n, G = len(idx), len(grid)
        P_aki, P_cure = np.zeros((n, G)), np.zeros((n, G))
        CMAX_MIC, CMIN = np.zeros((n, G)), np.zeros((n, G))
        for j, dose in enumerate(grid):
            r = dose / d0
            fr = dose_features(pipe, sub_raw, sub_nan, sub_tc, sub_pat, r)
            P_aki[:, j], P_cure[:, j] = predict_both(setup, engineer(pipe, fr))
            CMAX_MIC[:, j], CMIN[:, j] = fr['Cmax_MIC'].to_numpy(), fr['Cmin'].to_numpy()
        # True outcomes over the same grid
        so = o.iloc[idx].reset_index(drop=True)
        T_aki, T_cure, T_cmic, T_tr = (np.zeros((n, G)) for _ in range(4))
        for j, dose in enumerate(grid):
            T_aki[:, j], T_cure[:, j], T_cmic[:, j], T_tr[:, j] = true_outcomes(dose / d0, so, sub_pat)
        grid_scores[drug] = dict(idx=idx, grid=grid, d0=d0, limit=limit, P_aki=P_aki,
                                 P_cure=P_cure, CMAX_MIC=CMAX_MIC, CMIN=CMIN, T_aki=T_aki,
                                 T_cure=T_cure, T_cmic=T_cmic, T_tr=T_tr)
    t_opt = time.time() - t_opt

    # Optimise under each weight profile, and under the trough-threshold scenario
    profiles = {pname: optimise_profile(grid_scores, pat, w, None)
                for pname, w in WEIGHT_PROFILES.items()}
    trough_profiles = {name: optimise_profile(grid_scores, pat, WEIGHTS, limits)
                       for name, limits in TROUGH_SCENARIOS.items()}
    return pat, profiles, trough_profiles, dict(r1_check_max_dev=max_dev, scoring_seconds=t_opt,
                               total_seconds=time.time() - t_start,
                               n_patients=len(pat),
                               n_candidate_evaluations=int(sum(len(g['idx']) * len(g['grid'])
                                                               for g in grid_scores.values())))


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def boot_mean(x, n_boot=N_BOOT, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarise(df):
    pct = 100 * (df['optimal_dose'] - df['baseline_dose']) / df['baseline_dose']
    s = {
        'n': int(len(df)),
        'baseline_dose_mg': boot_mean(df['baseline_dose']),
        'optimal_dose_mg': boot_mean(df['optimal_dose']),
        'baseline_mgkg': boot_mean(df['baseline_mgkg']),
        'optimal_mgkg': boot_mean(df['optimal_mgkg']),
        'dose_change_pct': boot_mean(pct),
        'n_dose_reduced': int((df['optimal_dose'] < df['baseline_dose']).sum()),
        'n_dose_unchanged': int((df['optimal_dose'] == df['baseline_dose']).sum()),
        'n_dose_increased': int((df['optimal_dose'] > df['baseline_dose']).sum()),
        # ML-predicted changes (percentage points)
        'pred_aki_base_pct': boot_mean(100 * df['pred_aki_base']),
        'pred_aki_opt_pct': boot_mean(100 * df['pred_aki_opt']),
        'pred_aki_change_pp': boot_mean(100 * (df['pred_aki_opt'] - df['pred_aki_base'])),
        'pred_cure_base_pct': boot_mean(100 * df['pred_cure_base']),
        'pred_cure_opt_pct': boot_mean(100 * df['pred_cure_opt']),
        'pred_cure_change_pp': boot_mean(100 * (df['pred_cure_opt'] - df['pred_cure_base'])),
        # Ground truth (simulator) changes
        'true_aki_base_pct': boot_mean(100 * df['true_aki_base']),
        'true_aki_opt_pct': boot_mean(100 * df['true_aki_opt']),
        'true_aki_change_pp': boot_mean(100 * (df['true_aki_opt'] - df['true_aki_base'])),
        'true_cure_base_pct': boot_mean(100 * df['true_cure_base']),
        'true_cure_opt_pct': boot_mean(100 * df['true_cure_opt']),
        'true_cure_change_pp': boot_mean(100 * (df['true_cure_opt'] - df['true_cure_base'])),
        # Target attainment (dose-scaled observed exposures, as used by the optimiser)
        'cmax_mic_ge8_base_pct': float(100 * (df['cmax_mic_base'] >= 8).mean()),
        'cmax_mic_ge8_opt_pct': float(100 * (df['cmax_mic_opt'] >= 8).mean()),
        'trough_safe_base_pct': float(100 * (df['cmin_base'] < np.where(df['drug'] == 'amikacin', 2.5, 2.0)).mean()),
        'trough_safe_opt_pct': float(100 * (df['cmin_opt'] < np.where(df['drug'] == 'amikacin', 2.5, 2.0)).mean()),
        # Objective
        'pct_improved_ml_objective': float(100 * (df['score_opt'] > df['score_base'] + 1e-12).mean()),
        'pct_improved_true_objective': float(100 * (df['true_score_opt'] > df['true_score_base'] + 1e-12).mean()),
        'pct_worse_true_objective': float(100 * (df['true_score_opt'] < df['true_score_base'] - 1e-12).mean()),
        'pct_true_aki_lower': float(100 * (df['true_aki_opt'] < df['true_aki_base'] - 1e-12).mean()),
        'pct_true_cure_higher': float(100 * (df['true_cure_opt'] > df['true_cure_base'] + 1e-12).mean()),
        'pct_true_cure_lower': float(100 * (df['true_cure_opt'] < df['true_cure_base'] - 1e-12).mean()),
        # Agreement of predicted with true changes, and with the oracle-optimal dose
        'corr_pred_true_aki_change': float(np.corrcoef(df['pred_aki_opt'] - df['pred_aki_base'],
                                                       df['true_aki_opt'] - df['true_aki_base'])[0, 1]),
        'corr_pred_true_cure_change': float(np.corrcoef(df['pred_cure_opt'] - df['pred_cure_base'],
                                                        df['true_cure_opt'] - df['true_cure_base'])[0, 1]),
        'oracle_dose_mg': boot_mean(df['oracle_dose']),
        'ml_minus_oracle_dose_mg': boot_mean(df['optimal_dose'] - df['oracle_dose']),
        'median_abs_ml_minus_oracle_pct': float((100 * (df['optimal_dose'] - df['oracle_dose']).abs()
                                                 / df['oracle_dose']).median()),
        'corr_ml_oracle_dose': float(np.corrcoef(df['optimal_dose'], df['oracle_dose'])[0, 1]),
        'true_score_regret_vs_oracle': boot_mean(df['true_score_oracle'] - df['true_score_opt']),
        'true_aki_oracle_change_pp': boot_mean(100 * (df['true_aki_oracle'] - df['true_aki_base'])),
        'true_cure_oracle_change_pp': boot_mean(100 * (df['true_cure_oracle'] - df['true_cure_base'])),
    }
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pat, profiles, trough_profiles, timing = run()
    base = profiles['Baseline (0.40/0.30/0.20/0.10)']
    base.to_csv(OUT_DIR / 'dose_optimization_patients.csv', index=False)

    test_ids = set(pat['patient_id'].iloc[
        __import__('r2_ml_common').split_indices(
            __import__('r2_ml_common').load_pipeline().data['nephrotoxicity'].astype(int).to_numpy())[1]])
    summary = {'all': summarise(base),
               'held_out_test_patients': summarise(base[base['patient_id'].isin(test_ids)])}
    for drug in ['amikacin', 'gentamicin']:
        summary[drug] = summarise(base[base['drug'] == drug])
        for interval, g in base[base['drug'] == drug].groupby('interval_h'):
            summary[f'{drug}_q{int(interval)}h'] = {
                'n': int(len(g)), 'baseline_dose_mg': float(g['baseline_dose'].mean()),
                'optimal_dose_mg': float(g['optimal_dose'].mean()),
                'baseline_mgkg': float(g['baseline_mgkg'].mean()),
                'optimal_mgkg': float(g['optimal_mgkg'].mean())}
    for rg, g in base.groupby('renal_group', observed=True):
        summary[f'renal: {rg}'] = summarise(g)
    summary['settings'] = {
        'grids_mg': {k: [float(v[0]), float(v[-1]), float(v[1] - v[0])] for k, v in GRIDS.items()},
        'trough_limit_mg_per_L': TROUGH_LIMIT, 'weights': WEIGHTS,
        'baseline_dose_in_candidate_set': True, 'interval': "patient's own (unchanged)",
        'ci_method': f'percentile bootstrap over patients, {N_BOOT} resamples; '
                     'patient-sampling uncertainty only'}
    (OUT_DIR / 'dose_optimization_summary.json').write_text(json.dumps(summary, indent=2, default=float))

    wsum = {}
    for pname, df in profiles.items():
        s = summarise(df)
        wsum[pname] = {'weights': WEIGHT_PROFILES[pname],
                       **{k: s[k] for k in ['optimal_dose_mg', 'dose_change_pct', 'pred_aki_change_pp',
                                            'pred_cure_change_pp', 'true_aki_change_pp',
                                            'true_cure_change_pp', 'pct_improved_true_objective']}}
    (OUT_DIR / 'dose_optimization_weights.json').write_text(json.dumps(wsum, indent=2, default=float))
    (OUT_DIR / 'dose_optimization_timing.json').write_text(json.dumps(timing, indent=2))

    tsum = {}
    for name, df in trough_profiles.items():
        lim = TROUGH_SCENARIOS[name]
        entry = {'trough_limits_mg_per_L': lim, 'weights': WEIGHTS}
        for grp, sub in [('all', df), ('amikacin', df[df['drug'] == 'amikacin']),
                         ('gentamicin', df[df['drug'] == 'gentamicin']),
                         ('gentamicin_target_attainable', df[(df['drug'] == 'gentamicin') & (df['cmax_mic_base'] >= 8)]),
                         ('gentamicin_target_not_attainable', df[(df['drug'] == 'gentamicin') & (df['cmax_mic_base'] < 8)])]:
            s_ = summarise(sub)
            s_['optimal_mgkg_median_iqr'] = [float(v) for v in np.percentile(sub['optimal_mgkg'], [50, 25, 75])]
            s_['n_at_grid_min'] = int((sub['optimal_dose'] == sub['drug'].map(lambda d: GRIDS[d][0])).sum())
            s_['n_at_grid_max'] = int((sub['optimal_dose'] == sub['drug'].map(lambda d: GRIDS[d][-1])).sum())
            s_['pct_cmin_below_limit_opt'] = float(100 * (sub['cmin_opt'] < sub['drug'].map(lim)).mean())
            s_['pct_cmin_below_limit_base'] = float(100 * (sub['cmin_base'] < sub['drug'].map(lim)).mean())
            entry[grp] = s_
        tsum[name] = entry
    (OUT_DIR / 'dose_optimization_trough_sensitivity.json').write_text(json.dumps(tsum, indent=2, default=float))

    a = summary['all']
    print(json.dumps(timing, indent=1))
    for k in ['n', 'baseline_dose_mg', 'optimal_dose_mg', 'dose_change_pct', 'pred_aki_change_pp',
              'pred_cure_change_pp', 'true_aki_change_pp', 'true_cure_change_pp',
              'pct_improved_true_objective', 'pct_worse_true_objective', 'corr_pred_true_aki_change',
              'corr_pred_true_cure_change', 'oracle_dose_mg', 'corr_ml_oracle_dose',
              'median_abs_ml_minus_oracle_pct']:
        print(f'  {k:<32} {a[k]}')


if __name__ == '__main__':
    main()
