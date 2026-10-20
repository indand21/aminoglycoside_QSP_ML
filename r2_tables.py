#!/usr/bin/env python3
"""Revision 2: recompute Tables 2 and 4 from the canonical dataset.

Every value is computed here from data/ and results/phase3_pkpd/pkpd_indices.csv.
In R1, Table 2's dosing rows and Table 4 panels C and D were carried from an
earlier simulation (q8h/q12h/q24h intervals that do not occur in this dataset);
they are replaced here.

MIC categories. The simulator does not use breakpoints: it draws each MIC from a
two-component log-normal mixture (susceptible and resistant modes) with
pathogen-specific resistance probabilities, rounded to doubling dilutions.
Categories are applied afterwards for description only, using organism-specific
CLSI M100 (33rd ed., 2023) breakpoints. Gentamicin has no CLSI breakpoint for
P. aeruginosa in that edition, and the amikacin P. aeruginosa breakpoint applies
to urinary isolates only; other isolates are reported as "no breakpoint".

Output: results/revision2/table2_values.json, table4_values.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / 'results' / 'revision2'

# CLSI M100 33rd ed. (2023) systemic breakpoints, mg/L: (S <=, R >=); I in between.
CLSI_M100_ED33 = {
    'gentamicin': {'Enterobacterales': (2, 8), 'Acinetobacter': (4, 16), 'Pseudomonas': None},
    'amikacin': {'Enterobacterales': (4, 16), 'Acinetobacter': (16, 64), 'Pseudomonas': (16, 64)},
}
# M100-Ed33 limits the amikacin P. aeruginosa breakpoint to urinary isolates
URINE_ONLY = {('amikacin', 'Pseudomonas')}
ORGANISM_GROUP = {'klebsiella': 'Enterobacterales', 'e_coli': 'Enterobacterales',
                  'other': 'Enterobacterales', 'pseudomonas': 'Pseudomonas',
                  'acinetobacter': 'Acinetobacter'}
DILUTIONS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
TROUGH = {'amikacin': 2.5, 'gentamicin': 2.0}


def n_pct(mask, denom=None):
    mask = np.asarray(mask, bool)
    d = len(mask) if denom is None else denom
    return [int(mask.sum()), round(100 * mask.sum() / d, 1)]


def mean_sd(x, nd=1):
    x = pd.Series(x).dropna()
    return [round(float(x.mean()), nd), round(float(x.std()), nd)]


def load():
    p = pd.read_csv(REPO_ROOT / 'data' / 'patient_data.csv')
    o = pd.read_csv(REPO_ROOT / 'data' / 'outcomes.csv')
    d = pd.read_csv(REPO_ROOT / 'data' / 'dosing.csv')
    k = pd.read_csv(REPO_ROOT / 'results' / 'phase3_pkpd' / 'pkpd_indices.csv')
    dz = d.groupby('patient_id').agg(first_dose=('dose', 'first'), n_doses=('dose', 'size'))
    dz['interval_h'] = d.groupby('patient_id')['time'].apply(lambda t: float(np.diff(t).min()))
    df = (p.merge(o[['patient_id', 'nephrotoxicity', 'clinical_cure', 'icu_mortality',
                     'day_28_mortality']], on='patient_id')
           .merge(dz, left_on='patient_id', right_index=True)
           .merge(k[['patient_id', 'Cmax', 'Cmin', 'AUC24', 'MIC', 'Cmax_MIC', 'AUC_MIC']],
                  on='patient_id'))
    df['mic'] = np.where(df['drug'] == 'amikacin', df['mic_amikacin'], df['mic_gentamicin'])
    assert np.allclose(df['mic'], df['MIC']), 'pkpd MIC differs from drug-specific MIC'
    df['organism_group'] = df['pathogen'].map(ORGANISM_GROUP)
    df['trough_limit'] = df['drug'].map(TROUGH)
    df['renal'] = pd.cut(df['baseline_crcl'], [-np.inf, 50, 130, np.inf], right=False,
                         labels=['Impaired (<50)', 'Normal (50-130)', 'Augmented (>130)'])
    # Cut points: <50, 50-<=130 normal per R1 wording "Normal (50-130)", ">130" augmented
    df['renal'] = np.select([df['baseline_crcl'] < 50, df['baseline_crcl'] > 130],
                            ['Impaired (<50)', 'Augmented (>130)'], 'Normal (50-130)')
    return df


def clsi_category(drug, group, mic, site=None):
    bp = CLSI_M100_ED33[drug][group]
    if (drug, group) in URINE_ONLY and site != 'uti':
        bp = None
    if bp is None:
        return 'No CLSI breakpoint'
    s, r = bp
    return 'Susceptible' if mic <= s else ('Resistant' if mic >= r else 'Intermediate')


def table2(df):
    t = {}
    t['N'] = len(df)
    t['age'] = mean_sd(df['age']); t['age_range'] = [float(df['age'].min()), float(df['age'].max())]
    t['male'] = n_pct(df['sex'] == 'M')
    t['weight'] = mean_sd(df['weight']); t['height'] = mean_sd(df['height']); t['bmi'] = mean_sd(df['bmi'])
    t['amikacin'] = n_pct(df['drug'] == 'amikacin'); t['gentamicin'] = n_pct(df['drug'] == 'gentamicin')
    t['apache'] = mean_sd(df['apache_ii']); t['apache_range'] = [int(df['apache_ii'].min()), int(df['apache_ii'].max())]
    t['sofa'] = mean_sd(df['sofa_score']); t['sofa_range'] = [int(df['sofa_score'].min()), int(df['sofa_score'].max())]
    t['sepsis_type'] = {k: n_pct(df['sepsis_type'] == k) for k in ['sepsis', 'severe sepsis', 'septic shock']}
    t['mech_vent'] = n_pct(df['mechanical_ventilation']); t['vasopressor'] = n_pct(df['vasopressor_use'])
    t['infection_site'] = {k: n_pct(df['infection_site'] == k) for k in df['infection_site'].value_counts().index}
    t['pathogen'] = {k: n_pct(df['pathogen'] == k) for k in df['pathogen'].value_counts().index}
    mic = {}
    for drug in ['gentamicin', 'amikacin']:
        sub = df[df['drug'] == drug]
        col = f'mic_{drug}'
        mic[drug] = {
            'n': len(sub),
            'distribution': {str(v): int((sub[col] == v).sum()) for v in DILUTIONS},
            'geometric_mean': round(float(np.exp(np.log(sub[col]).mean())), 2),
            'MIC50': float(np.percentile(sub[col], 50, method='inverted_cdf')),
            'MIC90': float(np.percentile(sub[col], 90, method='inverted_cdf')),
        }
        cats = [clsi_category(drug, g, m, st) for g, m, st in zip(sub['organism_group'], sub[col], sub['infection_site'])]
        mic[drug]['clsi_m100_ed33'] = {c: n_pct(np.array(cats) == c, len(sub)) for c in
                                       ['Susceptible', 'Intermediate', 'Resistant', 'No CLSI breakpoint']}
        mic[drug]['clsi_by_organism'] = {}
        for g in ['Enterobacterales', 'Pseudomonas', 'Acinetobacter']:
            s2 = sub[sub['organism_group'] == g]
            c2 = [clsi_category(drug, g, m, st) for m, st in zip(s2[col], s2['infection_site'])]
            mic[drug]['clsi_by_organism'][g] = {'n': len(s2), **{c: int((np.array(c2) == c).sum()) for c in
                                                ['Susceptible', 'Intermediate', 'Resistant', 'No CLSI breakpoint']}}
        # R1 (simulation-defined) categories, kept for the response letter only
        s_cut = 2 if drug == 'gentamicin' else 4
        mic[drug]['r1_categories'] = {'S': n_pct(sub[col] <= s_cut, len(sub)),
                                      'I': n_pct((sub[col] > s_cut) & (sub[col] < 16), len(sub)),
                                      'R': n_pct(sub[col] >= 16, len(sub))}
    t['mic'] = mic
    t['scr'] = mean_sd(df['baseline_scr'], 2); t['crcl'] = mean_sd(df['baseline_crcl'])
    t['renal'] = {k: n_pct(df['renal'] == k) for k in ['Augmented (>130)', 'Normal (50-130)', 'Impaired (<50)']}
    t['egfr'] = mean_sd(df['baseline_egfr']); t['albumin'] = mean_sd(df['baseline_albumin'])
    t['diabetes'] = n_pct(df['diabetes'])
    t['ckd_stage'] = {k: n_pct(df['ckd_stage'].astype(str) == k) for k in sorted(df['ckd_stage'].astype(str).unique())}
    t['ckd_stages_1_4'] = n_pct(df['ckd_stage'].astype(str) != '0')
    dose = {}
    for drug in ['amikacin', 'gentamicin']:
        sub = df[df['drug'] == drug]
        dose[drug] = {'dose_mg': mean_sd(sub['first_dose'], 0),
                      'dose_mgkg': mean_sd(sub['first_dose'] / sub['weight'], 1)}
    t['dose'] = dose
    t['interval'] = {f'q{int(h)}h': n_pct(df['interval_h'] == h) for h in sorted(df['interval_h'].unique())}
    t['interval_by_drug'] = {drug: {f'q{int(h)}h': int(((df['drug'] == drug) & (df['interval_h'] == h)).sum())
                                    for h in sorted(df['interval_h'].unique())} for drug in ['amikacin', 'gentamicin']}
    t['n_doses'] = mean_sd(df['n_doses'])
    t['treatment_days'] = mean_sd(df['n_doses'] * df['interval_h'] / 24)
    t['nephrotoxicity'] = n_pct(df['nephrotoxicity']); t['cure'] = n_pct(df['clinical_cure'])
    t['icu_mortality'] = n_pct(df['icu_mortality']); t['day28_mortality'] = n_pct(df['day_28_mortality'])
    return t


def attain(sub):
    tl = sub['trough_limit']
    return {
        'n': len(sub),
        'cmax_mic_8': n_pct(sub['Cmax_MIC'] >= 8), 'cmax_mic_10': n_pct(sub['Cmax_MIC'] >= 10),
        'auc_mic_80': n_pct(sub['AUC_MIC'] >= 80), 'auc_mic_100': n_pct(sub['AUC_MIC'] >= 100),
        'cmin_drug': n_pct(sub['Cmin'] < tl), 'cmin_2': n_pct(sub['Cmin'] < 2), 'cmin_1': n_pct(sub['Cmin'] < 1),
        'combined': n_pct((sub['Cmax_MIC'] >= 8) & (sub['Cmin'] < tl)),
        'combined_optimal': n_pct((sub['Cmax_MIC'] >= 10) & (sub['Cmin'] < 1)),
    }


def table4(df):
    t = {'A': {'combined': attain(df), 'amikacin': attain(df[df['drug'] == 'amikacin']),
               'gentamicin': attain(df[df['drug'] == 'gentamicin'])}}
    t['B'] = {r: {g: attain(df[(df['renal'] == r) & ((df['drug'] == g) if g != 'combined' else True)])
                  for g in ['combined', 'amikacin', 'gentamicin']}
              for r in ['Augmented (>130)', 'Normal (50-130)', 'Impaired (<50)']}
    t['C'] = {f'q{int(h)}h': {g: attain(df[(df['interval_h'] == h) & ((df['drug'] == g) if g != 'combined' else True)])
                              for g in ['combined', 'amikacin', 'gentamicin']}
              for h in sorted(df['interval_h'].unique())}
    # Panel D: CFR = sum_i PTA(MIC_i) * f(MIC_i), per drug, against that drug's simulated MIC distribution
    cfr = {}
    for drug in ['amikacin', 'gentamicin']:
        sub = df[df['drug'] == drug]
        f = sub['mic'].value_counts(normalize=True).reindex(DILUTIONS, fill_value=0.0)
        rows = []
        for m in DILUTIONS:
            rows.append({'mic': m, 'fraction': round(float(f[m]), 4),
                         'pta_cmax8': round(float((sub['Cmax'] / m >= 8).mean()), 4),
                         'pta_auc80': round(float((sub['AUC24'] / m >= 80).mean()), 4)})
        cfr[drug] = {'rows': rows,
                     'cfr_cmax8': round(float(sum(r['fraction'] * r['pta_cmax8'] for r in rows)), 4),
                     'cfr_auc80': round(float(sum(r['fraction'] * r['pta_auc80'] for r in rows)), 4),
                     'n': len(sub)}
    w = {d: cfr[d]['n'] / len(df) for d in cfr}
    cfr['combined'] = {k: round(sum(w[d] * cfr[d][k] for d in w), 4) for k in ['cfr_cmax8', 'cfr_auc80']}
    t['D'] = cfr
    t['E'] = {g: {c: mean_sd(s[c], 2 if c == 'Cmin' else 1) for c in ['Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC']}
              for g, s in [('combined', df), ('amikacin', df[df['drug'] == 'amikacin']),
                           ('gentamicin', df[df['drug'] == 'gentamicin'])]}
    t['notes'] = {'n_auc24_zero': int((df['AUC24'] == 0).sum()),
                  'auc24_zero_by_interval': {f'q{int(h)}h': int(((df['AUC24'] == 0) & (df['interval_h'] == h)).sum())
                                             for h in sorted(df['interval_h'].unique())}}
    return t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    t2, t4 = table2(df), table4(df)
    (OUT_DIR / 'table2_values.json').write_text(json.dumps(t2, indent=1))
    (OUT_DIR / 'table4_values.json').write_text(json.dumps(t4, indent=1))
    print(json.dumps(t2, indent=None)[:6000])
    print()
    print(json.dumps(t4, indent=None)[:8000])


if __name__ == '__main__':
    main()
