#!/usr/bin/env python3
"""Revision 2: put the R1 PopPK estimates on the simulator's reference scale.

The R1 production fit (`refit_popk.py`, trace models/phase2_popk_trace_likelihood.nc)
used z-score-standardised covariates in an exponential model:

    CL = theta_CL * exp(b_CrCL*z_CrCL + b_WT*z_WT + b_AGE*z_AGE + dCL*AMK + eta)
    Vc = theta_Vc * exp(b_VcWT*z_WT + dVc*AMK + eta)

so theta is the typical value at the cohort-mean covariates and each b is the
log-change per SD of covariate. The simulator used power models referenced to
CrCL 100 mL/min and 70 kg. This script compares the two on common ground, three
ways, using every posterior draw:

  1. "projection": the simulator's typical-value model regressed onto the R1
     parameterisation over the same 1,450 patients (what a perfect fit of the
     R1 model would return);
  2. "implied exponent": b * mean(x)/SD(x), the local power exponent at the
     cohort mean, and typical CL/Vc re-referenced to CrCL 100, 70 kg, age 55;
  3. individual recovery: posterior-mean individual CL, Vc, Q, Vp against the
     true simulated values (r2_simulator_truth.py).

Run with the pinned PyMC venv (needs arviz/xarray). Output:
    results/revision2/popk_r1_reference_comparison.csv
    results/revision2/popk_individual_recovery.csv
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / 'results' / 'revision2'
TRACE = REPO_ROOT / 'models' / 'phase2_popk_trace_likelihood.nc'


def hdi(x, prob=0.94):
    lo, hi = az.hdi(np.asarray(x), hdi_prob=prob)
    return float(lo), float(hi)


def r1_covariates():
    """Rebuild the exact patient set and standardisation used by refit_popk.py."""
    df = pd.read_csv(REPO_ROOT / 'data' / 'processed' / 'popk_dataset.csv')
    obs = df[df['EVID'] == 0]
    counts = obs.groupby('ID').size()
    keep = counts[counts >= 3].index
    pat = (obs[obs['ID'].isin(keep)].groupby('ID').first()
           [['patient_id', 'AGE', 'WT', 'CRCL', 'DRUG']].sort_index().reset_index())
    pdat = pd.read_csv(REPO_ROOT / 'data' / 'patient_data.csv')[['patient_id', 'sepsis_type']]
    return pat.merge(pdat, on='patient_id', how='left')


def projection(pat):
    """Least-squares projection of the generating model onto the R1 model."""
    z = lambda s: (s - s.mean()) / s.std()
    amk = pat['DRUG'].to_numpy(float)
    shock = (pat['sepsis_type'] == 'septic shock').to_numpy(float)
    ln_cl = (np.log(5.5) + np.log(4.2 / 5.5) * amk + 0.75 * np.log(pat['CRCL'] / 100)
             + 0.75 * np.log(pat['WT'] / 70) - 0.15 * shock)
    ln_vc = (np.log(16) + np.log(1.5) * amk
             + np.where(amk == 1, 0.5, 1.0) * np.log(pat['WT'] / 70))
    X = np.column_stack([np.ones(len(pat)), z(pat['CRCL']), z(pat['WT']), z(pat['AGE'])])
    b_cl = np.linalg.lstsq(X, ln_cl - np.log(4.2 / 5.5) * amk, rcond=None)[0]
    Xv = np.column_stack([np.ones(len(pat)), z(pat['WT'])])
    b_vc = np.linalg.lstsq(Xv, ln_vc - np.log(1.5) * amk, rcond=None)[0]
    return {'theta_CL': np.exp(b_cl[0]), 'beta_CL_CRCL': b_cl[1], 'beta_CL_WT': b_cl[2],
            'beta_CL_AGE': b_cl[3], 'theta_Vc': np.exp(b_vc[0]), 'beta_Vc_WT': b_vc[1]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pat = r1_covariates()
    m = {k: pat[k].mean() for k in ['CRCL', 'WT', 'AGE']}
    s = {k: pat[k].std() for k in ['CRCL', 'WT', 'AGE']}
    p_shock = (pat['sepsis_type'] == 'septic shock').mean()
    print(f'R1 fit: {len(pat)} patients; means CrCL {m["CRCL"]:.1f}, WT {m["WT"]:.1f}, '
          f'AGE {m["AGE"]:.1f}; SDs {s["CRCL"]:.1f}, {s["WT"]:.2f}, {s["AGE"]:.2f}')

    post = az.from_netcdf(TRACE).posterior
    draw = lambda v: post[v].values.reshape(-1)
    th_cl, b_cr, b_wt, b_age = (draw('theta_CL'), draw('beta_CL_CRCL'),
                                draw('beta_CL_WT'), draw('beta_CL_AGE'))
    th_vc, b_vwt = draw('theta_Vc'), draw('beta_Vc_WT')
    proj = projection(pat)

    rows = []

    def add(name, r1_scale_draws, truth_same_scale, note):
        lo, hi = hdi(r1_scale_draws)
        rows.append({'quantity': name, 'estimate_mean': float(np.mean(r1_scale_draws)),
                     'hdi_3': lo, 'hdi_97': hi, 'truth_same_scale': float(truth_same_scale),
                     'truth_in_hdi': lo <= truth_same_scale <= hi, 'note': note})

    # 1. R1 scale vs projection of the true model onto the R1 scale
    add('theta_CL (R1 scale, cohort-mean covariates)', th_cl, proj['theta_CL'], 'projection')
    add('beta_CL_CRCL per SD (R1 scale)', b_cr, proj['beta_CL_CRCL'], 'projection')
    add('beta_CL_WT per SD (R1 scale)', b_wt, proj['beta_CL_WT'], 'projection')
    add('beta_CL_AGE per SD (R1 scale)', b_age, proj['beta_CL_AGE'], 'projection')
    add('theta_Vc (R1 scale, cohort-mean weight)', th_vc, proj['theta_Vc'], 'projection')
    add('beta_Vc_WT per SD (R1 scale)', b_vwt, proj['beta_Vc_WT'], 'projection')

    # 2. Re-expressed on the simulator's power scale and reference values
    add('CrCL->CL exponent (implied at cohort mean)', b_cr * m['CRCL'] / s['CRCL'], 0.75,
        'generating exponent')
    add('WT->CL exponent (implied at cohort mean)', b_wt * m['WT'] / s['WT'], 0.75,
        'generating exponent')
    vc_true_avg = 0.5 * pat['DRUG'].mean() + 1.0 * (1 - pat['DRUG'].mean())
    add('WT->Vc exponent (implied, drug-pooled)', b_vwt * m['WT'] / s['WT'], vc_true_avg,
        f'generating exponents 1.0 (GEN) / 0.5 (AMK), weighted by drug mix')
    cl_ref = th_cl * np.exp(b_cr * (100 - m['CRCL']) / s['CRCL']
                            + b_wt * (70 - m['WT']) / s['WT']
                            + b_age * (55 - m['AGE']) / s['AGE'])
    add('Typical GEN CL at CrCL 100, 70 kg, 55 y (L/h)', cl_ref,
        5.5 * np.exp(-0.15 * p_shock),
        f'5.5 L/h x exp(-0.15 x {p_shock:.3f} septic-shock fraction); R1 model has no shock term')
    cl_ref_proj = proj['theta_CL'] * np.exp(
        proj['beta_CL_CRCL'] * (100 - m['CRCL']) / s['CRCL']
        + proj['beta_CL_WT'] * (70 - m['WT']) / s['WT']
        + proj['beta_CL_AGE'] * (55 - m['AGE']) / s['AGE'])
    add('Typical GEN CL at CrCL 100, 70 kg, 55 y (L/h), vs projection', cl_ref, cl_ref_proj,
        'best value attainable by the R1 (log-linear) model at the same reference')
    vc_ref = th_vc * np.exp(b_vwt * (70 - m['WT']) / s['WT'])
    add('Typical GEN Vc at 70 kg (L)', vc_ref, 16.0, 'generating value')

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / 'popk_r1_reference_comparison.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_colwidth', 60):
        print(out.round(3).to_string(index=False))

    # 3. Individual-level recovery (R1 individual posterior means, as used downstream)
    truth = pd.read_csv(OUT_DIR / 'true_individual_pk.csv')
    ind = pd.read_csv(REPO_ROOT / 'results' / 'phase3_pkpd' / 'pkpd_indices.csv')
    ind = ind[['patient_id', 'CL_bayes', 'Vc_bayes', 'Q_bayes', 'Vp_bayes']].merge(truth, on='patient_id')
    ind = ind.dropna(subset=['CL_bayes'])
    rec = []
    for k in ['CL', 'Vc', 'Q', 'Vp']:
        est, tru = ind[f'{k}_bayes'], ind[f'{k}_individual']
        pe = 100 * (est - tru) / tru
        rec.append({'fit': 'R1 production', 'parameter': k, 'n': len(ind),
                    'r_log': np.corrcoef(np.log(est), np.log(tru))[0, 1],
                    'median_pct_error': pe.median(),
                    'median_abs_pct_error': pe.abs().median()})
    r2_ind = OUT_DIR / 'popk_power_individual.csv'
    if r2_ind.exists():
        ind2 = pd.read_csv(r2_ind).merge(truth, on='patient_id')
        for k in ['CL', 'Vc', 'Q', 'Vp']:
            est, tru = ind2[f'{k}_post_mean'], ind2[f'{k}_individual']
            pe = 100 * (est - tru) / tru
            rec.append({'fit': 'R2 correctly specified', 'parameter': k, 'n': len(ind2),
                        'r_log': np.corrcoef(np.log(est), np.log(tru))[0, 1],
                        'median_pct_error': pe.median(),
                        'median_abs_pct_error': pe.abs().median()})
    recdf = pd.DataFrame(rec)
    recdf.to_csv(OUT_DIR / 'popk_individual_recovery.csv', index=False)
    print('\nIndividual recovery (R1 posterior means vs true simulated values):')
    print(recdf.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
