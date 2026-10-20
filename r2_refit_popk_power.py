#!/usr/bin/env python3
"""Revision 2: correctly specified PopPK refit (simulation-estimation check).

Reviewer 3 (AAPSJ-D-26-00274, second round) asked why the estimated covariate
effects differed from the values used to simulate the data, and asked that the
two be compared on the same reference values.

Root cause: `refit_popk.py` (the R1 production fit) does not fit the power model
written in the R1 manuscript. It uses z-score-standardised covariates in an
exponential model, CL = theta * exp(beta * (CrCL - mean)/SD), so its betas are
log-changes per SD of covariate, not allometric exponents, and theta is the
typical value at the cohort-mean covariates. Projected onto that scale, the
simulator's own model gives beta_CrCL = 0.333, beta_WT = 0.143 (estimates 0.322,
0.148); see `r2_popk_parameterization_check.py`.

This script fits the model the simulator actually used, on its own reference
values, so that every estimate can be compared directly with its generating
value:

    CL_i = thCL * exp(dCL*AMK) * (CrCL/100)^bCrCL * (WT/70)^bWT * (AGE/55)^bAGE
                * exp(bSHOCK*SHOCK) * exp(eta_CL)
    Vc_i = thVc * exp(dVc*AMK) * (WT/70)^(bVcG*(1-AMK) + bVcA*AMK) * exp(eta_Vc)
    Q_i  = thQ  * exp(dQ*AMK)  * exp(eta_Q)
    Vp_i = thVp * exp(dVp*AMK) * exp(eta_Vp)

theta values refer to gentamicin, 70 kg, CrCL 100 mL/min, age 55 y, no septic
shock. Drug offsets are estimated (not fixed), and all covariate effects have
weakly informative priors centred on zero, so recovery cannot be attributed to
the priors. All 1,500 patients are included (no >=3-observation filter).

Requires the pinned PyMC stack (numpy 1.26.4, pymc 5.16.2, pytensor 2.25.5,
nutpie). Outputs:
    models/r2_popk_power_trace.nc
    results/revision2/popk_power_summary.csv
    results/revision2/popk_power_individual.csv

Usage:
    python r2_refit_popk_power.py --smoke     # 100 patients, 300 draws
    python r2_refit_popk_power.py             # full run, 4 chains x 1000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('PYTENSOR_FLAGS', 'device=cpu,floatX=float64')

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

REPO_ROOT = Path(__file__).resolve().parent
POPK_PATH = REPO_ROOT / 'data' / 'processed' / 'popk_dataset.csv'
PATIENT_PATH = REPO_ROOT / 'data' / 'patient_data.csv'
MODELS_DIR = REPO_ROOT / 'models'
OUT_DIR = REPO_ROOT / 'results' / 'revision2'

# Data-generating values (generate_synthetic_data_python.py, lines 79-95, 311-337)
TRUE_VALUES = {
    'theta_CL': 5.5, 'theta_Vc': 16.0, 'theta_Q': 12.0, 'theta_Vp': 10.0,
    'd_CL_AMK': np.log(4.2 / 5.5), 'd_Vc_AMK': np.log(24.0 / 16.0),
    'd_Q_AMK': np.log(1.1 / 12.0), 'd_Vp_AMK': np.log(11.7 / 10.0),
    'b_CL_CRCL': 0.75, 'b_CL_WT': 0.75, 'b_CL_AGE': 0.0, 'b_CL_SHOCK': -0.15,
    'b_Vc_WT_GEN': 1.0, 'b_Vc_WT_AMK': 0.5,
    'omega_CL': 0.40, 'omega_Vc': 0.30, 'omega_Q': 0.40, 'omega_Vp': 0.35,
    'sigma_prop': 0.15, 'sigma_add': 0.5,
}
SUMMARY_VARS = list(TRUE_VALUES)


def prepare_data(n_subset=None, seed=0):
    df = pd.read_csv(POPK_PATH)
    pdat = pd.read_csv(PATIENT_PATH)[['patient_id', 'sepsis_type']]
    obs = df[df['EVID'] == 0].copy()
    dose = df[df['EVID'] == 1].copy()

    pat = (obs.groupby('patient_id').first()[['AGE', 'WT', 'CRCL', 'DRUG']]
              .reset_index()
              .merge(pdat, on='patient_id', how='left')
              .sort_values('patient_id').reset_index(drop=True))
    if n_subset is not None:
        rng = np.random.default_rng(seed)
        pat = pat.iloc[np.sort(rng.choice(len(pat), n_subset, replace=False))]
        pat = pat.reset_index(drop=True)
    pat['pidx'] = np.arange(len(pat))
    pat['SHOCK'] = (pat['sepsis_type'] == 'septic shock').astype(float)
    id_map = dict(zip(pat['patient_id'], pat['pidx']))

    obs = obs[obs['patient_id'].isin(id_map)].sort_values(['patient_id', 'TIME'])
    dose = dose[dose['patient_id'].isin(id_map)].sort_values(['patient_id', 'TIME'])
    obs = obs.reset_index(drop=True)
    obs['pidx'] = obs['patient_id'].map(id_map).astype(int)

    max_doses = int(dose.groupby('patient_id').size().max())
    n_obs = len(obs)
    d_times = np.zeros((n_obs, max_doses))
    d_amts = np.zeros((n_obs, max_doses))
    d_mask = np.zeros((n_obs, max_doses))
    pat_doses = {pid: (s['TIME'].to_numpy(), s['AMT'].to_numpy())
                 for pid, s in dose.groupby('patient_id')}
    for i, (pid, t_obs) in enumerate(zip(obs['patient_id'], obs['TIME'])):
        dt, da = pat_doses[pid]
        prior = dt <= t_obs
        k = int(prior.sum())
        d_times[i, :k] = dt[prior]
        d_amts[i, :k] = da[prior]
        d_mask[i, :k] = 1.0

    n_per = obs.groupby('pidx').size()
    print(f'  Patients: {len(pat)}  Observations: {n_obs}  '
          f'obs/patient min {n_per.min()} median {int(n_per.median())} max {n_per.max()}')
    return {'pat': pat, 'obs': obs, 'd_times': d_times, 'd_amts': d_amts,
            'd_mask': d_mask, 'obs_time': obs['TIME'].to_numpy(),
            'dv': obs['DV'].to_numpy(), 'pidx': obs['pidx'].to_numpy()}


def two_comp_infusion(t, dose, CL, Vc, Q, Vp, tau=1.0):
    """Central concentration at time t after the start of a 1-h infusion."""
    k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
    s = k10 + k12 + k21
    sq = pt.sqrt(pt.maximum(s * s - 4 * k10 * k21, 1e-12))
    alpha, beta = 0.5 * (s + sq), 0.5 * (s - sq)
    A = (dose / Vc) * (alpha - k21) / (alpha - beta)
    B = (dose / Vc) * (beta - k21) / (beta - alpha)
    c_inf = A / alpha * (1 - pt.exp(-alpha * t)) + B / beta * (1 - pt.exp(-beta * t))
    c_post = (A / alpha * (1 - pt.exp(-alpha * tau)) * pt.exp(-alpha * (t - tau))
              + B / beta * (1 - pt.exp(-beta * tau)) * pt.exp(-beta * (t - tau)))
    return pt.switch(pt.le(t, tau), c_inf, c_post)


def build_model(d):
    pat = d['pat']
    n = len(pat)
    amk = pat['DRUG'].to_numpy(float)
    l_crcl = np.log(pat['CRCL'].to_numpy() / 100.0)
    l_wt = np.log(pat['WT'].to_numpy() / 70.0)
    l_age = np.log(pat['AGE'].to_numpy() / 55.0)
    shock = pat['SHOCK'].to_numpy()

    with pm.Model() as model:
        # Structural typical values (gentamicin, reference covariates) and drug
        # offsets: literature-anchored priors as in the R1 model. With peak and
        # trough sampling only, the distribution parameters (and the small
        # amikacin Q) are weakly identified, and broad priors on them leave
        # NUTS stuck between modes (smoke test: R-hat > 1.8).
        ln_thCL = pm.Normal('ln_theta_CL', np.log(5.5), 0.3)
        ln_thVc = pm.Normal('ln_theta_Vc', np.log(16.0), 0.3)
        ln_thQ = pm.Normal('ln_theta_Q', np.log(12.0), 0.5)
        ln_thVp = pm.Normal('ln_theta_Vp', np.log(10.0), 0.5)
        for name, v in [('theta_CL', ln_thCL), ('theta_Vc', ln_thVc),
                        ('theta_Q', ln_thQ), ('theta_Vp', ln_thVp)]:
            pm.Deterministic(name, pt.exp(v))

        dCL = pm.Normal('d_CL_AMK', np.log(4.2 / 5.5), 0.3)
        dVc = pm.Normal('d_Vc_AMK', np.log(24.0 / 16.0), 0.3)
        dQ = pm.Normal('d_Q_AMK', np.log(1.1 / 12.0), 0.5)
        dVp = pm.Normal('d_Vp_AMK', 0.0, 0.3)

        # Covariate effects (the quantities Reviewer 3 asked about): weakly
        # informative and centred on ZERO, so recovery of the generating values
        # cannot be attributed to the priors.
        bCrCL = pm.Normal('b_CL_CRCL', 0.0, 1.0)
        bWT = pm.Normal('b_CL_WT', 0.0, 1.0)
        bAGE = pm.Normal('b_CL_AGE', 0.0, 1.0)
        bSHOCK = pm.Normal('b_CL_SHOCK', 0.0, 1.0)
        bVcG = pm.Normal('b_Vc_WT_GEN', 0.0, 1.0)
        bVcA = pm.Normal('b_Vc_WT_AMK', 0.0, 1.0)

        omega = {k: pm.HalfNormal(f'omega_{k}', s) for k, s in
                 [('CL', 0.5), ('Vc', 0.5), ('Q', 0.4), ('Vp', 0.35)]}
        eta = {k: pm.Normal(f'eta_{k}', 0.0, 1.0, shape=n) for k in ['CL', 'Vc', 'Q', 'Vp']}

        CL = pm.Deterministic('CL', pt.exp(
            ln_thCL + dCL * amk + bCrCL * l_crcl + bWT * l_wt + bAGE * l_age
            + bSHOCK * shock + omega['CL'] * eta['CL']))
        Vc = pm.Deterministic('Vc', pt.exp(
            ln_thVc + dVc * amk + (bVcG * (1 - amk) + bVcA * amk) * l_wt
            + omega['Vc'] * eta['Vc']))
        Q = pm.Deterministic('Q', pt.exp(ln_thQ + dQ * amk + omega['Q'] * eta['Q']))
        Vp = pm.Deterministic('Vp', pt.exp(ln_thVp + dVp * amk + omega['Vp'] * eta['Vp']))

        sigma_prop = pm.HalfNormal('sigma_prop', 0.3)
        sigma_add = pm.HalfNormal('sigma_add', 1.0)

        p = d['pidx']
        t_since = pt.maximum(d['obs_time'][:, None] - d['d_times'], 0.0)
        c = two_comp_infusion(t_since, d['d_amts'], CL[p][:, None], Vc[p][:, None],
                              Q[p][:, None], Vp[p][:, None])
        c_pred = pt.maximum(pt.sum(c * d['d_mask'], axis=1), 1e-3)
        sd = pt.sqrt((sigma_prop * c_pred) ** 2 + sigma_add ** 2)
        pm.Normal('y_obs', mu=c_pred, sigma=sd, observed=d['dv'])
    return model


MAX_DIVERGENT_FRACTION = 0.5
MAX_LOGP_DEFICIT = 100.0


def drop_failed_chains(idata):
    """Exclude chains that failed to explore the posterior: more than half of the
    post-warm-up transitions divergent, or a mean log-density more than 100
    units below the best chain (a chain trapped in a region of negligible
    posterior mass; healthy chains differ by < 20 units in these fits).
    Returns the filtered InferenceData and a report."""
    st = idata.sample_stats
    div = st['diverging'].mean('draw').values
    lp = st['logp'].mean('draw').values
    chains = [int(c) for c in idata.posterior.chain.values]
    keep = [c for c, f, l in zip(chains, div, lp)
            if f <= MAX_DIVERGENT_FRACTION and l >= lp.max() - MAX_LOGP_DEFICIT]
    report = {'divergent_fraction_by_chain': [float(f) for f in div],
              'mean_logp_by_chain': [float(l) for l in lp],
              'chains_kept': keep, 'chains_excluded': [c for c in chains if c not in keep]}
    return idata.sel(chain=keep), report


def summarise(idata, d, out_dir, tag):
    import json
    idata, report = drop_failed_chains(idata)
    (out_dir / f'popk_power_chains{tag}.json').write_text(json.dumps(report, indent=1))
    print('Chain screening:', report)
    summ = az.summary(idata, var_names=SUMMARY_VARS, hdi_prob=0.94)
    summ.insert(0, 'true_value', [TRUE_VALUES[v] for v in summ.index])
    summ['true_in_hdi'] = ((summ['hdi_3%'] <= summ['true_value'])
                           & (summ['true_value'] <= summ['hdi_97%']))
    summ.to_csv(out_dir / f'popk_power_summary{tag}.csv')
    print(summ[['true_value', 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk',
                'r_hat', 'true_in_hdi']].to_string())

    post = idata.posterior
    ind = pd.DataFrame({'patient_id': d['pat']['patient_id']})
    for k in ['CL', 'Vc', 'Q', 'Vp']:
        ind[f'{k}_post_mean'] = post[k].mean(('chain', 'draw')).values
        ind[f'{k}_post_sd'] = post[k].std(('chain', 'draw')).values
    ind.to_csv(out_dir / f'popk_power_individual{tag}.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--resummarise', action='store_true',
                    help='re-summarise the saved trace without sampling')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        d = prepare_data(n_subset=100)
        draws, tune, chains, tag = 300, 300, 2, '_smoke'
    else:
        d = prepare_data()
        draws, tune, chains, tag = args.draws, args.tune, args.chains, ''

    if args.resummarise:
        summarise(az.from_netcdf(MODELS_DIR / f'r2_popk_power_trace{tag}.nc'), d, OUT_DIR, tag)
        return

    model = build_model(d)
    t0 = time.time()
    # Start at the prior centres (covariate effects at zero, not the truth)
    n = len(d['pat'])
    init = {'ln_theta_CL': np.log(5.5), 'ln_theta_Vc': np.log(16.0),
            'ln_theta_Q': np.log(12.0), 'ln_theta_Vp': np.log(10.0),
            'd_CL_AMK': np.log(4.2 / 5.5), 'd_Vc_AMK': np.log(1.5),
            'd_Q_AMK': np.log(1.1 / 12.0), 'd_Vp_AMK': 0.0,
            **{f'omega_{k}': 0.3 for k in ['CL', 'Vc', 'Q', 'Vp']},
            **{f'eta_{k}': np.zeros(n) for k in ['CL', 'Vc', 'Q', 'Vp']},
            **{b: 0.0 for b in ['b_CL_CRCL', 'b_CL_WT', 'b_CL_AGE', 'b_CL_SHOCK',
                                'b_Vc_WT_GEN', 'b_Vc_WT_AMK']},
            'sigma_prop': 0.2, 'sigma_add': 0.5}
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=args.seed,
                          initvals=init, target_accept=0.9, nuts_sampler='nutpie',
                          return_inferencedata=True, progressbar=False)
    print(f'Sampling time: {(time.time() - t0) / 60:.1f} min')
    idata.to_netcdf(MODELS_DIR / f'r2_popk_power_trace{tag}.nc')
    summarise(idata, d, OUT_DIR, tag)


if __name__ == '__main__':
    main()
