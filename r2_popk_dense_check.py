#!/usr/bin/env python3
"""Revision 2: is the poor recovery of distribution parameters a sampling-design
(identifiability) problem? A dense-sampling simulation-estimation check.

The correctly specified fit (r2_refit_popk_power.py) recovers the clearance
submodel but not the distribution parameters (Vc, Q, Vp and the weight effects
on Vc) from the study's peak-and-trough samples. To test whether this reflects
the sampling design rather than the estimation model, 400 patients are drawn at
random (seed 7); for each, concentrations after the first dose are re-simulated
from the patient's TRUE parameters (r2_simulator_truth.py) at eight times
(0.5, 1, 1.5, 2, 4, 8, 12 h and 0.5 h before the next dose), with the
simulator's residual-error model (15% exponential + 0.5 mg/L additive, floor
0.1 mg/L), and the same power model with the same priors is fitted.

Outputs: results/revision2/popk_dense_summary.csv, models/r2_popk_dense_trace.nc
Run with the PyMC venv.
"""

from __future__ import annotations

import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

import r2_refit_popk_power as R

N_PATIENTS = 400
SAMPLE_TIMES = [0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0]   # plus (interval - 0.5)


def conc(t, dose, CL, Vc, Q, Vp):
    k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
    s = k10 + k12 + k21
    sq = np.sqrt(s * s - 4 * k10 * k21)
    a, b = 0.5 * (s + sq), 0.5 * (s - sq)
    A, B = (a - k21) / (a - b) / Vc, (b - k21) / (b - a) / Vc
    ci = dose * (A * (1 - np.exp(-a * t)) / a + B * (1 - np.exp(-b * t)) / b)
    cp = dose * (A * (1 - np.exp(-a)) / a * np.exp(-a * (t - 1)) + B * (1 - np.exp(-b)) / b * np.exp(-b * (t - 1)))
    return np.where(t <= 1, ci, cp)


def make_data():
    rng = np.random.default_rng(7)
    full = R.prepare_data()
    pat = full['pat']
    pat = pat.iloc[np.sort(rng.choice(len(pat), N_PATIENTS, replace=False))].reset_index(drop=True)
    pat['pidx'] = np.arange(len(pat))
    truth = pd.read_csv(R.OUT_DIR / 'true_individual_pk.csv').set_index('patient_id')
    dosing = pd.read_csv(R.REPO_ROOT / 'data' / 'dosing.csv')
    first = dosing.groupby('patient_id').agg(dose=('dose', 'first'))
    interval = dosing.groupby('patient_id')['time'].apply(lambda s: float(np.diff(s.to_numpy()).min()))
    rows = []
    for i, pid in enumerate(pat['patient_id']):
        tp = truth.loc[pid]
        times = np.array(SAMPLE_TIMES + [interval[pid] - 0.5])
        c = conc(times, first.loc[pid, 'dose'], tp['CL_individual'], tp['Vc_individual'],
                 tp['Q_individual'], tp['Vp_individual'])
        y = np.maximum(c * np.exp(rng.normal(0, 0.15, len(c))) + rng.normal(0, 0.5, len(c)), 0.1)
        for t, v in zip(times, y):
            rows.append((i, t, v, first.loc[pid, 'dose']))
    obs = pd.DataFrame(rows, columns=['pidx', 'TIME', 'DV', 'AMT'])
    n = len(obs)
    return {'pat': pat, 'obs': obs, 'obs_time': obs['TIME'].to_numpy(), 'dv': obs['DV'].to_numpy(),
            'pidx': obs['pidx'].to_numpy(), 'd_times': np.zeros((n, 1)),
            'd_amts': obs['AMT'].to_numpy()[:, None], 'd_mask': np.ones((n, 1))}


def summarise(idata):
    import json
    idata, report = R.drop_failed_chains(idata)
    (R.OUT_DIR / 'popk_dense_chains.json').write_text(json.dumps(report, indent=1))
    print('Chain screening:', report)
    summ = az.summary(idata, var_names=R.SUMMARY_VARS, hdi_prob=0.94)
    summ.insert(0, 'true_value', [R.TRUE_VALUES[v] for v in summ.index])
    summ['true_in_hdi'] = (summ['hdi_3%'] <= summ['true_value']) & (summ['true_value'] <= summ['hdi_97%'])
    summ.to_csv(R.OUT_DIR / 'popk_dense_summary.csv')
    print(summ[['true_value', 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'r_hat', 'true_in_hdi']].to_string())


def main():
    import sys
    if '--resummarise' in sys.argv:
        summarise(az.from_netcdf(R.MODELS_DIR / 'r2_popk_dense_trace.nc'))
        return
    d = make_data()
    print(f'Dense design: {N_PATIENTS} patients, {len(d["dv"])} observations')
    model = R.build_model(d)
    n = len(d['pat'])
    init = {'ln_theta_CL': np.log(5.5), 'ln_theta_Vc': np.log(16.0), 'ln_theta_Q': np.log(12.0),
            'ln_theta_Vp': np.log(10.0), 'd_CL_AMK': np.log(4.2 / 5.5), 'd_Vc_AMK': np.log(1.5),
            'd_Q_AMK': np.log(1.1 / 12.0), 'd_Vp_AMK': 0.0,
            **{f'omega_{k}': 0.3 for k in ['CL', 'Vc', 'Q', 'Vp']},
            **{f'eta_{k}': np.zeros(n) for k in ['CL', 'Vc', 'Q', 'Vp']},
            **{b: 0.0 for b in ['b_CL_CRCL', 'b_CL_WT', 'b_CL_AGE', 'b_CL_SHOCK', 'b_Vc_WT_GEN', 'b_Vc_WT_AMK']},
            'sigma_prop': 0.2, 'sigma_add': 0.5}
    t0 = time.time()
    with model:
        idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=11, initvals=init,
                          target_accept=0.9, nuts_sampler='nutpie', return_inferencedata=True,
                          progressbar=False)
    print(f'Sampling time: {(time.time() - t0) / 60:.1f} min')
    idata.to_netcdf(R.MODELS_DIR / 'r2_popk_dense_trace.nc')
    summarise(idata)


if __name__ == '__main__':
    main()
