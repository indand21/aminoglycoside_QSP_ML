#!/usr/bin/env python3
"""Revision 2: Figure 2 (correctly specified PopPK fit) and Supplementary Figure S1
(production fit), computed from the posterior traces.

Figure 2 replaces the R1 file, which was an illustrative placeholder. Every panel
is computed here from models/r2_popk_power_trace.nc and the concentration data:
  A  observed concentrations and individual predictions (posterior-mean
     parameters) for four representative patients (median true CL in each
     drug x interval group shown);
  B  observed vs individual-predicted concentrations, R^2 on the log scale;
  C  individual weighted residuals (IWRES) vs time after dose;
  D  IWRES vs population prediction (eta = 0);
  E  covariate effects and drug offsets, posterior mean and 94% HDI, with the
     generating values marked (values identical to Table 3A);
  F  trace of typical gentamicin clearance by chain.
Supplementary Figure S1 shows the production fit (models/phase2_popk_trace_likelihood.nc):
  A  posterior densities of the covariate coefficients by chain;
  B  common-scale comparison with the generating model (Table 3B);
  C, D  individual CL and Vc (posterior mean) vs true values.

Run with the PyMC venv (needs arviz). Outputs: Revision2_Submission/Figures/
Figure2_PopPK_Diagnostics.{tiff,png}; Supplementary/figure_png/
FigureS1_PopPK_Production_Fit.png; results/revision2/figure2_values.json
"""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
R2 = REPO / 'results' / 'revision2'
FIG = REPO / 'Manuscript_AAPS_Journal' / 'Revision2_Submission' / 'Figures'
SUPP = REPO / 'Manuscript_AAPS_Journal' / 'Revision2_Submission' / 'Supplementary' / 'figure_png'
COL = {'amikacin': '#0072B2', 'gentamicin': '#E69F00'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})

COV_LABELS = [
    ('b_CL_CRCL', 'CrCL → CL (exponent)'), ('b_CL_WT', 'Weight → CL (exponent)'),
    ('b_CL_AGE', 'Age → CL (exponent)'), ('b_CL_SHOCK', 'Septic shock → CL (log)'),
    ('b_Vc_WT_GEN', 'Weight → Vc, gentamicin'), ('b_Vc_WT_AMK', 'Weight → Vc, amikacin'),
    ('d_CL_AMK', 'Amikacin offset, CL'), ('d_Vc_AMK', 'Amikacin offset, Vc'),
    ('d_Q_AMK', 'Amikacin offset, Q'), ('d_Vp_AMK', 'Amikacin offset, Vp'),
]


def letter(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom')


def conc_curve(t, dose_t, dose_amt, CL, Vc, Q, Vp):
    k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
    s = k10 + k12 + k21
    sq = np.sqrt(s * s - 4 * k10 * k21)
    a, b = 0.5 * (s + sq), 0.5 * (s - sq)
    A, B = (a - k21) / (a - b) / Vc, (b - k21) / (b - a) / Vc
    dt = t[:, None] - dose_t[None, :]
    on, inf = dt >= 0, dt <= 1
    ci = dose_amt * (A * (1 - np.exp(-a * dt)) / a + B * (1 - np.exp(-b * dt)) / b)
    cp = dose_amt * (A * (1 - np.exp(-a)) / a * np.exp(-a * (dt - 1))
                     + B * (1 - np.exp(-b)) / b * np.exp(-b * (dt - 1)))
    return np.where(on, np.where(inf, ci, cp), 0.0).sum(axis=1)


def hdi(x):
    lo, hi = az.hdi(np.asarray(x).ravel(), hdi_prob=0.94)
    return float(lo), float(hi)


def save(fig, path_png, tiff=None):
    fig.savefig(path_png, dpi=300, bbox_inches='tight', facecolor='white')
    if tiff:
        from PIL import Image
        fig.savefig(str(tiff).replace('.tiff', '_tmp.png'), dpi=300, bbox_inches='tight', facecolor='white')
        im = Image.open(str(tiff).replace('.tiff', '_tmp.png')).convert('RGB')
        im.save(tiff, compression='tiff_lzw', dpi=(300, 300))
        Path(str(tiff).replace('.tiff', '_tmp.png')).unlink()
    plt.close(fig)


def figure2(values):
    from r2_refit_popk_power import drop_failed_chains
    idata, _ = drop_failed_chains(az.from_netcdf(REPO / 'models' / 'r2_popk_power_trace.nc'))
    post = idata.posterior
    summ = pd.read_csv(R2 / 'popk_power_summary.csv', index_col=0)
    ind = pd.read_csv(R2 / 'popk_power_individual.csv')
    truth = pd.read_csv(R2 / 'true_individual_pk.csv')
    popk = pd.read_csv(REPO / 'data' / 'processed' / 'popk_dataset.csv')
    pdat = pd.read_csv(REPO / 'data' / 'patient_data.csv')[['patient_id', 'sepsis_type']]
    obs, dose = popk[popk['EVID'] == 0], popk[popk['EVID'] == 1]
    m = lambda v: float(post[v].mean())
    sp, sa = m('sigma_prop'), m('sigma_add')

    pat = (obs.groupby('patient_id').first()[['AGE', 'WT', 'CRCL', 'DRUG']].reset_index()
           .merge(pdat, on='patient_id').merge(ind, on='patient_id'))
    amk = pat['DRUG'].to_numpy(float)
    shock = (pat['sepsis_type'] == 'septic shock').to_numpy(float)
    lw = np.log(pat['WT'] / 70)
    pat['CL_pop'] = np.exp(np.log(m('theta_CL')) + m('d_CL_AMK') * amk + m('b_CL_CRCL') * np.log(pat['CRCL'] / 100)
                           + m('b_CL_WT') * lw + m('b_CL_AGE') * np.log(pat['AGE'] / 55) + m('b_CL_SHOCK') * shock)
    pat['Vc_pop'] = np.exp(np.log(m('theta_Vc')) + m('d_Vc_AMK') * amk
                           + (m('b_Vc_WT_GEN') * (1 - amk) + m('b_Vc_WT_AMK') * amk) * lw)
    pat['Q_pop'] = np.exp(np.log(m('theta_Q')) + m('d_Q_AMK') * amk)
    pat['Vp_pop'] = np.exp(np.log(m('theta_Vp')) + m('d_Vp_AMK') * amk)
    P = pat.set_index('patient_id')

    rows = []
    for pid, g in obs.groupby('patient_id'):
        dz = dose[dose['patient_id'] == pid]
        p = P.loc[pid]
        t = g['TIME'].to_numpy()
        ipred = conc_curve(t, dz['TIME'].to_numpy(), dz['AMT'].to_numpy(), p['CL_post_mean'],
                           p['Vc_post_mean'], p['Q_post_mean'], p['Vp_post_mean'])
        pred = conc_curve(t, dz['TIME'].to_numpy(), dz['AMT'].to_numpy(), p['CL_pop'], p['Vc_pop'],
                          p['Q_pop'], p['Vp_pop'])
        last = np.array([dz['TIME'].to_numpy()[dz['TIME'].to_numpy() <= ti].max() for ti in t])
        rows.append(pd.DataFrame({'patient_id': pid, 'time': t, 'tad': t - last, 'dv': g['DV'].to_numpy(),
                                  'ipred': ipred, 'pred': pred, 'drug': 'amikacin' if p['DRUG'] == 1 else 'gentamicin'}))
    d = pd.concat(rows, ignore_index=True)
    d['iwres'] = (d['dv'] - d['ipred']) / np.sqrt((sp * d['ipred']) ** 2 + sa ** 2)
    pos = (d['dv'] > 0) & (d['ipred'] > 0)
    r2 = float(np.corrcoef(np.log(d.loc[pos, 'dv']), np.log(d.loc[pos, 'ipred']))[0, 1] ** 2)
    values.update({'n_obs': int(len(d)), 'n_patients': int(d['patient_id'].nunique()),
                   'r2_log_obs_vs_ipred': r2, 'iwres_mean': float(d['iwres'].mean()),
                   'iwres_sd': float(d['iwres'].std()),
                   'pct_abs_iwres_lt2': float(100 * (d['iwres'].abs() < 2).mean())})

    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.45)

    # A: representative patients
    ax = fig.add_subplot(gs[0, 0]); letter(ax, 'A')
    dz_int = dose.groupby('patient_id')['TIME'].apply(lambda s: np.diff(s.to_numpy()).min())
    tr = truth.set_index('patient_id')
    chosen = []
    for drug_code, drug in [(0, 'gentamicin'), (1, 'amikacin')]:
        for h in [24, 48]:
            ids = [i for i in P.index if P.loc[i, 'DRUG'] == drug_code and dz_int.get(i) == h
                   and (obs['patient_id'] == i).sum() >= 5]
            if not ids:
                continue
            cl = tr.loc[ids, 'CL_individual']
            chosen.append((drug, h, (cl - cl.median()).abs().idxmin()))
    for drug, h, pid in chosen:
        dz = dose[dose['patient_id'] == pid]; p = P.loc[pid]
        tt = np.linspace(0, d.loc[d['patient_id'] == pid, 'time'].max() + 2, 600)
        cc = conc_curve(tt, dz['TIME'].to_numpy(), dz['AMT'].to_numpy(), p['CL_post_mean'],
                        p['Vc_post_mean'], p['Q_post_mean'], p['Vp_post_mean'])
        ls = '-' if h == 24 else '--'
        ax.plot(tt, cc, ls, color=COL[drug], lw=1, label=f'{drug.capitalize()}, q{h}h')
        g = d[d['patient_id'] == pid]
        ax.plot(g['time'], g['dv'], 'o', color=COL[drug], ms=3)
    ax.set_yscale('log'); ax.set_xlabel('Time (h)'); ax.set_ylabel('Concentration (mg/L)')
    ax.set_title('Representative patients', loc='left')
    ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)

    # B: observed vs IPRED
    ax = fig.add_subplot(gs[0, 1]); letter(ax, 'B')
    for drug in ['gentamicin', 'amikacin']:
        g = d[(d['drug'] == drug) & pos]
        ax.scatter(g['ipred'], g['dv'], s=3, alpha=0.25, color=COL[drug], label=drug.capitalize(), rasterized=True)
    lim = [0.05, d['dv'].max() * 1.3]
    ax.plot(lim, lim, 'k-', lw=0.8); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Individual prediction (mg/L)'); ax.set_ylabel('Observed (mg/L)')
    ax.set_title(f'Observed vs individual prediction\n(log-scale R² = {r2:.3f})', loc='left')
    ax.legend(frameon=False, markerscale=3)

    # C, D: IWRES
    for k, (xcol, xlab, L) in enumerate([('tad', 'Time after dose (h)', 'C'), ('pred', 'Population prediction (mg/L)', 'D')]):
        ax = fig.add_subplot(gs[1, k]); letter(ax, L)
        for drug in ['gentamicin', 'amikacin']:
            g = d[d['drug'] == drug]
            ax.scatter(g[xcol], g['iwres'], s=3, alpha=0.25, color=COL[drug], rasterized=True)
        ax.axhline(0, color='k', lw=0.8)
        for yv in (-2, 2):
            ax.axhline(yv, color='grey', lw=0.6, ls='--')
        if xcol == 'pred':
            ax.set_xscale('log')
        ax.set_ylim(-6, 6); ax.set_xlabel(xlab); ax.set_ylabel('IWRES')
        ax.set_title(f'IWRES vs {"time after dose" if xcol == "tad" else "population prediction"} '
                     f'({values["pct_abs_iwres_lt2"]:.1f}% within ±2)' if xcol == 'tad' else
                     'IWRES vs population prediction', loc='left')

    # E: forest
    ax = fig.add_subplot(gs[2, 0]); letter(ax, 'E')
    forest = []
    for i, (key, lab) in enumerate(COV_LABELS):
        r = summ.loc[key]
        y = len(COV_LABELS) - 1 - i
        ax.plot([r['hdi_3%'], r['hdi_97%']], [y, y], color='#333333', lw=1.5)
        ax.plot(r['mean'], y, 'o', color='#333333', ms=4)
        ax.plot(r['true_value'], y, 'x', color='#D55E00', ms=6, mew=1.5)
        forest.append({'parameter': key, 'mean': r['mean'], 'hdi_3%': r['hdi_3%'], 'hdi_97%': r['hdi_97%'],
                       'true': r['true_value']})
    ax.set_yticks(range(len(COV_LABELS))); ax.set_yticklabels([l for _, l in COV_LABELS][::-1])
    ax.axvline(0, color='grey', lw=0.6, ls=':')
    ax.set_xlabel('Posterior mean and 94% HDI'); ax.set_title('Covariate effects and offsets\n(× = generating value)', loc='left')
    values['forest'] = forest

    # F: trace
    ax = fig.add_subplot(gs[2, 1]); letter(ax, 'F')
    tc = post['theta_CL'].values
    for c in range(tc.shape[0]):
        ax.plot(tc[c], lw=0.5, alpha=0.8, label=f'Chain {int(post.chain.values[c]) + 1}')
    ax.axhline(5.5, color='#D55E00', lw=1, ls='--', label='Generating value')
    ax.set_xlabel('Draw'); ax.set_ylabel('θ$_{CL}$ (L/h)')
    lo = min(float(tc.min()), 5.5); hi = max(float(tc.max()), 5.5)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.15 * (hi - lo))
    ax.set_title(f'Typical gentamicin CL by chain\n(R-hat = {summ.loc["theta_CL", "r_hat"]:.2f})', loc='left')
    ax.legend(frameon=False, ncol=2, fontsize=7, loc='lower right')
    FIG.mkdir(parents=True, exist_ok=True)
    save(fig, FIG / 'Figure2_PopPK_Diagnostics.png', FIG / 'Figure2_PopPK_Diagnostics.tiff')


def figure_s1(values):
    idata = az.from_netcdf(REPO / 'models' / 'phase2_popk_trace_likelihood.nc')
    post = idata.posterior
    comp = pd.read_csv(R2 / 'popk_r1_reference_comparison.csv')
    truth = pd.read_csv(R2 / 'true_individual_pk.csv')
    ind = pd.read_csv(REPO / 'results' / 'phase3_pkpd' / 'pkpd_indices.csv')[['patient_id', 'CL_bayes', 'Vc_bayes']]
    ind = ind.dropna().merge(truth, on='patient_id')

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    ax = axs[0, 0]; letter(ax, 'A')
    for v, lab, c in [('beta_CL_CRCL', 'CrCL → CL', '#0072B2'), ('beta_CL_WT', 'WT → CL', '#009E73'),
                      ('beta_Vc_WT', 'WT → Vc', '#CC79A7'), ('beta_CL_AGE', 'Age → CL', '#E69F00')]:
        x = post[v].values
        for ch in range(x.shape[0]):
            az_kde = az.kde(x[ch])
            ax.plot(az_kde[0], az_kde[1], color=c, lw=1, alpha=0.8, label=lab if ch == 0 else None)
    ax.set_xlabel('Coefficient (log change per SD of covariate)'); ax.set_ylabel('Density')
    ax.set_title('Production fit: posterior by chain', loc='left'); ax.legend(frameon=False, fontsize=7)

    ax = axs[0, 1]; letter(ax, 'B')
    sel = comp[comp['quantity'].str.contains('per SD')]
    y = np.arange(len(sel))[::-1]
    ax.errorbar(sel['estimate_mean'], y, xerr=[sel['estimate_mean'] - sel['hdi_3'], sel['hdi_97'] - sel['estimate_mean']],
                fmt='o', color='#333333', ms=4, capsize=2, label='Estimate (94% HDI)')
    ax.plot(sel['truth_same_scale'], y, 'x', color='#D55E00', ms=7, mew=1.5, label='Generating model, same scale')
    names = {'beta_CL_CRCL': 'CrCL → CL', 'beta_CL_WT': 'WT → CL', 'beta_CL_AGE': 'Age → CL', 'beta_Vc_WT': 'WT → Vc'}
    ax.set_yticks(y); ax.set_yticklabels([names[q.split(' ')[0]] for q in sel['quantity']])
    ax.set_ylim(-0.6, len(sel) - 0.4 + 1.2)
    ax.set_xlabel('Coefficient per SD of covariate'); ax.set_title('Production fit vs generating model', loc='left')
    ax.legend(frameon=False, fontsize=7, loc='upper left')

    for k, (est, tru, lab, L) in enumerate([('CL_bayes', 'CL_individual', 'CL (L/h)', 'C'),
                                            ('Vc_bayes', 'Vc_individual', 'Vc (L)', 'D')]):
        ax = axs[1, k]; letter(ax, L)
        for drug in ['gentamicin', 'amikacin']:
            g = ind[ind['drug'] == drug]
            ax.scatter(g[tru], g[est], s=3, alpha=0.3, color=COL[drug], label=drug.capitalize(), rasterized=True)
        lim = [min(ind[tru].min(), ind[est].min()) * 0.8, max(ind[tru].max(), ind[est].max()) * 1.2]
        ax.plot(lim, lim, 'k-', lw=0.8); ax.set_xscale('log'); ax.set_yscale('log')
        r = np.corrcoef(np.log(ind[est]), np.log(ind[tru]))[0, 1]
        values[f's1_r_log_{est}'] = float(r)
        ax.set_xlabel(f'True {lab}'); ax.set_ylabel(f'Posterior mean {lab}')
        ax.set_title(f'Individual {lab.split()[0]} recovery (log-scale r = {r:.3f})', loc='left')
        ax.legend(frameon=False, markerscale=3, fontsize=7)
    SUPP.mkdir(parents=True, exist_ok=True)
    save(fig, SUPP / 'FigureS1_PopPK_Production_Fit.png')


def main():
    values = {}
    figure2(values)
    figure_s1(values)
    (R2 / 'figure2_values.json').write_text(json.dumps(values, indent=1, default=float))
    print(json.dumps({k: v for k, v in values.items() if k != 'forest'}, indent=1))


if __name__ == '__main__':
    main()
