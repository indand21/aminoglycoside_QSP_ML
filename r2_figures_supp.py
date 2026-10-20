#!/usr/bin/env python3
"""Revision 2: regenerate Supplementary Figures S2-S7 from released result files.

Every plotted or annotated value is computed at run time from the data and
result files below; nothing is simulated and no result value is typed in.
Design constants (MIC 8, trough limits 2.0/2.5 mg/L, risk cutoffs 0.15/0.35,
bootstrap settings) are the only literals. Every annotated number is written to
results/revision2/supp_figure_values.json.

Inputs
  data/concentrations.csv, data/dosing.csv
  results/phase3_pkpd/pkpd_indices.csv
  results/phase4_ml_enhanced/{test_predictions,algorithm_comparison}.json
  results/revision2/{table4_values,shap_importance,feature_ablation,lr_comparisons,
                     risk_stratification,dose_optimization_summary,
                     dose_optimization_weights}.json
  results/revision2/{shap_values_clinical_cure.csv, dose_optimization_patients.csv}
  production feature matrices via r2_ml_common.production_setup()

Outputs
  Manuscript_AAPS_Journal/Revision2_Submission/Supplementary/figure_png/FigureS2..S7*.png
  results/revision2/supp_figure_values.json

S1 (PopPK diagnostics) is produced elsewhere.

Usage (from the repo root):
    python r2_figures_supp.py
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent
R2 = REPO_ROOT / 'results' / 'revision2'
P4 = REPO_ROOT / 'results' / 'phase4_ml_enhanced'
FIG_DIR = REPO_ROOT / 'Manuscript_AAPS_Journal' / 'Revision2_Submission' / 'Supplementary' / 'figure_png'
VALUES_PATH = R2 / 'supp_figure_values.json'

# ----------------------------------------------------------------------------- #
# Style
# ----------------------------------------------------------------------------- #
OI = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73',
      'yellow': '#F0E442', 'blue': '#0072B2', 'vermilion': '#D55E00', 'purple': '#CC79A7',
      'grey': '#999999'}
DRUG_COL = {'amikacin': OI['blue'], 'gentamicin': OI['orange']}
DRUG_LAB = {'amikacin': 'Amikacin', 'gentamicin': 'Gentamicin'}
MODEL_LAB = {'nephrotoxicity_postdose': 'Post-dose nephrotoxicity',
             'nephrotoxicity_predose': 'Pre-dose nephrotoxicity',
             'clinical_cure': 'Clinical cure'}
MODEL_COL = {'nephrotoxicity_postdose': OI['vermilion'], 'nephrotoxicity_predose': OI['purple'],
             'clinical_cure': OI['green']}
RISK_COL = {'Low': OI['green'], 'Medium': OI['yellow'], 'High': OI['vermilion']}
PCT_CMAP = LinearSegmentedColormap.from_list('oi_div', [OI['blue'], '#DDDDDD', OI['vermilion']])

# Design constants
TROUGH = {'amikacin': 2.5, 'gentamicin': 2.0}   # mg/L, drug-specific trough safety limits
CMAX_MIC_TARGET = 8
AUC_MIC_TARGET = 80
NEPH_CUTS = (0.15, 0.35)
N_BOOT, SEED = 2000, 42
FIG_W = 7.0
DPI = 300

STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'legend.title_fontsize': 8, 'figure.titlesize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'legend.frameon': False, 'savefig.dpi': DPI,
}


def apply_style():
    """Phase 4 sets a seaborn style on import; reset to matplotlib defaults first."""
    plt.style.use('default')
    plt.rcParams.update(STYLE)


apply_style()

VALUES: dict = {}


def rec(fig, panel, key, value):
    """Record an annotated value."""
    VALUES.setdefault(fig, {}).setdefault(panel, {})[key] = value


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def load_json(name, base=R2):
    return json.loads((base / name).read_text())


def wilson(k, n, z=1.96):
    from r2_ml_analyses import wilson as _w
    return _w(k, n, z)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'  wrote {FIG_DIR / name}')


def plain_log(axis):
    """Plain-number tick labels on a log axis (mathtext superscripts would fall below 8 pt)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


ABBR = {'amikacin': 'AMK', 'gentamicin': 'GEN'}


def p_fmt(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


# ----------------------------------------------------------------------------- #
# Shared data
# ----------------------------------------------------------------------------- #
def cohort():
    """Patient table with drug, actual dosing interval, Cmin, CL_bayes (as in r2_tables.load)."""
    from r2_tables import load
    df = load()
    k = pd.read_csv(REPO_ROOT / 'results' / 'phase3_pkpd' / 'pkpd_indices.csv')
    return df.merge(k[['patient_id', 'CL_bayes']], on='patient_id', how='left')


# ----------------------------------------------------------------------------- #
# Figure S2: target attainment extras
# ----------------------------------------------------------------------------- #
def figure_s2(df):
    t4 = load_json('table4_values.json')
    fig = plt.figure(figsize=(FIG_W, 7.0))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.98, top=0.86, bottom=0.07,
                          hspace=0.62, wspace=0.42, height_ratios=[1, 1.15])

    # ---- A: PTA vs MIC over MIC distribution
    ax = fig.add_subplot(gs[0, 0])
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    D = t4['D']
    width = 0.36
    for j, drug in enumerate(['amikacin', 'gentamicin']):
        rows = D[drug]['rows']
        x = np.log2([r['mic'] for r in rows])
        frac = np.array([r['fraction'] for r in rows]) * 100
        ax2.bar(x + (j - 0.5) * width, frac, width=width, color=DRUG_COL[drug], alpha=0.28,
                edgecolor='none', zorder=0)
        ax.plot(x, [100 * r['pta_cmax8'] for r in rows], '-o', color=DRUG_COL[drug], ms=3, lw=1.3,
                zorder=3)
        ax.plot(x, [100 * r['pta_auc80'] for r in rows], '--s', color=DRUG_COL[drug], ms=3, lw=1.1,
                zorder=3)
        rec('S2', 'A', f'{drug}_n', D[drug]['n'])
        rec('S2', 'A', f'{drug}_cfr_cmax8_pct', round(100 * D[drug]['cfr_cmax8'], 1))
        rec('S2', 'A', f'{drug}_cfr_auc80_pct', round(100 * D[drug]['cfr_auc80'], 1))
        rec('S2', 'A', f'{drug}_rows', rows)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    mics = [r['mic'] for r in D['amikacin']['rows']]
    ax.set_xticks(np.log2(mics))
    ax.set_xticklabels([f'{m:g}' for m in mics], rotation=45)
    ax.set_xlim(np.log2(mics[0]) - 0.5, np.log2(mics[-1]) + 0.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel('MIC (mg/L)')
    ax.set_ylabel('Probability of target attainment (%)')
    ax2.set_ylabel('Isolates at MIC (%)')
    ymax2 = max(r['fraction'] for d in ['amikacin', 'gentamicin'] for r in D[d]['rows']) * 100
    ax2.set_ylim(0, ymax2 * 2.1)
    cfr = (f"CFR, Cmax/MIC ≥{CMAX_MIC_TARGET}: AMK {100 * D['amikacin']['cfr_cmax8']:.1f}%, "
           f"GEN {100 * D['gentamicin']['cfr_cmax8']:.1f}%\n"
           f"CFR, AUC24/MIC ≥{AUC_MIC_TARGET}: AMK {100 * D['amikacin']['cfr_auc80']:.1f}%, "
           f"GEN {100 * D['gentamicin']['cfr_auc80']:.1f}%")
    ax.set_title(cfr, loc='left')
    handles = [Line2D([], [], color=DRUG_COL['amikacin'], lw=1.3, label='Amikacin'),
               Line2D([], [], color=DRUG_COL['gentamicin'], lw=1.3, label='Gentamicin'),
               Line2D([], [], color='k', ls='-', marker='o', ms=3, lw=1, label=f'Cmax/MIC ≥{CMAX_MIC_TARGET}'),
               Line2D([], [], color='k', ls='--', marker='s', ms=3, lw=1, label=f'AUC24/MIC ≥{AUC_MIC_TARGET}'),
               Patch(facecolor=OI['grey'], alpha=0.4, label='MIC distribution')]
    ax.legend(handles=handles, loc='upper right', handlelength=1.8, borderaxespad=0.2)
    axA = ax

    # ---- B: attainment by actual dosing interval
    ax = fig.add_subplot(gs[0, 1])
    C = t4['C']
    metrics = [('cmax_mic_8', f'Efficacy (Cmax/MIC ≥{CMAX_MIC_TARGET})', OI['green']),
               ('cmin_drug', f"Trough safety (AMK <{TROUGH['amikacin']:g}, GEN <{TROUGH['gentamicin']:g} mg/L)",
                OI['sky']),
               ('combined', 'Combined', OI['purple'])]
    intervals = sorted(C.keys(), key=lambda s: int(re.sub(r'\D', '', s)))
    xs, labels = [], []
    pos = 0.0
    bw = 0.26
    group_centres = {}
    for drug in ['amikacin', 'gentamicin']:
        centres = []
        for iv in intervals:
            cell = C[iv][drug]
            for m, (key, _, col) in enumerate(metrics):
                ax.bar(pos + (m - 1) * bw, cell[key][1], width=bw, color=col, edgecolor='white', lw=0.3)
                rec('S2', 'B', f'{drug}_{iv}_{key}_pct', cell[key][1])
            xs.append(pos)
            labels.append(f"{iv}\nn={cell['n']}")
            rec('S2', 'B', f'{drug}_{iv}_n', cell['n'])
            centres.append(pos)
            pos += 1.0
        group_centres[drug] = np.mean(centres)
        pos += 0.5
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Patients attaining target (%)')
    for drug, c in group_centres.items():
        ax.text(c, -0.21, DRUG_LAB[drug], transform=ax.get_xaxis_transform(), ha='center', va='top',
                color=DRUG_COL[drug], fontweight='bold')
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in metrics], loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.2)
    axB = ax

    # ---- C: observed concentration-time data for three example patients
    conc = pd.read_csv(REPO_ROOT / 'data' / 'concentrations.csv')
    dosing = pd.read_csv(REPO_ROOT / 'data' / 'dosing.csv')
    lloq = float(conc['assay_lloq'].iloc[0])
    examples = [('gentamicin', 24.0), ('amikacin', 24.0), ('amikacin', 48.0)]
    sub_gs = gs[1, 0].subgridspec(3, 1, hspace=0.25)
    axes_c = []
    tmax = 0
    picks = []
    for i, (drug, iv) in enumerate(examples):
        cat = df[(df['drug'] == drug) & (df['interval_h'] == iv) & df['CL_bayes'].notna()]
        med = float(cat['CL_bayes'].median())
        cat = cat.assign(dist=(cat['CL_bayes'] - med).abs()).sort_values(['dist', 'patient_id'])
        row = cat.iloc[0]
        picks.append((drug, iv, row, med, len(cat)))
        c = conc[conc['patient_id'] == row['patient_id']]
        tmax = max(tmax, dosing.loc[dosing['patient_id'] == row['patient_id'], 'time'].max() + iv,
                   c['time'].max())
    for i, (drug, iv, row, med, ncat) in enumerate(picks):
        ax = fig.add_subplot(sub_gs[i])
        axes_c.append(ax)
        pid = row['patient_id']
        c = conc[conc['patient_id'] == pid].sort_values('time')
        d = dosing[dosing['patient_id'] == pid]
        col = DRUG_COL[drug]
        q = c[~c['bloq']]
        b = c[c['bloq']]
        ax.plot(q['time'], q['concentration'], '-', color=col, lw=0.6, alpha=0.5)
        ax.plot(q['time'], q['concentration'], 'o', color=col, ms=3)
        if len(b):
            ax.plot(b['time'], np.full(len(b), lloq), 'o', mfc='white', mec=col, ms=3)
        ax.axhline(TROUGH[drug], color=OI['vermilion'], ls='--', lw=0.9)
        ax.set_yscale('log')
        plain_log(ax.yaxis)
        ymin = min(lloq, c['concentration'].min()) * 0.2
        ymax = c['concentration'].max() * 40
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([t for t in (0.1, 10, 1000) if ymin <= t <= ymax])
        ax.plot(d['time'], np.full(len(d), 0.0), transform=ax.get_xaxis_transform(), marker='^',
                ls='none', color='k', ms=3.5, clip_on=False)
        ax.set_xlim(-4, tmax + 4)
        ax.text(0.99, 0.97, f"{DRUG_LAB[drug]} q{int(iv)}h, {pid}, CL {row['CL_bayes']:.2f} L/h",
                transform=ax.transAxes, ha='right', va='top')
        if i < 2:
            ax.tick_params(labelbottom=False)
        rec('S2', 'C', f'{drug}_q{int(iv)}h', {
            'patient_id': pid, 'CL_bayes_L_per_h': r4(row['CL_bayes']),
            'category_median_CL_bayes': r4(med), 'n_in_category_with_CL': int(ncat),
            'n_observations': int(len(c)), 'n_bloq': int(c['bloq'].sum()), 'n_doses': int(len(d)),
            'trough_limit_mg_per_L': TROUGH[drug]})
    axes_c[-1].set_xlabel('Time since first dose (h)')
    axes_c[1].set_ylabel('Concentration (mg/L)')
    axes_c[0].legend(handles=[Line2D([], [], marker='o', ls='none', color='k', ms=3, label='Observed'),
                              Line2D([], [], marker='o', ls='none', mfc='white', mec='k', ms=3,
                                     label=f'BLQ (at LLOQ {lloq:g})'),
                              Line2D([], [], marker='^', ls='none', color='k', ms=3.5, label='Dose'),
                              Line2D([], [], ls='--', color=OI['vermilion'], label='Trough limit')],
                     loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2,
                     columnspacing=0.8)
    rec('S2', 'C', 'lloq_mg_per_L', lloq)

    # ---- D: Cmin distribution by drug
    ax = fig.add_subplot(gs[1, 1])
    lo = np.floor(np.log10(df['Cmin'].min()) * 10) / 10
    hi = np.ceil(np.log10(df['Cmin'].max()) * 10) / 10
    bins = np.logspace(lo, hi, 36)
    ytop = 0
    for drug in ['amikacin', 'gentamicin']:
        x = df.loc[df['drug'] == drug, 'Cmin']
        w = np.full(len(x), 100 / len(x))
        h, _, _ = ax.hist(x, bins=bins, weights=w, histtype='stepfilled', color=DRUG_COL[drug],
                          alpha=0.35, lw=0)
        ax.hist(x, bins=bins, weights=w, histtype='step', color=DRUG_COL[drug], lw=1.0)
        ytop = max(ytop, h.max())
    ax.set_xscale('log')
    plain_log(ax.xaxis)
    ax.set_ylim(0, ytop * 1.75)
    txt = []
    for drug in ['amikacin', 'gentamicin']:
        x = df.loc[df['drug'] == drug, 'Cmin']
        pct = 100 * float((x < TROUGH[drug]).mean())
        nfloor = int((x <= x.min()).sum())
        ax.axvline(TROUGH[drug], ymax=0.62, color=DRUG_COL[drug], ls='--', lw=1.0)
        txt.append(f"{DRUG_LAB[drug]} (n={len(x)}): {pct:.1f}% <{TROUGH[drug]:g} mg/L;\n"
                   f"   {nfloor} at minimum value {x.min():g} mg/L")
        rec('S2', 'D', f'{drug}_n', int(len(x)))
        rec('S2', 'D', f'{drug}_pct_below_trough_limit', round(pct, 1))
        rec('S2', 'D', f'{drug}_median_cmin', r4(x.median()))
        rec('S2', 'D', f'{drug}_n_at_minimum_value', nfloor)
        rec('S2', 'D', f'{drug}_minimum_cmin', r4(x.min()))
        # Cross-check against Table 4A
        assert abs(pct - t4['A'][drug]['cmin_drug'][1]) < 0.06, 'Cmin attainment differs from Table 4A'
    ax.text(0.98, 0.98, '\n'.join(txt), transform=ax.transAxes, ha='right', va='top')
    ax.set_xlabel('Cmin (mg/L, log scale)')
    ax.set_ylabel('Patients (% of drug group)')
    ax.legend(handles=[Patch(facecolor=DRUG_COL[d], alpha=0.5, label=DRUG_LAB[d]) for d in DRUG_COL]
              + [Line2D([], [], color='k', ls='--', label='Drug trough limit')],
              loc='upper right', bbox_to_anchor=(1.0, 0.70))
    save_letters(fig, [(axA, 'A'), (axB, 'B'), (axes_c[0], 'C'), (ax, 'D')])
    save(fig, 'FigureS2_Target_Attainment_Extras.png')


# ----------------------------------------------------------------------------- #
# Figure S3: clinical-cure SHAP
# ----------------------------------------------------------------------------- #
def beeswarm_offsets(v, nbins=70, max_half=0.42):
    """Deterministic beeswarm: bin along x; within each bin alternate points above/below."""
    v = np.asarray(v, float)
    edges = np.linspace(v.min(), v.max() + 1e-12, nbins + 1)
    b = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    off = np.zeros(len(v))
    counts = np.bincount(b, minlength=nbins)
    step = max_half / max(1, (counts.max() + 1) // 2)
    for k in np.unique(b):
        idx = np.where(b == k)[0]
        idx = idx[np.argsort(v[idx], kind='stable')]
        j = np.arange(len(idx))
        off[idx] = np.where(j % 2 == 0, 1, -1) * ((j + 1) // 2) * step
    return off


def imp_bar(ax, imp, fig_id, panel, n=15, colour=OI['blue']):
    top = imp[:n]
    from r2_ml_common import ENGINEERED
    names = [r['feature'] for r in top][::-1]
    vals = [r['shap_pct'] for r in top][::-1]
    cols = [OI['purple'] if f in ENGINEERED else colour for f in names]
    ax.barh(range(len(names)), vals, color=cols, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f'{v:.1f}', va='center')
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel('Mean |SHAP| (% of total)')
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.legend(handles=[Patch(color=colour, label='Original'),
                       Patch(color=OI['purple'], label='Engineered')],
              title='Feature type', loc='lower right', bbox_to_anchor=(1.0, 0.02), handlelength=1.0,
              borderaxespad=0.1)
    rec(fig_id, panel, 'shap_pct_top15', {r['feature']: r4(r['shap_pct']) for r in top})


def figure_s3(setup, pdat):
    imp = load_json('shap_importance.json')
    fig = plt.figure(figsize=(FIG_W, 7.6))
    gs = fig.add_gridspec(2, 2, left=0.235, right=0.875, top=0.925, bottom=0.07, hspace=0.32,
                          wspace=0.95)

    # ---- A
    ax = fig.add_subplot(gs[0, 0])
    imp_bar(ax, imp['clinical_cure'], 'S3', 'A')
    axA = ax

    # ---- B: beeswarm
    ax = fig.add_subplot(gs[0, 1])
    axB = ax
    e = setup['clinical_cure']
    sh = pd.read_csv(R2 / 'shap_values_clinical_cure.csv')
    assert (sh['patient_id'].to_numpy() == e['patient_id']).all()
    assert list(sh.columns[:-1]) == e['features']
    top10 = [r['feature'] for r in imp['clinical_cure'][:10]]
    X = e['X']
    sc = None
    for i, f in enumerate(top10[::-1]):
        v = sh[f].to_numpy()
        pc = X[f].rank(pct=True).to_numpy()
        sc = ax.scatter(v, i + beeswarm_offsets(v), c=pc, cmap=PCT_CMAP, vmin=0, vmax=1, s=1.6,
                        lw=0, rasterized=True)
    ax.axvline(0, color=OI['grey'], lw=0.6)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10[::-1])
    ax.set_ylim(-0.6, len(top10) - 0.4)
    ax.set_xlabel('SHAP value (log-odds of cure)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, aspect=25)
    cb.set_label('Feature value (percentile)')
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['Low', 'Median', 'High'])
    rec('S3', 'B', 'features', top10)
    rec('S3', 'B', 'n_patients', int(len(sh)))

    # ---- C: SHAP(Cmax_MIC) vs Cmax_MIC by drug
    ax3 = fig.add_subplot(gs[1, 0])
    x = X['Cmax_MIC'].to_numpy()
    s = sh['Cmax_MIC'].to_numpy()
    drug = pdat['drug'].to_numpy()
    for d in ['amikacin', 'gentamicin']:
        m = drug == d
        ax3.scatter(x[m], s[m], s=3, color=DRUG_COL[d], alpha=0.45, lw=0, label=DRUG_LAB[d],
                    rasterized=True)
        rho = stats.spearmanr(x[m], s[m]).correlation
        rec('S3', 'C', f'{d}_spearman_rho', r4(rho))
        rec('S3', 'C', f'{d}_n', int(m.sum()))
    ax3.axvline(CMAX_MIC_TARGET, color='k', ls='--', lw=0.9)
    ax3.axhline(0, color=OI['grey'], lw=0.6)
    ax3.set_xscale('log')
    plain_log(ax3.xaxis)
    ax3.set_xlabel('Cmax/MIC (log scale)')
    ax3.set_ylabel('SHAP value for Cmax/MIC')
    ax3.text(CMAX_MIC_TARGET * 0.85, ax3.get_ylim()[1] - 0.02 * np.ptp(ax3.get_ylim()),
             f'Cmax/MIC = {CMAX_MIC_TARGET}', ha='right', va='top')
    lo, hi = s[x < CMAX_MIC_TARGET], s[x >= CMAX_MIC_TARGET]
    rec('S3', 'C', 'mean_shap_below_8', r4(lo.mean()))
    rec('S3', 'C', 'mean_shap_at_or_above_8', r4(hi.mean()))
    ax3.set_title(f'Mean SHAP: {lo.mean():+.2f} below {CMAX_MIC_TARGET}, '
                  f'{hi.mean():+.2f} at/above', loc='left')

    ax3.legend(loc='lower right', bbox_to_anchor=(1.0, 0.04), markerscale=2.5, handletextpad=0.05,
               handlelength=0.8, borderaxespad=0.1)

    # ---- D: post-dose nephrotoxicity SHAP % vs gain %
    ax = fig.add_subplot(gs[1, 1])
    post = imp['nephrotoxicity_postdose']
    top = post[:15]
    names = [r['feature'] for r in top][::-1]
    shp = [r['shap_pct'] for r in top][::-1]
    gn = [r['gain_pct'] for r in top][::-1]
    gr = [r['gain_rank'] for r in top][::-1]
    yy = np.arange(len(names))
    for i in yy:
        ax.plot([shp[i], gn[i]], [i, i], color=OI['grey'], lw=1.0, zorder=1)
    ax.scatter(shp, yy, color=OI['blue'], s=16, zorder=2, label='SHAP (% of total)')
    ax.scatter(gn, yy, color=OI['orange'], marker='D', s=13, zorder=2, label='Gain (% of total)')
    ax.set_yticks(yy)
    ax.set_yticklabels([f'{n} [{g}]' for n, g in zip(names, gr)])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel('Importance (% of total)')
    rho_all = stats.spearmanr([r['shap_pct'] for r in post], [r['gain_pct'] for r in post]).correlation
    top15_gain = {r['feature'] for r in sorted(post, key=lambda r: r['gain_rank'])[:15]}
    overlap = len(top15_gain & set(names))
    ax.set_xlim(0, max(max(shp), max(gn)) * 1.08)
    ax.text(0.98, 0.42, f'Spearman ρ = {rho_all:.2f}\n(all {len(post)} features)\n'
            f'Top-15 overlap: {overlap}/15', transform=ax.transAxes, ha='right', va='center')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), handletextpad=0.3)
    ax.set_title('Post-dose nephrotoxicity model\n[ ] = gain rank', loc='left')
    rec('S3', 'D', 'spearman_rho_shap_vs_gain_all_features', r4(rho_all))
    rec('S3', 'D', 'n_features', len(post))
    rec('S3', 'D', 'top15_overlap_shap_vs_gain', overlap)
    rec('S3', 'D', 'top15', {r['feature']: {'shap_pct': r4(r['shap_pct']), 'gain_pct': r4(r['gain_pct']),
                                            'gain_rank': r['gain_rank']} for r in top})

    save_letters(fig, [(axA, 'A'), (axB, 'B'), (ax3, 'C'), (ax, 'D')])
    save(fig, 'FigureS3_Cure_SHAP.png')


def save_letters(fig, pairs):
    """Place bold panel letters at the top-left of each axis' tight bounding box
    (which includes tick labels, titles and attached legends)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in pairs:
        bb = ax.get_tightbbox(r)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(max(x0 - 0.005, 0.003), min(y1 + 0.004, 0.995), s, fontsize=12, fontweight='bold',
                 ha='left', va='bottom')


# ----------------------------------------------------------------------------- #
# Comparison helper (S4 B and C)
# ----------------------------------------------------------------------------- #
def comparison_panel(ax, items, arm_labels, arm_cols, fig_id, panel):
    """items: list of (label, a_auc, a_ci, b_auc, b_ci, diff, diff_ci, p)."""
    n = len(items)
    for i, (lab, a, aci, b, bci, d, dci, p) in enumerate(items):
        y = n - 1 - i
        for off, v, ci, col, mk in [(0.10, a, aci, arm_cols[0], 'o'), (-0.12, b, bci, arm_cols[1], 's')]:
            ax.errorbar(v, y + off, xerr=[[v - ci[0]], [ci[1] - v]], fmt=mk, color=col, ms=4,
                        capsize=2, lw=1)
        ax.text(0.02, y + 0.36, lab, transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontweight='bold')
        ax.text(0.98, y - 0.36, f'Δ {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]; {p_fmt(p)}',
                transform=ax.get_yaxis_transform(), ha='right', va='center')
        rec(fig_id, panel, lab.replace('\n', ' '), {
            arm_labels[0]: r4(a), f'{arm_labels[0]}_ci': [r4(c) for c in aci],
            arm_labels[1]: r4(b), f'{arm_labels[1]}_ci': [r4(c) for c in bci],
            'difference': r4(d), 'difference_95ci': [r4(c) for c in dci], 'delong_p': r4(p)})
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    for k in range(1, n):
        ax.axhline(k - 0.5, color='0.85', lw=0.6)
    ax.set_ylim(-0.55, n - 0.45)
    allv = [v for it in items for v in (*it[2], *it[4])]
    ax.set_xlim(np.floor((min(allv) - 0.02) * 20) / 20, 1.0)
    ax.set_xlabel('Held-out AUROC (95% CI)')
    ax.legend(handles=[Line2D([], [], marker='o', color=arm_cols[0], ls='none', ms=4, label=arm_labels[0]),
                       Line2D([], [], marker='s', color=arm_cols[1], ls='none', ms=4, label=arm_labels[1])],
              loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2)


# ----------------------------------------------------------------------------- #
# Figure S4: ML extras (held-out N=300)
# ----------------------------------------------------------------------------- #
def figure_s4(setup):
    preds = load_json('test_predictions.json', P4)
    fig = plt.figure(figsize=(FIG_W, 6.4))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.98, top=0.92, bottom=0.11, hspace=0.50,
                          wspace=0.28)
    axes = []

    # ---- A: precision-recall
    ax = fig.add_subplot(gs[0, 0])
    axes.append(ax)
    for name in ['nephrotoxicity_postdose', 'nephrotoxicity_predose', 'clinical_cure']:
        y = np.asarray(preds[name]['y_test'])
        p = np.asarray(preds[name]['y_pred_proba'])
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        prev = y.mean()
        ax.plot(rc, pr, color=MODEL_COL[name], lw=1.3, label=f'{MODEL_LAB[name]} (AP {ap:.2f})')
        ax.axhline(prev, color=MODEL_COL[name], ls=':', lw=0.9)
        rec('S4', 'A', name, {'average_precision': r4(ap), 'prevalence': r4(prev),
                               'n_test': int(len(y)), 'n_events': int(y.sum())})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    hh, ll = ax.get_legend_handles_labels()
    hh.append(Line2D([], [], color=OI['grey'], ls=':', lw=0.9))
    ll.append('Event prevalence (dotted)')
    ax.legend(hh, ll, loc='lower left', handlelength=1.4, frameon=True, framealpha=1, facecolor='white', edgecolor='none')
    ax.set_title(f"Held-out test set (N = {len(preds['clinical_cure']['y_test'])})", loc='left')

    # ---- B: feature-engineering ablation
    ax = fig.add_subplot(gs[0, 1])
    axes.append(ax)
    abl_path = R2 / 'feature_ablation.json'
    if abl_path.exists():
        abl = json.loads(abl_path.read_text())
        items = []
        for name in ['nephrotoxicity_postdose', 'nephrotoxicity_predose', 'clinical_cure']:
            r = abl[name]
            lab = f"{MODEL_LAB[name]} ({r['n_features_with']} vs {r['n_features_without']} features)"
            items.append((lab, r['with_engineered'], r['with_engineered_ci'], r['without_engineered'],
                          r['without_engineered_ci'], r['difference'], r['difference_95ci'],
                          r['delong_p']))
        comparison_panel(ax, items, ['With engineered', 'Without engineered'],
                         [OI['vermilion'], OI['sky']], 'S4', 'B')
        rec('S4', 'B', 'status', 'computed from results/revision2/feature_ablation.json')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Feature-engineering ablation\nnot yet available:\n'
                'results/revision2/feature_ablation.json\nis missing. Run r2_ml_analyses.py,\n'
                'then re-run r2_figures_supp.py.', ha='center', va='center', transform=ax.transAxes,
                color=OI['vermilion'])
        rec('S4', 'B', 'status', 'MISSING results/revision2/feature_ablation.json; panel is a placeholder')
        print('  WARNING: feature_ablation.json missing; S4 panel B is a placeholder. Re-run later.')

    # ---- C: logistic regression vs XGBoost
    ax = fig.add_subplot(gs[1, 0])
    axes.append(ax)
    lr = load_json('lr_comparisons.json')
    n_post = len(setup['nephrotoxicity_postdose']['features'])
    n_pre = len(setup['nephrotoxicity_predose']['features'])
    spec = [('two_exposure_predictors', 'xgboost', 'logistic',
             f"{' + '.join(lr['two_exposure_predictors']['predictors'])} (fixed XGBoost)"),
            ('postdose_55_features', 'xgboost_production', 'logistic_l2', f'Post-dose ({n_post} features)'),
            ('predose_26_features', 'xgboost_production', 'logistic_l2', f'Pre-dose ({n_pre} features)')]
    items = []
    for key, a, b, lab in spec:
        r = lr[key]
        items.append((lab, r[a], r[f'{a}_ci'], r[b], r[f'{b}_ci'], r['difference'],
                      r['difference_95ci'], r['delong_p']))
    comparison_panel(ax, items, ['XGBoost', 'Logistic regression'], [OI['vermilion'], OI['blue']],
                     'S4', 'C')
    rec('S4', 'C', 'n_test', lr['postdose_55_features']['n_test'])
    rec('S4', 'C', 'n_events', lr['postdose_55_features']['n_events'])
    ax.text(0.0, -0.22, f"Nephrotoxicity, N = {lr['postdose_55_features']['n_test']} "
            f"({lr['postdose_55_features']['n_events']} events); Δ = XGBoost − LR",
            transform=ax.transAxes, ha='left', va='top')

    # ---- D: algorithm comparison Brier/ECE
    ax = fig.add_subplot(gs[1, 1])
    axes.append(ax)
    alg = load_json('algorithm_comparison.json', P4)
    fw = list(alg['frameworks'].keys())
    xs = np.arange(len(fw))
    bw = 0.36
    brier = [alg['frameworks'][f]['brier_score'] for f in fw]
    ece = [alg['frameworks'][f]['ece'] for f in fw]
    auc = [alg['frameworks'][f]['test_auc'] for f in fw]
    ax.bar(xs - bw / 2, brier, bw, color=OI['blue'], label='Brier score')
    ax.bar(xs + bw / 2, ece, bw, color=OI['sky'], label='ECE')
    top = max(brier + ece)
    for i in range(len(fw)):
        ax.text(xs[i] - bw / 2, brier[i] + top * 0.02, f'{brier[i]:.3f}', ha='center', va='bottom',
                rotation=90)
        ax.text(xs[i] + bw / 2, ece[i] + top * 0.02, f'{ece[i]:.3f}', ha='center', va='bottom',
                rotation=90)
        rec('S4', 'D', fw[i], {'brier': r4(brier[i]), 'ece': r4(ece[i]), 'test_auc': r4(auc[i])})
    pretty = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost', 'stacking': 'Stacking'}
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{pretty.get(f, f)}\nAUROC\n{a:.3f}" for f, a in zip(fw, auc)])
    ax.set_ylim(0, top * 1.55)
    ax.set_ylabel('Score (lower is better)')
    ax.legend(loc='upper left', ncol=2)
    ax.set_title(f"{MODEL_LAB[alg['source_model']]} ({alg['n_features']} features)", loc='left')
    rec('S4', 'D', 'source_model', alg['source_model'])
    rec('S4', 'D', 'n_trials', alg['n_trials'])

    save_letters(fig, list(zip(axes, 'ABCD')))
    save(fig, 'FigureS4_ML_Performance_Extras.png')


# ----------------------------------------------------------------------------- #
# Figure S5: pre-dose model
# ----------------------------------------------------------------------------- #
def figure_s5():
    from phase4_machine_learning_enhanced import expected_calibration_error
    from r2_ml_analyses import boot_auc_ci
    imp = load_json('shap_importance.json')
    preds = load_json('test_predictions.json', P4)['nephrotoxicity_predose']
    y = np.asarray(preds['y_test'])
    p = np.asarray(preds['y_pred_proba'])
    fig = plt.figure(figsize=(FIG_W, 6.6))
    gs = fig.add_gridspec(2, 2, left=0.215, right=0.98, top=0.93, bottom=0.08, hspace=0.42,
                          wspace=0.38)
    axes = []

    # ---- A
    ax = fig.add_subplot(gs[0, 0])
    axes.append(ax)
    imp_bar(ax, imp['nephrotoxicity_predose'], 'S5', 'A')

    # ---- B: ROC with bootstrap band
    ax = fig.add_subplot(gs[0, 1])
    axes.append(ax)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    lo, hi = boot_auc_ci(y, p, n_boot=N_BOOT, seed=SEED)
    grid = np.linspace(0, 1, 201)
    rng = np.random.default_rng(SEED)
    tprs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        f_, t_, _ = roc_curve(y[i], p[i])
        tt = np.interp(grid, f_, t_)
        tt[0] = 0.0
        tprs.append(tt)
    tprs = np.array(tprs)
    ax.fill_between(grid, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0),
                    color=OI['purple'], alpha=0.25, lw=0, label='95% bootstrap band')
    ax.plot(fpr, tpr, color=OI['purple'], lw=1.5, label=f'AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]')
    ax.plot([0, 1], [0, 1], ':', color=OI['grey'], lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('1 − specificity')
    ax.set_ylabel('Sensitivity')
    ax.legend(loc='lower right')
    ax.set_title(f'Held-out N = {len(y)} ({int(y.sum())} events)', loc='left')
    rec('S5', 'B', 'auroc', r4(auc))
    rec('S5', 'B', 'auroc_95ci', [r4(lo), r4(hi)])
    rec('S5', 'B', 'n_boot_used', int(len(tprs)))
    rec('S5', 'B', 'n_test', int(len(y)))
    rec('S5', 'B', 'n_events', int(y.sum()))

    # ---- C: reliability diagram
    ax = fig.add_subplot(gs[1, 0])
    axes.append(ax)
    edges = np.linspace(0, 1, 11)
    bins = np.clip(np.digitize(p, edges) - 1, 0, 9)
    rows = []
    for b in range(10):
        m = bins == b
        if m.sum() == 0:
            continue
        k, n = int(y[m].sum()), int(m.sum())
        wl, wh = wilson(k, n)
        rows.append((p[m].mean(), k / n, wl, wh, n))
    rows = np.array(rows)
    ax.plot([0, 1], [0, 1], ':', color=OI['grey'], lw=0.9, label='Perfect calibration')
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=[rows[:, 1] - rows[:, 2], rows[:, 3] - rows[:, 1]],
                fmt='-o', color=OI['purple'], ms=3.5, lw=1.1, capsize=2, label='Observed (Wilson 95% CI)')
    for r in rows:
        ax.text(r[0], min(r[3] + 0.03, 1.0), f'{int(r[4])}', ha='center', va='bottom', color='0.25')
    brier = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p, n_bins=10)
    ax.text(0.98, 0.04, f'Brier {brier:.3f}\nECE {ece:.3f}', transform=ax.transAxes, ha='right',
            va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed nephrotoxicity rate')
    ax.legend(loc='upper left')
    rec('S5', 'C', 'brier', r4(brier))
    rec('S5', 'C', 'ece_10bins', r4(ece))
    rec('S5', 'C', 'bins', [{'mean_pred': r4(r[0]), 'observed': r4(r[1]), 'n': int(r[4])} for r in rows])

    # ---- D: observed rate by pre-dose risk group (OOF, N=1500)
    ax = fig.add_subplot(gs[1, 1])
    axes.append(ax)
    rs = load_json('risk_stratification.json')['nephrotoxicity_predose']
    cuts = rs['cutoffs']
    labs = {'Low': f'Low\n(<{cuts[0]:g})', 'Medium': f'Medium\n({cuts[0]:g}–{cuts[1]:g})',
            'High': f'High\n(>{cuts[1]:g})'}
    groups = rs['groups']
    for i, g in enumerate(groups):
        wl, wh = wilson(g['events'], g['n'])
        assert abs(wl - g['rate_ci'][0]) < 1e-9 and abs(wh - g['rate_ci'][1]) < 1e-9
        ax.bar(i, 100 * g['rate'], color=RISK_COL[g['group']], edgecolor='k', lw=0.5, width=0.6)
        ax.errorbar(i, 100 * g['rate'], yerr=[[100 * (g['rate'] - wl)], [100 * (wh - g['rate'])]],
                    color='k', capsize=3, lw=1)
        ax.text(i, 100 * max(wh, g['pred_mean']) + 2.5, f"{100 * g['rate']:.1f}%\n{g['events']}/{g['n']}",
                ha='center', va='bottom')
        ax.plot(i, 100 * g['pred_mean'], marker='D', color='k', mfc='white', ms=4, ls='none')
        rec('S5', 'D', g['group'], {'n': g['n'], 'events': g['events'], 'rate_pct': round(100 * g['rate'], 1),
                                    'rate_ci_pct': [round(100 * wl, 1), round(100 * wh, 1)],
                                    'mean_predicted_pct': round(100 * g['pred_mean'], 1)})
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labs[g['group']] for g in groups])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Observed nephrotoxicity (%)')
    ax.set_xlabel('Pre-dose predicted risk group')
    ax.legend(handles=[Line2D([], [], marker='D', color='k', mfc='white', ms=4, ls='none',
                              label='Mean predicted risk')], loc='upper left')
    ax.set_title(f"Out-of-fold predictions, N = {sum(g['n'] for g in groups)}", loc='left')
    rec('S5', 'D', 'oof_auroc', r4(rs['oof_auroc']))

    save_letters(fig, list(zip(axes, 'ABCD')))
    save(fig, 'FigureS5_Predose_SHAP.png')


# ----------------------------------------------------------------------------- #
# Figure S6: risk-stratification extras (OOF N=1500)
# ----------------------------------------------------------------------------- #
def figure_s6():
    rs = load_json('risk_stratification.json')
    fig = plt.figure(figsize=(FIG_W, 6.8))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.92, bottom=0.60, wspace=0.75)
    gs2 = fig.add_gridspec(1, 2, left=0.285, right=0.98, top=0.43, bottom=0.08, wspace=0.42)
    axes = []

    # ---- A: tertiles
    ax = fig.add_subplot(gs[0, 0])
    axes.append(ax)
    models = ['nephrotoxicity_postdose', 'nephrotoxicity_predose', 'clinical_cure']
    bw = 0.26
    xl = []
    for i, name in enumerate(models):
        tg = rs[name]['tertile_groups']
        tc = rs[name]['tertile_cutoffs']
        for j, g in enumerate(tg):
            x = i + (j - 1) * bw
            ax.bar(x, 100 * g['rate'], bw, color=RISK_COL[g['group']], edgecolor='k', lw=0.5)
            ax.errorbar(x, 100 * g['rate'], yerr=[[100 * (g['rate'] - g['rate_ci'][0])],
                                                  [100 * (g['rate_ci'][1] - g['rate'])]],
                        color='k', capsize=1.5, lw=0.8)
            ax.text(x, 100 * g['rate_ci'][1] + 1.5, f"{100 * g['rate']:.0f}", ha='center', va='bottom')
        lab = MODEL_LAB[name].replace(' nephrotoxicity', '\nnephrotoxicity').replace('Clinical cure', 'Clinical\ncure')
        xl.append(f"{lab}\n({tc[0]:.3f}, {tc[1]:.3f})")
        rec('S6', 'A', name, {'tertile_cutoffs': [r4(c) for c in tc],
                               'groups': {g['group']: {'n': g['n'], 'events': g['events'],
                                                       'rate_pct': round(100 * g['rate'], 1),
                                                       'rate_ci_pct': [round(100 * c, 1) for c in g['rate_ci']]}
                                          for g in tg}})
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(xl)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Observed event rate (%)')
    ax.set_xlabel('Model (tertile cutoffs)')
    ax.legend(handles=[Patch(facecolor=RISK_COL[k], edgecolor='k', lw=0.5, label=f'{k} tertile')
                       for k in RISK_COL], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.2, columnspacing=0.8)

    # ---- B: subgroup RR forest
    subs = rs['subgroups_postdose']
    ax = fig.add_subplot(gs[0, 1])
    axes.append(ax)
    ylabs = [f"{s['subgroup']} (n={s['n']})" for s in subs]
    yy = np.arange(len(subs))[::-1]
    for y_, s in zip(yy, subs):
        col = DRUG_COL.get(s['subgroup'].lower(), 'k')
        ax.errorbar(s['rr'], y_, xerr=[[s['rr'] - s['rr_ci'][0]], [s['rr_ci'][1] - s['rr']]],
                    fmt='s', color=col, ms=4, capsize=2, lw=1)
        rec('S6', 'B', s['subgroup'], {'n': s['n'], 'rr': r4(s['rr']), 'rr_ci': [r4(c) for c in s['rr_ci']],
                                       'high_n': s['high_n'], 'high_events': s['high_events'],
                                       'low_n': s['low_n'], 'low_events': s['low_events']})
    ax.set_xscale('log')
    ax.axvline(1, color=OI['grey'], lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(ylabs)
    ax.set_ylim(-0.6, len(subs) - 0.4)
    ax.set_xlabel(f'Relative risk, high (>{NEPH_CUTS[1]:g}) vs low (<{NEPH_CUTS[0]:g})\n(log scale, 95% CI)')
    xmin = min(s['rr_ci'][0] for s in subs)
    xmax = max(s['rr_ci'][1] for s in subs)
    ax.set_xlim(min(0.8, xmin * 0.8), xmax * 1.3)
    from matplotlib.ticker import FixedLocator, NullFormatter
    ticks = [t for t in [1, 2, 5, 10, 20, 50, 100] if t <= xmax * 1.3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title('Post-dose nephrotoxicity, cutoffs ' f'{NEPH_CUTS[0]:g}/{NEPH_CUTS[1]:g}', loc='left')

    # ---- C: subgroup AUROC
    ax = fig.add_subplot(gs2[0, 0])
    axes.append(ax)
    overall = rs['nephrotoxicity_postdose']['oof_auroc']
    for y_, s in zip(yy, subs):
        col = DRUG_COL.get(s['subgroup'].lower(), 'k')
        ax.plot(s['auroc'], y_, 'o', color=col, ms=4.5)
        ax.text(s['auroc'] + 0.004, y_, f"{s['auroc']:.3f}", va='center', ha='left',
                bbox=dict(fc='white', ec='none', pad=0.5), zorder=3)
        rec('S6', 'C', s['subgroup'], {'auroc': r4(s['auroc']), 'n': s['n'], 'events': s['events']})
    ax.axvline(overall, color=OI['vermilion'], ls='--', lw=0.9)
    ax.text(overall + 0.002, len(subs) - 0.35, f'Overall {overall:.3f}', color=OI['vermilion'], ha='left',
            va='bottom')
    rec('S6', 'C', 'overall_oof_auroc', r4(overall))
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{s['subgroup']} ({s['events']}/{s['n']})" for s in subs])
    ax.set_ylim(-0.6, len(subs) + 0.3)
    lo_ = min(s['auroc'] for s in subs)
    hi_ = max(s['auroc'] for s in subs)
    ax.set_xlim(lo_ - 0.02, hi_ + 0.03)
    ax.set_xlabel('Out-of-fold AUROC (events/n)')

    # ---- D: cure by risk group
    ax = fig.add_subplot(gs2[0, 1])
    axes.append(ax)
    cr = rs['clinical_cure']
    cuts = cr['cutoffs']
    labs = {'Low': f'Low\n(<{cuts[0]:g})', 'Medium': f'Medium\n({cuts[0]:g}–{cuts[1]:g})',
            'High': f'High\n(>{cuts[1]:g})'}
    cure_col = {'Low': OI['vermilion'], 'Medium': OI['yellow'], 'High': OI['green']}
    for i, g in enumerate(cr['groups']):
        ax.bar(i, 100 * g['rate'], 0.6, color=cure_col[g['group']], edgecolor='k', lw=0.5)
        ax.errorbar(i, 100 * g['rate'], yerr=[[100 * (g['rate'] - g['rate_ci'][0])],
                                              [100 * (g['rate_ci'][1] - g['rate'])]],
                    color='k', capsize=3, lw=1)
        ax.text(i, 100 * max(g['rate_ci'][1], g['pred_mean']) + 2.5,
                f"{100 * g['rate']:.1f}%\n{g['events']}/{g['n']}", ha='center', va='bottom')
        ax.plot(i, 100 * g['pred_mean'], marker='D', color='k', mfc='white', ms=4, ls='none')
        rec('S6', 'D', g['group'], {'n': g['n'], 'events': g['events'], 'rate_pct': round(100 * g['rate'], 1),
                                    'rate_ci_pct': [round(100 * c, 1) for c in g['rate_ci']],
                                    'mean_predicted_pct': round(100 * g['pred_mean'], 1),
                                    **({'rr_vs_low': r4(g['rr_vs_low'])} if 'rr_vs_low' in g else {})})
    ax.set_xticks(range(3))
    ax.set_xticklabels([labs[g['group']] for g in cr['groups']])
    ax.set_ylim(0, 118)
    ax.set_ylabel('Observed clinical cure (%)')
    ax.set_xlabel('Predicted probability of cure')
    ax.legend(handles=[Line2D([], [], marker='D', color='k', mfc='white', ms=4, ls='none',
                              label='Mean predicted')], loc='upper left')
    ax.set_title(f"Clinical-cure model, OOF N = {sum(g['n'] for g in cr['groups'])}", loc='left')
    rec('S6', 'D', 'oof_auroc', r4(cr['oof_auroc']))

    save_letters(fig, list(zip(axes, 'ABCD')))
    save(fig, 'FigureS6_Risk_Stratification_Extras.png')


# ----------------------------------------------------------------------------- #
# Figure S7: dose-optimisation extras (N=1500)
# ----------------------------------------------------------------------------- #
def figure_s7():
    pts = pd.read_csv(R2 / 'dose_optimization_patients.csv')
    summ = load_json('dose_optimization_summary.json')
    wts = load_json('dose_optimization_weights.json')
    fig = plt.figure(figsize=(FIG_W, 6.4))
    gs = fig.add_gridspec(2, 1, left=0.10, right=0.98, top=0.90, bottom=0.10, hspace=0.50,
                          height_ratios=[1, 1.05])
    gA = gs[0].subgridspec(1, 2, wspace=0.08, width_ratios=[1, 1.25])
    gB = gs[1].subgridspec(1, 3, wspace=0.62)
    axes = []

    # ---- A: weight-profile sensitivity
    profiles = list(wts.keys())
    order = ['cure', 'safety', 'cmax_target', 'trough_safety']
    ylab = []
    for k in profiles:
        w = wts[k]['weights']
        base = re.sub(r'\s*\(.*\)\s*$', '', k)
        ylab.append(f"{base} ({'/'.join(f'{w[o]:.2f}' for o in order)})")
    yy = np.arange(len(profiles))[::-1]
    axd = fig.add_subplot(gA[0])
    axp = fig.add_subplot(gA[1], sharey=axd)
    axes.append(axd)
    for y_, k in zip(yy, profiles):
        m, lo, hi = wts[k]['dose_change_pct']
        axd.errorbar(m, y_, xerr=[[m - lo], [hi - m]], fmt='o', color='k', ms=4, capsize=2, lw=1)
        for key, col, mk, off, lab in [('true_aki_change_pp', OI['vermilion'], 's', 0.13, 'P(nephrotoxicity)'),
                                       ('true_cure_change_pp', OI['green'], 'o', -0.13, 'P(cure)')]:
            m2, lo2, hi2 = wts[k][key]
            axp.errorbar(m2, y_ + off, xerr=[[m2 - lo2], [hi2 - m2]], fmt=mk, color=col, ms=4,
                         capsize=2, lw=1, label=lab if y_ == yy[0] else None)
        rec('S7', 'A', k, {'weights': wts[k]['weights'],
                           'dose_change_pct': [r4(v) for v in wts[k]['dose_change_pct']],
                           'true_aki_change_pp': [r4(v) for v in wts[k]['true_aki_change_pp']],
                           'true_cure_change_pp': [r4(v) for v in wts[k]['true_cure_change_pp']],
                           'pct_improved_true_objective': r4(wts[k]['pct_improved_true_objective'])})
    for a in (axd, axp):
        a.axvline(0, color=OI['grey'], lw=0.8)
    axd.set_yticks(yy)
    axd.set_yticklabels(ylab)
    axd.set_ylim(-0.6, len(profiles) - 0.4)
    axd.set_xlabel('Mean dose change (%, 95% CI)')
    axd.set_title('Weights: cure/safety/Cmax target/trough safety', loc='left')
    axp.tick_params(labelleft=False)
    axp.set_xlabel('True-simulator change (percentage points, 95% CI)')
    axp.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2)
    # Shift the dose axis right so the long y labels fit in the left margin
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lab_w = max(t.get_window_extent(r).width for t in axd.get_yticklabels()) / fig.bbox.width
    pd_ = axd.get_position()
    pp_ = axp.get_position()
    new_left = 0.02 + lab_w + 0.01
    total = pp_.x1 - new_left
    wd = (total - 0.03) * 0.42
    axd.set_position([new_left, pd_.y0, wd, pd_.height])
    axp.set_position([new_left + wd + 0.03, pp_.y0, total - wd - 0.03, pp_.height])

    # ---- B/C: predicted vs true change
    d_aki_pred = 100 * (pts['pred_aki_opt'] - pts['pred_aki_base'])
    d_aki_true = 100 * (pts['true_aki_opt'] - pts['true_aki_base'])
    d_cure_pred = 100 * (pts['pred_cure_opt'] - pts['pred_cure_base'])
    d_cure_true = 100 * (pts['true_cure_opt'] - pts['true_cure_base'])
    for j, (xp, yt, lab, panel, skey) in enumerate([
            (d_aki_pred, d_aki_true, 'P(nephrotoxicity)', 'B', 'corr_pred_true_aki_change'),
            (d_cure_pred, d_cure_true, 'P(cure)', 'C', 'corr_pred_true_cure_change')]):
        ax = fig.add_subplot(gB[j])
        axes.append(ax)
        for d in ['amikacin', 'gentamicin']:
            m = pts['drug'] == d
            ax.scatter(xp[m], yt[m], s=3, color=DRUG_COL[d], alpha=0.4, lw=0, label=DRUG_LAB[d],
                       rasterized=True)
        lim = [min(xp.min(), yt.min()), max(xp.max(), yt.max())]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, 'k--', lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        r_all = stats.pearsonr(xp, yt)[0]
        assert abs(r_all - summ['all'][skey]) < 1e-6, f'{skey} differs from summary'
        # Text block doubles as the drug legend (lines coloured by drug)
        ax.text(0.98, 0.03 + 0.075 * 2, f'Pearson r = {r_all:.2f}', transform=ax.transAxes,
                ha='right', va='bottom')
        rec('S7', panel, 'pearson_r_all', r4(r_all))
        rec('S7', panel, 'n', int(len(xp)))
        for k, d in enumerate(['amikacin', 'gentamicin']):
            m = (pts['drug'] == d).to_numpy()
            rd = stats.pearsonr(xp[m], yt[m])[0]
            ax.text(0.98, 0.03 + 0.075 * (1 - k), f'{ABBR[d]} r = {rd:.2f}', transform=ax.transAxes,
                    ha='right', va='bottom', color=DRUG_COL[d], fontweight='bold')
            rec('S7', panel, f'pearson_r_{d}', r4(rd))
        ax.set_xlabel(f'ML-predicted change in\n{lab} (pp)')
        ax.set_ylabel(f'True change in\n{lab} (pp)')

    # ---- D: change in true objective score
    ax = fig.add_subplot(gB[2])
    axes.append(ax)
    ds = pts['true_score_opt'] - pts['true_score_base']
    bins = np.linspace(ds.min(), ds.max(), 41)
    for d in ['amikacin', 'gentamicin']:
        x = ds[pts['drug'] == d]
        ax.hist(x, bins=bins, weights=np.full(len(x), 100 / len(x)), histtype='stepfilled',
                color=DRUG_COL[d], alpha=0.35, lw=0)
        ax.hist(x, bins=bins, weights=np.full(len(x), 100 / len(x)), histtype='step',
                color=DRUG_COL[d], lw=1, label=DRUG_LAB[d])
    ax.axvline(0, ymax=0.58, color='k', lw=0.8)
    imp_ = 100 * float((ds > 0).mean())
    wor = 100 * float((ds < 0).mean())
    unc = 100 * float((ds == 0).mean())
    assert abs(imp_ - summ['all']['pct_improved_true_objective']) < 1e-6
    assert abs(wor - summ['all']['pct_worse_true_objective']) < 1e-6
    lines = [f'All: {imp_:.1f}% improved,', f'      {wor:.1f}% worse']
    rec('S7', 'D', 'all', {'n': int(len(ds)), 'pct_improved': round(imp_, 1), 'pct_worse': round(wor, 1),
                           'pct_unchanged': round(unc, 1), 'median_change': r4(ds.median())})
    for d in ['amikacin', 'gentamicin']:
        x = ds[pts['drug'] == d]
        pi, pw = 100 * float((x > 0).mean()), 100 * float((x < 0).mean())
        lines.append(f'{ABBR[d]}: {pi:.1f}% / {pw:.1f}%')
        rec('S7', 'D', d, {'n': int(len(x)), 'pct_improved': round(pi, 1), 'pct_worse': round(pw, 1),
                           'pct_unchanged': round(100 * float((x == 0).mean()), 1),
                           'median_change': r4(x.median())})
    ax.text(0.97, 0.97, '\n'.join(lines), transform=ax.transAxes, ha='right', va='top')
    ax.set_xlabel('Change in true objective score\n(optimised − baseline)')
    ax.set_ylabel('Patients (% of drug group)')
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax / 0.55)
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.52), handlelength=1.2)

    save_letters(fig, list(zip(axes, 'ABCD')))
    save(fig, 'FigureS7_Dose_Optimization_Extras.png')


# ----------------------------------------------------------------------------- #
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    os.chdir(REPO_ROOT)
    from r2_ml_common import production_setup
    print('Loading cohort and production feature matrices ...')
    df = cohort()
    pipe, setup = production_setup()
    pdat = pipe.data[['patient_id', 'drug']].reset_index(drop=True)
    assert (pdat['patient_id'].to_numpy() == setup['clinical_cure']['patient_id']).all()
    import phase4_machine_learning_enhanced  # noqa: F401  (its import-time style is overridden next)
    apply_style()

    print('Figure S2 ...')
    figure_s2(df)
    print('Figure S3 ...')
    figure_s3(setup, pdat)
    print('Figure S4 ...')
    figure_s4(setup)
    print('Figure S5 ...')
    figure_s5()
    print('Figure S6 ...')
    figure_s6()
    print('Figure S7 ...')
    figure_s7()

    VALUES['_meta'] = {'script': 'r2_figures_supp.py',
                       'figure_dir': str(FIG_DIR.relative_to(REPO_ROOT)).replace('\\', '/'),
                       'feature_ablation_available': (R2 / 'feature_ablation.json').exists()}
    VALUES_PATH.write_text(json.dumps(to_jsonable(VALUES), indent=1))
    print(f'  wrote {VALUES_PATH}')


if __name__ == '__main__':
    main()
