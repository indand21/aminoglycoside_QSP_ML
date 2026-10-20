#!/usr/bin/env python3
"""Revision 2: regenerate main-manuscript Figures 1, 3, 4, 5, 6, 7 and the
Graphical Abstract (AAPS Journal revision).

Every number, curve, point and annotation is computed at run time from saved
result files (no simulated or hard-coded result values). Only fixed design
constants appear in the code: PK/PD thresholds (Cmax/MIC 8; trough 2.0 mg/L
gentamicin, 2.5 mg/L amikacin), risk cutoffs (0.15/0.35 nephrotoxicity,
0.55/0.75 cure), CrCL categories (<50, 50-130, >130 mL/min) and bin counts.
Every annotated number is written to results/revision2/figure_values.json.

Inputs
    results/phase3_pkpd/pkpd_indices.csv, data/patient_data.csv
    results/revision2/table4_values.json
    results/phase4_ml_enhanced/test_predictions.json, algorithm_comparison.json
    results/revision2/shap_importance.json, shap_values_nephrotoxicity_postdose.csv
    results/revision2/oof_predictions.csv, risk_stratification.json
    results/revision2/dose_optimization_patients.csv, dose_optimization_summary.json
    production feature matrices via r2_ml_common.production_setup()

Outputs
    Manuscript_AAPS_Journal/Revision2_Submission/Figures/<name>.tiff (LZW, RGB)
    and <name>.png for Figure1_Framework_Overview (600 dpi),
    Figure3_Target_Attainment, Figure4_ML_Performance, Figure5_SHAP_Analysis,
    Figure6_Risk_Stratification, Figure7_Dose_Optimization, Graphical_Abstract
    (300 dpi); results/revision2/figure_values.json

Usage (from the repository root):
    python r2_figures_main.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve  # noqa: E402

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parent
RES = REPO / 'results'
R2 = RES / 'revision2'
FIG_DIR = REPO / 'Manuscript_AAPS_Journal' / 'Revision2_Submission' / 'Figures'
VALUES_JSON = R2 / 'figure_values.json'

# --------------------------------------------------------------------------- #
# Design constants (not results)
# --------------------------------------------------------------------------- #
CMAX_MIC_TARGET = 8.0
TROUGH = {'amikacin': 2.5, 'gentamicin': 2.0}
NEPH_CUTS = (0.15, 0.35)
CURE_CUTS = (0.55, 0.75)
RENAL_ORDER = ['Impaired (<50)', 'Normal (50-130)', 'Augmented (>130)']
RENAL_LABEL = {'Impaired (<50)': 'Impaired\n(<50)', 'Normal (50-130)': 'Normal\n(50–130)',
               'Augmented (>130)': 'Augmented\n(>130)'}
N_BINS = 10
OPTIM_WORD = 'optimization'      # American spelling, as in the manuscript text

# Okabe-Ito palette
OI = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73',
      'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7',
      'grey': '#999999', 'lightgrey': '#BBBBBB'}
DRUG_COL = {'amikacin': OI['blue'], 'gentamicin': OI['orange'], 'combined': '#555555'}
DRUG_MARK = {'amikacin': 'o', 'gentamicin': '^'}
MODEL_COL = {'nephrotoxicity_predose': OI['sky'], 'nephrotoxicity_postdose': OI['purple'],
             'clinical_cure': OI['green']}
MODEL_LABEL = {'nephrotoxicity_predose': 'Pre-dose nephrotoxicity',
               'nephrotoxicity_postdose': 'Post-dose nephrotoxicity',
               'clinical_cure': 'Post-dose clinical cure'}
NEPH_YES, NEPH_NO = OI['vermillion'], OI['lightgrey']
ZONE_COL = {'favourable': OI['green'], 'intermediate': OI['yellow'], 'unfavourable': OI['vermillion']}
ZONE_LABEL = {'favourable': 'Favorable', 'intermediate': 'Intermediate', 'unfavourable': 'Unfavorable'}

FEATURE_LABEL = {
    'Cmin': 'Cmin (trough)', 'k10_bayes': 'k10 (individual)', 'CL_bayes': 'CL (individual)',
    'CL_cv': 'CL CV', 'Vc_cv': 'Vc CV', 'age_x_baseline_scr': 'Age × SCr',
    'diabetes_x_baseline_crcl': 'Diabetes × CrCL', 't_half_beta': 't½β', 't_half_alpha': 't½α',
    'Vp_bayes': 'Vp (individual)', 'k21_bayes': 'k21 (individual)',
    'mechanical_ventilation': 'Mechanical ventilation', 'fluctuation_index': 'Fluctuation index',
    'baseline_albumin': 'Albumin',
    # additional readable names for features outside the requested map
    'age': 'Age', 'AUC24': 'AUC24', 'Cmax': 'Cmax', 'Cmax_MIC': 'Cmax/MIC', 'AUC_MIC': 'AUC/MIC',
    'baseline_crcl': 'CrCL', 'baseline_scr': 'SCr', 'baseline_egfr': 'eGFR', 'drug': 'Drug',
    'weight': 'Weight', 'height': 'Height', 'Vc_bayes': 'Vc (individual)', 'Q_bayes': 'Q (individual)',
    'k12_bayes': 'k12 (individual)', 'Vss_bayes': 'Vss (individual)', 'apache_ii': 'APACHE II',
    'sofa_score': 'SOFA', 'vasopressor_use': 'Vasopressor use', 'peak_trough_ratio': 'Peak/trough ratio',
    'time_above_MIC': 'Time above MIC', 'AUC_above_MIC': 'AUC above MIC',
}


def flabel(f):
    return FEATURE_LABEL.get(f, f.replace('_', ' '))


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def set_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = 'Arial' if 'Arial' in names else 'DejaVu Sans'
    plt.style.use('default')          # undo any theme set by imported modules
    plt.rcParams.update({'axes.grid': False,
        'font.family': 'sans-serif', 'font.sans-serif': [family, 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.title_fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
    })
    return family


# --------------------------------------------------------------------------- #
# Values registry and helpers
# --------------------------------------------------------------------------- #
VALUES: dict = {}
CHECKS: list = []


def rec(fig, panel, key, value):
    VALUES.setdefault(fig, {}).setdefault(panel, {})[key] = value


def check(label, a, b, tol=1e-6):
    ok = bool(np.isclose(a, b, atol=tol, rtol=0))
    CHECKS.append({'check': label, 'computed': float(a), 'reference': float(b), 'match': ok})
    if not ok:
        print(f'  [CHECK MISMATCH] {label}: computed {a} vs reference {b}')


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def fmt_pp(x, nd=1):
    s = f'{x:+.{nd}f}'
    return s.replace('-', '−')


def fmt_num(x, nd=1):
    return f'{x:.{nd}f}'.replace('-', '−')


def add_letters(fig, pairs, dx=0.0, dy=0.0):
    """Freeze the layout, then put bold panel letters at the top-left corner of
    each panel's tight bounding box (axis labels included)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fig.set_layout_engine('none')
    for ax, s in pairs:
        bb = ax.get_tightbbox(renderer).transformed(inv)
        fig.text(max(bb.x0 + dx, 0.004), min(bb.y1 + dy, 0.996), s, ha='left', va='top',
                 fontsize=11, fontweight='bold')


def save(fig, name, dpi):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{name}.png'
    tif = FIG_DIR / f'{name}.tiff'
    fig.savefig(png, dpi=dpi, facecolor='white')
    plt.close(fig)
    img = Image.open(png).convert('RGB')
    img.save(png, dpi=(dpi, dpi))
    img.save(tif, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))
    print(f'  wrote {tif.name} ({img.size[0]}x{img.size[1]} px, {dpi} dpi) + .png')
    return [str(tif), str(png)]


def reliability_bins(y, p, n_bins=N_BINS):
    """Equal-width bins identical to phase4 expected_calibration_error."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            rows.append((b, float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def swarm_offsets(v, nbins=60, width=0.40):
    """Deterministic beeswarm offsets: points sharing an x-bin are stacked
    alternately above and below the centre line (no random jitter)."""
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    edges = np.linspace(lo, hi if hi > lo else lo + 1, nbins + 1)
    b = np.clip(np.digitize(v, edges[1:-1]), 0, nbins - 1)
    off = np.zeros(len(v))
    counts = np.bincount(b, minlength=nbins)
    for k in np.nonzero(counts)[0]:
        ii = np.where(b == k)[0]
        ii = ii[np.argsort(v[ii], kind='stable')]
        j = np.arange(len(ii))
        off[ii] = np.where(j % 2 == 1, 1, -1) * ((j + 1) // 2)
    return off / (counts.max() / 2 + 1) * width


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_json(p):
    return json.loads(Path(p).read_text())


def load_pkpd():
    pk = pd.read_csv(RES / 'phase3_pkpd' / 'pkpd_indices.csv')
    pt = pd.read_csv(REPO / 'data' / 'patient_data.csv')
    df = pk.merge(pt[['patient_id', 'drug', 'baseline_crcl']], on='patient_id', how='left',
                  validate='one_to_one')
    df['neph'] = (df['nephrotoxicity'] > 0).astype(int)   # KDIGO stage >= 1
    df['renal'] = np.select([df['baseline_crcl'] < 50, df['baseline_crcl'] > 130],
                            ['Impaired (<50)', 'Augmented (>130)'], 'Normal (50-130)')
    return df


# =========================================================================== #
# Figure 1: framework overview (no numbers)
# =========================================================================== #
def figure1():
    W, H = 7.2, 5.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    bw, bh, gap = 2.05, 1.60, 0.375
    xs = [0.15, 0.15 + bw + gap, 0.15 + 2 * (bw + gap)]
    y_top, y_bot = 3.00, 0.85
    # snake layout: row 1 = Phases 1-3 (left to right), row 2 = Phases 6, 5, 4
    boxes = {
        1: (xs[0], y_top, OI['sky'], 'Data Preprocessing',
            ['• Synthetic ICU cohort', '• Quality control and', '   outlier detection',
             '• Covariate derivation', '   (CrCL, BMI, severity)']),
        2: (xs[1], y_top, OI['blue'], 'Population PK',
            ['• Bayesian two-compartment', '   PopPK model', '• Covariate model',
             '   (CrCL, weight, age)', '• Individual PK parameters']),
        3: (xs[2], y_top, OI['green'], 'PK/PD Analysis',
            ['• Exposure indices', '   (Cmax, Cmin, AUC24)', '• Probability of target',
             '   attainment (PTA)', '• Cumulative fraction of', '   response (CFR)']),
        4: (xs[2], y_bot, OI['purple'], 'Machine Learning',
            ['ML development steps 1–4:', '1. Baseline models', '2. Engineered features',
             '3. Algorithm comparison', '4. Interpretability (SHAP)', '    & decision support']),
        5: (xs[1], y_bot, OI['orange'], 'Dose Optimization',
            ['• Grid search over drug-', '   specific dose grids', '• ML outcome surrogates',
             '• Ground-truth simulator', '   check']),
        6: (xs[0], y_bot, OI['grey'], 'Verification &\nCompletion',
            ['• Parameter recovery', '• Cross-validation and', '   calibration',
             '• Simulator (ground-truth)', '   check']),
    }
    for k, (x, y, col, title, body) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc='white', ec='#333333', lw=1.0, zorder=2))
        n_title = title.count('\n') + 1
        head_h = 0.22 + 0.19 * n_title
        ax.add_patch(FancyBboxPatch((x, y + bh - head_h), bw, head_h,
                                    boxstyle='round,pad=0,rounding_size=0.08',
                                    fc=col, ec='#333333', lw=1.0, alpha=0.30, zorder=3))
        ax.text(x + bw / 2, y + bh - 0.08, f'Phase {k}\n{title}', ha='center', va='top',
                fontsize=11, fontweight='bold', linespacing=1.05, zorder=4)
        ax.text(x + 0.10, y + bh - head_h - 0.08, '\n'.join(body), ha='left', va='top',
                fontsize=9, linespacing=1.25, zorder=4)

    def mid(k):
        x, y = boxes[k][0], boxes[k][1]
        return x, y, x + bw, y + bh, x + bw / 2, y + bh / 2

    solid = dict(arrowstyle='-|>', lw=1.6, color='black', mutation_scale=14, shrinkA=0, shrinkB=0)
    for a, b in [(1, 2), (2, 3)]:
        _, _, ax1, _, _, ym = mid(a)
        bx0 = mid(b)[0]
        ax.annotate('', xy=(bx0, ym), xytext=(ax1, ym), arrowprops=solid, zorder=5)
    x3c = mid(3)[4]
    ax.annotate('', xy=(x3c, y_bot + bh), xytext=(x3c, y_top), arrowprops=solid, zorder=5)
    _, _, _, _, _, ym2 = mid(4)
    ax.annotate('', xy=(xs[1] + bw, ym2), xytext=(xs[2], ym2), arrowprops=solid, zorder=5)

    dcol = '#444444'

    def dashed_path(pts, label, lab_xy):
        px, py = zip(*pts)
        ax.plot(px[:-1] + (px[-1],), py[:-1] + (py[-1] - 0.0,), ls=(0, (4, 3)), lw=1.3,
                color=dcol, zorder=1, solid_capstyle='butt')
        # solid arrow head on the final segment
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ux, uy = np.sign(x1 - x0), np.sign(y1 - y0)
        ax.annotate('', xy=(x1, y1), xytext=(x1 - 0.12 * ux, y1 - 0.12 * uy),
                    arrowprops=dict(arrowstyle='-|>', lw=1.3, color=dcol, mutation_scale=12,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.text(*lab_xy, label, ha='center', va='center', fontsize=9, style='italic',
                color='#222222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=dcol, lw=0.6))

    y_gap = (y_top + y_bot + bh) / 2
    dashed_path([(xs[1] + 1.65, y_top), (xs[1] + 1.65, y_gap), (xs[0] + 0.45, y_gap),
                 (xs[0] + 0.45, y_bot + bh)],
                'Parameter recovery check', ((xs[0] + 0.45 + xs[1] + 1.65) / 2 + 0.15, y_gap))
    dashed_path([(xs[1] + 1.45, y_bot), (xs[1] + 1.45, 0.55), (xs[0] + 1.45, 0.55),
                 (xs[0] + 1.45, y_bot)],
                'Simulator (ground-truth) check', ((xs[0] + xs[1] + 2.9) / 2, 0.55))
    dashed_path([(xs[2] + 1.0, y_bot), (xs[2] + 1.0, 0.22), (xs[0] + 0.65, 0.22),
                 (xs[0] + 0.65, y_bot)],
                'Cross-validation / calibration', ((xs[0] + 0.65 + xs[2] + 1.0) / 2, 0.22))

    # legend
    ly = 4.85
    ax.annotate('', xy=(0.75, ly), xytext=(0.2, ly), arrowprops=solid)
    ax.text(0.85, ly, 'Data flow', va='center', fontsize=9)
    ax.plot([2.2, 2.68], [ly, ly], ls=(0, (4, 3)), lw=1.3, color=dcol)
    ax.annotate('', xy=(2.8, ly), xytext=(2.68, ly),
                arrowprops=dict(arrowstyle='-|>', lw=1.3, color=dcol, mutation_scale=12,
                                shrinkA=0, shrinkB=0))
    ax.text(2.9, ly, 'Verification check (feeds Phase 6)', va='center', fontsize=9)
    VALUES['Figure1'] = {'note': 'schematic; contains no result numbers'}
    return save(fig, 'Figure1_Framework_Overview', 600)


# =========================================================================== #
# Figure 3: PK/PD target attainment
# =========================================================================== #
def figure3(pkpd, t4):
    F = 'Figure3'
    fig = plt.figure(figsize=(7.0, 6.1), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    endpoints = [('cmax_mic_8', 'Efficacy\n(Cmax/MIC\n≥ 8)'),
                 ('cmin_drug', 'Trough\nsafety\n(drug-specific)'),
                 ('combined', 'Combined\n(both)')]
    ep_legend = {'cmax_mic_8': 'Efficacy (Cmax/MIC ≥ 8)',
                 'cmin_drug': 'Trough safety (drug-specific)', 'combined': 'Combined (both)'}
    groups = [('combined', 'Both drugs'), ('amikacin', 'Amikacin'), ('gentamicin', 'Gentamicin')]

    # cross-check Table 4A against the per-patient file
    for g, _ in groups:
        sub = pkpd if g == 'combined' else pkpd[pkpd['drug'] == g]
        tl = sub['drug'].map(TROUGH)
        check(f'F3A {g} n', len(sub), t4['A'][g]['n'])
        check(f'F3A {g} cmax_mic_8 pct', round(100 * (sub['Cmax_MIC'] >= 8).mean(), 1),
              t4['A'][g]['cmax_mic_8'][1])
        check(f'F3A {g} cmin_drug pct', round(100 * (sub['Cmin'] < tl).mean(), 1),
              t4['A'][g]['cmin_drug'][1])

    # A
    w = 0.26
    x = np.arange(len(endpoints))
    for j, (g, glab) in enumerate(groups):
        vals = [t4['A'][g][e][1] for e, _ in endpoints]
        n = t4['A'][g]['n']
        bars = axA.bar(x + (j - 1) * w, vals, w, color=DRUG_COL[g], label=f'{glab} (n = {n:,})',
                       edgecolor='white', lw=0.5)
        for bx, v in zip(bars, vals):
            axA.text(bx.get_x() + w / 2, v + 1.5, f'{v:.1f}', ha='center', va='bottom',
                     rotation=90, fontsize=8)
        for (e, _), v in zip(endpoints, vals):
            rec(F, 'A', f'{g}_{e}_pct', v)
            rec(F, 'A', f'{g}_{e}_n', t4['A'][g][e][0])
        rec(F, 'A', f'{g}_N', n)
    axA.set_xticks(x, [lab for _, lab in endpoints])
    axA.set_ylim(0, 138)
    axA.set_yticks(range(0, 101, 20))
    axA.set_ylabel('Patients attaining target (%)')
    axA.set_title('Target attainment by drug')
    axA.legend(loc='upper left', frameon=False, ncol=1, borderaxespad=0.1, handlelength=1.2)

    # B
    ep_col = [OI['green'], OI['sky'], OI['black']]
    xb = np.arange(len(RENAL_ORDER))
    for j, ((e, elab), col) in enumerate(zip(endpoints, ep_col)):
        vals = [t4['B'][rg]['combined'][e][1] for rg in RENAL_ORDER]
        bars = axB.bar(xb + (j - 1) * w, vals, w, color=col, edgecolor='white', lw=0.5,
                       label=ep_legend[e])
        for bx, v in zip(bars, vals):
            axB.text(bx.get_x() + w / 2, v + 1.5, f'{v:.1f}', ha='center', va='bottom',
                     rotation=90, fontsize=8)
        for rg, v in zip(RENAL_ORDER, vals):
            rec(F, 'B', f'{rg}_{e}_pct', v)
    ticks = []
    for rg in RENAL_ORDER:
        n = t4['B'][rg]['combined']['n']
        rec(F, 'B', f'{rg}_n', n)
        ticks.append(f'{RENAL_LABEL[rg]}\nn = {n:,}')
        check(f'F3B {rg} n', (pkpd['renal'] == rg).sum(), n)
    axB.set_xticks(xb, ticks)
    axB.set_xlabel('Renal function (CrCL, mL/min)')
    axB.set_ylim(0, 130)
    axB.set_yticks(range(0, 101, 20))
    axB.set_ylabel('Patients attaining target (%)')
    axB.set_title('By renal function (both drugs)')
    axB.legend(loc='upper left', frameon=False, ncol=1, borderaxespad=0.1, handlelength=1.2)

    # C
    cm = pkpd['Cmax_MIC']
    edges = np.logspace(np.log10(cm.min()), np.log10(cm.max()), 41)
    for g in ['amikacin', 'gentamicin']:
        v = pkpd.loc[pkpd['drug'] == g, 'Cmax_MIC']
        pct = 100 * (v >= CMAX_MIC_TARGET).mean()
        axC.hist(v, bins=edges, color=DRUG_COL[g], alpha=0.45, edgecolor=DRUG_COL[g], lw=0.8,
                 histtype='stepfilled',
                 label=f'{g.capitalize()} (n = {len(v):,}; {pct:.1f}% ≥ 8)')
        rec(F, 'C', f'{g}_n', int(len(v)))
        rec(F, 'C', f'{g}_pct_ge8', r(pct, 1))
        rec(F, 'C', f'{g}_median_cmax_mic', r(v.median(), 2))
    axC.axvline(CMAX_MIC_TARGET, color='black', ls='--', lw=1.1)
    ymaxC = axC.get_ylim()[1]
    axC.text(CMAX_MIC_TARGET * 1.12, ymaxC * 1.05, 'Cmax/MIC = 8', fontsize=8, va='bottom')
    axC.set_xscale('log')
    axC.set_xlabel('Cmax/MIC (log scale)')
    axC.set_ylabel('Patients')
    axC.set_title('Cmax/MIC distribution')
    axC.legend(loc='upper left', frameon=True, framealpha=1, edgecolor='none',
               borderaxespad=0.1, handlelength=1.2)
    axC.set_ylim(0, ymaxC * 1.45)

    # D
    for g in ['amikacin', 'gentamicin']:
        for yv, col, z in [(0, NEPH_NO, 1), (1, NEPH_YES, 2)]:
            s = pkpd[(pkpd['drug'] == g) & (pkpd['neph'] == yv)]
            axD.scatter(s['Cmax_MIC'], s['Cmin'], s=7, marker=DRUG_MARK[g], color=col,
                        alpha=0.65, lw=0, zorder=z)
    axD.set_xscale('log')
    axD.set_yscale('log')
    for g in ['amikacin', 'gentamicin']:
        axD.axhline(TROUGH[g], color=DRUG_COL[g], ls='--', lw=1.2, zorder=3)
    axD.axvline(CMAX_MIC_TARGET, color='black', ls=':', lw=1.2, zorder=3)
    n_yes, n_no = int(pkpd['neph'].sum()), int((pkpd['neph'] == 0).sum())
    rec(F, 'D', 'n_nephrotoxicity', n_yes)
    rec(F, 'D', 'n_no_nephrotoxicity', n_no)
    rec(F, 'D', 'n_cmin_at_floor_0.1', int((pkpd['Cmin'] <= 0.1).sum()))
    handles = [Line2D([], [], ls='', marker='s', color=NEPH_YES, label=f'Nephrotoxicity (n = {n_yes:,})'),
               Line2D([], [], ls='', marker='s', color=NEPH_NO, label=f'No nephrotoxicity (n = {n_no:,})'),
               Line2D([], [], ls='', marker='o', mfc='none', color='#333333', label='Amikacin'),
               Line2D([], [], ls='', marker='^', mfc='none', color='#333333', label='Gentamicin'),
               Line2D([], [], ls='--', color=DRUG_COL['amikacin'], label='Trough 2.5 mg/L (amikacin)'),
               Line2D([], [], ls='--', color=DRUG_COL['gentamicin'], label='Trough 2.0 mg/L (gentamicin)'),
               Line2D([], [], ls=':', color='black', label='Cmax/MIC = 8')]
    axD.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False,
               borderaxespad=0, handlelength=1.6)
    axD.set_xlabel('Cmax/MIC (log scale)')
    axD.set_ylabel('Cmin (mg/L, log scale)')
    axD.set_title('Exposure vs observed nephrotoxicity')

    add_letters(fig, [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')])
    return save(fig, 'Figure3_Target_Attainment', 300)


# =========================================================================== #
# Figure 4: ML performance (held-out N = 300)
# =========================================================================== #
def figure4(setup, testp, algo):
    from phase4_machine_learning_enhanced import expected_calibration_error
    from r2_ml_analyses import boot_auc_ci

    F = 'Figure4'
    fig = plt.figure(figsize=(7.0, 6.4), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    gsC = gs[1, 0].subgridspec(2, 1, height_ratios=[2.0, 1.15], hspace=0.08)
    axC1, axC2 = fig.add_subplot(gsC[0]), fig.add_subplot(gsC[1])
    axD = fig.add_subplot(gs[1, 1])

    models = ['nephrotoxicity_predose', 'nephrotoxicity_postdose', 'clinical_cure']
    res = {}
    for m in models:
        y = np.asarray(testp[m]['y_test'], int)
        p = np.asarray(testp[m]['y_pred_proba'], float)
        auc = roc_auc_score(y, p)
        lo, hi = boot_auc_ci(y, p)
        nf = len(setup[m]['features'])
        res[m] = dict(y=y, p=p, auc=auc, lo=lo, hi=hi, nf=nf)
        rec(F, 'A', f'{m}_auroc', r(auc))
        rec(F, 'A', f'{m}_ci95', [r(lo), r(hi)])
        rec(F, 'A', f'{m}_n_features', nf)
        rec(F, 'A', f'{m}_n_test', int(len(y)))
        rec(F, 'A', f'{m}_events_test', int(y.sum()))
        check(f'F4 {m} held-out y equals production split',
              float(np.array_equal(y, setup[m]['y'][setup[m]['test']])), 1.0)

    # A
    short = {'nephrotoxicity_predose': 'Pre-dose\nnephrotoxicity',
             'nephrotoxicity_postdose': 'Post-dose\nnephrotoxicity',
             'clinical_cure': 'Post-dose\nclinical cure'}
    for i, m in enumerate(models):
        d = res[m]
        axA.errorbar(i, d['auc'], yerr=[[d['auc'] - d['lo']], [d['hi'] - d['auc']]], fmt='o',
                     color=MODEL_COL[m], ms=7, capsize=4, lw=1.6, mec='black', mew=0.5)
        axA.text(i + 0.12, d['auc'], f"{d['auc']:.3f}\n({d['lo']:.3f}–{d['hi']:.3f})",
                 va='center', ha='left', fontsize=8)
    axA.set_xticks(range(3), [f"{short[m]}\n({res[m]['nf']} features)" for m in models])
    axA.set_xlim(-0.4, 2.85)
    axA.set_ylim(0.5, 1.0)
    axA.axhline(0.5, color='#999999', lw=0.8, ls=':')
    axA.set_ylabel('AUROC (95% bootstrap CI)')
    axA.set_title(f"Discrimination, held-out test set (N = {len(res[models[0]]['y'])})")

    # B
    for m in models:
        d = res[m]
        fpr, tpr, _ = roc_curve(d['y'], d['p'])
        axB.plot(fpr, tpr, color=MODEL_COL[m], lw=1.8,
                 label=f"{MODEL_LABEL[m]}\n(AUROC {d['auc']:.3f})")
    axB.plot([0, 1], [0, 1], color='#999999', ls=':', lw=0.9)
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1.01)
    axB.set_aspect('equal')
    axB.set_xlabel('1 − Specificity')
    axB.set_ylabel('Sensitivity')
    axB.set_title('ROC curves (held-out)')
    axB.legend(loc='lower right', frameon=False, borderaxespad=0.2, handlelength=1.4,
               labelspacing=0.6)

    # C
    fw_order = ['xgboost', 'lightgbm', 'catboost', 'stacking']
    fw_lab = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
              'stacking': 'Stacking'}
    fws = [f for f in fw_order if f in algo['frameworks']]
    aucs = [algo['frameworks'][f]['test_auc'] for f in fws]
    briers = [algo['frameworks'][f]['brier_score'] for f in fws]
    xc = np.arange(len(fws))
    axC1.plot(xc, aucs, 'o', color=MODEL_COL['nephrotoxicity_postdose'], ms=7, mec='black', mew=0.5)
    for xi, a in zip(xc, aucs):
        axC1.text(xi, a + 0.012, f'{a:.3f}', ha='center', va='bottom', fontsize=8)
    axC1.set_ylim(0.80, 1.0)
    axC1.set_ylabel('AUROC')
    axC1.set_xticks(xc, [])
    axC1.set_xlim(-0.5, len(fws) - 0.5)
    axC1.set_title('Similar AUROC point estimates\n'
                   f"(post-dose nephrotoxicity, {algo['n_features']} features, "
                   f"{algo['n_trials']} Optuna trials)", fontsize=9)
    axC2.bar(xc, briers, 0.55, color='#BBBBBB', edgecolor='#555555', lw=0.6)
    for xi, b in zip(xc, briers):
        axC2.text(xi, b + 0.004, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
    axC2.set_ylim(0, max(briers) * 1.45)
    axC2.set_ylabel('Brier')
    axC2.set_xticks(xc, [fw_lab[f] for f in fws])
    axC2.set_xlim(-0.5, len(fws) - 0.5)
    for f, a, b in zip(fws, aucs, briers):
        rec(F, 'C', f'{f}_auroc', r(a))
        rec(F, 'C', f'{f}_brier', r(b))
    rec(F, 'C', 'n_features', algo['n_features'])
    rec(F, 'C', 'n_trials', algo['n_trials'])

    # D
    axD.plot([0, 1], [0, 1], color='#999999', ls=':', lw=0.9)
    nmax = 0
    for m in ['nephrotoxicity_postdose', 'clinical_cure']:
        d = res[m]
        rows = reliability_bins(d['y'], d['p'])
        nmax = max(nmax, max(rw[3] for rw in rows))
    for m, mk in [('nephrotoxicity_postdose', 'o'), ('clinical_cure', 's')]:
        d = res[m]
        rows = reliability_bins(d['y'], d['p'])
        brier = brier_score_loss(d['y'], d['p'])
        ece = expected_calibration_error(d['y'], d['p'], n_bins=N_BINS)
        px = [rw[1] for rw in rows]
        py = [rw[2] for rw in rows]
        sz = [12 + 260 * rw[3] / nmax for rw in rows]
        axD.plot(px, py, '-', color=MODEL_COL[m], lw=1.2, zorder=2)
        axD.scatter(px, py, s=sz, marker=mk, color=MODEL_COL[m], edgecolor='black', lw=0.5,
                    zorder=3, alpha=0.9,
                    label=f'{MODEL_LABEL[m]}\nBrier {brier:.3f}; ECE {ece:.3f}')
        rec(F, 'D', f'{m}_brier', r(brier))
        rec(F, 'D', f'{m}_ece', r(ece))
        rec(F, 'D', f'{m}_bin_counts', [rw[3] for rw in rows])
    axD.set_xlim(-0.05, 1.07)
    axD.set_ylim(-0.05, 1.07)
    axD.set_aspect('equal')
    axD.set_xlabel('Mean predicted probability')
    axD.set_ylabel('Observed event rate')
    axD.set_title('Calibration (10 equal-width bins)')
    lg = axD.legend(loc='lower right', frameon=False, borderaxespad=0.2, labelspacing=0.8,
                    markerscale=0.6)
    for h in lg.legend_handles:
        h.set_sizes([40])
    axD.text(0.02, 0.98, 'Marker area proportional\nto patients per bin',
             transform=axD.transAxes, ha='left', va='top', fontsize=8, color='#444444')

    add_letters(fig, [(axA, 'A'), (axB, 'B'), (axC1, 'C'), (axD, 'D')])
    return save(fig, 'Figure4_ML_Performance', 300)


# =========================================================================== #
# Figure 5: SHAP (post-dose nephrotoxicity)
# =========================================================================== #
def figure5(setup, pipe, pkpd, shap_imp):
    F = 'Figure5'
    m = 'nephrotoxicity_postdose'
    e = setup[m]
    X = e['X']
    sv = pd.read_csv(R2 / f'shap_values_{m}.csv')
    assert (sv['patient_id'].to_numpy() == e['patient_id']).all(), 'SHAP rows not aligned'
    assert [c for c in sv.columns if c != 'patient_id'] == list(X.columns)
    drug = pipe.data['drug'].to_numpy()
    assert (pipe.data['patient_id'].to_numpy() == e['patient_id']).all()
    imputed = pkpd.set_index('patient_id').loc[e['patient_id'], 'CL_bayes'].isna().to_numpy()

    imp = sorted(shap_imp[m], key=lambda d: d['shap_rank'])
    # verify shap_pct against the SHAP matrix
    mabs = sv.drop(columns='patient_id').abs().mean()
    tot = mabs.sum()
    for d in imp[:15]:
        check(f"F5A shap_pct {d['feature']}", 100 * mabs[d['feature']] / tot, d['shap_pct'], tol=1e-3)

    fig = plt.figure(figsize=(7.0, 7.6), layout='constrained')
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0])
    gtop = gs[0, :].subgridspec(1, 2, width_ratios=[1.0, 1.12])
    axA = fig.add_subplot(gtop[0])
    axB = fig.add_subplot(gtop[1])
    axC, axD, axE = (fig.add_subplot(gs[1, i]) for i in range(3))

    # A
    top15 = imp[:15]
    labs = [flabel(d['feature']) for d in top15][::-1]
    vals = [d['shap_pct'] for d in top15][::-1]
    yy = np.arange(len(vals))
    axA.barh(yy, vals, color=MODEL_COL[m], edgecolor='black', lw=0.4, height=0.7)
    for yi, v in zip(yy, vals):
        axA.text(v + 0.4, yi, f'{v:.1f}', va='center', fontsize=8)
    axA.set_yticks(yy, labs)
    axA.set_xlim(0, max(vals) * 1.18)
    axA.set_xlabel('Mean |SHAP| (% of total)')
    axA.set_title('Top-15 features')
    for d in top15:
        rec(F, 'A', d['feature'], r(d['shap_pct'], 1))
    rec(F, 'A', 'top15_total_pct', r(sum(d['shap_pct'] for d in top15), 1))

    # B
    top10 = [d['feature'] for d in imp[:10]]
    cmap = plt.get_cmap('viridis')
    for i, f in enumerate(top10[::-1]):
        s = sv[f].to_numpy()
        xv = X[f].to_numpy()
        pct = stats.rankdata(xv) / len(xv) * 100
        off = swarm_offsets(s)
        sc = axB.scatter(s, i + off, c=pct, cmap=cmap, vmin=0, vmax=100, s=3.5, lw=0,
                         rasterized=True)
    axB.axvline(0, color='#555555', lw=0.8)
    axB.set_yticks(range(10), [flabel(f) for f in top10[::-1]])
    axB.set_ylim(-0.6, 9.6)
    axB.set_xlabel('SHAP value (log-odds)')
    axB.set_title('Top-10 features: per-patient SHAP')
    cb = fig.colorbar(sc, ax=axB, pad=0.01, aspect=30, fraction=0.05)
    cb.set_label('Feature value (percentile)', fontsize=9)
    cb.set_ticks([0, 50, 100], labels=['Low', '50', 'High'])
    cb.ax.tick_params(labelsize=8)

    # C, D, E dependence plots
    def dep(ax, f, xlab, title, logx=False, mark_imputed=False):
        s = sv[f].to_numpy()
        xv = X[f].to_numpy()
        for g in ['amikacin', 'gentamicin']:
            mk = (drug == g) & (~imputed if mark_imputed else True)
            ax.scatter(xv[mk], s[mk], s=6, color=DRUG_COL[g], alpha=0.6, lw=0,
                       marker=DRUG_MARK[g], label=f'{g.capitalize()} (n = {int(mk.sum()):,})',
                       rasterized=True)
        if mark_imputed:
            ax.scatter(xv[imputed], s[imputed], s=14, facecolor='none', edgecolor='black', lw=0.6,
                       marker='s', label=f'Median-imputed (n = {int(imputed.sum())})')
        ax.axhline(0, color='#555555', lw=0.8)
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xlab)
        ax.set_ylabel(f'SHAP value: {flabel(f)}')
        ax.set_title(title)

    dep(axC, 'Cmin', 'Cmin (mg/L, log scale)', 'Trough concentration', logx=True)
    for g in ['amikacin', 'gentamicin']:
        axC.axvline(TROUGH[g], color=DRUG_COL[g], ls='--', lw=1.1)
    hC, lC = axC.get_legend_handles_labels()
    hC += [Line2D([], [], color=DRUG_COL['gentamicin'], ls='--'),
           Line2D([], [], color=DRUG_COL['amikacin'], ls='--')]
    lC += ['2.0 mg/L (gentamicin)', '2.5 mg/L (amikacin)']
    axC.legend(hC, lC, loc='upper left', frameon=True, framealpha=1, edgecolor='none',
               borderaxespad=0.1, handlelength=1.3, markerscale=1.8)
    dep(axD, 'CL_bayes', 'Individual CL (L/h)', 'Individual clearance', mark_imputed=True)
    axD.legend(loc='upper right', frameon=False, borderaxespad=0.1, handlelength=1.0,
               markerscale=1.5)
    dep(axE, 'AUC24', 'AUC24 (mg·h/L)', 'Daily exposure')
    axE.legend(loc='upper left', frameon=False, borderaxespad=0.1, handlelength=1.0,
               markerscale=1.8)
    for ax in (axC, axD, axE):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.45)
    rec(F, 'C', 'n_amikacin', int((drug == 'amikacin').sum()))
    rec(F, 'C', 'n_gentamicin', int((drug == 'gentamicin').sum()))
    rec(F, 'D', 'n_CL_median_imputed', int(imputed.sum()))
    rec(F, 'D', 'n_amikacin_non_imputed', int(((drug == 'amikacin') & ~imputed).sum()))
    rec(F, 'D', 'n_gentamicin_non_imputed', int(((drug == 'gentamicin') & ~imputed).sum()))
    rec(F, 'E', 'n_auc24_zero', int((X['AUC24'] == 0).sum()))
    add_letters(fig, [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D'), (axE, 'E')])
    return save(fig, 'Figure5_SHAP_Analysis', 300)


# =========================================================================== #
# Figure 6: risk stratification (OOF, N = 1,500)
# =========================================================================== #
def figure6(oof, risk):
    from phase4_machine_learning_enhanced import expected_calibration_error
    from r2_ml_analyses import wilson

    F = 'Figure6'
    fig = plt.figure(figsize=(7.0, 6.4), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    N = len(oof)

    p = oof['oof_nephrotoxicity_postdose'].to_numpy()
    y = oof['y_nephrotoxicity_postdose'].to_numpy().astype(int)

    # A
    edges = np.linspace(0, 1, 41)
    for yv, col, lab in [(0, NEPH_NO, 'No nephrotoxicity'), (1, NEPH_YES, 'Nephrotoxicity')]:
        n = int((y == yv).sum())
        axA.hist(p[y == yv], bins=edges, color=col, alpha=0.75 if yv == 0 else 0.6,
                 edgecolor='white' if yv == 0 else NEPH_YES, lw=0.4,
                 label=f'{lab} (n = {n:,})')
        rec(F, 'A', f'n_y{yv}', n)
    axA.set_yscale('log')
    for c in NEPH_CUTS:
        axA.axvline(c, color='black', ls='--', lw=1.0)
    ytop = axA.get_ylim()[1]
    axA.set_ylim(top=ytop * 30)
    axA.set_xlim(0, 1)
    axA.set_xticks([0, NEPH_CUTS[0], NEPH_CUTS[1], 0.6, 0.8, 1.0],
                   ['0', '0.15', '0.35', '0.6', '0.8', '1'])
    axA.set_xlabel('Out-of-fold P(nephrotoxicity), post-dose model')
    axA.set_ylabel('Patients (log scale)')
    axA.set_title(f'Predicted risk by outcome (N = {N:,})')
    axA.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='none',
               borderaxespad=0.1, handlelength=1.2)

    # B
    grp_names = ['Low', 'Medium', 'High']
    grp_tick = ['Low\n(<0.15)', 'Medium\n(0.15–0.35)', 'High\n(>0.35)']
    w = 0.36
    for j, (m, lab) in enumerate([('nephrotoxicity_postdose', 'Post-dose model'),
                                  ('nephrotoxicity_predose', 'Pre-dose model')]):
        groups = {g['group']: g for g in risk[m]['groups']}
        pm = oof[f'oof_{m}'].to_numpy()
        ym = oof[f'y_{m}'].to_numpy().astype(int)
        gi = np.digitize(pm, NEPH_CUTS)
        for k, gname in enumerate(grp_names):
            g = groups[gname]
            mk = gi == k
            check(f'F6B {m} {gname} n', mk.sum(), g['n'])
            check(f'F6B {m} {gname} events', ym[mk].sum(), g['events'])
            lo, hi = wilson(int(ym[mk].sum()), int(mk.sum()))
            check(f'F6B {m} {gname} wilson lo', lo, g['rate_ci'][0])
            xpos = k + (j - 0.5) * w
            rate = 100 * g['rate']
            axB.bar(xpos, rate, w, color=MODEL_COL[m], edgecolor='black', lw=0.4,
                    label=lab if k == 0 else None)
            axB.errorbar(xpos, rate, yerr=[[rate - 100 * g['rate_ci'][0]],
                                           [100 * g['rate_ci'][1] - rate]],
                         fmt='none', ecolor='black', capsize=3, lw=1.0)
            axB.text(xpos, 100 * g['rate_ci'][1] + 2, f"n = {g['n']}", rotation=90,
                     ha='center', va='bottom', fontsize=8)
            rec(F, 'B', f'{m}_{gname}_n', g['n'])
            rec(F, 'B', f'{m}_{gname}_events', g['events'])
            rec(F, 'B', f'{m}_{gname}_rate_pct', r(rate, 1))
            rec(F, 'B', f'{m}_{gname}_wilson95_pct', [r(100 * g['rate_ci'][0], 1),
                                                       r(100 * g['rate_ci'][1], 1)])
    axB.set_xticks(range(3), grp_tick)
    axB.set_xlabel('Predicted-risk group')
    axB.set_ylim(0, 112)
    axB.set_yticks(range(0, 101, 20))
    axB.set_ylabel('Observed nephrotoxicity (%)\n(Wilson 95% CI)')
    axB.set_title('Observed rate by risk group')
    axB.legend(loc='upper left', frameon=False, borderaxespad=0.1, handlelength=1.2)

    # C
    rows = reliability_bins(y, p)
    brier = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p, n_bins=N_BINS)
    auc = roc_auc_score(y, p)
    check('F6 OOF AUROC post-dose', auc, risk['nephrotoxicity_postdose']['oof_auroc'])
    nmax = max(rw[3] for rw in rows)
    axC.plot([0, 1], [0, 1], color='#999999', ls=':', lw=0.9)
    axC.plot([rw[1] for rw in rows], [rw[2] for rw in rows], '-', color=MODEL_COL['nephrotoxicity_postdose'],
             lw=1.2)
    axC.scatter([rw[1] for rw in rows], [rw[2] for rw in rows],
                s=[12 + 260 * rw[3] / nmax for rw in rows], color=MODEL_COL['nephrotoxicity_postdose'],
                edgecolor='black', lw=0.5, zorder=3)
    for rw in rows:
        size = 12 + 260 * rw[3] / nmax
        below = rw[2] > rw[1] and rw[2] > 0.2
        dy = (np.sqrt(size) / 2 + 8) * (-1 if below else 1)
        axC.annotate(f'{rw[3]}', (rw[1], rw[2]), xytext=(0, dy),
                     textcoords='offset points', ha='center', va='center', fontsize=8,
                     color='#333333')
    axC.text(0.03, 0.97, f'Brier {brier:.3f}\nECE {ece:.3f}\nAUROC {auc:.3f}',
             transform=axC.transAxes, ha='left', va='top', fontsize=8)
    axC.text(0.98, 0.03, 'Labels: patients per bin', transform=axC.transAxes, ha='right',
             va='bottom', fontsize=8, color='#444444')
    axC.set_xlim(-0.05, 1.07)
    axC.set_ylim(-0.05, 1.07)
    axC.set_aspect('equal')
    axC.set_xlabel('Mean predicted probability (OOF)')
    axC.set_ylabel('Observed nephrotoxicity rate')
    axC.set_title('Calibration of OOF post-dose predictions')
    rec(F, 'C', 'brier', r(brier))
    rec(F, 'C', 'ece', r(ece))
    rec(F, 'C', 'oof_auroc', r(auc))
    rec(F, 'C', 'bin_counts', [rw[3] for rw in rows])

    # D
    pc = oof['oof_clinical_cure'].to_numpy()
    zone = oof['action_zone'].to_numpy()
    z_counts = risk['action_zones']['counts']
    # recompute zones from the stated rule
    rule = np.where((p < NEPH_CUTS[0]) & (pc > CURE_CUTS[1]), 'favourable',
                    np.where((p > NEPH_CUTS[1]) | (pc < CURE_CUTS[0]), 'unfavourable', 'intermediate'))
    check('F6D zone rule reproduces stored zones', float((rule == zone).mean()), 1.0)
    for zname in ['unfavourable', 'intermediate', 'favourable']:
        mk = zone == zname
        check(f'F6D {zname} count', mk.sum(), z_counts[zname])
        axD.scatter(p[mk], pc[mk], s=6, color=ZONE_COL[zname], lw=0.2,
                    edgecolor='#444444' if zname == 'intermediate' else 'none', alpha=0.8,
                    label=f'{ZONE_LABEL[zname]} (n = {z_counts[zname]:,})', rasterized=True)
        rec(F, 'D', f'{zname}_n', z_counts[zname])
    for c in NEPH_CUTS:
        axD.axvline(c, color='black', ls='--', lw=0.9)
    for c in CURE_CUTS:
        axD.axhline(c, color='black', ls=':', lw=1.0)
    axD.set_xlim(-0.02, 1.02)
    axD.set_ylim(-0.02, 1.30)
    axD.set_xticks([0, NEPH_CUTS[0], NEPH_CUTS[1], 0.6, 0.8, 1.0],
                   ['0', '0.15', '0.35', '0.6', '0.8', '1'])
    axD.set_yticks([0, 0.2, 0.4, CURE_CUTS[0], CURE_CUTS[1], 1.0],
                   ['0', '0.2', '0.4', '0.55', '0.75', '1'])
    axD.set_xlabel('OOF P(nephrotoxicity), post-dose model')
    axD.set_ylabel('OOF P(clinical cure)')
    axD.set_title('Decision-support action zones')
    hd, ld = axD.get_legend_handles_labels()
    order = [ld.index(f'{ZONE_LABEL[z]} (n = {z_counts[z]:,})')
             for z in ['favourable', 'intermediate', 'unfavourable']]
    lgd = axD.legend([hd[i] for i in order], [ld[i] for i in order], loc='upper center',
                     ncol=2, frameon=True, framealpha=1, edgecolor='none', borderaxespad=0.1,
                     handlelength=1.0, columnspacing=1.0, markerscale=2.5)

    add_letters(fig, [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')])
    return save(fig, 'Figure6_Risk_Stratification', 300)


# =========================================================================== #
# Figure 7: grid-search dose optimization (N = 1,500)
# =========================================================================== #
def figure7(dp, dsum):
    F = 'Figure7'
    dp = dp.copy()
    dp['chg_pct'] = (dp['optimal_dose'] / dp['baseline_dose'] - 1) * 100
    N = len(dp)
    fig = plt.figure(figsize=(7.0, 6.6), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    gA = gs[0, 0].subgridspec(1, 2, wspace=0.05)
    axA1, axA2 = fig.add_subplot(gA[0]), fig.add_subplot(gA[1])
    axB = fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    fig.suptitle(f'Grid-search dose {OPTIM_WORD} (full cohort, N = {N:,}; '
                 'same patients at baseline and after optimization)', fontsize=10,
                 fontweight='bold')

    # A
    for ax, g in [(axA1, 'amikacin'), (axA2, 'gentamicin')]:
        sub = dp[dp['drug'] == g]
        data = [sub['baseline_mgkg'].to_numpy(), sub['optimal_mgkg'].to_numpy()]
        vp = ax.violinplot(data, positions=[0, 1], widths=0.8, showextrema=False)
        for body, a in zip(vp['bodies'], [0.25, 0.6]):
            body.set_facecolor(DRUG_COL[g])
            body.set_edgecolor(DRUG_COL[g])
            body.set_alpha(a)
        ax.boxplot(data, positions=[0, 1], widths=0.16, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='white', lw=0.8), medianprops=dict(color='black', lw=1.2),
                   whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
        mb, mo = data[0].mean(), data[1].mean()
        check(f'F7A {g} baseline mean mg/kg', mb, dsum[g]['baseline_mgkg'][0])
        check(f'F7A {g} optimal mean mg/kg', mo, dsum[g]['optimal_mgkg'][0])
        ax.set_xticks([0, 1], ['Baseline', 'Optimized'])
        ax.set_title(f'{g.capitalize()} (n = {len(sub)})', fontsize=9)
        ytop = max(d.max() for d in data)
        ax.set_ylim(0, ytop * 1.22)
        ax.text(0.5, 0.985, f'Mean {mb:.1f} → {mo:.1f} mg/kg', transform=ax.transAxes,
                ha='center', va='top', fontsize=8)
        rec(F, 'A', f'{g}_n', int(len(sub)))
        rec(F, 'A', f'{g}_baseline_mean_mgkg', r(mb, 1))
        rec(F, 'A', f'{g}_optimized_mean_mgkg', r(mo, 1))
        rec(F, 'A', f'{g}_baseline_median_mgkg', r(np.median(data[0]), 1))
        rec(F, 'A', f'{g}_optimized_median_mgkg', r(np.median(data[1]), 1))
    axA1.set_ylabel('Dose (mg/kg)')

    # B
    outcomes = [('aki', 'P(nephrotoxicity)'), ('cure', 'P(cure)')]
    xlabels, pos = [], []
    w = 0.36
    k = 0
    for g in ['amikacin', 'gentamicin']:
        sub = dp[dp['drug'] == g]
        for o, olab in outcomes:
            for j, (kind, klab, col, hatch) in enumerate([('pred', 'ML-predicted', OI['sky'], None),
                                                         ('true', 'True (simulator)', OI['vermillion'], None)]):
                mean, lo, hi = dsum[g][f'{kind}_{o}_change_pp']
                recomputed = 100 * (sub[f'{kind}_{o}_opt'] - sub[f'{kind}_{o}_base']).mean()
                check(f'F7B {g} {kind}_{o}_change_pp', recomputed, mean)
                xp = k + (j - 0.5) * w
                axB.bar(xp, mean, w, color=col, edgecolor='black', lw=0.4,
                        label=klab if k == 0 else None)
                axB.errorbar(xp, mean, yerr=[[mean - lo], [hi - mean]], fmt='none', ecolor='black',
                             capsize=2.5, lw=0.9)
                ytxt = hi + 0.8 if mean >= 0 else lo - 0.8
                axB.text(xp, ytxt, fmt_pp(mean), ha='center', va='bottom' if mean >= 0 else 'top',
                         fontsize=8, rotation=90)
                rec(F, 'B', f'{g}_{kind}_{o}_change_pp', [r(mean, 2), r(lo, 2), r(hi, 2)])
            xlabels.append(f'{olab}\n{g.capitalize()}')
            pos.append(k)
            k += 1
    axB.axhline(0, color='black', lw=0.8)
    axB.axvline(1.5, color='#BBBBBB', lw=0.8, ls='-')
    axB.set_xticks(pos, [lab.replace('P(nephrotoxicity)', 'P(nephro-\ntoxicity)') for lab in xlabels])
    lo_all = min(dsum[g][f'{kd}_{o}_change_pp'][1] for g in ['amikacin', 'gentamicin']
                 for kd in ['pred', 'true'] for o in ['aki', 'cure'])
    hi_all = max(dsum[g][f'{kd}_{o}_change_pp'][2] for g in ['amikacin', 'gentamicin']
                 for kd in ['pred', 'true'] for o in ['aki', 'cure'])
    axB.set_ylim(lo_all - 9, hi_all + 9)
    axB.set_ylabel('Mean change from baseline\n(percentage points, 95% CI)')
    axB.set_title('Change in outcome probabilities')
    axB.legend(loc='lower right', frameon=False, borderaxespad=0.1, handlelength=1.2)

    # C
    gap = 0.38
    centers = np.arange(len(RENAL_ORDER))
    for j, g in enumerate(['amikacin', 'gentamicin']):
        data, xp = [], []
        for i, rg in enumerate(RENAL_ORDER):
            v = dp.loc[(dp['drug'] == g) & (dp['renal_group'] == rg), 'chg_pct'].to_numpy()
            data.append(v)
            xp.append(centers[i] + (j - 0.5) * gap)
            rec(F, 'C', f'{g}_{rg}_n', int(len(v)))
            rec(F, 'C', f'{g}_{rg}_median_pct', r(np.median(v), 1))
        bp = axC.boxplot(data, positions=xp, widths=0.3, patch_artist=True, showfliers=True,
                         flierprops=dict(marker='o', ms=2, mfc=DRUG_COL[g], mec='none', alpha=0.6),
                         medianprops=dict(color='black', lw=1.2), whiskerprops=dict(lw=0.8),
                         capprops=dict(lw=0.8), boxprops=dict(lw=0.8))
        for b in bp['boxes']:
            b.set_facecolor(DRUG_COL[g])
            b.set_alpha(0.75)
        for x_, v in zip(xp, data):
            axC.text(x_, 1.0, f'{len(v)}', transform=axC.get_xaxis_transform(), ha='center',
                     va='top', fontsize=8)
    axC.text(-0.62, 1.0, 'n:', transform=axC.get_xaxis_transform(), ha='left', va='top',
             fontsize=8)
    axC.axhline(0, color='black', lw=0.9)
    axC.set_xticks(centers, [RENAL_LABEL[rg] for rg in RENAL_ORDER])
    axC.set_xlim(-0.65, len(RENAL_ORDER) - 0.35)
    lo, hi = axC.get_ylim()
    axC.set_ylim(lo, hi + (hi - lo) * 0.30)
    axC.set_xlabel('Renal function (CrCL, mL/min)')
    axC.set_ylabel('Dose change from baseline (%)')
    axC.set_title('Dose change by renal function', pad=14)
    axC.legend(handles=[Patch(color=DRUG_COL[g], alpha=0.75, label=g.capitalize())
                        for g in ['amikacin', 'gentamicin']],
               loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False,
               borderaxespad=0.1, handlelength=1.0)

    # D
    for g in ['amikacin', 'gentamicin']:
        sub = dp[dp['drug'] == g]
        rr = stats.pearsonr(sub['oracle_mgkg'], sub['optimal_mgkg'])[0]
        axD.scatter(sub['oracle_mgkg'], sub['optimal_mgkg'], s=7, marker=DRUG_MARK[g],
                    color=DRUG_COL[g], alpha=0.55, lw=0,
                    label=f'{g.capitalize()} (n = {len(sub)}), r = {rr:.2f}', rasterized=True)
        rec(F, 'D', f'{g}_pearson_r', r(rr, 2))
        rec(F, 'D', f'{g}_n', int(len(sub)))
    lim = [min(dp['oracle_mgkg'].min(), dp['optimal_mgkg'].min()) * 0.8,
           max(dp['oracle_mgkg'].max(), dp['optimal_mgkg'].max()) * 1.6]
    axD.plot(lim, lim, color='black', ls='--', lw=0.9, label='Identity')
    axD.set_xscale('log')
    axD.set_yscale('log')
    axD.set_xlim(lim)
    axD.set_ylim(lim[0], lim[1] * 8)    # headroom for the legend above the data
    axD.set_xlabel('Oracle-optimal dose (mg/kg, log scale)')
    axD.set_ylabel('ML-recommended dose\n(mg/kg, log scale)')
    axD.set_title('ML-recommended vs oracle-optimal dose')
    axD.legend(loc='upper left', frameon=False, borderaxespad=0.1, handlelength=1.2,
               markerscale=1.8)

    add_letters(fig, [(axA1, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')])
    return save(fig, 'Figure7_Dose_Optimization', 300)


# =========================================================================== #
# Graphical abstract
# =========================================================================== #
def graphical_abstract(pkpd, t4, testp, shap_imp, dp, dsum):
    F = 'Graphical_Abstract'
    W, H = 13.3, 5.5
    pal = {'cohort': '#1f6f8b', 'popk': '#21897e', 'pkpd': '#d68438', 'ml': '#a14a76',
           'optim': '#3b5f7a', 'ink': '#1a1a1a', 'subtle': '#4a5961', 'bg': '#f6f4ee',
           'band': '#eaf1f1', 'panel': '#ffffff'}
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis('off')
    bg.add_patch(plt.Rectangle((0, 0), W, H, color=pal['bg'], zorder=0))
    bg.add_patch(plt.Rectangle((0, H - 0.78), W, 0.78, color=pal['band'], zorder=1))
    bg.add_patch(plt.Rectangle((0, 0), W, 0.45, color=pal['band'], zorder=1))
    bg.text(W / 2, H - 0.28, 'An Integrated Bayesian PopPK – Interpretable ML Framework',
            ha='center', va='center', fontsize=16, fontweight='bold', color=pal['ink'], zorder=2)
    bg.text(W / 2, H - 0.60, 'for Aminoglycoside Precision Dosing in Critical Illness  ·  '
            'In-Silico Development & Evaluation', ha='center', va='center', fontsize=12,
            style='italic', color=pal['subtle'], zorder=2)

    N = len(pkpd)
    n_ami = int((pkpd['drug'] == 'amikacin').sum())
    pct_ami = 100 * n_ami / N
    bg.text(W / 2, 0.225, f'Synthetic ICU cohort (N = {N:,})  ·  Bayesian PopPK + PK/PD targets + '
            f'interpretable ML + grid-search dose {OPTIM_WORD}  ·  '
            'Prospective validation required before clinical use',
            ha='center', va='center', fontsize=10, color=pal['ink'], zorder=2)

    # computed values
    auc_pre = roc_auc_score(testp['nephrotoxicity_predose']['y_test'],
                            testp['nephrotoxicity_predose']['y_pred_proba'])
    auc_post = roc_auc_score(testp['nephrotoxicity_postdose']['y_test'],
                             testp['nephrotoxicity_postdose']['y_pred_proba'])
    renal_pct = {rg: 100 * (pkpd['renal'] == rg).mean() for rg in RENAL_ORDER}
    cl = pkpd['CL_bayes'].dropna()
    ami, gen = dsum['amikacin'], dsum['gentamicin']
    for g in ['amikacin', 'gentamicin']:
        sub = dp[dp['drug'] == g]
        check(f'GA {g} dose_change_pct', 100 * (sub['optimal_dose'] / sub['baseline_dose'] - 1).mean(),
              dsum[g]['dose_change_pct'][0])
    rec(F, 'cohort', 'N', N)
    rec(F, 'cohort', 'pct_amikacin', r(pct_ami, 0))
    rec(F, 'cohort', 'pct_gentamicin', r(100 - pct_ami, 0))
    rec(F, 'cohort', 'renal_pct', {k: r(v, 0) for k, v in renal_pct.items()})
    rec(F, 'popk', 'CL_median_L_per_h', r(cl.median(), 1))
    rec(F, 'popk', 'n_CL_estimated', int(len(cl)))
    rec(F, 'pkpd', 'cmax_mic_ge8_pct', {g: t4['A'][g]['cmax_mic_8'][1] for g in ['combined', 'amikacin', 'gentamicin']})
    rec(F, 'pkpd', 'combined_target_pct', t4['A']['combined']['combined'][1])
    rec(F, 'ml', 'predose_auroc_heldout', r(auc_pre))
    rec(F, 'ml', 'postdose_auroc_positive_control', r(auc_post))
    rec(F, 'optim', 'amikacin_dose_change_pct', [r(v, 1) for v in ami['dose_change_pct']])
    rec(F, 'optim', 'amikacin_true_aki_change_pp', [r(v, 1) for v in ami['true_aki_change_pp']])
    rec(F, 'optim', 'gentamicin_dose_change_pct', [r(v, 1) for v in gen['dose_change_pct']])
    rec(F, 'optim', 'gentamicin_true_cure_change_pp', [r(v, 1) for v in gen['true_cure_change_pp']])

    titles = ['Virtual ICU Cohort', 'Bayesian PopPK', 'PK/PD Target\nAttainment',
              'Interpretable ML', f'Grid-search Dose\n{OPTIM_WORD.capitalize()}']
    cols = [pal['cohort'], pal['popk'], pal['pkpd'], pal['ml'], pal['optim']]
    bullets = [
        [f'N = {N:,} synthetic ICU patients',
         f'{pct_ami:.0f}% amikacin · {100 - pct_ami:.0f}% gentamicin',
         f"CrCL <50 / 50–130 / >130 mL/min:\n{renal_pct['Impaired (<50)']:.0f}% / "
         f"{renal_pct['Normal (50-130)']:.0f}% / {renal_pct['Augmented (>130)']:.0f}%"],
        ['Two-compartment model',
         'Covariates: CrCL, weight, age',
         f'Individual CL: median {cl.median():.1f} L/h'],
        [f"Cmax/MIC ≥ 8: {t4['A']['combined']['cmax_mic_8'][1]:.1f}% overall",
         f"Amikacin {t4['A']['amikacin']['cmax_mic_8'][1]:.1f}%,\n"
         f"gentamicin {t4['A']['gentamicin']['cmax_mic_8'][1]:.1f}%",
         f"Efficacy + trough target:\n{t4['A']['combined']['combined'][1]:.1f}% of patients"],
        ['XGBoost + SHAP explanations',
         f'Pre-dose AUROC {auc_pre:.3f} (held-out)',
         f'Post-dose AUROC {auc_post:.3f}\n(positive control)'],
        ['Grid search over ML surrogates',
         f"Amikacin: dose {fmt_num(ami['dose_change_pct'][0], 0)}%,\n"
         f"true P(nephrotoxicity) {fmt_pp(ami['true_aki_change_pp'][0])} pp",
         f"Gentamicin: dose {fmt_pp(gen['dose_change_pct'][0], 0)}%,\n"
         f"true P(cure) {fmt_pp(gen['true_cure_change_pp'][0])} pp"],
    ]

    n = 5
    mx, gx = 0.22, 0.18
    pw = (W - 2 * mx - (n - 1) * gx) / n
    py0, ph = 0.62, 3.9
    for i in range(n):
        x0 = mx + i * (pw + gx)
        bg.add_patch(FancyBboxPatch((x0, py0), pw, ph, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc=pal['panel'], ec='#cdd4d8', lw=1.0, zorder=2))
        bg.add_patch(plt.Rectangle((x0 + 0.08, py0 + ph - 0.09), pw - 0.16, 0.05, color=cols[i],
                                   zorder=3))
        bg.add_patch(plt.Circle((x0 + 0.27, py0 + ph - 0.38), 0.17, color=cols[i], zorder=3))
        bg.text(x0 + 0.27, py0 + ph - 0.38, str(i + 1), ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', zorder=4)
        bg.text(x0 + 0.54, py0 + ph - 0.38, titles[i], ha='left', va='center',
                fontsize=11.5, fontweight='bold', color=pal['ink'], linespacing=1.0, zorder=4)
        # bullets
        yb = py0 + 1.30
        for b in bullets[i]:
            bg.text(x0 + 0.14, yb, '•', ha='left', va='top', fontsize=10, color=cols[i],
                    fontweight='bold', zorder=4)
            bg.text(x0 + 0.30, yb, b, ha='left', va='top', fontsize=9.5, color=pal['ink'],
                    linespacing=1.15, zorder=4)
            yb -= 0.27 + 0.19 * b.count('\n')
        if i < n - 1:
            bg.annotate('', xy=(x0 + pw + gx - 0.01, py0 + ph / 2 + 0.3),
                        xytext=(x0 + pw + 0.01, py0 + ph / 2 + 0.3),
                        arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#7d8a93',
                                        mutation_scale=14, shrinkA=0, shrinkB=0), zorder=5)

    def inset(i, left=0.42, right=0.15, bottom=1.98, top=0.82):
        x0 = mx + i * (pw + gx)
        rect = [(x0 + left) / W, (py0 + bottom) / H, (pw - left - right) / W,
                (ph - bottom - top) / H]
        a = fig.add_axes(rect)
        a.patch.set_alpha(0)
        for sp in ['top', 'right']:
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8.5)
        return a

    # 1 cohort: stacked bar of drug x renal group counts (computed)
    a = inset(0, left=0.55)
    bottom = np.zeros(2)
    renal_cols = ['#c9e3ef', '#6fb0cf', pal['cohort']]
    for rg, c in zip(RENAL_ORDER, renal_cols):
        cnt = np.array([((pkpd['drug'] == g) & (pkpd['renal'] == rg)).sum() for g in ['amikacin', 'gentamicin']])
        a.barh([1, 0], cnt, left=bottom, color=c, edgecolor='white', height=0.6,
               label=rg.split(' ')[0])
        bottom += cnt
    a.set_yticks([1, 0], ['Amikacin', 'Gentamicin'])
    a.set_xlabel('Patients', fontsize=9)
    a.legend(loc='upper center', bbox_to_anchor=(0.42, 1.04), ncol=3, frameon=False, fontsize=8.5,
             handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
    a.set_ylim(-0.5, 2.0)
    a.spines['left'].set_visible(False)

    # 2 PopPK: individual CL distribution by drug
    a = inset(1, left=0.45)
    edges = np.linspace(0, np.percentile(cl, 99.5), 30)
    for g in ['amikacin', 'gentamicin']:
        v = pkpd.loc[pkpd['drug'] == g, 'CL_bayes'].dropna()
        a.hist(v, bins=edges, color=DRUG_COL[g], alpha=0.5, histtype='stepfilled',
               edgecolor=DRUG_COL[g], label=g.capitalize())
    a.set_xlabel('Individual CL (L/h)', fontsize=9)
    a.set_yticks([])
    a.spines['left'].set_visible(False)
    a.set_ylim(0, a.get_ylim()[1] * 1.15)
    a.legend(frameon=False, fontsize=8.5, loc='upper right', handlelength=0.9)

    # 3 PK/PD: Cmax/MIC distribution with target line
    a = inset(2, left=0.35)
    cm = pkpd['Cmax_MIC']
    edges = np.logspace(np.log10(cm.min()), np.log10(cm.max()), 30)
    for g in ['amikacin', 'gentamicin']:
        a.hist(pkpd.loc[pkpd['drug'] == g, 'Cmax_MIC'], bins=edges, color=DRUG_COL[g], alpha=0.5,
               histtype='stepfilled', edgecolor=DRUG_COL[g], label=g.capitalize())
    a.set_xscale('log')
    a.axvline(CMAX_MIC_TARGET, color='black', ls='--', lw=1.2)
    yt = a.get_ylim()[1]
    a.set_ylim(0, yt * 1.45)
    a.text(CMAX_MIC_TARGET * 1.2, yt * 1.02, '8', fontsize=9, va='bottom')
    a.set_xlabel('Cmax/MIC (log)', fontsize=9)
    a.set_yticks([])
    a.spines['left'].set_visible(False)
    a.legend(frameon=False, fontsize=8.5, loc='upper left', handlelength=0.9, ncol=2,
             bbox_to_anchor=(0.0, 1.08), columnspacing=0.8)

    # 4 ML: top-5 SHAP features of the post-dose nephrotoxicity model
    a = inset(3, left=1.05)
    imp = sorted(shap_imp['nephrotoxicity_postdose'], key=lambda d: d['shap_rank'])[:5]
    vals = [d['shap_pct'] for d in imp][::-1]
    a.barh(range(5), vals, color=pal['ml'], height=0.65)
    a.set_yticks(range(5), [flabel(d['feature']) for d in imp][::-1], fontsize=8.5)
    a.set_xlabel('Mean |SHAP| (%)', fontsize=9)
    for k_, v in enumerate(vals):
        a.text(v + 0.5, k_, f'{v:.0f}', va='center', fontsize=8.5)
    a.set_xlim(0, max(vals) * 1.25)
    rec(F, 'ml', 'top5_shap_pct', {d['feature']: r(d['shap_pct'], 0) for d in imp})

    # 5 optimisation: true-simulator change by drug
    a = inset(4, left=0.5)
    xs_ = np.arange(2)
    for j, (o, lab, c) in enumerate([('aki', 'P(nephrotoxicity)', OI['vermillion']),
                                     ('cure', 'P(cure)', OI['green'])]):
        m_ = [dsum[g][f'true_{o}_change_pp'] for g in ['amikacin', 'gentamicin']]
        a.bar(xs_ + (j - 0.5) * 0.36, [v[0] for v in m_], 0.36, color=c, label=lab,
              yerr=[[v[0] - v[1] for v in m_], [v[2] - v[0] for v in m_]], capsize=2,
              error_kw=dict(lw=0.8))
    a.axhline(0, color='black', lw=0.8)
    a.set_xticks(xs_, ['Amikacin', 'Gentamicin'])
    a.set_ylabel('True Δ (pp)', fontsize=9)
    a.legend(frameon=False, fontsize=8.5, loc='lower right', handlelength=0.9)
    rec(F, 'optim', 'inset_true_change_pp', {g: {o: [r(v, 1) for v in dsum[g][f'true_{o}_change_pp']]
                                                 for o in ['aki', 'cure']} for g in ['amikacin', 'gentamicin']})

    out = save(fig, 'Graphical_Abstract', 300)
    return out


# =========================================================================== #
def main():
    from r2_ml_common import production_setup
    from phase4_machine_learning_enhanced import expected_calibration_error  # noqa: F401
    import r2_ml_analyses  # noqa: F401
    family = set_style()   # after imports: phase4 sets a seaborn theme on import
    print(f'Font: {family}')

    pkpd = load_pkpd()
    t4 = load_json(R2 / 'table4_values.json')
    testp = load_json(RES / 'phase4_ml_enhanced' / 'test_predictions.json')
    algo = load_json(RES / 'phase4_ml_enhanced' / 'algorithm_comparison.json')
    shap_imp = load_json(R2 / 'shap_importance.json')
    oof = pd.read_csv(R2 / 'oof_predictions.csv')
    risk = load_json(R2 / 'risk_stratification.json')
    dp = pd.read_csv(R2 / 'dose_optimization_patients.csv')
    dsum = load_json(R2 / 'dose_optimization_summary.json')
    print('Reproducing production feature matrices (r2_ml_common.production_setup)...')
    pipe, setup = production_setup()

    written = []
    print('Figure 1');  written += figure1()
    print('Figure 3');  written += figure3(pkpd, t4)
    print('Figure 4');  written += figure4(setup, testp, algo)
    print('Figure 5');  written += figure5(setup, pipe, pkpd, shap_imp)
    print('Figure 6');  written += figure6(oof, risk)
    print('Figure 7');  written += figure7(dp, dsum)
    print('Graphical abstract');  written += graphical_abstract(pkpd, t4, testp, shap_imp, dp, dsum)

    # data notes discovered while building the figures
    renal_dp = dp['renal_group'].value_counts().to_dict()
    renal_t4 = {rg: t4['B'][rg]['combined']['n'] for rg in RENAL_ORDER}
    VALUES['data_notes'] = {
        'renal_group_counts_dose_optimization': renal_dp,
        'renal_group_counts_table4': renal_t4,
        'n_crcl_exactly_50': int((pkpd['baseline_crcl'] == 50).sum()),
        'n_cmin_at_0.1_floor': int((pkpd['Cmin'] <= 0.1).sum()),
        'n_auc24_zero': int((pkpd['AUC24'] == 0).sum()),
        'n_CL_bayes_missing_in_pkpd_indices': int(pkpd['CL_bayes'].isna().sum()),
    }
    VALUES['checks'] = CHECKS
    VALUES['_meta'] = {'script': 'r2_figures_main.py', 'files': written}
    VALUES_JSON.write_text(json.dumps(VALUES, indent=2, ensure_ascii=False))
    n_bad = sum(not c['match'] for c in CHECKS)
    print(f'Consistency checks: {len(CHECKS) - n_bad}/{len(CHECKS)} match')
    print(f'Wrote {VALUES_JSON}')


if __name__ == '__main__':
    main()
