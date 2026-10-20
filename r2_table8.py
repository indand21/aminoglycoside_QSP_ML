#!/usr/bin/env python3
"""Revision 2: Table 8 (drug-specific baseline and optimised doses).

Reads results/revision2/dose_optimization_patients.csv (r2_dose_optimization.py)
and writes results/revision2/table8_values.json: for each drug, overall and by
dosing interval and by whether the efficacy target was attainable at the
administered dose (baseline Cmax/MIC >= 8), the baseline and optimised dose in
mg and mg/kg, the number of recommendations at the grid limits, and the
ML-predicted and true (simulator) changes in outcome probability.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from r2_dose_optimization import GRIDS
from r2_ml_common import OUT_DIR


def msd(x, nd=0):
    return [round(float(np.mean(x)), nd), round(float(np.std(x, ddof=1)), nd)]


def miqr(x, nd=1):
    q = np.percentile(x, [50, 25, 75])
    return [round(float(v), nd) for v in q]


def block(s, drug):
    lo, hi = GRIDS[drug][0], GRIDS[drug][-1]
    pct = 100 * (s['optimal_dose'] - s['baseline_dose']) / s['baseline_dose']
    return {
        'n': int(len(s)),
        'baseline_mg': msd(s['baseline_dose']), 'baseline_mgkg': msd(s['baseline_mgkg'], 1),
        'optimal_mg': msd(s['optimal_dose']), 'optimal_mgkg': msd(s['optimal_mgkg'], 1),
        'optimal_mgkg_median_iqr': miqr(s['optimal_mgkg']),
        'oracle_mgkg': msd(s['oracle_mgkg'], 1),
        'pct_change_mean': round(float(pct.mean()), 1), 'pct_change_median': round(float(pct.median()), 1),
        'n_reduced': int((s['optimal_dose'] < s['baseline_dose']).sum()),
        'n_unchanged': int((s['optimal_dose'] == s['baseline_dose']).sum()),
        'n_increased': int((s['optimal_dose'] > s['baseline_dose']).sum()),
        'n_at_grid_min': int((s['optimal_dose'] == lo).sum()),
        'n_at_grid_max': int((s['optimal_dose'] == hi).sum()),
        'pred_aki_change_pp': round(float(100 * (s['pred_aki_opt'] - s['pred_aki_base']).mean()), 1),
        'true_aki_change_pp': round(float(100 * (s['true_aki_opt'] - s['true_aki_base']).mean()), 1),
        'pred_cure_change_pp': round(float(100 * (s['pred_cure_opt'] - s['pred_cure_base']).mean()), 1),
        'true_cure_change_pp': round(float(100 * (s['true_cure_opt'] - s['true_cure_base']).mean()), 1),
        'pct_true_objective_worse': round(float(100 * (s['true_score_opt'] < s['true_score_base'] - 1e-12).mean()), 1),
    }


def main():
    d = pd.read_csv(OUT_DIR / 'dose_optimization_patients.csv')
    out = {'grids_mg': {k: [float(v[0]), float(v[-1]), float(v[1] - v[0])] for k, v in GRIDS.items()}}
    n_boundary = 0
    for drug in ['amikacin', 'gentamicin']:
        s = d[d['drug'] == drug]
        t = {'all': block(s, drug)}
        for h, g in s.groupby('interval_h'):
            t[f'q{int(h)}h'] = block(g, drug)
        t['target_attainable_at_baseline'] = block(s[s['cmax_mic_base'] >= 8], drug)
        t['target_not_attainable_at_baseline'] = block(s[s['cmax_mic_base'] < 8], drug)
        n_boundary += t['all']['n_at_grid_min'] + t['all']['n_at_grid_max']
        out[drug] = t
    out['n_at_grid_boundary_total'] = n_boundary
    out['pct_at_grid_boundary_total'] = round(100 * n_boundary / len(d), 1)
    (OUT_DIR / 'table8_values.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
