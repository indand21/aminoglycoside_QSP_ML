#!/usr/bin/env python3
"""Revision 2: one-at-a-time sensitivity sweep over the simulation assumptions
that have no published source (Supplementary Table S1).

Each assumption is moved, one at a time, to a low and a high value spanning a
plausible range, and five cohorts (the same seeds as the primary analysis) are
regenerated and evaluated with the production pre-dose procedure, exactly as in
r2_sensitivity_cohorts.py (26 admission features, Optuna with 100 trials,
logistic regression on the same features, realised event rates, combined
target attainment). Each variant is compared seed by seed with the primary
parameters, so differences reflect the parameter change only.

The parameters that define the simulator's ground truth for the PopPK layer
(between-subject variability, residual error, gentamicin PK values) and the
outcome-model coefficients are not swept: the former are tested by parameter
recovery (Table 3A, Supplementary Methods S2.3), and the latter define the
exposure-outcome mapping that the study is about.

Output: results/revision2/sensitivity_sweep.json
Usage:  python r2_sensitivity_sweep.py [--workers 11]
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from r2_sensitivity_cohorts import SEEDS, run_spec

REPO = Path(__file__).resolve().parent
OUT = REPO / 'results' / 'revision2' / 'sensitivity_sweep.json'

PATHOGENS = ['klebsiella', 'pseudomonas', 'e_coli', 'acinetobacter', 'other']
PATHOGEN_P = [0.35, 0.25, 0.20, 0.15, 0.05]
SITES = ['bloodstream', 'pneumonia', 'uti', 'iai', 'other']
SITE_P = [0.30, 0.35, 0.15, 0.12, 0.08]
GENT_RES = {'klebsiella': 0.54, 'pseudomonas': 0.40, 'e_coli': 0.45, 'acinetobacter': 0.65, 'other': 0.30}


def reweight(names, probs, name, share):
    """Set one category's probability to `share`, rescaling the others proportionally."""
    i = names.index(name)
    rest = 1 - probs[i]
    return [share if j == i else p * (1 - share) / rest for j, p in enumerate(probs)]


# (assumption, label, primary value, low, high, spec builder)
SWEEP = [
    ('Weight (mean)', 'kg', 62, 52, 72, lambda v: {'params': {'weight_mean': v}}),
    ('Mechanical ventilation', 'probability', 0.65, 0.45, 0.85, lambda v: {'params': {'mechanical_ventilation': v}}),
    ('Vasopressor use', 'probability', 0.50, 0.30, 0.70, lambda v: {'params': {'vasopressor_use': v}}),
    ('Diabetes', 'probability', 0.35, 0.20, 0.50, lambda v: {'params': {'diabetes_prevalence': v}}),
    ('Augmented renal clearance (prevalence)', 'probability', 0.25, 0.15, 0.40,
     lambda v: {'params': {'crcl_arc_proportion': v}}),
    ('Augmented renal clearance (mean CrCL)', 'mL/min', 160, 140, 180, lambda v: {'params': {'crcl_arc_mean': v}}),
    ('Pneumonia share of infection sites', 'probability', 0.35, 0.20, 0.50,
     lambda v: {'params': {'infection_probs': reweight(SITES, SITE_P, 'pneumonia', v)}}),
    ('Acinetobacter share of pathogens', 'probability', 0.15, 0.05, 0.25,
     lambda v: {'params': {'pathogen_probs': reweight(PATHOGENS, PATHOGEN_P, 'acinetobacter', v)}}),
    ('Gentamicin resistance (all pathogens, multiplier)', 'x', 1.0, 0.7, 1.3,
     lambda v: {'params': {'gent_resistance': {k: min(r * v, 0.95) for k, r in GENT_RES.items()}}}),
    ('Resistant MIC mode (both drugs, multiplier)', 'x', 1.0, 0.5, 2.0,
     lambda v: {'params': {'gent_mic_resistant_meanlog': float(np.log(32 * v)),
                           'ami_mic_resistant_meanlog': float(np.log(64 * v))}}),
    ('Septic-shock effect on CL (log scale)', 'log units', -0.15, 0.0, -0.30,
     lambda v: {'patches': [('generate_patient_demographics', "'septic shock', -0.15, 0)",
                             f"'septic shock', {v}, 0)")]}),
]
# Published values that differ materially from the simulator's (verified sources;
# Supplementary Table S1), each tested directly as a single variant
PUBLISHED = [
    ('Pathogen mix, ICMR 2024 ICU bloodstream isolates', 'Klebsiella 36.0%, Pseudomonas 9.5%, E. coli 8.7%, Acinetobacter 31.3%, other 14.5%',
     {'params': {'pathogen_probs': [0.360, 0.095, 0.087, 0.313, 0.145]}}),
    ('Gentamicin resistance, ICMR 2024 ICU bloodstream isolates', 'Klebsiella 0.691, Pseudomonas 0.510, E. coli 0.487, Acinetobacter 0.845',
     {'params': {'gent_resistance': {'klebsiella': 0.691, 'pseudomonas': 0.510, 'e_coli': 0.487, 'acinetobacter': 0.845}}}),
    ('Mechanical ventilation, SEPSIS INDIA', '0.48', {'params': {'mechanical_ventilation': 0.48}}),
]
METRICS = ['nephrotoxicity_pct', 'cure_pct', 'combined_attainment_pct', 'predose_auroc_xgb', 'predose_auroc_lr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=11)
    args = ap.parse_args()
    jobs = [(s, 'primary', None) for s in SEEDS]
    for name, unit, base, lo, hi, build in SWEEP:
        for level, v in [('low', lo), ('high', hi)]:
            jobs += [(s, f'{name} | {level} | {v}', build(v)) for s in SEEDS]
    for name, desc, spec in PUBLISHED:
        jobs += [(s, f'{name} | published', spec) for s in SEEDS]
    t0 = time.time()
    # max_tasks_per_child=1: every cohort runs in a fresh process, because the
    # simulator module is modified in place for each parameter specification
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
        rows = list(ex.map(run_spec, jobs))
    prim = {r['seed']: r for r in rows if r['parameters'] == 'primary'}

    out = {'seeds': SEEDS, 'primary': {m: {'mean': float(np.mean([prim[s][m] for s in SEEDS])),
                                           'sd': float(np.std([prim[s][m] for s in SEEDS], ddof=1))}
                                       for m in METRICS},
           'variants': [], 'published': [], 'wall_seconds': None, 'per_replicate': rows}
    for name, unit, base, lo, hi, _ in SWEEP:
        for level, v in [('low', lo), ('high', hi)]:
            lab = f'{name} | {level} | {v}'
            rr = {r['seed']: r for r in rows if r['parameters'] == lab}
            entry = {'assumption': name, 'unit': unit, 'primary_value': base, 'level': level, 'value': v}
            for m in METRICS:
                d = [rr[s][m] - prim[s][m] for s in SEEDS]
                entry[m] = {'mean': float(np.mean([rr[s][m] for s in SEEDS])),
                            'diff_mean': float(np.mean(d)), 'diff_min': float(np.min(d)), 'diff_max': float(np.max(d))}
            out['variants'].append(entry)
    for name, desc, _ in PUBLISHED:
        rr = {r['seed']: r for r in rows if r['parameters'] == f'{name} | published'}
        entry = {'assumption': name, 'value': desc, 'level': 'published'}
        for m in METRICS:
            d = [rr[s][m] - prim[s][m] for s in SEEDS]
            entry[m] = {'mean': float(np.mean([rr[s][m] for s in SEEDS])),
                        'diff_mean': float(np.mean(d)), 'diff_min': float(np.min(d)), 'diff_max': float(np.max(d))}
        out['published'].append(entry)
    out['wall_seconds'] = time.time() - t0
    # Largest absolute mean paired change for each metric, across all variants
    out['max_abs_mean_change'] = {m: max(abs(e[m]['diff_mean']) for e in out['variants'] + out['published'])
                                  for m in METRICS}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for e in out['variants'] + out['published']:
        print(f"{e['assumption'][:44]:<44} {e['level']:<4} {e['value']!s:<6} "
              + ' | '.join(f"{m.split('_')[0][:6]}{'-xgb' if m.endswith('xgb') else '-lr' if m.endswith('lr') else ''} "
                           f"{e[m]['diff_mean']:+.3f}" for m in METRICS))
    print('max |mean change|:', {k: round(v, 3) for k, v in out['max_abs_mean_change'].items()})
    print(f'Wall time {out["wall_seconds"] / 60:.1f} min for {len(jobs)} cohorts')


if __name__ == '__main__':
    main()
