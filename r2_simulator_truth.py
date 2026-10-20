#!/usr/bin/env python3
"""Revision 2: recover the simulator's ground truth for each synthetic patient.

`generate_synthetic_data_python.py` seeds NumPy's global RNG (12345) at import
and drops the true individual PK parameters before writing data/. Re-running its
generation functions in the original order reproduces the saved dataset exactly,
so the true parameters can be recovered in memory without touching data/.

The script verifies the reproduction against every saved file and writes
results/revision2/true_individual_pk.csv (true CL, Vc, Q, Vp per patient).
Other R2 scripts import `load_truth()`, and use the outcome-model functions
below for the oracle check of the dose optimizer.

Usage:
    python r2_simulator_truth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / 'results' / 'revision2'
TRUTH_PATH = OUT_DIR / 'true_individual_pk.csv'


def regenerate():
    """Re-run the simulator in its original call order; return all tables."""
    sys.path.insert(0, str(REPO_ROOT))
    if 'generate_synthetic_data_python' in sys.modules:
        raise RuntimeError('simulator module already imported; RNG state is not fresh')
    import generate_synthetic_data_python as g   # np.random.seed(12345) on import

    patient_data = g.generate_patient_demographics(1500)
    dosing = g.generate_dosing_records(patient_data)
    conc = g.generate_concentrations(patient_data, dosing)
    time_varying = g.generate_time_varying(patient_data, 7)
    outcomes = g.generate_outcomes(patient_data, conc, dosing)
    return patient_data, dosing, conc, time_varying, outcomes


def verify(patient_data, dosing, conc, outcomes):
    saved_p = pd.read_csv(REPO_ROOT / 'data' / 'patient_data.csv')
    saved_d = pd.read_csv(REPO_ROOT / 'data' / 'dosing.csv')
    saved_c = pd.read_csv(REPO_ROOT / 'data' / 'concentrations.csv')
    saved_o = pd.read_csv(REPO_ROOT / 'data' / 'outcomes.csv')
    checks = {
        'patients': np.allclose(patient_data['baseline_crcl'], saved_p['baseline_crcl'])
                    and (patient_data['drug'].values == saved_p['drug'].values).all(),
        'dosing': len(dosing) == len(saved_d) and np.allclose(dosing['dose'], saved_d['dose']),
        'concentrations': len(conc) == len(saved_c)
                          and np.allclose(conc['concentration'], saved_c['concentration']),
        'nephrotoxicity': (outcomes['nephrotoxicity'].values
                           == saved_o['nephrotoxicity'].values).all(),
        'clinical_cure': (outcomes['clinical_cure'].values
                          == saved_o['clinical_cure'].values).all(),
    }
    for k, v in checks.items():
        print(f'  reproduces saved {k:<15} {v}')
    if not all(checks.values()):
        raise RuntimeError('regenerated data do not match data/; truth not recoverable')


def load_truth() -> pd.DataFrame:
    return pd.read_csv(TRUTH_PATH)


# --------------------------------------------------------------------------- #
# Outcome-generating model (generate_synthetic_data_python.py, lines 597-640),
# written as probabilities so that dose changes can be scored without noise.
# --------------------------------------------------------------------------- #
def true_p_nephrotoxicity(mean_trough, drug_is_amikacin, age, diabetes, scr,
                          mech_vent, vasopressor, albumin, sofa):
    thr = np.where(drug_is_amikacin, 2.5, 2.0)
    logit = (-5.0 + 2.5 * np.log(mean_trough / thr + 0.1) + 0.06 * age
             + 1.2 * diabetes + 1.0 * (scr > 1.5) + 0.8 * mech_vent
             + 0.6 * vasopressor - 0.5 * albumin + 0.05 * sofa)
    return 1.0 / (1.0 + np.exp(-logit))


def true_p_cure(cmax_mic, auc_mic, mean_trough, apache, sofa, acinetobacter,
                albumin, crcl):
    logit = (-0.5 + 1.2 * np.log(np.clip(cmax_mic, 0.01, None) + 0.1)
             + 0.8 * np.log(np.clip(auc_mic, 0.01, None) + 0.1)
             - 0.12 * apache - 0.06 * sofa - 1.8 * acinetobacter + 0.4 * albumin
             - 0.8 * (mean_trough > 2) + 0.3 * (crcl > 60))
    return np.clip(1.0 / (1.0 + np.exp(-logit)), 0.001, 0.999)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_data, dosing, conc, _, outcomes = regenerate()
    print('\nVerifying reproduction against data/ ...')
    verify(patient_data, dosing, conc, outcomes)
    truth = patient_data[['patient_id', 'drug', 'sepsis_type', 'CL_individual',
                          'Vc_individual', 'Q_individual', 'Vp_individual']].copy()
    truth.to_csv(TRUTH_PATH, index=False)
    print(f'Wrote {TRUTH_PATH} ({len(truth)} patients)')


if __name__ == '__main__':
    main()
