#!/usr/bin/env python3
"""Revision 2: regenerate every R2 result, table, figure and document in order.

Two Python environments are used:
  * BASE  - the main environment (NumPy 2.x, XGBoost 3.1.2, scikit-learn, Optuna);
  * PYMC  - a pinned environment for the Bayesian fits (NumPy 1.26.4,
            PyMC 5.16.2, PyTensor 2.25.5, nutpie 0.13.2, numba 0.60.0, ArviZ,
            Matplotlib), e.g. C:/Users/<user>/venvs/popk_r2.

Usage:
    python r2_run_all.py --pymc C:/Users/<user>/venvs/popk_r2/Scripts/python.exe
    python r2_run_all.py --pymc ... --skip-refit      # reuse the PopPK refit (~2-3 h)

Steps marked (slow) take tens of minutes to hours.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'Manuscript_AAPS_Journal' / 'Revision2_Submission' / '_source'


def run(py, *args, cwd=ROOT):
    print(f'\n>>> {Path(py).name} {" ".join(map(str, args))}', flush=True)
    subprocess.run([py, *map(str, args)], cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pymc', required=True, help='python executable of the pinned PyMC environment')
    ap.add_argument('--skip-refit', action='store_true')
    ap.add_argument('--skip-ablation', action='store_true')
    a = ap.parse_args()
    base = sys.executable

    run(base, 'r2_simulator_truth.py')                       # true parameters; checks data/ reproduces
    run(a.pymc, 'r2_popk_parameterization_check.py')          # Table 3B-C
    if not a.skip_refit:
        run(a.pymc, 'r2_refit_popk_power.py')                 # Table 3A (slow)
        run(a.pymc, 'r2_popk_parameterization_check.py')      # adds R2 individual recovery
    run(base, 'r2_tables.py')                                 # Tables 2 and 4
    run(base, 'r2_ml_common.py')                              # verifies production predictions
    run(base, 'r2_ml_analyses.py', *(['--skip-ablation'] if a.skip_ablation else []))  # Tables 5C, 7 (slow)
    run(base, 'r2_shap.py')                                   # Table 6
    run(base, 'r2_dose_optimization.py')                      # Table 8 inputs, Figure 7
    run(base, 'r2_table8.py')
    run(base, 'r2_sensitivity_cohorts.py')                    # Table S2 (10 cohorts in parallel)
    run(a.pymc, 'r2_figure2_popk.py')                         # Figure 2, Figure S1
    run(base, 'r2_figures_main.py')                           # Figures 1, 3-7, Graphical Abstract
    run(base, 'r2_figures_supp.py')                           # Figures S2-S7
    run(base, 'make_tables.py', cwd=SRC)                      # tables_R2.md
    run(base, 'build_r2.py', cwd=SRC)                         # Word documents
    run(base, 'make_tracked.py', cwd=SRC)                     # tracked-changes manuscript (needs Word)


if __name__ == '__main__':
    main()
