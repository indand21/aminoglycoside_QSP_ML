#!/usr/bin/env python3
"""Revision 2: regenerate every R2 result, table, figure and document in order.

Two Python environments are used:
  * BASE  - the main environment (NumPy 2.x, XGBoost 3.1.2, scikit-learn, Optuna);
  * PYMC  - a pinned environment for the Bayesian fits (NumPy 1.26.4,
            PyMC 5.16.2, PyTensor 2.25.5, nutpie 0.13.2, numba 0.60.0, ArviZ,
            Matplotlib), e.g. C:/Users/<user>/venvs/popk_r2.

Posterior traces (models/*.nc) exceed GitHub's file-size limit and are not in
the repository. On a clean checkout the R1 production trace is therefore
regenerated first with refit_popk.py (slow), and the R2 refits are run unless
--skip-refit is given, in which case their traces must already exist.

The manuscript build (tables, Word documents, tracked-changes copy) lives in
Manuscript_AAPS_Journal/, which is not part of the public repository; those
steps run only when that folder is present.

The console output (stdout and stderr) of the steps that have a log in
results/revision2/ is shown live and also written to that log, so the logs
always describe the run that produced the results. A log is replaced only when
its step succeeds; a failed step leaves its output in <name>.log.partial.

Usage:
    python r2_run_all.py --pymc C:/Users/<user>/venvs/popk_r2/Scripts/python.exe
    python r2_run_all.py --pymc ... --skip-refit      # reuse existing R2 PopPK traces

Steps marked (slow) take tens of minutes to hours.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'Manuscript_AAPS_Journal' / 'Revision2_Submission' / '_source'
MODELS = ROOT / 'models'
LOGS = ROOT / 'results' / 'revision2'
R1_TRACE = MODELS / 'phase2_popk_trace_likelihood.nc'
R2_TRACES = [MODELS / 'r2_popk_power_trace.nc', MODELS / 'r2_popk_dense_trace.nc']


def run(py, *args, cwd=ROOT, log=None):
    """Run one step; with `log`, also write its console output to results/revision2/<log>."""
    print(f'\n>>> {Path(py).name} {" ".join(map(str, args))}', flush=True)
    cmd = [py, *map(str, args)]
    if log is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return
    final = LOGS / log
    partial = final.with_name(final.name + '.partial')
    # TF_CPP_MIN_LOG_LEVEL=1 hides TensorFlow's INFO banner (printed once per worker
    # process by the optional TensorFlow import); its warnings and errors still appear
    env = {'TF_CPP_MIN_LOG_LEVEL': '1', **os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
    con = sys.stdout.encoding or 'utf-8'                      # e.g. cp1252 on a Windows console
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace') as proc, \
            open(partial, 'w', encoding='utf-8', newline='\n') as fh:
        for line in proc.stdout:
            fh.write(line)                                    # the log keeps the full UTF-8 text
            sys.stdout.write(line.encode(con, 'replace').decode(con))
            sys.stdout.flush()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    os.replace(partial, final)                                # keep the old log if the step failed
    print(f'    log written to {final.relative_to(ROOT)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pymc', required=True, help='python executable of the pinned PyMC environment')
    ap.add_argument('--skip-refit', action='store_true',
                    help='reuse existing R2 PopPK traces instead of refitting')
    ap.add_argument('--skip-ablation', action='store_true')
    a = ap.parse_args()
    base = sys.executable

    if a.skip_refit:
        missing = [str(p.relative_to(ROOT)) for p in R2_TRACES if not p.exists()]
        if missing:
            sys.exit(f'--skip-refit needs the R2 PopPK traces, which are missing: {", ".join(missing)}. '
                     'Run without --skip-refit to regenerate them.')

    run(base, 'r2_simulator_truth.py')                       # true parameters; checks data/ reproduces
    if not R1_TRACE.exists():                                 # git-ignored; regenerate on a clean checkout
        print(f'\n{R1_TRACE.relative_to(ROOT)} not found: regenerating the R1 production fit (slow)')
        run(a.pymc, 'refit_popk.py')
    run(a.pymc, 'r2_popk_parameterization_check.py')          # Table 3B-C
    if not a.skip_refit:
        run(a.pymc, 'r2_refit_popk_power.py', log='popk_power_fit.log')   # Table 3A (slow)
        run(a.pymc, 'r2_popk_dense_check.py', log='popk_dense.log')       # dense-sampling check, S2.3 (slow)
        run(a.pymc, 'r2_popk_parameterization_check.py')      # adds R2 individual recovery
    run(base, 'r2_tables.py')                                 # Tables 2 and 4
    run(base, 'r2_ml_common.py')                              # verifies production predictions
    run(base, 'r2_ml_analyses.py', *(['--skip-ablation'] if a.skip_ablation else []),
        log='oof_nested.log')                                 # Tables 5C, 7 (slow)
    run(base, 'r2_shap.py')                                   # Table 6
    run(base, 'r2_dose_optimization.py')                      # Table 8 inputs, Figure 7, Table S3
    run(base, 'r2_table8.py')
    run(base, 'r2_sensitivity_cohorts.py', log='sensitivity_cohorts.log')  # Table S2 (10 cohorts in parallel)
    run(base, 'r2_sensitivity_sweep.py', log='sensitivity_sweep.log')      # Table S4 (130 cohorts, slow)
    run(a.pymc, 'r2_figure2_popk.py')                         # Figure 2, Figure S1
    run(base, 'r2_figures_main.py', log='figures_main.log')   # Figures 1, 3-7, Graphical Abstract
    run(base, 'r2_figures_supp.py', log='figures_supp.log')   # Figures S2-S7

    if SRC.exists():                                          # private manuscript build
        run(base, 'make_tables.py', cwd=SRC)                  # tables_R2.md
        run(base, 'build_r2.py', cwd=SRC)                     # Word documents
        run(base, 'make_tracked.py', cwd=SRC)                 # tracked-changes manuscript (needs Word)
    else:
        print('\nManuscript sources not present (not part of the public repository); '
              'skipping the manuscript build. All results and figures are complete.')


if __name__ == '__main__':
    main()
