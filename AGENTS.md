# Agent Guide: Aminoglycoside QSP-ML Project

This guide provides essential information for AI agents working in this repository. It documents the project structure, essential commands, code patterns, and gotchas specific to this Python-based Quantitative Systems Pharmacology (QSP) and Machine Learning (ML) framework for personalized aminoglycoside dosing in ICU patients.

## Project Overview

- **Objective**: Combine mechanistic PK/PD modeling with machine learning to optimize aminoglycoside dosing for Indian ICU patients, balancing efficacy (clinical cure) and safety (nephrotoxicity).
- **Language**: Python 3.12, as declared in `requirements.txt` (shebang `#!/usr/bin/env python3`). The Revision 2 scripts need at least Python 3.11 (`ProcessPoolExecutor(max_tasks_per_child=...)`). The project started with R documentation but all production code is Python.
- **Key Libraries**: PyMC (Bayesian PK), XGBoost, scikit-learn, Optuna, Plotly, pandas, NumPy, SciPy, imbalanced-learn, TensorFlow (optional), LightGBM/CatBoost (optional).
- **Workflow**: Five sequential phases (preprocessing → population PK → PK/PD modeling → ML → dose optimization), each implemented as a standalone Python script.
- **Data**: Synthetic patient data stored in `data/` (CSV, PKL). Processed data in `data/processed/`. Results in `results/`. Models in `models/`.

## Essential Commands

### Dependency Installation

`requirements.txt` is the pinned dependency manifest; its versions match those reported in the manuscript Methods. Install from it rather than from unpinned commands:

```bash
python -m pip install -r requirements.txt

# Optional (deep learning)
python -m pip install tensorflow
```

**Note**: `requirements.txt` pins NumPy 1.26.4, PyMC 5.16.2 and PyTensor 2.25.5; NumPy 2.x cannot import this PyMC. `r2_run_all.py` (Revision 2) instead uses two environments, a main one for the ML scripts and a pinned PyMC environment passed with `--pymc`; see its docstring.

### Running the Pipeline

Each phase can be run independently. The typical order is:

```bash
# Phase 1: Data preprocessing
python phase1_preprocessing.py

# Phase 2: Bayesian population PK modeling (30–90 minutes)
python phase2_population_pk.py

# Phase 3: PK/PD analysis (requires Phase 2 output)
python phase3_pkpd_modeling.py

# Phase 4: Enhanced machine learning (requires Phase 3 output)
python phase4_machine_learning_enhanced.py

# Phase 5: Dose optimization
python phase5_dose_optimization.py
```

**Pipeline scripts** (convenience wrappers) are also available:

```bash
python run_phase2_pipeline.py    # Runs Phase 2 enhanced ML pipeline
python run_phase3_pipeline.py    # Runs Phase 3 PK/PD pipeline
python run_dose_optimization_analysis.py  # Runs full dose optimization
```

### Testing

Test scripts validate each phase’s improvements. They are not unit tests but integration checks.

```bash
python test_phase1_improvements.py
python test_phase2_enhancements.py
python test_phase3_enhancements.py
python test_phase4_enhancements.py
```

**Note**: Some tests require output files from previous phases (e.g., `models/phase2_popk_trace.nc`). If missing, they will skip with a warning.

### Utility Scripts

- `generate_synthetic_data_python.py` – Generates synthetic ICU patient data (Python version).
- `generate_manuscript_figures.py` – Creates figures for the manuscript.
- `generate_population_characteristics_figure.py` – Generates population characteristics figure.
- `generate_shap_figure_panel.py` – Creates SHAP analysis figure panels.
- `enhanced_dataflow_visualization.py` – Visualizes the dataflow.
- `workflow_visualization.py` – Generates workflow diagrams.
- `check_feature_importance.py` – Checks feature importance of trained models.
- `fill_poster.py` – Fills poster templates with results.

## Revision 2 Scripts (`r2_*.py`)

The `r2_*.py` scripts produce every number, table and figure of the second revision of the AAPS Journal manuscript (AAPSJ-D-26-00274). They read the Phase 1–5 outputs and write to `results/revision2/`; they do not modify `data/` or the Phase 4 production models.

### Running them

```bash
# Everything, in dependency order (two environments; see below)
python r2_run_all.py --pymc <path to the PyMC environment's python.exe>
python r2_run_all.py --pymc ... --skip-refit      # reuse existing R2 PopPK traces
python r2_run_all.py --pymc ... --skip-ablation   # skip the feature-engineering ablation
```

`r2_run_all.py` uses two Python environments:

- **Main environment** (NumPy 2.x, XGBoost 3.1.2, scikit-learn, Optuna) for the ML, dose-optimization, table and figure scripts.
- **Pinned PyMC environment** (NumPy 1.26.4, PyMC 5.16.2, PyTensor 2.25.5, nutpie 0.13.2, numba 0.60.0, ArviZ, Matplotlib), passed with `--pymc`, for the scripts that read or fit posterior traces (marked *PyMC* below).

### Scripts, in run order

| Script | Purpose | Main outputs (`results/revision2/`) | Time |
|---|---|---|---|
| `r2_simulator_truth.py` | Re-runs the simulator (seed 12345) in memory, checks the result against key columns of `data/` (baseline CrCL, drug, doses, concentrations, both outcomes; `time_varying.csv` is not checked), and recovers each patient's true PK parameters; `load_truth()` and the outcome functions are imported by other scripts | `true_individual_pk.csv` | seconds |
| `refit_popk.py` | R1 production PopPK fit (*PyMC*); run automatically only if its trace is missing | `models/phase2_popk_trace_likelihood.nc` | slow |
| `r2_popk_parameterization_check.py` | Puts the R1 PopPK estimates (z-standardized covariates) on the simulator's reference scale (*PyMC*) | `popk_r1_reference_comparison.csv`, `popk_individual_recovery.csv` | ~20 s |
| `r2_refit_popk_power.py` | Correctly specified power-model refit (*PyMC*; `--seed`, default 2026; `--smoke` for a quick test) | `popk_power_summary.csv`, `popk_power_chains.json`, `models/r2_popk_power_trace.nc` | ~70 min |
| `r2_popk_dense_check.py` | Dense-sampling simulation-estimation check of identifiability (*PyMC*) | `popk_dense_summary.csv`, `models/r2_popk_dense_trace.nc` | ~20 min |
| `r2_tables.py` | Tables 2 and 4 from `data/` and Phase 3 PK/PD indices (CLSI M100 2023 breakpoints) | `table2_values.json`, `table4_values.json` | seconds |
| `r2_ml_common.py` | Shared helpers that reproduce the Phase 4 production setup; run directly to verify the held-out predictions match `results/phase4_ml_enhanced/test_predictions.json` | – | ~20 s |
| `r2_ml_analyses.py` | Nested out-of-fold predictions, risk stratification (Table 7), feature-engineering ablation, logistic-regression comparisons, imputed-PK sensitivity | `oof_predictions.csv`, `risk_stratification.json`, `feature_ablation.json`, `lr_comparisons.json` | ~11 min without ablation; longer with it |
| `r2_shap.py` | Exact TreeSHAP and gain importance, each model with its own scaler | `shap_importance.json`, `shap_values_*.csv` | ~15 s |
| `r2_dose_optimization.py` | Vectorized full-cohort grid-search dose optimization with a ground-truth (simulator) check and a gentamicin trough-threshold scenario | `dose_optimization_*.json/.csv` | ~45 s |
| `r2_table8.py` | Table 8 (baseline and optimized doses by drug) | `table8_values.json` | seconds |
| `r2_sensitivity_cohorts.py` | Source-aligned cohorts: published values replace author-chosen ones, 5 paired seeds (Table S2) | `sensitivity_cohorts.json` | ~2 min |
| `r2_sensitivity_sweep.py` | One-at-a-time sweep of assumptions without a published source, plus three published values (Table S4); 130 cohorts | `sensitivity_sweep.json` | ~26 min |
| `r2_figure2_popk.py` | Figure 2 and Supplementary Figure S1 from the posterior traces (*PyMC*) | `figure2_values.json` | ~20 s |
| `r2_figures_main.py` | Figures 1 and 3–7 and the Graphical Abstract; cross-checks plotted values against the result files | `figure_values.json` | ~35 s |
| `r2_figures_supp.py` | Supplementary Figures S2–S7 (asserts its values against the result files) | `supp_figure_values.json` | ~25 s |

Times are from a 12-core Windows workstation. The scripts print to the console; `r2_run_all.py` also writes the console output (stdout and stderr) of the PopPK refits, `r2_ml_analyses.py`, both sensitivity scripts and the two figure scripts to their `*.log` files in `results/revision2/`, replacing a log only when its step succeeds (a failed step leaves `<name>.log.partial`). Running a script directly does not update its log. The figure scripts write images to `Manuscript_AAPS_Journal/Revision2_Submission/` (created if missing; git-ignored, so figures are never committed). `r2_run_all.py` also runs the private manuscript build (`make_tables.py`, `build_r2.py`, `make_tracked.py`) only when that folder exists.

### Conventions

- **No hand-typed results.** Every number is computed by a script and stored in `results/revision2/`; tables and the manuscript text are filled from those files. Change the script and re-run it rather than editing a result file.
- **Fixed seeds.** Simulator 12345; PopPK refit 2026 (2027 for the repeat); sensitivity cohorts `[12345, 111, 222, 333, 444]`; Optuna TPE seed 42 (Phase 4's own tuning code). Re-running a script reproduces its outputs value for value, apart from recorded run times.
- **Production models are reused, not retrained.** `r2_ml_common.py` rebuilds the Phase 4 features, split and scaler exactly. The saved XGBoost JSON models do not store the tuned hyperparameters, so any refit (nested CV, ablation, sensitivity cohorts) re-tunes with Optuna (100 trials).
- **Parallelism.** The sensitivity scripts use `ProcessPoolExecutor(max_tasks_per_child=1)` (Python ≥ 3.11) so that each cohort runs in a fresh process; `--workers` sets the pool size. CPU processes are used rather than the GPU, which was slower for these small XGBoost fits.

## Code Organization

```
aminoglycoside_QSP_ML/
├── data/                    # Raw and processed data
│   ├── *.csv               # Raw CSV files
│   └── processed/          # Processed datasets (ml_dataset.csv, popk_dataset.csv, etc.)
├── models/                 # Saved model artifacts
│   ├── enhanced/           # Enhanced ML models (JSON)
│   ├── phase2_popk_trace.nc  # Bayesian posterior trace (NetCDF)
│   └── phase2_popk_results.pkl
├── results/                # Outputs from each phase
│   ├── phase2_diagnostics/
│   ├── phase3_pkpd/
│   ├── phase4_ml_enhanced/
│   ├── phase5_optimization/
│   └── revision2/          # Revision 2 results (r2_*.py); _optuna_scratch/ is git-ignored
├── utils/                  # Utility modules
│   ├── __init__.py
│   └── extract_individual_pk.py  # Extract individual PK parameters from Bayesian trace
├── *.py                    # Main phase scripts (phase1_preprocessing.py, …)
├── test_*.py               # Test scripts
├── run_*.py                # Pipeline scripts
├── r2_*.py                 # Revision 2 analysis scripts (see above); refit_popk.py = R1 production PopPK fit
└── *.md                    # Extensive documentation (README, guides, summaries)
```

**Key Modules**:

- `phase1_preprocessing.py` – Contains `AminoglycosidePreprocessor` class.
- `phase2_population_pk.py` – Bayesian two‑compartment PK model using PyMC.
- `phase3_pkpd_modeling.py` – Computes PK/PD indices and time‑based PD features.
- `phase4_machine_learning_enhanced.py` – `EnhancedMLPipeline` class; builds ML models for nephrotoxicity, clinical cure, and PK surrogates.
- `phase5_dose_optimization.py` – Dose optimization algorithms.
- `utils.extract_individual_pk` – Extracts individual PK parameters from the Bayesian posterior.

## Key Patterns and Conventions

### Import Style

- Standard library imports first, then third‑party, then local modules.
- Use `warnings.filterwarnings('ignore')` to suppress warnings (common in ML scripts).
- Conditional imports for optional libraries (TensorFlow, LightGBM, CatBoost) with `try/except` and a flag (`TF_AVAILABLE`, etc.).

### Data Handling

- Data is loaded via pandas `read_csv()` from `data/` directory.
- Processed datasets are saved as CSV in `data/processed/` (e.g., `ml_dataset.csv`, `popk_dataset.csv`).
- Models and large objects are saved as JSON, pickle, or NetCDF (for PyMC traces) in `models/`.
- Results (metrics, plots) are saved in `results/` with subdirectories per phase.

### Class and Function Patterns

- Each phase script defines a main class (`AminoglycosidePreprocessor`, `EnhancedMLPipeline`) with a `run_*` or `run_complete_pipeline` method.
- Functions include verbose logging with `print` statements and separators (`"=" * 80`).
- Configuration is often passed via constructor arguments; paths are relative to the project root.

### Bayesian Modeling (PyMC)

- PyMC models are built using the `with pm.Model()` context manager.
- **Critical for Windows**: Set `PYTENSOR_FLAGS` environment variable before importing PyMC to avoid DLL issues. This is already done in `phase2_population_pk.py`:

```python
import os
os.environ.setdefault('PYTENSOR_FLAGS', 'device=cpu,floatX=float64,optimizer=fast_compile,cxx=')
import pymc as pm
```

- The Bayesian trace is saved as NetCDF (`models/phase2_popk_trace.nc`) for later individual PK extraction.

### Machine Learning Pipeline

- The `EnhancedMLPipeline` class orchestrates data loading, feature engineering, model training, evaluation, and saving.
- Imbalance handling via `class_weight` (default) or SMOTE (optional).
- Hyperparameter optimization uses Optuna (TPE sampler, seed 42, 100 trials in the production models); `RandomizedSearchCV` is used only as a fallback when Optuna is unavailable or fails.
- Performance metrics (ROC‑AUC, R², etc.) are stored in JSON files under `results/phase4_ml_enhanced/`.

## Testing Approach

- No formal unit‑test framework (pytest, unittest) is used.
- Test scripts are integration‑style checks that verify the presence of expected files, run key functions, and print success/failure messages.
- Tests are designed to be run after the corresponding phase completes; they may skip if dependencies are missing.
- To run all tests, execute each `test_*.py` script in order (Phase 1 → Phase 4).

## Gotchas and Non‑Obvious Patterns

### Windows‑Specific PyTensor Issues

On Windows, importing PyMC may fail with DLL errors. The project already sets `PYTENSOR_FLAGS` in relevant scripts. If you encounter import errors:

1. Ensure the environment variable is set **before** importing `pymc` or `pytensor`.
2. Install Microsoft Visual C++ Redistributable 2015‑2022 (x64).
3. Consider using Conda with MKL‑backed NumPy/SciPy.
4. Detailed troubleshooting steps are in `WINDOWS_PYTENSOR_TROUBLESHOOTING.md`.

### Missing Phase 2 Trace File

Many downstream steps (individual PK extraction, some tests) require `models/phase2_popk_trace.nc`. If this file is missing:

- Run `python phase2_population_pk.py` (takes 30‑90 minutes).
- If you cannot run Phase 2, those steps will skip or fall back to using only population PK parameters.

### Which PopPK Trace Is the Posterior

`phase2_population_pk.py` defines priors but no observation likelihood, so `models/phase2_popk_trace.nc` is a prior sample. `refit_popk.py` adds the likelihood and writes `models/phase2_popk_trace_likelihood.nc`, which is the R1 production fit used by `phase3_pkpd_modeling.py` and the Revision 2 scripts. That fit uses z-standardized log-linear covariates, so its coefficients are log-changes per SD, not allometric exponents; `r2_popk_parameterization_check.py` converts them. All `*.nc` traces (0.2–0.7 GB) are git-ignored and must be regenerated on a clean checkout.

### Outdated R Documentation

The main `README.md` references R scripts and directory structures that do not exist. **Ignore those sections**. The actual implementation is entirely Python.

### Dependency Versions

- NumPy 2.x is incompatible; stick to NumPy 1.26.4.
- PyMC 5.16.2 with PyTensor 2.25.5 is required; newer versions may cause issues.
- Phase 4 depends on Optuna for tuning; without it, results fall back to `RandomizedSearchCV` and will not match the production models.

### Data Paths

All scripts assume the working directory is the project root. Relative paths like `data/patient_data.csv` are used. Do not change the working directory inside scripts.

**Note**: The `catboost_info/` directory contains training logs and TensorBoard events generated by CatBoost when `train_dir` is set. The `phase4_output.log` file is a log from a previous Phase 4 run. These can be safely ignored or deleted.

### Performance Expectations

- Phase 2 (Bayesian MCMC) is computationally intensive (30‑90 minutes on a typical desktop).
- Phase 4 (ML training) is faster but can take several minutes depending on hyperparameter search space.
- The project is designed to run on a single machine; no distributed computing setup is configured.

## References to Existing Documentation

- `DEPENDENCY_INSTALLATION_AND_RESULTS.md` – Detailed installation steps, performance results, and remaining work.
- `WINDOWS_PYTENSOR_TROUBLESHOOTING.md` – Windows‑specific fixes for PyTensor.
- `PYTHON_IMPLEMENTATION_ROADMAP.md` – High‑level roadmap of the Python implementation.
- `PROJECT_EXECUTION_WORKFLOW.md` – Overview of the execution workflow.
- `README.md` – General project description (note: contains outdated R references).
- Many other markdown files in the root provide summaries of each phase, validation, literature comparison, etc.

## Quick Checklist for Agents

1. **Install dependencies** with `python -m pip install -r requirements.txt`.
2. **Run phases sequentially** (Phase 1 → Phase 5).
3. **Check for output files** after each phase (e.g., `models/phase2_popk_trace.nc`).
4. **Run tests** to verify improvements.
5. **Consult existing documentation** for gotchas (Windows, dependency versions).
6. **Ignore R‑related instructions**; focus on Python scripts.
7. **For Revision 2 work**, change the relevant `r2_*.py` script and re-run it (or `r2_run_all.py`); never edit files in `results/revision2/` by hand.

---

*This document is based on the codebase as of February 2026. Update it as the project evolves.*