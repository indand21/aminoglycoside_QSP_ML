# Agent Guide: Aminoglycoside QSP-ML Project

This guide provides essential information for AI agents working in this repository. It documents the project structure, essential commands, code patterns, and gotchas specific to this Python-based Quantitative Systems Pharmacology (QSP) and Machine Learning (ML) framework for personalized aminoglycoside dosing in ICU patients.

## Project Overview

- **Objective**: Combine mechanistic PK/PD modeling with machine learning to optimize aminoglycoside dosing for Indian ICU patients, balancing efficacy (clinical cure) and safety (nephrotoxicity).
- **Language**: Python 3.10+ (shebang `#!/usr/bin/env python3`). The project started with R documentation but all production code is Python.
- **Key Libraries**: PyMC (Bayesian PK), XGBoost, scikit-learn, Optuna, Plotly, pandas, NumPy, SciPy, imbalanced-learn, TensorFlow (optional), LightGBM/CatBoost (optional).
- **Workflow**: Five sequential phases (preprocessing → population PK → PK/PD modeling → ML → dose optimization), each implemented as a standalone Python script.
- **Data**: Synthetic patient data stored in `data/` (CSV, PKL). Processed data in `data/processed/`. Results in `results/`. Models in `models/`.

## Essential Commands

### Dependency Installation

No `requirements.txt` or `pyproject.toml` exists; dependencies are installed via pip commands documented in `DEPENDENCY_INSTALLATION_AND_RESULTS.md`. The following commands install the required packages:

```bash
# Core dependencies (run in order)
python -m pip install pymc arviz optuna plotly
python -m pip install --upgrade --force-reinstall scipy scikit-learn
python -m pip install --upgrade pandas pyarrow xgboost
python -m pip install --upgrade imbalanced-learn
python -m pip install optuna-integration[xgboost]

# Optional (deep learning, other ML libraries)
python -m pip install tensorflow lightgbm catboost
```

**Note**: NumPy 2.x is incompatible; ensure NumPy 1.26.4 is used. PyMC 5.16.2 with PyTensor 2.25.5 is required.

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
│   └── phase5_optimization/
├── utils/                  # Utility modules
│   ├── __init__.py
│   └── extract_individual_pk.py  # Extract individual PK parameters from Bayesian trace
├── *.py                    # Main phase scripts (phase1_preprocessing.py, …)
├── test_*.py               # Test scripts
├── run_*.py                # Pipeline scripts
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
- Hyperparameter optimization uses `RandomizedSearchCV`; Optuna is installed but not yet integrated (see `DEPENDENCY_INSTALLATION_AND_RESULTS.md`).
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

### Outdated R Documentation

The main `README.md` references R scripts and directory structures that do not exist. **Ignore those sections**. The actual implementation is entirely Python.

### Dependency Versions

- NumPy 2.x is incompatible; stick to NumPy 1.26.4.
- PyMC 5.16.2 with PyTensor 2.25.5 is required; newer versions may cause issues.
- Optuna is installed but not yet integrated into the ML build methods (see `DEPENDENCY_INSTALLATION_AND_RESULTS.md` for remaining work).

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

1. **Install dependencies** using the pip commands above.
2. **Run phases sequentially** (Phase 1 → Phase 5).
3. **Check for output files** after each phase (e.g., `models/phase2_popk_trace.nc`).
4. **Run tests** to verify improvements.
5. **Consult existing documentation** for gotchas (Windows, dependency versions).
6. **Ignore R‑related instructions**; focus on Python scripts.

---

*This document is based on the codebase as of February 2026. Update it as the project evolves.*