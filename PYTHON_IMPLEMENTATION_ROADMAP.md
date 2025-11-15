# Python Implementation Roadmap

**Project:** Aminoglycoside QSP-ML Optimization Framework
**Implementation:** Python-based (complete pipeline)
**Status:** Phase 1 Complete ✅

---

## Overview

This document outlines the complete Python implementation strategy for all 6 phases of the aminoglycoside optimization framework.

---

## ✅ Phase 1: Data Preprocessing (COMPLETE)

**Status:** ✅ DONE
**Script:** `phase1_preprocessing.py`
**Output:** `data/processed/`

### Completed Tasks:
- ✅ Load synthetic data (300 patients)
- ✅ Calculate derived variables (BSA, IBW, ABW)
- ✅ Compute creatinine clearance (time-varying)
- ✅ Calculate PK/PD metrics (Cmax/MIC)
- ✅ Detect outliers and quality issues
- ✅ Create ML dataset (300×63)
- ✅ Create PopPK dataset (3,631 rows, NONMEM format)

**Key Outputs:**
- `ml_dataset.csv` - Ready for ML models
- `popk_dataset.csv` - Ready for PK modeling
- `pk_metrics.csv` - PK/PD indices

---

## 🔄 Phase 2: Population PK Modeling (NEXT)

**Status:** 🟡 IN PLANNING
**Target:** Fit two-compartment PK model with covariate effects

### Objectives:
1. Fit two-compartment PK model to concentration data
2. Estimate population parameters (CL, Vc, Q, Vp)
3. Identify significant covariates
4. Quantify between-subject variability (BSV)
5. Validate model performance

### Python Implementation Options:

#### **Option A: PyMC (Bayesian)**
**Pros:**
- ✅ Fully Bayesian (uncertainty quantification)
- ✅ Flexible model specification
- ✅ Python-native
- ✅ Great for dose optimization (Phase 5)

**Cons:**
- ⚠️ Slower than frequentist methods
- ⚠️ Requires MCMC expertise

**Implementation:**
```python
import pymc as pm
import pytensor.tensor as pt

# Define two-compartment model
with pm.Model() as pk_model:
    # Population parameters
    theta_CL = pm.Lognormal('theta_CL', mu=np.log(5), sigma=0.5)
    theta_Vc = pm.Lognormal('theta_Vc', mu=np.log(15), sigma=0.5)
    # ... (covariate effects, BSV, residual error)

    # Fit model
    trace = pm.sample(2000, tune=1000)
```

#### **Option B: SciPy + Custom ODE**
**Pros:**
- ✅ Full control over model
- ✅ Fast optimization
- ✅ No additional dependencies

**Cons:**
- ⚠️ More coding required
- ⚠️ No built-in diagnostics

**Implementation:**
```python
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

def two_comp_ode(y, t, CL, Vc, Q, Vp, dose_rate):
    A_central, A_peripheral = y
    dA_central = dose_rate - (CL/Vc)*A_central - (Q/Vc)*A_central + (Q/Vp)*A_peripheral
    dA_peripheral = (Q/Vc)*A_central - (Q/Vp)*A_peripheral
    return [dA_central, dA_peripheral]

# Fit to data using maximum likelihood
```

#### **Option C: PINTS (PK-specific)**
**Pros:**
- ✅ Designed for PK modeling
- ✅ Built-in optimization algorithms
- ✅ Good documentation

**Cons:**
- ⚠️ Less popular than PyMC
- ⚠️ Additional dependency

#### **Option D: Interface with R (nlmixr2)**
**Pros:**
- ✅ Industry-standard tool
- ✅ Comprehensive diagnostics
- ✅ R is already installed

**Cons:**
- ⚠️ Requires R-Python bridge (rpy2)
- ⚠️ Package installation limitations in this environment

### Recommended Approach: **PyMC (Option A)**

**Rationale:**
1. Native Python implementation
2. Bayesian framework perfect for Phase 5 (dose optimization)
3. Excellent for uncertainty quantification
4. Well-documented and actively maintained
5. Can recover known parameters from synthetic data

### Implementation Plan:

**Step 1: Install PyMC**
```bash
pip install pymc arviz
```

**Step 2: Create `phase2_population_pk.py`**
```python
# Implement two-compartment Bayesian PK model
# Estimate: CL, Vc, Q, Vp
# Covariates: CrCL, weight, age, sepsis
# BSV on all parameters
# Combined residual error
```

**Step 3: Model Diagnostics**
- Trace plots
- Posterior predictive checks
- VPC (Visual Predictive Check)
- Parameter correlation
- Individual fits

**Step 4: Validation**
- Compare to known parameters (synthetic data truth)
- Goodness-of-fit plots
- Residual analysis

---

## 📊 Phase 3: PK/PD Modeling

**Status:** 🟡 PENDING
**Prerequisites:** Phase 2 complete

### Objectives:
1. Calculate PK/PD indices using individual PK parameters
2. Link PK/PD indices to clinical outcomes
3. Perform target attainment analysis
4. Identify optimal PK/PD targets

### Python Implementation:

**Tools:**
- NumPy/SciPy for calculations
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization

**Key Tasks:**
```python
# 1. Calculate Cmax, AUC24, Cmax/MIC, AUC/MIC
# 2. Probability of target attainment (PTA)
# 3. Cumulative fraction of response (CFR)
# 4. Link to clinical cure (logistic regression)
# 5. Link to nephrotoxicity (logistic regression)
```

**Script:** `phase3_pkpd_modeling.py`

---

## 🤖 Phase 4: Machine Learning

**Status:** 🟡 PENDING
**Prerequisites:** Phase 1 complete (ML dataset ready)

### Objectives:
1. Train models for nephrotoxicity prediction
2. Train models for clinical cure prediction
3. Train models for PK parameter prediction (surrogate)
4. Feature importance analysis
5. Model validation

### Python Implementation:

**Tools:**
- **scikit-learn** - Preprocessing, model selection, validation
- **XGBoost** - Gradient boosting models
- **imbalanced-learn** - SMOTE for class imbalance
- **SHAP** - Feature importance and explainability

**Models:**

#### Model 1: Nephrotoxicity Prediction (Classification)
```python
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_res, y_res)

# Evaluate
roc_auc = cross_val_score(model, X, y, cv=10, scoring='roc_auc')
```

#### Model 2: Clinical Cure Prediction (Classification)
```python
# Similar structure to nephrotoxicity model
# Features: demographics, severity, PK metrics, renal function
```

#### Model 3: PK Parameter Surrogate (Regression)
```python
from sklearn.ensemble import RandomForestRegressor

# Predict individual CL and Vc from baseline covariates
# Allows rapid dose calculation without full PopPK
```

**Script:** `phase4_machine_learning.py`

**Outputs:**
- Trained models (pickle files)
- ROC curves, calibration plots
- Feature importance plots
- Model performance metrics

---

## 🎯 Phase 5: Integrated Optimization

**Status:** 🟡 PENDING
**Prerequisites:** Phases 2, 3, 4 complete

### Objectives:
1. Bayesian dose optimization for individual patients
2. Multi-objective optimization (efficacy + safety)
3. Real-time dose recommendations
4. Uncertainty quantification

### Python Implementation:

**Tools:**
- **PyMC** - Bayesian PK model (from Phase 2)
- **scipy.optimize** - Multi-objective optimization
- **pymoo** - Advanced multi-objective optimization

**Workflow:**

#### Step 1: Bayesian Individual Forecasting
```python
# Use population PK model from Phase 2
# Update with individual patient observations
# Get posterior distributions of PK parameters

with pk_model:
    # Condition on patient data
    posterior = pm.sample_posterior_predictive(
        trace,
        var_names=['CL', 'Vc', 'Q', 'Vp']
    )
```

#### Step 2: Multi-Objective Optimization
```python
from scipy.optimize import minimize

def objective_function(dose_interval, patient_params):
    """
    Optimize dose and interval
    Objective 1: Maximize P(Cmax/MIC >= 8)
    Objective 2: Minimize P(nephrotoxicity)
    """
    dose, interval = dose_interval

    # Simulate PK profile
    cmax = simulate_pk(dose, interval, patient_params)
    cmax_mic = cmax / patient_mic

    # Predict toxicity using ML model
    tox_prob = ml_model.predict_proba([patient_features])[0, 1]

    # Combined objective (weighted)
    return -(0.6 * (cmax_mic >= 8) - 0.4 * tox_prob)

# Optimize
result = minimize(objective_function, x0=[1000, 24],
                 bounds=[(100, 3000), (12, 48)])
```

**Script:** `phase5_dose_optimization.py`

**Outputs:**
- Recommended dose and interval
- Predicted Cmax/MIC
- Predicted nephrotoxicity risk
- Uncertainty bounds

---

## ✅ Phase 6: Validation & Simulation

**Status:** 🟡 PENDING
**Prerequisites:** All phases complete

### Objectives:
1. External validation on independent data
2. Monte Carlo simulation across patient population
3. Performance assessment
4. Final model validation

### Python Implementation:

**Monte Carlo Simulation:**
```python
import numpy as np
from multiprocessing import Pool

def simulate_regimen(patient, dose, interval):
    """Simulate one patient-regimen combination."""
    # Use PK model to predict Cmax, Cmin
    # Calculate target attainment
    # Predict toxicity
    return results

# Simulate 1000 patients × 10 regimens in parallel
with Pool() as pool:
    results = pool.starmap(simulate_regimen, patient_regimen_combinations)
```

**External Validation:**
```python
# Load external dataset
# Apply preprocessing
# Predict using trained models
# Calculate performance metrics
```

**Script:** `phase6_validation_simulation.py`

**Outputs:**
- PTA by regimen and renal function
- Efficacy-safety trade-off plots
- Model performance on external data
- Final validation report

---

## Implementation Timeline

| Phase | Status | Estimated Time | Priority |
|-------|--------|----------------|----------|
| Phase 1 | ✅ Complete | - | - |
| Phase 2 | 🟡 Next | 2-3 days | **HIGH** |
| Phase 3 | 🟡 Pending | 1-2 days | HIGH |
| Phase 4 | 🟡 Pending | 2-3 days | HIGH |
| Phase 5 | 🟡 Pending | 2-3 days | MEDIUM |
| Phase 6 | 🟡 Pending | 1-2 days | MEDIUM |

**Total estimated time:** 8-13 days for complete implementation

---

## Required Python Packages

### Currently Installed ✅
- numpy
- pandas
- scipy
- pickle
- json

### To Install for Phases 2-6

#### Phase 2 (PK Modeling):
```bash
pip install pymc arviz
pip install matplotlib seaborn
```

#### Phase 4 (Machine Learning):
```bash
pip install scikit-learn
pip install xgboost
pip install imbalanced-learn
pip install shap
```

#### Phase 5 (Optimization):
```bash
pip install pymoo  # Multi-objective optimization
```

#### Visualization (All Phases):
```bash
pip install plotly  # Interactive plots
pip install seaborn  # Statistical visualization
```

---

## File Structure

```
aminoglycoside_QSP_ML/
│
├── data/
│   ├── [synthetic data]
│   └── processed/
│       ├── ml_dataset.csv              ✅
│       ├── popk_dataset.csv            ✅
│       └── [other processed files]     ✅
│
├── models/                             [To create]
│   ├── popk_model.pkl
│   ├── nephrotox_model.pkl
│   └── cure_model.pkl
│
├── results/                            [To create]
│   ├── phase2_pk_diagnostics/
│   ├── phase4_ml_performance/
│   └── phase6_validation/
│
├── phase1_preprocessing.py             ✅ DONE
├── phase2_population_pk.py             🟡 TODO
├── phase3_pkpd_modeling.py             🟡 TODO
├── phase4_machine_learning.py          🟡 TODO
├── phase5_dose_optimization.py         🟡 TODO
├── phase6_validation_simulation.py     🟡 TODO
│
├── generate_synthetic_data_python.py   ✅ DONE
├── PYTHON_IMPLEMENTATION_ROADMAP.md    ✅ THIS FILE
├── PHASE1_COMPLETE.md                  ✅ DONE
└── README.md                           ✅ DONE
```

---

## Next Immediate Steps

### 1. Install PyMC (for Phase 2)
```bash
pip install --user pymc arviz matplotlib
```

### 2. Create Phase 2 Script
- Implement two-compartment Bayesian PK model
- Fit to `popk_dataset.csv`
- Generate diagnostics and validation plots

### 3. Validate Parameter Recovery
- Compare estimated parameters to known truth
- Verify model can recover:
  - CL ~5.5 L/h
  - Vc ~16 L
  - Q ~12 L/h
  - Vp ~10 L
  - Covariate effects

### 4. Proceed to Phase 3
- Use individual PK parameters from Phase 2
- Calculate PK/PD indices
- Link to outcomes

---

## Success Criteria

### Phase 2:
- ✅ Model converges successfully
- ✅ Parameter estimates within 20% of truth
- ✅ VPC shows good agreement
- ✅ Residuals are randomly distributed

### Phase 3:
- ✅ PK/PD indices calculated correctly
- ✅ Significant relationship with clinical cure
- ✅ Cmax/MIC >8 associated with higher cure rate

### Phase 4:
- ✅ ROC-AUC >0.75 for outcome prediction
- ✅ Good calibration (observed vs predicted)
- ✅ Interpretable feature importance

### Phase 5:
- ✅ Optimized doses achieve target PK/PD
- ✅ Toxicity risk minimized
- ✅ Uncertainty quantified

### Phase 6:
- ✅ External validation metrics acceptable
- ✅ Monte Carlo simulations realistic
- ✅ Framework ready for clinical use

---

## Questions or Issues?

See the main project README.md or specific phase documentation.

**Current Status:** Ready to begin Phase 2! 🚀
