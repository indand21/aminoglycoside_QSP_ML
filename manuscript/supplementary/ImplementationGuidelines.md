# Implementation Guidelines: QSP-ML Framework for Aminoglycoside Dosing

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Data Preparation](#3-data-preparation)
4. [Running the Framework](#4-running-the-framework)
5. [Interpreting Results](#5-interpreting-results)
6. [Clinical Deployment](#6-clinical-deployment)
7. [Troubleshooting](#7-troubleshooting)
8. [Validation Guidelines](#8-validation-guidelines)

---

## 1. System Requirements

### 1.1 Hardware Requirements

**Minimum Specifications:**
- CPU: 4 cores, 2.0 GHz or higher
- RAM: 16 GB
- Storage: 10 GB free disk space
- GPU: Not required (CPU-only implementation)

**Recommended Specifications:**
- CPU: 8+ cores, 2.4 GHz or higher
- RAM: 32 GB
- Storage: 50 GB SSD
- GPU: Optional, NVIDIA GPU with CUDA support for neural network models

**Performance Benchmarks:**
- Minimum system: ~6 hours for complete pipeline (1,500 patients)
- Recommended system: ~2 hours for complete pipeline
- Single patient dose optimization: <5 seconds

### 1.2 Software Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ or equivalent) - **Recommended**
- macOS 11.0+
- Windows 10+ (with WSL2 recommended)

**Python:**
- Python 3.8 - 3.11 (3.11 recommended)
- Not compatible with Python 3.12+ (PyMC dependency limitation)

**Core Dependencies:**
- NumPy ≥ 1.24
- Pandas ≥ 2.0
- SciPy ≥ 1.10
- Scikit-learn ≥ 1.4
- XGBoost ≥ 2.0
- LightGBM ≥ 4.0
- PyMC ≥ 5.10
- ArviZ ≥ 0.16 (for Bayesian diagnostics)
- Matplotlib ≥ 3.8
- Seaborn ≥ 0.13

**Optional Dependencies:**
- TensorFlow ≥ 2.13 (for neural network models)
- scikit-optimize ≥ 0.9 (for Bayesian optimization)
- imbalanced-learn ≥ 0.11 (for SMOTE)
- SHAP ≥ 0.43 (for model interpretability)

---

## 2. Installation

### 2.1 Quick Start Installation

**Using Conda (Recommended):**

```bash
# Create new conda environment
conda create -n aminoglycoside python=3.11
conda activate aminoglycoside

# Install core dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn

# Install specialized packages
pip install pymc==5.10.0 arviz==0.16.1
pip install xgboost==2.0.3 lightgbm==4.1.0
pip install imbalanced-learn==0.11.0
pip install scikit-optimize==0.9.0

# Optional: Install TensorFlow for neural networks
pip install tensorflow==2.15.0

# Clone repository
git clone https://github.com/[organization]/aminoglycoside_QSP_ML.git
cd aminoglycoside_QSP_ML

# Install package in development mode
pip install -e .
```

**Using pip with Virtual Environment:**

```bash
# Create virtual environment
python3.11 -m venv aminogly_env
source aminogly_env/bin/activate  # On Windows: aminogly_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone and install package
git clone https://github.com/[organization]/aminoglycoside_QSP_ML.git
cd aminoglycoside_QSP_ML
pip install -e .
```

### 2.2 Verifying Installation

```bash
# Test imports
python -c "import pymc; import xgboost; import sklearn; print('Installation successful!')"

# Run test suite
python -m pytest tests/

# Check versions
python scripts/check_dependencies.py
```

### 2.3 Docker Installation (Reproducibility)

```bash
# Pull Docker image
docker pull [organization]/aminoglycoside_qsp_ml:latest

# Run container
docker run -it -v $(pwd):/workspace aminoglycoside_qsp_ml:latest

# Inside container, activate environment and run analysis
```

---

## 3. Data Preparation

### 3.1 Input Data Format

**Required Data Structure:**

Create a CSV file with the following columns:

```csv
patient_id,age,sex,weight,height,baseline_scr,crcl,egfr,apache_ii,sofa,
diabetes,ckd_stage,mechanical_vent,vasopressor,septic_shock,albumin,
bilirubin,infection_site,pathogen,mic,aminoglycoside,dose,infusion_time,
dosing_interval,treatment_duration,nephrotoxicity,clinical_cure
```

**Column Descriptions:**

| Column | Type | Unit | Description | Required |
|--------|------|------|-------------|----------|
| patient_id | String | - | Unique patient identifier | Yes |
| age | Integer | years | Patient age | Yes |
| sex | String | - | "M" or "F" | Yes |
| weight | Float | kg | Body weight | Yes |
| height | Float | cm | Height | Yes |
| baseline_scr | Float | mg/dL | Baseline serum creatinine | Yes |
| crcl | Float | mL/min | Creatinine clearance (Cockcroft-Gault) | Yes |
| egfr | Float | mL/min/1.73m² | Estimated GFR (MDRD or CKD-EPI) | No |
| apache_ii | Integer | - | APACHE II score (0-71) | Yes |
| sofa | Integer | - | SOFA score (0-24) | Yes |
| diabetes | Integer | - | 1=present, 0=absent | Yes |
| ckd_stage | Integer | - | CKD stage (0-5) | Yes |
| mechanical_vent | Integer | - | 1=yes, 0=no | Yes |
| vasopressor | Integer | - | 1=yes, 0=no | Yes |
| septic_shock | Integer | - | 1=yes, 0=no | Yes |
| albumin | Float | g/dL | Serum albumin | No |
| bilirubin | Float | mg/dL | Total bilirubin | No |
| infection_site | String | - | "respiratory", "bloodstream", "urinary", "other" | Yes |
| pathogen | String | - | Identified organism species | Yes |
| mic | Float | mg/L | Minimum inhibitory concentration | Yes |
| aminoglycoside | String | - | "amikacin" or "gentamicin" | Yes |
| dose | Float | mg | Administered dose | Yes |
| infusion_time | Float | hours | Infusion duration (typically 0.5h) | Yes |
| dosing_interval | Float | hours | Interval between doses | Yes |
| treatment_duration | Float | days | Total treatment duration | Yes |
| nephrotoxicity | Integer | - | 1=AKI occurred, 0=no AKI | Yes |
| clinical_cure | Integer | - | 1=cured, 0=failure | Yes |

**Missing Data Handling:**

- Required fields with >10% missing: Impute using median (continuous) or mode (categorical)
- Optional fields: Can be left blank (will use population defaults)
- Do not use listwise deletion unless <5% patients affected

### 3.2 Data Quality Checks

**Run automated quality checks:**

```bash
python scripts/data_quality_check.py --input data/patient_data.csv --output reports/quality_report.html
```

**Manual Checks:**

1. **Physiological Plausibility:**
   - Age: 18-100 years
   - Weight: 30-200 kg
   - Height: 120-220 cm
   - CrCL: 5-200 mL/min
   - MIC: 0.125-128 mg/L

2. **Logical Consistency:**
   - Dose should be 5-25 mg/kg for gentamicin, 15-30 mg/kg for amikacin
   - Patients with CrCL <30 should have extended dosing intervals
   - APACHE II and SOFA scores should correlate

3. **Outlier Detection:**
   - Flag patients >3 SD from mean for key variables
   - Review extreme PK/PD indices (Cmax >100 mg/L, Cmin >10 mg/L)

### 3.3 Data Preprocessing Script

```bash
# Automated preprocessing with default settings
python scripts/preprocess_data.py \
  --input data/patient_data.csv \
  --output data/processed_data.csv \
  --impute-missing \
  --remove-outliers \
  --validate

# Custom preprocessing
python scripts/preprocess_data.py \
  --input data/patient_data.csv \
  --output data/processed_data.csv \
  --impute-method median \
  --outlier-threshold 4.0 \
  --min-patients 100
```

---

## 4. Running the Framework

### 4.1 Complete Pipeline Execution

**Run all phases sequentially:**

```bash
# Execute complete analysis pipeline
python run_complete_pipeline.py \
  --input data/processed_data.csv \
  --output-dir results/ \
  --n-mcmc-samples 2000 \
  --ml-optimization \
  --dose-optimization \
  --generate-reports

# Estimated runtime: 2-6 hours depending on hardware
```

### 4.2 Individual Phase Execution

**Phase 1: Data Preprocessing**

```bash
python phase1_preprocessing.py \
  --input data/patient_data.csv \
  --output data/phase1_processed.csv
```

**Phase 2: Population Pharmacokinetic Modeling**

```bash
python phase2_popk_modeling.py \
  --input data/phase1_processed.csv \
  --n-chains 4 \
  --n-samples 2000 \
  --n-tune 1000 \
  --output results/phase2_popk/

# Output files:
# - parameter_estimates.csv
# - trace.nc (NetCDF format for ArviZ)
# - diagnostics.png
# - posterior_predictive.png
```

**Phase 3: PK/PD Target Attainment Analysis**

```bash
python phase3_pkpd_analysis.py \
  --input data/phase1_processed.csv \
  --popk-results results/phase2_popk/trace.nc \
  --output results/phase3_pkpd/

# Output files:
# - pkpd_indices.csv
# - pta_analysis.csv
# - cfr_analysis.csv
# - target_attainment_summary.txt
```

**Phase 4: Machine Learning Model Development**

```bash
python phase4_machine_learning_enhanced.py \
  --input data/phase1_processed.csv \
  --pkpd-data results/phase3_pkpd/pkpd_indices.csv \
  --test-size 0.2 \
  --cv-folds 5 \
  --use-smote \
  --hyperparameter-tuning \
  --n-iter 50 \
  --output results/phase4_ml/

# Output files:
# - nephrotoxicity_model.pkl
# - clinical_cure_model.pkl
# - performance_metrics.csv
# - feature_importance.csv
# - roc_curves.png
# - calibration_plots.png
```

**Phase 5: Dose Optimization**

```bash
python phase5_dose_optimization.py \
  --input data/phase1_processed.csv \
  --ml-models results/phase4_ml/ \
  --n-patients 50 \
  --output results/phase5_optimization/

# Output files:
# - dose_recommendations.csv
# - optimization_curves.png
# - target_attainment_comparison.csv
```

**Phase 6: Comprehensive Reporting**

```bash
python phase6_generate_reports.py \
  --results-dir results/ \
  --output reports/

# Output files:
# - executive_summary.pdf
# - technical_report.html
# - clinical_recommendations.docx
```

### 4.3 Single Patient Dose Optimization

**For real-time clinical use:**

```bash
python demo_simple_dose_optimization.py \
  --patient-file patient.json \
  --output dose_recommendation.pdf

# Example patient.json:
{
  "age": 65,
  "weight": 70,
  "sex": "M",
  "crcl": 60,
  "baseline_scr": 1.5,
  "diabetes": 1,
  "mechanical_vent": 1,
  "apache_ii": 22,
  "mic": 2.0,
  "aminoglycoside": "amikacin"
}
```

### 4.4 Parallel Processing

**For large datasets (>5,000 patients):**

```bash
# Use multiple cores for ML training
python phase4_machine_learning_enhanced.py \
  --input data/large_dataset.csv \
  --n-jobs -1  # Use all available cores
  --parallel-backend loky

# Parallelize MCMC chains across cores
python phase2_popk_modeling.py \
  --input data/large_dataset.csv \
  --n-chains 8 \
  --cores 8
```

---

## 5. Interpreting Results

### 5.1 Population Pharmacokinetic Results

**Key Files:**

1. **parameter_estimates.csv**: Population parameter values with credible intervals
2. **trace.nc**: Full MCMC trace for advanced analysis
3. **diagnostics.png**: Convergence diagnostics (R-hat, trace plots)

**Interpretation Guidelines:**

```
Parameter    | Expected Range | Clinical Meaning
-------------|----------------|------------------
CL (L/h)     | 4-7           | Higher CL → lower concentrations, need higher doses
Vc (L)       | 12-22         | Larger Vc → lower Cmax, need higher doses
Q (L/h)      | 8-18          | Distribution rate between compartments
Vp (L)       | 7-15          | Peripheral distribution volume
t½ (hours)   | 2-4           | Half-life for once-daily dosing assessment
```

**Quality Indicators:**

✓ **Good Model:**
- R-hat < 1.01 for all parameters
- Effective sample size >1,000
- Posterior predictive checks show good agreement
- Parameter estimates physiologically plausible

✗ **Problematic Model:**
- R-hat > 1.05 (lack of convergence)
- Wide credible intervals (>50% relative to mean)
- Systematic bias in posterior predictive checks

### 5.2 PK/PD Analysis Results

**Target Attainment Interpretation:**

```
PTA Metric               | Value  | Clinical Interpretation
------------------------|--------|---------------------------
Cmax/MIC ≥8             | <40%   | Inadequate efficacy, dose increase needed
                        | 40-60% | Borderline, consider optimization
                        | >60%   | Adequate efficacy likely

AUC/MIC ≥80             | <40%   | High resistance risk
                        | 40-60% | Moderate risk
                        | >60%   | Lower resistance risk

Combined Targets        | <40%   | Substantial improvement possible
                        | 40-55% | Moderate improvement possible
                        | >55%   | Already optimized
```

**Heatmap Guidance:**

- Dark blue/green regions: High PTA (>80%), optimal dosing
- Yellow regions: Moderate PTA (50-80%), consider individualization
- Red regions: Low PTA (<50%), inadequate dosing

### 5.3 Machine Learning Model Results

**Performance Benchmarks:**

```
ROC-AUC     | Clinical Utility
------------|------------------
<0.60       | Poor, not clinically useful
0.60-0.70   | Modest, limited utility
0.70-0.80   | Good, clinically useful
0.80-0.90   | Excellent
>0.90       | Outstanding (check for overfitting)
```

**Feature Importance Analysis:**

Top 5 features should explain >40% of total importance:
- If dominated by PK/PD indices: Good, validates mechanistic understanding
- If dominated by clinical severity: Consider recalibration, may be predicting illness rather than drug effect
- If many features ~equal importance: May indicate noise, consider feature selection

**Model Calibration:**

Check calibration plots:
- Points close to diagonal = well-calibrated
- Points above diagonal = underestimation of risk
- Points below diagonal = overestimation of risk

Brier score:
- <0.15: Excellent calibration
- 0.15-0.25: Good calibration
- >0.25: Poor calibration, recalibration needed

### 5.4 Dose Optimization Results

**For Individual Patients:**

```
Objective Score | Interpretation
----------------|----------------
<0.50           | High risk patient, consider alternative therapy
0.50-0.65       | Suboptimal, benefit from optimization
0.65-0.80       | Good, likely favorable outcome
>0.80           | Excellent, optimal dosing achieved
```

**Dose Recommendations:**

Compare optimized vs standard dosing:
- <10% difference: Standard dosing acceptable
- 10-30% difference: Optimization recommended
- >30% difference: Optimization strongly recommended (extremes of weight, renal function, or MIC)

---

## 6. Clinical Deployment

### 6.1 Integration with Electronic Health Records (EHR)

**API Endpoint Example:**

```python
# FastAPI server for dose recommendations
from fastapi import FastAPI
import joblib

app = FastAPI()

# Load trained models
ml_model = joblib.load('models/clinical_cure_model.pkl')
pk_model = joblib.load('models/popk_model.pkl')

@app.post("/optimize_dose")
def optimize_dose(patient: PatientData):
    """
    Real-time dose optimization endpoint

    Input: Patient characteristics (JSON)
    Output: Dose recommendation with probabilities
    """
    # Extract features
    features = extract_features(patient)

    # Predict PK parameters
    pk_params = predict_pk(features, pk_model)

    # Optimize dose
    optimal_dose = optimize_objective(features, pk_params, ml_model)

    # Calculate outcome probabilities
    probs = calculate_probabilities(optimal_dose, features, ml_model)

    return {
        "recommended_dose": optimal_dose,
        "cure_probability": probs['cure'],
        "nephrotoxicity_risk": probs['nephrotox'],
        "confidence_interval": [optimal_dose * 0.9, optimal_dose * 1.1],
        "rationale": generate_rationale(optimal_dose, features, probs)
    }
```

**EHR Integration Steps:**

1. **Data Extraction:**
   - Create FHIR resource mappings for required patient data
   - Automated extraction from EHR using HL7/FHIR APIs
   - Real-time updates when labs available

2. **Decision Support Alert:**
   - Trigger when aminoglycoside ordered
   - Display dose recommendation with probabilities
   - Allow clinician override with documentation

3. **Documentation:**
   - Auto-generate clinical note with:
     * Recommended dose and rationale
     * Expected PK/PD indices
     * Target attainment probabilities
     * Key risk factors identified

### 6.2 Clinical Validation Workflow

**Before Clinical Use:**

1. **Retrospective Validation:**
   ```bash
   python validation/retrospective_validation.py \
     --historical-data data/historical_cohort.csv \
     --trained-models models/ \
     --output reports/validation_report.pdf
   ```

2. **Prospective Observational Study:**
   - Run model in shadow mode (predictions recorded but not acted on)
   - Compare model predictions to actual outcomes
   - Minimum 100 patients for initial validation

3. **Randomized Controlled Trial (Optional):**
   - Control: Standard dosing
   - Intervention: Model-guided dosing
   - Primary endpoint: Target attainment rate
   - Secondary: Clinical cure, nephrotoxicity

### 6.3 Clinical Use Protocol

**Standard Operating Procedure:**

1. **Patient Assessment:**
   - Verify aminoglycoside therapy appropriate
   - Collect required baseline data
   - Ensure MIC available or use epidemiological breakpoints

2. **Dose Calculation:**
   - Input patient data into system
   - Review automated dose recommendation
   - Check for alerts/warnings (e.g., severe renal impairment)

3. **Clinical Judgment:**
   - Model provides recommendation, not mandate
   - Consider patient-specific factors not in model
   - Document reason if deviating from recommendation

4. **Therapeutic Drug Monitoring:**
   - Obtain peak concentration (1h post-infusion)
   - Obtain trough concentration (pre-next dose)
   - Update model with measured concentrations
   - Recalculate dose if needed

5. **Outcome Assessment:**
   - Monitor renal function daily
   - Assess clinical response at 48-72h
   - Adjust therapy based on culture results and response

### 6.4 Safety Guardrails

**Automated Alerts:**

```python
def check_safety_guardrails(dose, patient):
    """Implement safety checks before dose recommendation"""

    alerts = []

    # Maximum dose limits
    if dose > 1800:
        alerts.append("WARNING: Dose exceeds typical maximum (1800mg)")

    # Minimum dose limits
    if dose < 200:
        alerts.append("WARNING: Dose below typical minimum (200mg)")

    # Renal function checks
    if patient.crcl < 30 and dose > 800:
        alerts.append("CRITICAL: High dose with severe renal impairment")

    # Elderly checks
    if patient.age > 75 and dose > 1200:
        alerts.append("CAUTION: High dose in elderly patient")

    # Obesity checks
    if patient.bmi > 35 and dose > 2000:
        alerts.append("WARNING: Dose may exceed recommendations for obesity")

    return alerts
```

**Required Clinical Oversight:**

- Pharmacist verification of all recommendations
- Attending physician approval for >20% deviation from standard
- Mandatory TDM for high-risk patients
- Regular audit of outcomes (quarterly)

---

## 7. Troubleshooting

### 7.1 Common Installation Issues

**Problem: PyMC installation fails**

```
Solution:
# Use conda for PyMC (easier than pip)
conda install -c conda-forge pymc==5.10.0

# If still failing, install dependencies separately
conda install numpy scipy matplotlib
pip install pymc==5.10.0
```

**Problem: XGBoost ImportError on macOS**

```
Solution:
# Install XGBoost with libomp
brew install libomp
pip install xgboost
```

**Problem: TensorFlow not using GPU**

```
Solution:
# Verify CUDA installation
nvidia-smi

# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Test GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 7.2 Common Runtime Issues

**Problem: MCMC sampling very slow**

```
Solutions:
1. Reduce number of samples:
   python phase2_popk_modeling.py --n-samples 1000 --n-tune 500

2. Use fewer chains:
   --n-chains 2  # Instead of 4

3. Initialize from prior modes:
   --init-method adapt_diag

4. Increase target_accept (trades speed for accuracy):
   --target-accept 0.90  # Default is 0.95
```

**Problem: R-hat > 1.01 (non-convergence)**

```
Solutions:
1. Increase warmup iterations:
   --n-tune 2000  # Double the warmup

2. Increase target_accept:
   --target-accept 0.99  # More conservative sampling

3. Check for multimodality:
   python scripts/check_trace_multimodality.py --trace results/phase2_popk/trace.nc

4. Reparameterize model (contact support)
```

**Problem: ML model overfitting (train-test gap >0.15)**

```
Solutions:
1. Increase regularization:
   # In XGBoost, increase gamma and lambda
   params = {'gamma': 0.5, 'lambda': 2.0}

2. Reduce model complexity:
   params = {'max_depth': 5}  # Instead of 7

3. Use more training data or fewer features

4. Apply more aggressive SMOTE:
   --smote-k-neighbors 3  # Instead of 5
```

**Problem: Dose optimization recommends extreme doses**

```
Solutions:
1. Check input data quality:
   python scripts/validate_patient_data.py --input patient.json

2. Adjust objective function weights:
   # Increase safety weight if doses too high
   --weights 0.3,0.4,0.2,0.1  # cure,safety,cmax,trough

3. Add stricter constraints:
   --min-dose 300 --max-dose 1400

4. Review surrogate model calibration
```

### 7.3 Performance Optimization

**Speed up ML training:**

```bash
# Use all CPU cores
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Run with parallelization
python phase4_machine_learning_enhanced.py --n-jobs -1

# Use faster tree method for XGBoost
--tree-method hist  # Instead of exact
```

**Reduce memory usage:**

```bash
# Process data in chunks
python phase4_machine_learning_enhanced.py --batch-size 500

# Use sparse matrices for large datasets
--use-sparse

# Reduce MCMC storage
--thin 2  # Store every 2nd sample
```

---

## 8. Validation Guidelines

### 8.1 Internal Validation Checklist

Before using models on new data, verify:

- [ ] Model convergence (R-hat < 1.01 for all parameters)
- [ ] Adequate effective sample size (>1,000 per parameter)
- [ ] Posterior predictive checks show good agreement
- [ ] ML cross-validation AUC within 10% of test AUC
- [ ] Calibration plots show good agreement (slope 0.9-1.1)
- [ ] Feature importance makes clinical sense
- [ ] No extreme dose recommendations in validation set
- [ ] Safety guardrails trigger appropriately

### 8.2 External Validation Protocol

**Dataset Requirements:**

- Minimum 200 patients (preferably 500+)
- Same inclusion/exclusion criteria
- Similar patient population
- Complete baseline characteristics
- Verified outcomes (nephrotoxicity, clinical cure)
- Known MIC values

**Validation Script:**

```bash
python validation/external_validation.py \
  --external-data data/external_cohort.csv \
  --trained-models models/ \
  --output reports/external_validation.pdf \
  --bootstrap-ci \
  --n-bootstrap 1000
```

**Success Criteria:**

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| ROC-AUC decline | <0.05 | Good generalization |
| Calibration slope | 0.85-1.15 | Acceptable calibration |
| Target attainment improvement | >10% absolute | Clinically meaningful |
| No serious safety events | 100% | Safe for deployment |

### 8.3 Continuous Monitoring

**After Clinical Deployment:**

```bash
# Monthly performance monitoring
python monitoring/monthly_performance.py \
  --clinical-data data/monthly_cohort.csv \
  --reference-performance models/baseline_performance.json \
  --alert-threshold 0.05 \
  --output reports/monthly_$(date +%Y%m).pdf
```

**Triggers for Model Retraining:**

- AUC decline >0.05
- Calibration slope <0.85 or >1.15
- Systematic bias in predictions (>10% in any subgroup)
- Change in patient population or practice patterns
- New evidence for PK/PD targets
- Annual update recommended regardless

---

## 9. Citation and Attribution

If you use this framework in research or clinical practice, please cite:

```
[Author names]. Integration of Quantitative Systems Pharmacology and Machine Learning
for Personalized Aminoglycoside Dosing in Critically Ill Patients. [Journal]. [Year];[Vol]:[Pages].
```

**License:** [To be specified - likely MIT or Apache 2.0 for open source]

**Contact:**
- Technical support: [email]
- Clinical questions: [email]
- Bug reports: GitHub issues
- Feature requests: GitHub discussions

---

## 10. Additional Resources

**Documentation:**
- Full API reference: `docs/api_reference.html`
- Tutorial videos: [YouTube playlist]
- Example datasets: `data/examples/`
- Validation case studies: `docs/case_studies/`

**Training Materials:**
- Webinar series: [Link]
- Hands-on workshop materials: `docs/workshop/`
- FAQ: `docs/FAQ.md`

**Community:**
- Discussion forum: [Link]
- Slack channel: [Link]
- Quarterly user group meetings: [Link]

---

## Appendix A: Complete Parameter Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--n-chains` | 4 | 2-8 | Number of MCMC chains |
| `--n-samples` | 2000 | 500-5000 | MCMC samples per chain |
| `--n-tune` | 1000 | 500-2000 | MCMC warmup iterations |
| `--target-accept` | 0.95 | 0.80-0.99 | MCMC acceptance rate target |
| `--test-size` | 0.2 | 0.1-0.3 | ML test set fraction |
| `--cv-folds` | 5 | 3-10 | Cross-validation folds |
| `--n-iter` | 50 | 20-200 | Hyperparameter search iterations |
| `--smote-k` | 5 | 3-10 | SMOTE nearest neighbors |
| `--max-dose` | 1600 | 1000-2500 | Maximum allowed dose (mg) |
| `--min-dose` | 200 | 100-500 | Minimum allowed dose (mg) |

## Appendix B: Error Codes Reference

| Code | Message | Solution |
|------|---------|----------|
| E001 | Insufficient data | Minimum 100 patients required |
| E002 | Missing required columns | Check data format section 3.1 |
| E003 | MCMC divergence | Increase target_accept or tune iterations |
| E004 | Model file not found | Run prior phases first |
| E005 | Invalid patient data | Check data quality guidelines |
| E006 | Optimization timeout | Reduce n-iterations or simplify objective |
| E007 | Insufficient memory | Reduce batch size or use sparse matrices |
| E008 | GPU not available | Install CUDA or use CPU-only mode |

---

**Document Version:** 1.0
**Last Updated:** 2025-01-16
**Maintainer:** [Name/Organization]
