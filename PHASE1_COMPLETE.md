# Phase 1 Complete: Data Preprocessing ✅

**Status:** COMPLETED
**Date:** 2025-11-15
**Implementation:** Python

---

## Summary

Phase 1 data preprocessing has been successfully completed for 300 synthetic Indian ICU patients. The data is now ready for Population PK modeling (Phase 2) and Machine Learning (Phase 4).

---

## What Was Accomplished

### ✅ **Derived Variables Calculated**

| Variable | Mean ± SD | Description |
|----------|-----------|-------------|
| BSA | 1.67 ± 0.16 m² | Body surface area (Mosteller formula) |
| IBW | 56.7 ± 8.7 kg | Ideal body weight (Devine formula) |
| ABW | 59.0 ± 7.0 kg | Adjusted body weight (for obese) |

### ✅ **Obesity Classification**

- Normal weight: 141 (47.0%)
- Overweight: 86 (28.7%)
- Underweight: 37 (12.3%)
- Obese Class I: 31 (10.3%)
- Obese Class II: 3 (1.0%)
- Obese Class III: 2 (0.7%)

### ✅ **Creatinine Clearance Calculated**

- **Baseline CrCL:** Already calculated during data generation
- **Time-varying CrCL:** Updated for 2,400 records
- **Mean time-varying CrCL:** 112.3 ± 69.8 mL/min
- **ARC patients identified:** 74 (24.7%)

### ✅ **PK/PD Metrics Calculated**

| Metric | Value |
|--------|-------|
| Mean Cmax | 51.84 mg/L |
| Mean Cmax/MIC | 10.72 |
| Target Cmax/MIC ≥8 | 96 patients (32.0%) |
| Target Cmax/MIC ≥10 | 77 patients (25.7%) |

### ✅ **Outlier Detection**

- **Concentration outliers:** 0 (all within normal range)
- **Suspicious peak samples:** 5 (concentration <10 mg/L)
- **Suspicious trough samples:** 0

### ✅ **Data Quality Checks**

All quality checks passed:
- ✓ No negative times
- ✓ No missing essential covariates (weight, age, SCr)
- ✓ No extreme doses (<100 or >3000 mg)
- ✓ No duplicate patient IDs
- ✓ All values within expected ranges

---

## Generated Datasets

### 1. **ML Dataset** (`ml_dataset.csv`)
- **300 rows × 63 columns**
- **Format:** Wide (one row per patient)
- **Purpose:** Machine learning models, outcome prediction

**Features include:**
- Demographics: age, sex, weight, height, BMI, BSA, IBW, ABW
- Clinical severity: APACHE II, SOFA, sepsis type
- Renal function: baseline SCr, CrCL, eGFR, ARC flag
- Infection: site, pathogen, MIC values
- Comorbidities: diabetes, CKD, ventilation, vasopressors
- PK metrics: Cmax, Cmin, Cmax/MIC ratios
- Dosing: first dose, mean dose, frequency
- Time-varying summaries: peak SCr, min CrCL, max fluid balance, max WBC, max CRP
- Outcomes: clinical cure, nephrotoxicity, mortality, LOS

**Usage:**
```python
import pandas as pd
ml_data = pd.read_csv('data/processed/ml_dataset.csv')
```

### 2. **PopPK Dataset** (`popk_dataset.csv`)
- **3,631 rows** (dosing + observation records)
- **Format:** Long (NONMEM-compatible)
- **Purpose:** Population PK modeling

**Structure:**
- `ID`: Numeric patient ID (1-300)
- `TIME`: Time from first dose (hours)
- `EVID`: Event ID (1=dose, 0=observation)
- `AMT`: Dose amount (mg)
- `DV`: Dependent variable (concentration, mg/L)
- `CMT`: Compartment (1=central, 2=observation)
- `RATE`: Infusion rate
- `MDV`: Missing DV flag
- **Covariates:** AGE, WT, HT, SEX, SCR, CRCL, APACHE, SOFA

**Usage:**
```python
popk_data = pd.read_csv('data/processed/popk_dataset.csv')
```

### 3. **PK Metrics** (`pk_metrics.csv`)
- **300 rows** (one per patient)
- Observed Cmax, Cmin, mean concentration
- Number of samples, peak/trough availability
- Cmax/MIC ratios and target attainment

### 4. **Processed Patient Data** (`patient_data_processed.csv`)
- Enhanced patient demographics with derived variables
- Obesity classification, ARC flag

### 5. **Processed Time-Varying** (`time_varying_processed.csv`)
- Updated creatinine clearance calculations

### 6. **Processed Concentrations** (`concentrations_processed.csv`)
- Outlier flags, suspicious sample flags

### 7. **Processed Data Bundle** (`processed_data.pkl`)
- Python pickle containing all datasets
- Preprocessing metadata

---

## Data Quality Summary

| Check | Result |
|-------|--------|
| Completeness | ✓ 100% (all 300 patients) |
| Missing values | ✓ None in critical variables |
| Outliers | ✓ 0 concentration outliers |
| Value ranges | ✓ All within expected bounds |
| Duplicates | ✓ None found |

---

## File Structure

```
data/
├── [original synthetic data files]
└── processed/
    ├── ml_dataset.csv                    # ML-ready (300×63)
    ├── popk_dataset.csv                  # PopPK-ready (3631 rows)
    ├── pk_metrics.csv                    # PK/PD metrics
    ├── patient_data_processed.csv        # Enhanced demographics
    ├── time_varying_processed.csv        # Updated CrCL
    ├── concentrations_processed.csv      # Flagged outliers
    ├── processed_data.pkl                # Python bundle
    └── preprocessing_metadata.json       # Metadata

phase1_preprocessing.py                   # Preprocessing script
```

---

## How to Use Preprocessed Data

### Load ML Dataset
```python
import pandas as pd
import pickle

# Option 1: Load CSV
ml_data = pd.read_csv('data/processed/ml_dataset.csv')

# Option 2: Load from pickle (includes all datasets)
with open('data/processed/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

ml_data = data['ml_dataset']
popk_data = data['popk_dataset']
pk_metrics = data['pk_metrics']
```

### Explore the Data
```python
# View dimensions
print(f"ML dataset: {ml_data.shape}")
print(f"PopPK dataset: {popk_data.shape}")

# Summary statistics
print(ml_data.describe())

# Check features
print(ml_data.columns.tolist())

# View first few rows
print(ml_data.head())
```

---

## Key Findings

### Demographics
- **Age:** 51.2 ± 15.4 years (Indian ICU population)
- **Weight:** 62.6 ± 11.3 kg (lower than Western populations)
- **BMI:** 24.2 ± 5.2 kg/m² (mostly normal to overweight)

### Clinical Characteristics
- **High severity:** APACHE II 21.9 ± 7.9
- **Septic shock:** 40.3% of patients
- **Mechanical ventilation:** 67.3%
- **Diabetes:** 34.7% (high Indian prevalence)

### Renal Function
- **Baseline CrCL:** 104.2 ± 44.1 mL/min
- **Bimodal distribution:** Normal + ARC patients
- **ARC prevalence:** 24.7%

### PK/PD Performance
- **Mean Cmax:** 51.84 mg/L
- **Mean Cmax/MIC:** 10.72
- **Target attainment:** Only 32% achieving Cmax/MIC ≥8
  - **Implication:** Many patients are underdosed
  - **Need:** Optimization of dosing regimens

---

## Next Steps

### ✅ **Immediate Next: Phase 2 - Population PK Modeling**

**Objectives:**
1. Fit two-compartment PK model to the data
2. Identify significant covariates (CrCL, weight, age, sepsis)
3. Quantify between-subject variability (BSV)
4. Validate model with VPC and diagnostics

**Implementation approach:**
- **Option 1:** Python-based PK modeling using `pypkpd` or custom ODE solver
- **Option 2:** Interface with R packages (nlmixr2) if needed
- **Option 3:** Bayesian approach using PyMC or Stan (Python interface)

**Data ready:** `popk_dataset.csv` (3,631 records in NONMEM format)

### ⏭️ **Phase 3: PK/PD Modeling**

Use PK parameters from Phase 2 to:
- Calculate PK/PD indices (Cmax/MIC, AUC/MIC)
- Perform target attainment analysis
- Link to clinical outcomes

### ⏭️ **Phase 4: Machine Learning**

Use `ml_dataset.csv` to:
- Train models for nephrotoxicity prediction
- Train models for clinical cure prediction
- Feature importance analysis
- Model validation

### ⏭️ **Phase 5: Dose Optimization**

Integrate PK model + ML models for:
- Bayesian dose optimization
- Multi-objective optimization (efficacy + safety)
- Patient-specific recommendations

### ⏭️ **Phase 6: Validation**

- External validation
- Monte Carlo simulation
- Performance assessment

---

## Code to Reproduce

```bash
# Run Phase 1 preprocessing
python3 phase1_preprocessing.py

# Outputs will be saved to data/processed/
```

---

## Validation Metrics

All preprocessing steps have been validated:

✅ **Derived variables:** Formulas verified against medical literature
✅ **CrCL calculation:** Cockcroft-Gault formula applied correctly
✅ **PK metrics:** Calculated from observed concentrations
✅ **Data quality:** Comprehensive checks performed
✅ **Outputs:** Multiple formats for different analyses

---

## References

### Formulas Used

1. **BSA (Mosteller):**
   ```
   BSA (m²) = √[(height(cm) × weight(kg)) / 3600]
   ```

2. **IBW (Devine):**
   ```
   Male: IBW = 50 + 2.3 × (height(inches) - 60)
   Female: IBW = 45.5 + 2.3 × (height(inches) - 60)
   ```

3. **ABW (Adjusted for obesity):**
   ```
   ABW = IBW + 0.4 × (actual weight - IBW)
   ```

4. **CrCL (Cockcroft-Gault):**
   ```
   CrCL = [(140 - age) × weight × (0.85 if female)] / (72 × SCr)
   ```

---

## Summary Statistics

| Category | Metric | Value |
|----------|--------|-------|
| **Patients** | Total | 300 |
| **Data Quality** | Complete | 100% |
| **Derived Variables** | BSA | 1.67 ± 0.16 m² |
| | IBW | 56.7 ± 8.7 kg |
| | ABW | 59.0 ± 7.0 kg |
| **PK Metrics** | Mean Cmax | 51.84 mg/L |
| | Mean Cmax/MIC | 10.72 |
| | Target ≥8 | 32.0% |
| **Datasets Created** | ML dataset | 300×63 |
| | PopPK dataset | 3,631 rows |

---

**Phase 1 Status: ✅ COMPLETE AND VALIDATED**

Ready to proceed to **Phase 2: Population PK Modeling**!
