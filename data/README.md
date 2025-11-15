# Synthetic Aminoglycoside Data - Indian ICU Setting

**Generated:** 2025-11-15
**Population:** Indian ICU patients
**Sample Size:** 300 patients

---

## Overview

This directory contains scientifically valid synthetic data for testing the aminoglycoside QSP-ML optimization framework. The data reflects real-world characteristics of Indian ICU patients receiving aminoglycoside antibiotics.

---

## Files

### 1. **patient_data.csv** (300 rows)
Patient demographics and baseline characteristics

**Key Variables:**
- `patient_id`: Unique identifier (IND_0001 to IND_0300)
- `age`, `sex`, `weight`, `height`, `bmi`: Demographics
- `apache_ii`, `sofa_score`, `sepsis_type`: Clinical severity
- `baseline_scr`, `baseline_crcl`, `baseline_egfr`: Renal function
- `infection_site`, `pathogen`, `mic_amikacin`, `mic_gentamicin`: Infection data
- `diabetes`, `ckd_stage`, `mechanical_ventilation`, `vasopressor_use`: Comorbidities
- `hospital_id`, `icu_type`: Study site information

### 2. **dosing.csv** (1,432 rows)
Aminoglycoside dosing records

**Key Variables:**
- `patient_id`: Links to patient_data
- `time`: Hours from study start
- `dose`: Dose in mg
- `infusion_duration`: Duration in hours (typically 1.0)
- `route`: Route of administration (IV)

**Summary:**
- ~4.8 doses per patient
- 75% amikacin, 25% gentamicin
- Amikacin: 15-20 mg/kg once daily
- Gentamicin: 5-7 mg/kg once daily
- Intervals adjusted for renal function (24-48 hours)

### 3. **concentrations.csv** (2,199 rows)
PK concentration measurements

**Key Variables:**
- `patient_id`: Links to patient_data
- `time`: Hours from first dose
- `sample_time_from_dose`: Hours from last dose
- `concentration`: Drug concentration (mg/L)
- `sample_type`: peak/trough/random
- `bloq`: Below limit of quantification flag
- `dose_number`: Which dose this relates to

**Summary:**
- ~7.3 samples per patient
- 53.0% peak samples (1 hour post-infusion)
- 42.1% trough samples (pre-dose)
- 4.9% random samples
- 19.7% BLOQ (mostly trough samples)

### 4. **time_varying.csv** (2,400 rows)
Daily time-varying covariates (8 days × 300 patients)

**Key Variables:**
- `patient_id`: Links to patient_data
- `time`: Hours from study start
- `scr`, `crcl_cg`: Renal function markers
- `urine_output`: Daily urine output (mL/24h)
- `fluid_balance`: Net fluid balance (L)
- `wbc`, `crp`, `procalcitonin`: Inflammatory markers
- `rrt_status`, `rrt_type`: Renal replacement therapy
- `map`, `norepinephrine_dose`: Hemodynamics

**Summary:**
- Daily measurements over 7 days
- Realistic trajectories (improving inflammation, variable renal function)
- 18% initiate RRT during study period

### 5. **outcomes.csv** (300 rows)
Clinical outcomes

**Key Variables:**

*Efficacy:*
- `clinical_cure`: TRUE/FALSE
- `microbiological_eradication`: TRUE/FALSE
- `time_to_clinical_improvement`: Days

*Safety:*
- `nephrotoxicity`: TRUE/FALSE (RIFLE/KDIGO criteria)
- `aki_stage`: 0/1/2/3
- `peak_scr`: Maximum creatinine
- `ototoxicity`, `neurotoxicity`: TRUE/FALSE

*Overall:*
- `icu_los`, `hospital_los`: Length of stay (days)
- `icu_mortality`, `day_28_mortality`: TRUE/FALSE

*PK/PD:*
- `achieved_cmax_mic`: Observed Cmax/MIC ratio
- `achieved_auc_mic`: Estimated AUC/MIC ratio

**Summary:**
- Clinical cure: 53.7%
- Nephrotoxicity: 20.0%
- ICU mortality: 20.3%
- Mean Cmax/MIC: 11.5
- Target attainment (Cmax/MIC ≥8): 35.3%

### 6. **synthetic_aminoglycoside_data.pkl**
Python pickle file containing all datasets in a single object

**Load in Python:**
```python
import pickle
with open('data/synthetic_aminoglycoside_data.pkl', 'rb') as f:
    data = pickle.load(f)

patient_data = data['patient_data']
concentrations = data['concentrations']
# etc.
```

### 7. **metadata.json**
Generation metadata and parameters

---

## Population Characteristics

### Demographics (Indian ICU)
- **Age:** 51.2 ± 15.4 years (range: 18-90)
- **Weight:** 62.6 ± 11.3 kg (lower than Western populations)
- **Height:** 161.5 ± 9.3 cm
- **BMI:** 24.2 ± 5.2 kg/m²
- **Male:** 65.0%

### Clinical Severity
- **APACHE II:** 21.9 ± 7.9 (higher than Western ICUs)
- **SOFA:** 9.3 ± 4.2
- **Septic shock:** 40.3% (high proportion)
- **Mechanical ventilation:** 67.3%
- **Vasopressor use:** 50.3%

### Comorbidities
- **Diabetes:** 34.7% (high in Indian population)
- **CKD stages 2-4:** ~30%
- **Augmented renal clearance (CrCL >130):** 24.7%

### Infection Profile
**Sites:**
- Pneumonia: 34.7%
- Bloodstream: 32.7%
- UTI: 12.7%
- Intra-abdominal: 12.3%

**Pathogens:**
- Klebsiella: 38.3% (very common in India)
- Pseudomonas: 24.0%
- E. coli: 16.7%
- Acinetobacter: 15.0% (increasing resistance)

**MIC Distribution:**
- Gentamicin: Median 4.91 μg/mL (higher resistance)
- Amikacin: Median 11.0 μg/mL

---

## Scientific Validity

### PK Model
- **Two-compartment model** with first-order elimination
- **Typical parameters:**
  - CL: ~5.5 L/h (scaled by renal function and weight)
  - Vc: ~16 L (scaled by weight)
  - Q: ~12 L/h
  - Vp: ~10 L
- **Covariate effects:**
  - CL ∝ CrCL^0.75 × Weight^0.75
  - Vc ∝ Weight^1.0
  - Septic shock: -15% CL
- **BSV:** CL 25%, Vc 20%, Q 40%, Vp 35%
- **Residual error:** 15% proportional + 0.5 mg/L additive

### PK/PD Relationships
**Clinical cure:**
```
logit(P_cure) = -2 + 1.5×log(Cmax/MIC) - 0.03×APACHE_II - 0.5×(Acinetobacter)
```

**Nephrotoxicity:**
```
logit(P_nephrotox) = -3 + 1.2×log(Trough) + 0.02×Age + 0.5×Diabetes + 0.4×(SCr>1.5)
```

**Mortality:**
```
logit(P_death) = -4 + 0.08×APACHE_II + 0.1×SOFA - 0.8×Cure + 0.5×Nephrotox
```

---

## Usage

### Load in R
```r
# Load individual CSV files
library(readr)
patient_data <- read_csv("data/patient_data.csv")
dosing <- read_csv("data/dosing.csv")
concentrations <- read_csv("data/concentrations.csv")
time_varying <- read_csv("data/time_varying.csv")
outcomes <- read_csv("data/outcomes.csv")
```

### Load in Python
```python
import pandas as pd

patient_data = pd.read_csv('data/patient_data.csv')
dosing = pd.read_csv('data/dosing.csv')
concentrations = pd.read_csv('data/concentrations.csv')
time_varying = pd.read_csv('data/time_varying.csv')
outcomes = pd.read_csv('data/outcomes.csv')

# Or load all at once
import pickle
with open('data/synthetic_aminoglycoside_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

---

## Data Quality

✅ **Completeness:**
- No missing values in critical variables
- All patients have dosing, concentration, and outcome data
- 300/300 patients complete (100%)

✅ **Validity:**
- All values within biologically plausible ranges
- No duplicate patient IDs
- Consistent patient IDs across datasets

✅ **Realism:**
- Demographics match published Indian ICU data
- MIC distributions reflect increasing resistance
- Outcome rates align with literature
- PK/PD relationships are scientifically sound

---

## Next Steps

1. **Explore the data:**
   ```r
   summary(patient_data)
   table(patient_data$sepsis_type)
   hist(concentrations$concentration)
   ```

2. **Run preprocessing:**
   ```r
   source("PHASE 1: DATA COLLECTION & PREPROCESSING/1.2 Data Preprocessing Script.R")
   processed <- preprocess_aminoglycoside_data("data/synthetic_aminoglycoside_data.rds")
   ```

3. **Fit PopPK model:**
   ```r
   source("PHASE 2: POPULATION PK MODELING/2.2 Base Structural Model Development.R")
   ```

4. **Continue with Phases 3-6...**

---

## Regeneration

To regenerate with different parameters:

```bash
# Python version
python3 generate_synthetic_data_python.py

# Or R version (if R is available)
Rscript generate_synthetic_data.R
```

Modify population parameters in the script to customize:
- Sample size (n_patients)
- Study duration
- Population characteristics (age, weight, severity)
- Random seed for reproducibility

---

## Citation

If using this data in publications:

```
Synthetic aminoglycoside data for Indian ICU settings
Generated: 2025-11-15
Framework: Aminoglycoside QSP-ML Optimization
```

---

**For questions or issues, see the main project README.md**
