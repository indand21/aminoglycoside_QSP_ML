# Synthetic Data Generation for Aminoglycoside QSP-ML Project

## Overview

This document describes how to generate scientifically valid synthetic data for the aminoglycoside optimization project, reflecting real-world Indian ICU settings.

## Quick Start

### Using R (Recommended)

```bash
# Make sure R is installed (version 4.0+)
Rscript generate_synthetic_data.R
```

This will generate:
- **300 synthetic ICU patients** with complete data
- All required datasets (patient demographics, dosing, PK concentrations, time-varying covariates, outcomes)
- Data saved in `data/` directory as both RDS and CSV files

## Data Generated

### 1. Patient Demographics (`patient_data.csv`)
- **n = 300 patients** from Indian ICU settings
- Demographics: Age, sex, weight, height, BMI
- Clinical severity: APACHE II, SOFA scores, sepsis type
- Baseline labs: Creatinine, eGFR, albumin, bilirubin
- Infection data: Site, pathogen, MIC values
- Comorbidities: Diabetes (35%), CKD, mechanical ventilation, vasopressors

**Key Characteristics (Indian Population):**
- Age: 52 ± 16 years (younger than Western ICUs)
- Weight: 62 ± 12 kg (lower BMI)
- Height: 162 ± 10 cm
- APACHE II: 22 ± 8 (higher severity)
- Diabetes: 35% (higher prevalence)
- Septic shock: ~40% (higher proportion)

### 2. Dosing Records (`dosing.csv`)
- **Drug:** 75% amikacin, 25% gentamicin
- **Amikacin:** 15-20 mg/kg once daily
- **Gentamicin:** 5-7 mg/kg once daily
- **Interval:** Adjusted for renal function (24-48 hours)
- **Duration:** 3-7 days of treatment

### 3. PK Concentrations (`concentrations.csv`)
- **~1,200 concentration samples** across all patients
- **Peak samples:** 1 hour after infusion start
- **Trough samples:** Before next dose
- **Random samples:** ~10% of patients
- Simulated using **two-compartment PK model** with:
  - Clearance (CL): Scaled by renal function and weight
  - Volume (Vc): Scaled by body weight
  - Residual error: 15% CV + 0.5 mg/L additive

### 4. Time-Varying Covariates (`time_varying.csv`)
Daily measurements over 7 days:
- **Renal function:** SCr, CrCL, urine output
- **Fluid status:** Input/output, fluid balance
- **Inflammation:** WBC, CRP, procalcitonin
- **Organ support:** RRT status (~18% initiate RRT)
- **Hemodynamics:** MAP, vasopressor doses

### 5. Clinical Outcomes (`outcomes.csv`)
**Efficacy:**
- Clinical cure: ~55-65% (PK/PD dependent)
- Microbiological eradication
- Time to improvement

**Safety:**
- Nephrotoxicity: ~15-25% (trough-dependent)
- AKI staging (KDIGO)
- Ototoxicity: ~3%

**Overall:**
- ICU LOS: 10 ± 4 days
- Hospital LOS: 17 ± 5 days
- ICU mortality: ~20-30%

## Scientific Validity

### Pharmacokinetic Model
- **Two-compartment model** with first-order elimination
- **Covariate effects:**
  - CL: ∝ CrCL^0.75 × Weight^0.75
  - Vc: ∝ Weight^1.0
  - Septic shock: -15% CL
- **Between-subject variability:**
  - CL: 25% CV
  - Vc: 20% CV
  - Q: 40% CV
  - Vp: 35% CV

### MIC Distribution (Indian Settings)
Reflects higher antimicrobial resistance:
- **Gentamicin:** Geometric mean ~4 μg/mL (higher for Acinetobacter, Pseudomonas)
- **Amikacin:** Geometric mean ~8 μg/mL
- Distribution: Log-normal with wider spread

### Pathogen Distribution
Based on Indian ICU epidemiology:
- Klebsiella: 35% (very common)
- Pseudomonas: 25%
- E. coli: 20%
- Acinetobacter: 15% (increasing trend)

### Outcome Models

**Clinical Cure:**
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

## Data Quality Features

✅ **Realistic distributions** based on published Indian ICU data
✅ **Biologically plausible** PK parameters and concentration-time profiles
✅ **PK/PD relationships** linking exposure to efficacy and toxicity
✅ **Correlated variables** (e.g., APACHE II with mortality)
✅ **Missing data handling** (some random samples, optional trough missing)
✅ **Outlier-free** but realistic variability

## Using the Generated Data

### Load in R

```r
# Load complete dataset
synthetic_data <- readRDS("data/synthetic_aminoglycoside_data.rds")

# Access individual components
patient_data <- synthetic_data$patient_data
dosing <- synthetic_data$dosing
concentrations <- synthetic_data$concentrations
time_varying <- synthetic_data$time_varying
outcomes <- synthetic_data$outcomes

# Or load CSVs individually
library(readr)
patient_data <- read_csv("data/patient_data.csv")
```

### Proceed with Analysis Pipeline

Once data is generated, you can run the analysis phases:

```r
# Phase 1: Preprocessing
source("PHASE 1: DATA COLLECTION & PREPROCESSING/1.2 Data Preprocessing Script.R")
processed <- preprocess_aminoglycoside_data("data/synthetic_aminoglycoside_data.rds")

# Phase 2: Population PK Modeling
source("PHASE 2: POPULATION PK MODELING/2.2 Base Structural Model Development.R")

# Phase 3: PK/PD Modeling
source("PHASE 3: PHARMACODYNAMIC MODELING/3.1 PK-PD Linkage Models.R")

# And so on...
```

## Customization

You can modify the generation parameters:

```r
# Edit generate_synthetic_data.R

# Change sample size
synthetic_data <- generate_synthetic_aminoglycoside_data(
  n_patients = 500,           # Increase to 500 patients
  study_duration_days = 10,   # Extend to 10 days
  seed = 54321                # Different random seed
)

# Modify population characteristics
indian_icu_params$age_mean <- 55  # Older population
indian_icu_params$diabetes_prevalence <- 0.45  # Higher diabetes rate
```

## Comparison: Synthetic vs Real Data

| Characteristic | Synthetic | Expected Real (Indian ICU) |
|----------------|-----------|----------------------------|
| Age (years) | 52 ± 16 | 48-55 ± 15-18 |
| Weight (kg) | 62 ± 12 | 55-65 ± 10-15 |
| APACHE II | 22 ± 8 | 18-25 ± 7-10 |
| Diabetes | 35% | 30-40% |
| Septic shock | 40% | 35-45% |
| MV requirement | 65% | 60-70% |
| Nephrotoxicity | 15-25% | 15-30% |
| ICU mortality | 20-30% | 20-35% |
| Cmax/MIC ≥8 | 50-70% | 40-65% |

## References

The synthetic data generation is based on:

1. **Indian ICU epidemiology:**
   - Indian J Crit Care Med 2019-2023 issues
   - INDICAPS study data

2. **Aminoglycoside PK:**
   - Nicolau et al. (1995) - Two-compartment model
   - Begg et al. (2001) - Dosing in obesity

3. **PK/PD targets:**
   - Moore et al. (1987) - Cmax/MIC ≥8-10
   - Kashuba et al. (1999) - AUC/MIC targets

4. **Nephrotoxicity:**
   - Rybak et al. (2020) - Aminoglycoside toxicity
   - KDIGO guidelines for AKI

## Troubleshooting

**Issue:** R packages not installed

```r
# Install all required packages
install.packages(c("tidyverse", "lubridate", "MASS"))
```

**Issue:** Need different population characteristics

Edit `indian_icu_params` at the top of `generate_synthetic_data.R`

**Issue:** Want to generate data in batches

```r
# Generate multiple datasets with different seeds
for (i in 1:5) {
  data <- generate_synthetic_aminoglycoside_data(
    n_patients = 100,
    seed = 10000 + i
  )
  saveRDS(data, paste0("data/synthetic_batch_", i, ".rds"))
}
```

## Next Steps

After generating synthetic data:

1. ✅ **Verify data quality** - Check distributions, ranges, correlations
2. ✅ **Run preprocessing** - Use Phase 1 scripts
3. ✅ **Develop PopPK model** - Phase 2 (should recover input parameters)
4. ✅ **Build ML models** - Phase 4
5. ✅ **Test optimization** - Phase 5
6. ✅ **Validate framework** - Phase 6

This synthetic dataset allows you to:
- Test the entire analytical pipeline
- Develop and debug code
- Demonstrate the framework
- Train ML models
- Validate that PopPK can recover known parameters

**When you have real data, simply replace the synthetic dataset with your actual clinical data in the same format!**
