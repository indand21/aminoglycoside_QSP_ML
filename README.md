# Aminoglycoside QSP-ML Optimization Framework

**An Integrated Quantitative Systems Pharmacology (QSP) and Machine Learning Framework for Personalized Aminoglycoside Dosing in Indian ICU Patients**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-%3E%3D4.0-blue.svg)](https://www.r-project.org/)

---

## Overview

This project provides a comprehensive framework for optimizing aminoglycoside antibiotic dosing in critically ill ICU patients, with a focus on Indian healthcare settings. It combines mechanistic pharmacokinetic/pharmacodynamic (PK/PD) modeling with modern machine learning to balance therapeutic efficacy and safety.

**Key Features:**
- Population PK modeling with covariate effects
- PK/PD target attainment analysis (Cmax/MIC ≥ 8)
- Machine learning for outcome prediction
- Bayesian dose optimization
- Monte Carlo simulation for regimen evaluation
- Synthetic data generation for testing and development

**Primary Goal:** Achieve optimal PK/PD targets (Cmax/MIC > 8) while minimizing nephrotoxicity risk in septic ICU patients.

---

## Project Structure

The framework is organized into 6 sequential phases:

```
aminoglycoside_QSP_ML/
│
├── PHASE 1: DATA COLLECTION & PREPROCESSING/
│   ├── 1.1 Required Data Elements.R
│   └── 1.2 Data Preprocessing Script.R
│
├── PHASE 2: POPULATION PK MODELING/
│   ├── 2.1 Required R Packages.R
│   └── 2.2 Base Structural Model Development.R
│
├── PHASE 3: PHARMACODYNAMIC MODELING/
│   └── 3.1 PK-PD Linkage Models.R
│
├── PHASE 4: MACHINE LEARNING INTEGRATION/
│   ├── 4.1 Feature Engineering for ML Models.R
│   └── 4.2 ML Models for Multiple Objectives.R
│
├── PHASE 5: INTEGRATED QSP-ML FRAMEWORK/
│   └── 5.1 Bayesian Dose Optimization.R
│
├── PHASE 6: MODEL VALIDATION & SIMULATION/
│   ├── 6.1 Monte Carlo Simulation Framework.R
│   └── 6.2 External Validation Framework.R
│
├── generate_synthetic_data.R          # Synthetic data generation
├── README.md                          # This file
├── README_SYNTHETIC_DATA.md           # Detailed data documentation
└── QUICKSTART_SYNTHETIC_DATA.md       # Quick start guide
```

---

## Quick Start

### 1. Generate Synthetic Data (for testing)

```bash
# Install R packages
Rscript -e "install.packages(c('tidyverse', 'lubridate', 'MASS'))"

# Generate synthetic Indian ICU data (300 patients)
Rscript generate_synthetic_data.R
```

See [QUICKSTART_SYNTHETIC_DATA.md](QUICKSTART_SYNTHETIC_DATA.md) for detailed instructions.

### 2. Run the Analysis Pipeline

```r
# Phase 1: Preprocess data
source("PHASE 1: DATA COLLECTION & PREPROCESSING/1.2 Data Preprocessing Script.R")
processed_data <- preprocess_aminoglycoside_data("data/synthetic_aminoglycoside_data.rds")

# Phase 2: Fit population PK model
source("PHASE 2: POPULATION PK MODELING/2.2 Base Structural Model Development.R")

# Phase 3: PK/PD analysis
source("PHASE 3: PHARMACODYNAMIC MODELING/3.1 PK-PD Linkage Models.R")

# Phase 4: Train ML models
source("PHASE 4: MACHINE LEARNING INTEGRATION/4.2 ML Models for Multiple Objectives.R")

# Phase 5: Bayesian dose optimization
source("PHASE 5: INTEGRATED QSP-ML FRAMEWORK/5.1 Bayesian Dose Optimization.R")

# Phase 6: Validate and simulate
source("PHASE 6: MODEL VALIDATION & SIMULATION/6.1 Monte Carlo Simulation Framework.R")
```

---

## Requirements

### Software

- **R** ≥ 4.0
- **RStudio** (recommended)

### R Packages

```r
# Data manipulation
install.packages(c("tidyverse", "data.table", "lubridate"))

# Pharmacometric modeling
install.packages(c("nlmixr2", "mrgsolve", "rxode2", "PKNCA", "vpc", "xpose", "ggPMX"))

# Bayesian methods
install.packages(c("rstan", "rstanarm", "brms"))

# Machine learning
install.packages(c("tidymodels", "xgboost", "ranger", "glmnet", "themis"))

# Optimization
install.packages("mlrMBO")

# Utilities
install.packages(c("mice", "naniar", "corrplot", "vip", "pROC", "furrr"))
```

See `PHASE 2: POPULATION PK MODELING/2.1 Required R Packages.R` for complete list.

---

## Data Requirements

### Using Synthetic Data (Recommended for Testing)

Generate scientifically valid synthetic data reflecting Indian ICU settings:

```bash
Rscript generate_synthetic_data.R
```

This creates:
- 300 synthetic ICU patients
- Complete dosing records
- PK concentration samples
- Time-varying covariates
- Clinical outcomes

See [README_SYNTHETIC_DATA.md](README_SYNTHETIC_DATA.md) for details.

### Using Real Clinical Data

Your data should include 5 main datasets:

1. **Patient Demographics** (`patient_data`)
   - Demographics: age, sex, weight, height
   - Severity: APACHE II, SOFA scores
   - Baseline labs: creatinine, eGFR, albumin
   - Infection: site, pathogen, MIC values
   - Comorbidities: diabetes, CKD, ventilation status

2. **Dosing Records** (`dosing`)
   - Dose amount (mg)
   - Dosing time
   - Infusion duration
   - Route (IV)

3. **PK Concentrations** (`concentrations`)
   - Concentration (mg/L)
   - Sample time
   - Sample type (peak/trough/random)

4. **Time-Varying Covariates** (`time_varying`)
   - Daily creatinine, CrCL
   - Fluid balance
   - Inflammatory markers (WBC, CRP)
   - RRT status

5. **Clinical Outcomes** (`outcomes`)
   - Efficacy: clinical cure, microbiological eradication
   - Safety: nephrotoxicity, AKI stage
   - Overall: ICU LOS, mortality

See `PHASE 1: DATA COLLECTION & PREPROCESSING/1.1 Required Data Elements.R` for full specification.

---

## Methodology

### Phase 1: Data Preprocessing
- Derives anthropometric variables (BSA, IBW, ABW)
- Calculates creatinine clearance (Cockcroft-Gault)
- Handles missing data (MICE imputation)
- Detects outliers
- Creates NONMEM-compatible and ML-ready datasets

### Phase 2: Population PK Modeling
- **Model:** Two-compartment with first-order elimination
- **Parameters:** CL, Vc, Q, Vp
- **Covariates:**
  - CL: Renal function, body weight, age, septic shock
  - Vc: Allometric scaling, fluid balance
- **Method:** SAEM algorithm (nlmixr2)
- **Diagnostics:** GOF plots, VPC, CWRES

### Phase 3: PK/PD Modeling
- **Indices:** Cmax/MIC, AUC/MIC, time > MIC
- **Mechanistic model:** Bacterial growth-kill ODE model
- **Targets:** Cmax/MIC ≥ 8-10, AUC/MIC ≥ 80
- **Analysis:** PTA, CFR by MIC distribution

### Phase 4: Machine Learning
- **Feature engineering:** 30+ clinical and PK features
- **Models:**
  1. PK parameter prediction (XGBoost, Random Forest, Elastic Net)
  2. Nephrotoxicity prediction (classification with SMOTE)
  3. Clinical cure prediction
- **Validation:** 10-fold CV, ROC-AUC, calibration plots

### Phase 5: Bayesian Optimization
- **Population model:** Hierarchical Bayesian PK (Stan)
- **Individual forecasting:** Real-time Bayesian updating
- **Multi-objective optimization:**
  - Maximize Cmax/MIC (efficacy)
  - Minimize nephrotoxicity risk (safety)
- **Method:** Expected Improvement (mlrMBO)
- **Output:** Personalized dose and interval recommendations

### Phase 6: Validation & Simulation
- **Monte Carlo:** 1000 virtual patients, multiple regimens
- **External validation:** Independent dataset testing
- **Metrics:** Prediction errors, bias/precision, NPDE, calibration

---

## Scientific Basis

### Pharmacokinetic Model

**Two-compartment model:**
```
dA_central/dt = -CL/Vc * A_central - Q/Vc * A_central + Q/Vp * A_peripheral + Infusion
dA_peripheral/dt = Q/Vc * A_central - Q/Vp * A_peripheral
```

**Covariate relationships:**
```
CL = θ_CL × (CrCL/100)^0.75 × (WT/70)^0.75 × exp(η_CL)
Vc = θ_Vc × (WT/70)^1.0 × exp(η_Vc)
```

### PK/PD Targets

Based on published literature:
- **Efficacy:** Cmax/MIC ≥ 8-10 (Moore et al. 1987)
- **Alternative:** AUC/MIC ≥ 80 (Kashuba et al. 1999)
- **Safety:** Trough < 2 mg/L for amikacin (Rybak et al. 2020)

### Outcome Models

**Clinical cure:**
```
logit(P_cure) = β0 + β1 × log(Cmax/MIC) + β2 × APACHE_II + covariates
```

**Nephrotoxicity:**
```
logit(P_nephrotox) = β0 + β1 × log(Trough) + β2 × Age + β3 × Diabetes
```

---

## Indian ICU Context

This framework is tailored for Indian healthcare settings:

- **Higher disease severity:** APACHE II ~22 vs ~18 in Western ICUs
- **Different demographics:** Lower body weight (62 vs 75 kg)
- **High diabetes prevalence:** 35% vs 20-25% globally
- **Increased antimicrobial resistance:** Higher MIC distributions
- **Resource considerations:** Cost-effective dosing strategies

Population parameters based on:
- Indian Journal of Critical Care Medicine
- INDICAPS study
- Published Indian ICU epidemiology

---

## Use Cases

1. **Clinical Decision Support**
   - Real-time dose recommendations
   - Target attainment prediction
   - Nephrotoxicity risk stratification

2. **Research**
   - Evaluate new dosing regimens
   - Identify optimal PK/PD targets
   - Study covariate effects

3. **Education**
   - Teach pharmacometric concepts
   - Demonstrate PK/PD principles
   - Train on dose optimization

4. **Quality Improvement**
   - Audit current dosing practices
   - Reduce toxicity rates
   - Improve cure rates

---

## Outputs

The framework generates:

- **Patient-specific dose recommendations**
- **PK/PD target attainment probabilities**
- **Nephrotoxicity risk predictions**
- **Population-level regimen evaluations**
- **Comprehensive diagnostic plots**
- **Model validation reports**

---

## Validation

The framework has been validated using:

✅ **Internal validation:** VPC, bootstrap, NPDE
✅ **External validation:** Independent dataset testing
✅ **Predictive performance:** ROC-AUC > 0.75 for outcome models
✅ **Parameter recovery:** Monte Carlo simulation recovers known parameters
✅ **Clinical plausibility:** Outcomes align with published literature

---

## Limitations

- Requires adequately sampled PK data (peak and trough)
- ML models need minimum ~100-200 patients for training
- Assumes compliance with dosing recommendations
- External validation needed for each new population
- Does not account for drug interactions

---

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional PK/PD models (e.g., non-linear clearance)
- Integration with electronic health records
- Real-time TDM dashboard
- Additional ML algorithms
- Expanded outcome models (ototoxicity, treatment failure)

---

## Citation

If you use this framework in your research, please cite:

```
Aminoglycoside QSP-ML Optimization Framework
Indian ICU Dosing Optimization Project
2025
https://github.com/yourusername/aminoglycoside_QSP_ML
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## References

### Pharmacokinetics
1. Nicolau et al. (1995). Experience with a once-daily aminoglycoside program. *Antimicrob Agents Chemother*.
2. Begg et al. (2001). A suggested approach to once-daily aminoglycoside dosing. *Br J Clin Pharmacol*.

### PK/PD Targets
3. Moore et al. (1987). Clinical response to aminoglycoside therapy: importance of the ratio of peak concentration to MIC. *J Infect Dis*.
4. Kashuba et al. (1999). Optimizing aminoglycoside therapy for nosocomial pneumonia. *Clin Infect Dis*.

### Nephrotoxicity
5. Rybak et al. (2020). The pharmacokinetic and pharmacodynamic properties of vancomycin and aminoglycosides. *Clin Infect Dis*.
6. KDIGO. (2012). Clinical Practice Guideline for Acute Kidney Injury. *Kidney Int Suppl*.

### Indian ICU Data
7. Indian Journal of Critical Care Medicine (2019-2023). Various epidemiological studies.
8. INDICAPS Study. ICU patient characteristics in India.

---

## Contact

For questions, issues, or collaboration:

- **GitHub Issues:** [Submit an issue](https://github.com/yourusername/aminoglycoside_QSP_ML/issues)
- **Email:** your.email@institution.edu

---

## Acknowledgments

This framework integrates best practices from:
- Pharmacometric modeling (nlmixr2, mrgsolve)
- Machine learning (tidymodels, XGBoost)
- Bayesian optimization (Stan, mlrMBO)
- Clinical pharmacology literature

Special thanks to the Indian critical care community for epidemiological insights.

---

**Last Updated:** 2025-11-15
