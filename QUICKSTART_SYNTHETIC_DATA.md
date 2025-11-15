# Quick Start: Synthetic Data Generation

## Prerequisites

Ensure you have R installed (version 4.0 or higher):

```bash
R --version
```

## Step 1: Install Required Packages

Open R and run:

```r
install.packages(c("tidyverse", "lubridate", "MASS"))
```

## Step 2: Generate Synthetic Data

From the project root directory:

```bash
Rscript generate_synthetic_data.R
```

**Expected output:**
```
================================================================================
SYNTHETIC DATA GENERATION - INDIAN ICU AMINOGLYCOSIDE STUDY
================================================================================

Generating synthetic aminoglycoside data for 300 patients...
Population: Indian ICU setting

Step 1/5: Generating patient demographics and baseline data...
Step 2/5: Generating dosing records...
Step 3/5: Simulating PK concentrations...
Step 4/5: Generating time-varying covariates...
Step 5/5: Simulating clinical outcomes...

Data generation complete!
Summary:
  Patients: 300
  Dosing records: ~1200
  PK samples: ~1200
  Time-varying records: 2400

Data saved to: data/synthetic_aminoglycoside_data.rds

CSV files also saved to 'data/' directory

================================================================================
DATA SUMMARY
================================================================================

Patient Demographics:
  Age:  52 ± 16 years
  Weight:  62 ± 12 kg
  Male:  65 %

Clinical Severity:
  APACHE II:  22 ± 8
  SOFA:  9 ± 4
  Septic shock:  40 %

Outcomes:
  Clinical cure:  55-65 %
  Nephrotoxicity:  15-25 %
  ICU mortality:  20-30 %

PK/PD Metrics:
  Mean Cmax/MIC:  9-12
  Target (Cmax/MIC ≥8):  50-70 %

================================================================================
READY FOR ANALYSIS!
================================================================================
```

## Step 3: Verify Data Files

Check that the following files were created:

```bash
ls -lh data/

# Expected files:
# - synthetic_aminoglycoside_data.rds  (main R data file)
# - patient_data.csv
# - dosing.csv
# - concentrations.csv
# - time_varying.csv
# - outcomes.csv
```

## Step 4: Load and Inspect Data

```r
# Load the complete dataset
synthetic_data <- readRDS("data/synthetic_aminoglycoside_data.rds")

# View structure
names(synthetic_data)
# [1] "patient_data"  "time_varying"  "dosing"  "concentrations"  "outcomes"  "metadata"

# Inspect patient data
head(synthetic_data$patient_data)
summary(synthetic_data$patient_data)

# Check sample sizes
nrow(synthetic_data$patient_data)        # 300 patients
nrow(synthetic_data$concentrations)      # ~1200 samples
nrow(synthetic_data$dosing)             # ~1200 dose records

# View metadata
synthetic_data$metadata
```

## Step 5: Run Preprocessing

```r
source("PHASE 1: DATA COLLECTION & PREPROCESSING/1.2 Data Preprocessing Script.R")

# Preprocess the synthetic data
processed_data <- preprocess_aminoglycoside_data("data/synthetic_aminoglycoside_data.rds")

# View preprocessing results
processed_data$preprocessing_metadata
```

## Step 6: Proceed with Analysis

You can now run the complete analytical pipeline:

### Phase 2: Population PK Modeling

```r
source("PHASE 2: POPULATION PK MODELING/2.2 Base Structural Model Development.R")
# Run PopPK model fitting...
```

### Phase 3: PK/PD Modeling

```r
source("PHASE 3: PHARMACODYNAMIC MODELING/3.1 PK-PD Linkage Models.R")
# Run PK/PD analysis...
```

### Phase 4: Machine Learning

```r
source("PHASE 4: MACHINE LEARNING INTEGRATION/4.1 Feature Engineering for ML Models.R")
source("PHASE 4: MACHINE LEARNING INTEGRATION/4.2 ML Models for Multiple Objectives.R")
# Train ML models...
```

### Phase 5: Bayesian Optimization

```r
source("PHASE 5: INTEGRATED QSP-ML FRAMEWORK/5.1 Bayesian Dose Optimization.R")
# Run dose optimization...
```

### Phase 6: Validation

```r
source("PHASE 6: MODEL VALIDATION & SIMULATION/6.1 Monte Carlo Simulation Framework.R")
source("PHASE 6: MODEL VALIDATION & SIMULATION/6.2 External Validation Framework.R")
# Run validation analyses...
```

## Customization

### Change Number of Patients

Edit `generate_synthetic_data.R` and modify the function call:

```r
synthetic_data <- generate_synthetic_aminoglycoside_data(
  n_patients = 500,           # Change from 300 to 500
  study_duration_days = 7,
  seed = 12345
)
```

### Modify Population Characteristics

Edit the `indian_icu_params` list in `generate_synthetic_data.R`:

```r
indian_icu_params <- list(
  age_mean = 55,              # Increase average age
  diabetes_prevalence = 0.45, # Higher diabetes rate
  apache_ii_mean = 25,        # Higher severity
  # ... other parameters
)
```

### Use Different Random Seed

```r
synthetic_data <- generate_synthetic_aminoglycoside_data(
  n_patients = 300,
  study_duration_days = 7,
  seed = 99999               # Different seed for different data
)
```

## Troubleshooting

### Error: "Package 'tidyverse' not found"

```r
install.packages("tidyverse")
```

### Error: "Cannot find function"

Make sure you're running the script from the project root directory:

```bash
cd /path/to/aminoglycoside_QSP_ML
Rscript generate_synthetic_data.R
```

### Data looks unrealistic

- Check that covariate ranges make sense
- Verify PK/PD relationships are correct
- Ensure outcome rates align with literature

### Want to see individual patient profiles

```r
library(tidyverse)

# Load data
data <- readRDS("data/synthetic_aminoglycoside_data.rds")

# View one patient's complete data
patient_id <- "IND_0001"

# Demographics
data$patient_data %>% filter(patient_id == !!patient_id)

# Doses
data$dosing %>% filter(patient_id == !!patient_id)

# Concentrations
data$concentrations %>% filter(patient_id == !!patient_id)

# Time-varying
data$time_varying %>% filter(patient_id == !!patient_id)

# Outcome
data$outcomes %>% filter(patient_id == !!patient_id)
```

## Next Steps

After generating and verifying synthetic data:

1. ✅ **Explore the data** - Create visualizations, check distributions
2. ✅ **Run preprocessing** - Clean and prepare for analysis
3. ✅ **Fit PopPK model** - Should recover input parameters
4. ✅ **Train ML models** - Use for outcome prediction
5. ✅ **Test optimization** - Bayesian dose optimization
6. ✅ **Validate framework** - Monte Carlo simulation

When you have **real clinical data**, simply replace the synthetic dataset with your actual data in the same format!

## Questions?

See `README_SYNTHETIC_DATA.md` for detailed documentation on:
- Data specifications
- Scientific validity
- Population characteristics
- Model equations
- References
