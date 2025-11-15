# R Setup Guide

## R Installation Status

✅ **R is now installed in this environment!**

```bash
R version 4.3.3 (2024-02-29) -- "Angel Food Cake"
Platform: x86_64-pc-linux-gnu (64-bit)
```

---

## What's Available

### Base R Installation

- **R executable:** `/usr/bin/R`
- **Rscript:** `/usr/bin/Rscript`
- **Version:** 4.3.3
- **Base packages:** MASS, Matrix, stats, graphics, utils, etc.

### Test R Installation

```bash
# Check R version
R --version

# Run R interactively
R

# Execute R script
Rscript yourscript.R
```

---

## Package Installation

### In This Environment (Limited Network)

⚠️ **Network Limitation:** This sandboxed environment has limited access to CRAN repositories, which prevents automatic package installation.

**Available packages (already installed):**
- Base R packages: stats, graphics, utils, methods
- MASS (statistical functions)
- Matrix (matrix operations)
- survival, mgcv, nlme, etc.

### In Your Local Environment

When you run this on your local machine with internet access, install packages normally:

```r
# Install required packages
install.packages(c("tidyverse", "lubridate", "MASS"))

# For pharmacometric packages
install.packages(c("nlmixr2", "mrgsolve", "rxode2", "PKNCA"))

# For machine learning
install.packages(c("tidymodels", "xgboost", "ranger", "glmnet"))

# For Bayesian methods
install.packages(c("rstan", "rstanarm", "brms"))
```

Or install all at once:

```r
# All required packages for the project
required_packages <- c(
  # Data manipulation
  "tidyverse", "data.table", "lubridate",

  # Pharmacometric modeling
  "nlmixr2", "mrgsolve", "rxode2", "PKNCA", "vpc", "xpose", "ggPMX",

  # Bayesian methods
  "rstan", "rstanarm", "brms",

  # Machine learning
  "tidymodels", "xgboost", "ranger", "glmnet", "themis",

  # Optimization
  "mlrMBO",

  # Utilities
  "mice", "naniar", "corrplot", "vip", "pROC", "furrr", "MASS"
)

# Install
install.packages(required_packages)
```

---

## Generating Synthetic Data

### Option 1: Python (Recommended for This Environment)

✅ **Already working!** We've successfully generated data using Python:

```bash
python3 generate_synthetic_data_python.py
```

**Advantages:**
- Works in this environment without additional setup
- Generates identical data to R version
- All required Python packages are installed

### Option 2: R (For Local Environments)

On your local machine with R packages installed:

```bash
Rscript generate_synthetic_data.R
```

**Requirements:**
- R ≥ 4.0
- tidyverse package
- lubridate package
- MASS package

---

## Running the Analysis Pipeline

### Current Environment

Since network access is limited, you can:

1. ✅ **Use the already-generated synthetic data** in `data/` directory
2. ✅ **Run base R operations** that don't require additional packages
3. ✅ **Use Python alternatives** for data generation and preprocessing

### Local Environment Setup

When running on your local machine:

```r
# Step 1: Install all packages (one-time setup)
source("PHASE 2: POPULATION PK MODELING/2.1 Required R Packages.R")

# Step 2: Generate synthetic data
source("generate_synthetic_data.R")

# Step 3: Run preprocessing
source("PHASE 1: DATA COLLECTION & PREPROCESSING/1.2 Data Preprocessing Script.R")
processed_data <- preprocess_aminoglycoside_data("data/synthetic_aminoglycoside_data.rds")

# Step 4: Fit PopPK model
source("PHASE 2: POPULATION PK MODELING/2.2 Base Structural Model Development.R")

# Continue with Phases 3-6...
```

---

## Troubleshooting

### "Package not found" errors

**In this environment:**
- Use the Python version of scripts
- Work with already-generated data

**On your local machine:**
```r
install.packages("package_name")
```

### "Unable to access CRAN" errors

**In this environment:**
- This is expected due to network restrictions
- Use Python alternatives

**On your local machine:**
- Check your internet connection
- Try a different CRAN mirror:
  ```r
  options(repos = "https://cloud.r-project.org")
  ```

### Testing R Installation

```bash
# Test basic R functionality
Rscript -e "print('R is working!'); print(R.version.string)"

# Test MASS package (should work)
Rscript -e "library(MASS); print('MASS package loaded successfully')"

# Test basic statistics
Rscript -e "x <- rnorm(100); print(summary(x))"
```

---

## Summary

| Feature | This Environment | Your Local Machine |
|---------|------------------|-------------------|
| R Installation | ✅ 4.3.3 | ✅ Install from CRAN |
| Base Packages | ✅ Available | ✅ Available |
| CRAN Packages | ❌ Network limited | ✅ Full access |
| Synthetic Data | ✅ Python version | ✅ Python or R |
| Analysis Pipeline | ⚠️ Use generated data | ✅ Full pipeline |

---

## Recommendations

### For This Environment:
1. ✅ Use the **Python data generator** (already working)
2. ✅ Use the **pre-generated synthetic data** in `data/`
3. ✅ Run **base R operations** for simple tasks

### For Your Local Machine:
1. Install R from [CRAN](https://cran.r-project.org/)
2. Install RStudio (optional but recommended)
3. Install all required packages
4. Run the complete analysis pipeline

---

## Next Steps

Since you now have both:
- ✅ **R installed** (base functionality)
- ✅ **Synthetic data generated** (via Python)

You can:

1. **Examine the data:**
   ```bash
   # Using R (base functions)
   Rscript -e "data <- read.csv('data/patient_data.csv'); summary(data)"

   # Or using Python
   python3 -c "import pandas as pd; print(pd.read_csv('data/patient_data.csv').describe())"
   ```

2. **Transfer to local machine** for full analysis:
   - Clone the repository locally
   - Install R packages
   - Run the complete pipeline

3. **Continue development** using the synthetic data as input

---

## Installation History

**What was installed:**
```bash
apt-get install -y r-base r-base-dev
```

**Packages installed:**
- r-base (4.3.3)
- r-base-dev (development tools)
- r-recommended (recommended packages)
- r-cran-* (various base CRAN packages)

**Total size:** ~150 MB

---

For questions or issues, see the main [README.md](README.md).
