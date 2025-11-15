# ============================================================================
# AMINOGLYCOSIDE DATA PREPROCESSING PIPELINE
# ============================================================================

# Required packages
required_packages <- c(
  "tidyverse",      # Data manipulation
  "data.table",     # Fast data handling
  "lubridate",      # Date/time handling
  "mice",           # Multiple imputation
  "caret",          # ML preprocessing
  "moments",        # Distribution analysis
  "naniar",         # Missing data visualization
  "corrplot"        # Correlation matrices
)

lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
})

# ----------------------------------------------------------------------------
# 1. DATA IMPORT AND INITIAL CLEANING
# ----------------------------------------------------------------------------

preprocess_aminoglycoside_data <- function(raw_data_path) {
  
  # Import data
  cat("Loading raw data...\n")
  data <- readRDS(raw_data_path)
  
  # ----------------------------------------------------------------------------
  # 2. DERIVED VARIABLES CALCULATION
  # ----------------------------------------------------------------------------
  
  cat("Calculating derived variables...\n")
  
  # Body surface area (Mosteller formula)
  data$patient_data <- data$patient_data %>%
    mutate(
      bsa = sqrt((height * weight) / 3600),
      
      # Ideal body weight (Devine formula)
      ibw = case_when(
        sex == "M" ~ 50 + 2.3 * ((height / 2.54) - 60),
        sex == "F" ~ 45.5 + 2.3 * ((height / 2.54) - 60)
      ),
      
      # Adjusted body weight (for obese patients)
      abw = ibw + 0.4 * (weight - ibw),
      
      # Obesity classification
      obesity_class = case_when(
        bmi < 18.5 ~ "underweight",
        bmi < 25 ~ "normal",
        bmi < 30 ~ "overweight",
        bmi < 35 ~ "obese_class1",
        bmi < 40 ~ "obese_class2",
        TRUE ~ "obese_class3"
      )
    )
  
  # Calculate creatinine clearance (Cockcroft-Gault)
  data$patient_data <- data$patient_data %>%
    mutate(
      baseline_crcl = ((140 - age) * weight * ifelse(sex == "M", 1, 0.85)) / 
                      (72 * baseline_scr),
      
      # Augmented renal clearance flag
      arc = baseline_crcl > 130
    )
  
  # Time-varying CrCl
  data$time_varying <- data$time_varying %>%
    left_join(select(data$patient_data, patient_id, age, sex, weight), 
              by = "patient_id") %>%
    mutate(
      crcl_cg = ((140 - age) * weight * ifelse(sex == "M", 1, 0.85)) / 
                (72 * scr)
    )
  
  # ----------------------------------------------------------------------------
  # 3. PK/PD METRICS CALCULATION
  # ----------------------------------------------------------------------------
  
  cat("Calculating PK/PD metrics...\n")
  
  # For each patient, calculate actual Cmax and AUC from observed concentrations
  pk_metrics <- data$concentrations %>%
    group_by(patient_id) %>%
    summarize(
      observed_cmax = max(concentration, na.rm = TRUE),
      n_samples = n(),
      has_peak = any(sample_type == "peak"),
      has_trough = any(sample_type == "trough")
    )
  
  # Join with dosing and MIC to get PK/PD indices
  data$outcomes <- data$outcomes %>%
    left_join(pk_metrics, by = "patient_id") %>%
    left_join(
      data$patient_data %>% 
        select(patient_id, mic_amikacin, mic_gentamicin),
      by = "patient_id"
    ) %>%
    mutate(
      # Calculate Cmax/MIC (use appropriate MIC based on drug)
      observed_cmax_mic_ratio = observed_cmax / 
        coalesce(mic_amikacin, mic_gentamicin)
    )
  
  # ----------------------------------------------------------------------------
  # 4. MISSING DATA ANALYSIS
  # ----------------------------------------------------------------------------
  
  cat("Analyzing missing data patterns...\n")
  
  # Visualize missing data
  missing_summary <- data$patient_data %>%
    summarize(across(everything(), ~sum(is.na(.))/n()*100)) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
    arrange(desc(pct_missing))
  
  print(missing_summary)
  
  # Missing data visualization
  vis_miss(data$patient_data)
  
  # ----------------------------------------------------------------------------
  # 5. OUTLIER DETECTION
  # ----------------------------------------------------------------------------
  
  cat("Detecting outliers...\n")
  
  # Concentration outliers (>3 SD or biologically implausible)
  conc_stats <- data$concentrations %>%
    summarize(
      mean_conc = mean(concentration, na.rm = TRUE),
      sd_conc = sd(concentration, na.rm = TRUE),
      q99 = quantile(concentration, 0.99, na.rm = TRUE)
    )
  
  data$concentrations <- data$concentrations %>%
    mutate(
      outlier_flag = concentration > (conc_stats$mean_conc + 3*conc_stats$sd_conc) |
                     concentration > 100 | # Amikacin rarely >100 mg/L
                     concentration < 0,
      
      # Flag potentially erroneous peak/trough labels
      suspicious_peak = sample_type == "peak" & concentration < 10,
      suspicious_trough = sample_type == "trough" & concentration > 20
    )
  
  # Review flagged samples
  outlier_summary <- data$concentrations %>%
    filter(outlier_flag | suspicious_peak | suspicious_trough) %>%
    select(patient_id, time, concentration, sample_type, 
           outlier_flag, suspicious_peak, suspicious_trough)
  
  cat(sprintf("Flagged %d potentially problematic samples\n", 
              nrow(outlier_summary)))
  
  # ----------------------------------------------------------------------------
  # 6. DATA QUALITY CHECKS
  # ----------------------------------------------------------------------------
  
  cat("Performing data quality checks...\n")
  
  quality_checks <- list(
    # Check for negative times
    negative_times = data$concentrations %>% 
      filter(time < 0) %>% nrow(),
    
    # Check for samples before first dose
    samples_before_dose = data$concentrations %>%
      anti_join(data$dosing, by = "patient_id") %>% nrow(),
    
    # Check for missing essential covariates
    missing_weight = sum(is.na(data$patient_data$weight)),
    missing_age = sum(is.na(data$patient_data$age)),
    missing_scr = sum(is.na(data$patient_data$baseline_scr)),
    
    # Check dose ranges
    extreme_doses = data$dosing %>%
      filter(dose < 100 | dose > 3000) %>% nrow(),
    
    # Check for duplicate records
    duplicate_patients = data$patient_data %>%
      group_by(patient_id) %>%
      filter(n() > 1) %>%
      nrow()
  )
  
  print("Data Quality Summary:")
  print(quality_checks)
  
  # ----------------------------------------------------------------------------
  # 7. HANDLE MISSING DATA
  # ----------------------------------------------------------------------------
  
  cat("Handling missing data...\n")
  
  # For critical covariates, use multiple imputation
  # Only impute if <30% missing
  
  # Identify variables for imputation
  vars_to_impute <- missing_summary %>%
    filter(pct_missing > 0 & pct_missing < 30) %>%
    pull(variable)
  
  if (length(vars_to_impute) > 0) {
    # Multiple imputation using MICE
    imputation_data <- data$patient_data %>%
      select(all_of(c("patient_id", vars_to_impute, 
                      "age", "weight", "sex", "baseline_scr")))
    
    # Perform imputation (m=5 imputations)
    imp <- mice(imputation_data %>% select(-patient_id), 
                m = 5, method = "pmm", seed = 123, print = FALSE)
    
    # Use first completed dataset (or pool results in actual analysis)
    data$patient_data_imputed <- complete(imp, 1) %>%
      bind_cols(patient_id = imputation_data$patient_id)
  }
  
  # ----------------------------------------------------------------------------
  # 8. CREATE ANALYSIS DATASETS
  # ----------------------------------------------------------------------------
  
  cat("Creating analysis datasets...\n")
  
  # NONMEM-compatible dataset for population PK
  nonmem_dataset <- create_nonmem_dataset(data)
  
  # Wide-format dataset for ML
  ml_dataset <- create_ml_dataset(data)
  
  # Save processed data
  processed_data <- list(
    patient_data = data$patient_data,
    patient_data_imputed = data$patient_data_imputed,
    time_varying = data$time_varying,
    dosing = data$dosing,
    concentrations = data$concentrations,
    outcomes = data$outcomes,
    nonmem_dataset = nonmem_dataset,
    ml_dataset = ml_dataset,
    preprocessing_metadata = list(
      date_processed = Sys.time(),
      n_patients = length(unique(data$patient_data$patient_id)),
      n_concentrations = nrow(data$concentrations),
      missing_summary = missing_summary,
      quality_checks = quality_checks,
      outlier_summary = outlier_summary
    )
  )
  
  return(processed_data)
}

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

create_nonmem_dataset <- function(data) {
  # Create NONMEM-compatible dataset
  # Standard format: one row per observation/dose
  
  # Combine dosing and concentration records
  dose_records <- data$dosing %>%
    mutate(
      EVID = 1,  # Dosing event
      AMT = dose,
      DV = NA,
      CMT = 1,   # Central compartment
      RATE = AMT / infusion_duration,
      MDV = 1    # Missing DV
    ) %>%
    select(patient_id, TIME = time, EVID, AMT, DV, CMT, RATE, MDV)
  
  conc_records <- data$concentrations %>%
    filter(!outlier_flag) %>%
    mutate(
      EVID = 0,  # Observation
      AMT = 0,
      DV = concentration,
      CMT = 2,   # Observation compartment
      RATE = 0,
      MDV = as.integer(bloq)
    ) %>%
    select(patient_id, TIME = time, EVID, AMT, DV, CMT, RATE, MDV)
  
  # Combine and sort
  nm_data <- bind_rows(dose_records, conc_records) %>%
    arrange(patient_id, TIME, desc(EVID))
  
  # Add covariates (baseline, time-invariant)
  nm_data <- nm_data %>%
    left_join(
      data$patient_data %>%
        select(patient_id, AGE = age, WT = weight, HT = height,
               SEX = sex, SCR = baseline_scr, CRCL = baseline_crcl,
               APACHE = apache_ii, SOFA = sofa_score),
      by = "patient_id"
    ) %>%
    mutate(
      ID = as.integer(factor(patient_id)),
      SEX = as.integer(SEX == "M")  # 1 = male, 0 = female
    )
  
  return(nm_data)
}

create_ml_dataset <- function(data) {
  # Create wide-format dataset for ML models
  # One row per patient with all features
  
  # Aggregate PK features
  pk_features <- data$concentrations %>%
    group_by(patient_id) %>%
    summarize(
      n_samples = n(),
      first_peak = first(concentration[sample_type == "peak"]),
      mean_trough = mean(concentration[sample_type == "trough"], na.rm = TRUE),
      cv_concentration = sd(concentration) / mean(concentration) * 100
    )
  
  # Aggregate dosing features
  dose_features <- data$dosing %>%
    group_by(patient_id) %>%
    summarize(
      total_dose = sum(dose),
      mean_dose = mean(dose),
      n_doses = n(),
      dose_frequency = n() / max(time) * 24  # doses per day
    )
  
  # Time-varying features (use baseline or mean/max)
  tv_features <- data$time_varying %>%
    group_by(patient_id) %>%
    summarize(
      baseline_crcl_measured = first(crcl_cg),
      mean_crcl = mean(crcl_cg, na.rm = TRUE),
      min_crcl = min(crcl_cg, na.rm = TRUE),
      max_fluid_balance = max(fluid_balance, na.rm = TRUE),
      max_crp = max(crp, na.rm = TRUE),
      rrt_ever = any(rrt_status, na.rm = TRUE)
    )
  
  # Combine all features
  ml_data <- data$patient_data %>%
    left_join(pk_features, by = "patient_id") %>%
    left_join(dose_features, by = "patient_id") %>%
    left_join(tv_features, by = "patient_id") %>%
    left_join(data$outcomes, by = "patient_id")
  
  return(ml_data)
}
