# ============================================================================
# MACHINE LEARNING FEATURE ENGINEERING
# ============================================================================

library(recipes)
library(caret)
library(tidymodels)

# ----------------------------------------------------------------------------
# CREATE COMPREHENSIVE FEATURE SET
# ----------------------------------------------------------------------------

engineer_ml_features <- function(processed_data) {
  
  # Start with base patient data
  ml_features <- processed_data$ml_dataset
  
  # --------------------------------------------------------------------
  # 1. DEMOGRAPHIC FEATURES (already present)
  # --------------------------------------------------------------------
  # age, sex, weight, height, bmi, ibw, abw, bsa
  
  # --------------------------------------------------------------------
  # 2. DERIVED ANTHROPOMETRIC FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # Body composition indices
      obesity_index = weight / ibw,
      bmi_category = cut(bmi, 
                         breaks = c(0, 18.5, 25, 30, 35, 100),
                         labels = c("underweight", "normal", "overweight",
                                   "obese", "morbidly_obese")),
      
      # Body surface area normalized weight
      wt_bsa_ratio = weight / bsa
    )
  
  # --------------------------------------------------------------------
  # 3. RENAL FUNCTION FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # CKD-EPI eGFR
      egfr_ckdepi = calculate_egfr_ckdepi(baseline_scr, age, sex),
      
      # Renal function categories
      renal_cat = cut(baseline_crcl,
                      breaks = c(0, 30, 50, 80, 130, 1000),
                      labels = c("severe_impair", "moderate_impair",
                                "mild_impair", "normal", "ARC")),
      
      # ARC flag (very important for aminoglycosides)
      arc_flag = baseline_crcl > 130,
      
      # Renal function trajectory (if time-varying data available)
      delta_crcl = (mean_crcl - baseline_crcl_measured) / baseline_crcl_measured,
      crcl_variability = (max(mean_crcl, baseline_crcl_measured) - 
                          min_crcl) / mean_crcl
    )
  
  # --------------------------------------------------------------------
  # 4. ILLNESS SEVERITY FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # Combined severity score
      severity_composite = apache_ii + sofa_score,
      
      # High severity flag
      high_severity = apache_ii > 25 | sofa_score > 10,
      
      # Sepsis severity encoding
      sepsis_severity = case_when(
        sepsis_type == "septic shock" ~ 3,
        sepsis_type == "severe sepsis" ~ 2,
        sepsis_type == "sepsis" ~ 1,
        TRUE ~ 0
      )
    )
  
  # --------------------------------------------------------------------
  # 5. FLUID STATUS FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # Fluid overload
      fluid_overload_pct = (max_fluid_balance / weight) * 100,
      fluid_overload_flag = max_fluid_balance > (weight * 0.1), # >10% BW
      
      # Normalized fluid balance
      fluid_balance_per_kg = max_fluid_balance / weight
    )
  
  # --------------------------------------------------------------------
  # 6. INFECTION CHARACTERISTICS
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # High-risk infection sites
      high_risk_site = infection_site %in% c("bloodstream", "pneumonia"),
      
      # MIC categories (for amikacin - typical breakpoints)
      mic_category = cut(mic_amikacin,
                        breaks = c(0, 4, 16, 64, 1000),
                        labels = c("susceptible", "intermediate",
                                  "resistant", "highly_resistant")),
      
      # Log-transformed MIC (for modeling)
      log_mic = log2(mic_amikacin)
    )
  
  # --------------------------------------------------------------------
  # 7. DOSING FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # Dose normalization
      dose_per_kg = mean_dose / weight,
      dose_per_ibw = mean_dose / ibw,
      dose_per_abw = mean_dose / abw,
      
      # Dosing intensity
      total_exposure = total_dose / weight,
      
      # Dosing frequency category
      freq_category = cut(dose_frequency,
                         breaks = c(0, 0.75, 1.5, 2.5, 10),
                         labels = c("q24h", "q12h", "q8h", "more_frequent"))
    )
  
  # --------------------------------------------------------------------
  # 8. PK-DERIVED FEATURES (from PopPK model predictions)
  # --------------------------------------------------------------------
  # These require running the PopPK model first
  # We'll add placeholder for now - fill in after PopPK fitting
  
  ml_features <- ml_features %>%
    mutate(
      # Individual PK parameters (Empirical Bayes estimates)
      ipred_cl = NA_real_,
      ipred_vc = NA_real_,
      ipred_cmax = NA_real_,
      ipred_auc24 = NA_real_,
      ipred_cmax_mic = NA_real_,
      
      # PK variability
      pk_variability = cv_concentration  # From observed data
    )
  
  # --------------------------------------------------------------------
  # 9. TIME-BASED FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # Treatment duration
      treatment_duration = n_doses * (24 / dose_frequency),
      
      # Early vs late sampling
      early_pk_samples = sum(concentrations$time < 24, na.rm = TRUE)
    )
  
  # --------------------------------------------------------------------
  # 10. INTERACTION FEATURES
  # --------------------------------------------------------------------
  ml_features <- ml_features %>%
    mutate(
      # Critical interactions
      renal_severity = baseline_crcl * (apache_ii / 10),
      dose_renal = dose_per_kg * baseline_crcl,
      weight_renal = (weight / ibw) * baseline_crcl,
      
      # Age-renal interaction
      age_renal_risk = (age > 65) & (baseline_crcl < 60),
      
      # Sepsis-fluid interaction
      sepsis_fluid = (sepsis_type == "septic shock") * fluid_overload_pct
    )
  
  return(ml_features)
}

# Helper function for CKD-EPI eGFR
calculate_egfr_ckdepi <- function(scr, age, sex) {
  kappa <- ifelse(sex == "F", 0.7, 0.9)
  alpha <- ifelse(sex == "F", -0.329, -0.411)
  sex_factor <- ifelse(sex == "F", 1.018, 1)
  
  egfr <- 141 * pmin(scr/kappa, 1)^alpha * 
          pmax(scr/kappa, 1)^(-1.209) * 
          0.993^age * sex_factor
  
  return(egfr)
}

# ----------------------------------------------------------------------------
# RECIPE FOR ML PREPROCESSING
# ----------------------------------------------------------------------------

create_ml_recipe <- function(ml_features, outcome_variable) {
  
  # Define recipe
  rec <- recipe(as.formula(paste(outcome_variable, "~ .")), 
                data = ml_features) %>%
    
    # Remove ID variables
    update_role(patient_id, new_role = "ID") %>%
    
    # Handle missing data
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    
    # Remove zero variance predictors
    step_zv(all_predictors()) %>%
    
    # Remove highly correlated features (r > 0.9)
    step_corr(all_numeric_predictors(), threshold = 0.9) %>%
    
    # Normalize numeric predictors
    step_normalize(all_numeric_predictors()) %>%
    
    # Create dummy variables for factors
    step_dummy(all_nominal_predictors()) %>%
    
    # Optional: Principal component analysis for dimension reduction
    # step_pca(all_numeric_predictors(), threshold = 0.95) %>%
    
    # Optional: Interaction terms (if specified)
    step_interact(terms = ~ age:baseline_crcl + 
                           weight:baseline_crcl +
                           dose_per_kg:baseline_crcl)
  
  return(rec)
}
