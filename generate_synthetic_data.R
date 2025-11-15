# ============================================================================
# SYNTHETIC DATA GENERATION FOR INDIAN ICU AMINOGLYCOSIDE STUDY
# ============================================================================
#
# This script generates scientifically valid synthetic data reflecting
# real-world aminoglycoside use in Indian ICU settings
#
# Author: Generated for aminoglycoside QSP-ML project
# Date: 2025-11-15
# ============================================================================

# Required packages
required_packages <- c(
  "tidyverse",
  "lubridate",
  "MASS"  # For multivariate normal distributions
)

lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
})

# ============================================================================
# POPULATION CHARACTERISTICS - INDIAN ICU SETTINGS
# ============================================================================

# Based on published literature from Indian ICUs:
# - Indian Journal of Critical Care Medicine
# - Studies on aminoglycoside use in Indian hospitals
# - APACHE II and SOFA scores from Indian sepsis cohorts

indian_icu_params <- list(
  # Demographics
  age_mean = 52,              # Years (Indian ICU patients tend to be slightly younger)
  age_sd = 16,
  weight_mean = 62,           # kg (lower than Western populations)
  weight_sd = 12,
  height_mean = 162,          # cm (Indian average height)
  height_sd = 10,
  male_proportion = 0.65,     # Higher proportion of males in ICU

  # Clinical severity (Indian ICUs often have higher severity)
  apache_ii_mean = 22,        # Higher than Western ICUs
  apache_ii_sd = 8,
  sofa_mean = 9,
  sofa_sd = 4,

  # Infection epidemiology
  infection_sites = list(
    bloodstream = 0.30,       # High in Indian ICUs
    pneumonia = 0.35,         # VAP and HAP common
    uti = 0.15,
    iai = 0.12,               # Intra-abdominal infections
    other = 0.08
  ),

  # Pathogen distribution (Gram-negative predominance)
  pathogens = list(
    klebsiella = 0.35,        # Very common in India
    pseudomonas = 0.25,
    e_coli = 0.20,
    acinetobacter = 0.15,     # Increasing in Indian ICUs
    other = 0.05
  ),

  # MIC distribution (higher resistance in India)
  mic_gentamicin_meanlog = log(4),   # Geometric mean ~4 μg/mL
  mic_gentamicin_sdlog = 1.2,
  mic_amikacin_meanlog = log(8),     # Geometric mean ~8 μg/mL
  mic_amikacin_sdlog = 1.3,

  # Comorbidities
  diabetes_prevalence = 0.35,        # High in India
  ckd_prevalence = 0.25,
  mechanical_ventilation = 0.65,
  vasopressor_use = 0.50,
  rrt_rate = 0.18,                   # AKI requiring RRT

  # Renal function distribution
  crcl_normal_mean = 85,
  crcl_normal_sd = 25,
  crcl_arc_proportion = 0.25,        # Augmented renal clearance
  crcl_arc_mean = 160,
  crcl_arc_sd = 35,

  # Fluid overload (common in sepsis)
  fluid_balance_mean = 2.5,          # Liters
  fluid_balance_sd = 2.0
)

# ============================================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================================

generate_synthetic_aminoglycoside_data <- function(
    n_patients = 300,
    study_duration_days = 7,
    seed = 12345
) {

  set.seed(seed)

  cat("Generating synthetic aminoglycoside data for", n_patients, "patients...\n")
  cat("Population: Indian ICU setting\n\n")

  # --------------------------------------------------------------------------
  # 1. PATIENT DEMOGRAPHICS & BASELINE CHARACTERISTICS
  # --------------------------------------------------------------------------

  cat("Step 1/5: Generating patient demographics and baseline data...\n")

  patient_data <- tibble(
    patient_id = sprintf("IND_%04d", 1:n_patients),

    # Demographics
    age = rnorm(n_patients,
                indian_icu_params$age_mean,
                indian_icu_params$age_sd) %>%
      pmax(18) %>% pmin(90) %>% round(1),

    sex = sample(c("M", "F"), n_patients, replace = TRUE,
                 prob = c(indian_icu_params$male_proportion,
                         1 - indian_icu_params$male_proportion)) %>%
      factor(),

    weight = rnorm(n_patients,
                   indian_icu_params$weight_mean,
                   indian_icu_params$weight_sd) %>%
      pmax(35) %>% pmin(120) %>% round(1),

    height = rnorm(n_patients,
                   indian_icu_params$height_mean,
                   indian_icu_params$height_sd) %>%
      pmax(140) %>% pmin(190) %>% round(1),

    bmi = (weight / (height/100)^2) %>% round(1),

    # Clinical severity scores
    apache_ii = rnorm(n_patients,
                      indian_icu_params$apache_ii_mean,
                      indian_icu_params$apache_ii_sd) %>%
      pmax(5) %>% pmin(45) %>% round(0),

    sofa_score = rnorm(n_patients,
                       indian_icu_params$sofa_mean,
                       indian_icu_params$sofa_sd) %>%
      pmax(2) %>% pmin(20) %>% round(0),

    sepsis_type = sample(
      c("sepsis", "severe sepsis", "septic shock"),
      n_patients, replace = TRUE,
      prob = c(0.25, 0.35, 0.40)  # High shock rate in Indian ICUs
    ) %>% factor(),

    # Renal function - bimodal distribution
    renal_group = sample(c("normal", "arc"), n_patients, replace = TRUE,
                         prob = c(1 - indian_icu_params$crcl_arc_proportion,
                                 indian_icu_params$crcl_arc_proportion)),

    baseline_crcl = ifelse(
      renal_group == "normal",
      rnorm(n_patients, indian_icu_params$crcl_normal_mean,
            indian_icu_params$crcl_normal_sd),
      rnorm(n_patients, indian_icu_params$crcl_arc_mean,
            indian_icu_params$crcl_arc_sd)
    ) %>% pmax(15) %>% pmin(250) %>% round(1),

    # Calculate baseline serum creatinine from CrCL (reverse Cockcroft-Gault)
    baseline_scr = ((140 - age) * weight * ifelse(sex == "M", 1, 0.85)) /
      (72 * baseline_crcl) %>% round(2),

    # eGFR (CKD-EPI equation approximation)
    baseline_egfr = (140 - age) * weight / (baseline_scr * 72) *
      ifelse(sex == "M", 1, 0.85) %>%
      pmax(10) %>% pmin(180) %>% round(1),

    # Other baseline labs
    baseline_albumin = rnorm(n_patients, 2.8, 0.6) %>%
      pmax(1.5) %>% pmin(4.5) %>% round(1),  # Low albumin in sepsis

    baseline_bilirubin = rlnorm(n_patients, log(1.2), 0.6) %>%
      pmax(0.3) %>% pmin(15) %>% round(1),

    # Infection characteristics
    infection_site = sample(
      names(indian_icu_params$infection_sites),
      n_patients, replace = TRUE,
      prob = unlist(indian_icu_params$infection_sites)
    ) %>% factor(),

    pathogen = sample(
      names(indian_icu_params$pathogens),
      n_patients, replace = TRUE,
      prob = unlist(indian_icu_params$pathogens)
    ),

    # MIC values - higher for resistant organisms
    mic_base_gentamicin = rlnorm(
      n_patients,
      indian_icu_params$mic_gentamicin_meanlog,
      indian_icu_params$mic_gentamicin_sdlog
    ),

    mic_base_amikacin = rlnorm(
      n_patients,
      indian_icu_params$mic_amikacin_meanlog,
      indian_icu_params$mic_amikacin_sdlog
    ),

    # Higher MICs for Acinetobacter and Pseudomonas
    mic_gentamicin = case_when(
      pathogen == "acinetobacter" ~ mic_base_gentamicin * 2,
      pathogen == "pseudomonas" ~ mic_base_gentamicin * 1.5,
      TRUE ~ mic_base_gentamicin
    ) %>% pmax(0.25) %>% pmin(128) %>% round(2),

    mic_amikacin = case_when(
      pathogen == "acinetobacter" ~ mic_base_amikacin * 2,
      pathogen == "pseudomonas" ~ mic_base_amikacin * 1.5,
      TRUE ~ mic_base_amikacin
    ) %>% pmax(0.5) %>% pmin(256) %>% round(2),

    # Comorbidities
    diabetes = rbinom(n_patients, 1, indian_icu_params$diabetes_prevalence) == 1,

    ckd_stage = case_when(
      baseline_egfr >= 90 ~ "0",
      baseline_egfr >= 60 ~ "1",
      baseline_egfr >= 45 ~ "2",
      baseline_egfr >= 30 ~ "3a",
      baseline_egfr >= 15 ~ "3b",
      TRUE ~ "4"
    ) %>% factor(),

    mechanical_ventilation = rbinom(
      n_patients, 1,
      indian_icu_params$mechanical_ventilation
    ) == 1,

    vasopressor_use = rbinom(
      n_patients, 1,
      indian_icu_params$vasopressor_use
    ) == 1,

    # Study site info
    hospital_id = sample(
      c("AIIMS_Delhi", "CMC_Vellore", "PGIMER_Chandigarh",
        "SGPGI_Lucknow", "JIPMER_Puducherry"),
      n_patients, replace = TRUE
    ) %>% factor(),

    icu_type = sample(
      c("medical", "surgical", "mixed"),
      n_patients, replace = TRUE,
      prob = c(0.5, 0.3, 0.2)
    ) %>% factor()
  ) %>%
    select(-renal_group, -mic_base_gentamicin, -mic_base_amikacin)

  # --------------------------------------------------------------------------
  # 2. INDIVIDUAL PK PARAMETERS
  # --------------------------------------------------------------------------

  # Generate individual PK parameters with covariate effects
  # Two-compartment model: CL, Vc, Q, Vp

  patient_data <- patient_data %>%
    mutate(
      # Typical values with covariate effects
      CL_typical = exp(
        log(5.5) +                              # Base CL (L/h) for amikacin
          0.75 * log(baseline_crcl / 100) +     # Renal function effect
          0.75 * log(weight / 70) +             # Body size
          ifelse(sepsis_type == "septic shock", -0.15, 0)  # Shock reduces CL
      ),

      Vc_typical = exp(
        log(16) +                               # Base Vc (L)
          1.0 * log(weight / 70)                # Allometric scaling
      ),

      Q_typical = 12,                           # Intercompartmental clearance
      Vp_typical = 10,                          # Peripheral volume

      # Individual parameters with between-subject variability (BSV)
      # Log-normal distribution
      CL_individual = CL_typical * exp(rnorm(n_patients, 0, 0.25)),
      Vc_individual = Vc_typical * exp(rnorm(n_patients, 0, 0.20)),
      Q_individual = Q_typical * exp(rnorm(n_patients, 0, 0.40)),
      Vp_individual = Vp_typical * exp(rnorm(n_patients, 0, 0.35))
    )

  # --------------------------------------------------------------------------
  # 3. DOSING DATA
  # --------------------------------------------------------------------------

  cat("Step 2/5: Generating dosing records...\n")

  # Dosing strategy: mostly amikacin, some gentamicin
  # Common regimens in Indian ICUs:
  # - Amikacin: 15-20 mg/kg once daily
  # - Gentamicin: 5-7 mg/kg once daily

  dosing <- patient_data %>%
    mutate(
      drug = sample(c("amikacin", "gentamicin"), n_patients,
                    replace = TRUE, prob = c(0.75, 0.25)),

      # Dose based on weight and drug
      dose_per_kg = case_when(
        drug == "amikacin" ~ runif(n_patients, 15, 20),
        drug == "gentamicin" ~ runif(n_patients, 5, 7)
      ),

      # Calculate actual dose, round to nearest 50 mg
      dose_calculated = dose_per_kg * weight,
      dose_mg = round(dose_calculated / 50) * 50,

      # Dosing interval based on renal function
      interval_hours = case_when(
        baseline_crcl > 90 ~ 24,
        baseline_crcl > 60 ~ 24,
        baseline_crcl > 40 ~ 36,
        baseline_crcl > 20 ~ 48,
        TRUE ~ 48
      ),

      # Number of doses (treatment duration 3-7 days)
      treatment_days = sample(3:7, n_patients, replace = TRUE),
      n_doses = ceiling(treatment_days * 24 / interval_hours)
    ) %>%
    select(patient_id, drug, dose_mg, dose_per_kg, interval_hours, n_doses)

  # Expand to individual dose records
  dosing_records <- dosing %>%
    rowwise() %>%
    mutate(dose_times = list(seq(0, (n_doses - 1) * interval_hours,
                                 by = interval_hours))) %>%
    unnest(dose_times) %>%
    mutate(
      time = dose_times,
      dose = dose_mg,
      infusion_duration = 1,  # 1 hour infusion
      route = factor("IV")
    ) %>%
    select(patient_id, time, dose, infusion_duration, route) %>%
    ungroup()

  # --------------------------------------------------------------------------
  # 4. SIMULATE PK CONCENTRATIONS
  # --------------------------------------------------------------------------

  cat("Step 3/5: Simulating PK concentrations...\n")

  # Function to simulate two-compartment PK
  simulate_two_compartment_pk <- function(time, dose_times, doses, CL, Vc, Q, Vp, tau) {

    # Calculate concentration at specific time
    calc_conc <- function(t) {
      conc_total <- 0

      for (i in seq_along(dose_times)) {
        if (t >= dose_times[i]) {
          time_since_dose <- t - dose_times[i]

          # Two-compartment model with 1-hour infusion
          k10 <- CL / Vc
          k12 <- Q / Vc
          k21 <- Q / Vp

          alpha <- 0.5 * ((k10 + k12 + k21) +
                          sqrt((k10 + k12 + k21)^2 - 4*k10*k21))
          beta <- 0.5 * ((k10 + k12 + k21) -
                         sqrt((k10 + k12 + k21)^2 - 4*k10*k21))

          A <- doses[i] / Vc * (alpha - k21) / (alpha - beta)
          B <- doses[i] / Vc * (beta - k21) / (beta - alpha)

          # During infusion (0-1 hour)
          if (time_since_dose <= 1) {
            conc <- A * (1 - exp(-alpha * time_since_dose)) / alpha +
              B * (1 - exp(-beta * time_since_dose)) / beta
          } else {
            # After infusion
            conc <- A * (1 - exp(-alpha * 1)) / alpha * exp(-alpha * (time_since_dose - 1)) +
              B * (1 - exp(-beta * 1)) / beta * exp(-beta * (time_since_dose - 1))
          }

          conc_total <- conc_total + conc
        }
      }
      return(conc_total)
    }

    return(sapply(time, calc_conc))
  }

  # Generate concentration samples for each patient
  concentrations <- map_dfr(1:nrow(patient_data), function(i) {

    patient <- patient_data[i, ]
    patient_doses <- dosing_records %>% filter(patient_id == patient$patient_id)

    # PK parameters
    CL <- patient$CL_individual
    Vc <- patient$Vc_individual
    Q <- patient$Q_individual
    Vp <- patient$Vp_individual

    # Sampling strategy: peak (1h after start of infusion) and trough
    n_doses <- nrow(patient_doses)

    # Sample times
    sample_times <- c()
    sample_types <- c()
    dose_numbers <- c()

    for (d in 1:min(n_doses, 5)) {  # Sample first 5 doses
      dose_time <- patient_doses$time[d]

      # Peak: 1 hour after start (end of infusion)
      if (runif(1) > 0.1) {  # 90% chance of peak sample
        sample_times <- c(sample_times, dose_time + 1)
        sample_types <- c(sample_types, "peak")
        dose_numbers <- c(dose_numbers, d)
      }

      # Trough: just before next dose (if not last dose)
      if (d < n_doses && runif(1) > 0.15) {  # 85% chance of trough
        next_dose_time <- patient_doses$time[d + 1]
        sample_times <- c(sample_times, next_dose_time - 0.5)
        sample_types <- c(sample_types, "trough")
        dose_numbers <- c(dose_numbers, d)
      }

      # Random sample (10% of patients)
      if (runif(1) > 0.9 && d < n_doses) {
        random_time <- dose_time + runif(1, 4, 20)
        sample_times <- c(sample_times, random_time)
        sample_types <- c(sample_types, "random")
        dose_numbers <- c(dose_numbers, d)
      }
    }

    # Simulate concentrations
    conc_pred <- simulate_two_compartment_pk(
      time = sample_times,
      dose_times = patient_doses$time,
      doses = patient_doses$dose,
      CL = CL, Vc = Vc, Q = Q, Vp = Vp,
      tau = patient_doses$time[2] - patient_doses$time[1]
    )

    # Add residual error (proportional + additive)
    prop_error <- 0.15  # 15% CV
    add_error <- 0.5    # 0.5 mg/L

    conc_obs <- conc_pred * exp(rnorm(length(conc_pred), 0, prop_error)) +
      rnorm(length(conc_pred), 0, add_error)
    conc_obs <- pmax(conc_obs, 0.1)  # Lower limit

    tibble(
      patient_id = patient$patient_id,
      time = sample_times,
      sample_time_from_dose = sample_times -
        patient_doses$time[dose_numbers],
      concentration = round(conc_obs, 2),
      assay_lloq = 0.5,
      bloq = concentration < 0.5,
      sample_type = factor(sample_types),
      dose_number = dose_numbers
    )
  })

  # --------------------------------------------------------------------------
  # 5. TIME-VARYING COVARIATES
  # --------------------------------------------------------------------------

  cat("Step 4/5: Generating time-varying covariates...\n")

  time_varying <- map_dfr(1:nrow(patient_data), function(i) {

    patient <- patient_data[i, ]

    # Daily measurements for study duration
    days <- 0:study_duration_days

    # Serum creatinine trajectory
    # Can worsen (nephrotoxicity) or improve
    scr_baseline <- patient$baseline_scr

    # Random walk for SCr
    scr_change <- cumsum(c(0, rnorm(length(days) - 1,
                                    mean = ifelse(patient$apache_ii > 25, 0.1, -0.05),
                                    sd = 0.2)))
    scr_trajectory <- scr_baseline + scr_change
    scr_trajectory <- pmax(scr_trajectory, 0.3)
    scr_trajectory <- pmin(scr_trajectory, 8)

    # Calculate CrCL from SCr
    crcl_trajectory <- ((140 - patient$age) * patient$weight *
                          ifelse(patient$sex == "M", 1, 0.85)) /
      (72 * scr_trajectory)

    # Fluid balance (improves over time in responders)
    fluid_balance_trajectory <- rnorm(
      length(days),
      mean = seq(3, 0, length.out = length(days)),
      sd = 1.5
    ) %>% pmax(-2) %>% pmin(10)

    # Inflammatory markers (decline in responders)
    wbc_trajectory <- rnorm(
      length(days),
      mean = seq(18, 10, length.out = length(days)),
      sd = 3
    ) %>% pmax(2) %>% pmin(40)

    crp_trajectory <- rlnorm(
      length(days),
      meanlog = log(seq(150, 30, length.out = length(days))),
      sdlog = 0.4
    ) %>% pmax(5) %>% pmin(400)

    # RRT initiation (for some patients with AKI)
    rrt_prob <- ifelse(patient$apache_ii > 25 & patient$baseline_scr > 2, 0.3, 0.05)
    rrt_initiated <- rbinom(1, 1, rrt_prob) == 1

    if (rrt_initiated) {
      rrt_start_day <- sample(1:3, 1)
      rrt_status <- days >= rrt_start_day
      rrt_type <- rep("CVVHDF", length(days))
      rrt_type[days < rrt_start_day] <- "none"
    } else {
      rrt_status <- rep(FALSE, length(days))
      rrt_type <- rep("none", length(days))
    }

    tibble(
      patient_id = patient$patient_id,
      time = days * 24,  # Convert to hours

      # Renal function
      scr = round(scr_trajectory, 2),
      crcl_cg = round(crcl_trajectory, 1),
      crcl_measured = NA_real_,
      urine_output = rnorm(length(days), 1800, 600) %>%
        pmax(200) %>% pmin(4000) %>% round(0),

      # Fluid balance
      cumulative_fluid_input = cumsum(rnorm(length(days), 2.5, 0.8)) %>%
        pmax(0) %>% round(1),
      cumulative_fluid_output = cumsum(rnorm(length(days), 2.0, 0.8)) %>%
        pmax(0) %>% round(1),
      fluid_balance = round(fluid_balance_trajectory, 1),

      # Inflammatory markers
      wbc = round(wbc_trajectory, 1),
      crp = round(crp_trajectory, 1),
      procalcitonin = rlnorm(length(days), log(5), 1) %>%
        pmax(0.1) %>% pmin(100) %>% round(2),

      # Other labs
      albumin = rnorm(length(days), patient$baseline_albumin, 0.2) %>%
        pmax(1.5) %>% pmin(4) %>% round(1),

      # Organ support
      rrt_status = rrt_status,
      rrt_type = factor(rrt_type),
      rrt_flow_rate = ifelse(rrt_status,
                             rnorm(length(days), 2000, 300),
                             NA_real_),

      # Hemodynamics
      map = rnorm(length(days),
                  ifelse(patient$sepsis_type == "septic shock", 65, 75),
                  8) %>% pmax(50) %>% pmin(110) %>% round(0),

      cardiac_output = NA_real_,

      norepinephrine_dose = if(patient$vasopressor_use) {
        # Taper down over time
        rnorm(length(days),
              seq(0.3, 0.05, length.out = length(days)),
              0.1) %>% pmax(0) %>% pmin(2) %>% round(2)
      } else {
        rep(0, length(days))
      }
    )
  })

  # --------------------------------------------------------------------------
  # 6. CLINICAL OUTCOMES
  # --------------------------------------------------------------------------

  cat("Step 5/5: Simulating clinical outcomes...\n")

  # Calculate PK/PD metrics for each patient
  pk_metrics <- concentrations %>%
    group_by(patient_id) %>%
    summarize(
      observed_cmax = max(concentration[sample_type == "peak"], na.rm = TRUE),
      mean_trough = mean(concentration[sample_type == "trough"], na.rm = TRUE),
      n_samples = n()
    )

  # Join with patient data and drug info
  outcomes <- patient_data %>%
    left_join(pk_metrics, by = "patient_id") %>%
    left_join(dosing %>% select(patient_id, drug), by = "patient_id") %>%
    mutate(
      # Get appropriate MIC
      mic = ifelse(drug == "amikacin", mic_amikacin, mic_gentamicin),

      # Calculate Cmax/MIC
      achieved_cmax_mic = observed_cmax / mic,

      # Estimate AUC24 (simplified: AUC24 ≈ Dose/CL)
      dose_total = map_dbl(patient_id, function(pid) {
        sum(dosing_records$dose[dosing_records$patient_id == pid][1])
      }),

      estimated_auc24 = dose_total / CL_individual,
      achieved_auc_mic = estimated_auc24 / mic,

      # Efficacy outcomes based on PK/PD target attainment
      # Higher Cmax/MIC increases cure probability
      cure_prob = plogis(
        -2 +                                    # Baseline log-odds
          1.5 * log(achieved_cmax_mic + 0.1) +  # PK/PD effect
          -0.03 * apache_ii +                   # Severity effect
          -0.5 * (pathogen == "acinetobacter")  # Resistant organisms
      ),

      clinical_cure = rbinom(n(), 1, cure_prob) == 1,
      microbiological_eradication = rbinom(n(), 1, cure_prob * 0.9) == 1,

      time_to_clinical_improvement = if_else(
        clinical_cure,
        rnorm(n(), 3, 1) %>% pmax(1) %>% pmin(7) %>% round(1),
        NA_real_
      ),

      # Nephrotoxicity - based on high trough and duration
      nephrotox_prob = plogis(
        -3 +                                    # Baseline
          1.2 * log(mean_trough + 0.1) +        # Trough effect
          0.02 * age +                          # Age
          0.5 * (diabetes == TRUE) +            # Diabetes
          0.4 * (baseline_scr > 1.5)            # Baseline renal dysfunction
      ),

      nephrotoxicity = rbinom(n(), 1, nephrotox_prob) == 1,

      aki_stage = case_when(
        !nephrotoxicity ~ "0",
        nephrotoxicity & runif(n()) < 0.5 ~ "1",
        nephrotoxicity & runif(n()) < 0.3 ~ "2",
        TRUE ~ "3"
      ) %>% factor(),

      peak_scr = baseline_scr * if_else(nephrotoxicity,
                                         runif(n(), 1.5, 3.0),
                                         runif(n(), 0.9, 1.3)) %>% round(2),

      # Ototoxicity (rare)
      ototoxicity = rbinom(n(), 1, 0.03) == 1,
      neurotoxicity = rbinom(n(), 1, 0.02) == 1,

      # Overall outcomes
      icu_los = rnorm(n(),
                      10 + apache_ii * 0.2 - clinical_cure * 3,
                      4) %>%
        pmax(2) %>% pmin(30) %>% round(1),

      hospital_los = icu_los + rnorm(n(), 7, 3) %>%
        pmax(0) %>% pmin(60) %>% round(1),

      # Mortality
      mortality_prob = plogis(
        -4 +
          0.08 * apache_ii +
          0.1 * sofa_score +
          -0.8 * clinical_cure +
          0.5 * nephrotoxicity
      ),

      icu_mortality = rbinom(n(), 1, mortality_prob) == 1,
      day_28_mortality = rbinom(n(), 1, mortality_prob * 1.2) == 1
    ) %>%
    select(
      patient_id, clinical_cure, microbiological_eradication,
      time_to_clinical_improvement, nephrotoxicity, aki_stage,
      peak_scr, ototoxicity, neurotoxicity, icu_los, hospital_los,
      icu_mortality, day_28_mortality, achieved_cmax_mic, achieved_auc_mic
    )

  # --------------------------------------------------------------------------
  # 7. COMPILE AND RETURN DATA
  # --------------------------------------------------------------------------

  cat("\nData generation complete!\n")
  cat("Summary:\n")
  cat("  Patients:", n_patients, "\n")
  cat("  Dosing records:", nrow(dosing_records), "\n")
  cat("  PK samples:", nrow(concentrations), "\n")
  cat("  Time-varying records:", nrow(time_varying), "\n")

  # Return as list matching expected structure
  synthetic_data <- list(
    patient_data = patient_data %>%
      select(patient_id, age, sex, weight, height, bmi,
             apache_ii, sofa_score, sepsis_type,
             baseline_scr, baseline_egfr, baseline_albumin, baseline_bilirubin,
             infection_site, pathogen, mic_amikacin, mic_gentamicin,
             diabetes, ckd_stage, mechanical_ventilation, vasopressor_use,
             hospital_id, icu_type),

    time_varying = time_varying,

    dosing = dosing_records,

    concentrations = concentrations,

    outcomes = outcomes,

    # Additional metadata
    metadata = list(
      n_patients = n_patients,
      study_duration_days = study_duration_days,
      generation_date = Sys.time(),
      seed = seed,
      population = "Indian ICU",
      description = "Synthetic aminoglycoside data for Indian ICU setting"
    )
  )

  return(synthetic_data)
}

# ============================================================================
# GENERATE AND SAVE DATA
# ============================================================================

# Generate data
cat("\n")
cat("="*80, "\n")
cat("SYNTHETIC DATA GENERATION - INDIAN ICU AMINOGLYCOSIDE STUDY\n")
cat("="*80, "\n\n")

synthetic_data <- generate_synthetic_aminoglycoside_data(
  n_patients = 300,
  study_duration_days = 7,
  seed = 12345
)

# Save in required format
output_dir <- "data"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

saveRDS(synthetic_data, file.path(output_dir, "synthetic_aminoglycoside_data.rds"))
cat("\nData saved to:", file.path(output_dir, "synthetic_aminoglycoside_data.rds"), "\n")

# Also save as CSV files for easy inspection
write_csv(synthetic_data$patient_data,
          file.path(output_dir, "patient_data.csv"))
write_csv(synthetic_data$time_varying,
          file.path(output_dir, "time_varying.csv"))
write_csv(synthetic_data$dosing,
          file.path(output_dir, "dosing.csv"))
write_csv(synthetic_data$concentrations,
          file.path(output_dir, "concentrations.csv"))
write_csv(synthetic_data$outcomes,
          file.path(output_dir, "outcomes.csv"))

cat("\nCSV files also saved to 'data/' directory\n")

# Print summary statistics
cat("\n")
cat("="*80, "\n")
cat("DATA SUMMARY\n")
cat("="*80, "\n\n")

cat("Patient Demographics:\n")
cat("  Age: ", round(mean(synthetic_data$patient_data$age), 1), "±",
    round(sd(synthetic_data$patient_data$age), 1), "years\n")
cat("  Weight: ", round(mean(synthetic_data$patient_data$weight), 1), "±",
    round(sd(synthetic_data$patient_data$weight), 1), "kg\n")
cat("  Male: ", round(mean(synthetic_data$patient_data$sex == "M") * 100, 1), "%\n\n")

cat("Clinical Severity:\n")
cat("  APACHE II: ", round(mean(synthetic_data$patient_data$apache_ii), 1), "±",
    round(sd(synthetic_data$patient_data$apache_ii), 1), "\n")
cat("  SOFA: ", round(mean(synthetic_data$patient_data$sofa_score), 1), "±",
    round(sd(synthetic_data$patient_data$sofa_score), 1), "\n")
cat("  Septic shock: ", round(mean(synthetic_data$patient_data$sepsis_type == "septic shock") * 100, 1), "%\n\n")

cat("Outcomes:\n")
cat("  Clinical cure: ", round(mean(synthetic_data$outcomes$clinical_cure) * 100, 1), "%\n")
cat("  Nephrotoxicity: ", round(mean(synthetic_data$outcomes$nephrotoxicity) * 100, 1), "%\n")
cat("  ICU mortality: ", round(mean(synthetic_data$outcomes$icu_mortality) * 100, 1), "%\n\n")

cat("PK/PD Metrics:\n")
cat("  Mean Cmax/MIC: ", round(mean(synthetic_data$outcomes$achieved_cmax_mic, na.rm = TRUE), 1), "\n")
cat("  Target (Cmax/MIC ≥8): ",
    round(mean(synthetic_data$outcomes$achieved_cmax_mic >= 8, na.rm = TRUE) * 100, 1), "%\n")

cat("\n")
cat("="*80, "\n")
cat("READY FOR ANALYSIS!\n")
cat("="*80, "\n")
