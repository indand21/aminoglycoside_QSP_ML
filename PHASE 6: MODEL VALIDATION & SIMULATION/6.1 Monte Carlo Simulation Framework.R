# ============================================================================
# MONTE CARLO SIMULATION FOR DOSE REGIMEN EVALUATION
# ============================================================================

# ----------------------------------------------------------------------------
# GENERATE VIRTUAL PATIENT POPULATION
# ----------------------------------------------------------------------------

generate_virtual_population <- function(n_patients = 1000, 
                                        population_characteristics) {
  
  set.seed(123)
  
  # Sample from realistic distributions based on Indian ICU data
  
  virtual_pop <- tibble(
    patient_id = paste0("VP_", 1:n_patients),
    
    # Demographics
    age = rnorm(n_patients, 
                mean = population_characteristics$age_mean,
                sd = population_characteristics$age_sd) %>%
      pmax(18) %>% pmin(90),
    
    sex = sample(c("M", "F"), n_patients, replace = TRUE,
                 prob = c(0.6, 0.4)),
    
    weight = rnorm(n_patients,
                   mean = population_characteristics$weight_mean,
                   sd = population_characteristics$weight_sd) %>%
      pmax(40) %>% pmin(150),
    
    height = rnorm(n_patients, 165, 10) %>%
      pmax(140) %>% pmin(200),
    
    # Calculate derived
    bmi = weight / (height/100)^2,
    
    # Renal function - bimodal distribution (normal + ARC)
    crcl = c(
      rnorm(n_patients * 0.7, 80, 30),  # Normal renal function
      rnorm(n_patients * 0.3, 150, 40)  # ARC patients
    )[1:n_patients] %>%
      pmax(10) %>% pmin(250),
    
    # Baseline serum creatinine (calculated from CrCL)
    scr = ((140 - age) * weight * ifelse(sex == "M", 1, 0.85)) / (72 * crcl),
    
    # Severity scores
    apache_ii = rnbinom(n_patients, mu = 20, size = 5) %>%
      pmin(45),
    
    sofa_score = rnbinom(n_patients, mu = 8, size = 3) %>%
      pmin(20),
    
    # Sepsis type
    sepsis_type = sample(
      c("sepsis", "severe sepsis", "septic shock"),
      n_patients, replace = TRUE,
      prob = c(0.3, 0.4, 0.3)
    ),
    
    # Fluid balance (L)
    fluid_balance = rnorm(n_patients, 3, 2) %>%
      pmax(-2) %>% pmin(15),
    
    # Vasopressor use
    vasopressor = rbinom(n_patients, 1, 0.45),
    
    # Mechanical ventilation
    vent = rbinom(n_patients, 1, 0.6),
    
    # RRT
    rrt = rbinom(n_patients, 1, 0.15),
    
    # Pathogen MIC (log-normal distribution)
    mic_amikacin = rlnorm(n_patients, log(2), 1) %>%
      pmax(0.5) %>% pmin(64)
  )
  
  # Sample individual PK parameters from population distributions
  # Using lognormal with inter-individual variability
  
  virtual_pop <- virtual_pop %>%
    mutate(
      # Typical values with covariate effects
      CL_typical = exp(log(5) + 
                       0.75 * log(crcl/100) +
                       0.75 * log(weight/70)),
      
      Vc_typical = exp(log(15) + 
                       1.0 * log(weight/70) +
                       0.02 * (fluid_balance/10)),
      
      Q_typical = exp(log(10)),
      Vp_typical = exp(log(10)),
      
      # Individual parameters with BSV
      CL = CL_typical * exp(rnorm(n_patients, 0, 0.2)),
      Vc = Vc_typical * exp(rnorm(n_patients, 0, 0.15)),
      Q = Q_typical * exp(rnorm(n_patients, 0, 0.4)),
      Vp = Vp_typical * exp(rnorm(n_patients, 0, 0.3))
    )
  
  return(virtual_pop)
}

# ----------------------------------------------------------------------------
# MONTE CARLO SIMULATION OF DOSING REGIMENS
# ----------------------------------------------------------------------------

monte_carlo_dosing_simulation <- function(virtual_population, 
                                          dosing_regimens,
                                          n_simulations = 1000) {
  
  library(furrr)
  plan(multisession, workers = parallel::detectCores() - 1)
  
  results <- map_dfr(dosing_regimens, function(regimen) {
    
    cat(sprintf("Simulating regimen: %g mg q%gh\n", 
                regimen$dose, regimen$interval))
    
    # Parallel simulation across virtual patients
    sim_results <- future_map_dfr(1:nrow(virtual_population), function(i) {
      
      patient <- virtual_population[i, ]
      
      # Individual PK parameters
      pk_params <- list(
        CL = list(mean = patient$CL, sd = 0),
        Vc = list(mean = patient$Vc, sd = 0),
        Q = list(mean = patient$Q, sd = 0),
        Vp = list(mean = patient$Vp, sd = 0)
      )
      
      # Simulate PK profile
      pkpd <- simulate_pk_profile(
        patient_params = pk_params,
        dose = regimen$dose,
        interval = regimen$interval,
        mic = patient$mic_amikacin,
        n_doses = 5
      )
      
      # Evaluate targets
      target_cmax_mic_8 <- pkpd$Cmax_MIC >= 8
      target_cmax_mic_10 <- pkpd$Cmax_MIC >= 10
      target_auc_mic <- pkpd$AUC24_MIC >= 80
      
      # Predict toxicity (simplified - use threshold)
      high_trough_risk <- pkpd$Ctrough_ss > 2  # Amikacin trough >2 mg/L
      high_cmax_risk <- pkpd$Cmax > 80         # Very high peak
      
      tibble(
        patient_id = patient$patient_id,
        regimen_id = regimen$id,
        dose = regimen$dose,
        interval = regimen$interval,
        
        # Patient characteristics
        age = patient$age,
        weight = patient$weight,
        crcl = patient$crcl,
        mic = patient$mic_amikacin,
        
        # PK/PD outcomes
        Cmax = pkpd$Cmax,
        Ctrough = pkpd$Ctrough_ss,
        AUC24 = pkpd$AUC24,
        Cmax_MIC = pkpd$Cmax_MIC,
        AUC24_MIC = pkpd$AUC24_MIC,
        
        # Target attainment
        target_cmax_8 = target_cmax_mic_8,
        target_cmax_10 = target_cmax_mic_10,
        target_auc = target_auc_mic,
        
        # Safety flags
        high_trough = high_trough_risk,
        high_cmax = high_cmax_risk
      )
    }, .progress = TRUE, .options = furrr_options(seed = TRUE))
    
    return(sim_results)
  })
  
  plan(sequential)  # Reset to sequential
  
  return(results)
}

# ----------------------------------------------------------------------------
# ANALYZE MONTE CARLO RESULTS
# ----------------------------------------------------------------------------

analyze_monte_carlo_results <- function(mc_results) {
  
  # Calculate summary statistics by regimen
  regimen_summary <- mc_results %>%
    group_by(regimen_id, dose, interval) %>%
    summarize(
      n_patients = n(),
      
      # Target attainment rates
      PTA_Cmax8 = mean(target_cmax_8) * 100,
      PTA_Cmax10 = mean(target_cmax_10) * 100,
      PTA_AUC = mean(target_auc) * 100,
      
      # Safety metrics
      high_trough_rate = mean(high_trough) * 100,
      high_cmax_rate = mean(high_cmax) * 100,
      
      # PK/PD distribution
      median_Cmax_MIC = median(Cmax_MIC),
      q25_Cmax_MIC = quantile(Cmax_MIC, 0.25),
      q75_Cmax_MIC = quantile(Cmax_MIC, 0.75),
      
      # Composite score (efficacy - toxicity)
      composite_score = PTA_Cmax8 - high_trough_rate,
      
      .groups = 'drop'
    )
  
  # Stratified analysis by renal function
  stratified_summary <- mc_results %>%
    mutate(
      renal_group = cut(crcl,
                       breaks = c(0, 50, 90, 130, 300),
                       labels = c("Impaired", "Normal", "High-normal", "ARC"))
    ) %>%
    group_by(regimen_id, dose, interval, renal_group) %>%
    summarize(
      n_patients = n(),
      PTA_Cmax8 = mean(target_cmax_8) * 100,
      high_trough_rate = mean(high_trough) * 100,
      .groups = 'drop'
    )
  
  # Visualization: PTA vs toxicity trade-off
  tradeoff_plot <- ggplot(regimen_summary, 
                          aes(x = high_trough_rate, y = PTA_Cmax8)) +
    geom_point(aes(size = dose, color = interval), alpha = 0.7) +
    geom_text(aes(label = sprintf("%gq%g", dose, interval)),
              vjust = -0.5, size = 3) +
    geom_hline(yintercept = 90, linetype = "dashed", color = "blue") +
    geom_vline(xintercept = 10, linetype = "dashed", color = "red") +
    labs(
      x = "High Trough Rate (%)",
      y = "PTA for Cmax/MIC ≥ 8 (%)",
      title = "Efficacy-Safety Trade-off Analysis",
      subtitle = "Target: PTA >90%, Toxicity <10%"
    ) +
    theme_bw()
  
  # Heatmap: PTA by MIC and regimen
  pta_heatmap <- mc_results %>%
    mutate(mic_bin = cut(mic, breaks = c(0, 1, 2, 4, 8, 16, 64))) %>%
    group_by(regimen_id, dose, interval, mic_bin) %>%
    summarize(PTA = mean(target_cmax_8) * 100, .groups = 'drop') %>%
    mutate(regimen_label = sprintf("%g mg q%gh", dose, interval)) %>%
    ggplot(aes(x = mic_bin, y = regimen_label, fill = PTA)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.0f%%", PTA)), size = 3) +
    scale_fill_gradient2(low = "red", mid = "yellow", high = "green",
                        midpoint = 75, limits = c(0, 100)) +
    labs(x = "MIC (mg/L)", y = "Dosing Regimen",
         title = "PTA Heatmap by MIC",
         fill = "PTA (%)") +
    theme_minimal()
  
  # Distribution plots
  cmax_mic_dist <- ggplot(mc_results, 
                          aes(x = Cmax_MIC, 
                              color = factor(paste(dose, "mg q", interval, "h")))) +
    geom_density(alpha = 0.5) +
    geom_vline(xintercept = c(8, 10), linetype = "dashed") +
    scale_x_log10() +
    labs(x = "Cmax/MIC Ratio (log scale)",
         y = "Density",
         color = "Regimen") +
    theme_bw()
  
  return(list(
    regimen_summary = regimen_summary,
    stratified_summary = stratified_summary,
    tradeoff_plot = tradeoff_plot,
    pta_heatmap = pta_heatmap,
    cmax_mic_distribution = cmax_mic_dist
  ))
}
