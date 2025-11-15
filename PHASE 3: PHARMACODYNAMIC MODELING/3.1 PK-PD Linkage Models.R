# ============================================================================
# AMINOGLYCOSIDE PK/PD MODELS
# ============================================================================

library(mrgsolve)
library(deSolve)

# ----------------------------------------------------------------------------
# PK/PD MODEL 1: Empirical Cmax/MIC and AUC/MIC
# ----------------------------------------------------------------------------

# Calculate PK/PD indices from individual PK parameters
calculate_pkpd_indices <- function(pk_params, mic, dosing_regimen) {
  
  # Individual PK parameters from population model
  CL <- pk_params$CL
  Vc <- pk_params$Vc
  Q <- pk_params$Q
  Vp <- pk_params$Vp
  
  # Dosing
  dose <- dosing_regimen$dose
  tau <- dosing_regimen$interval  # hours
  tinf <- dosing_regimen$infusion_duration
  
  # Micro-constants
  k10 <- CL / Vc
  k12 <- Q / Vc
  k21 <- Q / Vp
  
  # Eigenvalues for 2-compartment model
  lambda1 <- 0.5 * ((k10 + k12 + k21) + 
                    sqrt((k10 + k12 + k21)^2 - 4*k10*k21))
  lambda2 <- 0.5 * ((k10 + k12 + k21) - 
                    sqrt((k10 + k12 + k21)^2 - 4*k10*k21))
  
  # Coefficients
  A <- (dose/Vc) * (1/tinf) * ((k21 - lambda1)/(lambda2 - lambda1)) * 
       (1 - exp(-lambda1*tinf))/lambda1
  B <- (dose/Vc) * (1/tinf) * ((k21 - lambda2)/(lambda1 - lambda2)) * 
       (1 - exp(-lambda2*tinf))/lambda2
  
  # Cmax at steady state (end of infusion)
  Cmax_ss <- (A * (1 - exp(-lambda1*tau))/(1 - exp(-lambda1*tau))) * 
              exp(-lambda1*0) +
             (B * (1 - exp(-lambda2*tau))/(1 - exp(-lambda2*tau))) * 
              exp(-lambda2*0)
  
  # Cmin at steady state (just before next dose)
  Cmin_ss <- (A * (1 - exp(-lambda1*tau))/(1 - exp(-lambda1*tau))) * 
              exp(-lambda1*(tau-tinf)) +
             (B * (1 - exp(-lambda2*tau))/(1 - exp(-lambda2*tau))) * 
              exp(-lambda2*(tau-tinf))
  
  # AUC over dosing interval at steady state
  AUC_ss <- (A/lambda1) * (1 - exp(-lambda1*tau))/(1 - exp(-lambda1*tau)) * 
             (1 - exp(-lambda1*tau)) +
            (B/lambda2) * (1 - exp(-lambda2*tau))/(1 - exp(-lambda2*tau)) * 
             (1 - exp(-lambda2*tau))
  
  # PK/PD indices
  pkpd <- list(
    Cmax = Cmax_ss,
    Cmin = Cmin_ss,
    AUC24 = AUC_ss * (24/tau),
    Cmax_MIC = Cmax_ss / mic,
    AUC_MIC = AUC_ss / mic,
    AUC24_MIC = (AUC_ss * 24/tau) / mic,
    T_above_MIC = calculate_time_above_mic(A, B, lambda1, lambda2, 
                                            tau, tinf, mic)
  )
  
  return(pkpd)
}

# Helper function to calculate time above MIC
calculate_time_above_mic <- function(A, B, lambda1, lambda2, tau, tinf, mic) {
  
  # Concentration as function of time
  conc_func <- function(t) {
    if (t <= tinf) {
      # During infusion
      A * (1 - exp(-lambda1*t)) + B * (1 - exp(-lambda2*t))
    } else {
      # After infusion
      A * (1 - exp(-lambda1*tinf)) * exp(-lambda1*(t-tinf)) +
      B * (1 - exp(-lambda2*tinf)) * exp(-lambda2*(t-tinf))
    }
  }
  
  # Find time when concentration drops below MIC
  # Use root finding
  time_below_mic <- tryCatch({
    uniroot(function(t) conc_func(t) - mic, 
            interval = c(tinf, tau))$root
  }, error = function(e) {
    tau  # If always above MIC during interval
  })
  
  t_above_mic <- time_below_mic - 0  # Time from dose start
  
  return(t_above_mic)
}

# ----------------------------------------------------------------------------
# PK/PD MODEL 2: Bacterial Growth-Kill Model (mrgsolve)
# ----------------------------------------------------------------------------

# Mechanistic model linking aminoglycoside concentrations to bacterial killing

bacterial_kill_model <- '
$PARAM 
// PK parameters (from individual patient PopPK estimates)
CL = 5, Vc = 15, Q = 10, Vp = 10

// PD parameters
Kgrowth = 0.5    // Bacterial growth rate constant (1/h)
Popmax = 10      // Maximum bacterial population (log10 CFU/mL)
Pop0 = 6         // Initial bacterial population (log10 CFU/mL)
Kmax = 5         // Maximum kill rate (1/h)
KC50 = 8         // Concentration for 50% of Kmax (mg/L)
H = 2            // Hill coefficient (sigmoidicity)

// MIC
MIC = 2          // Pathogen MIC (mg/L)

$CMT 
central peripheral bacteria

$MAIN
// Initial bacterial count
bacteria_0 = Pop0;

$ODE
// PK equations
double k10 = CL / Vc;
double k12 = Q / Vc;
double k21 = Q / Vp;

dxdt_central = -k10*central - k12*central + k21*peripheral;
dxdt_peripheral = k12*central - k21*peripheral;

// Concentration
double Cp = central / Vc;

// PD equations - bacterial dynamics
double Kgrowth_net = Kgrowth * (1 - bacteria/Popmax);  // Logistic growth
double Kkill = (Kmax * pow(Cp, H)) / (pow(KC50, H) + pow(Cp, H));
double dBdt = Kgrowth_net - Kkill;

dxdt_bacteria = bacteria * dBdt;

$CAPTURE
Cp bacteria Kgrowth_net Kkill
'

# Compile model
bacterial_model <- mcode("bacterial_kill", bacterial_kill_model)

# Simulate bacterial kill curve
simulate_bacterial_kill <- function(pk_params, pd_params, dosing, mic) {
  
  # Update model with individual parameters
  mod <- bacterial_model %>%
    param(CL = pk_params$CL,
          Vc = pk_params$Vc,
          Q = pk_params$Q,
          Vp = pk_params$Vp,
          Kmax = pd_params$Kmax,
          KC50 = pd_params$KC50,
          H = pd_params$H,
          MIC = mic)
  
  # Create dosing events
  doses <- ev(amt = dosing$dose,
              ii = dosing$interval,
              addl = dosing$n_doses - 1,
              rate = dosing$dose / dosing$infusion_duration,
              cmt = 1)
  
  # Simulate
  out <- mod %>%
    ev(doses) %>%
    mrgsim(end = dosing$interval * dosing$n_doses, delta = 0.1)
  
  return(as.data.frame(out))
}

# ----------------------------------------------------------------------------
# PK/PD TARGET ATTAINMENT ANALYSIS
# ----------------------------------------------------------------------------

analyze_target_attainment <- function(population_pk_fit, mic_distribution, 
                                       dosing_regimen) {
  
  # Extract individual PK parameters from population fit
  ind_params <- population_pk_fit$eta
  
  # Targets for aminoglycosides:
  # - Optimal: Cmax/MIC > 10
  # - Adequate: Cmax/MIC > 8
  # - Minimal: Cmax/MIC > 5
  
  targets <- c(5, 8, 10)
  
  # For each patient
  pta_results <- map_dfr(1:nrow(ind_params), function(i) {
    
    # Individual PK parameters
    pk_params <- list(
      CL = exp(population_pk_fit$parFixed['tvcl'] + ind_params$eta_cl[i]),
      Vc = exp(population_pk_fit$parFixed['tvvc'] + ind_params$eta_vc[i]),
      Q = exp(population_pk_fit$parFixed['tvq'] + ind_params$eta_q[i]),
      Vp = exp(population_pk_fit$parFixed['tvvp'] + ind_params$eta_vp[i])
    )
    
    # Calculate PK/PD indices for each MIC value
    map_dfr(mic_distribution$mic_values, function(mic) {
      
      pkpd <- calculate_pkpd_indices(pk_params, mic, dosing_regimen)
      
      # Check target attainment
      tibble(
        patient_id = i,
        mic = mic,
        mic_probability = mic_distribution$probability[
          mic_distribution$mic_values == mic],
        Cmax_MIC = pkpd$Cmax_MIC,
        AUC24_MIC = pkpd$AUC24_MIC,
        target_5 = pkpd$Cmax_MIC > 5,
        target_8 = pkpd$Cmax_MIC > 8,
        target_10 = pkpd$Cmax_MIC > 10
      )
    })
  })
  
  # Calculate PTA (Probability of Target Attainment)
  pta_summary <- pta_results %>%
    group_by(mic) %>%
    summarize(
      mic_probability = first(mic_probability),
      PTA_5 = mean(target_5) * 100,
      PTA_8 = mean(target_8) * 100,
      PTA_10 = mean(target_10) * 100,
      mean_Cmax_MIC = mean(Cmax_MIC),
      .groups = 'drop'
    )
  
  # Calculate CFR (Cumulative Fraction of Response)
  # Weight PTA by MIC probability
  cfr <- pta_summary %>%
    summarize(
      CFR_5 = sum(PTA_5 * mic_probability) / 100,
      CFR_8 = sum(PTA_8 * mic_probability) / 100,
      CFR_10 = sum(PTA_10 * mic_probability) / 100
    )
  
  # Visualization
  pta_plot <- ggplot(pta_summary, aes(x = mic)) +
    geom_line(aes(y = PTA_5, color = "Cmax/MIC > 5")) +
    geom_line(aes(y = PTA_8, color = "Cmax/MIC > 8")) +
    geom_line(aes(y = PTA_10, color = "Cmax/MIC > 10")) +
    geom_hline(yintercept = 90, linetype = "dashed", color = "red") +
    scale_x_continuous(trans = "log2") +
    labs(x = "MIC (mg/L)", 
         y = "Probability of Target Attainment (%)",
         title = sprintf("PTA Analysis: %g mg q%gh", 
                        dosing_regimen$dose, dosing_regimen$interval),
         subtitle = sprintf("CFR (Cmax/MIC>8): %.1f%%", cfr$CFR_8)) +
    theme_bw()
  
  return(list(
    pta_by_mic = pta_summary,
    cfr = cfr,
    pta_plot = pta_plot
  ))
}
