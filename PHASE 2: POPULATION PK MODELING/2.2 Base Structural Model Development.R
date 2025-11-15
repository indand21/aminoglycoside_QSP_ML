# ============================================================================
# AMINOGLYCOSIDE POPULATION PK MODEL
# ============================================================================

library(nlmixr2)
library(mrgsolve)
library(tidyverse)

# ----------------------------------------------------------------------------
# MODEL 1: Two-Compartment Model with First-Order Elimination
# ----------------------------------------------------------------------------

# This is the base structural model for aminoglycosides
# Most aminoglycosides follow 2-compartment PK

aminoglycoside_2cmt_base <- function() {
  ini({
    # Typical values (population parameters)
    tvcl <- log(5)      # Clearance (L/h) - log-transformed
    tvvc <- log(15)     # Central volume (L)
    tvq <- log(10)      # Intercompartmental clearance (L/h)
    tvvp <- log(10)     # Peripheral volume (L)
    
    # Inter-individual variability (exponential error model)
    eta_cl ~ 0.3        # IIV on CL (CV%)
    eta_vc ~ 0.2        # IIV on Vc
    eta_q ~ 0.4         # IIV on Q
    eta_vp ~ 0.3        # IIV on Vp
    
    # Residual error (proportional + additive)
    prop_err <- 0.15    # Proportional error (15%)
    add_err <- 0.5      # Additive error (mg/L)
  })
  
  model({
    # Individual PK parameters
    CL <- exp(tvcl + eta_cl)
    Vc <- exp(tvvc + eta_vc)
    Q <- exp(tvq + eta_q)
    Vp <- exp(tvvp + eta_vp)
    
    # Micro-constants
    k10 <- CL / Vc
    k12 <- Q / Vc
    k21 <- Q / Vp
    
    # Differential equations
    d/dt(central) <- -k10 * central - k12 * central + k21 * peripheral
    d/dt(peripheral) <- k12 * central - k21 * peripheral
    
    # Concentration
    Cp <- central / Vc
    
    # Observation model (combined error)
    Cp ~ prop(prop_err) + add(add_err)
  })
}

# ----------------------------------------------------------------------------
# MODEL 2: Covariate Model Development
# ----------------------------------------------------------------------------

# Stepwise covariate modeling approach
# Test clinically relevant covariates on CL and Vc

aminoglycoside_2cmt_covariate <- function() {
  ini({
    # Typical values at reference covariate values
    # Reference: 70kg, CrCL 100 mL/min, Age 40y
    tvcl <- log(5)
    tvvc <- log(15)
    tvq <- log(10)
    tvvp <- log(10)
    
    # Covariate effects
    cl_crcl <- 0.75     # Effect of CrCL on CL (power model)
    cl_wt <- 0.75       # Effect of weight on CL
    vc_wt <- 1.0        # Effect of weight on Vc (allometric)
    cl_age <- -0.01     # Effect of age on CL (linear)
    cl_sepsis <- 0      # Effect of septic shock on CL
    vc_fluid <- 0       # Effect of fluid balance on Vc
    
    # IIV
    eta_cl ~ 0.2        # Reduced after covariates
    eta_vc ~ 0.15
    eta_q ~ 0.4
    eta_vp ~ 0.3
    
    # Residual error
    prop_err <- 0.15
    add_err <- 0.5
  })
  
  model({
    # Covariate effects on CL
    CL <- exp(tvcl) * 
          (CRCL/100)^cl_crcl *          # Renal function
          (WT/70)^cl_wt *                # Body size
          exp(cl_age * (AGE - 40)) *     # Age effect
          exp(cl_sepsis * SEPSHOCK) *    # Septic shock
          exp(eta_cl)
    
    # Covariate effects on Vc
    Vc <- exp(tvvc) * 
          (WT/70)^vc_wt *                # Allometric scaling
          (1 + vc_fluid * (FLUIDBAL/10)) * # Fluid balance (L)
          exp(eta_vc)
    
    Q <- exp(tvq + eta_q)
    Vp <- exp(tvvp + eta_vp)
    
    # Model dynamics (same as base model)
    k10 <- CL / Vc
    k12 <- Q / Vc
    k21 <- Q / Vp
    
    d/dt(central) <- -k10 * central - k12 * central + k21 * peripheral
    d/dt(peripheral) <- k12 * central - k21 * peripheral
    
    Cp <- central / Vc
    Cp ~ prop(prop_err) + add(add_err)
  })
}

# ----------------------------------------------------------------------------
# FIT POPULATION PK MODEL
# ----------------------------------------------------------------------------

fit_popPK_model <- function(data, model_function, control_options = NULL) {
  
  # Default control options
  if (is.null(control_options)) {
    control_options <- nlmixr2Control(
      method = "saem",      # SAEM algorithm
      nBurn = 500,          # Burn-in iterations
      nEm = 300,            # EM iterations
      print = 50            # Print every 50 iterations
    )
  }
  
  # Fit model
  cat("Fitting population PK model...\n")
  fit <- nlmixr2(
    object = model_function,
    data = data,
    est = "saem",
    control = control_options
  )
  
  return(fit)
}

# ----------------------------------------------------------------------------
# MODEL DIAGNOSTICS
# ----------------------------------------------------------------------------

perform_model_diagnostics <- function(fit, data) {
  
  library(xpose)
  library(vpc)
  
  # Create xpose database
  xpdb <- xpose_data(fit)
  
  # 1. Goodness-of-fit plots
  gof_plots <- list(
    dv_vs_pred = dv_vs_pred(xpdb) + 
      labs(title = "Observations vs Population Predictions"),
    
    dv_vs_ipred = dv_vs_ipred(xpdb) +
      labs(title = "Observations vs Individual Predictions"),
    
    res_vs_pred = res_vs_pred(xpdb, res = "CWRES") +
      labs(title = "CWRES vs Predictions"),
    
    res_vs_idv = res_vs_idv(xpdb, res = "CWRES") +
      labs(title = "CWRES vs Time")
  )
  
  # 2. Individual fits
  ind_plots <- ind_plots(xpdb, page = 1:9)
  
  # 3. Parameter distribution
  eta_plots <- eta_distrib(xpdb)
  
  # 4. Visual Predictive Check (VPC)
  vpc_plot <- vpc(
    fit,
    n = 1000,           # Number of simulations
    stratify = "SEPSHOCK",  # Stratify by septic shock
    show = list(
      obs_dv = TRUE,
      obs_ci = TRUE,
      pi = TRUE,
      pi_ci = TRUE
    )
  )
  
  # 5. Bootstrap for parameter uncertainty (if needed)
  # This is computationally intensive - run separately
  # bootstrap_results <- bootstrap_fit(fit, nboot = 200)
  
  diagnostics <- list(
    gof_plots = gof_plots,
    ind_plots = ind_plots,
    eta_plots = eta_plots,
    vpc_plot = vpc_plot,
    parameter_estimates = fit$parFixed,
    omega_matrix = fit$omega,
    objective_function = fit$objective
  )
  
  return(diagnostics)
}

# ----------------------------------------------------------------------------
# COVARIATE MODEL BUILDING
# ----------------------------------------------------------------------------

# Stepwise covariate model (SCM) approach
perform_scm <- function(base_fit, data, covariates) {
  
  # Covariates to test
  # Format: list(parameter = c("covariate1", "covariate2"))
  # Example:
  # covariates <- list(
  #   CL = c("CRCL", "WT", "AGE", "SEPSHOCK", "APACHE"),
  #   Vc = c("WT", "FLUIDBAL", "SEPSHOCK")
  # )
  
  results <- list()
  current_model <- base_fit
  selected_covariates <- list()
  
  # Forward selection (alpha = 0.05, dOFV > 3.84)
  cat("Starting forward selection...\n")
  
  for (param in names(covariates)) {
    for (cov in covariates[[param]]) {
      
      cat(sprintf("Testing %s on %s...\n", cov, param))
      
      # Create model with this covariate
      test_model <- add_covariate_to_model(current_model, param, cov)
      
      # Fit model
      test_fit <- fit_popPK_model(data, test_model)
      
      # Calculate delta OFV
      delta_ofv <- current_model$objective - test_fit$objective
      
      results[[sprintf("%s_%s", param, cov)]] <- list(
        delta_ofv = delta_ofv,
        parameter_estimate = test_fit$parFixed[cov],
        fit = test_fit
      )
      
      # If significant improvement (dOFV > 3.84 for p<0.05)
      if (delta_ofv > 3.84) {
        current_model <- test_fit
        selected_covariates[[param]] <- c(selected_covariates[[param]], cov)
        cat(sprintf("  -> Selected (dOFV = %.2f)\n", delta_ofv))
      }
    }
  }
  
  # Backward elimination (alpha = 0.01, dOFV > 6.63)
  cat("\nStarting backward elimination...\n")
  
  # Test removal of each selected covariate
  # (Implementation similar to forward, but testing removal)
  
  final_scm <- list(
    selected_covariates = selected_covariates,
    forward_results = results,
    final_model = current_model
  )
  
  return(final_scm)
}
