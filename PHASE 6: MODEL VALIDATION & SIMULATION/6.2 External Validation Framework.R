# ============================================================================
# EXTERNAL VALIDATION FRAMEWORK
# ============================================================================

# ----------------------------------------------------------------------------
# VALIDATE PopPK MODEL ON EXTERNAL DATA
# ----------------------------------------------------------------------------

validate_popPK_external <- function(fitted_model, external_data) {
  
  # 1. Predict concentrations in external dataset
  external_pred <- predict(fitted_model, external_data)
  
  # 2. Calculate prediction errors
  pred_errors <- external_data %>%
    mutate(
      pred = external_pred,
      error = DV - pred,
      pct_error = (error / DV) * 100,
      abs_pct_error = abs(pct_error)
    )
  
  # 3. Bias and precision metrics
  bias_precision <- pred_errors %>%
    filter(EVID == 0) %>%  # Observations only
    summarize(
      # Bias
      MPE = mean(pct_error),
      MdPE = median(pct_error),
      
      # Precision
      MAPE = mean(abs_pct_error),
      MdAPE = median(abs_pct_error),
      RMSE = sqrt(mean(error^2)),
      
      # Within X% criteria
      within_20pct = mean(abs_pct_error <= 20) * 100,
      within_30pct = mean(abs_pct_error <= 30) * 100
    )
  
  # 4. Prediction-corrected VPC on external data
  pcvpc <- vpc(
    fitted_model,
    obs = external_data,
    n = 500,
    pred_corr = TRUE
  )
  
  # 5. Normalized prediction distribution errors (NPDE)
  library(npde)
  
  npde_results <- autonpde(
    namobs = external_data,
    namsim = generate_simulated_data(fitted_model, external_data, nsim = 1000),
    iid = "ID",
    ix = "TIME",
    iy = "DV",
    imdv = "MDV"
  )
  
  # Check NPDE distribution (should be N(0,1))
  npde_test <- list(
    mean_npde = mean(npde_results@results@res$npde),
    sd_npde = sd(npde_results@results@res$npde),
    shapiro_pvalue = shapiro.test(npde_results@results@res$npde)$p.value,
    ks_pvalue = ks.test(npde_results@results@res$npde, "pnorm")$p.value
  )
  
  validation_results <- list(
    bias_precision = bias_precision,
    prediction_errors = pred_errors,
    pcvpc_plot = pcvpc,
    npde_results = npde_results,
    npde_tests = npde_test
  )
  
  return(validation_results)
}

# ----------------------------------------------------------------------------
# VALIDATE ML MODELS ON EXTERNAL DATA
# ----------------------------------------------------------------------------

validate_ml_external <- function(ml_models, external_data) {
  
  # Prepare external data
  external_features <- engineer_ml_features(external_data)
  
  results <- list()
  
  # 1. Validate PK parameter prediction
  if (!is.null(ml_models$pk_predictor)) {
    
    external_pk_pred <- predict(ml_models$pk_predictor$cl_model, 
                                external_features)
    
    pk_metrics <- external_features %>%
      mutate(pred_cl = external_pk_pred$.pred) %>%
      filter(!is.na(ipred_cl)) %>%
      summarize(
        r2 = cor(pred_cl, ipred_cl)^2,
        rmse = sqrt(mean((pred_cl - ipred_cl)^2)),
        mae = mean(abs(pred_cl - ipred_cl)),
        mape = mean(abs((pred_cl - ipred_cl)/ipred_cl)) * 100
      )
    
    results$pk_validation = pk_metrics
  }
  
  # 2. Validate outcome predictions
  if (!is.null(ml_models$outcome_predictor)) {
    
    # Nephrotoxicity prediction
    nephro_pred <- predict(
      ml_models$outcome_predictor$nephrotoxicity_model,
      external_features,
      type = "prob"
    )
    
    nephro_actual <- external_features$nephrotoxicity
    
    # Calculate metrics
    library(pROC)
    
    nephro_roc <- roc(nephro_actual, nephro_pred$.pred_Yes)
    
    # Calibration
    nephro_calib <- calibration_plot(
      observed = nephro_actual,
      predicted = nephro_pred$.pred_Yes
    )
    
    # Brier score
    brier <- mean((nephro_pred$.pred_Yes - as.numeric(nephro_actual))^2)
    
    results$nephrotoxicity_validation <- list(
      auc = auc(nephro_roc),
      sensitivity = coords(nephro_roc, "best", ret = "sensitivity"),
      specificity = coords(nephro_roc, "best", ret = "specificity"),
      brier_score = brier,
      calibration_plot = nephro_calib
    )
  }
  
  return(results)
}

# ----------------------------------------------------------------------------
# CALIBRATION PLOT HELPER
# ----------------------------------------------------------------------------

calibration_plot <- function(observed, predicted, n_bins = 10) {
  
  calib_data <- tibble(
    observed = observed,
    predicted = predicted
  ) %>%
    mutate(
      pred_bin = cut(predicted, 
                     breaks = seq(0, 1, length.out = n_bins + 1),
                     include.lowest = TRUE)
    ) %>%
    group_by(pred_bin) %>%
    summarize(
      mean_predicted = mean(predicted),
      mean_observed = mean(as.numeric(observed)),
      n = n(),
      se = sqrt(mean_observed * (1 - mean_observed) / n),
      .groups = 'drop'
    )
  
  plot <- ggplot(calib_data, aes(x = mean_predicted, y = mean_observed)) +
    geom_point(aes(size = n)) +
    geom_errorbar(aes(ymin = mean_observed - 1.96*se,
                      ymax = mean_observed + 1.96*se),
                  width = 0.02) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = TRUE, color = "blue") +
    xlim(0, 1) + ylim(0, 1) +
    labs(x = "Predicted Probability",
         y = "Observed Frequency",
         title = "Calibration Plot",
         size = "N") +
    theme_bw()
  
  return(plot)
}
