# ============================================================================
# MACHINE LEARNING MODELS FOR AMINOGLYCOSIDE OPTIMIZATION
# ============================================================================

library(tidymodels)
library(xgboost)
library(ranger)
library(glmnet)
library(kernlab)

# ----------------------------------------------------------------------------
# MODEL 1: PK Parameter Prediction (surrogate for full PopPK)
# ----------------------------------------------------------------------------

# Predict individual CL and Vc from baseline covariates
# Useful for rapid dose calculation without full NONMEM run

train_pk_predictor <- function(ml_features, popPK_predictions) {
  
  # Merge PopPK predictions with features
  modeling_data <- ml_features %>%
    left_join(popPK_predictions, by = "patient_id") %>%
    filter(!is.na(ipred_cl), !is.na(ipred_vc))
  
  # Data split
  set.seed(123)
  data_split <- initial_split(modeling_data, prop = 0.8, strata = renal_cat)
  train_data <- training(data_split)
  test_data <- testing(data_split)
  
  # Cross-validation folds
  cv_folds <- vfold_cv(train_data, v = 10, strata = renal_cat)
  
  # Define model specifications
  
  # XGBoost
  xgb_spec <- boost_tree(
    trees = tune(),
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    learn_rate = tune()
  ) %>%
    set_engine("xgboost") %>%
    set_mode("regression")
  
  # Random Forest
  rf_spec <- rand_forest(
    trees = 1000,
    mtry = tune(),
    min_n = tune()
  ) %>%
    set_engine("ranger", importance = "impurity") %>%
    set_mode("regression")
  
  # Elastic Net
  enet_spec <- linear_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine("glmnet")
  
  # Create recipe for CL prediction
  rec_cl <- create_ml_recipe(train_data, "ipred_cl")
  
  # Create workflow set
  workflows_cl <- workflow_set(
    preproc = list(rec = rec_cl),
    models = list(xgb = xgb_spec, rf = rf_spec, enet = enet_spec)
  )
  
  # Tune all models
  grid_ctrl <- control_grid(save_pred = TRUE, parallel_over = "everything")
  
  tuned_models_cl <- workflows_cl %>%
    workflow_map(
      fn = "tune_grid",
      resamples = cv_folds,
      grid = 25,
      control = grid_ctrl,
      metrics = metric_set(rmse, rsq, mae),
      verbose = TRUE
    )
  
  # Select best model
  best_cl_model <- tuned_models_cl %>%
    rank_results(rank_metric = "rmse") %>%
    filter(.metric == "rmse") %>%
    slice(1)
  
  # Finalize and fit best model on full training set
  final_wf_cl <- tuned_models_cl %>%
    extract_workflow(best_cl_model$wflow_id) %>%
    finalize_workflow(
      select_best(
        extract_workflow_set_result(tuned_models_cl, best_cl_model$wflow_id),
        metric = "rmse"
      )
    )
  
  final_fit_cl <- fit(final_wf_cl, train_data)
  
  # Repeat for Vc prediction
  rec_vc <- create_ml_recipe(train_data, "ipred_vc")
  # ... (similar process)
  
  # Test set performance
  test_predictions_cl <- predict(final_fit_cl, test_data) %>%
    bind_cols(test_data %>% select(patient_id, ipred_cl))
  
  test_metrics_cl <- test_predictions_cl %>%
    metrics(truth = ipred_cl, estimate = .pred)
  
  return(list(
    cl_model = final_fit_cl,
    vc_model = final_fit_vc,  # From similar process
    test_metrics_cl = test_metrics_cl,
    test_metrics_vc = test_metrics_vc,
    test_predictions = test_predictions_cl
  ))
}

# ----------------------------------------------------------------------------
# MODEL 2: Clinical Outcome Prediction
# ----------------------------------------------------------------------------

# Predict nephrotoxicity and clinical cure
train_outcome_predictor <- function(ml_features) {
  
  # Two separate models:
  # 1. Nephrotoxicity (binary classification)
  # 2. Clinical cure (binary classification)
  
  # --- NEPHROTOXICITY MODEL ---
  
  # Data preparation
  nephrotox_data <- ml_features %>%
    filter(!is.na(nephrotoxicity)) %>%
    mutate(nephrotoxicity = factor(nephrotoxicity, 
                                   levels = c(FALSE, TRUE),
                                   labels = c("No", "Yes")))
  
  # Split data
  set.seed(456)
  nephro_split <- initial_split(nephrotox_data, prop = 0.75, 
                                strata = nephrotoxicity)
  nephro_train <- training(nephro_split)
  nephro_test <- testing(nephro_split)
  
  # Handle class imbalance with SMOTE
  library(themis)
  
  nephro_recipe <- create_ml_recipe(nephro_train, "nephrotoxicity") %>%
    step_smote(nephrotoxicity, over_ratio = 0.8)
  
  # Model specifications for classification
  
  # XGBoost
  xgb_class_spec <- boost_tree(
    trees = 500,
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    learn_rate = tune()
  ) %>%
    set_engine("xgboost", eval_metric = "auc") %>%
    set_mode("classification")
  
  # Random Forest
  rf_class_spec <- rand_forest(
    trees = 1000,
    mtry = tune(),
    min_n = tune()
  ) %>%
    set_engine("ranger", importance = "permutation") %>%
    set_mode("classification")
  
  # Logistic Regression with Elastic Net
  lr_spec <- logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine("glmnet")
  
  # Create workflow
  nephro_wf <- workflow_set(
    preproc = list(rec = nephro_recipe),
    models = list(xgb = xgb_class_spec, 
                  rf = rf_class_spec, 
                  lr = lr_spec)
  )
  
  # Cross-validation
  nephro_folds <- vfold_cv(nephro_train, v = 10, strata = nephrotoxicity)
  
  # Tune models
  nephro_tuned <- nephro_wf %>%
    workflow_map(
      fn = "tune_grid",
      resamples = nephro_folds,
      grid = 25,
      control = control_grid(save_pred = TRUE),
      metrics = metric_set(roc_auc, accuracy, sensitivity, specificity),
      verbose = TRUE
    )
  
  # Select best model (maximize AUC)
  best_nephro <- nephro_tuned %>%
    rank_results(rank_metric = "roc_auc") %>%
    filter(.metric == "roc_auc") %>%
    slice(1)
  
  # Finalize workflow
  final_nephro_wf <- nephro_tuned %>%
    extract_workflow(best_nephro$wflow_id) %>%
    finalize_workflow(
      select_best(
        extract_workflow_set_result(nephro_tuned, best_nephro$wflow_id),
        metric = "roc_auc"
      )
    )
  
  # Fit final model
  final_nephro_fit <- fit(final_nephro_wf, nephro_train)
  
  # Test set evaluation
  nephro_test_pred <- predict(final_nephro_fit, nephro_test, type = "prob") %>%
    bind_cols(predict(final_nephro_fit, nephro_test)) %>%
    bind_cols(nephro_test %>% select(patient_id, nephrotoxicity))
  
  # Calculate metrics
  nephro_metrics <- nephro_test_pred %>%
    metrics(truth = nephrotoxicity, estimate = .pred_class, .pred_Yes)
  
  # ROC curve
  nephro_roc <- nephro_test_pred %>%
    roc_curve(truth = nephrotoxicity, .pred_Yes) %>%
    autoplot()
  
  # Calibration plot
  nephro_cal <- nephro_test_pred %>%
    mutate(pred_bin = cut(.pred_Yes, breaks = seq(0, 1, 0.1))) %>%
    group_by(pred_bin) %>%
    summarize(
      predicted = mean(.pred_Yes),
      observed = mean(nephrotoxicity == "Yes"),
      n = n()
    ) %>%
    ggplot(aes(predicted, observed)) +
    geom_point(aes(size = n)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = "Nephrotoxicity Prediction Calibration",
         x = "Predicted Probability",
         y = "Observed Frequency") +
    theme_bw()
  
  # Variable importance
  nephro_vip <- final_nephro_fit %>%
    extract_fit_parsnip() %>%
    vip::vip(num_features = 20)
  
  # --- Repeat similar process for CLINICAL CURE ---
  # (Code structure similar to nephrotoxicity)
  
  return(list(
    nephrotoxicity_model = final_nephro_fit,
    nephro_metrics = nephro_metrics,
    nephro_roc = nephro_roc,
    nephro_calibration = nephro_cal,
    nephro_importance = nephro_vip,
    nephro_test_predictions = nephro_test_pred
    # ... clinical_cure_model results
  ))
}

# ----------------------------------------------------------------------------
# MODEL 3: Optimal Dose Prediction (Regression)
# ----------------------------------------------------------------------------

# Predict optimal dose to achieve Cmax/MIC > 8 while minimizing toxicity

train_optimal_dose_predictor <- function(ml_features) {
  
  # Target: dose that achieves PK/PD target with lowest risk
  # This is a constrained optimization problem
  # We'll use regression to predict needed dose
  
  modeling_data <- ml_features %>%
    filter(
      !is.na(achieved_cmax_mic),
      !is.na(mean_dose)
    ) %>%
    # Create target: dose adjustment factor
    mutate(
      target_met = achieved_cmax_mic > 8,
      
      # If target not met, estimate needed dose increase
      # If target met, estimate if dose could be reduced
      optimal_dose_factor = case_when(
        achieved_cmax_mic < 5 ~ mean_dose * (8 / achieved_cmax_mic),
        achieved_cmax_mic >= 8 & achieved_cmax_mic < 12 ~ mean_dose,
        achieved_cmax_mic >= 12 ~ mean_dose * (10 / achieved_cmax_mic),
        TRUE ~ mean_dose
      ),
      
      # Optimal dose per kg
      optimal_dose_kg = optimal_dose_factor / weight
    )
  
  # ... Model training similar to previous examples
  
  return(optimal_dose_model)
}
