# Supplementary Tables

## Table S1: Complete Machine Learning Model Performance Comparison

| **Model** | **Outcome** | **ROC-AUC** | **95% CI** | **Accuracy** | **Sensitivity** | **Specificity** | **PPV** | **NPV** | **F1-Score** | **Brier Score** | **Average Precision** |
|-----------|-------------|-------------|------------|--------------|-----------------|-----------------|---------|---------|--------------|-----------------|----------------------|
| **Ensemble Stacking (Final)** | | | | | | | | | | | |
| XGBoost+RF+GB+LightGBM | Nephrotoxicity | 0.739 | 0.717-0.760 | 0.720 | 0.680 | 0.750 | 0.640 | 0.780 | 0.660 | 0.180 | 0.515 |
| XGBoost+RF+GB+LightGBM | Clinical Cure | 0.742 | 0.696-0.788 | 0.730 | 0.710 | 0.760 | 0.690 | 0.770 | 0.700 | 0.170 | 0.943 |
| **Individual Base Learners** | | | | | | | | | | | |
| XGBoost (Optimized) | Nephrotoxicity | 0.722 | 0.698-0.746 | 0.700 | 0.650 | 0.740 | 0.620 | 0.760 | 0.630 | 0.190 | 0.492 |
| XGBoost (Optimized) | Clinical Cure | 0.728 | 0.682-0.774 | 0.710 | 0.680 | 0.740 | 0.670 | 0.750 | 0.670 | 0.180 | 0.928 |
| Random Forest | Nephrotoxicity | 0.698 | 0.672-0.724 | 0.680 | 0.620 | 0.720 | 0.590 | 0.740 | 0.600 | 0.210 | 0.468 |
| Random Forest | Clinical Cure | 0.702 | 0.654-0.750 | 0.690 | 0.650 | 0.730 | 0.640 | 0.730 | 0.640 | 0.200 | 0.912 |
| Gradient Boosting | Nephrotoxicity | 0.715 | 0.690-0.740 | 0.690 | 0.640 | 0.730 | 0.610 | 0.750 | 0.620 | 0.200 | 0.483 |
| Gradient Boosting | Clinical Cure | 0.719 | 0.672-0.766 | 0.700 | 0.670 | 0.740 | 0.660 | 0.740 | 0.660 | 0.190 | 0.920 |
| LightGBM | Nephrotoxicity | 0.718 | 0.693-0.743 | 0.700 | 0.640 | 0.740 | 0.620 | 0.750 | 0.630 | 0.190 | 0.486 |
| LightGBM | Clinical Cure | 0.723 | 0.676-0.770 | 0.710 | 0.680 | 0.750 | 0.670 | 0.750 | 0.670 | 0.180 | 0.925 |
| Logistic Regression | Nephrotoxicity | 0.658 | 0.631-0.685 | 0.640 | 0.580 | 0.680 | 0.550 | 0.700 | 0.560 | 0.230 | 0.424 |
| Logistic Regression | Clinical Cure | 0.662 | 0.612-0.712 | 0.650 | 0.600 | 0.690 | 0.590 | 0.700 | 0.590 | 0.220 | 0.885 |
| **Deep Neural Network** | | | | | | | | | | | |
| 4-Layer DNN (256-128-64-32) | Nephrotoxicity | 0.691 | 0.664-0.718 | 0.670 | 0.610 | 0.710 | 0.580 | 0.730 | 0.590 | 0.210 | 0.458 |
| 4-Layer DNN (256-128-64-32) | Clinical Cure | 0.695 | 0.646-0.744 | 0.680 | 0.630 | 0.720 | 0.620 | 0.720 | 0.620 | 0.200 | 0.905 |
| **Baseline (No Optimization)** | | | | | | | | | | | |
| XGBoost (Default) | Nephrotoxicity | 0.550 | 0.522-0.578 | 0.580 | 0.520 | 0.620 | 0.490 | 0.640 | 0.500 | 0.280 | 0.358 |
| XGBoost (Default) | Clinical Cure | 0.558 | 0.506-0.610 | 0.590 | 0.540 | 0.630 | 0.520 | 0.650 | 0.530 | 0.270 | 0.842 |

**Abbreviations:** ROC-AUC, area under receiver operating characteristic curve; CI, confidence interval; PPV, positive predictive value; NPV, negative predictive value; RF, Random Forest; GB, Gradient Boosting; DNN, Deep Neural Network.

**Notes:**
- All metrics calculated on held-out test set (N=300 patients, 20% of total dataset)
- 95% confidence intervals calculated using 1,000 bootstrap resamples
- Classification threshold optimized to maximize F1-score
- Training set: N=900 (60%), Validation set: N=300 (20%), Test set: N=300 (20%)

---

## Table S2: Cross-Validation Performance Metrics (5-Fold)

| **Model** | **Outcome** | **Fold 1 AUC** | **Fold 2 AUC** | **Fold 3 AUC** | **Fold 4 AUC** | **Fold 5 AUC** | **Mean AUC** | **SD** | **CV%** |
|-----------|-------------|----------------|----------------|----------------|----------------|----------------|--------------|--------|---------|
| **Ensemble Stacking** | | | | | | | | | |
| Final Model | Nephrotoxicity | 0.728 | 0.715 | 0.701 | 0.723 | 0.718 | 0.717 | 0.021 | 2.9% |
| Final Model | Clinical Cure | 0.712 | 0.695 | 0.681 | 0.704 | 0.688 | 0.696 | 0.038 | 5.5% |
| **XGBoost (Optimized)** | | | | | | | | | |
| Optimized | Nephrotoxicity | 0.715 | 0.702 | 0.688 | 0.710 | 0.705 | 0.704 | 0.018 | 2.6% |
| Optimized | Clinical Cure | 0.698 | 0.682 | 0.668 | 0.691 | 0.675 | 0.683 | 0.033 | 4.8% |

**Notes:**
- Stratified 5-fold cross-validation on training set (N=900)
- SMOTE applied within each fold to prevent data leakage
- Hyperparameters fixed at optimal values for consistency
- CV% = Coefficient of variation (SD/Mean × 100%)
- Low CV% indicates stable, robust performance across folds

---

## Table S3: Complete Feature Importance Rankings - Nephrotoxicity Prediction

| **Rank** | **Feature** | **Importance** | **95% CI** | **Cumulative %** | **Category** |
|----------|-------------|----------------|------------|------------------|--------------|
| 1 | Mechanical Ventilation | 8.42% | 7.85-9.01% | 8.42% | Clinical Severity |
| 2 | Baseline Serum Creatinine | 8.15% | 7.61-8.72% | 16.57% | Renal Function |
| 3 | Infection Site | 7.68% | 7.12-8.28% | 24.25% | Clinical |
| 4 | APACHE II Score | 7.32% | 6.78-7.89% | 31.57% | Clinical Severity |
| 5 | Body Weight | 6.58% | 6.05-7.14% | 38.15% | Demographics |
| 6 | Age | 6.21% | 5.68-6.76% | 44.36% | Demographics |
| 7 | Baseline Creatinine Clearance | 5.92% | 5.39-6.48% | 50.28% | Renal Function |
| 8 | Diabetes Mellitus | 5.45% | 4.93-6.00% | 55.73% | Comorbidities |
| 9 | Septic Shock | 5.18% | 4.66-5.73% | 60.91% | Clinical Severity |
| 10 | SOFA Score | 4.87% | 4.36-5.41% | 65.78% | Clinical Severity |
| 11 | Chronic Kidney Disease Stage | 4.52% | 4.02-5.05% | 70.30% | Comorbidities |
| 12 | Vasopressor Use | 4.28% | 3.78-4.81% | 74.58% | Clinical Severity |
| 13 | Baseline eGFR | 3.95% | 3.46-4.47% | 78.53% | Renal Function |
| 14 | Baseline Albumin | 3.62% | 3.14-4.13% | 82.15% | Laboratory |
| 15 | Age × Baseline Creatinine (Interaction) | 3.21% | 2.74-3.71% | 85.36% | Engineered |
| 16 | Body Mass Index | 2.85% | 2.39-3.35% | 88.21% | Demographics |
| 17 | Baseline Bilirubin | 2.42% | 1.98-2.90% | 90.63% | Laboratory |
| 18 | Diabetes × CrCL (Interaction) | 2.08% | 1.65-2.55% | 92.71% | Engineered |
| 19 | Composite Severity Score | 1.75% | 1.33-2.21% | 94.46% | Engineered |
| 20 | Sex | 1.52% | 1.11-1.97% | 95.98% | Demographics |
| 21 | Renal Function Score | 1.28% | 0.88-1.72% | 97.26% | Engineered |
| 22 | Height | 1.05% | 0.66-1.48% | 98.31% | Demographics |
| 23 | Elderly Indicator (Age ≥65) | 0.82% | 0.45-1.23% | 99.13% | Engineered |
| 24 | Impaired Renal Function Indicator | 0.58% | 0.23-0.97% | 99.71% | Engineered |
| 25 | Obesity Indicator (BMI ≥30) | 0.29% | 0.05-0.57% | 100.00% | Engineered |

**Notes:**
- Importance calculated using gain-based importance from XGBoost ensemble model
- 95% CI calculated across 5 cross-validation folds
- Top 10 features account for 65.78% of total model importance
- Top 20 features account for 95.98% of total model importance

---

## Table S4: Complete Feature Importance Rankings - Clinical Cure Prediction

| **Rank** | **Feature** | **Importance** | **95% CI** | **Cumulative %** | **Category** |
|----------|-------------|----------------|------------|------------------|--------------|
| 1 | AUC/MIC Ratio | 6.18% | 5.65-6.74% | 6.18% | PK/PD Index |
| 2 | Cmax | 5.82% | 5.29-6.38% | 12.00% | PK/PD Index |
| 3 | Cmax/MIC Ratio | 5.51% | 4.98-6.07% | 17.51% | PK/PD Index |
| 4 | Baseline eGFR | 5.48% | 4.95-6.04% | 22.99% | Renal Function |
| 5 | Body Weight | 5.22% | 4.69-5.78% | 28.21% | Demographics |
| 6 | APACHE II Score | 4.95% | 4.42-5.51% | 33.16% | Clinical Severity |
| 7 | Infection Site | 4.68% | 4.16-5.23% | 37.84% | Clinical |
| 8 | Age | 4.42% | 3.90-4.97% | 42.26% | Demographics |
| 9 | MIC Value | 4.15% | 3.64-4.70% | 46.41% | Microbiological |
| 10 | Baseline Creatinine Clearance | 3.88% | 3.37-4.42% | 50.29% | Renal Function |
| 11 | AUC24 | 3.62% | 3.12-4.15% | 53.91% | PK/PD Index |
| 12 | Mechanical Ventilation | 3.35% | 2.85-3.88% | 57.26% | Clinical Severity |
| 13 | SOFA Score | 3.08% | 2.59-3.61% | 60.34% | Clinical Severity |
| 14 | Target Attainment Score | 2.82% | 2.34-3.34% | 63.16% | Engineered PK/PD |
| 15 | Septic Shock | 2.55% | 2.08-3.06% | 65.71% | Clinical Severity |
| 16 | PK Composite (Cmax/MIC × AUC/MIC) | 2.28% | 1.82-2.78% | 67.99% | Engineered PK/PD |
| 17 | Baseline Serum Creatinine | 2.02% | 1.57-2.51% | 70.01% | Renal Function |
| 18 | Diabetes Mellitus | 1.85% | 1.40-2.33% | 71.86% | Comorbidities |
| 19 | Cmin (Trough) | 1.68% | 1.24-2.16% | 73.54% | PK/PD Index |
| 20 | Pathogen Species | 1.52% | 1.09-1.99% | 75.06% | Microbiological |
| 21 | AUC/MIC × APACHE II (Interaction) | 1.35% | 0.92-1.82% | 76.41% | Engineered PK/PD |
| 22 | Chronic Kidney Disease Stage | 1.18% | 0.76-1.64% | 77.59% | Comorbidities |
| 23 | Vasopressor Use | 1.02% | 0.61-1.47% | 78.61% | Clinical Severity |
| 24 | Cmax × Weight (Interaction) | 0.95% | 0.54-1.40% | 79.56% | Engineered PK/PD |
| 25 | PK Ratio (Cmax/MIC ÷ AUC/MIC) | 0.88% | 0.48-1.32% | 80.44% | Engineered PK/PD |
| 26 | Baseline Albumin | 0.82% | 0.42-1.26% | 81.26% | Laboratory |
| 27 | Cmax/MIC × CrCL (Interaction) | 0.75% | 0.36-1.18% | 82.01% | Engineered PK/PD |
| 28 | Body Mass Index | 0.68% | 0.30-1.10% | 82.69% | Demographics |
| 29 | Treatment Duration | 0.62% | 0.24-1.04% | 83.31% | Clinical |
| 30 | Baseline Bilirubin | 0.55% | 0.18-0.96% | 83.86% | Laboratory |
| 31 | Age × Baseline Creatinine (Interaction) | 0.48% | 0.12-0.88% | 84.34% | Engineered |
| 32 | Composite Severity Score | 0.42% | 0.06-0.82% | 84.76% | Engineered |
| 33 | Diabetes × CrCL (Interaction) | 0.35% | 0.01-0.73% | 85.11% | Engineered |
| 34 | Sex | 0.28% | -0.05-0.65% | 85.39% | Demographics |
| 35 | Renal Function Score | 0.22% | -0.10-0.58% | 85.61% | Engineered |
| 36 | Height | 0.15% | -0.16-0.50% | 85.76% | Demographics |

**Notes:**
- Importance calculated using gain-based importance from XGBoost ensemble model
- 95% CI calculated across 5 cross-validation folds
- Top 3 PK/PD indices (AUC/MIC, Cmax, Cmax/MIC) account for 17.51% of total importance
- Top 10 features account for 50.29% of total model importance
- Top 20 features account for 75.06% of total model importance
- Negative lower CI bounds for low-importance features indicate high uncertainty

---

## Table S5: Hyperparameter Optimization Results - Top 10 Configurations

### Nephrotoxicity Model

| **Rank** | **n_estimators** | **max_depth** | **learning_rate** | **subsample** | **colsample** | **gamma** | **min_child_weight** | **CV AUC** | **SD** |
|----------|------------------|---------------|-------------------|---------------|---------------|-----------|---------------------|------------|--------|
| 1 | 300 | 7 | 0.05 | 0.8 | 0.8 | 0.1 | 3 | 0.717 | 0.021 |
| 2 | 500 | 5 | 0.05 | 0.9 | 0.9 | 0.1 | 3 | 0.714 | 0.023 |
| 3 | 300 | 10 | 0.05 | 0.8 | 0.9 | 0 | 1 | 0.712 | 0.024 |
| 4 | 200 | 7 | 0.1 | 0.8 | 0.8 | 0.1 | 5 | 0.709 | 0.022 |
| 5 | 500 | 7 | 0.01 | 0.9 | 0.8 | 0.5 | 3 | 0.706 | 0.025 |
| 6 | 300 | 5 | 0.1 | 0.6 | 0.9 | 0.1 | 3 | 0.704 | 0.026 |
| 7 | 200 | 10 | 0.05 | 0.9 | 0.6 | 0 | 3 | 0.702 | 0.027 |
| 8 | 500 | 15 | 0.01 | 0.8 | 0.8 | 0.1 | 1 | 0.699 | 0.028 |
| 9 | 100 | 7 | 0.1 | 0.9 | 0.9 | 0 | 5 | 0.697 | 0.024 |
| 10 | 300 | 5 | 0.05 | 1.0 | 0.6 | 0.5 | 7 | 0.695 | 0.029 |

### Clinical Cure Model

| **Rank** | **n_estimators** | **max_depth** | **learning_rate** | **subsample** | **colsample** | **gamma** | **min_child_weight** | **CV AUC** | **SD** |
|----------|------------------|---------------|-------------------|---------------|---------------|-----------|---------------------|------------|--------|
| 1 | 300 | 10 | 0.05 | 0.9 | 0.8 | 0 | 3 | 0.696 | 0.038 |
| 2 | 500 | 7 | 0.05 | 0.8 | 0.9 | 0.1 | 1 | 0.693 | 0.041 |
| 3 | 200 | 10 | 0.1 | 0.9 | 0.8 | 0 | 3 | 0.690 | 0.039 |
| 4 | 300 | 7 | 0.05 | 0.8 | 0.8 | 0.1 | 5 | 0.687 | 0.042 |
| 5 | 500 | 5 | 0.05 | 0.9 | 0.9 | 0.5 | 3 | 0.684 | 0.044 |
| 6 | 300 | 15 | 0.01 | 0.8 | 0.8 | 0.1 | 1 | 0.681 | 0.045 |
| 7 | 200 | 7 | 0.1 | 0.6 | 0.9 | 0 | 3 | 0.679 | 0.043 |
| 8 | 500 | 10 | 0.05 | 0.9 | 0.6 | 0.1 | 3 | 0.676 | 0.046 |
| 9 | 100 | 7 | 0.1 | 0.9 | 0.9 | 0 | 5 | 0.673 | 0.041 |
| 10 | 300 | 5 | 0.05 | 1.0 | 0.6 | 0.5 | 7 | 0.670 | 0.047 |

**Notes:**
- Total configurations tested: 50 for each model
- Search method: RandomizedSearchCV with 5-fold stratified cross-validation
- Optimization metric: ROC-AUC
- Total computational time: ~4.5 hours on 8-core CPU
- colsample = colsample_bytree parameter in XGBoost

---

## Table S6: Patient Characteristics Stratified by Outcomes

### Nephrotoxicity Stratification

| **Characteristic** | **No Nephrotoxicity (N=1,095)** | **Nephrotoxicity (N=405)** | **p-value** | **Effect Size** |
|-------------------|--------------------------------|----------------------------|-------------|-----------------|
| **Demographics** | | | | |
| Age, years (mean ± SD) | 56.2 ± 15.3 | 63.8 ± 14.8 | <0.001 | 0.51 |
| Weight, kg (mean ± SD) | 63.1 ± 11.6 | 59.8 ± 12.5 | <0.001 | 0.28 |
| Male sex, n (%) | 648 (59%) | 252 (62%) | 0.312 | 0.06 |
| **Clinical Severity** | | | | |
| APACHE II (mean ± SD) | 20.5 ± 7.8 | 26.2 ± 7.9 | <0.001 | 0.73 |
| SOFA (mean ± SD) | 7.6 ± 3.2 | 9.5 ± 3.4 | <0.001 | 0.58 |
| Septic shock, n (%) | 285 (26%) | 165 (41%) | <0.001 | 0.32 |
| Mechanical ventilation, n (%) | 328 (30%) | 197 (49%) | <0.001 | 0.39 |
| Vasopressor use, n (%) | 547 (50%) | 278 (69%) | <0.001 | 0.39 |
| **Comorbidities** | | | | |
| Diabetes mellitus, n (%) | 350 (32%) | 175 (43%) | <0.001 | 0.23 |
| Chronic kidney disease, n (%) | 130 (12%) | 95 (23%) | <0.001 | 0.30 |
| **Renal Function** | | | | |
| Baseline SCr, mg/dL (mean ± SD) | 1.08 ± 0.65 | 1.52 ± 0.88 | <0.001 | 0.57 |
| CrCL, mL/min (mean ± SD) | 92.5 ± 42.8 | 72.3 ± 46.5 | <0.001 | 0.46 |
| eGFR, mL/min/1.73m² (mean ± SD) | 81.2 ± 35.6 | 62.8 ± 38.9 | <0.001 | 0.50 |
| **Laboratory Values** | | | | |
| Albumin, g/dL (mean ± SD) | 2.95 ± 0.55 | 2.52 ± 0.58 | <0.001 | 0.76 |
| Bilirubin, mg/dL (mean ± SD) | 1.32 ± 1.08 | 1.88 ± 1.32 | <0.001 | 0.47 |
| **PK/PD Indices** | | | | |
| Cmax, mg/L (mean ± SD) | 49.5 ± 18.2 | 52.1 ± 19.5 | 0.024 | 0.14 |
| Cmin, mg/L (mean ± SD) | 1.52 ± 1.85 | 2.65 ± 3.25 | <0.001 | 0.42 |
| AUC24, mg·h/L (mean ± SD) | 478 ± 248 | 532 ± 275 | <0.001 | 0.21 |

### Clinical Cure Stratification

| **Characteristic** | **Clinical Cure (N=1,332)** | **Treatment Failure (N=168)** | **p-value** | **Effect Size** |
|-------------------|----------------------------|------------------------------|-------------|-----------------|
| **Demographics** | | | | |
| Age, years (mean ± SD) | 57.5 ± 15.6 | 61.2 ± 16.2 | 0.005 | 0.23 |
| Weight, kg (mean ± SD) | 62.8 ± 11.8 | 59.5 ± 12.5 | <0.001 | 0.27 |
| Male sex, n (%) | 798 (60%) | 102 (61%) | 0.825 | 0.02 |
| **Clinical Severity** | | | | |
| APACHE II (mean ± SD) | 21.2 ± 8.0 | 25.8 ± 8.5 | <0.001 | 0.56 |
| SOFA (mean ± SD) | 7.9 ± 3.3 | 9.2 ± 3.6 | <0.001 | 0.38 |
| Septic shock, n (%) | 378 (28%) | 72 (43%) | <0.001 | 0.31 |
| Mechanical ventilation, n (%) | 442 (33%) | 83 (49%) | <0.001 | 0.33 |
| **PK/PD Indices** | | | | |
| Cmax/MIC (mean ± SD) | 19.8 ± 13.2 | 11.5 ± 8.8 | <0.001 | 0.72 |
| AUC/MIC (mean ± SD) | 192 ± 148 | 98 ± 82 | <0.001 | 0.76 |
| Cmax, mg/L (mean ± SD) | 51.2 ± 18.5 | 45.8 ± 17.2 | <0.001 | 0.30 |
| AUC24, mg·h/L (mean ± SD) | 512 ± 262 | 398 ± 218 | <0.001 | 0.47 |
| Cmin, mg/L (mean ± SD) | 1.78 ± 2.25 | 1.95 ± 2.82 | 0.325 | 0.07 |
| **Target Attainment** | | | | |
| Cmax/MIC ≥8, n (%) | 748 (56%) | 22 (13%) | <0.001 | 0.88 |
| AUC/MIC ≥80, n (%) | 715 (54%) | 39 (23%) | <0.001 | 0.64 |
| Combined targets, n (%) | 632 (47%) | 37 (22%) | <0.001 | 0.54 |

**Notes:**
- p-values from independent t-tests (continuous) or chi-square tests (categorical)
- Effect size: Cohen's d for continuous variables, Cramér's V for categorical variables
- Small effect: 0.2, Medium effect: 0.5, Large effect: 0.8
- SCr = serum creatinine, CrCL = creatinine clearance, eGFR = estimated glomerular filtration rate

---

## Table S7: Pharmacokinetic Surrogate Model Performance

| **Model** | **Target** | **Training R²** | **Test R²** | **RMSE** | **MAE** | **CV R² (mean ± SD)** | **Pearson r** | **Spearman ρ** |
|-----------|-----------|----------------|-------------|----------|---------|----------------------|---------------|----------------|
| **Cmax Surrogate** | | | | | | | | |
| XGBoost | Peak Concentration | 0.812 | 0.759 | 9.98 mg/L | 7.36 mg/L | 0.730 ± 0.029 | 0.871 | 0.865 |
| Random Forest | Peak Concentration | 0.785 | 0.732 | 10.52 mg/L | 7.82 mg/L | 0.715 ± 0.035 | 0.856 | 0.849 |
| Gradient Boosting | Peak Concentration | 0.798 | 0.745 | 10.25 mg/L | 7.58 mg/L | 0.722 ± 0.032 | 0.863 | 0.857 |
| **AUC24 Surrogate** | | | | | | | | |
| XGBoost | Area Under Curve | 0.425 | 0.357 | 235 mg·h/L | 163 mg·h/L | 0.293 ± 0.059 | 0.598 | 0.612 |
| Random Forest | Area Under Curve | 0.398 | 0.332 | 248 mg·h/L | 175 mg·h/L | 0.278 ± 0.064 | 0.576 | 0.591 |
| Gradient Boosting | Area Under Curve | 0.412 | 0.345 | 241 mg·h/L | 168 mg·h/L | 0.285 ± 0.061 | 0.587 | 0.602 |

**Top 5 Predictive Features - Cmax Surrogate:**
1. Dose (36.2% importance)
2. Body Weight (18.3% importance)
3. Creatinine Clearance (12.1% importance)
4. Body Surface Area (8.4% importance)
5. Volume of Distribution Estimate (7.5% importance)

**Top 5 Predictive Features - AUC24 Surrogate:**
1. Dose (28.5% importance)
2. Creatinine Clearance (21.3% importance)
3. Clearance Estimate (15.8% importance)
4. Body Weight (12.2% importance)
5. Age (8.7% importance)

**Notes:**
- RMSE = Root Mean Squared Error
- MAE = Mean Absolute Error
- Pearson r = Pearson correlation coefficient
- Spearman ρ = Spearman rank correlation coefficient
- Test set: N=300 patients
- 5-fold cross-validation performed on training set (N=900)

---

## Table S8: Dose Optimization Framework Performance Metrics

| **Metric** | **Value** | **95% CI** | **Notes** |
|-----------|----------|------------|-----------|
| **Computational Performance** | | | |
| Mean optimization time per patient | 4.2 seconds | 3.8-4.6 s | N=50 test patients |
| Median optimization time | 4.1 seconds | - | Range: 3.2-5.8 s |
| Iterations to convergence (mean) | 28 | 25-31 | Maximum: 30 iterations |
| **Objective Function Performance** | | | |
| Mean objective score (observed dosing) | 0.512 | 0.485-0.539 | Scale: 0-1 |
| Mean objective score (optimized dosing) | 0.658 | 0.632-0.684 | Scale: 0-1 |
| Relative improvement | 28.5% | 24.2-32.8% | p<0.001 |
| **Target Attainment (Optimized vs Observed)** | | | |
| Cmax/MIC ≥8 | 62% vs 48% | - | +29% relative |
| AUC/MIC ≥80 | 58% vs 46% | - | +26% relative |
| Trough <2 mg/L | 94% vs 88% | - | +7% relative |
| Combined targets | 52% vs 38% | - | +37% relative |
| **Predicted Clinical Outcomes** | | | |
| Probability of cure (optimized) | 0.82 | 0.78-0.86 | vs 0.75 observed |
| Probability of nephrotoxicity (optimized) | 0.18 | 0.15-0.21 | vs 0.25 observed |
| **Dose Recommendations** | | | |
| Mean optimized dose | 856 mg | 782-930 mg | vs 918 mg observed |
| Dose range | 420-1480 mg | - | Observed: 480-1600 mg |
| Coefficient of variation | 32% | - | vs 42% observed |

**Notes:**
- Optimization performed using Bayesian optimization with Gaussian process surrogate
- Objective function weights: 40% cure, 30% safety, 20% Cmax/MIC target, 10% trough safety
- All p-values <0.001 for target attainment improvements (McNemar's test)
- Computational performance measured on standard hardware (Intel Xeon E5-2680 v4, 2.4 GHz)
