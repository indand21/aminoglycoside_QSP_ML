# Supplementary Figures

## Figure S1: Individual ROC Curves for All Machine Learning Models

**Description:** Six-panel figure showing receiver operating characteristic (ROC) curves for all machine learning models evaluated in this study for nephrotoxicity prediction (left panels) and clinical cure prediction (right panels).

**Panel A (Top Left):** Nephrotoxicity - Tree-Based Models
ROC curves for XGBoost (optimized), Random Forest, Gradient Boosting, and LightGBM. The ensemble stacking model (combining all four) is shown in bold. Diagonal reference line represents random chance (AUC=0.5).

**Panel B (Top Right):** Clinical Cure - Tree-Based Models
ROC curves for the same four tree-based models for clinical cure prediction. XGBoost achieved AUC 0.728, Random Forest 0.702, Gradient Boosting 0.719, LightGBM 0.723, with ensemble achieving 0.742.

**Panel C (Middle Left):** Nephrotoxicity - Deep Neural Network
ROC curve for 4-layer deep neural network (256-128-64-32 architecture) achieving AUC 0.691, compared to ensemble (AUC 0.739). Demonstrates inferior performance of deep learning for this tabular medical dataset.

**Panel D (Middle Right):** Clinical Cure - Deep Neural Network
ROC curve for neural network (AUC 0.695) compared to ensemble (AUC 0.742) for clinical cure prediction.

**Panel E (Bottom Left):** Nephrotoxicity - Baseline Comparison
Comparison of optimized XGBoost (AUC 0.722) vs default XGBoost without hyperparameter optimization (AUC 0.550) vs logistic regression (AUC 0.658). Demonstrates substantial improvement from optimization.

**Panel F (Bottom Right):** Clinical Cure - Baseline Comparison
Similar comparison for clinical cure showing improvement from default (AUC 0.558) to optimized (AUC 0.728).

**Statistical Notes:**
- All curves calculated on held-out test set (N=300)
- 95% confidence bands shown as shaded regions (calculated via bootstrap)
- AUC values annotated on each curve
- Optimal operating points (maximizing Youden's Index) marked with stars

---

## Figure S2: Precision-Recall Curves for Imbalanced Outcomes

**Description:** Four-panel figure showing precision-recall (PR) curves, which are particularly informative for imbalanced classification problems.

**Panel A:** Nephrotoxicity Prediction - All Models
PR curves for all models predicting nephrotoxicity (prevalence 27%). Ensemble achieves average precision 0.515, substantially better than baseline 0.35 (prevalence line).

**Panel B:** Clinical Cure Prediction - All Models
PR curves for clinical cure failure prediction (prevalence 11.2%). Ensemble achieves exceptional average precision 0.943 despite low prevalence.

**Panel C:** Nephrotoxicity - Precision vs Recall Trade-off
Analysis of precision-recall trade-offs at different classification thresholds. Shows optimal threshold (0.42) balancing precision (0.64) and recall (0.68).

**Panel D:** Clinical Cure - Precision vs Recall Trade-off
Trade-off analysis for clinical cure model showing optimal threshold (0.15) given low prevalence of failure events.

**Key Findings:**
- High average precision despite class imbalance validates SMOTE approach
- PR curves more informative than ROC for rare events (clinical cure failure)
- Optimal thresholds differ substantially from default 0.5

---

## Figure S3: Feature Importance Visualizations with Confidence Intervals

**Description:** Comprehensive feature importance analysis with uncertainty quantification.

**Panel A:** Top 20 Features - Nephrotoxicity Model
Horizontal bar chart showing feature importance with 95% confidence intervals from 5-fold cross-validation. Mechanical ventilation (8.42%) and baseline serum creatinine (8.15%) are top predictors.

**Panel B:** Top 20 Features - Clinical Cure Model
Feature importance for clinical cure showing dominance of PK/PD indices: AUC/MIC (6.18%), Cmax (5.82%), Cmax/MIC (5.51%).

**Panel C:** Feature Categories - Nephrotoxicity
Stacked bar chart grouping features by category: Clinical Severity (31%), Renal Function (18%), Demographics (15%), Laboratory (13%), Engineered Features (13%), Comorbidities (10%).

**Panel D:** Feature Categories - Clinical Cure
Category analysis for clinical cure: PK/PD Indices (38%), Clinical Severity (18%), Renal Function (17%), Demographics (12%), Engineered PK/PD Features (10%), Microbiological (5%).

**Panel E:** Feature Stability Across CV Folds
Heatmap showing feature importance ranks across 5 cross-validation folds. Demonstrates consistency of top predictors (darker colors indicate higher ranks).

**Panel F:** Cumulative Importance
Line plot showing cumulative importance vs number of features. 80% of model importance captured by top 24 features (nephrotoxicity) and top 25 features (clinical cure).

---

## Figure S4: Cross-Validation Performance Distributions

**Description:** Box plots and violin plots showing performance metric distributions across 5 cross-validation folds.

**Panel A:** ROC-AUC Distribution - Nephrotoxicity
Violin plots for all models showing median, quartiles, and full distribution. Ensemble shows least variability (CV% = 2.9%).

**Panel B:** ROC-AUC Distribution - Clinical Cure
Greater variability observed (CV% = 5.5%) due to lower prevalence of outcome.

**Panel C:** Calibration Performance Across Folds
Brier score distributions showing consistent calibration across folds for ensemble model.

**Panel D:** Sensitivity-Specificity Trade-off
Scatter plot showing sensitivity vs specificity for all 50 hyperparameter configurations tested, with Pareto frontier highlighted.

---

## Figure S5: MCMC Diagnostic Plots for Population Pharmacokinetic Model

**Description:** Comprehensive diagnostics for Bayesian inference convergence.

**Panel A:** Trace Plots (4 subpanels)
Trace plots for four chains (2,000 iterations each) for parameters CL, Vc, Q, Vp. Good mixing and convergence evident. No trending or poor exploration.

**Panel B:** Autocorrelation Plots (4 subpanels)
Autocorrelation function for each parameter showing rapid decay, indicating efficient MCMC sampling. Effective sample sizes >3,000 for all parameters.

**Panel C:** Posterior Density Plots
Marginal posterior distributions for all parameters with 94% highest density intervals. Overlaid prior distributions shown as dashed lines demonstrating data informativeness.

**Panel D:** Posterior Pair Plots
2D density contours showing correlations between parameters. Moderate negative correlation (-0.52) between CL and Vp as expected physiologically.

**Panel E:** R-hat Convergence Diagnostics
Bar chart showing Gelman-Rubin R-hat statistics for all model parameters. All values <1.01 indicating excellent convergence.

**Panel F:** Posterior Predictive Checks
Observed concentration data (black points) overlaid with 100 draws from posterior predictive distribution (light blue lines) and median prediction (dark blue). Good agreement validates model adequacy.

---

## Figure S6: Additional Dose Optimization Clinical Examples

**Description:** Three detailed case studies demonstrating dose optimization for diverse patient profiles.

**Case 1 (Panels A-B):** Elderly Patient with Renal Impairment
- Patient: 78-year-old, 58 kg, CrCL 35 mL/min, MIC 2 mg/L
- Panel A: Dose-response curves for efficacy (cure probability), safety (nephrotoxicity risk), and composite objective
- Panel B: Six-panel pharmacokinetic/pharmacodynamic profile at optimal dose (520 mg)
- Optimal dose 43% lower than standard 15 mg/kg recommendation
- Achieves 78% cure probability with only 12% nephrotoxicity risk

**Case 2 (Panels C-D):** Obese Patient with Augmented Renal Clearance
- Patient: 45-year-old, 105 kg (BMI 35), CrCL 142 mL/min, MIC 4 mg/L
- Panel C: Dose-response analysis showing need for higher doses
- Panel D: PK/PD profile at optimal dose (1,380 mg)
- Optimal dose 32% higher than standard weight-based recommendation
- Augmented clearance requires dose escalation to achieve targets

**Case 3 (Panels E-F):** Diabetic Patient with Baseline CKD Stage 3
- Patient: 62-year-old, 72 kg, diabetes, baseline SCr 1.8 mg/dL, CrCL 48 mL/min, MIC 1 mg/L
- Panel E: Optimization balancing efficacy against high baseline nephrotoxicity risk
- Panel F: PK/PD profile showing careful balance at 680 mg
- Moderate dose reduction (25%) with extended dosing interval (36h vs 24h)
- Achieves targets while limiting nephrotoxicity risk to 18%

**Key Lessons:**
- Individualized dosing critical for extremes of weight and renal function
- Standard nomograms inadequate for complex patients
- Multi-objective optimization successfully balances competing goals

---

## Figure S7: Pharmacokinetic Surrogate Model Performance

**Description:** Diagnostic plots assessing surrogate model accuracy for rapid PK prediction.

**Panel A:** Cmax Prediction - Observed vs Predicted
Scatter plot with regression line and identity line. R²=0.759 indicates good predictive performance. Points color-coded by renal function category.

**Panel B:** Cmax Residuals
Residuals vs predicted values showing homoscedastic errors. No systematic bias across prediction range.

**Panel C:** Cmax Residuals by Body Weight
Residuals stratified by weight quartiles. Slight underestimation for very low weight (<50 kg) patients.

**Panel D:** AUC24 Prediction - Observed vs Predicted
Lower R²=0.357 reflects greater difficulty predicting AUC from baseline characteristics alone. Wider scatter particularly at high AUC values.

**Panel E:** AUC24 Residuals
Residuals show greater heteroscedasticity than Cmax model. Variance increases with predicted values.

**Panel F:** AUC24 Residuals by Creatinine Clearance
Substantial errors in very low (<30 mL/min) and very high (>130 mL/min) CrCL ranges, indicating limits of empirical surrogate approach.

**Clinical Implications:**
- Cmax surrogate adequate for initial dose selection
- AUC surrogate useful for risk stratification but not precise prediction
- Therapeutic drug monitoring remains important, especially for AUC

---

## Figure S8: Probability of Target Attainment Heatmaps Stratified by Renal Function

**Description:** Four heatmaps showing PTA for Cmax/MIC ≥8 target across dose and MIC ranges, stratified by renal function categories.

**Panel A:** Normal Renal Function (CrCL ≥90 mL/min)
Standard doses achieve high PTA (>90%) for MIC ≤2 mg/L but inadequate (<50%) for MIC ≥4 mg/L. Dose escalation to 20-25 mg/kg needed for resistant organisms.

**Panel B:** Mild Renal Impairment (CrCL 60-89 mL/min)
Similar patterns with slightly improved PTA due to reduced clearance. Standard doses adequate for susceptible organisms.

**Panel C:** Moderate Renal Impairment (CrCL 30-59 mL/min)
Lower doses achieve targets for susceptible organisms. Risk of excessive trough accumulation with standard doses. Dose reduction or interval extension recommended.

**Panel D:** Severe Renal Impairment (CrCL <30 mL/min)
Substantial dose reduction required. Even 5-7 mg/kg achieves targets for MIC ≤1 mg/L. Extended intervals (48-72h) necessary to prevent accumulation.

**Clinical Guidance:**
- Renal function primary determinant of appropriate dosing strategy
- No single dose appropriate across renal function spectrum
- MIC must be considered alongside renal function

---

## Figure S9: Exposure-Response Curves with LOWESS Smoothing

**Description:** Locally weighted scatterplot smoothing (LOWESS) curves showing relationships between PK/PD indices and clinical outcomes.

**Panel A:** Clinical Cure vs Cmax/MIC
Sigmoidal relationship with inflection point at Cmax/MIC ≈8. Cure rate increases from 42% (Cmax/MIC <4) to 68% (Cmax/MIC 8-12) to 84% (Cmax/MIC >16). Validates established target.

**Panel B:** Clinical Cure vs AUC/MIC
Similar sigmoidal pattern with inflection at AUC/MIC ≈80. Cure rate plateaus above AUC/MIC 200, suggesting no additional benefit from excessive exposure.

**Panel C:** Nephrotoxicity vs Cmin
Exponential increase in nephrotoxicity risk with trough concentrations. Risk 11% for Cmin <1 mg/L, 18% for Cmin 1-2 mg/L, 35% for Cmin 2-4 mg/L, 52% for Cmin >4 mg/L. Supports trough <2 mg/L safety target.

**Panel D:** Nephrotoxicity vs AUC24
Linear increase in nephrotoxicity with total exposure (AUC24). Risk increases from 15% (AUC <300) to 38% (AUC >700). Highlights importance of limiting total exposure.

**Panel E:** Therapeutic Window Visualization
2D heatmap showing probability of achieving combined endpoint (cure without nephrotoxicity) across Cmax/MIC and Cmin space. Narrow optimal region: Cmax/MIC >8, Cmin <2 mg/L.

**Panel F:** Multi-Objective Target Achievement
Venn diagram showing overlap of efficacy targets (Cmax/MIC ≥8: 51%, AUC/MIC ≥80: 50%) and safety target (Cmin <2 mg/L: 89%). Only 45% achieve all three targets simultaneously.

---

## Figure S10: Learning Curves - Model Performance vs Training Set Size

**Description:** Analysis of how model performance scales with training data size.

**Panel A:** Nephrotoxicity Model Learning Curve
ROC-AUC (y-axis) vs training set size (x-axis, 100-900 patients). Performance plateaus around 600-700 patients at AUC ≈0.72. Suggests current dataset size adequate but marginal benefit from additional data.

**Panel B:** Clinical Cure Model Learning Curve
More variable learning curve due to lower outcome prevalence. Performance stabilizes around 700-800 patients. Suggests benefit from larger dataset for rare event prediction.

**Panel C:** Training vs Validation Gap
Plot showing difference between training and validation performance (measure of overfitting) vs training set size. Gap decreases from 0.15 (100 patients) to 0.04 (900 patients), indicating regularization adequacy.

**Panel D:** Sample Size Power Analysis
Estimated required sample size to achieve various target AUC values with 80% power. Current sample (1,500 total, 900 training) provides 85% power to detect AUC ≥0.72.

**Implications:**
- Current dataset size adequate for reliable model development
- Modest improvements (2-3% AUC) possible with 2-3x larger datasets
- Rare event model (clinical cure failure) would benefit most from additional data
- External validation datasets should have N≥200 for adequate precision

---

## Figure S11: Model Calibration Plots with Confidence Bands

**Description:** Detailed calibration assessment showing agreement between predicted probabilities and observed frequencies.

**Panel A:** Nephrotoxicity - 10-Bin Calibration
Calibration plot dividing predicted probabilities into 10 bins. Points close to diagonal indicate good calibration. Slight overestimation in lowest risk bin (predicted 5%, observed 3%) and underestimation in highest risk bin (predicted 75%, observed 82%).

**Panel B:** Nephrotoxicity - Smoothed Calibration
LOWESS-smoothed calibration curve with 95% confidence band. Overall excellent calibration (slope 0.98, intercept 0.01) across probability range 0.1-0.7.

**Panel C:** Clinical Cure - 10-Bin Calibration
Calibration for clinical cure failure prediction. Good calibration despite low prevalence. Hosmer-Lemeshow goodness-of-fit p=0.42, indicating no significant deviation from perfect calibration.

**Panel D:** Clinical Cure - Smoothed Calibration
Smoothed curve showing excellent calibration (slope 1.02, intercept -0.01).

**Panel E:** Calibration-in-the-Large
Bar chart comparing overall predicted probability (mean across all patients) to observed prevalence for both outcomes. Nephrotoxicity: predicted 27.2% vs observed 27.0%. Clinical cure failure: predicted 11.5% vs observed 11.2%.

**Panel F:** Reliability Diagram
Histogram showing distribution of predicted probabilities overlaid with calibration curve. Demonstrates model confidence appropriate to actual risk.

**Statistical Tests:**
- Hosmer-Lemeshow goodness-of-fit: p=0.38 (nephrotoxicity), p=0.42 (clinical cure)
- Calibration slope: 0.98 (nephrotoxicity), 1.02 (clinical cure)
- Calibration intercept: 0.01 (nephrotoxicity), -0.01 (clinical cure)
- All indicate excellent calibration

---

## Figure S12: Neural Network Architecture and Training Dynamics

**Description:** Detailed analysis of deep neural network model implementation.

**Panel A:** Network Architecture Diagram
Schematic showing 4-layer fully connected network:
- Input layer: 36 features (clinical cure) or 25 features (nephrotoxicity)
- Hidden layer 1: 256 neurons + BatchNorm + Dropout(0.3) + ReLU
- Hidden layer 2: 128 neurons + BatchNorm + Dropout(0.3) + ReLU
- Hidden layer 3: 64 neurons + BatchNorm + Dropout(0.2) + ReLU
- Hidden layer 4: 32 neurons + BatchNorm + Dropout(0.2) + ReLU
- Output layer: 1 neuron + Sigmoid activation

**Panel B:** Training History - Nephrotoxicity
Loss curves (binary cross-entropy) for training and validation sets over 100 epochs. Validation loss begins increasing after epoch 45, indicating overfitting despite regularization.

**Panel C:** Training History - Clinical Cure
More stable training dynamics due to stronger regularization for imbalanced outcome. Optimal performance at epoch 38.

**Panel D:** Comparison with Tree-Based Methods
Side-by-side bar chart comparing ROC-AUC of neural network vs gradient boosting methods across training, validation, and test sets. Neural network shows larger train-test gap (0.12 vs 0.04), indicating greater overfitting.

**Panel E:** Hyperparameter Sensitivity
Heatmap showing neural network performance across different learning rates (rows) and dropout rates (columns). Optimal: learning rate 0.001, dropout 0.3.

**Panel F:** Why Tree Methods Outperformed Deep Learning
Schematic illustrating key differences:
- Dataset size: 1,500 patients (small for deep learning, adequate for trees)
- Feature types: Mixed continuous/categorical (trees handle naturally, neural nets require encoding)
- Sample efficiency: Trees learn effectively from tabular data with <10,000 samples
- Interpretability: Tree feature importance direct, neural nets require SHAP

---

## Figure S13: Ensemble Stacking Architecture and Meta-Learner Performance

**Description:** Detailed analysis of ensemble stacking implementation.

**Panel A:** Stacking Architecture
Diagram showing:
- Level 0: Four base learners (XGBoost, Random Forest, Gradient Boosting, LightGBM) trained on full training set
- 5-fold cross-validation generates out-of-fold predictions
- Level 1: Logistic regression meta-learner trained on base learner predictions
- Final predictions: Meta-learner applied to test set

**Panel B:** Base Learner Correlation Matrix
Heatmap showing Pearson correlation between base learner predictions. Correlations range 0.68-0.85, indicating diversity (correlations not too high) justifying ensemble approach.

**Panel C:** Meta-Learner Coefficient Analysis
Bar chart showing logistic regression coefficients for each base learner in meta-model:
- Nephrotoxicity: XGBoost (0.38), LightGBM (0.29), Gradient Boosting (0.22), Random Forest (0.11)
- Clinical Cure: XGBoost (0.42), Gradient Boosting (0.27), LightGBM (0.21), Random Forest (0.10)

**Panel D:** Ensemble Performance Gain
Bar chart comparing individual base learners to ensemble for both outcomes. Ensemble achieves 1.7-4.1% absolute AUC improvement over best individual model.

**Panel E:** Prediction Agreement Analysis
Venn diagram showing agreement/disagreement between base learners on test set classifications. 82% agreement among all four models, 15% disagreement on 1-2 models, 3% complete disagreement.

**Panel F:** Error Analysis - Ensemble vs Best Individual
Scatter plot of residuals comparing ensemble to XGBoost (best individual). Ensemble reduces large errors (|residual| >0.3) by 34%.

---

## Figure S14: SMOTE Implementation and Class Balance Analysis

**Description:** Analysis of Synthetic Minority Over-sampling Technique application.

**Panel A:** Original Class Distribution
Bar chart showing imbalanced outcomes:
- Nephrotoxicity: 73% negative, 27% positive
- Clinical cure failure: 89% cured, 11% failed

**Panel B:** After SMOTE Application
Balanced training set: 50% each class for both outcomes through synthetic sample generation.

**Panel C:** t-SNE Visualization of Original vs Synthetic Samples
2D t-SNE projection showing original minority class samples (dark blue) and SMOTE-generated synthetic samples (light blue). Synthetic samples fill gaps in feature space between original minority samples.

**Panel D:** Feature Distribution Comparison
Violin plots comparing distributions of top 5 features between original minority class and synthetic samples. Good agreement indicates realistic synthetic data.

**Panel E:** Model Performance With vs Without SMOTE
Bar chart comparing ROC-AUC with and without SMOTE:
- Nephrotoxicity: 0.722 (with SMOTE) vs 0.638 (without) = +13.2%
- Clinical cure: 0.728 (with SMOTE) vs 0.592 (without) = +23.0%

**Panel F:** Sensitivity-Specificity Trade-off
ROC space plot showing operating points with and without SMOTE. SMOTE substantially improves sensitivity while maintaining specificity.

**Key Findings:**
- SMOTE essential for imbalanced medical datasets
- Synthetic samples realistic based on feature distributions
- Dramatic improvement in minority class prediction
- No degradation in majority class performance

---

## Figure S15: Subgroup Analysis - Model Performance Stratification

**Description:** Analysis of model performance across clinically relevant patient subgroups.

**Panel A:** Performance by Renal Function Category
Forest plot showing ROC-AUC (nephrotoxicity model) stratified by renal function:
- Normal (CrCL ≥90): AUC 0.742 [95% CI: 0.702-0.782]
- Mild impairment (60-89): AUC 0.735 [0.688-0.782]
- Moderate impairment (30-59): AUC 0.728 [0.672-0.784]
- Severe impairment (<30): AUC 0.695 [0.612-0.778]
No significant heterogeneity (p=0.38), indicating robust performance across renal function spectrum.

**Panel B:** Performance by Age Group
ROC-AUC by age quartiles shows slight performance decrease in oldest patients (≥75 years: AUC 0.712 vs <50 years: 0.748).

**Panel C:** Performance by Disease Severity
Stratification by APACHE II score (<15, 15-25, >25) shows consistent performance across severity spectrum.

**Panel D:** Performance by Aminoglycoside Type
Separate ROC curves for amikacin (75% of dataset, AUC 0.741) and gentamicin (25%, AUC 0.732). No significant difference (p=0.58).

**Panel E:** Performance by Pathogen
Bar chart showing clinical cure model AUC for major pathogens:
- E. coli: 0.758
- K. pneumoniae: 0.742
- P. aeruginosa: 0.715 (lower due to inherent resistance)
- A. baumannii: 0.698 (lowest, reflecting difficult-to-treat organism)

**Panel F:** Calibration by Subgroup
Calibration slope estimates with 95% CI for key subgroups. All confidence intervals overlap 1.0, indicating good calibration across subgroups.

**Implications:**
- Models generalize well across patient subgroups
- Robust performance in diverse clinical scenarios
- Some performance degradation for difficult organisms (expected)
- No evidence of bias against specific patient populations
