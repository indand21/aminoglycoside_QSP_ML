# Table 3: Machine Learning Model Performance Metrics

| **Model** | **Outcome** | **ROC-AUC** | **Accuracy** | **Sensitivity** | **Specificity** | **PPV** | **NPV** | **F1-Score** | **Brier Score** |
|-----------|-------------|-------------|--------------|-----------------|-----------------|---------|---------|--------------|-----------------|
| **Ensemble Stacking (Final Model)** | | | | | | | | | |
| XGBoost + RF + GB + LightGBM | Nephrotoxicity | **0.739** | 0.72 | 0.68 | 0.75 | 0.64 | 0.78 | 0.66 | 0.18 |
| XGBoost + RF + GB + LightGBM | Clinical Cure | **0.742** | 0.73 | 0.71 | 0.76 | 0.69 | 0.77 | 0.70 | 0.17 |
| **Individual Base Learners** | | | | | | | | | |
| XGBoost | Nephrotoxicity | 0.722 | 0.70 | 0.65 | 0.74 | 0.62 | 0.76 | 0.63 | 0.19 |
| XGBoost | Clinical Cure | 0.728 | 0.71 | 0.68 | 0.74 | 0.67 | 0.75 | 0.67 | 0.18 |
| Random Forest | Nephrotoxicity | 0.698 | 0.68 | 0.62 | 0.72 | 0.59 | 0.74 | 0.60 | 0.21 |
| Random Forest | Clinical Cure | 0.702 | 0.69 | 0.65 | 0.73 | 0.64 | 0.73 | 0.64 | 0.20 |
| Gradient Boosting | Nephrotoxicity | 0.715 | 0.69 | 0.64 | 0.73 | 0.61 | 0.75 | 0.62 | 0.20 |
| Gradient Boosting | Clinical Cure | 0.719 | 0.70 | 0.67 | 0.74 | 0.66 | 0.74 | 0.66 | 0.19 |
| LightGBM | Nephrotoxicity | 0.718 | 0.70 | 0.64 | 0.74 | 0.62 | 0.75 | 0.63 | 0.19 |
| LightGBM | Clinical Cure | 0.723 | 0.71 | 0.68 | 0.75 | 0.67 | 0.75 | 0.67 | 0.18 |
| Logistic Regression | Nephrotoxicity | 0.658 | 0.64 | 0.58 | 0.68 | 0.55 | 0.70 | 0.56 | 0.23 |
| Logistic Regression | Clinical Cure | 0.662 | 0.65 | 0.60 | 0.69 | 0.59 | 0.70 | 0.59 | 0.22 |
| **Deep Neural Network** | | | | | | | | | |
| 4-layer DNN (256→128→64→32) | Nephrotoxicity | 0.691 | 0.67 | 0.61 | 0.71 | 0.58 | 0.73 | 0.59 | 0.21 |
| 4-layer DNN (256→128→64→32) | Clinical Cure | 0.695 | 0.68 | 0.63 | 0.72 | 0.62 | 0.72 | 0.62 | 0.20 |

**Abbreviations:** ROC-AUC, area under the receiver operating characteristic curve; PPV, positive predictive value; NPV, negative predictive value; RF, Random Forest; GB, Gradient Boosting; DNN, Deep Neural Network.

**Model Development Details:**
- **Training/Validation/Test Split:** 60%/20%/20% (900/300/300 patients)
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) applied to training set
- **Hyperparameter Optimization:** RandomizedSearchCV with 50 iterations and 5-fold cross-validation
- **Ensemble Method:** Stacked generalization with logistic regression meta-learner
- **Feature Engineering:** 36 engineered features derived from 18 baseline patient characteristics and pharmacokinetic indices

**Performance Interpretation:** The ensemble stacking approach achieved clinically useful discriminative ability (ROC-AUC >0.70) for both outcomes, exceeding individual base learner performance by 2-6%. The nephrotoxicity model (AUC 0.739) and clinical cure model (AUC 0.742) demonstrated balanced sensitivity and specificity, with acceptable calibration (Brier scores 0.17-0.18). Deep neural networks underperformed tree-based methods, consistent with recent literature demonstrating superiority of gradient boosting for tabular medical data.

**Clinical Utility Threshold:** ROC-AUC values ≥0.70 are generally considered clinically useful for risk prediction models in critical care. The achieved performance (0.74) is comparable to published aminoglycoside nephrotoxicity prediction models (0.62-0.73) and superior to traditional clinical scoring systems.
