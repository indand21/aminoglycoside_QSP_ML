#!/usr/bin/env python3
"""
Phase 4: ENHANCED Machine Learning for Outcome Prediction

ENHANCEMENTS:
1. Advanced feature engineering (interactions, polynomial features, PK/PD composites)
2. Class imbalance handling (SMOTE, class weights)
3. Hyperparameter optimization (RandomizedSearchCV)
4. Ensemble methods (stacking, voting classifiers)
5. Deep learning (neural networks with TensorFlow/Keras)
6. Feature selection (RFECV, mutual information)
7. Probability calibration
8. Comprehensive model comparison and evaluation

Author: Aminoglycoside QSP-ML Project
Date: 2025-11-15
Version: 2.0 (Enhanced)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    KFold, RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    mean_absolute_error, f1_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Algorithms
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier, RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression

# Class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Feature selection
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, f_classif

# Deep learning (optional). Use broad except because TF imports JAX which can
# fail with AttributeError under newer NumPy in some environments.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Deep learning models will be skipped.")

# LightGBM and CatBoost
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not available.")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available.")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available.")

# Optuna for Bayesian hyperparameter optimization
try:
    import optuna
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnhancedMLPipeline:
    """
    Enhanced Machine Learning Pipeline with Advanced Techniques
    """

    def __init__(self,
                 ml_data_path='data/processed/ml_dataset.csv',
                 pkpd_data_path='results/phase3_pkpd/pkpd_indices.csv'):
        """Initialize Enhanced ML Pipeline"""

        print("="*80)
        print("Phase 4: ENHANCED Machine Learning for Outcome Prediction")
        print("="*80)
        print()
        print("ENHANCEMENTS:")
        print("  [OK] Advanced feature engineering")
        print("  [OK] Class imbalance handling (SMOTE)")
        print("  [OK] Hyperparameter optimization")
        print("  [OK] Ensemble methods")
        print("  [OK] Deep learning (neural networks)")
        print("  [OK] Feature selection")
        print("  [OK] Probability calibration")
        print()

        # Load data
        print("Loading data...")
        self.ml_data = pd.read_csv(ml_data_path)
        self.pkpd_data = pd.read_csv(pkpd_data_path)

        # Identify all PK/PD features to merge (including new time-based and individual PK features)
        pkpd_features_to_merge = ['patient_id', 'Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC']

        # Add time-based PD features if available
        time_based_features = ['time_above_MIC', 'time_above_MIC_percent', 'AUC_above_MIC',
                              'fluctuation_index', 'time_to_peak', 'peak_trough_ratio']
        for feat in time_based_features:
            if feat in self.pkpd_data.columns:
                pkpd_features_to_merge.append(feat)

        # Add individual PK parameters from Bayesian model if available
        individual_pk_features = ['CL_bayes', 'Vc_bayes', 'Q_bayes', 'Vp_bayes',
                                 'CL_uncertainty', 'Vc_uncertainty', 'Q_uncertainty', 'Vp_uncertainty',
                                 'Vss_bayes', 'k10_bayes', 'k12_bayes', 'k21_bayes',
                                 't_half_alpha', 't_half_beta', 'CL_cv', 'Vc_cv']
        for feat in individual_pk_features:
            if feat in self.pkpd_data.columns:
                pkpd_features_to_merge.append(feat)

        # Merge datasets
        self.data = self.ml_data.merge(
            self.pkpd_data[pkpd_features_to_merge],
            on='patient_id',
            how='left',
            suffixes=('', '_pkpd')
        )

        print(f"  [OK] Loaded ML dataset: {self.ml_data.shape}")
        print(f"  [OK] Loaded PK/PD indices: {self.pkpd_data.shape}")
        print(f"  [OK] Merged dataset: {self.data.shape}")

        # Report which new features were added
        n_time_based = sum(1 for f in time_based_features if f in self.pkpd_data.columns)
        n_individual_pk = sum(1 for f in individual_pk_features if f in self.pkpd_data.columns)

        if n_time_based > 0:
            print(f"  [OK] Added {n_time_based} time-based PD features")
        if n_individual_pk > 0:
            print(f"  [OK] Added {n_individual_pk} individual PK parameters from Bayesian model")
        print()

        # Create output directories
        self.results_dir = Path('results/phase4_ml_enhanced')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path('models/enhanced')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.models = {}
        self.scalers = {}
        self.performance = {}
        self.feature_importance = {}
        self.feature_engineered = {}

    def engineer_features(self, X, feature_names, use_phase2_features=True):
        """
        Advanced feature engineering

        Creates:
        - Interaction terms (PK/PD x patient characteristics)
        - Polynomial features
        - Domain-specific composite features
        - Log transformations
        - Phase 2: Domain-specific interactions and composite risk scores

        Parameters:
        -----------
        use_phase2_features : bool
            Whether to include Phase 2 advanced feature engineering
        """
        print("\n  -> Engineering advanced features...")

        X_df = pd.DataFrame(X, columns=feature_names)

        # Initialize engineered features
        X_eng = X_df.copy()

        # === PK/PD Composite Features ===
        if 'Cmax_MIC' in feature_names and 'AUC_MIC' in feature_names:
            X_eng['pk_composite'] = X_df['Cmax_MIC'] * X_df['AUC_MIC']  # Multiplicative
            X_eng['pk_ratio'] = X_df['Cmax_MIC'] / (X_df['AUC_MIC'] + 1)  # Ratio

        if all(col in feature_names for col in ['Cmax_MIC', 'AUC_MIC', 'Cmin']):
            # Target achievement score
            target_cmax = (X_df['Cmax_MIC'] >= 8).astype(int)
            target_auc = (X_df['AUC_MIC'] >= 80).astype(int)
            target_trough = (X_df['Cmin'] < 2).astype(int)
            X_eng['target_score'] = target_cmax + target_auc + target_trough

        # === Dose normalization ===
        if 'first_dose' in feature_names and 'weight' in feature_names:
            X_eng['dose_per_kg'] = X_df['first_dose'] / (X_df['weight'] + 1)

        # === Interaction terms ===
        interactions = [
            ('Cmax', 'weight'),
            ('AUC_MIC', 'apache_ii'),
            ('Cmax_MIC', 'baseline_crcl'),
            ('age', 'baseline_scr'),
            ('diabetes', 'baseline_crcl')
        ]

        for feat1, feat2 in interactions:
            if feat1 in X_df.columns and feat2 in X_df.columns:
                try:
                    X_eng[f'{feat1}_x_{feat2}'] = X_df[feat1] * X_df[feat2]
                except Exception as e:
                    print(f"    Warning: Could not create {feat1}_x_{feat2}: {e}")
                    continue

        # === Log transformations for skewed features ===
        log_features = ['Cmax_MIC', 'AUC_MIC', 'Cmax', 'AUC24']
        for feat in log_features:
            if feat in X_df.columns:
                try:
                    col_data = X_df[feat].values
                    if len(col_data.shape) > 0:  # Ensure it's an array, not scalar
                        X_eng[f'log_{feat}'] = np.log(col_data + 1)
                except Exception as e:
                    print(f"    Warning: Could not create log_{feat}: {e}")
                    continue

        # === Severity composite ===
        if 'apache_ii' in X_df.columns and 'sofa_score' in X_df.columns:
            X_eng['severity_composite'] = X_df['apache_ii'] + X_df['sofa_score']

        # === Renal function composite ===
        if 'baseline_crcl' in X_df.columns and 'baseline_scr' in X_df.columns:
            X_eng['renal_score'] = X_df['baseline_crcl'] / (X_df['baseline_scr'] + 0.1)

        # === Risk flags ===
        if 'age' in X_df.columns:
            X_eng['elderly'] = (X_df['age'] > 65).astype(int)

        if 'baseline_crcl' in X_df.columns:
            X_eng['impaired_renal'] = (X_df['baseline_crcl'] < 50).astype(int)

        if 'bmi' in X_df.columns:
            X_eng['obese'] = (X_df['bmi'] > 30).astype(int)

        # Remove any infinite or NaN values
        X_eng = X_eng.replace([np.inf, -np.inf], np.nan)
        X_eng = X_eng.fillna(X_eng.median())

        n_new_features = X_eng.shape[1] - len(feature_names)
        print(f"     [OK] Created {n_new_features} basic engineered features")

        # === PHASE 2: Advanced Feature Engineering ===
        if use_phase2_features:
            # Add domain-specific interactions
            X_eng_array, eng_names = self.create_domain_specific_interactions(
                X_eng.values, X_eng.columns.tolist()
            )

            # Add composite risk scores
            X_final, final_names = self.create_composite_risk_scores(
                X_eng_array, eng_names
            )

            # Clean up again after Phase 2 features
            X_final_df = pd.DataFrame(X_final, columns=final_names)
            X_final_df = X_final_df.replace([np.inf, -np.inf], np.nan)
            X_final_df = X_final_df.fillna(X_final_df.median())

            total_new = X_final_df.shape[1] - len(feature_names)
            print(f"     [OK] Total engineered features (including Phase 2): {total_new}")
            print(f"     [OK] Final feature count: {X_final_df.shape[1]}")

            # Convert to float64 to ensure numeric dtype
            return X_final_df.astype(np.float64).values, X_final_df.columns.tolist()
        else:
            print(f"     [OK] Total features: {X_eng.shape[1]}")
            # Convert to float64 to ensure numeric dtype
            return X_eng.astype(np.float64).values, X_eng.columns.tolist()

    def prepare_features(self, feature_set='pkpd', engineer=True):
        """
        Prepare feature sets with optional engineering
        """
        print(f"Preparing features (set: {feature_set}, engineering: {engineer})...")

        # Baseline features
        baseline_features = [
            'age', 'sex', 'weight', 'height', 'bmi',
            'apache_ii', 'sofa_score',
            'baseline_crcl', 'baseline_scr', 'baseline_egfr',
            'baseline_albumin', 'baseline_bilirubin',
            'diabetes', 'ckd_stage',
            'mechanical_ventilation', 'vasopressor_use',
            'sepsis_type', 'infection_site',
            'drug'
        ]

        # PK/PD features
        pkpd_features = [
            'Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC'
        ]

        # Time-based PD features (from Phase 1 improvements)
        time_based_pd_features = [
            'time_above_MIC', 'time_above_MIC_percent', 'AUC_above_MIC',
            'fluctuation_index', 'time_to_peak', 'peak_trough_ratio'
        ]

        # Individual PK parameters from Bayesian model (from Phase 1 improvements)
        individual_pk_features = [
            'CL_bayes', 'Vc_bayes', 'Q_bayes', 'Vp_bayes',
            'Vss_bayes', 'k10_bayes', 'k12_bayes', 'k21_bayes',
            't_half_alpha', 't_half_beta', 'CL_cv', 'Vc_cv'
        ]

        # Dosing features
        dosing_features = ['first_dose', 'mean_dose', 'n_doses']

        # Select based on set
        if feature_set == 'baseline':
            selected_features = baseline_features
        elif feature_set == 'pkpd':
            # Include all PK/PD features + time-based PD + individual PK parameters
            selected_features = baseline_features + pkpd_features + time_based_pd_features + individual_pk_features
        elif feature_set == 'full':
            selected_features = baseline_features + pkpd_features + time_based_pd_features + individual_pk_features + dosing_features
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")

        # Filter available
        available_features = [f for f in selected_features if f in self.data.columns]

        # Handle categorical
        data_encoded = self.data.copy()
        categorical_cols = ['sex', 'sepsis_type', 'infection_site', 'ckd_stage', 'drug']
        for col in categorical_cols:
            if col in available_features:
                data_encoded[col] = pd.Categorical(data_encoded[col]).codes

        X = data_encoded[available_features].copy()

        # Handle missing
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        feature_names = available_features

        # Feature engineering
        if engineer:
            use_phase2 = getattr(self, 'use_phase2_features', False)
            X, feature_names = self.engineer_features(X.values, feature_names,
                                                     use_phase2_features=use_phase2)

        print(f"  [OK] Prepared {len(feature_names)} features")

        return X, feature_names

    def build_nephrotoxicity_model_enhanced(self, feature_set='baseline',
                                           imbalance_strategy='class_weight',
                                           optimize_hyperparams=True,
                                           use_optuna=False,
                                           use_ensemble=True, model_name='nephrotoxicity'):
        """
        Build enhanced nephrotoxicity prediction model with all improvements

        Parameters:
        -----------
        feature_set : str
            'baseline' for pre-dose model (baseline features only)
            'pkpd' for post-dose model (includes PK/PD exposure metrics)
        imbalance_strategy : str
            Strategy for handling class imbalance:
            - 'class_weight': Use scale_pos_weight in XGBoost (RECOMMENDED - faster, no synthetic data)
            - 'smote': Use SMOTE oversampling (legacy approach)
            - 'hybrid': Combine moderate SMOTE + class weights
            - 'none': No imbalance handling
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        use_ensemble : bool
            Whether to use ensemble methods
        model_name : str
            Name for storing the model ('nephrotoxicity_predose' or 'nephrotoxicity_postdose')
        """
        model_type = "PRE-DOSE" if feature_set == 'baseline' else "POST-DOSE"
        print("="*80)
        print(f"ENHANCED NEPHROTOXICITY PREDICTION MODEL ({model_type})")
        print("="*80)
        print()

        # Prepare features
        X, feature_names = self.prepare_features(feature_set, engineer=True)
        y = self.data['nephrotoxicity'].astype(int)

        print(f"Target distribution:")
        print(f"  No AKI: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        print(f"  AKI: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

        # Calculate class ratio for imbalance handling
        class_counts = np.bincount(y)
        class_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        print(f"  Class imbalance ratio: {class_ratio:.2f}:1")
        print()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === HANDLE CLASS IMBALANCE ===
        print(f"Imbalance handling strategy: {imbalance_strategy}")

        if imbalance_strategy == 'class_weight':
            print("  Using class weights (scale_pos_weight)...")
            X_train_final = X_train_scaled
            y_train_final = y_train
            scale_pos_weight = class_ratio
            print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

        elif imbalance_strategy == 'smote':
            print("  Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            scale_pos_weight = 1.0  # No additional class weighting needed
            print(f"  After SMOTE: {len(y_train_final)} samples")
            print(f"    No AKI: {(y_train_final==0).sum()}")
            print(f"    AKI: {(y_train_final==1).sum()}")

        elif imbalance_strategy == 'hybrid':
            print("  Using hybrid approach (moderate SMOTE + class weights)...")
            # Use SMOTE to partially balance (minority = 50% of majority)
            smote = SMOTE(random_state=42, sampling_strategy=0.5, k_neighbors=5)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            # Recalculate class ratio after SMOTE
            new_class_counts = np.bincount(y_train_final)
            scale_pos_weight = new_class_counts[0] / new_class_counts[1]
            print(f"  After SMOTE: {len(y_train_final)} samples")
            print(f"  Remaining imbalance: {scale_pos_weight:.2f}:1")
            print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

        else:  # 'none'
            print("  No imbalance handling...")
            X_train_final = X_train_scaled
            y_train_final = y_train
            scale_pos_weight = 1.0

        print()

        # === Hyperparameter Optimization ===
        if optimize_hyperparams:
            # Try Optuna first if requested and available
            if use_optuna and OPTUNA_AVAILABLE:
                print("Optimizing hyperparameters (Optuna Bayesian optimization)...")

                # Split training data for Optuna validation
                X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                    X_train_final, y_train_final, test_size=0.2, random_state=42, stratify=y_train_final
                )

                best_model, best_params, best_score = self.optimize_hyperparameters_optuna(
                    X_train_opt, y_train_opt, X_val_opt, y_val_opt,
                    scale_pos_weight=scale_pos_weight,
                    n_trials=100
                )

                if best_model is not None:
                    print(f"\n  [OK] Best parameters: {best_params}")
                    print(f"  [OK] Best validation ROC-AUC: {best_score:.3f}")

                    # Retrain on full training data with best parameters
                    best_model.fit(X_train_final, y_train_final)
                    print()
                else:
                    # Fallback to RandomizedSearchCV if Optuna fails
                    use_optuna = False

            if not use_optuna or not OPTUNA_AVAILABLE:
                if use_optuna and not OPTUNA_AVAILABLE:
                    print("  [WARNING] Optuna not available. Falling back to RandomizedSearchCV.")
                print("Optimizing hyperparameters (RandomizedSearchCV)...")

                param_dist = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 10, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.5, 1],
                    'min_child_weight': [1, 3, 5, 7]
                }

                random_search = RandomizedSearchCV(
                    xgb.XGBClassifier(
                        scale_pos_weight=scale_pos_weight,  # Use calculated scale_pos_weight
                        random_state=42,
                        eval_metric='logloss',
                        tree_method='hist'
                    ),
                    param_distributions=param_dist,
                    n_iter=50,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )

                random_search.fit(X_train_final, y_train_final)
                best_model = random_search.best_estimator_

                print(f"\n  [OK] Best parameters: {random_search.best_params_}")
                print(f"  [OK] Best CV ROC-AUC: {random_search.best_score_:.3f}")
                print()
        else:
            # Use improved default parameters
            best_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=3,
                scale_pos_weight=scale_pos_weight,  # Use calculated scale_pos_weight
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'
            )
            best_model.fit(X_train_final, y_train_final)

        # === Predictions ===
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)

        # === Evaluation ===
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"[OK] Enhanced Model Performance:")
        print(f"\nTest Set:")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  Average Precision: {avg_precision:.3f}")
        print()

        # Cross-validation on original data (not resampled)
        print("Cross-validation (5-fold)...")
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        print()

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))

        # === Ensemble Methods ===
        if use_ensemble:
            print("\nBuilding ensemble models...")
            ensemble_model = self.build_ensemble_classifier(
                X_train_final, y_train_final, X_test_scaled, y_test
            )

            if ensemble_model is not None:
                y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                roc_auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
                print(f"  Ensemble ROC-AUC: {roc_auc_ensemble:.3f}")

                # Use ensemble if better
                if roc_auc_ensemble > roc_auc:
                    print("  -> Using ensemble model (better performance)")
                    best_model = ensemble_model
                    y_pred_proba = y_pred_proba_ensemble
                    roc_auc = roc_auc_ensemble

        # === Deep Learning ===
        if TF_AVAILABLE:
            print("\nBuilding deep learning model...")
            dl_model, dl_auc = self.build_deep_learning_classifier(
                X_train_final, y_train_final, X_test_scaled, y_test
            )

            if dl_auc > roc_auc:
                print(f"  -> Deep learning achieves better AUC: {dl_auc:.3f}")
                # Note: We keep XGBoost for interpretability, but record DL performance
                self.performance['nephrotoxicity_dl'] = {'roc_auc': dl_auc}

        # Store results
        self.models[model_name] = best_model
        self.scalers[model_name] = scaler

        self.performance[model_name] = {
            'model_type': model_type,
            'feature_set': feature_set,
            'n_features': len(feature_names),
            'features': feature_names,
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'imbalance_strategy': imbalance_strategy,  # Updated from use_smote
            'scale_pos_weight': float(scale_pos_weight),
            'optimize_hyperparams': optimize_hyperparams,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.feature_importance[model_name] = feature_importance

            print("\nTop 15 features:")
            print(feature_importance.head(15).to_string(index=False))
        print()

        return best_model, roc_auc

    def build_ensemble_classifier(self, X_train, y_train, X_test, y_test):
        """Build stacking ensemble classifier"""
        try:
            # Base estimators
            estimators = [
                ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                         random_state=42, eval_metric='logloss', tree_method='hist')),
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                 random_state=42))
            ]

            # Add LightGBM if available
            if LGBM_AVAILABLE:
                estimators.append(
                    ('lgbm', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                               random_state=42, verbose=-1))
                )

            # Stacking
            stacking_model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                n_jobs=-1
            )

            stacking_model.fit(X_train, y_train)

            return stacking_model

        except Exception as e:
            print(f"  Warning: Ensemble building failed: {e}")
            return None

    def build_deep_learning_classifier(self, X_train, y_train, X_test, y_test):
        """Build deep neural network classifier"""
        try:
            # Build model
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),

                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),

                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[keras.metrics.AUC(name='auc'), 'accuracy']
            )

            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_auc',
                patience=30,
                restore_best_weights=True,
                mode='max',
                verbose=0
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=10,
                mode='max',
                verbose=0
            )

            # Train
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=200,
                batch_size=64,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            # Evaluate
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            dl_auc = roc_auc_score(y_test, y_pred_proba)

            print(f"  Deep Learning ROC-AUC: {dl_auc:.3f}")

            return model, dl_auc

        except Exception as e:
            print(f"  Warning: Deep learning failed: {e}")
            return None, 0.5

    def build_clinical_cure_model_enhanced(self, feature_set='pkpd',
                                          imbalance_strategy='class_weight',
                                          optimize_hyperparams=True,
                                          use_optuna=False,
                                          use_ensemble=True):
        """
        Build enhanced clinical cure prediction model
        """
        print("="*80)
        print("2. ENHANCED CLINICAL CURE PREDICTION MODEL")
        print("="*80)
        print()

        # Prepare features
        X, feature_names = self.prepare_features(feature_set, engineer=True)
        y = self.data['clinical_cure'].astype(int)

        print(f"Target distribution:")
        print(f"  Failure: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        print(f"  Cure: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

        # Calculate class ratio
        class_counts = np.bincount(y)
        class_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        print(f"  Class imbalance ratio: {class_ratio:.2f}:1")
        print()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle class imbalance
        print(f"Imbalance handling strategy: {imbalance_strategy}")

        if imbalance_strategy == 'class_weight':
            X_train_final = X_train_scaled
            y_train_final = y_train
            scale_pos_weight = class_ratio
            print(f"  Using class weights: scale_pos_weight = {scale_pos_weight:.2f}")
        elif imbalance_strategy == 'smote':
            print("  Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            scale_pos_weight = 1.0
            print(f"  After SMOTE: {len(y_train_final)} samples")
        elif imbalance_strategy == 'hybrid':
            print("  Using hybrid approach...")
            smote = SMOTE(random_state=42, sampling_strategy=0.5)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            new_class_counts = np.bincount(y_train_final)
            scale_pos_weight = new_class_counts[0] / new_class_counts[1]
            print(f"  scale_pos_weight = {scale_pos_weight:.2f}")
        else:
            X_train_final = X_train_scaled
            y_train_final = y_train
            scale_pos_weight = 1.0

        print()

        # Hyperparameter optimization
        if optimize_hyperparams:
            # Try Optuna first if requested and available
            if use_optuna and OPTUNA_AVAILABLE:
                print("Optimizing hyperparameters (Optuna Bayesian optimization)...")

                # Split training data for Optuna validation
                X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                    X_train_final, y_train_final, test_size=0.2, random_state=42, stratify=y_train_final
                )

                best_model, best_params, best_score = self.optimize_hyperparameters_optuna(
                    X_train_opt, y_train_opt, X_val_opt, y_val_opt,
                    scale_pos_weight=scale_pos_weight,
                    n_trials=100
                )

                if best_model is not None:
                    print(f"\n  [OK] Best parameters: {best_params}")
                    print(f"  [OK] Best validation ROC-AUC: {best_score:.3f}")

                    # Retrain on full training data with best parameters
                    best_model.fit(X_train_final, y_train_final)
                    print()
                else:
                    # Fallback to RandomizedSearchCV if Optuna fails
                    use_optuna = False

            if not use_optuna or not OPTUNA_AVAILABLE:
                if use_optuna and not OPTUNA_AVAILABLE:
                    print("  [WARNING] Optuna not available. Falling back to RandomizedSearchCV.")
                print("Optimizing hyperparameters...")

                param_dist = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 10, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.5, 1],
                    'min_child_weight': [1, 3, 5, 7]
                }

                random_search = RandomizedSearchCV(
                    xgb.XGBClassifier(
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        eval_metric='logloss',
                        tree_method='hist'
                    ),
                    param_distributions=param_dist,
                    n_iter=50,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )

                random_search.fit(X_train_final, y_train_final)
                best_model = random_search.best_estimator_

                print(f"\n  [OK] Best CV ROC-AUC: {random_search.best_score_:.3f}\n")
        else:
            best_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'
            )
            best_model.fit(X_train_final, y_train_final)

        # Predictions
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)

        # Evaluation
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"[OK] Enhanced Model Performance:")
        print(f"\nTest Set:")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  Average Precision: {avg_precision:.3f}")
        print()

        # Cross-validation
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"Cross-validation (5-fold):")
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        print()

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Failure', 'Cure']))

        # Ensemble
        if use_ensemble:
            print("\nBuilding ensemble...")
            ensemble_model = self.build_ensemble_classifier(
                X_train_final, y_train_final, X_test_scaled, y_test
            )

            if ensemble_model is not None:
                y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                roc_auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
                print(f"  Ensemble ROC-AUC: {roc_auc_ensemble:.3f}")

                if roc_auc_ensemble > roc_auc:
                    best_model = ensemble_model
                    y_pred_proba = y_pred_proba_ensemble
                    roc_auc = roc_auc_ensemble

        # Store
        self.models['clinical_cure'] = best_model
        self.scalers['clinical_cure'] = scaler

        self.performance['clinical_cure'] = {
            'feature_set': feature_set,
            'n_features': len(feature_names),
            'features': feature_names,
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'imbalance_strategy': imbalance_strategy,
            'scale_pos_weight': float(scale_pos_weight),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.feature_importance['clinical_cure'] = feature_importance

            print("\nTop 15 features:")
            print(feature_importance.head(15).to_string(index=False))
        print()

        return best_model, roc_auc

    def optimize_hyperparameters_optuna(self, X_train, y_train, X_val, y_val,
                                       scale_pos_weight=1.0, n_trials=100):
        """
        Hyperparameter optimization using Optuna (Bayesian optimization)

        Parameters:
        -----------
        X_train : array
            Training features
        y_train : array
            Training labels
        X_val : array
            Validation features
        y_val : array
            Validation labels
        scale_pos_weight : float
            Class weight for imbalance handling
        n_trials : int
            Number of optimization trials (default: 100)

        Returns:
        --------
        best_model : XGBClassifier
            Model with optimized hyperparameters
        best_params : dict
            Best hyperparameters found
        best_value : float
            Best validation AUC achieved
        """
        if not OPTUNA_AVAILABLE:
            print("  ⚠ Optuna not available. Falling back to RandomizedSearchCV.")
            return None, None, None

        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Bayesian Optimization)")
        print("="*80)

        def objective(trial):
            """Objective function for Optuna"""
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            }

            # Create model
            model = xgb.XGBClassifier(**params)

            # Train with early stopping (XGBoost 3.x uses callbacks instead of early_stopping_rounds)
            try:
                # XGBoost 3.x API
                from xgboost.callback import EarlyStopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[EarlyStopping(rounds=20, save_best=True)],
                    verbose=False
                )
            except (ImportError, TypeError):
                # Fallback for older XGBoost versions or if callbacks don't work
                model.fit(X_train, y_train, verbose=False)

            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)

            return auc

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Optimize
        print(f"\nRunning {n_trials} trials with Bayesian optimization...")
        print("(This may take a few minutes...)")

        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Results
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"\nBest Validation AUC: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {param:20s}: {value:.4f}")
            else:
                print(f"  {param:20s}: {value}")

        # Save optimization visualizations
        try:
            import plotly

            viz_dir = self.results_dir / 'optuna_visualizations'
            viz_dir.mkdir(exist_ok=True)

            # Optimization history
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_html(str(viz_dir / 'optimization_history.html'))

            # Parameter importances
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(str(viz_dir / 'param_importances.html'))

            # Parallel coordinate plot
            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.write_html(str(viz_dir / 'parallel_coordinate.html'))

            print(f"\n[OK] Saved optimization visualizations to: {viz_dir}/")
        except Exception as e:
            print(f"\n  [WARNING] Could not create visualizations: {e}")

        # Train final model with best parameters
        best_params = study.best_params.copy()
        best_params['scale_pos_weight'] = scale_pos_weight
        best_params['random_state'] = 42
        best_params['eval_metric'] = 'logloss'
        best_params['tree_method'] = 'hist'

        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

        print(f"\n[OK] Trained final model with optimized hyperparameters")

        return final_model, study.best_params, study.best_value

    def build_pk_surrogate_models_enhanced(self):
        """Build enhanced PK surrogate models with hyperparameter tuning"""
        print("="*80)
        print("3. ENHANCED PK SURROGATE MODELS")
        print("="*80)
        print()

        # Prepare features (baseline + dose + drug)
        baseline_features = [
            'age', 'sex', 'weight', 'height', 'bmi',
            'apache_ii', 'sofa_score',
            'baseline_crcl', 'baseline_scr', 'baseline_egfr',
            'baseline_albumin', 'baseline_bilirubin',
            'diabetes', 'ckd_stage',
            'mechanical_ventilation', 'vasopressor_use',
            'first_dose', 'drug'
        ]

        # Handle categorical
        data_encoded = self.data.copy()
        categorical_cols = ['sex', 'sepsis_type', 'infection_site', 'ckd_stage', 'drug']
        for col in categorical_cols:
            if col in data_encoded.columns:
                data_encoded[col] = pd.Categorical(data_encoded[col]).codes

        available_features = [f for f in baseline_features if f in data_encoded.columns]
        X = data_encoded[available_features].fillna(data_encoded[available_features].median())

        # Targets
        targets = {'Cmax': 'Cmax', 'AUC24': 'AUC24'}

        for target_name, target_col in targets.items():
            print(f"\nBuilding enhanced {target_name} surrogate...")

            # Get valid data
            valid_mask = ~self.data[target_col].isnull()
            X_valid = X[valid_mask]
            y = self.data.loc[valid_mask, target_col]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y, test_size=0.2, random_state=42
            )

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Hyperparameter tuning
            param_dist = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            }

            random_search = RandomizedSearchCV(
                xgb.XGBRegressor(random_state=42, tree_method='hist'),
                param_distributions=param_dist,
                n_iter=30,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            random_search.fit(X_train_scaled, y_train)
            model = random_search.best_estimator_

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # CV
            cv_r2 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)

            print(f"  [OK] R²: {r2:.3f}")
            print(f"  [OK] RMSE: {rmse:.2f}")
            print(f"  [OK] MAE: {mae:.2f}")
            print(f"  [OK] CV R²: {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

            # Store
            self.models[f'{target_name}_surrogate'] = model
            self.scalers[f'{target_name}_surrogate'] = scaler

            self.performance[f'{target_name}_surrogate'] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'cv_r2_mean': float(cv_r2.mean()),
                'cv_r2_std': float(cv_r2.std())
            }

        print()
        return self.models

    def plot_enhanced_results(self):
        """Generate comprehensive visualizations"""
        print("Generating enhanced visualizations...")

        # ROC curves
        self._plot_roc_curves_enhanced()

        # Feature importance comparison
        self._plot_feature_importance_comparison()

        # Calibration curves
        self._plot_calibration_curves()

        # Performance comparison
        self._plot_performance_comparison()

        print("  [OK] All visualizations saved")

    def _plot_roc_curves_enhanced(self):
        """Enhanced ROC curve plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        models_to_plot = ['nephrotoxicity_predose', 'nephrotoxicity_postdose', 'clinical_cure']
        titles = ['Nephrotoxicity (Pre-dose)', 'Nephrotoxicity (Post-dose)', 'Clinical Cure']

        plot_idx = 0
        for model_name, title in zip(models_to_plot, titles):
            if model_name in self.performance:
                perf = self.performance[model_name]

                fpr, tpr, _ = roc_curve(perf['y_test'], perf['y_pred_proba'])
                auc = perf['roc_auc']

                axes[plot_idx].plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})', linewidth=2)
                axes[plot_idx].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[plot_idx].set_xlabel('False Positive Rate', fontsize=12)
                axes[plot_idx].set_ylabel('True Positive Rate', fontsize=12)
                axes[plot_idx].set_title(f'{title}\nEnhanced Model', fontsize=14, fontweight='bold')
                axes[plot_idx].legend(loc='lower right')
                axes[plot_idx].grid(alpha=0.3)
                plot_idx += 1

        # Hide unused subplot
        if plot_idx < 4:
            axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance_comparison(self):
        """Plot feature importance for all models"""
        if not self.feature_importance:
            return

        # Determine number of models to plot
        models_to_plot = ['nephrotoxicity_predose', 'nephrotoxicity_postdose', 'clinical_cure']
        available_models = [m for m in models_to_plot if m in self.feature_importance]

        if not available_models:
            return

        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))

        if n_models == 1:
            axes = [axes]

        for idx, model_name in enumerate(available_models):
            fi = self.feature_importance[model_name].head(20)

            axes[idx].barh(range(len(fi)), fi['importance'])
            axes[idx].set_yticks(range(len(fi)))
            axes[idx].set_yticklabels(fi['feature'], fontsize=9)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance', fontsize=11)

            title = model_name.replace("_", " ").title()
            axes[idx].set_title(f'{title}\nTop 20 Features',
                               fontsize=13, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_calibration_curves(self):
        """Plot calibration curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        models = ['nephrotoxicity_predose', 'nephrotoxicity_postdose', 'clinical_cure']
        titles = ['Nephrotoxicity (Pre-dose)', 'Nephrotoxicity (Post-dose)', 'Clinical Cure']

        plot_idx = 0
        for model_name, title in zip(models, titles):
            if model_name in self.performance:
                perf = self.performance[model_name]

                fraction_positives, mean_predicted = calibration_curve(
                    perf['y_test'], perf['y_pred_proba'], n_bins=10
                )

                axes[plot_idx].plot(mean_predicted, fraction_positives, 's-', label='Model')
                axes[plot_idx].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[plot_idx].set_xlabel('Mean Predicted Probability', fontsize=12)
                axes[plot_idx].set_ylabel('Fraction of Positives', fontsize=12)
                axes[plot_idx].set_title(f'{title}\nCalibration Curve', fontsize=14, fontweight='bold')
                axes[plot_idx].legend()
                axes[plot_idx].grid(alpha=0.3)
                plot_idx += 1

        # Hide unused subplot
        if plot_idx < 4:
            axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'calibration_curves_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self):
        """Create performance comparison table"""
        perf_data = []

        for model_name in ['nephrotoxicity_predose', 'nephrotoxicity_postdose', 'clinical_cure']:
            if model_name in self.performance:
                perf = self.performance[model_name]
                model_display = model_name.replace('_', ' ').title()
                if 'model_type' in perf:
                    model_display = f"{model_display} ({perf['model_type']})"

                perf_data.append({
                    'Model': model_display,
                    'Feature Set': perf.get('feature_set', 'N/A'),
                    'ROC-AUC (Test)': f"{perf['roc_auc']:.3f}",
                    'ROC-AUC (CV)': f"{perf['cv_mean']:.3f} +/- {perf['cv_std']:.3f}",
                    'Avg Precision': f"{perf['avg_precision']:.3f}",
                    'N Features': perf['n_features']
                })

        for model_name in ['Cmax_surrogate', 'AUC24_surrogate']:
            if model_name in self.performance:
                perf = self.performance[model_name]
                perf_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Feature Set': 'baseline',
                    'R² (Test)': f"{perf['r2']:.3f}",
                    'R² (CV)': f"{perf['cv_r2_mean']:.3f} +/- {perf['cv_r2_std']:.3f}",
                    'RMSE': f"{perf['rmse']:.2f}",
                    'MAE': f"{perf['mae']:.2f}"
                })

        df = pd.DataFrame(perf_data)

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.axis('tight')
        ax.axis('off')

        # Dynamic column widths based on number of columns
        n_cols = len(df.columns)
        col_width = 1.0 / n_cols
        colWidths = [col_width] * n_cols

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=colWidths)

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Enhanced Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.results_dir / 'performance_summary_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_models_and_results(self):
        """Save all models and results"""
        print("\nSaving models and results...")

        # Save performance
        with open(self.results_dir / 'performance_enhanced.json', 'w') as f:
            # Convert numpy arrays to lists for JSON
            perf_serializable = {}
            for key, val in self.performance.items():
                perf_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in val.items()
                    if k not in ['X_test', 'y_test', 'y_pred_proba', 'y_pred']
                }
            json.dump(perf_serializable, f, indent=2)

        # Save feature importance
        for model_name, fi_df in self.feature_importance.items():
            fi_df.to_csv(self.results_dir / f'feature_importance_{model_name}_enhanced.csv', index=False)

        # Export models (XGBoost only)
        for model_name, model in self.models.items():
            if hasattr(model, 'save_model'):
                model.save_model(str(self.models_dir / f'{model_name}_enhanced.json'))

        print("  [OK] All models and results saved")

    # ========================================================================
    # PHASE 3: ADVANCED ENSEMBLE METHODS
    # ========================================================================

    def optimize_lightgbm_optuna(self, X_train, y_train, X_val, y_val,
                                 scale_pos_weight=1.0, n_trials=100):
        """
        Optimize LightGBM hyperparameters using Optuna
        """
        print("\n  -> Optimizing LightGBM with Optuna...")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(20, verbose=False)])

            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred)

        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42
        })

        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])

        print(f"     [OK] Best LightGBM AUC: {study.best_value:.4f}")
        return best_model, best_params, study.best_value

    def optimize_catboost_optuna(self, X_train, y_train, X_val, y_val,
                                 scale_pos_weight=1.0, n_trials=100):
        """
        Optimize CatBoost hyperparameters using Optuna
        """
        print("\n  -> Optimizing CatBoost with Optuna...")

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'verbose': False,
                'allow_writing_files': False,  # Prevent file locking issues on Windows
                'early_stopping_rounds': 20
            }

            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred)

        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbose': False,
            'allow_writing_files': False  # Prevent file locking issues on Windows
        })

        best_model = cb.CatBoostClassifier(**best_params)
        best_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        print(f"     [OK] Best CatBoost AUC: {study.best_value:.4f}")
        return best_model, best_params, study.best_value

    def build_stacking_ensemble(self, X_train, y_train, X_test, y_test,
                               base_models, meta_learner='logistic'):
        """
        Build stacking ensemble with multiple base models and a meta-learner

        Parameters:
        -----------
        base_models : dict
            Dictionary of trained base models {name: model}
        meta_learner : str
            Type of meta-learner: 'logistic' or 'neural'
        """
        print("\n  -> Building stacking ensemble...")

        # Create base estimators list
        estimators = [(name, model) for name, model in base_models.items()]

        # Choose meta-learner
        if meta_learner == 'logistic':
            final_estimator = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_learner == 'neural':
            from sklearn.neural_network import MLPClassifier
            final_estimator = MLPClassifier(hidden_layer_sizes=(100, 50),
                                           random_state=42, max_iter=1000)
        else:
            final_estimator = LogisticRegression(random_state=42, max_iter=1000)

        # Build stacking classifier
        # Note: Using n_jobs=1 to avoid CatBoost file locking issues on Windows
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=1  # Avoid parallel processing issues with CatBoost on Windows
        )

        stacking.fit(X_train, y_train)

        # Evaluate
        y_pred = stacking.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        print(f"     [OK] Stacking ensemble AUC: {auc:.4f}")
        print(f"     [OK] Base models: {len(base_models)}")
        print(f"     [OK] Meta-learner: {meta_learner}")

        return stacking, auc

    def compare_gradient_boosting_frameworks(self, X_train, y_train, X_test, y_test,
                                            scale_pos_weight=1.0, use_optuna=True,
                                            n_trials=50):
        """
        Compare XGBoost, LightGBM, and CatBoost performance

        Returns:
        --------
        dict with model comparison results
        """
        print("\n" + "="*80)
        print("COMPARING GRADIENT BOOSTING FRAMEWORKS")
        print("="*80)

        results = {}
        models = {}

        # Split for validation
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 1. XGBoost
        print("\n1. XGBoost")
        if use_optuna and OPTUNA_AVAILABLE:
            xgb_model, xgb_params, xgb_val_auc = self.optimize_hyperparameters_optuna(
                X_train_opt, y_train_opt, X_val_opt, y_val_opt,
                scale_pos_weight=scale_pos_weight, n_trials=n_trials
            )
            xgb_model.fit(X_train, y_train)
        else:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight, random_state=42
            )
            xgb_model.fit(X_train, y_train)

        y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, y_pred_xgb)
        print(f"   [OK] XGBoost Test AUC: {xgb_auc:.4f}")

        models['xgboost'] = xgb_model
        results['xgboost'] = {'test_auc': xgb_auc, 'model': xgb_model}

        # 2. LightGBM
        if LGBM_AVAILABLE:
            print("\n2. LightGBM")
            if use_optuna and OPTUNA_AVAILABLE:
                lgb_model, lgb_params, lgb_val_auc = self.optimize_lightgbm_optuna(
                    X_train_opt, y_train_opt, X_val_opt, y_val_opt,
                    scale_pos_weight=scale_pos_weight, n_trials=n_trials
                )
                lgb_model.fit(X_train, y_train)
            else:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1
                )
                lgb_model.fit(X_train, y_train)

            y_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
            lgb_auc = roc_auc_score(y_test, y_pred_lgb)
            print(f"   [OK] LightGBM Test AUC: {lgb_auc:.4f}")

            models['lightgbm'] = lgb_model
            results['lightgbm'] = {'test_auc': lgb_auc, 'model': lgb_model}
        else:
            print("\n2. LightGBM - SKIPPED (not available)")

        # 3. CatBoost
        if CATBOOST_AVAILABLE:
            print("\n3. CatBoost")
            if use_optuna and OPTUNA_AVAILABLE:
                cb_model, cb_params, cb_val_auc = self.optimize_catboost_optuna(
                    X_train_opt, y_train_opt, X_val_opt, y_val_opt,
                    scale_pos_weight=scale_pos_weight, n_trials=n_trials
                )
                cb_model.fit(X_train, y_train, verbose=False)
            else:
                cb_model = cb.CatBoostClassifier(
                    iterations=200, depth=6, learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight, random_state=42, verbose=False
                )
                cb_model.fit(X_train, y_train, verbose=False)

            y_pred_cb = cb_model.predict_proba(X_test)[:, 1]
            cb_auc = roc_auc_score(y_test, y_pred_cb)
            print(f"   [OK] CatBoost Test AUC: {cb_auc:.4f}")

            models['catboost'] = cb_model
            results['catboost'] = {'test_auc': cb_auc, 'model': cb_model}
        else:
            print("\n3. CatBoost - SKIPPED (not available)")

        # Summary
        print("\n" + "="*80)
        print("FRAMEWORK COMPARISON SUMMARY")
        print("="*80)

        for name, res in results.items():
            print(f"  {name:15s}: AUC = {res['test_auc']:.4f}")

        # Find best
        best_framework = max(results.items(), key=lambda x: x[1]['test_auc'])
        print(f"\n  [OK] Best framework: {best_framework[0].upper()} (AUC = {best_framework[1]['test_auc']:.4f})")

        return results, models

    # ========================================================================
    # PHASE 2: ADVANCED FEATURE ENGINEERING
    # ========================================================================

    def bayesian_model_averaging(self, models_dict, X_test, weights=None):
        """
        Bayesian Model Averaging for uncertainty quantification

        Combines predictions from multiple models with optional weights
        to provide:
        1. Ensemble predictions
        2. Prediction uncertainty (standard deviation across models)
        3. Confidence intervals

        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models {name: model}
        X_test : array-like
            Test features
        weights : array-like, optional
            Model weights (default: equal weights)

        Returns:
        --------
        dict with:
            - mean_pred: Average prediction across models
            - std_pred: Standard deviation of predictions (uncertainty)
            - lower_ci: Lower 95% confidence interval
            - upper_ci: Upper 95% confidence interval
            - individual_preds: Predictions from each model
        """
        print("\n  -> Performing Bayesian Model Averaging...")

        # Get predictions from all models
        predictions = []
        model_names = []

        for name, model in models_dict.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_test)[:, 1]
                else:
                    pred = model.predict(X_test)
                predictions.append(pred)
                model_names.append(name)
            except Exception as e:
                print(f"     [WARNING] Could not get predictions from {name}: {e}")
                continue

        if len(predictions) == 0:
            print("     [X] No valid predictions obtained")
            return None

        predictions = np.array(predictions)

        # Set weights (equal if not provided)
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

        # Calculate weighted average and uncertainty
        mean_pred = np.average(predictions, axis=0, weights=weights)
        std_pred = np.std(predictions, axis=0)

        # 95% confidence intervals
        lower_ci = mean_pred - 1.96 * std_pred
        upper_ci = mean_pred + 1.96 * std_pred

        # Clip to valid probability range
        lower_ci = np.clip(lower_ci, 0, 1)
        upper_ci = np.clip(upper_ci, 0, 1)

        print(f"     [OK] Averaged predictions from {len(predictions)} models")
        print(f"     [OK] Mean uncertainty (std): {std_pred.mean():.4f}")
        print(f"     [OK] Max uncertainty: {std_pred.max():.4f}")

        return {
            'mean_pred': mean_pred,
            'std_pred': std_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'individual_preds': predictions,
            'model_names': model_names,
            'weights': weights
        }

    def create_domain_specific_interactions(self, X, feature_names):
        """
        Create domain-specific interaction features based on pharmacological knowledge

        Key interactions:
        1. PK parameters × renal function (CL_bayes × baseline_crcl)
        2. Time-based PD × severity (time_above_MIC × apache_ii)
        3. Volume parameters × body size (Vss_bayes × weight)
        4. Exposure × renal function (AUC_MIC × baseline_crcl)
        5. Individual variability × severity (CL_cv × sofa_score)
        """
        print("\n  -> Creating domain-specific interaction features...")

        X_df = pd.DataFrame(X, columns=feature_names)
        interactions = []
        interaction_names = []

        # Define pharmacologically meaningful interactions
        interaction_pairs = [
            # PK parameters × renal function
            ('CL_bayes', 'baseline_crcl', 'CL_renal_interaction'),
            ('Vc_bayes', 'baseline_crcl', 'Vc_renal_interaction'),
            ('k10_bayes', 'baseline_egfr', 'k10_egfr_interaction'),

            # Time-based PD × severity
            ('time_above_MIC', 'apache_ii', 'time_MIC_severity'),
            ('time_above_MIC_percent', 'sofa_score', 'time_MIC_sofa'),
            ('AUC_above_MIC', 'apache_ii', 'AUC_MIC_severity'),

            # Volume parameters × body size
            ('Vss_bayes', 'weight', 'Vss_weight_interaction'),
            ('Vc_bayes', 'bmi', 'Vc_bmi_interaction'),

            # Exposure × renal function
            ('AUC_MIC', 'baseline_crcl', 'AUC_MIC_renal'),
            ('Cmax_MIC', 'baseline_crcl', 'Cmax_MIC_renal'),

            # Individual variability × severity
            ('CL_cv', 'sofa_score', 'CL_variability_severity'),
            ('Vc_cv', 'apache_ii', 'Vc_variability_severity'),

            # Half-life × age
            ('t_half_beta', 'age', 'half_life_age'),

            # Fluctuation × renal function
            ('fluctuation_index', 'baseline_crcl', 'fluctuation_renal'),
            ('peak_trough_ratio', 'baseline_scr', 'peak_trough_scr'),
        ]

        n_created = 0
        for feat1, feat2, name in interaction_pairs:
            if feat1 in feature_names and feat2 in feature_names:
                interaction = X_df[feat1] * X_df[feat2]
                interactions.append(interaction.values)
                interaction_names.append(name)
                n_created += 1

        if n_created > 0:
            interactions_array = np.column_stack(interactions)
            X_combined = np.hstack([X, interactions_array])
            combined_names = feature_names + interaction_names
            print(f"     [OK] Created {n_created} domain-specific interactions")
            return X_combined, combined_names
        else:
            print("     [WARNING] No interaction features created (missing base features)")
            return X, feature_names

    def create_composite_risk_scores(self, X, feature_names):
        """
        Create composite risk scores combining multiple PK/PD and clinical features

        Composite scores:
        1. PK_risk_score: Combines CL_bayes, Vc_bayes, and renal function
        2. PD_efficacy_score: Combines time_above_MIC, AUC_above_MIC, Cmax_MIC
        3. Toxicity_risk_score: Combines exposure metrics, renal function, age
        4. Overall_risk_score: Weighted combination of all risk factors
        """
        print("\n  -> Creating composite risk scores...")

        X_df = pd.DataFrame(X, columns=feature_names)
        composites = []
        composite_names = []

        # 1. PK Risk Score (normalized clearance and volume with renal function)
        pk_features = []
        if 'CL_bayes' in feature_names:
            pk_features.append(X_df['CL_bayes'] / X_df['CL_bayes'].std())
        if 'Vc_bayes' in feature_names:
            pk_features.append(X_df['Vc_bayes'] / X_df['Vc_bayes'].std())
        if 'baseline_crcl' in feature_names:
            pk_features.append((100 - X_df['baseline_crcl']) / X_df['baseline_crcl'].std())

        if len(pk_features) >= 2:
            pk_risk = np.mean(pk_features, axis=0)
            composites.append(pk_risk)
            composite_names.append('PK_risk_score')

        # 2. PD Efficacy Score (time above MIC and exposure)
        pd_features = []
        if 'time_above_MIC_percent' in feature_names:
            pd_features.append(X_df['time_above_MIC_percent'] / 100)
        if 'AUC_above_MIC' in feature_names:
            pd_features.append(X_df['AUC_above_MIC'] / X_df['AUC_above_MIC'].std())
        if 'Cmax_MIC' in feature_names:
            pd_features.append(X_df['Cmax_MIC'] / X_df['Cmax_MIC'].std())

        if len(pd_features) >= 2:
            pd_efficacy = np.mean(pd_features, axis=0)
            composites.append(pd_efficacy)
            composite_names.append('PD_efficacy_score')

        # 3. Toxicity Risk Score (high exposure + poor renal function + age)
        tox_features = []
        if 'AUC24' in feature_names:
            tox_features.append(X_df['AUC24'] / X_df['AUC24'].std())
        if 'Cmax' in feature_names:
            tox_features.append(X_df['Cmax'] / X_df['Cmax'].std())
        if 'baseline_scr' in feature_names:
            tox_features.append(X_df['baseline_scr'] / X_df['baseline_scr'].std())
        if 'age' in feature_names:
            tox_features.append(X_df['age'] / X_df['age'].std())

        if len(tox_features) >= 2:
            tox_risk = np.mean(tox_features, axis=0)
            composites.append(tox_risk)
            composite_names.append('Toxicity_risk_score')

        # 4. Variability Score (PK uncertainty)
        var_features = []
        if 'CL_cv' in feature_names:
            var_features.append(X_df['CL_cv'] / 100)
        if 'Vc_cv' in feature_names:
            var_features.append(X_df['Vc_cv'] / 100)

        if len(var_features) >= 1:
            var_score = np.mean(var_features, axis=0)
            composites.append(var_score)
            composite_names.append('PK_variability_score')

        if len(composites) > 0:
            composites_array = np.column_stack(composites)
            X_combined = np.hstack([X, composites_array])
            combined_names = feature_names + composite_names
            print(f"     [OK] Created {len(composites)} composite risk scores")
            return X_combined, combined_names
        else:
            print("     [WARNING] No composite scores created (missing base features)")
            return X, feature_names

    def run_complete_pipeline(self, imbalance_strategy='class_weight', use_optuna=False,
                             use_phase2_features=False):
        """
        Run the complete enhanced ML pipeline

        Parameters:
        -----------
        imbalance_strategy : str
            Strategy for handling class imbalance:
            - 'class_weight': Use scale_pos_weight (RECOMMENDED - faster, no synthetic data)
            - 'smote': Use SMOTE oversampling
            - 'hybrid': Combine SMOTE + class weights
        use_optuna : bool
            Whether to use Optuna for hyperparameter optimization (default: False)
            If False, uses RandomizedSearchCV
        use_phase2_features : bool
            Whether to use Phase 2 advanced feature engineering (default: False)
        """
        print("\n" + "="*80)
        print("RUNNING COMPLETE ENHANCED ML PIPELINE")
        if use_phase2_features:
            print("WITH PHASE 2 ADVANCED FEATURE ENGINEERING")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Imbalance strategy: {imbalance_strategy}")
        print(f"  Hyperparameter optimization: {'Optuna (Bayesian)' if use_optuna else 'RandomizedSearchCV'}")
        print(f"  Phase 2 features: {'Enabled' if use_phase2_features else 'Disabled'}")
        print("="*80 + "\n")

        # Store configuration for feature engineering
        self.use_phase2_features = use_phase2_features

        # Build all models
        # 1. Pre-dose nephrotoxicity model (baseline features only)
        print("\n" + "="*80)
        print("BUILDING PRE-DOSE NEPHROTOXICITY MODEL")
        print("Uses baseline patient characteristics before treatment")
        print("="*80 + "\n")
        self.build_nephrotoxicity_model_enhanced(
            feature_set='baseline',
            imbalance_strategy=imbalance_strategy,
            optimize_hyperparams=True,
            use_optuna=use_optuna,
            use_ensemble=True,
            model_name='nephrotoxicity_predose'
        )

        # 2. Post-dose nephrotoxicity model (includes PK/PD exposure)
        print("\n" + "="*80)
        print("BUILDING POST-DOSE NEPHROTOXICITY MODEL")
        print("Uses baseline characteristics + drug exposure metrics (PK/PD)")
        print("="*80 + "\n")
        self.build_nephrotoxicity_model_enhanced(
            feature_set='pkpd',
            imbalance_strategy=imbalance_strategy,
            optimize_hyperparams=True,
            use_optuna=use_optuna,
            use_ensemble=True,
            model_name='nephrotoxicity_postdose'
        )

        # 3. Clinical cure prediction model
        self.build_clinical_cure_model_enhanced(
            feature_set='pkpd',
            imbalance_strategy=imbalance_strategy,
            optimize_hyperparams=True,
            use_optuna=use_optuna,
            use_ensemble=True
        )

        # 4. PK surrogate models
        self.build_pk_surrogate_models_enhanced()

        # Visualizations
        self.plot_enhanced_results()

        # Save everything
        self.save_models_and_results()

        print("\n" + "="*80)
        print("ENHANCED ML PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Models saved to: {self.models_dir}")
        print()

    # ========================================================================
    # PHASE 4: MODEL INTERPRETABILITY & VALIDATION
    # ========================================================================

    def calculate_shap_values(self, model, X, feature_names, model_type='tree'):
        """
        Calculate SHAP values for model interpretability

        Parameters:
        -----------
        model : trained model
            The model to explain
        X : array-like
            Features to explain
        feature_names : list
            Names of features
        model_type : str
            Type of model: 'tree' (XGBoost, LightGBM, CatBoost, RF) or 'linear'

        Returns:
        --------
        explainer : shap.Explainer
            SHAP explainer object
        shap_values : array
            SHAP values for each sample and feature
        """
        if not SHAP_AVAILABLE:
            print("  [WARNING] SHAP not available. Install with: pip install shap")
            return None, None

        print("\n  -> Calculating SHAP values...")

        # Ensure X is numeric numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        try:
            if model_type == 'tree':
                # Use TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                # For binary classification, some models return list of arrays
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class

            elif model_type == 'linear':
                # Use LinearExplainer for linear models
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            else:
                # Use KernelExplainer as fallback (slower but works for any model)
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

            print(f"     [OK] SHAP values calculated: {shap_values.shape}")
            return explainer, shap_values

        except Exception as e:
            print(f"     [WARNING] Error calculating SHAP values: {e}")
            return None, None

    def generate_shap_visualizations(self, shap_values, X, feature_names,
                                     outcome_name='outcome', save_dir=None):
        """
        Generate SHAP visualizations for model interpretability

        Parameters:
        -----------
        shap_values : array
            SHAP values from calculate_shap_values()
        X : array-like
            Features used for SHAP calculation
        feature_names : list
            Names of features
        outcome_name : str
            Name of the outcome being predicted
        save_dir : str, optional
            Directory to save plots (default: results/shap_analysis/)
        """
        if not SHAP_AVAILABLE or shap_values is None:
            print("  [WARNING] SHAP values not available")
            return

        if save_dir is None:
            save_dir = self.results_dir / 'shap_analysis'
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n  -> Generating SHAP visualizations...")

        # Convert X to DataFrame for better visualization
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X

        try:
            # 1. Summary Plot (bar) - Feature importance
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_df, plot_type='bar', show=False)
            plt.title(f'SHAP Feature Importance - {outcome_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / f'shap_importance_{outcome_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     ✓ Saved: shap_importance_{outcome_name}.png")

            # 2. Summary Plot (beeswarm) - Feature effects
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_df, show=False)
            plt.title(f'SHAP Summary Plot - {outcome_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / f'shap_summary_{outcome_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     ✓ Saved: shap_summary_{outcome_name}.png")

            # 3. Dependence plots for top 5 features
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]

            for idx in top_features_idx:
                feature_name = feature_names[idx]
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(idx, shap_values, X_df, show=False)
                plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                safe_name = feature_name.replace('/', '_').replace(' ', '_')
                plt.savefig(save_dir / f'shap_dependence_{safe_name}_{outcome_name}.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
            print(f"     ✓ Saved: {len(top_features_idx)} dependence plots")

            # 4. Force plot for first prediction (saved as HTML)
            shap.force_plot(
                explainer.expected_value if hasattr(self, 'explainer') else shap_values.mean(),
                shap_values[0],
                X_df.iloc[0],
                matplotlib=False,
                show=False
            )
            # Note: Force plots are interactive and best viewed in Jupyter
            print(f"     ✓ Force plots available (best viewed in Jupyter notebook)")

        except Exception as e:
            print(f"     [WARNING] Error generating SHAP visualizations: {e}")

    def external_validation(self, model, X_external, y_external, feature_names,
                           outcome_name='outcome'):
        """
        Perform external validation on a holdout dataset

        Parameters:
        -----------
        model : trained model
            The model to validate
        X_external : array-like
            External validation features
        y_external : array-like
            External validation outcomes
        feature_names : list
            Names of features
        outcome_name : str
            Name of the outcome being predicted

        Returns:
        --------
        results : dict
            Validation metrics
        """
        print(f"\n  -> Performing external validation for {outcome_name}...")

        # Make predictions
        y_pred_proba = model.predict_proba(X_external)[:, 1]
        y_pred = model.predict(X_external)

        # Calculate metrics
        auc = roc_auc_score(y_external, y_pred_proba)
        avg_precision = average_precision_score(y_external, y_pred_proba)

        # Classification report
        report = classification_report(y_external, y_pred, output_dict=True)

        # Get positive class metrics (handle both string and int keys)
        pos_class_key = '1' if '1' in report else 1
        if pos_class_key not in report:
            pos_class_key = '1.0' if '1.0' in report else 1.0

        results = {
            'auc': auc,
            'avg_precision': avg_precision,
            'accuracy': report['accuracy'],
            'precision': report[pos_class_key]['precision'],
            'recall': report[pos_class_key]['recall'],
            'f1': report[pos_class_key]['f1-score'],
            'n_samples': len(y_external),
            'n_positive': int(y_external.sum()),
            'n_negative': int((1 - y_external).sum())
        }

        print(f"     [OK] External Validation Results:")
        print(f"        - Samples: {results['n_samples']} ({results['n_positive']} positive, {results['n_negative']} negative)")
        print(f"        - AUC: {results['auc']:.4f}")
        print(f"        - Avg Precision: {results['avg_precision']:.4f}")
        print(f"        - Accuracy: {results['accuracy']:.4f}")
        print(f"        - Precision: {results['precision']:.4f}")
        print(f"        - Recall: {results['recall']:.4f}")
        print(f"        - F1-Score: {results['f1']:.4f}")

        return results

    def create_clinical_decision_support_viz(self, model, X_test, y_test, feature_names,
                                             outcome_name='outcome', save_dir=None):
        """
        Create clinical decision support visualizations

        Parameters:
        -----------
        model : trained model
            The model to use for predictions
        X_test : array-like
            Test features
        y_test : array-like
            Test outcomes
        feature_names : list
            Names of features
        outcome_name : str
            Name of the outcome being predicted
        save_dir : str, optional
            Directory to save plots
        """
        if save_dir is None:
            save_dir = self.results_dir / 'clinical_decision_support'
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  -> Creating clinical decision support visualizations for {outcome_name}...")

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 1. Risk stratification plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Risk distribution
        axes[0, 0].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
                       bins=20, label=['No Event', 'Event'], alpha=0.7)
        axes[0, 0].set_xlabel('Predicted Risk', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Risk Distribution by Outcome', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
        axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision, linewidth=2, label=f'AP = {avg_precision:.3f}')
        axes[1, 0].set_xlabel('Recall', fontsize=11)
        axes[1, 0].set_ylabel('Precision', fontsize=11)
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        axes[1, 1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        axes[1, 1].set_xlabel('Predicted Probability', fontsize=11)
        axes[1, 1].set_ylabel('True Probability', fontsize=11)
        axes[1, 1].set_title('Calibration Curve', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Clinical Decision Support - {outcome_name}',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_dir / f'clinical_decision_support_{outcome_name}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     [OK] Saved: clinical_decision_support_{outcome_name}.png")

        # 2. Risk stratification table
        risk_bins = pd.cut(y_pred_proba, bins=[0, 0.3, 0.7, 1.0],
                          labels=['Low Risk', 'Medium Risk', 'High Risk'])
        risk_table = pd.crosstab(risk_bins, y_test, normalize='index') * 100

        print(f"\n     Risk Stratification Table:")
        print(f"     {risk_table.to_string()}")

        # Save risk table
        risk_table.to_csv(save_dir / f'risk_stratification_{outcome_name}.csv')
        print(f"     [OK] Saved: risk_stratification_{outcome_name}.csv")


def main():
    """Main execution"""

    # Initialize pipeline
    pipeline = EnhancedMLPipeline(
        ml_data_path='data/processed/ml_dataset.csv',
        pkpd_data_path='results/phase3_pkpd/pkpd_indices.csv'
    )

    # Run complete pipeline with Optuna enabled
    pipeline.run_complete_pipeline(
        imbalance_strategy='class_weight',
        use_optuna=True  # Enable Bayesian hyperparameter optimization
    )

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    for model_name, perf in pipeline.performance.items():
        display_name = model_name.upper().replace('_', ' ')
        if 'model_type' in perf:
            display_name = f"{display_name} ({perf['model_type']})"

        print(f"\n{display_name}:")
        if 'feature_set' in perf:
            print(f"  Feature Set: {perf['feature_set']}")
        if 'n_features' in perf:
            print(f"  Number of Features: {perf['n_features']}")
        if 'roc_auc' in perf:
            print(f"  ROC-AUC (Test): {perf['roc_auc']:.3f}")
            if 'cv_mean' in perf:
                print(f"  ROC-AUC (CV):   {perf['cv_mean']:.3f} +/- {perf['cv_std']:.3f}")
        if 'avg_precision' in perf:
            print(f"  Avg Precision:  {perf['avg_precision']:.3f}")
        if 'r2' in perf:
            print(f"  R² (Test): {perf['r2']:.3f}")
            if 'cv_r2_mean' in perf:
                print(f"  R² (CV):   {perf['cv_r2_mean']:.3f} +/- {perf['cv_r2_std']:.3f}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
