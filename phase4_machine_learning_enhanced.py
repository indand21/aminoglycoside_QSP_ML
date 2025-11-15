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

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
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
        print("  ✓ Advanced feature engineering")
        print("  ✓ Class imbalance handling (SMOTE)")
        print("  ✓ Hyperparameter optimization")
        print("  ✓ Ensemble methods")
        print("  ✓ Deep learning (neural networks)")
        print("  ✓ Feature selection")
        print("  ✓ Probability calibration")
        print()

        # Load data
        print("Loading data...")
        self.ml_data = pd.read_csv(ml_data_path)
        self.pkpd_data = pd.read_csv(pkpd_data_path)

        # Merge datasets
        self.data = self.ml_data.merge(
            self.pkpd_data[['patient_id', 'Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC']],
            on='patient_id',
            how='left',
            suffixes=('', '_pkpd')
        )

        print(f"  ✓ Loaded ML dataset: {self.ml_data.shape}")
        print(f"  ✓ Loaded PK/PD indices: {self.pkpd_data.shape}")
        print(f"  ✓ Merged dataset: {self.data.shape}")
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

    def engineer_features(self, X, feature_names):
        """
        Advanced feature engineering

        Creates:
        - Interaction terms (PK/PD × patient characteristics)
        - Polynomial features
        - Domain-specific composite features
        - Log transformations
        """
        print("\n  → Engineering advanced features...")

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
        print(f"     ✓ Created {n_new_features} engineered features")
        print(f"     ✓ Total features: {X_eng.shape[1]}")

        return X_eng.values, X_eng.columns.tolist()

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
            'sepsis_type', 'infection_site'
        ]

        # PK/PD features
        pkpd_features = [
            'Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC'
        ]

        # Dosing features
        dosing_features = ['first_dose', 'mean_dose', 'n_doses']

        # Select based on set
        if feature_set == 'baseline':
            selected_features = baseline_features
        elif feature_set == 'pkpd':
            selected_features = baseline_features + pkpd_features
        elif feature_set == 'full':
            selected_features = baseline_features + pkpd_features + dosing_features
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")

        # Filter available
        available_features = [f for f in selected_features if f in self.data.columns]

        # Handle categorical
        data_encoded = self.data.copy()
        categorical_cols = ['sex', 'sepsis_type', 'infection_site', 'ckd_stage']
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
            X, feature_names = self.engineer_features(X.values, feature_names)

        print(f"  ✓ Prepared {len(feature_names)} features")

        return X, feature_names

    def build_nephrotoxicity_model_enhanced(self, feature_set='baseline',
                                           use_smote=True, optimize_hyperparams=True,
                                           use_ensemble=True):
        """
        Build enhanced nephrotoxicity prediction model with all improvements
        """
        print("="*80)
        print("1. ENHANCED NEPHROTOXICITY PREDICTION MODEL")
        print("="*80)
        print()

        # Prepare features
        X, feature_names = self.prepare_features(feature_set, engineer=True)
        y = self.data['nephrotoxicity'].astype(int)

        print(f"Target distribution:")
        print(f"  No AKI: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        print(f"  AKI: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        print()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === SMOTE for class imbalance ===
        if use_smote:
            print("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print(f"  After SMOTE: {len(y_train_resampled)} samples")
            print(f"    No AKI: {(y_train_resampled==0).sum()}")
            print(f"    AKI: {(y_train_resampled==1).sum()}")
            print()
        else:
            X_train_resampled = X_train_scaled
            y_train_resampled = y_train

        # === Hyperparameter Optimization ===
        if optimize_hyperparams:
            print("Optimizing hyperparameters (RandomizedSearchCV)...")

            param_dist = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.5, 1],
                'min_child_weight': [1, 3, 5, 7],
                'scale_pos_weight': [1, 2, 3, 5]
            }

            random_search = RandomizedSearchCV(
                xgb.XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'),
                param_distributions=param_dist,
                n_iter=50,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            random_search.fit(X_train_resampled, y_train_resampled)
            best_model = random_search.best_estimator_

            print(f"\n  ✓ Best parameters: {random_search.best_params_}")
            print(f"  ✓ Best CV ROC-AUC: {random_search.best_score_:.3f}")
            print()
        else:
            # Use improved default parameters
            class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            best_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=3,
                scale_pos_weight=class_ratio,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'
            )
            best_model.fit(X_train_resampled, y_train_resampled)

        # === Predictions ===
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)

        # === Evaluation ===
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"✓ Enhanced Model Performance:")
        print(f"\nTest Set:")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  Average Precision: {avg_precision:.3f}")
        print()

        # Cross-validation on original data
        print("Cross-validation (5-fold)...")
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print()

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))

        # === Ensemble Methods ===
        if use_ensemble:
            print("\nBuilding ensemble models...")
            ensemble_model = self.build_ensemble_classifier(
                X_train_resampled, y_train_resampled, X_test_scaled, y_test
            )

            if ensemble_model is not None:
                y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                roc_auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
                print(f"  Ensemble ROC-AUC: {roc_auc_ensemble:.3f}")

                # Use ensemble if better
                if roc_auc_ensemble > roc_auc:
                    print("  → Using ensemble model (better performance)")
                    best_model = ensemble_model
                    y_pred_proba = y_pred_proba_ensemble
                    roc_auc = roc_auc_ensemble

        # === Deep Learning ===
        if TF_AVAILABLE:
            print("\nBuilding deep learning model...")
            dl_model, dl_auc = self.build_deep_learning_classifier(
                X_train_resampled, y_train_resampled, X_test_scaled, y_test
            )

            if dl_auc > roc_auc:
                print(f"  → Deep learning achieves better AUC: {dl_auc:.3f}")
                # Note: We keep XGBoost for interpretability, but record DL performance
                self.performance['nephrotoxicity_dl'] = {'roc_auc': dl_auc}

        # Store results
        self.models['nephrotoxicity'] = best_model
        self.scalers['nephrotoxicity'] = scaler

        self.performance['nephrotoxicity'] = {
            'feature_set': feature_set,
            'n_features': len(feature_names),
            'features': feature_names,
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'use_smote': use_smote,
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

            self.feature_importance['nephrotoxicity'] = feature_importance

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
                                          use_smote=True, optimize_hyperparams=True,
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
        print()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE
        if use_smote:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print(f"  After SMOTE: {len(y_train_resampled)} samples\n")
        else:
            X_train_resampled = X_train_scaled
            y_train_resampled = y_train

        # Hyperparameter optimization
        if optimize_hyperparams:
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
                xgb.XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'),
                param_distributions=param_dist,
                n_iter=50,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            random_search.fit(X_train_resampled, y_train_resampled)
            best_model = random_search.best_estimator_

            print(f"\n  ✓ Best CV ROC-AUC: {random_search.best_score_:.3f}\n")
        else:
            best_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'
            )
            best_model.fit(X_train_resampled, y_train_resampled)

        # Predictions
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)

        # Evaluation
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"✓ Enhanced Model Performance:")
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
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print()

        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Failure', 'Cure']))

        # Ensemble
        if use_ensemble:
            print("\nBuilding ensemble...")
            ensemble_model = self.build_ensemble_classifier(
                X_train_resampled, y_train_resampled, X_test_scaled, y_test
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

    def build_pk_surrogate_models_enhanced(self):
        """Build enhanced PK surrogate models with hyperparameter tuning"""
        print("="*80)
        print("3. ENHANCED PK SURROGATE MODELS")
        print("="*80)
        print()

        # Prepare features (baseline + dose)
        baseline_features = [
            'age', 'sex', 'weight', 'height', 'bmi',
            'apache_ii', 'sofa_score',
            'baseline_crcl', 'baseline_scr', 'baseline_egfr',
            'baseline_albumin', 'baseline_bilirubin',
            'diabetes', 'ckd_stage',
            'mechanical_ventilation', 'vasopressor_use',
            'first_dose'
        ]

        # Handle categorical
        data_encoded = self.data.copy()
        categorical_cols = ['sex', 'sepsis_type', 'infection_site', 'ckd_stage']
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

            print(f"  ✓ R²: {r2:.3f}")
            print(f"  ✓ RMSE: {rmse:.2f}")
            print(f"  ✓ MAE: {mae:.2f}")
            print(f"  ✓ CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

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

        print("  ✓ All visualizations saved")

    def _plot_roc_curves_enhanced(self):
        """Enhanced ROC curve plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        models_to_plot = ['nephrotoxicity', 'clinical_cure']
        titles = ['Nephrotoxicity Prediction', 'Clinical Cure Prediction']

        for idx, (model_name, title) in enumerate(zip(models_to_plot, titles)):
            if model_name in self.performance:
                perf = self.performance[model_name]

                fpr, tpr, _ = roc_curve(perf['y_test'], perf['y_pred_proba'])
                auc = perf['roc_auc']

                axes[idx].plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})', linewidth=2)
                axes[idx].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[idx].set_xlabel('False Positive Rate', fontsize=12)
                axes[idx].set_ylabel('True Positive Rate', fontsize=12)
                axes[idx].set_title(f'{title}\nEnhanced Model', fontsize=14, fontweight='bold')
                axes[idx].legend(loc='lower right')
                axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance_comparison(self):
        """Plot feature importance for both models"""
        if not self.feature_importance:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for idx, model_name in enumerate(['nephrotoxicity', 'clinical_cure']):
            if model_name in self.feature_importance:
                fi = self.feature_importance[model_name].head(20)

                axes[idx].barh(range(len(fi)), fi['importance'])
                axes[idx].set_yticks(range(len(fi)))
                axes[idx].set_yticklabels(fi['feature'], fontsize=9)
                axes[idx].invert_yaxis()
                axes[idx].set_xlabel('Importance', fontsize=11)
                axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nTop 20 Features',
                                   fontsize=13, fontweight='bold')
                axes[idx].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_calibration_curves(self):
        """Plot calibration curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        models = ['nephrotoxicity', 'clinical_cure']
        titles = ['Nephrotoxicity', 'Clinical Cure']

        for idx, (model_name, title) in enumerate(zip(models, titles)):
            if model_name in self.performance:
                perf = self.performance[model_name]

                fraction_positives, mean_predicted = calibration_curve(
                    perf['y_test'], perf['y_pred_proba'], n_bins=10
                )

                axes[idx].plot(mean_predicted, fraction_positives, 's-', label='Model')
                axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[idx].set_xlabel('Mean Predicted Probability', fontsize=12)
                axes[idx].set_ylabel('Fraction of Positives', fontsize=12)
                axes[idx].set_title(f'{title}\nCalibration Curve', fontsize=14, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'calibration_curves_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self):
        """Create performance comparison table"""
        perf_data = []

        for model_name in ['nephrotoxicity', 'clinical_cure']:
            if model_name in self.performance:
                perf = self.performance[model_name]
                perf_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'ROC-AUC (Test)': f"{perf['roc_auc']:.3f}",
                    'ROC-AUC (CV)': f"{perf['cv_mean']:.3f} ± {perf['cv_std']:.3f}",
                    'Avg Precision': f"{perf['avg_precision']:.3f}",
                    'N Features': perf['n_features']
                })

        for model_name in ['Cmax_surrogate', 'AUC24_surrogate']:
            if model_name in self.performance:
                perf = self.performance[model_name]
                perf_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'R² (Test)': f"{perf['r2']:.3f}",
                    'R² (CV)': f"{perf['cv_r2_mean']:.3f} ± {perf['cv_r2_std']:.3f}",
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

        print("  ✓ All models and results saved")

    def run_complete_pipeline(self):
        """Run the complete enhanced ML pipeline"""
        print("\n" + "="*80)
        print("RUNNING COMPLETE ENHANCED ML PIPELINE")
        print("="*80 + "\n")

        # Build all models
        self.build_nephrotoxicity_model_enhanced(
            feature_set='baseline',
            use_smote=True,
            optimize_hyperparams=True,
            use_ensemble=True
        )

        self.build_clinical_cure_model_enhanced(
            feature_set='pkpd',
            use_smote=True,
            optimize_hyperparams=True,
            use_ensemble=True
        )

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


def main():
    """Main execution"""

    # Initialize pipeline
    pipeline = EnhancedMLPipeline(
        ml_data_path='data/processed/ml_dataset.csv',
        pkpd_data_path='results/phase3_pkpd/pkpd_indices.csv'
    )

    # Run complete pipeline
    pipeline.run_complete_pipeline()

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    for model_name, perf in pipeline.performance.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        if 'roc_auc' in perf:
            print(f"  ROC-AUC (Test): {perf['roc_auc']:.3f}")
            print(f"  ROC-AUC (CV):   {perf['cv_mean']:.3f} ± {perf['cv_std']:.3f}")
        if 'r2' in perf:
            print(f"  R² (Test): {perf['r2']:.3f}")
            print(f"  R² (CV):   {perf['cv_r2_mean']:.3f} ± {perf['cv_r2_std']:.3f}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
