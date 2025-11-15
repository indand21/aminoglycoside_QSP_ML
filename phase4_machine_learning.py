#!/usr/bin/env python3
"""
Phase 4: Machine Learning for Outcome Prediction

This script:
1. Builds nephrotoxicity prediction model (XGBoost classification)
2. Builds clinical cure prediction model (XGBoost classification)
3. Builds PK parameter surrogate models (regression)
4. Performs feature importance analysis (SHAP)
5. Validates models with cross-validation
6. Generates comprehensive visualizations and reports

Author: Aminoglycoside QSP-ML Project
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    mean_absolute_error
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature importance will use built-in XGBoost importance.")

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLPipeline:
    """
    Machine Learning Pipeline for Aminoglycoside Outcome Prediction
    """

    def __init__(self,
                 ml_data_path='data/processed/ml_dataset.csv',
                 pkpd_data_path='results/phase3_pkpd/pkpd_indices.csv'):
        """
        Initialize ML Pipeline

        Parameters:
        -----------
        ml_data_path : str
            Path to ML dataset from Phase 1
        pkpd_data_path : str
            Path to PK/PD indices from Phase 3
        """
        print("="*80)
        print("Phase 4: Machine Learning for Outcome Prediction")
        print("="*80)
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
        self.results_dir = Path('results/phase4_ml')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.models = {}
        self.scalers = {}
        self.performance = {}
        self.feature_importance = {}

    def prepare_features(self, feature_set='baseline'):
        """
        Prepare feature sets for modeling

        Parameters:
        -----------
        feature_set : str
            'baseline' - baseline covariates only
            'pkpd' - baseline + PK/PD indices
            'full' - all available features
        """
        print(f"Preparing features (set: {feature_set})...")

        # Baseline features (available before treatment)
        baseline_features = [
            'age', 'sex', 'weight', 'height', 'bmi',
            'apache_ii', 'sofa_score',
            'baseline_crcl', 'baseline_scr', 'baseline_egfr',
            'baseline_albumin', 'baseline_bilirubin',
            'diabetes', 'ckd_stage',
            'mechanical_ventilation', 'vasopressor_use',
            'sepsis_type', 'infection_site'
        ]

        # PK/PD features (from Phase 3)
        pkpd_features = [
            'Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC'
        ]

        # Dosing features
        dosing_features = [
            'first_dose', 'mean_dose', 'n_doses'
        ]

        # Select features based on set
        if feature_set == 'baseline':
            selected_features = baseline_features
        elif feature_set == 'pkpd':
            selected_features = baseline_features + pkpd_features
        elif feature_set == 'full':
            selected_features = baseline_features + pkpd_features + dosing_features
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")

        # Filter to available columns
        available_features = [f for f in selected_features if f in self.data.columns]

        # Handle categorical variables
        data_encoded = self.data.copy()

        # Encode categorical variables
        categorical_cols = ['sex', 'sepsis_type', 'infection_site', 'ckd_stage']
        for col in categorical_cols:
            if col in available_features:
                data_encoded[col] = pd.Categorical(data_encoded[col]).codes

        # Get feature matrix
        X = data_encoded[available_features].copy()

        # Handle missing values (simple imputation with median)
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        print(f"  ✓ Selected {len(available_features)} features")
        print(f"  ✓ Feature matrix shape: {X.shape}")
        print()

        return X, available_features

    def build_nephrotoxicity_model(self, feature_set='baseline'):
        """
        Build nephrotoxicity prediction model (binary classification)

        Target: aki_stage > 0 (any AKI vs none)
        """
        print("="*80)
        print("1. NEPHROTOXICITY PREDICTION MODEL")
        print("="*80)
        print()

        # Prepare features and target
        X, feature_names = self.prepare_features(feature_set)
        y = (self.data['aki_stage'] > 0).astype(int)

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

        # Build XGBoost model
        print("Training XGBoost classifier...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"\n✓ Model trained successfully")
        print(f"\nTest Set Performance:")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  Average Precision: {avg_precision:.3f}")
        print()

        # Cross-validation
        print("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print()

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))

        # Store results
        self.models['nephrotoxicity'] = model
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
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_importance['nephrotoxicity'] = feature_importance

        print("Top 10 features:")
        print(feature_importance.head(10).to_string(index=False))
        print()

        return model, roc_auc

    def build_clinical_cure_model(self, feature_set='pkpd'):
        """
        Build clinical cure prediction model (binary classification)
        """
        print("="*80)
        print("2. CLINICAL CURE PREDICTION MODEL")
        print("="*80)
        print()

        # Prepare features and target
        X, feature_names = self.prepare_features(feature_set)
        y = self.data['clinical_cure'].astype(int)

        print(f"Target distribution:")
        print(f"  Failure: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        print(f"  Cure: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        print()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build XGBoost model
        print("Training XGBoost classifier...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"\n✓ Model trained successfully")
        print(f"\nTest Set Performance:")
        print(f"  ROC-AUC: {roc_auc:.3f}")
        print(f"  Average Precision: {avg_precision:.3f}")
        print()

        # Cross-validation
        print("Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print()

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Failure', 'Cure']))

        # Store results
        self.models['clinical_cure'] = model
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
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_importance['clinical_cure'] = feature_importance

        print("Top 10 features:")
        print(feature_importance.head(10).to_string(index=False))
        print()

        return model, roc_auc

    def build_pk_surrogate_models(self):
        """
        Build PK parameter surrogate models (regression)

        Predicts Cmax and AUC24 from baseline covariates and dose
        """
        print("="*80)
        print("3. PK PARAMETER SURROGATE MODELS")
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

        # Filter to available columns
        available_features = [f for f in baseline_features if f in self.data.columns]

        # Handle categorical variables
        data_encoded = self.data.copy()
        categorical_cols = ['sex', 'sepsis_type', 'infection_site', 'ckd_stage']
        for col in categorical_cols:
            if col in available_features:
                data_encoded[col] = pd.Categorical(data_encoded[col]).codes

        X = data_encoded[available_features].copy()

        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        # Build models for Cmax and AUC24
        targets = {
            'Cmax': 'Cmax',
            'AUC24': 'AUC24'
        }

        for target_name, target_col in targets.items():
            print(f"\nBuilding {target_name} surrogate model...")

            # Get target (remove NaNs)
            valid_mask = ~self.data[target_col].isnull()
            X_valid = X[valid_mask]
            y = self.data.loc[valid_mask, target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Build XGBoost regressor
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"  ✓ {target_name} model trained")
            print(f"    R²: {r2:.3f}")
            print(f"    RMSE: {rmse:.2f}")
            print(f"    MAE: {mae:.2f}")

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring='r2'
            )
            print(f"    CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Store results
            self.models[f'pk_{target_name.lower()}'] = model
            self.scalers[f'pk_{target_name.lower()}'] = scaler

            self.performance[f'pk_{target_name.lower()}'] = {
                'features': available_features,
                'n_features': len(available_features),
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'y_test': y_test.values,
                'y_pred': y_pred
            }

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.feature_importance[f'pk_{target_name.lower()}'] = feature_importance

        print()
        return

    def perform_shap_analysis(self, model_name='nephrotoxicity'):
        """
        Perform SHAP analysis for model interpretability
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP analysis.")
            return None

        print(f"\nPerforming SHAP analysis for {model_name} model...")

        model = self.models[model_name]
        X_test = self.performance[model_name]['X_test']

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        print(f"  ✓ SHAP values calculated")

        # Store SHAP results
        self.performance[model_name]['shap_values'] = shap_values
        self.performance[model_name]['shap_explainer'] = explainer

        return shap_values

    def generate_visualizations(self):
        """
        Generate comprehensive ML visualizations
        """
        print("="*80)
        print("Generating visualizations...")
        print("="*80)
        print()

        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.size'] = 10

        # 1. ROC Curves for classification models
        print("  Creating ROC curves...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        classification_models = ['nephrotoxicity', 'clinical_cure']
        titles = ['Nephrotoxicity Prediction', 'Clinical Cure Prediction']

        for idx, (model_name, title) in enumerate(zip(classification_models, titles)):
            perf = self.performance[model_name]

            # ROC curve
            fpr, tpr, _ = roc_curve(perf['y_test'], perf['y_pred_proba'])
            roc_auc = perf['roc_auc']

            axes[idx].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(title)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Precision-Recall Curves
        print("  Creating precision-recall curves...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (model_name, title) in enumerate(zip(classification_models, titles)):
            perf = self.performance[model_name]

            precision, recall, _ = precision_recall_curve(perf['y_test'], perf['y_pred_proba'])
            avg_precision = perf['avg_precision']

            axes[idx].plot(recall, precision, linewidth=2,
                          label=f'PR (AP = {avg_precision:.3f})')
            axes[idx].set_xlabel('Recall')
            axes[idx].set_ylabel('Precision')
            axes[idx].set_title(title)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Confusion Matrices
        print("  Creating confusion matrices...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (model_name, title) in enumerate(zip(classification_models, titles)):
            perf = self.performance[model_name]

            cm = confusion_matrix(perf['y_test'], perf['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(title)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Feature Importance
        print("  Creating feature importance plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        model_names = ['nephrotoxicity', 'clinical_cure', 'pk_cmax', 'pk_auc24']
        titles = ['Nephrotoxicity', 'Clinical Cure', 'Cmax Surrogate', 'AUC24 Surrogate']

        for idx, (model_name, title) in enumerate(zip(model_names, titles)):
            fi = self.feature_importance[model_name].head(10)

            axes[idx].barh(range(len(fi)), fi['importance'], alpha=0.8)
            axes[idx].set_yticks(range(len(fi)))
            axes[idx].set_yticklabels(fi['feature'])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{title} - Top 10 Features')
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. PK Surrogate Model Performance
        print("  Creating PK surrogate performance plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        pk_models = ['pk_cmax', 'pk_auc24']
        titles = ['Cmax Prediction', 'AUC24 Prediction']

        for idx, (model_name, title) in enumerate(zip(pk_models, titles)):
            perf = self.performance[model_name]

            y_test = perf['y_test']
            y_pred = perf['y_pred']
            r2 = perf['r2']

            axes[idx].scatter(y_test, y_pred, alpha=0.5, s=30)
            axes[idx].plot([y_test.min(), y_test.max()],
                          [y_test.min(), y_test.max()],
                          'r--', linewidth=2, label='Perfect prediction')
            axes[idx].set_xlabel('Observed')
            axes[idx].set_ylabel('Predicted')
            axes[idx].set_title(f'{title} (R² = {r2:.3f})')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'pk_surrogate_performance.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Calibration Curves
        print("  Creating calibration curves...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (model_name, title) in enumerate(zip(classification_models, titles)):
            perf = self.performance[model_name]

            prob_true, prob_pred = calibration_curve(
                perf['y_test'], perf['y_pred_proba'], n_bins=10
            )

            axes[idx].plot(prob_pred, prob_true, marker='o', linewidth=2,
                          label='Model')
            axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
            axes[idx].set_xlabel('Predicted Probability')
            axes[idx].set_ylabel('Observed Frequency')
            axes[idx].set_title(f'{title} - Calibration')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ All visualizations saved to: {self.results_dir}/")
        print()

    def save_models(self):
        """
        Save trained models
        """
        print("Saving models...")

        for model_name, model in self.models.items():
            model_path = self.models_dir / f'{model_name}_model.json'
            model.save_model(str(model_path))
            print(f"  ✓ Saved {model_name} model to {model_path}")

        print()

    def save_performance_summary(self):
        """
        Save performance summary to JSON
        """
        print("Saving performance summary...")

        # Prepare summary (exclude numpy arrays)
        summary = {}
        for model_name, perf in self.performance.items():
            summary[model_name] = {
                k: v for k, v in perf.items()
                if k not in ['X_test', 'y_test', 'y_pred', 'y_pred_proba',
                            'shap_values', 'shap_explainer']
            }

        # Save to JSON
        output_file = self.results_dir / 'model_performance.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Saved performance summary to {output_file}")
        print()

    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("Generating summary report...")

        report = []
        report.append("="*80)
        report.append("Phase 4: Machine Learning - Summary Report")
        report.append("="*80)
        report.append("")
        report.append(f"Analysis Date: 2025-11-15")
        report.append(f"Number of Patients: {len(self.data)}")
        report.append("")

        report.append("="*80)
        report.append("1. NEPHROTOXICITY PREDICTION MODEL")
        report.append("="*80)
        report.append("")

        neph_perf = self.performance['nephrotoxicity']
        report.append(f"Features: {neph_perf['feature_set']} ({neph_perf['n_features']} features)")
        report.append(f"ROC-AUC (test): {neph_perf['roc_auc']:.3f}")
        report.append(f"Average Precision: {neph_perf['avg_precision']:.3f}")
        report.append(f"CV ROC-AUC: {neph_perf['cv_mean']:.3f} ± {neph_perf['cv_std']:.3f}")
        report.append("")

        report.append("Top 5 Predictive Features:")
        for i, row in self.feature_importance['nephrotoxicity'].head(5).iterrows():
            report.append(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        report.append("")

        report.append("="*80)
        report.append("2. CLINICAL CURE PREDICTION MODEL")
        report.append("="*80)
        report.append("")

        cure_perf = self.performance['clinical_cure']
        report.append(f"Features: {cure_perf['feature_set']} ({cure_perf['n_features']} features)")
        report.append(f"ROC-AUC (test): {cure_perf['roc_auc']:.3f}")
        report.append(f"Average Precision: {cure_perf['avg_precision']:.3f}")
        report.append(f"CV ROC-AUC: {cure_perf['cv_mean']:.3f} ± {cure_perf['cv_std']:.3f}")
        report.append("")

        report.append("Top 5 Predictive Features:")
        for i, row in self.feature_importance['clinical_cure'].head(5).iterrows():
            report.append(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        report.append("")

        report.append("="*80)
        report.append("3. PK SURROGATE MODELS")
        report.append("="*80)
        report.append("")

        cmax_perf = self.performance['pk_cmax']
        auc_perf = self.performance['pk_auc24']

        report.append(f"Cmax Model:")
        report.append(f"  R²: {cmax_perf['r2']:.3f}")
        report.append(f"  RMSE: {cmax_perf['rmse']:.2f} mg/L")
        report.append(f"  MAE: {cmax_perf['mae']:.2f} mg/L")
        report.append(f"  CV R²: {cmax_perf['cv_mean']:.3f} ± {cmax_perf['cv_std']:.3f}")
        report.append("")

        report.append(f"AUC24 Model:")
        report.append(f"  R²: {auc_perf['r2']:.3f}")
        report.append(f"  RMSE: {auc_perf['rmse']:.2f} mg·h/L")
        report.append(f"  MAE: {auc_perf['mae']:.2f} mg·h/L")
        report.append(f"  CV R²: {auc_perf['cv_mean']:.3f} ± {auc_perf['cv_std']:.3f}")
        report.append("")

        report.append("="*80)
        report.append("4. KEY FINDINGS")
        report.append("="*80)
        report.append("")

        # Determine if models are good
        neph_auc = neph_perf['roc_auc']
        cure_auc = cure_perf['roc_auc']

        if neph_auc > 0.8:
            neph_quality = "Excellent"
        elif neph_auc > 0.7:
            neph_quality = "Good"
        else:
            neph_quality = "Moderate"

        if cure_auc > 0.8:
            cure_quality = "Excellent"
        elif cure_auc > 0.7:
            cure_quality = "Good"
        else:
            cure_quality = "Moderate"

        report.append(f"1. Nephrotoxicity prediction: {neph_quality} (AUC = {neph_auc:.3f})")
        report.append(f"2. Clinical cure prediction: {cure_quality} (AUC = {cure_auc:.3f})")
        report.append(f"3. PK surrogate models show {'good' if cmax_perf['r2'] > 0.5 else 'moderate'} predictive performance")
        report.append("")

        report.append("="*80)
        report.append("5. FILES GENERATED")
        report.append("="*80)
        report.append("")

        report.append("Models:")
        report.append("  - models/nephrotoxicity_model.json")
        report.append("  - models/clinical_cure_model.json")
        report.append("  - models/pk_cmax_model.json")
        report.append("  - models/pk_auc24_model.json")
        report.append("")

        report.append("Visualizations:")
        report.append("  - results/phase4_ml/roc_curves.png")
        report.append("  - results/phase4_ml/pr_curves.png")
        report.append("  - results/phase4_ml/confusion_matrices.png")
        report.append("  - results/phase4_ml/feature_importance.png")
        report.append("  - results/phase4_ml/pk_surrogate_performance.png")
        report.append("  - results/phase4_ml/calibration_curves.png")
        report.append("")

        report.append("Data:")
        report.append("  - results/phase4_ml/model_performance.json")
        report.append("")

        report.append("="*80)
        report.append("Phase 4 Analysis Complete!")
        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        output_file = self.results_dir / 'PHASE4_SUMMARY.txt'
        with open(output_file, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✓ Summary report saved to: {output_file}")
        print()

    def run_complete_pipeline(self):
        """
        Run complete ML pipeline
        """
        print("Starting Phase 4 complete ML pipeline...")
        print()

        # Build models
        self.build_nephrotoxicity_model(feature_set='baseline')
        self.build_clinical_cure_model(feature_set='pkpd')
        self.build_pk_surrogate_models()

        # SHAP analysis (if available)
        if SHAP_AVAILABLE:
            try:
                self.perform_shap_analysis('nephrotoxicity')
                self.perform_shap_analysis('clinical_cure')
            except Exception as e:
                print(f"SHAP analysis failed: {e}")

        # Generate visualizations
        self.generate_visualizations()

        # Save models and results
        self.save_models()
        self.save_performance_summary()
        self.generate_summary_report()

        print("="*80)
        print("✅ PHASE 4 COMPLETE!")
        print("="*80)
        print()
        print("Next step: Phase 5 - Bayesian Dose Optimization")
        print("  Run: python3 phase5_dose_optimization.py")
        print()


def main():
    """Main execution function"""

    # Initialize pipeline
    pipeline = MLPipeline()

    # Run complete pipeline
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
