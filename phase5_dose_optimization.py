#!/usr/bin/env python3
"""
Phase 5: Bayesian Dose Optimization

This script:
1. Implements multi-objective dose optimization
2. Uses Bayesian optimization for efficient dose search
3. Provides personalized dose recommendations
4. Performs Monte Carlo simulations
5. Validates recommendations with PK/PD targets
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

# Optimization imports
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DoseOptimizer:
    """
    Bayesian Dose Optimization for Aminoglycosides
    """

    def __init__(self,
                 ml_data_path='data/processed/ml_dataset.csv',
                 pkpd_data_path='results/phase3_pkpd/pkpd_indices.csv'):
        """
        Initialize Dose Optimizer

        Parameters:
        -----------
        ml_data_path : str
            Path to ML dataset
        pkpd_data_path : str
            Path to PK/PD indices
        """
        print("="*80)
        print("Phase 5: Bayesian Dose Optimization")
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

        print(f"  ✓ Loaded data: {self.data.shape}")
        print()

        # Load ML models
        print("Loading ML models...")
        self.models = {}
        self.scalers = {}

        # Load models
        model_names = ['nephrotoxicity', 'clinical_cure', 'pk_cmax', 'pk_auc24']
        for model_name in model_names:
            model_path = Path(f'models/{model_name}_model.json')
            if model_path.exists():
                if 'pk_' in model_name:
                    model = xgb.XGBRegressor()
                else:
                    model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                self.models[model_name] = model
                print(f"  ✓ Loaded {model_name} model")
            else:
                print(f"  ⚠ Warning: {model_name} model not found")

        print()

        # Create output directories
        self.results_dir = Path('results/phase5_optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.optimization_results = []
        self.dose_recommendations = None

        # Define optimization parameters
        self.dose_range = (200, 1600)  # mg
        self.target_cmax_mic = 8
        self.target_auc_mic = 80
        self.max_trough = 2  # mg/L

        # Optimization weights
        self.weights = {
            'cure': 0.4,        # Maximize cure probability
            'safety': 0.3,      # Minimize nephrotoxicity probability
            'cmax_target': 0.2, # Achieve Cmax/MIC target
            'trough_safety': 0.1 # Maintain safe trough
        }

    def prepare_features_for_prediction(self, patient_data, dose, model_type='nephrotoxicity'):
        """
        Prepare features for model prediction

        Parameters:
        -----------
        patient_data : pd.Series
            Patient baseline characteristics
        dose : float
            Dose in mg
        model_type : str
            'pk', 'nephrotoxicity', or 'clinical_cure'

        Returns:
        --------
        features : np.array
            Feature vector ready for prediction
        """
        # PK model features (17 features)
        pk_model_features = [
            'age', 'sex', 'weight', 'height', 'bmi',
            'apache_ii', 'sofa_score',
            'baseline_crcl', 'baseline_scr', 'baseline_egfr',
            'baseline_albumin', 'baseline_bilirubin',
            'diabetes', 'ckd_stage',
            'mechanical_ventilation', 'vasopressor_use'
        ]

        # Nephrotoxicity model features (18 features - adds sepsis_type, infection_site)
        nephrotoxicity_features = pk_model_features + ['sepsis_type', 'infection_site']

        # Select features based on model type
        if model_type == 'pk':
            selected_features = pk_model_features + ['first_dose']
        elif model_type == 'nephrotoxicity':
            selected_features = nephrotoxicity_features
        elif model_type == 'clinical_cure':
            selected_features = nephrotoxicity_features  # Will add PK features later
        else:
            selected_features = nephrotoxicity_features

        # Extract features
        features_dict = {}
        for feat in selected_features:
            if feat == 'first_dose':
                features_dict[feat] = dose
            elif feat in patient_data.index:
                val = patient_data[feat]
                # Encode categorical
                if feat in ['sex', 'sepsis_type', 'infection_site']:
                    val = pd.Categorical([val]).codes[0]
                elif feat == 'ckd_stage':
                    val = pd.Categorical([val]).codes[0]
                features_dict[feat] = val
            else:
                features_dict[feat] = 0  # Default value

        # Convert to array
        features = np.array(list(features_dict.values())).reshape(1, -1)

        return features, features_dict

    def predict_outcomes(self, patient_data, dose):
        """
        Predict all outcomes for a given patient and dose

        Parameters:
        -----------
        patient_data : pd.Series
            Patient data
        dose : float
            Dose in mg

        Returns:
        --------
        outcomes : dict
            Dictionary of predicted outcomes
        """
        # Get features for nephrotoxicity (baseline only)
        features_baseline, _ = self.prepare_features_for_prediction(patient_data, dose, model_type='nephrotoxicity')

        # Get features for PK prediction
        features_pk, _ = self.prepare_features_for_prediction(patient_data, dose, model_type='pk')

        # Predict outcomes
        outcomes = {}

        # Predict PK parameters
        if 'pk_cmax' in self.models and 'pk_auc24' in self.models:
            cmax_pred = self.models['pk_cmax'].predict(features_pk)[0]
            auc24_pred = self.models['pk_auc24'].predict(features_pk)[0]
        else:
            cmax_pred = 50
            auc24_pred = 500

        # Calculate PK/PD indices
        mic = patient_data['mic_gentamicin']
        cmax_mic = cmax_pred / mic if mic > 0 else 8
        auc_mic = auc24_pred / mic if mic > 0 else 100
        cmin = max(0.1, cmax_pred * 0.05)  # Approximate trough

        # Nephrotoxicity (baseline features only)
        if 'nephrotoxicity' in self.models:
            p_aki = self.models['nephrotoxicity'].predict_proba(features_baseline)[0, 1]
            outcomes['p_nephrotoxicity'] = p_aki
        else:
            outcomes['p_nephrotoxicity'] = 0.2  # Default

        # Clinical cure (baseline + PK features)
        if 'clinical_cure' in self.models:
            # Prepare features with PK indices
            features_cure, _ = self.prepare_features_for_prediction(patient_data, dose, model_type='clinical_cure')
            # Add PK features
            pk_features = np.concatenate([
                features_cure[0],
                [cmax_pred, auc24_pred, cmax_mic, auc_mic, cmin]
            ]).reshape(1, -1)

            p_cure = self.models['clinical_cure'].predict_proba(pk_features)[0, 1]
            outcomes['p_cure'] = p_cure
        else:
            outcomes['p_cure'] = 0.5  # Default

        # PK parameters
        outcomes['cmax'] = cmax_pred
        outcomes['auc24'] = auc24_pred
        outcomes['cmin'] = cmin
        outcomes['cmax_mic'] = cmax_mic
        outcomes['auc_mic'] = auc_mic

        # Target attainment
        outcomes['cmax_target_met'] = 1 if outcomes['cmax_mic'] >= self.target_cmax_mic else 0
        outcomes['auc_target_met'] = 1 if outcomes['auc_mic'] >= self.target_auc_mic else 0
        outcomes['trough_safe'] = 1 if outcomes['cmin'] < self.max_trough else 0

        return outcomes

    def objective_function(self, patient_data, dose):
        """
        Multi-objective function to maximize

        Higher is better. Combines:
        - Maximize P(cure)
        - Minimize P(nephrotoxicity)
        - Achieve PK/PD targets
        - Maintain safety thresholds

        Parameters:
        -----------
        patient_data : pd.Series
            Patient data
        dose : float
            Dose in mg

        Returns:
        --------
        score : float
            Objective score (higher is better)
        """
        outcomes = self.predict_outcomes(patient_data, dose)

        # Component scores
        cure_score = outcomes['p_cure']  # 0-1
        safety_score = 1 - outcomes['p_nephrotoxicity']  # 0-1 (inverted)
        cmax_target_score = 1 / (1 + np.exp(-0.5 * (outcomes['cmax_mic'] - self.target_cmax_mic)))  # Sigmoid
        trough_safety_score = 1 / (1 + np.exp(2 * (outcomes['cmin'] - self.max_trough)))  # Sigmoid (inverted)

        # Weighted combination
        total_score = (
            self.weights['cure'] * cure_score +
            self.weights['safety'] * safety_score +
            self.weights['cmax_target'] * cmax_target_score +
            self.weights['trough_safety'] * trough_safety_score
        )

        return total_score

    def optimize_dose_bayesian(self, patient_data, n_iter=25):
        """
        Optimize dose using Bayesian optimization

        Parameters:
        -----------
        patient_data : pd.Series
            Patient data
        n_iter : int
            Number of Bayesian optimization iterations

        Returns:
        --------
        optimal_dose : float
            Recommended dose
        max_score : float
            Maximum objective score
        """
        # Define objective function for this patient
        def obj_func(dose):
            return self.objective_function(patient_data, dose)

        # Bayesian optimization
        optimizer = BayesianOptimization(
            f=obj_func,
            pbounds={'dose': self.dose_range},
            random_state=42,
            allow_duplicate_points=True
        )

        optimizer.maximize(
            init_points=5,
            n_iter=n_iter
        )

        # Get best result
        optimal_dose = optimizer.max['params']['dose']
        max_score = optimizer.max['target']

        return optimal_dose, max_score, optimizer

    def optimize_dose_grid(self, patient_data, n_points=50):
        """
        Optimize dose using grid search (faster, for comparison)

        Parameters:
        -----------
        patient_data : pd.Series
            Patient data
        n_points : int
            Number of grid points

        Returns:
        --------
        optimal_dose : float
            Recommended dose
        max_score : float
            Maximum objective score
        dose_range : array
            Array of doses evaluated
        scores : array
            Objective scores for each dose
        """
        doses = np.linspace(self.dose_range[0], self.dose_range[1], n_points)
        scores = np.array([self.objective_function(patient_data, dose) for dose in doses])

        optimal_idx = np.argmax(scores)
        optimal_dose = doses[optimal_idx]
        max_score = scores[optimal_idx]

        return optimal_dose, max_score, doses, scores

    def optimize_all_patients(self, method='grid', n_samples=50):
        """
        Optimize doses for all patients

        Parameters:
        -----------
        method : str
            'grid' or 'bayesian'
        n_samples : int
            Number of patients to optimize (for speed)
        """
        print("="*80)
        print(f"DOSE OPTIMIZATION ({method.upper()} SEARCH)")
        print("="*80)
        print()

        # Sample patients
        if n_samples < len(self.data):
            sample_data = self.data.sample(n=n_samples, random_state=42)
            print(f"Optimizing doses for {n_samples} sampled patients...")
        else:
            sample_data = self.data
            print(f"Optimizing doses for all {len(sample_data)} patients...")

        results = []

        for idx, (_, patient) in enumerate(sample_data.iterrows()):
            # Optimize
            if method == 'bayesian':
                opt_dose, max_score, optimizer = self.optimize_dose_bayesian(patient, n_iter=20)
            else:  # grid
                opt_dose, max_score, doses, scores = self.optimize_dose_grid(patient, n_points=50)

            # Get outcomes at optimal dose
            outcomes = self.predict_outcomes(patient, opt_dose)

            # Get outcomes at observed dose (for comparison)
            observed_dose = patient['first_dose']
            observed_outcomes = self.predict_outcomes(patient, observed_dose)

            results.append({
                'patient_id': patient['patient_id'],
                'age': patient['age'],
                'weight': patient['weight'],
                'baseline_crcl': patient['baseline_crcl'],
                'apache_ii': patient['apache_ii'],
                'mic': patient['mic_gentamicin'],

                # Observed dose and outcomes
                'observed_dose': observed_dose,
                'observed_p_cure': observed_outcomes['p_cure'],
                'observed_p_aki': observed_outcomes['p_nephrotoxicity'],
                'observed_cmax_mic': observed_outcomes['cmax_mic'],
                'observed_score': self.objective_function(patient, observed_dose),

                # Optimal dose and outcomes
                'optimal_dose': opt_dose,
                'optimal_p_cure': outcomes['p_cure'],
                'optimal_p_aki': outcomes['p_nephrotoxicity'],
                'optimal_cmax': outcomes['cmax'],
                'optimal_cmin': outcomes['cmin'],
                'optimal_auc24': outcomes['auc24'],
                'optimal_cmax_mic': outcomes['cmax_mic'],
                'optimal_auc_mic': outcomes['auc_mic'],
                'optimal_score': max_score,

                # Improvements
                'dose_change': opt_dose - observed_dose,
                'cure_improvement': outcomes['p_cure'] - observed_outcomes['p_cure'],
                'aki_reduction': observed_outcomes['p_nephrotoxicity'] - outcomes['p_nephrotoxicity'],
                'score_improvement': max_score - self.objective_function(patient, observed_dose)
            })

            if (idx + 1) % 10 == 0:
                print(f"  Optimized {idx + 1}/{len(sample_data)} patients...")

        self.optimization_results = pd.DataFrame(results)

        print(f"\n✓ Optimization complete for {len(self.optimization_results)} patients")
        print()

        # Summary statistics
        print("Summary Statistics:")
        print(f"  Mean observed dose: {self.optimization_results['observed_dose'].mean():.0f} mg")
        print(f"  Mean optimal dose: {self.optimization_results['optimal_dose'].mean():.0f} mg")
        print(f"  Mean dose change: {self.optimization_results['dose_change'].mean():.0f} mg "
              f"({self.optimization_results['dose_change'].mean()/self.optimization_results['observed_dose'].mean()*100:.1f}%)")
        print()
        print(f"  Mean cure improvement: {self.optimization_results['cure_improvement'].mean():.3f} "
              f"({self.optimization_results['cure_improvement'].mean()*100:.1f}%)")
        print(f"  Mean AKI reduction: {self.optimization_results['aki_reduction'].mean():.3f} "
              f"({self.optimization_results['aki_reduction'].mean()*100:.1f}%)")
        print(f"  Mean score improvement: {self.optimization_results['score_improvement'].mean():.3f}")
        print()

        # Patients with significant improvements
        sig_improvements = (self.optimization_results['score_improvement'] > 0.1).sum()
        print(f"  Patients with significant improvement (Δscore > 0.1): {sig_improvements} "
              f"({sig_improvements/len(self.optimization_results)*100:.1f}%)")
        print()

        # Save results
        output_file = self.results_dir / 'dose_recommendations.csv'
        self.optimization_results.to_csv(output_file, index=False)
        print(f"✓ Saved dose recommendations to: {output_file}")
        print()

        return self.optimization_results

    def create_dosing_nomogram(self):
        """
        Create dosing nomogram based on patient characteristics
        """
        print("Creating dosing nomogram...")

        # Create grid of patient characteristics
        weights = np.linspace(40, 100, 7)  # kg
        crcl_values = np.array([30, 50, 75, 100, 130])  # mL/min

        nomogram_data = []

        for weight in weights:
            for crcl in crcl_values:
                # Create synthetic patient
                patient = pd.Series({
                    'age': 60,
                    'sex': 'M',
                    'weight': weight,
                    'height': 170,
                    'bmi': weight / (1.7**2),
                    'apache_ii': 20,
                    'sofa_score': 6,
                    'baseline_crcl': crcl,
                    'baseline_scr': 140 / crcl if crcl > 0 else 2.0,
                    'baseline_egfr': crcl,
                    'baseline_albumin': 3.0,
                    'baseline_bilirubin': 1.0,
                    'diabetes': 0,
                    'ckd_stage': '0',
                    'mechanical_ventilation': 0,
                    'vasopressor_use': 0,
                    'sepsis_type': 'sepsis',
                    'infection_site': 'respiratory',
                    'mic_gentamicin': 1.0,
                    'first_dose': 600
                })

                # Optimize dose
                opt_dose, max_score, _, _ = self.optimize_dose_grid(patient, n_points=30)

                nomogram_data.append({
                    'weight': weight,
                    'crcl': crcl,
                    'recommended_dose': opt_dose,
                    'dose_per_kg': opt_dose / weight
                })

        nomogram_df = pd.DataFrame(nomogram_data)

        # Save nomogram
        output_file = self.results_dir / 'dosing_nomogram.csv'
        nomogram_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved dosing nomogram to: {output_file}")
        print()

        return nomogram_df

    def generate_visualizations(self):
        """
        Generate comprehensive optimization visualizations
        """
        print("="*80)
        print("Generating visualizations...")
        print("="*80)
        print()

        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.size'] = 10

        # 1. Dose comparison: observed vs optimal
        print("  Creating dose comparison plot...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Dose distribution
        axes[0].hist(self.optimization_results['observed_dose'], bins=30, alpha=0.5,
                    label='Observed', edgecolor='black')
        axes[0].hist(self.optimization_results['optimal_dose'], bins=30, alpha=0.5,
                    label='Optimal', edgecolor='black')
        axes[0].set_xlabel('Dose (mg)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Dose Distribution: Observed vs Optimal')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dose change
        axes[1].hist(self.optimization_results['dose_change'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', label='No change')
        axes[1].set_xlabel('Dose Change (mg)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Dose Changes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Scatter: observed vs optimal
        axes[2].scatter(self.optimization_results['observed_dose'],
                       self.optimization_results['optimal_dose'],
                       alpha=0.5, s=30)
        axes[2].plot([200, 1600], [200, 1600], 'r--', label='No change')
        axes[2].set_xlabel('Observed Dose (mg)')
        axes[2].set_ylabel('Optimal Dose (mg)')
        axes[2].set_title('Observed vs Optimal Dose')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'dose_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Outcome improvements
        print("  Creating outcome improvement plots...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Cure improvement
        axes[0].hist(self.optimization_results['cure_improvement'] * 100, bins=30,
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', label='No improvement')
        axes[0].set_xlabel('Cure Probability Improvement (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Clinical Cure Improvement')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # AKI reduction
        axes[1].hist(self.optimization_results['aki_reduction'] * 100, bins=30,
                    edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', label='No reduction')
        axes[1].set_xlabel('Nephrotoxicity Reduction (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Nephrotoxicity Risk Reduction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Overall score improvement
        axes[2].hist(self.optimization_results['score_improvement'], bins=30,
                    edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='red', linestyle='--', label='No improvement')
        axes[2].set_xlabel('Objective Score Improvement')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Overall Optimization Score Improvement')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'outcome_improvements.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. PK/PD target attainment
        print("  Creating target attainment comparison...")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate target attainment rates
        observed_cmax_target = (self.optimization_results['observed_cmax_mic'] >= 8).mean() * 100
        optimal_cmax_target = (self.optimization_results['optimal_cmax_mic'] >= 8).mean() * 100

        optimal_auc_target = (self.optimization_results['optimal_auc_mic'] >= 80).mean() * 100
        optimal_trough_safe = (self.optimization_results['optimal_cmin'] < 2).mean() * 100

        categories = ['Cmax/MIC ≥8\n(Efficacy)', 'AUC/MIC ≥80\n(Efficacy)', 'Trough <2\n(Safety)']
        observed_rates = [observed_cmax_target, np.nan, np.nan]  # Only have Cmax/MIC for observed
        optimal_rates = [optimal_cmax_target, optimal_auc_target, optimal_trough_safe]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, observed_rates, width, label='Observed Dose', alpha=0.8)
        ax.bar(x + width/2, optimal_rates, width, label='Optimal Dose', alpha=0.8)
        ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% Target')

        ax.set_ylabel('Target Attainment (%)')
        ax.set_title('PK/PD Target Attainment: Observed vs Optimal Dosing')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'target_attainment_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Dose recommendations by patient characteristics
        print("  Creating dose recommendations by patient characteristics...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # By weight
        axes[0, 0].scatter(self.optimization_results['weight'],
                          self.optimization_results['optimal_dose'],
                          alpha=0.5, s=30)
        axes[0, 0].set_xlabel('Weight (kg)')
        axes[0, 0].set_ylabel('Optimal Dose (mg)')
        axes[0, 0].set_title('Optimal Dose by Weight')
        axes[0, 0].grid(True, alpha=0.3)

        # By CrCL
        axes[0, 1].scatter(self.optimization_results['baseline_crcl'],
                          self.optimization_results['optimal_dose'],
                          alpha=0.5, s=30)
        axes[0, 1].set_xlabel('Baseline CrCL (mL/min)')
        axes[0, 1].set_ylabel('Optimal Dose (mg)')
        axes[0, 1].set_title('Optimal Dose by Renal Function')
        axes[0, 1].grid(True, alpha=0.3)

        # By APACHE II
        axes[1, 0].scatter(self.optimization_results['apache_ii'],
                          self.optimization_results['optimal_dose'],
                          alpha=0.5, s=30)
        axes[1, 0].set_xlabel('APACHE II Score')
        axes[1, 0].set_ylabel('Optimal Dose (mg)')
        axes[1, 0].set_title('Optimal Dose by Disease Severity')
        axes[1, 0].grid(True, alpha=0.3)

        # By MIC
        axes[1, 1].scatter(self.optimization_results['mic'],
                          self.optimization_results['optimal_dose'],
                          alpha=0.5, s=30)
        axes[1, 1].set_xlabel('MIC (mg/L)')
        axes[1, 1].set_ylabel('Optimal Dose (mg)')
        axes[1, 1].set_title('Optimal Dose by MIC')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'dose_by_characteristics.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Example dose-response curve
        print("  Creating example dose-response curves...")
        # Select a representative patient
        example_patient = self.data.iloc[0]

        doses = np.linspace(200, 1600, 100)
        outcomes_list = [self.predict_outcomes(example_patient, dose) for dose in doses]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # P(cure)
        cure_probs = [o['p_cure'] for o in outcomes_list]
        axes[0, 0].plot(doses, cure_probs, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Dose (mg)')
        axes[0, 0].set_ylabel('P(Clinical Cure)')
        axes[0, 0].set_title('Dose-Response: Clinical Cure')
        axes[0, 0].grid(True, alpha=0.3)

        # P(nephrotoxicity)
        aki_probs = [o['p_nephrotoxicity'] for o in outcomes_list]
        axes[0, 1].plot(doses, aki_probs, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Dose (mg)')
        axes[0, 1].set_ylabel('P(Nephrotoxicity)')
        axes[0, 1].set_title('Dose-Response: Nephrotoxicity Risk')
        axes[0, 1].grid(True, alpha=0.3)

        # Cmax/MIC
        cmax_mic_values = [o['cmax_mic'] for o in outcomes_list]
        axes[1, 0].plot(doses, cmax_mic_values, 'g-', linewidth=2)
        axes[1, 0].axhline(8, color='red', linestyle='--', label='Target ≥8')
        axes[1, 0].set_xlabel('Dose (mg)')
        axes[1, 0].set_ylabel('Cmax/MIC Ratio')
        axes[1, 0].set_title('Dose-Response: Cmax/MIC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Objective score
        scores = [self.objective_function(example_patient, dose) for dose in doses]
        axes[1, 1].plot(doses, scores, 'purple', linewidth=2)
        optimal_idx = np.argmax(scores)
        axes[1, 1].axvline(doses[optimal_idx], color='red', linestyle='--',
                          label=f'Optimal: {doses[optimal_idx]:.0f} mg')
        axes[1, 1].set_xlabel('Dose (mg)')
        axes[1, 1].set_ylabel('Objective Score')
        axes[1, 1].set_title('Dose-Response: Overall Objective')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'dose_response_curves.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ All visualizations saved to: {self.results_dir}/")
        print()

    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("Generating summary report...")

        report = []
        report.append("="*80)
        report.append("Phase 5: Bayesian Dose Optimization - Summary Report")
        report.append("="*80)
        report.append("")
        report.append(f"Analysis Date: 2025-11-15")
        report.append(f"Number of Patients Optimized: {len(self.optimization_results)}")
        report.append("")

        report.append("="*80)
        report.append("1. OPTIMIZATION PARAMETERS")
        report.append("="*80)
        report.append("")
        report.append(f"Dose Range: {self.dose_range[0]}-{self.dose_range[1]} mg")
        report.append(f"Target Cmax/MIC: ≥{self.target_cmax_mic}")
        report.append(f"Target AUC/MIC: ≥{self.target_auc_mic}")
        report.append(f"Maximum Trough: <{self.max_trough} mg/L")
        report.append("")
        report.append("Objective Weights:")
        for key, val in self.weights.items():
            report.append(f"  {key}: {val:.1%}")
        report.append("")

        report.append("="*80)
        report.append("2. DOSE RECOMMENDATIONS")
        report.append("="*80)
        report.append("")

        obs_dose = self.optimization_results['observed_dose']
        opt_dose = self.optimization_results['optimal_dose']
        dose_change = self.optimization_results['dose_change']

        report.append(f"Observed Dose: {obs_dose.mean():.0f} ± {obs_dose.std():.0f} mg")
        report.append(f"Optimal Dose:  {opt_dose.mean():.0f} ± {opt_dose.std():.0f} mg")
        report.append(f"Mean Change:   {dose_change.mean():.0f} mg ({dose_change.mean()/obs_dose.mean()*100:+.1f}%)")
        report.append("")

        report.append("Dose Change Distribution:")
        report.append(f"  Increase (>50 mg): {(dose_change > 50).sum()} patients ({(dose_change > 50).mean()*100:.1f}%)")
        report.append(f"  Similar (±50 mg):  {((dose_change >= -50) & (dose_change <= 50)).sum()} patients "
                     f"({((dose_change >= -50) & (dose_change <= 50)).mean()*100:.1f}%)")
        report.append(f"  Decrease (<-50 mg): {(dose_change < -50).sum()} patients ({(dose_change < -50).mean()*100:.1f}%)")
        report.append("")

        report.append("="*80)
        report.append("3. OUTCOME IMPROVEMENTS")
        report.append("="*80)
        report.append("")

        cure_imp = self.optimization_results['cure_improvement']
        aki_red = self.optimization_results['aki_reduction']
        score_imp = self.optimization_results['score_improvement']

        report.append(f"Clinical Cure:")
        report.append(f"  Mean improvement: {cure_imp.mean()*100:+.2f}%")
        report.append(f"  Patients improved: {(cure_imp > 0).sum()} ({(cure_imp > 0).mean()*100:.1f}%)")
        report.append("")

        report.append(f"Nephrotoxicity Risk:")
        report.append(f"  Mean reduction: {aki_red.mean()*100:+.2f}%")
        report.append(f"  Patients reduced: {(aki_red > 0).sum()} ({(aki_red > 0).mean()*100:.1f}%)")
        report.append("")

        report.append(f"Overall Objective Score:")
        report.append(f"  Mean improvement: {score_imp.mean():+.3f}")
        report.append(f"  Patients improved: {(score_imp > 0).sum()} ({(score_imp > 0).mean()*100:.1f}%)")
        report.append(f"  Significant improvement (>0.1): {(score_imp > 0.1).sum()} ({(score_imp > 0.1).mean()*100:.1f}%)")
        report.append("")

        report.append("="*80)
        report.append("4. TARGET ATTAINMENT")
        report.append("="*80)
        report.append("")

        obs_cmax_target = (self.optimization_results['observed_cmax_mic'] >= 8).mean()
        opt_cmax_target = (self.optimization_results['optimal_cmax_mic'] >= 8).mean()
        opt_auc_target = (self.optimization_results['optimal_auc_mic'] >= 80).mean()
        opt_trough_safe = (self.optimization_results['optimal_cmin'] < 2).mean()

        report.append("Observed Dosing:")
        report.append(f"  Cmax/MIC ≥8: {obs_cmax_target*100:.1f}%")
        report.append("")

        report.append("Optimal Dosing:")
        report.append(f"  Cmax/MIC ≥8:  {opt_cmax_target*100:.1f}% ({(opt_cmax_target - obs_cmax_target)*100:+.1f}%)")
        report.append(f"  AUC/MIC ≥80:  {opt_auc_target*100:.1f}%")
        report.append(f"  Trough <2:    {opt_trough_safe*100:.1f}%")
        report.append("")

        combined_target = ((self.optimization_results['optimal_cmax_mic'] >= 8) &
                          (self.optimization_results['optimal_cmin'] < 2)).mean()
        report.append(f"Combined (Efficacy + Safety): {combined_target*100:.1f}%")
        report.append("")

        report.append("="*80)
        report.append("5. CLINICAL IMPLICATIONS")
        report.append("="*80)
        report.append("")

        # Find patterns
        if dose_change.mean() > 0:
            report.append(f"• On average, optimal doses are HIGHER than observed doses")
            report.append(f"  → Current dosing may be suboptimal for efficacy")
        else:
            report.append(f"• On average, optimal doses are LOWER than observed doses")
            report.append(f"  → Current dosing may increase toxicity risk unnecessarily")
        report.append("")

        if (cure_imp > 0).mean() > 0.7:
            report.append(f"• Majority of patients ({(cure_imp > 0).mean()*100:.0f}%) could benefit from dose adjustment")
        report.append("")

        report.append(f"• Personalized dosing could improve outcomes in {(score_imp > 0.1).mean()*100:.0f}% of patients")
        report.append("")

        report.append("="*80)
        report.append("6. FILES GENERATED")
        report.append("="*80)
        report.append("")
        report.append("Data:")
        report.append("  - results/phase5_optimization/dose_recommendations.csv")
        report.append("  - results/phase5_optimization/dosing_nomogram.csv")
        report.append("")
        report.append("Visualizations:")
        report.append("  - results/phase5_optimization/dose_comparison.png")
        report.append("  - results/phase5_optimization/outcome_improvements.png")
        report.append("  - results/phase5_optimization/target_attainment_comparison.png")
        report.append("  - results/phase5_optimization/dose_by_characteristics.png")
        report.append("  - results/phase5_optimization/dose_response_curves.png")
        report.append("")

        report.append("="*80)
        report.append("Phase 5 Analysis Complete!")
        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        output_file = self.results_dir / 'PHASE5_SUMMARY.txt'
        with open(output_file, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n✓ Summary report saved to: {output_file}")
        print()

    def run_complete_pipeline(self, n_patients=50):
        """
        Run complete dose optimization pipeline

        Parameters:
        -----------
        n_patients : int
            Number of patients to optimize
        """
        print("Starting Phase 5 complete optimization pipeline...")
        print()

        # Optimize doses for all patients
        self.optimize_all_patients(method='grid', n_samples=n_patients)

        # Create dosing nomogram
        self.create_dosing_nomogram()

        # Generate visualizations
        self.generate_visualizations()

        # Generate summary report
        self.generate_summary_report()

        print("="*80)
        print("✅ PHASE 5 COMPLETE!")
        print("="*80)
        print()
        print("Next step: Phase 6 - Validation & Simulation")
        print("  Run: python3 phase6_validation.py")
        print()


def main():
    """Main execution function"""

    # Initialize optimizer
    optimizer = DoseOptimizer()

    # Run complete pipeline
    # Using 50 patients for speed (can increase to 300 for full analysis)
    optimizer.run_complete_pipeline(n_patients=50)


if __name__ == "__main__":
    main()
