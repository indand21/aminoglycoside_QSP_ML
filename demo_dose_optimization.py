#!/usr/bin/env python3
"""
Demonstration: Personalized Dose Optimization for Individual Patients

This script demonstrates how to use the aminoglycoside QSP-ML framework
to generate personalized dose recommendations for specific patients.

Author: Aminoglycoside QSP-ML Project
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PersonalizedDoseOptimizer:
    """
    Personalized aminoglycoside dose optimizer using ML predictions
    """

    def __init__(self):
        """Initialize optimizer with trained models"""
        print("="*80)
        print("Personalized Aminoglycoside Dose Optimizer")
        print("="*80)
        print()

        # Load ML models
        print("Loading trained ML models...")
        self.models = {}

        # Try enhanced models first, fall back to original
        model_paths = {
            'Cmax': 'models/enhanced/Cmax_surrogate_enhanced.json',
            'AUC24': 'models/enhanced/AUC24_surrogate_enhanced.json',
            'nephrotoxicity': 'models/nephrotoxicity_model.json',
            'clinical_cure': 'models/clinical_cure_model.json'
        }

        for model_name, path in model_paths.items():
            if Path(path).exists():
                if 'surrogate' in path:
                    self.models[model_name] = xgb.XGBRegressor()
                else:
                    self.models[model_name] = xgb.XGBClassifier()
                self.models[model_name].load_model(path)
                print(f"  ✓ Loaded {model_name} model")
            else:
                print(f"  ⚠ Warning: {model_name} model not found at {path}")

        print()

        # Define PK/PD targets
        self.targets = {
            'cmax_mic_min': 8,      # Minimum Cmax/MIC for efficacy
            'auc_mic_min': 80,      # Minimum AUC/MIC for efficacy
            'trough_max': 2.0,      # Maximum trough to avoid toxicity (mg/L)
            'dose_min': 200,        # Minimum dose (mg)
            'dose_max': 1600        # Maximum dose (mg)
        }

    def predict_pk_outcomes(self, patient_params, dose):
        """
        Predict PK parameters and outcomes for a given dose

        Parameters:
        -----------
        patient_params : dict
            Patient characteristics
        dose : float
            Dose in mg

        Returns:
        --------
        predictions : dict
            Predicted PK parameters and outcomes
        """
        # Prepare features for PK prediction
        pk_features = [
            patient_params['weight'],
            patient_params['baseline_crcl'],
            dose
        ]

        # Predict Cmax and AUC24
        if 'Cmax' in self.models and 'AUC24' in self.models:
            cmax = self.models['Cmax'].predict(np.array(pk_features).reshape(1, -1))[0]
            auc24 = self.models['AUC24'].predict(np.array(pk_features).reshape(1, -1))[0]
        else:
            # Fallback: simple PK equations
            cmax = dose / patient_params['weight'] * 1.5  # Simplified
            auc24 = cmax * 8  # Simplified

        # Estimate trough (simplified: 5% of Cmax)
        cmin = max(0.1, cmax * 0.05)

        # Calculate PK/PD indices
        mic = patient_params.get('mic', 2.0)  # Default MIC if not provided
        cmax_mic = cmax / mic
        auc_mic = auc24 / mic

        # Predict clinical outcomes (if models available)
        p_cure = 0.5  # Default
        p_aki = 0.2   # Default

        # Note: Full feature preparation would require all baseline covariates
        # This is simplified for demonstration

        predictions = {
            'cmax': cmax,
            'cmin': cmin,
            'auc24': auc24,
            'cmax_mic': cmax_mic,
            'auc_mic': auc_mic,
            'p_cure': p_cure,
            'p_aki': p_aki,
            'dose': dose
        }

        return predictions

    def evaluate_dose(self, patient_params, dose):
        """
        Evaluate a dose against clinical targets

        Returns score (0-100) and detailed assessment
        """
        pred = self.predict_pk_outcomes(patient_params, dose)

        # Target attainment
        targets_met = {
            'cmax_mic': pred['cmax_mic'] >= self.targets['cmax_mic_min'],
            'auc_mic': pred['auc_mic'] >= self.targets['auc_mic_min'],
            'trough_safe': pred['cmin'] < self.targets['trough_max']
        }

        # Calculate score components
        efficacy_score = 0
        if targets_met['cmax_mic']:
            efficacy_score += 30
        if targets_met['auc_mic']:
            efficacy_score += 30

        safety_score = 0
        if targets_met['trough_safe']:
            safety_score += 40

        total_score = efficacy_score + safety_score

        return {
            'score': total_score,
            'predictions': pred,
            'targets_met': targets_met,
            'assessment': self._generate_assessment(pred, targets_met)
        }

    def _generate_assessment(self, pred, targets_met):
        """Generate clinical assessment text"""
        assessment = []

        # Efficacy assessment
        if targets_met['cmax_mic'] and targets_met['auc_mic']:
            assessment.append("✅ EXCELLENT efficacy potential (both PK/PD targets achieved)")
        elif targets_met['cmax_mic'] or targets_met['auc_mic']:
            assessment.append("⚠️  MODERATE efficacy potential (partial target attainment)")
        else:
            assessment.append("❌ SUBOPTIMAL efficacy (PK/PD targets not achieved)")

        # Safety assessment
        if targets_met['trough_safe']:
            assessment.append("✅ LOW nephrotoxicity risk (trough <2 mg/L)")
        else:
            assessment.append("⚠️  ELEVATED nephrotoxicity risk (trough ≥2 mg/L)")

        return " | ".join(assessment)

    def optimize_dose(self, patient_params, n_doses=50):
        """
        Find optimal dose by grid search

        Parameters:
        -----------
        patient_params : dict
            Patient characteristics
        n_doses : int
            Number of doses to evaluate

        Returns:
        --------
        optimal_dose : float
            Best dose
        results : pd.DataFrame
            All evaluated doses
        """
        print(f"Optimizing dose for patient...")
        print(f"  Weight: {patient_params['weight']} kg")
        print(f"  CrCL: {patient_params['baseline_crcl']} mL/min")
        print(f"  MIC: {patient_params.get('mic', 2.0)} mg/L")
        print()

        # Evaluate doses across range
        doses = np.linspace(self.targets['dose_min'],
                           self.targets['dose_max'],
                           n_doses)

        results = []
        for dose in doses:
            eval_result = self.evaluate_dose(patient_params, dose)
            results.append({
                'dose': dose,
                'score': eval_result['score'],
                **eval_result['predictions'],
                **{f'target_{k}': v for k, v in eval_result['targets_met'].items()}
            })

        results_df = pd.DataFrame(results)

        # Find optimal dose (highest score)
        optimal_idx = results_df['score'].idxmax()
        optimal_dose = results_df.loc[optimal_idx, 'dose']

        print(f"✓ Optimization complete!")
        print(f"  Optimal dose: {optimal_dose:.0f} mg ({optimal_dose/patient_params['weight']:.1f} mg/kg)")
        print()

        return optimal_dose, results_df

    def plot_dose_response(self, results_df, patient_params, save_path=None):
        """
        Plot dose-response curves
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: PK/PD Indices
        ax = axes[0, 0]
        ax.plot(results_df['dose'], results_df['cmax_mic'], 'o-',
                label='Cmax/MIC', linewidth=2, markersize=4)
        ax.axhline(self.targets['cmax_mic_min'], color='red', linestyle='--',
                   label=f'Target (≥{self.targets["cmax_mic_min"]})')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Cmax/MIC Ratio', fontsize=11)
        ax.set_title('Efficacy Target: Peak Concentration', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Safety (Trough)
        ax = axes[0, 1]
        ax.plot(results_df['dose'], results_df['cmin'], 'o-',
                color='orange', linewidth=2, markersize=4)
        ax.axhline(self.targets['trough_max'], color='red', linestyle='--',
                   label=f'Safety limit (<{self.targets["trough_max"]} mg/L)')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Trough Concentration (mg/L)', fontsize=11)
        ax.set_title('Safety: Trough Concentration', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: AUC/MIC
        ax = axes[1, 0]
        ax.plot(results_df['dose'], results_df['auc_mic'], 'o-',
                color='green', linewidth=2, markersize=4)
        ax.axhline(self.targets['auc_mic_min'], color='red', linestyle='--',
                   label=f'Target (≥{self.targets["auc_mic_min"]})')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('AUC/MIC Ratio', fontsize=11)
        ax.set_title('Efficacy Target: AUC/MIC', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Overall Score
        ax = axes[1, 1]
        ax.plot(results_df['dose'], results_df['score'], 'o-',
                color='purple', linewidth=2, markersize=4)
        optimal_dose = results_df.loc[results_df['score'].idxmax(), 'dose']
        optimal_score = results_df['score'].max()
        ax.axvline(optimal_dose, color='red', linestyle='--',
                   label=f'Optimal: {optimal_dose:.0f} mg (Score: {optimal_score:.0f})')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Optimization Score (0-100)', fontsize=11)
        ax.set_title('Overall Dose Optimization Score', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Dose Optimization: {patient_params["weight"]:.0f} kg patient, '
                    f'CrCL {patient_params["baseline_crcl"]:.0f} mL/min',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Dose-response plot saved: {save_path}")

        return fig

    def generate_recommendation_report(self, patient_params, optimal_dose, results_df):
        """
        Generate detailed clinical recommendation report
        """
        # Get optimal dose predictions
        optimal_result = self.evaluate_dose(patient_params, optimal_dose)
        pred = optimal_result['predictions']
        targets = optimal_result['targets_met']

        report = []
        report.append("="*80)
        report.append("PERSONALIZED AMINOGLYCOSIDE DOSING RECOMMENDATION")
        report.append("="*80)
        report.append("")

        # Patient characteristics
        report.append("PATIENT CHARACTERISTICS:")
        report.append(f"  Weight: {patient_params['weight']:.1f} kg")
        report.append(f"  Creatinine Clearance: {patient_params['baseline_crcl']:.1f} mL/min")
        report.append(f"  Pathogen MIC: {patient_params.get('mic', 2.0):.2f} mg/L")
        report.append("")

        # Dose recommendation
        report.append("RECOMMENDED DOSE:")
        report.append(f"  ✓ {optimal_dose:.0f} mg once daily")
        report.append(f"  ✓ {optimal_dose/patient_params['weight']:.1f} mg/kg")
        report.append("")

        # Predicted PK parameters
        report.append("PREDICTED PK PARAMETERS:")
        report.append(f"  Peak (Cmax): {pred['cmax']:.1f} mg/L")
        report.append(f"  Trough (Cmin): {pred['cmin']:.2f} mg/L")
        report.append(f"  AUC24: {pred['auc24']:.1f} mg·h/L")
        report.append("")

        # PK/PD target attainment
        report.append("PK/PD TARGET ATTAINMENT:")
        status_cmax = "✅ ACHIEVED" if targets['cmax_mic'] else "❌ NOT ACHIEVED"
        status_auc = "✅ ACHIEVED" if targets['auc_mic'] else "❌ NOT ACHIEVED"
        status_trough = "✅ SAFE" if targets['trough_safe'] else "⚠️  ELEVATED RISK"

        report.append(f"  Cmax/MIC = {pred['cmax_mic']:.1f} (target ≥{self.targets['cmax_mic_min']}) - {status_cmax}")
        report.append(f"  AUC/MIC = {pred['auc_mic']:.1f} (target ≥{self.targets['auc_mic_min']}) - {status_auc}")
        report.append(f"  Trough = {pred['cmin']:.2f} mg/L (target <{self.targets['trough_max']}) - {status_trough}")
        report.append("")

        # Clinical interpretation
        report.append("CLINICAL INTERPRETATION:")
        report.append(f"  {optimal_result['assessment']}")
        report.append("")

        # Monitoring recommendations
        report.append("MONITORING RECOMMENDATIONS:")
        report.append("  • Measure trough concentration before 3rd or 4th dose")
        report.append("  • Target trough: <2 mg/L (once-daily dosing)")
        report.append("  • Monitor serum creatinine daily for first 3 days")
        report.append("  • Adjust dose if CrCL changes >20%")
        report.append("")

        # Alternative doses
        report.append("ALTERNATIVE DOSE OPTIONS:")
        # Find doses within 90% of optimal score
        threshold = optimal_result['score'] * 0.9
        alternatives = results_df[results_df['score'] >= threshold].sort_values('score', ascending=False)

        for idx, row in alternatives.head(3).iterrows():
            if abs(row['dose'] - optimal_dose) > 50:  # Different from optimal
                report.append(f"  • {row['dose']:.0f} mg - Score: {row['score']:.0f}/100")

        report.append("")
        report.append("="*80)
        report.append("Generated by: Aminoglycoside QSP-ML Framework v2.0")
        report.append("="*80)

        return "\n".join(report)


def demo_patient_1():
    """Example 1: Standard adult patient with normal renal function"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Standard Adult Patient")
    print("="*80 + "\n")

    optimizer = PersonalizedDoseOptimizer()

    # Define patient
    patient = {
        'weight': 70,           # kg
        'baseline_crcl': 100,   # mL/min (normal)
        'mic': 2.0              # mg/L (susceptible organism)
    }

    # Optimize dose
    optimal_dose, results = optimizer.optimize_dose(patient, n_doses=30)

    # Generate report
    report = optimizer.generate_recommendation_report(patient, optimal_dose, results)
    print(report)

    # Plot
    fig = optimizer.plot_dose_response(results, patient,
                                       save_path='results/demo_patient1_dose_optimization.png')
    plt.show()

    return optimizer, patient, optimal_dose, results


def demo_patient_2():
    """Example 2: Obese patient with renal impairment"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Obese Patient with Renal Impairment")
    print("="*80 + "\n")

    optimizer = PersonalizedDoseOptimizer()

    # Define patient
    patient = {
        'weight': 120,          # kg (obese)
        'baseline_crcl': 40,    # mL/min (moderate renal impairment)
        'mic': 4.0              # mg/L (higher MIC)
    }

    # Optimize dose
    optimal_dose, results = optimizer.optimize_dose(patient, n_doses=30)

    # Generate report
    report = optimizer.generate_recommendation_report(patient, optimal_dose, results)
    print(report)

    # Plot
    fig = optimizer.plot_dose_response(results, patient,
                                       save_path='results/demo_patient2_dose_optimization.png')
    plt.show()

    return optimizer, patient, optimal_dose, results


def demo_patient_3():
    """Example 3: Elderly patient with good renal function"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Elderly Patient with Preserved Renal Function")
    print("="*80 + "\n")

    optimizer = PersonalizedDoseOptimizer()

    # Define patient
    patient = {
        'weight': 55,           # kg (lower weight)
        'baseline_crcl': 80,    # mL/min (good for age)
        'mic': 1.0              # mg/L (very susceptible)
    }

    # Optimize dose
    optimal_dose, results = optimizer.optimize_dose(patient, n_doses=30)

    # Generate report
    report = optimizer.generate_recommendation_report(patient, optimal_dose, results)
    print(report)

    # Plot
    fig = optimizer.plot_dose_response(results, patient,
                                       save_path='results/demo_patient3_dose_optimization.png')
    plt.show()

    return optimizer, patient, optimal_dose, results


def main():
    """Run all demonstration examples"""
    print("\n" + "="*80)
    print("DEMONSTRATION: Personalized Aminoglycoside Dose Optimization")
    print("="*80)
    print("\nThis script demonstrates how to use the QSP-ML framework")
    print("to generate personalized dose recommendations for different")
    print("patient scenarios.\n")

    # Create results directory
    Path('results').mkdir(exist_ok=True)

    # Run examples
    print("\nRunning 3 patient examples...\n")

    # Example 1: Standard patient
    opt1, pat1, dose1, res1 = demo_patient_1()

    # Example 2: Obese with renal impairment
    opt2, pat2, dose2, res2 = demo_patient_2()

    # Example 3: Elderly patient
    opt3, pat3, dose3, res3 = demo_patient_3()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/demo_patient1_dose_optimization.png")
    print("  - results/demo_patient2_dose_optimization.png")
    print("  - results/demo_patient3_dose_optimization.png")
    print("\nYou can modify this script to optimize doses for your own patients!")
    print()


if __name__ == '__main__':
    main()
