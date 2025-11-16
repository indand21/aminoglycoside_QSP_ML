#!/usr/bin/env python3
"""
Simple Demonstration: Personalized Dose Optimization

This demonstrates personalized aminoglycoside dosing using
simplified PK/PD equations based on the ML model insights.

Author: Aminoglycoside QSP-ML Project
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class SimpleDoseOptimizer:
    """
    Simplified aminoglycoside dose optimizer using PK/PD principles
    learned from the ML models
    """

    def __init__(self):
        print("="*80)
        print("Personalized Aminoglycoside Dose Optimizer")
        print("Using PK/PD principles from ML model analysis")
        print("="*80)
        print()

        # PK/PD targets based on ML model findings
        self.targets = {
            'cmax_mic_min': 8,      # Top predictor of clinical cure (6.2% importance)
            'auc_mic_min': 80,      # Second top predictor (5.8% importance)
            'trough_max': 2.0,      # Safety threshold (mg/L)
            'dose_min': 200,        # Minimum dose (mg)
            'dose_max': 1600        # Maximum dose (mg)
        }

    def predict_pk_parameters(self, weight, crcl, dose):
        """
        Predict PK parameters using population PK equations

        Based on typical aminoglycoside (gentamicin) PK:
        - Vd ~ 0.25 L/kg (extracellular fluid)
        - CL ~ CrCL * 0.7 (renal elimination)
        - Half-life ~ 2-3 hours (normal renal function)

        Parameters:
        -----------
        weight : float
            Body weight (kg)
        crcl : float
            Creatinine clearance (mL/min)
        dose : float
            Dose (mg)

        Returns:
        --------
        pk_params : dict
            Predicted PK parameters
        """
        # Volume of distribution (L)
        vd = 0.25 * weight  # L/kg

        # Clearance (L/h) - related to renal function
        cl = (crcl / 100) * 5.0  # Scaled based on CrCL

        # Peak concentration (mg/L) - immediately after infusion
        cmax = dose / vd

        # Half-life (hours)
        t_half = 0.693 * vd / cl

        # Elimination rate constant (1/h)
        ke = 0.693 / t_half

        # Trough at 24h (once-daily dosing)
        cmin = cmax * np.exp(-ke * 24)

        # AUC24 (mg·h/L)
        auc24 = dose / cl

        return {
            'cmax': cmax,
            'cmin': cmin,
            'auc24': auc24,
            'vd': vd,
            'cl': cl,
            't_half': t_half
        }

    def calculate_pkpd_indices(self, pk_params, mic):
        """Calculate PK/PD indices (top ML predictors)"""
        return {
            'cmax_mic': pk_params['cmax'] / mic,
            'auc_mic': pk_params['auc24'] / mic
        }

    def estimate_outcomes(self, pkpd_indices, cmin):
        """
        Estimate clinical outcomes based on ML model insights

        From ML analysis:
        - AUC/MIC is #1 predictor of cure (6.2% importance)
        - Cmax is #2 predictor (5.8% importance)
        - Higher PK/PD indices → better cure rates
        - Higher trough → higher nephrotoxicity risk
        """
        # Simplified logistic models based on ML findings

        # Clinical cure probability (from ML SHAP analysis)
        # Higher AUC/MIC increases cure (+0.05 per 10-unit increase)
        cure_logit = -1.0 + 0.05 * pkpd_indices['auc_mic'] + 0.03 * pkpd_indices['cmax_mic'] * 10
        p_cure = 1 / (1 + np.exp(-cure_logit))

        # Nephrotoxicity probability (from nephrotoxicity model)
        # Higher trough increases AKI risk
        aki_logit = -2.0 + 1.0 * cmin
        p_aki = 1 / (1 + np.exp(-aki_logit))

        return {
            'p_cure': min(max(p_cure, 0.1), 0.95),  # Bound between 10-95%
            'p_aki': min(max(p_aki, 0.05), 0.8)     # Bound between 5-80%
        }

    def evaluate_dose(self, weight, crcl, mic, dose):
        """
        Evaluate a dose against all criteria

        Returns comprehensive assessment
        """
        # Predict PK
        pk = self.predict_pk_parameters(weight, crcl, dose)

        # Calculate PK/PD indices
        pkpd = self.calculate_pkpd_indices(pk, mic)

        # Estimate outcomes
        outcomes = self.estimate_outcomes(pkpd, pk['cmin'])

        # Target attainment
        targets_met = {
            'cmax_mic': pkpd['cmax_mic'] >= self.targets['cmax_mic_min'],
            'auc_mic': pkpd['auc_mic'] >= self.targets['auc_mic_min'],
            'trough_safe': pk['cmin'] < self.targets['trough_max']
        }

        # Calculate optimization score (0-100)
        # Weights from Phase 5: cure 40%, safety 30%, targets 30%
        efficacy_score = 40 * outcomes['p_cure']
        safety_score = 30 * (1 - outcomes['p_aki'])
        target_score = 15 * int(targets_met['cmax_mic']) + 15 * int(targets_met['auc_mic'])

        total_score = efficacy_score + safety_score + target_score

        return {
            'dose': dose,
            'pk': pk,
            'pkpd': pkpd,
            'outcomes': outcomes,
            'targets_met': targets_met,
            'score': total_score,
            'dose_per_kg': dose / weight
        }

    def optimize_dose(self, weight, crcl, mic, n_doses=50):
        """
        Find optimal dose by grid search

        Returns:
        --------
        optimal_dose : float
            Best dose
        all_results : pd.DataFrame
            All evaluated doses
        """
        print(f"Optimizing dose...")
        print(f"  Weight: {weight} kg")
        print(f"  CrCL: {crcl} mL/min")
        print(f"  Pathogen MIC: {mic} mg/L")
        print()

        # Evaluate doses across range
        doses = np.linspace(self.targets['dose_min'], self.targets['dose_max'], n_doses)

        results = []
        for dose in doses:
            result = self.evaluate_dose(weight, crcl, mic, dose)
            results.append({
                'dose': dose,
                'cmax': result['pk']['cmax'],
                'cmin': result['pk']['cmin'],
                'auc24': result['pk']['auc24'],
                'cmax_mic': result['pkpd']['cmax_mic'],
                'auc_mic': result['pkpd']['auc_mic'],
                'p_cure': result['outcomes']['p_cure'],
                'p_aki': result['outcomes']['p_aki'],
                'target_cmax_mic': result['targets_met']['cmax_mic'],
                'target_auc_mic': result['targets_met']['auc_mic'],
                'target_trough': result['targets_met']['trough_safe'],
                'score': result['score']
            })

        results_df = pd.DataFrame(results)

        # Find optimal
        optimal_idx = results_df['score'].idxmax()
        optimal_dose = results_df.loc[optimal_idx, 'dose']
        optimal_score = results_df.loc[optimal_idx, 'score']

        print(f"✓ Optimization complete!")
        print(f"  Optimal dose: {optimal_dose:.0f} mg ({optimal_dose/weight:.1f} mg/kg)")
        print(f"  Optimization score: {optimal_score:.1f}/100")
        print()

        return optimal_dose, results_df

    def generate_report(self, weight, crcl, mic, optimal_dose, results_df):
        """Generate clinical recommendation report"""
        # Get optimal dose details
        optimal_result = self.evaluate_dose(weight, crcl, mic, optimal_dose)

        report = []
        report.append("="*80)
        report.append("PERSONALIZED AMINOGLYCOSIDE DOSING RECOMMENDATION")
        report.append("="*80)
        report.append("")

        # Patient characteristics
        report.append("PATIENT CHARACTERISTICS:")
        report.append(f"  Weight: {weight:.1f} kg")
        report.append(f"  Creatinine Clearance: {crcl:.1f} mL/min")
        if crcl >= 90:
            renal_status = "Normal"
        elif crcl >= 60:
            renal_status = "Mild impairment"
        elif crcl >= 30:
            renal_status = "Moderate impairment"
        else:
            renal_status = "Severe impairment"
        report.append(f"  Renal Function: {renal_status}")
        report.append(f"  Pathogen MIC: {mic:.2f} mg/L")
        report.append("")

        # Recommended dose
        report.append("RECOMMENDED DOSE:")
        report.append(f"  ✓ {optimal_dose:.0f} mg once daily (IV infusion over 30-60 min)")
        report.append(f"  ✓ Dose intensity: {optimal_result['dose_per_kg']:.1f} mg/kg")
        report.append("")

        # Predicted PK
        report.append("PREDICTED PHARMACOKINETICS:")
        report.append(f"  Peak concentration (Cmax): {optimal_result['pk']['cmax']:.1f} mg/L")
        report.append(f"  Trough concentration (24h): {optimal_result['pk']['cmin']:.2f} mg/L")
        report.append(f"  AUC (0-24h): {optimal_result['pk']['auc24']:.0f} mg·h/L")
        report.append(f"  Half-life: {optimal_result['pk']['t_half']:.1f} hours")
        report.append("")

        # PK/PD target attainment
        report.append("PK/PD TARGET ATTAINMENT:")
        report.append(f"  Cmax/MIC = {optimal_result['pkpd']['cmax_mic']:.1f}")
        status = "✅ ACHIEVED" if optimal_result['targets_met']['cmax_mic'] else "❌ SUBOPTIMAL"
        report.append(f"    Target ≥{self.targets['cmax_mic_min']} - {status}")

        report.append(f"  AUC/MIC = {optimal_result['pkpd']['auc_mic']:.0f}")
        status = "✅ ACHIEVED" if optimal_result['targets_met']['auc_mic'] else "❌ SUBOPTIMAL"
        report.append(f"    Target ≥{self.targets['auc_mic_min']} - {status}")

        report.append(f"  Trough = {optimal_result['pk']['cmin']:.2f} mg/L")
        status = "✅ SAFE" if optimal_result['targets_met']['trough_safe'] else "⚠️  ELEVATED RISK"
        report.append(f"    Safety limit <{self.targets['trough_max']} mg/L - {status}")
        report.append("")

        # Predicted outcomes
        report.append("PREDICTED CLINICAL OUTCOMES:")
        report.append(f"  Probability of clinical cure: {optimal_result['outcomes']['p_cure']*100:.0f}%")
        report.append(f"  Probability of nephrotoxicity: {optimal_result['outcomes']['p_aki']*100:.0f}%")
        report.append("")

        # Clinical interpretation
        report.append("CLINICAL INTERPRETATION:")
        if optimal_result['targets_met']['cmax_mic'] and optimal_result['targets_met']['auc_mic']:
            report.append("  ✅ EXCELLENT efficacy potential (both PK/PD targets achieved)")
        elif optimal_result['targets_met']['cmax_mic'] or optimal_result['targets_met']['auc_mic']:
            report.append("  ⚠️  MODERATE efficacy (partial target attainment)")
        else:
            report.append("  ❌ SUBOPTIMAL efficacy (consider higher dose or alternative agent)")

        if optimal_result['targets_met']['trough_safe']:
            report.append("  ✅ LOW nephrotoxicity risk (safe trough level)")
        else:
            report.append("  ⚠️  ELEVATED toxicity risk (enhanced monitoring required)")
        report.append("")

        # Monitoring
        report.append("THERAPEUTIC DRUG MONITORING:")
        report.append("  • Measure trough before 3rd or 4th dose")
        report.append("  • Target trough: <2 mg/L (once-daily dosing)")
        report.append("  • Measure peak 30-60 min after infusion (optional)")
        report.append("  • Monitor SCr daily × 3 days, then every 2-3 days")
        report.append("  • Adjust dose if CrCL changes >20% or trough elevated")
        report.append("")

        # Dose adjustment guidance
        report.append("DOSE ADJUSTMENT GUIDANCE:")
        report.append("  If trough >2 mg/L:")
        report.append("    → Decrease dose by 20-25%")
        report.append("    → Consider extending interval to 36-48h")
        report.append("  If targets not achieved AND trough safe:")
        report.append("    → Increase dose by 20-25%")
        report.append("  If CrCL decreases:")
        report.append("    → Reduce dose proportionally or extend interval")
        report.append("")

        report.append("="*80)
        report.append("Based on ML model findings:")
        report.append("  • AUC/MIC is #1 predictor of clinical cure (6.2% importance)")
        report.append("  • Cmax is #2 predictor (5.8% importance)")
        report.append("  • Cmax/MIC is #3 predictor (5.5% importance)")
        report.append("="*80)
        report.append("Generated by: Aminoglycoside QSP-ML Framework v2.0")
        report.append("="*80)

        return "\n".join(report)

    def plot_dose_response(self, results_df, weight, crcl, mic, save_path=None):
        """Plot comprehensive dose-response curves"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Plot 1: Cmax/MIC
        ax = axes[0, 0]
        ax.plot(results_df['dose'], results_df['cmax_mic'], 'o-', linewidth=2, markersize=4)
        ax.axhline(self.targets['cmax_mic_min'], color='red', linestyle='--',
                   label=f'Target (≥{self.targets["cmax_mic_min"]})', linewidth=2)
        ax.fill_between(results_df['dose'],
                        self.targets['cmax_mic_min'],
                        results_df['cmax_mic'].max() * 1.1,
                        where=results_df['cmax_mic'] >= self.targets['cmax_mic_min'],
                        alpha=0.2, color='green', label='Target zone')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Cmax/MIC Ratio', fontsize=11)
        ax.set_title('Top ML Predictor: Cmax/MIC (6.2% importance)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: AUC/MIC
        ax = axes[0, 1]
        ax.plot(results_df['dose'], results_df['auc_mic'], 'o-', color='green', linewidth=2, markersize=4)
        ax.axhline(self.targets['auc_mic_min'], color='red', linestyle='--',
                   label=f'Target (≥{self.targets["auc_mic_min"]})', linewidth=2)
        ax.fill_between(results_df['dose'],
                        self.targets['auc_mic_min'],
                        results_df['auc_mic'].max() * 1.1,
                        where=results_df['auc_mic'] >= self.targets['auc_mic_min'],
                        alpha=0.2, color='green', label='Target zone')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('AUC/MIC Ratio', fontsize=11)
        ax.set_title('2nd ML Predictor: AUC/MIC (5.8% importance)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 3: Trough concentration
        ax = axes[0, 2]
        ax.plot(results_df['dose'], results_df['cmin'], 'o-', color='orange', linewidth=2, markersize=4)
        ax.axhline(self.targets['trough_max'], color='red', linestyle='--',
                   label=f'Safety limit (<{self.targets["trough_max"]} mg/L)', linewidth=2)
        ax.fill_between(results_df['dose'],
                        0,
                        self.targets['trough_max'],
                        alpha=0.2, color='green', label='Safe zone')
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Trough Concentration (mg/L)', fontsize=11)
        ax.set_title('Safety: Trough Concentration', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 4: Clinical cure probability
        ax = axes[1, 0]
        ax.plot(results_df['dose'], results_df['p_cure'] * 100, 'o-',
                color='darkblue', linewidth=2, markersize=4)
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Probability of Cure (%)', fontsize=11)
        ax.set_title('Predicted Efficacy', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)

        # Plot 5: Nephrotoxicity probability
        ax = axes[1, 1]
        ax.plot(results_df['dose'], results_df['p_aki'] * 100, 'o-',
                color='darkred', linewidth=2, markersize=4)
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Probability of Nephrotoxicity (%)', fontsize=11)
        ax.set_title('Predicted Safety', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)

        # Plot 6: Overall score
        ax = axes[1, 2]
        ax.plot(results_df['dose'], results_df['score'], 'o-',
                color='purple', linewidth=2, markersize=4)
        optimal_dose = results_df.loc[results_df['score'].idxmax(), 'dose']
        optimal_score = results_df['score'].max()
        ax.axvline(optimal_dose, color='red', linestyle='--',
                   label=f'Optimal: {optimal_dose:.0f} mg', linewidth=2)
        ax.scatter([optimal_dose], [optimal_score], color='red', s=200, zorder=5,
                  marker='*', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('Dose (mg)', fontsize=11)
        ax.set_ylabel('Optimization Score (0-100)', fontsize=11)
        ax.set_title('Overall Dose Optimization', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Personalized Dose Optimization: {weight:.0f} kg, CrCL {crcl:.0f} mL/min, MIC {mic} mg/L',
                    fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dose-response plots saved: {save_path}")

        return fig


def demo_example_1():
    """Example 1: Standard adult with normal renal function"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Standard Adult with Normal Renal Function")
    print("="*80 + "\n")

    optimizer = SimpleDoseOptimizer()

    # Patient parameters
    weight = 70      # kg
    crcl = 100       # mL/min (normal)
    mic = 2.0        # mg/L (susceptible pathogen)

    # Optimize
    optimal_dose, results = optimizer.optimize_dose(weight, crcl, mic, n_doses=40)

    # Report
    report = optimizer.generate_report(weight, crcl, mic, optimal_dose, results)
    print(report)
    print()

    # Plot
    fig = optimizer.plot_dose_response(results, weight, crcl, mic,
                                       save_path='results/demo_standard_patient.png')

    return optimizer, optimal_dose, results


def demo_example_2():
    """Example 2: Obese patient with renal impairment"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Obese Patient with Moderate Renal Impairment")
    print("="*80 + "\n")

    optimizer = SimpleDoseOptimizer()

    # Patient parameters
    weight = 120     # kg (obese)
    crcl = 45        # mL/min (moderate impairment)
    mic = 4.0        # mg/L (higher MIC - less susceptible)

    # Optimize
    optimal_dose, results = optimizer.optimize_dose(weight, crcl, mic, n_doses=40)

    # Report
    report = optimizer.generate_report(weight, crcl, mic, optimal_dose, results)
    print(report)
    print()

    # Plot
    fig = optimizer.plot_dose_response(results, weight, crcl, mic,
                                       save_path='results/demo_obese_renal_impairment.png')

    return optimizer, optimal_dose, results


def demo_example_3():
    """Example 3: Low weight elderly with preserved function"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Elderly Patient (Low Weight, Preserved Renal Function)")
    print("="*80 + "\n")

    optimizer = SimpleDoseOptimizer()

    # Patient parameters
    weight = 55      # kg (lower weight)
    crcl = 75        # mL/min (good for age)
    mic = 1.0        # mg/L (very susceptible)

    # Optimize
    optimal_dose, results = optimizer.optimize_dose(weight, crcl, mic, n_doses=40)

    # Report
    report = optimizer.generate_report(weight, crcl, mic, optimal_dose, results)
    print(report)
    print()

    # Plot
    fig = optimizer.plot_dose_response(results, weight, crcl, mic,
                                       save_path='results/demo_elderly_patient.png')

    return optimizer, optimal_dose, results


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("PERSONALIZED AMINOGLYCOSIDE DOSE OPTIMIZATION")
    print("Using ML-Derived PK/PD Principles")
    print("="*80)
    print("\nThis tool uses insights from the QSP-ML framework ML models:")
    print("  • Top predictor: AUC/MIC (6.2% importance)")
    print("  • 2nd predictor: Cmax (5.8% importance)")
    print("  • 3rd predictor: Cmax/MIC (5.5% importance)")
    print("\nDemonstrating 3 patient scenarios...\n")

    # Create output directory
    Path('results').mkdir(exist_ok=True)

    # Run examples
    opt1, dose1, res1 = demo_example_1()
    opt2, dose2, res2 = demo_example_2()
    opt3, dose3, res3 = demo_example_3()

    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nDose Recommendations Summary:")
    print(f"  Example 1 (70 kg, CrCL 100): {dose1:.0f} mg ({dose1/70:.1f} mg/kg)")
    print(f"  Example 2 (120 kg, CrCL 45): {dose2:.0f} mg ({dose2/120:.1f} mg/kg)")
    print(f"  Example 3 (55 kg, CrCL 75): {dose3:.0f} mg ({dose3/55:.1f} mg/kg)")
    print("\nGenerated visualizations:")
    print("  • results/demo_standard_patient.png")
    print("  • results/demo_obese_renal_impairment.png")
    print("  • results/demo_elderly_patient.png")
    print("\n" + "="*80)
    print("You can modify patient parameters to optimize for your own cases!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
