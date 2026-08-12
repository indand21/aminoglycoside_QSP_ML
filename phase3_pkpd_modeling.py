#!/usr/bin/env python3
"""
Phase 3: PK/PD Modeling and Target Attainment Analysis

This script:
1. Calculates PK/PD indices (Cmax, AUC, Cmax/MIC, AUC/MIC)
2. Performs target attainment analysis (PTA, CFR)
3. Links PK/PD indices to clinical outcomes
4. Generates visualizations and reports

Author: Aminoglycoside QSP-ML Project
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PKPDAnalyzer:
    """
    Pharmacokinetic/Pharmacodynamic Analysis for Aminoglycosides
    """

    def __init__(self, data_path='data/processed/ml_dataset.csv',
                 conc_path='data/concentrations.csv',
                 dose_path='data/dosing.csv',
                 patient_path='data/patient_data.csv'):
        """
        Initialize PK/PD Analyzer

        Parameters:
        -----------
        data_path : str
            Path to ML dataset from Phase 1
        conc_path : str
            Path to concentration-time data
        dose_path : str
            Path to dosing data
        patient_path : str
            Path to patient demographics
        """
        print("="*80)
        print("Phase 3: PK/PD Modeling and Target Attainment Analysis")
        print("="*80)
        print()

        # Load data
        print("Loading data...")
        self.ml_data = pd.read_csv(data_path)
        self.conc_data = pd.read_csv(conc_path)
        self.dose_data = pd.read_csv(dose_path)
        self.patient_data = pd.read_csv(patient_path)

        print(f"  [OK] ML dataset: {self.ml_data.shape}")
        print(f"  [OK] Concentration data: {self.conc_data.shape}")
        print(f"  [OK] Dosing data: {self.dose_data.shape}")
        print(f"  [OK] Patient data: {self.patient_data.shape}")
        print()

        # Create output directories
        self.results_dir = Path('results/phase3_pkpd')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.pkpd_indices = None
        self.pta_results = None
        self.outcome_models = {}

    def two_compartment_ode(self, y, t, CL, Vc, Q, Vp, dose_rate):
        """
        Two-compartment PK model ODEs

        Parameters:
        -----------
        y : array
            [A_central, A_peripheral] - amounts in compartments
        t : float
            Time (hours)
        CL : float
            Clearance (L/h)
        Vc : float
            Central volume (L)
        Q : float
            Intercompartmental clearance (L/h)
        Vp : float
            Peripheral volume (L)
        dose_rate : float
            Infusion rate (mg/h)

        Returns:
        --------
        dydt : array
            Derivatives [dA_central/dt, dA_peripheral/dt]
        """
        A_central, A_peripheral = y

        dA_central = (dose_rate -
                     (CL/Vc) * A_central -
                     (Q/Vc) * A_central +
                     (Q/Vp) * A_peripheral)

        dA_peripheral = (Q/Vc) * A_central - (Q/Vp) * A_peripheral

        return [dA_central, dA_peripheral]

    def simulate_concentration_profile(self, dose, interval, CL, Vc, Q, Vp,
                                      infusion_time=1.0, n_doses=3):
        """
        Simulate concentration-time profile for multiple doses

        Parameters:
        -----------
        dose : float
            Dose amount (mg)
        interval : float
            Dosing interval (hours)
        CL, Vc, Q, Vp : float
            PK parameters
        infusion_time : float
            Duration of infusion (hours)
        n_doses : int
            Number of doses to simulate

        Returns:
        --------
        times : array
            Time points
        concentrations : array
            Concentration at each time point
        """
        # Time points
        total_time = interval * n_doses
        times = np.linspace(0, total_time, 1000)

        # Initial conditions
        y0 = [0, 0]  # Start with zero drug

        concentrations = np.zeros_like(times)

        for i, t in enumerate(times):
            # Determine current dose number and time since last dose
            dose_num = int(t / interval)
            time_since_dose = t - dose_num * interval

            if dose_num >= n_doses:
                dose_rate = 0
            elif time_since_dose < infusion_time:
                dose_rate = dose / infusion_time
            else:
                dose_rate = 0

            # Simulate from start to current time
            t_span = np.linspace(0, t, int(t*10)+1)

            # Track dose rates for each segment
            solution_segments = []
            current_y = y0.copy()

            for dose_idx in range(min(dose_num + 1, n_doses)):
                dose_start = dose_idx * interval
                dose_end = dose_start + infusion_time
                post_infusion_end = (dose_idx + 1) * interval

                # During infusion
                if dose_start <= t:
                    t_infusion = np.linspace(dose_start,
                                            min(dose_end, t),
                                            max(2, int((min(dose_end, t) - dose_start)*10)))
                    sol = odeint(self.two_compartment_ode, current_y, t_infusion,
                               args=(CL, Vc, Q, Vp, dose/infusion_time))
                    current_y = sol[-1]

                # After infusion
                if dose_end < t and dose_end < post_infusion_end:
                    t_post = np.linspace(dose_end,
                                        min(post_infusion_end, t),
                                        max(2, int((min(post_infusion_end, t) - dose_end)*10)))
                    sol = odeint(self.two_compartment_ode, current_y, t_post,
                               args=(CL, Vc, Q, Vp, 0))
                    current_y = sol[-1]

            concentrations[i] = current_y[0] / Vc

        return times, concentrations

    def calculate_pk_indices(self, dose, interval, CL, Vc, Q, Vp, infusion_time=1.0):
        """
        Calculate PK indices for a dosing regimen

        Returns:
        --------
        dict with Cmax, Cmin, AUC24, t_half
        """
        # Simulate until steady state (assume reached by dose 3)
        times, conc = self.simulate_concentration_profile(
            dose, interval, CL, Vc, Q, Vp, infusion_time, n_doses=5
        )

        # Focus on last dosing interval (steady state)
        last_interval_mask = times >= 4 * interval
        times_ss = times[last_interval_mask] - 4 * interval
        conc_ss = conc[last_interval_mask]

        # Calculate indices
        Cmax = np.max(conc_ss)
        Cmin = conc_ss[-1]  # Trough at end of interval

        # AUC using trapezoidal rule
        AUC_interval = np.trapz(conc_ss, times_ss)
        AUC24 = AUC_interval * (24 / interval)  # Normalize to 24h

        # Half-life (terminal phase approximation)
        # Find elimination phase (last 50% of interval)
        elim_phase_idx = len(conc_ss) // 2
        conc_elim = conc_ss[elim_phase_idx:]
        times_elim = times_ss[elim_phase_idx:]

        if len(conc_elim) > 2 and np.all(conc_elim > 0):
            log_conc = np.log(conc_elim)
            # Linear regression for elimination rate
            slope = np.polyfit(times_elim, log_conc, 1)[0]
            k_el = -slope
            t_half = 0.693 / k_el if k_el > 0 else np.nan
        else:
            t_half = np.nan

        return {
            'Cmax': Cmax,
            'Cmin': Cmin,
            'AUC24': AUC24,
            't_half': t_half,
            'concentration_profile': (times_ss, conc_ss)
        }

    def calculate_all_pk_indices(self):
        """
        Calculate PK indices for all patients using observed concentration data
        """
        print("Calculating PK indices for all patients...")

        results = []

        for idx, row in self.ml_data.iterrows():
            patient_id = row['patient_id']

            # Get this patient's concentration data
            patient_conc = self.conc_data[self.conc_data['patient_id'] == patient_id].copy()

            if len(patient_conc) == 0:
                print(f"  Warning: No concentration data for patient {patient_id}")
                continue

            # Sort by time
            patient_conc = patient_conc.sort_values('time')

            # Calculate PK indices from observed concentrations
            Cmax = patient_conc['concentration'].max()
            Cmin = patient_conc['concentration'].min()  # Approximate trough as minimum

            # Calculate AUC using trapezoidal rule
            # Use concentrations from the last dosing interval for steady-state approximation
            times = patient_conc['time'].values
            concs = patient_conc['concentration'].values

            # Find last dosing interval (approximate as last 24 hours)
            max_time = times.max()
            if max_time >= 24:
                last_interval_mask = times >= (max_time - 24)
                times_interval = times[last_interval_mask]
                concs_interval = concs[last_interval_mask]
            else:
                times_interval = times
                concs_interval = concs

            # Calculate AUC for the interval
            if len(times_interval) > 1:
                auc_interval = np.trapz(concs_interval, times_interval)
                # Normalize to 24h if interval is different
                interval_duration = times_interval[-1] - times_interval[0]
                if interval_duration > 0:
                    AUC24 = auc_interval * (24 / interval_duration)
                else:
                    AUC24 = 0
            else:
                AUC24 = 0

            # Get dosing info
            patient_doses = self.dose_data[self.dose_data['patient_id'] == patient_id]
            if len(patient_doses) > 0:
                dose = patient_doses['dose'].iloc[0]  # First dose
                mean_dose = patient_doses['dose'].mean()
            else:
                dose = row['first_dose'] if 'first_dose' in row else np.nan
                mean_dose = row['mean_dose'] if 'mean_dose' in row else np.nan

            # Get drug-specific MIC for this patient
            drug = row.get('drug', 'gentamicin')
            mic = row['mic_amikacin'] if drug == 'amikacin' else row['mic_gentamicin']

            # Drug-specific safety trough threshold, matching the data-generating
            # model (gentamicin 2.0 mg/L, amikacin 2.5 mg/L).
            trough_threshold = 2.5 if drug == 'amikacin' else 2.0

            # Calculate PK/PD indices
            results.append({
                'patient_id': patient_id,
                'dose': dose,
                'mean_dose': mean_dose,
                'Cmax': Cmax,
                'Cmin': Cmin,
                'AUC24': AUC24,
                'MIC': mic,
                'Cmax_MIC': Cmax / mic if mic > 0 else np.nan,
                'AUC_MIC': AUC24 / mic if mic > 0 else np.nan,
                'trough_target': 1 if Cmin < trough_threshold else 0,  # trough below drug-specific safety threshold
                'cmax_target': 1 if (Cmax / mic) >= 8 else 0,  # Target: Cmax/MIC ≥8
                'auc_target': 1 if (AUC24 / mic) >= 80 else 0,  # Target: AUC/MIC ≥80
                'clinical_cure': row['clinical_cure'],
                'nephrotoxicity': row['aki_stage']
            })

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(self.ml_data)} patients...")

        self.pkpd_indices = pd.DataFrame(results)

        print(f"\n[OK] Calculated PK/PD indices for {len(self.pkpd_indices)} patients")
        print(f"\nSummary statistics:")
        print(f"  Cmax/MIC:  {self.pkpd_indices['Cmax_MIC'].mean():.2f} +/- {self.pkpd_indices['Cmax_MIC'].std():.2f}")
        print(f"  AUC/MIC:   {self.pkpd_indices['AUC_MIC'].mean():.2f} +/- {self.pkpd_indices['AUC_MIC'].std():.2f}")
        print(f"  Cmax (mg/L): {self.pkpd_indices['Cmax'].mean():.2f} +/- {self.pkpd_indices['Cmax'].std():.2f}")
        print(f"  Cmin (mg/L): {self.pkpd_indices['Cmin'].mean():.2f} +/- {self.pkpd_indices['Cmin'].std():.2f}")
        print(f"\nTarget attainment:")
        print(f"  Cmax/MIC >=8:  {self.pkpd_indices['cmax_target'].mean()*100:.1f}%")
        print(f"  AUC/MIC >=80:  {self.pkpd_indices['auc_target'].mean()*100:.1f}%")
        print(f"  Trough below safety threshold: {self.pkpd_indices['trough_target'].mean()*100:.1f}%")
        print()

        # Save results (will be updated with time-based features later)
        output_file = self.results_dir / 'pkpd_indices.csv'
        self.pkpd_indices.to_csv(output_file, index=False)
        print(f"[OK] Saved PK/PD indices to: {output_file}")
        print()

    def calculate_time_based_pd_features(self):
        """
        Calculate time-based pharmacodynamic features

        Adds the following features to pkpd_indices:
        - time_above_MIC: Time (hours) concentration is above MIC in dosing interval
        - time_above_MIC_percent: Percentage of dosing interval above MIC
        - AUC_above_MIC: Area between concentration curve and MIC (AUBC)
        - fluctuation_index: (Cmax - Cmin) / Cmin
        - time_to_peak: Time to reach Cmax (hours)
        - peak_trough_ratio: Cmax / Cmin

        These features capture the time-dependent killing characteristics of aminoglycosides
        """
        print("\n" + "="*80)
        print("CALCULATING TIME-BASED PHARMACODYNAMIC FEATURES")
        print("="*80)

        if self.pkpd_indices is None:
            raise ValueError("Must run calculate_all_pk_indices() first")

        print(f"\nProcessing {len(self.pkpd_indices)} patients...")

        # Initialize new columns
        time_above_mic = []
        time_above_mic_percent = []
        auc_above_mic = []
        fluctuation_index = []
        time_to_peak = []
        peak_trough_ratio = []

        for idx, row in self.pkpd_indices.iterrows():
            patient_id = row['patient_id']
            mic = row['MIC']

            # Get concentration-time data for this patient
            patient_conc = self.conc_data[self.conc_data['patient_id'] == patient_id].copy()

            if len(patient_conc) == 0:
                # No concentration data - use NaN
                time_above_mic.append(np.nan)
                time_above_mic_percent.append(np.nan)
                auc_above_mic.append(np.nan)
                fluctuation_index.append(np.nan)
                time_to_peak.append(np.nan)
                peak_trough_ratio.append(np.nan)
                continue

            # Sort by time
            patient_conc = patient_conc.sort_values('time')
            times = patient_conc['time'].values
            concs = patient_conc['concentration'].values

            # Focus on last dosing interval (steady state)
            max_time = times.max()
            dosing_interval = 24  # Assume 24h dosing interval

            if max_time >= dosing_interval:
                last_interval_mask = times >= (max_time - dosing_interval)
                times_interval = times[last_interval_mask] - (max_time - dosing_interval)
                concs_interval = concs[last_interval_mask]
            else:
                times_interval = times
                concs_interval = concs
                dosing_interval = max_time if max_time > 0 else 24

            # === 1. Time above MIC ===
            # Interpolate to get finer time resolution
            time_fine = np.linspace(times_interval[0], times_interval[-1], 1000)
            conc_fine = np.interp(time_fine, times_interval, concs_interval)

            # Calculate time above MIC
            above_mic_mask = conc_fine > mic
            t_above_mic = np.sum(above_mic_mask) / len(time_fine) * (time_fine[-1] - time_fine[0])
            time_above_mic.append(t_above_mic)
            time_above_mic_percent.append((t_above_mic / dosing_interval) * 100)

            # === 2. AUC above MIC (AUBC - Area Under Bacterial Curve) ===
            # Calculate area between concentration curve and MIC line
            conc_above_mic = np.maximum(conc_fine - mic, 0)
            aubc = np.trapz(conc_above_mic, time_fine)
            auc_above_mic.append(aubc)

            # === 3. Fluctuation Index ===
            cmax = row['Cmax']
            cmin = row['Cmin']
            if cmin > 0:
                fluct_idx = (cmax - cmin) / cmin
            else:
                fluct_idx = np.nan
            fluctuation_index.append(fluct_idx)

            # === 4. Time to Peak ===
            peak_idx = np.argmax(concs_interval)
            t_peak = times_interval[peak_idx]
            time_to_peak.append(t_peak)

            # === 5. Peak-to-Trough Ratio ===
            if cmin > 0:
                pt_ratio = cmax / cmin
            else:
                pt_ratio = np.nan
            peak_trough_ratio.append(pt_ratio)

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(self.pkpd_indices)} patients...")

        # Add new features to pkpd_indices
        self.pkpd_indices['time_above_MIC'] = time_above_mic
        self.pkpd_indices['time_above_MIC_percent'] = time_above_mic_percent
        self.pkpd_indices['AUC_above_MIC'] = auc_above_mic
        self.pkpd_indices['fluctuation_index'] = fluctuation_index
        self.pkpd_indices['time_to_peak'] = time_to_peak
        self.pkpd_indices['peak_trough_ratio'] = peak_trough_ratio

        # Print summary statistics
        print(f"\n[OK] Calculated time-based PD features for {len(self.pkpd_indices)} patients")
        print("\nSummary Statistics:")
        print("-" * 80)
        print(f"Time above MIC (hours):     {np.nanmean(time_above_mic):.2f} +/- {np.nanstd(time_above_mic):.2f}")
        print(f"Time above MIC (%):         {np.nanmean(time_above_mic_percent):.1f}% +/- {np.nanstd(time_above_mic_percent):.1f}%")
        print(f"AUC above MIC (mg*h/L):     {np.nanmean(auc_above_mic):.2f} +/- {np.nanstd(auc_above_mic):.2f}")
        print(f"Fluctuation index:          {np.nanmean(fluctuation_index):.2f} +/- {np.nanstd(fluctuation_index):.2f}")
        print(f"Time to peak (hours):       {np.nanmean(time_to_peak):.2f} +/- {np.nanstd(time_to_peak):.2f}")
        print(f"Peak-to-trough ratio:       {np.nanmean(peak_trough_ratio):.2f} +/- {np.nanstd(peak_trough_ratio):.2f}")

        # Save updated results
        output_file = self.results_dir / 'pkpd_indices.csv'
        self.pkpd_indices.to_csv(output_file, index=False)
        print(f"\n[OK] Updated PK/PD indices saved to: {output_file}")
        print()

        return self.pkpd_indices

    def perform_pta_analysis(self, mic_range=None):
        """
        Probability of Target Attainment (PTA) analysis

        Parameters:
        -----------
        mic_range : array-like
            MIC values to evaluate (default: 0.25 to 16 mg/L)
        """
        print("Performing Probability of Target Attainment (PTA) analysis...")

        if mic_range is None:
            # Standard MIC doubling dilutions
            mic_range = np.array([0.25, 0.5, 1, 2, 4, 8, 16])

        pta_results = []

        # Group by dose levels
        dose_levels = self.pkpd_indices['dose'].unique()
        dose_levels.sort()

        for dose in dose_levels:
            dose_subset = self.pkpd_indices[self.pkpd_indices['dose'] == dose]

            for mic in mic_range:
                # Recalculate targets for this MIC
                cmax_mic = dose_subset['Cmax'] / mic
                auc_mic = dose_subset['AUC24'] / mic

                # PTA = proportion achieving target. Trough safety uses the
                # drug-specific threshold already encoded in 'trough_target'
                # (gentamicin 2.0 mg/L, amikacin 2.5 mg/L).
                trough_safe = dose_subset['trough_target'] == 1
                pta_cmax = (cmax_mic >= 8).mean()
                pta_auc = (auc_mic >= 80).mean()
                pta_trough = trough_safe.mean()

                # Combined PTA (all targets)
                pta_combined = ((cmax_mic >= 8) &
                              (auc_mic >= 80) &
                              trough_safe).mean()

                pta_results.append({
                    'dose': dose,
                    'MIC': mic,
                    'PTA_Cmax_MIC': pta_cmax,
                    'PTA_AUC_MIC': pta_auc,
                    'PTA_trough': pta_trough,
                    'PTA_combined': pta_combined,
                    'n_patients': len(dose_subset)
                })

        self.pta_results = pd.DataFrame(pta_results)

        print(f"[OK] PTA analysis complete")
        print(f"  Evaluated {len(dose_levels)} dose levels")
        print(f"  Across {len(mic_range)} MIC values")
        print()

        # Save results
        output_file = self.results_dir / 'pta_analysis.csv'
        self.pta_results.to_csv(output_file, index=False)
        print(f"[OK] Saved PTA results to: {output_file}")
        print()

        return self.pta_results

    def calculate_cfr(self, mic_distribution=None):
        """
        Cumulative Fraction of Response (CFR)

        Parameters:
        -----------
        mic_distribution : dict
            MIC distribution {MIC: probability}
            Default: Typical Gram-negative distribution
        """
        print("Calculating Cumulative Fraction of Response (CFR)...")

        if mic_distribution is None:
            # Typical MIC distribution for Gram-negative bacteria
            # (Example: E. coli, K. pneumoniae susceptibility)
            mic_distribution = {
                0.25: 0.05,
                0.5: 0.15,
                1: 0.30,
                2: 0.25,
                4: 0.15,
                8: 0.07,
                16: 0.03
            }

        # Normalize to ensure sum = 1
        total_prob = sum(mic_distribution.values())
        mic_distribution = {k: v/total_prob for k, v in mic_distribution.items()}

        cfr_results = []

        dose_levels = self.pta_results['dose'].unique()

        for dose in dose_levels:
            dose_pta = self.pta_results[self.pta_results['dose'] == dose]

            # CFR = sum(PTA(MIC) * P(MIC)) for all MICs
            cfr_cmax = sum(
                row['PTA_Cmax_MIC'] * mic_distribution.get(row['MIC'], 0)
                for _, row in dose_pta.iterrows()
            )

            cfr_auc = sum(
                row['PTA_AUC_MIC'] * mic_distribution.get(row['MIC'], 0)
                for _, row in dose_pta.iterrows()
            )

            cfr_combined = sum(
                row['PTA_combined'] * mic_distribution.get(row['MIC'], 0)
                for _, row in dose_pta.iterrows()
            )

            cfr_results.append({
                'dose': dose,
                'CFR_Cmax_MIC': cfr_cmax,
                'CFR_AUC_MIC': cfr_auc,
                'CFR_combined': cfr_combined
            })

        cfr_df = pd.DataFrame(cfr_results)

        print("[OK] CFR calculation complete")
        print(f"\nCFR by dose level (Cmax/MIC >=8 target):")
        for _, row in cfr_df.iterrows():
            print(f"  {row['dose']:.0f} mg: {row['CFR_Cmax_MIC']*100:.1f}%")
        print()

        # Save results
        output_file = self.results_dir / 'cfr_analysis.csv'
        cfr_df.to_csv(output_file, index=False)
        print(f"[OK] Saved CFR results to: {output_file}")
        print()

        return cfr_df

    def link_pkpd_to_outcomes(self):
        """
        Link PK/PD indices to clinical outcomes using logistic regression
        """
        print("Linking PK/PD indices to clinical outcomes...")

        from scipy.optimize import minimize
        from scipy.special import expit  # logistic function

        # Model 1: Cmax/MIC -> Clinical Cure
        print("\n1. Clinical Cure Model (Cmax/MIC -> Cure)")

        X_cure = self.pkpd_indices['Cmax_MIC'].values
        y_cure = self.pkpd_indices['clinical_cure'].values

        # Logistic regression: P(cure) = 1 / (1 + exp(-(beta0 + beta1*Cmax/MIC)))
        def logistic_nll(params, X, y):
            """Negative log-likelihood for logistic regression"""
            beta0, beta1 = params
            p = expit(beta0 + beta1 * X)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Fit model
        result_cure = minimize(logistic_nll, x0=[0, 0], args=(X_cure, y_cure),
                              method='BFGS')
        beta0_cure, beta1_cure = result_cure.x

        # Calculate predictions
        cmax_mic_range = np.linspace(0, 25, 100)
        p_cure = expit(beta0_cure + beta1_cure * cmax_mic_range)

        print(f"  beta0 (intercept): {beta0_cure:.3f}")
        print(f"  beta1 (Cmax/MIC):  {beta1_cure:.3f}")
        print(f"  P(cure | Cmax/MIC=8):  {expit(beta0_cure + beta1_cure * 8):.1%}")
        print(f"  P(cure | Cmax/MIC=10): {expit(beta0_cure + beta1_cure * 10):.1%}")

        self.outcome_models['cure'] = {
            'beta0': beta0_cure,
            'beta1': beta1_cure,
            'cmax_mic_range': cmax_mic_range,
            'p_cure': p_cure
        }

        # Model 2: Trough -> Nephrotoxicity
        print("\n2. Nephrotoxicity Model (Trough -> AKI)")

        X_tox = self.pkpd_indices['Cmin'].values
        y_tox = (self.pkpd_indices['nephrotoxicity'] > 0).astype(int).values

        result_tox = minimize(logistic_nll, x0=[0, 0], args=(X_tox, y_tox),
                            method='BFGS')
        beta0_tox, beta1_tox = result_tox.x

        # Calculate predictions
        trough_range = np.linspace(0, 10, 100)
        p_tox = expit(beta0_tox + beta1_tox * trough_range)

        print(f"  beta0 (intercept): {beta0_tox:.3f}")
        print(f"  beta1 (trough):    {beta1_tox:.3f}")
        print(f"  P(AKI | trough=1):  {expit(beta0_tox + beta1_tox * 1):.1%}")
        print(f"  P(AKI | trough=2):  {expit(beta0_tox + beta1_tox * 2):.1%}")
        print(f"  P(AKI | trough=3):  {expit(beta0_tox + beta1_tox * 3):.1%}")

        self.outcome_models['toxicity'] = {
            'beta0': beta0_tox,
            'beta1': beta1_tox,
            'trough_range': trough_range,
            'p_tox': p_tox
        }

        # Save model coefficients
        model_summary = {
            'clinical_cure': {
                'predictors': 'Cmax/MIC',
                'beta0': float(beta0_cure),
                'beta1': float(beta1_cure),
                'interpretation': 'Higher Cmax/MIC increases cure probability'
            },
            'nephrotoxicity': {
                'predictors': 'Trough concentration',
                'beta0': float(beta0_tox),
                'beta1': float(beta1_tox),
                'interpretation': 'Higher trough increases AKI risk'
            }
        }

        output_file = self.results_dir / 'outcome_models.json'
        with open(output_file, 'w') as f:
            json.dump(model_summary, f, indent=2)

        print(f"\n[OK] Saved outcome models to: {output_file}")
        print()

        return self.outcome_models

    def generate_visualizations(self):
        """
        Generate comprehensive PK/PD visualizations
        """
        print("Generating visualizations...")

        # Set up figure style
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.size'] = 10

        # 1. PK/PD Indices Distribution
        print("  Creating PK/PD indices distributions...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(self.pkpd_indices['Cmax_MIC'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(8, color='red', linestyle='--', label='Target ≥8')
        axes[0, 0].set_xlabel('Cmax/MIC Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Cmax/MIC')
        axes[0, 0].legend()

        axes[0, 1].hist(self.pkpd_indices['AUC_MIC'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(80, color='red', linestyle='--', label='Target ≥80')
        axes[0, 1].set_xlabel('AUC/MIC Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of AUC/MIC')
        axes[0, 1].legend()

        axes[1, 0].hist(self.pkpd_indices['Cmax'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Cmax (mg/L)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Peak Concentration')

        axes[1, 1].hist(self.pkpd_indices['Cmin'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(2, color='red', linestyle='--', label='Safety limit <2')
        axes[1, 1].set_xlabel('Cmin (mg/L)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Trough Concentration')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.results_dir / 'pkpd_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. PTA Heatmap
        print("  Creating PTA heatmap...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Pivot for heatmap
        pta_pivot_cmax = self.pta_results.pivot(index='MIC', columns='dose',
                                                values='PTA_Cmax_MIC')
        pta_pivot_combined = self.pta_results.pivot(index='MIC', columns='dose',
                                                    values='PTA_combined')

        sns.heatmap(pta_pivot_cmax, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'PTA'},
                   annot_kws={'fontsize': 6})
        axes[0].set_title('PTA: Cmax/MIC ≥8', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Dose (mg)', fontsize=10)
        axes[0].set_ylabel('MIC (mg/L)', fontsize=10)
        axes[0].tick_params(axis='both', labelsize=8)

        sns.heatmap(pta_pivot_combined, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'PTA'},
                   annot_kws={'fontsize': 6})
        axes[1].set_title('PTA: All Targets (Efficacy + Safety)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Dose (mg)', fontsize=10)
        axes[1].set_ylabel('MIC (mg/L)', fontsize=10)
        axes[1].tick_params(axis='both', labelsize=8)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'pta_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Outcome relationships
        print("  Creating outcome relationship plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Cure vs Cmax/MIC
        axes[0].scatter(self.pkpd_indices['Cmax_MIC'],
                       self.pkpd_indices['clinical_cure'],
                       alpha=0.3, s=20)
        axes[0].plot(self.outcome_models['cure']['cmax_mic_range'],
                    self.outcome_models['cure']['p_cure'],
                    'r-', linewidth=2, label='Logistic fit')
        axes[0].axvline(8, color='gray', linestyle='--', alpha=0.5, label='Target')
        axes[0].set_xlabel('Cmax/MIC Ratio')
        axes[0].set_ylabel('Probability of Clinical Cure')
        axes[0].set_title('Efficacy: Cmax/MIC → Clinical Cure')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # AKI vs Trough
        axes[1].scatter(self.pkpd_indices['Cmin'],
                       (self.pkpd_indices['nephrotoxicity'] > 0).astype(int),
                       alpha=0.3, s=20)
        axes[1].plot(self.outcome_models['toxicity']['trough_range'],
                    self.outcome_models['toxicity']['p_tox'],
                    'r-', linewidth=2, label='Logistic fit')
        axes[1].axvline(2, color='gray', linestyle='--', alpha=0.5, label='Safety limit')
        axes[1].set_xlabel('Trough Concentration (mg/L)')
        axes[1].set_ylabel('Probability of Nephrotoxicity (AKI)')
        axes[1].set_title('Safety: Trough → Nephrotoxicity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'outcome_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Target attainment by dose
        print("  Creating target attainment summary...")
        fig, ax = plt.subplots(figsize=(10, 6))

        dose_summary = self.pkpd_indices.groupby('dose').agg({
            'cmax_target': 'mean',
            'auc_target': 'mean',
            'trough_target': 'mean'
        }).reset_index()

        x = np.arange(len(dose_summary))
        width = 0.25

        ax.bar(x - width, dose_summary['cmax_target']*100, width,
              label='Cmax/MIC ≥8', alpha=0.8)
        ax.bar(x, dose_summary['auc_target']*100, width,
              label='AUC/MIC ≥80', alpha=0.8)
        ax.bar(x + width, dose_summary['trough_target']*100, width,
              label='Trough <2', alpha=0.8)

        ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% target')
        ax.set_xlabel('Dose (mg)')
        ax.set_ylabel('Target Attainment (%)')
        ax.set_title('Target Attainment by Dose Level')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:.0f}" for d in dose_summary['dose']])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'target_attainment_by_dose.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[OK] All visualizations saved to: {self.results_dir}/")
        print()

    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("Generating summary report...")

        report = []
        report.append("="*80)
        report.append("Phase 3: PK/PD Modeling - Summary Report")
        report.append("="*80)
        report.append("")
        report.append(f"Analysis Date: 2025-11-15")
        report.append(f"Number of Patients: {len(self.pkpd_indices)}")
        report.append("")

        report.append("="*80)
        report.append("1. PK/PD INDICES SUMMARY")
        report.append("="*80)
        report.append("")

        summary_stats = self.pkpd_indices[['Cmax', 'Cmin', 'AUC24',
                                           'Cmax_MIC', 'AUC_MIC']].describe()
        report.append(summary_stats.to_string())
        report.append("")

        report.append("="*80)
        report.append("2. TARGET ATTAINMENT")
        report.append("="*80)
        report.append("")
        report.append(f"Cmax/MIC >=8:   {self.pkpd_indices['cmax_target'].mean()*100:.1f}% "
                     f"({self.pkpd_indices['cmax_target'].sum()}/{len(self.pkpd_indices)} patients)")
        report.append(f"AUC/MIC >=80:   {self.pkpd_indices['auc_target'].mean()*100:.1f}% "
                     f"({self.pkpd_indices['auc_target'].sum()}/{len(self.pkpd_indices)} patients)")
        report.append(f"Trough below safety threshold (gent 2.0/ami 2.5 mg/L): {self.pkpd_indices['trough_target'].mean()*100:.1f}% "
                     f"({self.pkpd_indices['trough_target'].sum()}/{len(self.pkpd_indices)} patients)")
        report.append("")

        combined_target = ((self.pkpd_indices['cmax_target'] == 1) &
                          (self.pkpd_indices['trough_target'] == 1)).mean()
        report.append(f"Combined (Efficacy + Safety): {combined_target*100:.1f}%")
        report.append("")

        report.append("="*80)
        report.append("3. OUTCOME MODELS")
        report.append("="*80)
        report.append("")

        report.append("Clinical Cure Model: P(cure) = 1/(1 + exp(-(beta0 + beta1*Cmax/MIC)))")
        report.append(f"  beta0 (intercept): {self.outcome_models['cure']['beta0']:.3f}")
        report.append(f"  beta1 (Cmax/MIC):  {self.outcome_models['cure']['beta1']:.3f}")
        report.append("")

        report.append("Nephrotoxicity Model: P(AKI) = 1/(1 + exp(-(beta0 + beta1*Trough)))")
        report.append(f"  beta0 (intercept): {self.outcome_models['toxicity']['beta0']:.3f}")
        report.append(f"  beta1 (Trough):    {self.outcome_models['toxicity']['beta1']:.3f}")
        report.append("")

        report.append("="*80)
        report.append("4. CLINICAL IMPLICATIONS")
        report.append("="*80)
        report.append("")

        # Calculate optimal dose recommendation
        dose_performance = self.pkpd_indices.groupby('dose').agg({
            'cmax_target': 'mean',
            'trough_target': 'mean',
            'clinical_cure': 'mean',
            'nephrotoxicity': lambda x: (x > 0).mean()
        })

        # Find dose with best combined performance
        dose_performance['score'] = (dose_performance['cmax_target'] * 0.4 +
                                     dose_performance['trough_target'] * 0.3 +
                                     dose_performance['clinical_cure'] * 0.2 -
                                     dose_performance['nephrotoxicity'] * 0.1)

        best_dose = dose_performance['score'].idxmax()

        report.append(f"Recommended Starting Dose: {best_dose:.0f} mg")
        report.append(f"  - Achieves Cmax/MIC >=8: {dose_performance.loc[best_dose, 'cmax_target']*100:.1f}%")
        report.append(f"  - Safe trough (below drug-specific threshold): {dose_performance.loc[best_dose, 'trough_target']*100:.1f}%")
        report.append(f"  - Clinical cure rate: {dose_performance.loc[best_dose, 'clinical_cure']*100:.1f}%")
        report.append(f"  - Nephrotoxicity rate: {dose_performance.loc[best_dose, 'nephrotoxicity']*100:.1f}%")
        report.append("")

        report.append("="*80)
        report.append("5. FILES GENERATED")
        report.append("="*80)
        report.append("")
        report.append(f"  - {self.results_dir}/pkpd_indices.csv")
        report.append(f"  - {self.results_dir}/pta_analysis.csv")
        report.append(f"  - {self.results_dir}/cfr_analysis.csv")
        report.append(f"  - {self.results_dir}/outcome_models.json")
        report.append(f"  - {self.results_dir}/pkpd_distributions.png")
        report.append(f"  - {self.results_dir}/pta_heatmap.png")
        report.append(f"  - {self.results_dir}/outcome_relationships.png")
        report.append(f"  - {self.results_dir}/target_attainment_by_dose.png")
        report.append("")

        report.append("="*80)
        report.append("Phase 3 Analysis Complete!")
        report.append("="*80)

        # Save report
        report_text = "\n".join(report)
        output_file = self.results_dir / 'PHASE3_SUMMARY.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n[OK] Summary report saved to: {output_file}")
        print()

    def integrate_individual_pk_parameters(self, trace_path='models/phase2_popk_trace_likelihood.nc'):
        """
        Extract and integrate individual PK parameters from Bayesian posterior

        Parameters:
        -----------
        trace_path : str
            Path to the Phase 2 Bayesian trace file

        This method:
        1. Extracts individual PK parameters (CL_i, Vc_i, Q_i, Vp_i) from Bayesian posterior
        2. Calculates derived parameters (Vss, half-lives, rate constants)
        3. Merges with pkpd_indices DataFrame
        4. Saves updated results
        """
        print("\n" + "="*80)
        print("INTEGRATING INDIVIDUAL PK PARAMETERS FROM BAYESIAN MODEL")
        print("="*80)

        # Check if trace file exists
        if not Path(trace_path).exists():
            print(f"\n[WARNING] Trace file not found: {trace_path}")
            print("  Skipping individual PK parameter extraction.")
            print("  This is optional - Phase 3 can continue without it.")
            return

        try:
            # Import the utility function
            from utils.extract_individual_pk import extract_individual_pk_parameters

            # Determine the patient-id ordering that matches the trace's patient
            # dimension. The PopPK model (refit_popk.py / phase2_population_pk.py)
            # filters to patients with >= 3 observations and indexes them in
            # ascending integer-ID order. Naively slicing the full 1500-patient
            # list would scramble the per-patient parameters.
            popk_df = pd.read_csv(Path(__file__).resolve().parent /
                                   'data' / 'processed' / 'popk_dataset.csv')
            _obs = popk_df[popk_df['EVID'] == 0]
            _counts = _obs.groupby('ID').size()
            _keep_ids = sorted(_counts[_counts >= 3].index.tolist())
            _id_to_pid = (popk_df[['patient_id', 'ID']]
                           .drop_duplicates()
                           .set_index('ID')['patient_id'])
            patient_ids = [_id_to_pid.loc[i] for i in _keep_ids]

            # Extract individual PK parameters
            output_path = self.results_dir / 'individual_pk_parameters.csv'
            individual_pk = extract_individual_pk_parameters(
                trace_path=trace_path,
                patient_ids=patient_ids,
                output_path=str(output_path)
            )

            # Merge with pkpd_indices
            print("\nMerging individual PK parameters with PK/PD indices...")
            self.pkpd_indices = self.pkpd_indices.merge(
                individual_pk,
                on='patient_id',
                how='left'
            )

            # Save updated pkpd_indices
            output_file = self.results_dir / 'pkpd_indices.csv'
            self.pkpd_indices.to_csv(output_file, index=False)
            print(f"[OK] Updated PK/PD indices with individual PK parameters")
            print(f"[OK] Saved to: {output_file}")

        except ImportError as e:
            print(f"\n[WARNING] Could not import extract_individual_pk: {e}")
            print("  Skipping individual PK parameter extraction.")
        except Exception as e:
            print(f"\n[WARNING] Failed to extract individual PK parameters: {e}")
            print("  Continuing without individual PK parameters.")

    def run_complete_analysis(self, extract_individual_pk=True):
        """
        Run complete Phase 3 analysis pipeline

        Parameters:
        -----------
        extract_individual_pk : bool
            Whether to extract individual PK parameters from Bayesian model (default: True)
        """
        print("Starting Phase 3 complete analysis pipeline...")
        print()

        # Step 1: Calculate PK/PD indices
        self.calculate_all_pk_indices()

        # Step 2: Calculate time-based PD features (NEW)
        self.calculate_time_based_pd_features()

        # Step 3: Integrate individual PK parameters from Bayesian model (NEW)
        if extract_individual_pk:
            self.integrate_individual_pk_parameters()

        # Step 4: PTA analysis
        self.perform_pta_analysis()

        # Step 5: CFR calculation
        self.calculate_cfr()

        # Step 6: Link to outcomes
        self.link_pkpd_to_outcomes()

        # Step 7: Generate visualizations
        self.generate_visualizations()

        # Step 8: Generate summary report
        self.generate_summary_report()

        print("="*80)
        print("[OK] PHASE 3 COMPLETE!")
        print("="*80)
        print()
        print("Next step: Phase 4 - Machine Learning")
        print("  Run: python3 phase4_machine_learning_enhanced.py")
        print()


def main():
    """Main execution function"""

    # Initialize analyzer
    analyzer = PKPDAnalyzer()

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
