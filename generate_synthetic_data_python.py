#!/usr/bin/env python3
"""
Synthetic Data Generation for Indian ICU Aminoglycoside Study

This script generates scientifically valid synthetic data reflecting
real-world aminoglycoside use in Indian ICU settings.

Author: Generated for aminoglycoside QSP-ML project
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import os
import pickle
import json

# Set random seed for reproducibility
np.random.seed(12345)

# ============================================================================
# POPULATION CHARACTERISTICS - INDIAN ICU SETTINGS
# ============================================================================

INDIAN_ICU_PARAMS = {
    # Demographics
    'age_mean': 52,
    'age_sd': 16,
    'weight_mean': 62,
    'weight_sd': 12,
    'height_mean': 162,
    'height_sd': 10,
    'male_proportion': 0.65,

    # Clinical severity
    'apache_ii_mean': 22,
    'apache_ii_sd': 8,
    'sofa_mean': 9,
    'sofa_sd': 4,

    # Infection sites
    'infection_sites': ['bloodstream', 'pneumonia', 'uti', 'iai', 'other'],
    'infection_probs': [0.30, 0.35, 0.15, 0.12, 0.08],

    # Pathogens
    'pathogens': ['klebsiella', 'pseudomonas', 'e_coli', 'acinetobacter', 'other'],
    'pathogen_probs': [0.35, 0.25, 0.20, 0.15, 0.05],

    # MIC distribution (log-normal)
    'mic_gentamicin_meanlog': np.log(4),
    'mic_gentamicin_sdlog': 1.2,
    'mic_amikacin_meanlog': np.log(8),
    'mic_amikacin_sdlog': 1.3,

    # Comorbidities
    'diabetes_prevalence': 0.35,
    'ckd_prevalence': 0.25,
    'mechanical_ventilation': 0.65,
    'vasopressor_use': 0.50,
    'rrt_rate': 0.18,

    # Renal function
    'crcl_normal_mean': 85,
    'crcl_normal_sd': 25,
    'crcl_arc_proportion': 0.25,
    'crcl_arc_mean': 160,
    'crcl_arc_sd': 35,

    # Fluid balance
    'fluid_balance_mean': 2.5,
    'fluid_balance_sd': 2.0
}


def generate_patient_demographics(n_patients=300):
    """Generate patient demographics and baseline characteristics."""

    print("Step 1/5: Generating patient demographics and baseline data...")

    # Basic demographics
    patient_data = pd.DataFrame({
        'patient_id': [f'IND_{i:04d}' for i in range(1, n_patients + 1)],
        'age': np.clip(np.random.normal(INDIAN_ICU_PARAMS['age_mean'],
                                        INDIAN_ICU_PARAMS['age_sd'],
                                        n_patients), 18, 90).round(1),
        'sex': np.random.choice(['M', 'F'], n_patients,
                               p=[INDIAN_ICU_PARAMS['male_proportion'],
                                  1 - INDIAN_ICU_PARAMS['male_proportion']]),
        'weight': np.clip(np.random.normal(INDIAN_ICU_PARAMS['weight_mean'],
                                          INDIAN_ICU_PARAMS['weight_sd'],
                                          n_patients), 35, 120).round(1),
        'height': np.clip(np.random.normal(INDIAN_ICU_PARAMS['height_mean'],
                                          INDIAN_ICU_PARAMS['height_sd'],
                                          n_patients), 140, 190).round(1)
    })

    # Calculate BMI
    patient_data['bmi'] = (patient_data['weight'] /
                          (patient_data['height'] / 100) ** 2).round(1)

    # Clinical severity
    patient_data['apache_ii'] = np.clip(
        np.random.normal(INDIAN_ICU_PARAMS['apache_ii_mean'],
                        INDIAN_ICU_PARAMS['apache_ii_sd'],
                        n_patients), 5, 45).round(0).astype(int)

    patient_data['sofa_score'] = np.clip(
        np.random.normal(INDIAN_ICU_PARAMS['sofa_mean'],
                        INDIAN_ICU_PARAMS['sofa_sd'],
                        n_patients), 2, 20).round(0).astype(int)

    patient_data['sepsis_type'] = np.random.choice(
        ['sepsis', 'severe sepsis', 'septic shock'],
        n_patients, p=[0.25, 0.35, 0.40]
    )

    # Renal function (bimodal distribution)
    renal_group = np.random.choice(['normal', 'arc'], n_patients,
                                   p=[1 - INDIAN_ICU_PARAMS['crcl_arc_proportion'],
                                      INDIAN_ICU_PARAMS['crcl_arc_proportion']])

    baseline_crcl = np.where(
        renal_group == 'normal',
        np.random.normal(INDIAN_ICU_PARAMS['crcl_normal_mean'],
                        INDIAN_ICU_PARAMS['crcl_normal_sd'],
                        n_patients),
        np.random.normal(INDIAN_ICU_PARAMS['crcl_arc_mean'],
                        INDIAN_ICU_PARAMS['crcl_arc_sd'],
                        n_patients)
    )
    patient_data['baseline_crcl'] = np.clip(baseline_crcl, 15, 250).round(1)

    # Calculate serum creatinine (reverse Cockcroft-Gault)
    sex_factor = np.where(patient_data['sex'] == 'M', 1, 0.85)
    patient_data['baseline_scr'] = (
        ((140 - patient_data['age']) * patient_data['weight'] * sex_factor) /
        (72 * patient_data['baseline_crcl'])
    ).round(2)

    # eGFR approximation
    patient_data['baseline_egfr'] = np.clip(
        (140 - patient_data['age']) * patient_data['weight'] /
        (patient_data['baseline_scr'] * 72) * sex_factor,
        10, 180
    ).round(1)

    # Other baseline labs
    patient_data['baseline_albumin'] = np.clip(
        np.random.normal(2.8, 0.6, n_patients), 1.5, 4.5
    ).round(1)

    patient_data['baseline_bilirubin'] = np.clip(
        stats.lognorm.rvs(0.6, scale=np.exp(np.log(1.2)), size=n_patients),
        0.3, 15
    ).round(1)

    # Infection characteristics
    patient_data['infection_site'] = np.random.choice(
        INDIAN_ICU_PARAMS['infection_sites'],
        n_patients,
        p=INDIAN_ICU_PARAMS['infection_probs']
    )

    patient_data['pathogen'] = np.random.choice(
        INDIAN_ICU_PARAMS['pathogens'],
        n_patients,
        p=INDIAN_ICU_PARAMS['pathogen_probs']
    )

    # MIC values
    mic_gent_base = stats.lognorm.rvs(
        INDIAN_ICU_PARAMS['mic_gentamicin_sdlog'],
        scale=np.exp(INDIAN_ICU_PARAMS['mic_gentamicin_meanlog']),
        size=n_patients
    )

    mic_ami_base = stats.lognorm.rvs(
        INDIAN_ICU_PARAMS['mic_amikacin_sdlog'],
        scale=np.exp(INDIAN_ICU_PARAMS['mic_amikacin_meanlog']),
        size=n_patients
    )

    # Adjust for resistant organisms
    mic_mult_gent = np.where(patient_data['pathogen'] == 'acinetobacter', 2.0,
                             np.where(patient_data['pathogen'] == 'pseudomonas', 1.5, 1.0))
    mic_mult_ami = np.where(patient_data['pathogen'] == 'acinetobacter', 2.0,
                           np.where(patient_data['pathogen'] == 'pseudomonas', 1.5, 1.0))

    patient_data['mic_gentamicin'] = np.clip(
        mic_gent_base * mic_mult_gent, 0.25, 128
    ).round(2)

    patient_data['mic_amikacin'] = np.clip(
        mic_ami_base * mic_mult_ami, 0.5, 256
    ).round(2)

    # Comorbidities
    patient_data['diabetes'] = np.random.binomial(
        1, INDIAN_ICU_PARAMS['diabetes_prevalence'], n_patients
    ).astype(bool)

    # CKD stage based on eGFR
    patient_data['ckd_stage'] = pd.cut(
        patient_data['baseline_egfr'],
        bins=[0, 15, 30, 45, 60, 90, 200],
        labels=['4', '3b', '3a', '2', '1', '0'],
        include_lowest=True
    ).astype(str)

    patient_data['mechanical_ventilation'] = np.random.binomial(
        1, INDIAN_ICU_PARAMS['mechanical_ventilation'], n_patients
    ).astype(bool)

    patient_data['vasopressor_use'] = np.random.binomial(
        1, INDIAN_ICU_PARAMS['vasopressor_use'], n_patients
    ).astype(bool)

    # Hospital info
    patient_data['hospital_id'] = np.random.choice(
        ['AIIMS_Delhi', 'CMC_Vellore', 'PGIMER_Chandigarh',
         'SGPGI_Lucknow', 'JIPMER_Puducherry'],
        n_patients
    )

    patient_data['icu_type'] = np.random.choice(
        ['medical', 'surgical', 'mixed'],
        n_patients,
        p=[0.5, 0.3, 0.2]
    )

    # Individual PK parameters
    CL_typical = np.exp(
        np.log(5.5) +
        0.75 * np.log(patient_data['baseline_crcl'] / 100) +
        0.75 * np.log(patient_data['weight'] / 70) +
        np.where(patient_data['sepsis_type'] == 'septic shock', -0.15, 0)
    )

    Vc_typical = np.exp(
        np.log(16) +
        1.0 * np.log(patient_data['weight'] / 70)
    )

    patient_data['CL_individual'] = CL_typical * np.exp(np.random.normal(0, 0.25, n_patients))
    patient_data['Vc_individual'] = Vc_typical * np.exp(np.random.normal(0, 0.20, n_patients))
    patient_data['Q_individual'] = 12 * np.exp(np.random.normal(0, 0.40, n_patients))
    patient_data['Vp_individual'] = 10 * np.exp(np.random.normal(0, 0.35, n_patients))

    return patient_data


def generate_dosing_records(patient_data):
    """Generate dosing records for each patient."""

    print("Step 2/5: Generating dosing records...")

    dosing_list = []

    for _, patient in patient_data.iterrows():
        # Choose drug
        drug = np.random.choice(['amikacin', 'gentamicin'], p=[0.75, 0.25])

        # Dose per kg
        if drug == 'amikacin':
            dose_per_kg = np.random.uniform(15, 20)
        else:
            dose_per_kg = np.random.uniform(5, 7)

        # Calculate dose, round to nearest 50 mg
        dose_mg = round(dose_per_kg * patient['weight'] / 50) * 50

        # Interval based on renal function
        if patient['baseline_crcl'] > 90:
            interval_hours = 24
        elif patient['baseline_crcl'] > 60:
            interval_hours = 24
        elif patient['baseline_crcl'] > 40:
            interval_hours = 36
        else:
            interval_hours = 48

        # Treatment duration
        treatment_days = np.random.randint(3, 8)
        n_doses = int(np.ceil(treatment_days * 24 / interval_hours))

        # Create dose records
        for dose_num in range(n_doses):
            dosing_list.append({
                'patient_id': patient['patient_id'],
                'time': dose_num * interval_hours,
                'dose': dose_mg,
                'infusion_duration': 1.0,
                'route': 'IV',
                'drug': drug
            })

    return pd.DataFrame(dosing_list)


def simulate_two_compartment_pk(times, dose_times, doses, CL, Vc, Q, Vp):
    """Simulate two-compartment PK model."""

    concentrations = np.zeros_like(times, dtype=float)

    k10 = CL / Vc
    k12 = Q / Vc
    k21 = Q / Vp

    alpha = 0.5 * ((k10 + k12 + k21) +
                   np.sqrt((k10 + k12 + k21)**2 - 4*k10*k21))
    beta = 0.5 * ((k10 + k12 + k21) -
                  np.sqrt((k10 + k12 + k21)**2 - 4*k10*k21))

    A_coef = 1 / Vc * (alpha - k21) / (alpha - beta)
    B_coef = 1 / Vc * (beta - k21) / (beta - alpha)

    for i, t in enumerate(times):
        conc_total = 0

        for dose_time, dose in zip(dose_times, doses):
            if t >= dose_time:
                time_since_dose = t - dose_time

                # During infusion (0-1 hour)
                if time_since_dose <= 1:
                    conc = (dose * A_coef * (1 - np.exp(-alpha * time_since_dose)) / alpha +
                           dose * B_coef * (1 - np.exp(-beta * time_since_dose)) / beta)
                else:
                    # After infusion
                    conc = (dose * A_coef * (1 - np.exp(-alpha * 1)) / alpha *
                           np.exp(-alpha * (time_since_dose - 1)) +
                           dose * B_coef * (1 - np.exp(-beta * 1)) / beta *
                           np.exp(-beta * (time_since_dose - 1)))

                conc_total += conc

        concentrations[i] = conc_total

    return concentrations


def generate_concentrations(patient_data, dosing_records):
    """Generate PK concentration samples."""

    print("Step 3/5: Simulating PK concentrations...")

    conc_list = []

    for _, patient in patient_data.iterrows():
        patient_doses = dosing_records[dosing_records['patient_id'] == patient['patient_id']]

        if len(patient_doses) == 0:
            continue

        # PK parameters
        CL = patient['CL_individual']
        Vc = patient['Vc_individual']
        Q = patient['Q_individual']
        Vp = patient['Vp_individual']

        # Generate sampling times
        sample_times = []
        sample_types = []
        dose_numbers = []

        for dose_idx, (_, dose_record) in enumerate(patient_doses.head(5).iterrows(), 1):
            dose_time = dose_record['time']

            # Peak sample (90% chance)
            if np.random.rand() > 0.1:
                sample_times.append(dose_time + 1)
                sample_types.append('peak')
                dose_numbers.append(dose_idx)

            # Trough sample (85% chance, if not last dose)
            if dose_idx < len(patient_doses) and np.random.rand() > 0.15:
                next_dose_time = patient_doses.iloc[dose_idx]['time']
                sample_times.append(next_dose_time - 0.5)
                sample_types.append('trough')
                dose_numbers.append(dose_idx)

            # Random sample (10% chance)
            if np.random.rand() > 0.9 and dose_idx < len(patient_doses):
                random_time = dose_time + np.random.uniform(4, 20)
                sample_times.append(random_time)
                sample_types.append('random')
                dose_numbers.append(dose_idx)

        if len(sample_times) == 0:
            continue

        # Simulate concentrations
        sample_times = np.array(sample_times)
        conc_pred = simulate_two_compartment_pk(
            sample_times,
            patient_doses['time'].values,
            patient_doses['dose'].values,
            CL, Vc, Q, Vp
        )

        # Add residual error
        prop_error = 0.15
        add_error = 0.5

        conc_obs = (conc_pred * np.exp(np.random.normal(0, prop_error, len(conc_pred))) +
                   np.random.normal(0, add_error, len(conc_pred)))
        conc_obs = np.maximum(conc_obs, 0.1)

        # Create records
        for time, conc, stype, dnum in zip(sample_times, conc_obs, sample_types, dose_numbers):
            dose_time = patient_doses.iloc[dnum - 1]['time']
            conc_list.append({
                'patient_id': patient['patient_id'],
                'time': time,
                'sample_time_from_dose': time - dose_time,
                'concentration': round(conc, 2),
                'assay_lloq': 0.5,
                'bloq': conc < 0.5,
                'sample_type': stype,
                'dose_number': dnum
            })

    return pd.DataFrame(conc_list)


def generate_time_varying(patient_data, study_duration_days=7):
    """Generate time-varying covariates."""

    print("Step 4/5: Generating time-varying covariates...")

    tv_list = []

    for _, patient in patient_data.iterrows():
        days = np.arange(study_duration_days + 1)

        # SCr trajectory (random walk)
        scr_baseline = patient['baseline_scr']
        scr_mean_change = 0.1 if patient['apache_ii'] > 25 else -0.05
        scr_change = np.cumsum(np.concatenate([[0],
                              np.random.normal(scr_mean_change, 0.2, len(days) - 1)]))
        scr_trajectory = np.clip(scr_baseline + scr_change, 0.3, 8)

        # CrCL from SCr
        sex_factor = 1 if patient['sex'] == 'M' else 0.85
        crcl_trajectory = ((140 - patient['age']) * patient['weight'] * sex_factor) / (72 * scr_trajectory)

        # Fluid balance
        fluid_balance = np.clip(
            np.random.normal(np.linspace(3, 0, len(days)), 1.5),
            -2, 10
        )

        # Inflammatory markers
        wbc = np.clip(np.random.normal(np.linspace(18, 10, len(days)), 3), 2, 40)
        crp = np.clip(
            stats.lognorm.rvs(0.4, scale=np.exp(np.log(np.linspace(150, 30, len(days)))),
                            size=len(days)),
            5, 400
        )

        # RRT
        rrt_prob = 0.3 if (patient['apache_ii'] > 25 and patient['baseline_scr'] > 2) else 0.05
        rrt_initiated = np.random.binomial(1, rrt_prob) == 1

        if rrt_initiated:
            rrt_start_day = np.random.randint(1, 4)
            rrt_status = days >= rrt_start_day
            rrt_type = np.where(rrt_status, 'CVVHDF', 'none')
        else:
            rrt_status = np.zeros(len(days), dtype=bool)
            rrt_type = np.full(len(days), 'none')

        # Create records
        for i, day in enumerate(days):
            tv_list.append({
                'patient_id': patient['patient_id'],
                'time': day * 24,
                'scr': round(scr_trajectory[i], 2),
                'crcl_cg': round(crcl_trajectory[i], 1),
                'crcl_measured': np.nan,
                'urine_output': round(np.clip(np.random.normal(1800, 600), 200, 4000), 0),
                'cumulative_fluid_input': round(np.random.gamma(2, 1.5) * (i + 1), 1),
                'cumulative_fluid_output': round(np.random.gamma(2, 1.2) * (i + 1), 1),
                'fluid_balance': round(fluid_balance[i], 1),
                'wbc': round(wbc[i], 1),
                'crp': round(crp[i], 1),
                'procalcitonin': round(np.clip(stats.lognorm.rvs(1, scale=np.exp(np.log(5))),
                                              0.1, 100), 2),
                'albumin': round(np.clip(np.random.normal(patient['baseline_albumin'], 0.2),
                                        1.5, 4), 1),
                'rrt_status': bool(rrt_status[i]),
                'rrt_type': rrt_type[i],
                'rrt_flow_rate': round(np.random.normal(2000, 300), 0) if rrt_status[i] else np.nan,
                'map': round(np.clip(np.random.normal(
                    65 if patient['sepsis_type'] == 'septic shock' else 75, 8), 50, 110), 0),
                'cardiac_output': np.nan,
                'norepinephrine_dose': round(np.clip(np.random.normal(
                    max(0.3 - i * 0.04, 0), 0.1), 0, 2), 2) if patient['vasopressor_use'] else 0
            })

    return pd.DataFrame(tv_list)


def generate_outcomes(patient_data, concentrations, dosing_records):
    """Generate clinical outcomes."""

    print("Step 5/5: Simulating clinical outcomes...")

    # Calculate PK metrics
    pk_metrics = concentrations.groupby('patient_id').agg({
        'concentration': [
            ('observed_cmax', lambda x: x[concentrations.loc[x.index, 'sample_type'] == 'peak'].max()
             if any(concentrations.loc[x.index, 'sample_type'] == 'peak') else np.nan),
            ('mean_trough', lambda x: x[concentrations.loc[x.index, 'sample_type'] == 'trough'].mean()
             if any(concentrations.loc[x.index, 'sample_type'] == 'trough') else np.nan)
        ]
    })
    pk_metrics.columns = ['observed_cmax', 'mean_trough']
    pk_metrics = pk_metrics.reset_index()

    # Merge with patient data
    outcomes = patient_data.merge(pk_metrics, on='patient_id', how='left')
    outcomes = outcomes.merge(
        dosing_records[['patient_id', 'drug']].drop_duplicates(),
        on='patient_id', how='left'
    )

    # Get MIC
    outcomes['mic'] = np.where(
        outcomes['drug'] == 'amikacin',
        outcomes['mic_amikacin'],
        outcomes['mic_gentamicin']
    )

    # Calculate Cmax/MIC
    outcomes['achieved_cmax_mic'] = outcomes['observed_cmax'] / outcomes['mic']

    # Estimate AUC
    first_dose = dosing_records.groupby('patient_id')['dose'].first()
    outcomes = outcomes.merge(first_dose.rename('first_dose'), on='patient_id', how='left')
    outcomes['estimated_auc24'] = outcomes['first_dose'] / outcomes['CL_individual']
    outcomes['achieved_auc_mic'] = outcomes['estimated_auc24'] / outcomes['mic']

    # Clinical cure
    cure_logit = (-2 +
                  1.5 * np.log(outcomes['achieved_cmax_mic'] + 0.1) -
                  0.03 * outcomes['apache_ii'] -
                  0.5 * (outcomes['pathogen'] == 'acinetobacter').astype(int))
    cure_prob = 1 / (1 + np.exp(-cure_logit))
    outcomes['clinical_cure'] = np.random.binomial(1, cure_prob).astype(bool)
    outcomes['microbiological_eradication'] = np.random.binomial(1, cure_prob * 0.9).astype(bool)
    outcomes['time_to_clinical_improvement'] = np.where(
        outcomes['clinical_cure'],
        np.clip(np.random.normal(3, 1, len(outcomes)), 1, 7).round(1),
        np.nan
    )

    # Nephrotoxicity
    nephrotox_logit = (-3 +
                       1.2 * np.log(outcomes['mean_trough'].fillna(1) + 0.1) +
                       0.02 * outcomes['age'] +
                       0.5 * outcomes['diabetes'].astype(int) +
                       0.4 * (outcomes['baseline_scr'] > 1.5).astype(int))
    nephrotox_prob = 1 / (1 + np.exp(-nephrotox_logit))
    outcomes['nephrotoxicity'] = np.random.binomial(1, nephrotox_prob).astype(bool)

    # AKI stage
    aki_probs = np.where(outcomes['nephrotoxicity'],
                        np.random.rand(len(outcomes)), 1.0)
    outcomes['aki_stage'] = np.where(
        ~outcomes['nephrotoxicity'], '0',
        np.where(aki_probs < 0.5, '1',
                np.where(aki_probs < 0.8, '2', '3'))
    )

    outcomes['peak_scr'] = (outcomes['baseline_scr'] *
                           np.where(outcomes['nephrotoxicity'],
                                   np.random.uniform(1.5, 3.0, len(outcomes)),
                                   np.random.uniform(0.9, 1.3, len(outcomes)))).round(2)

    # Ototoxicity and neurotoxicity
    outcomes['ototoxicity'] = np.random.binomial(1, 0.03, len(outcomes)).astype(bool)
    outcomes['neurotoxicity'] = np.random.binomial(1, 0.02, len(outcomes)).astype(bool)

    # Length of stay
    outcomes['icu_los'] = np.clip(
        np.random.normal(10 + outcomes['apache_ii'] * 0.2 -
                        outcomes['clinical_cure'].astype(int) * 3, 4),
        2, 30
    ).round(1)
    outcomes['hospital_los'] = (outcomes['icu_los'] +
                               np.clip(np.random.normal(7, 3, len(outcomes)), 0, 60)).round(1)

    # Mortality
    mortality_logit = (-4 +
                       0.08 * outcomes['apache_ii'] +
                       0.1 * outcomes['sofa_score'] -
                       0.8 * outcomes['clinical_cure'].astype(int) +
                       0.5 * outcomes['nephrotoxicity'].astype(int))
    mortality_prob = 1 / (1 + np.exp(-mortality_logit))
    outcomes['icu_mortality'] = np.random.binomial(1, mortality_prob).astype(bool)
    outcomes['day_28_mortality'] = np.random.binomial(1, np.minimum(mortality_prob * 1.2, 1)).astype(bool)

    # Select outcome columns
    outcome_cols = ['patient_id', 'clinical_cure', 'microbiological_eradication',
                   'time_to_clinical_improvement', 'nephrotoxicity', 'aki_stage',
                   'peak_scr', 'ototoxicity', 'neurotoxicity', 'icu_los', 'hospital_los',
                   'icu_mortality', 'day_28_mortality', 'achieved_cmax_mic', 'achieved_auc_mic']

    return outcomes[outcome_cols]


def main():
    """Main data generation function."""

    print("=" * 80)
    print("SYNTHETIC DATA GENERATION - INDIAN ICU AMINOGLYCOSIDE STUDY")
    print("=" * 80)
    print()

    n_patients = 300
    study_duration_days = 7

    # Generate data
    patient_data = generate_patient_demographics(n_patients)
    dosing_records = generate_dosing_records(patient_data)
    concentrations = generate_concentrations(patient_data, dosing_records)
    time_varying = generate_time_varying(patient_data, study_duration_days)
    outcomes = generate_outcomes(patient_data, concentrations, dosing_records)

    # Clean patient data (remove PK parameters for export)
    patient_data_clean = patient_data.drop(
        columns=['CL_individual', 'Vc_individual', 'Q_individual', 'Vp_individual']
    )

    # Create output directory
    os.makedirs('data', exist_ok=True)

    # Save as CSV
    print("\nSaving data files...")
    patient_data_clean.to_csv('data/patient_data.csv', index=False)
    dosing_records.drop(columns=['drug']).to_csv('data/dosing.csv', index=False)
    concentrations.to_csv('data/concentrations.csv', index=False)
    time_varying.to_csv('data/time_varying.csv', index=False)
    outcomes.to_csv('data/outcomes.csv', index=False)

    # Save as pickle (includes all data)
    synthetic_data = {
        'patient_data': patient_data_clean,
        'time_varying': time_varying,
        'dosing': dosing_records.drop(columns=['drug']),
        'concentrations': concentrations,
        'outcomes': outcomes,
        'metadata': {
            'n_patients': n_patients,
            'study_duration_days': study_duration_days,
            'generation_date': datetime.now().isoformat(),
            'seed': 12345,
            'population': 'Indian ICU',
            'description': 'Synthetic aminoglycoside data for Indian ICU setting'
        }
    }

    with open('data/synthetic_aminoglycoside_data.pkl', 'wb') as f:
        pickle.dump(synthetic_data, f)

    # Also save metadata as JSON for easy inspection
    with open('data/metadata.json', 'w') as f:
        json.dump(synthetic_data['metadata'], f, indent=2)

    print("\nData saved to 'data/' directory:")
    print("  - patient_data.csv")
    print("  - dosing.csv")
    print("  - concentrations.csv")
    print("  - time_varying.csv")
    print("  - outcomes.csv")
    print("  - synthetic_aminoglycoside_data.pkl (Python pickle)")
    print("  - metadata.json")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print()
    print("Patient Demographics:")
    print(f"  Age: {patient_data['age'].mean():.1f} ± {patient_data['age'].std():.1f} years")
    print(f"  Weight: {patient_data['weight'].mean():.1f} ± {patient_data['weight'].std():.1f} kg")
    print(f"  Male: {(patient_data['sex'] == 'M').mean() * 100:.1f}%")
    print()
    print("Clinical Severity:")
    print(f"  APACHE II: {patient_data['apache_ii'].mean():.1f} ± {patient_data['apache_ii'].std():.1f}")
    print(f"  SOFA: {patient_data['sofa_score'].mean():.1f} ± {patient_data['sofa_score'].std():.1f}")
    print(f"  Septic shock: {(patient_data['sepsis_type'] == 'septic shock').mean() * 100:.1f}%")
    print()
    print("Outcomes:")
    print(f"  Clinical cure: {outcomes['clinical_cure'].mean() * 100:.1f}%")
    print(f"  Nephrotoxicity: {outcomes['nephrotoxicity'].mean() * 100:.1f}%")
    print(f"  ICU mortality: {outcomes['icu_mortality'].mean() * 100:.1f}%")
    print()
    print("PK/PD Metrics:")
    print(f"  Mean Cmax/MIC: {outcomes['achieved_cmax_mic'].mean():.1f}")
    print(f"  Target (Cmax/MIC ≥8): {(outcomes['achieved_cmax_mic'] >= 8).mean() * 100:.1f}%")
    print()
    print("Data Records:")
    print(f"  Patients: {len(patient_data_clean)}")
    print(f"  Dosing records: {len(dosing_records)}")
    print(f"  PK samples: {len(concentrations)}")
    print(f"  Time-varying records: {len(time_varying)}")
    print()
    print("=" * 80)
    print("READY FOR ANALYSIS!")
    print("=" * 80)


if __name__ == '__main__':
    main()
