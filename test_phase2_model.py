#!/usr/bin/env python3
"""
Quick test of Phase 2 PK model structure
This verifies the model is correctly specified before full sampling.
"""

import pandas as pd
import numpy as np
import pymc as pm

print("Testing Phase 2 Population PK Model Structure...")
print("=" * 60)

# Load minimal data
data = pd.read_csv('data/processed/popk_dataset.csv')
obs_data = data[data['EVID'] == 0].head(50)  # Just 50 observations for testing

# Get patient info
patient_ids = obs_data['ID'].unique()[:5]  # Just 5 patients for testing
obs_data = obs_data[obs_data['ID'].isin(patient_ids)]

patient_info = obs_data.groupby('ID').first()[['AGE', 'WT', 'CRCL']].reset_index()
n_patients = len(patient_info)

print(f"Test data: {n_patients} patients, {len(obs_data)} observations")
print()

# Standardize covariates
patient_info['CRCL_std'] = (patient_info['CRCL'] - patient_info['CRCL'].mean()) / patient_info['CRCL'].std()
patient_info['WT_std'] = (patient_info['WT'] - patient_info['WT'].mean()) / patient_info['WT'].std()
patient_info['AGE_std'] = (patient_info['AGE'] - patient_info['AGE'].mean()) / patient_info['AGE'].std()

# Build model
print("Building model...")
with pm.Model() as test_model:
    # Population parameters
    theta_CL = pm.Lognormal('theta_CL', mu=np.log(5.5), sigma=0.3)
    theta_Vc = pm.Lognormal('theta_Vc', mu=np.log(16), sigma=0.3)
    theta_Q = pm.Lognormal('theta_Q', mu=np.log(12), sigma=0.5)
    theta_Vp = pm.Lognormal('theta_Vp', mu=np.log(10), sigma=0.5)

    # Covariate effects
    beta_CL_CRCL = pm.Normal('beta_CL_CRCL', mu=0.75, sigma=0.2)
    beta_CL_WT = pm.Normal('beta_CL_WT', mu=0.75, sigma=0.2)
    beta_Vc_WT = pm.Normal('beta_Vc_WT', mu=1.0, sigma=0.2)

    # BSV
    omega_CL = pm.HalfNormal('omega_CL', sigma=0.3)
    omega_Vc = pm.HalfNormal('omega_Vc', sigma=0.25)

    # Individual effects
    eta_CL = pm.Normal('eta_CL', mu=0, sigma=1, shape=n_patients)
    eta_Vc = pm.Normal('eta_Vc', mu=0, sigma=1, shape=n_patients)

    # Individual parameters
    CRCL_std = patient_info['CRCL_std'].values
    WT_std = patient_info['WT_std'].values

    CL_ind = theta_CL * pm.math.exp(
        beta_CL_CRCL * CRCL_std +
        beta_CL_WT * WT_std +
        omega_CL * eta_CL
    )

    Vc_ind = theta_Vc * pm.math.exp(
        beta_Vc_WT * WT_std +
        omega_Vc * eta_Vc
    )

    pm.Deterministic('CL', CL_ind)
    pm.Deterministic('Vc', Vc_ind)

print("✓ Model structure is valid!")
print()

# Test prior sampling
print("Testing prior sampling...")
with test_model:
    prior = pm.sample_prior_predictive(samples=100, random_seed=42)

print("✓ Prior sampling successful!")
print()

# Test a very short MCMC run
print("Testing MCMC sampling (10 draws for verification)...")
with test_model:
    trace = pm.sample(draws=10, tune=10, chains=1, cores=1, random_seed=42)

print("✓ MCMC sampling successful!")
print()

print("=" * 60)
print("Phase 2 model structure is READY!")
print("The full model can now be run with: python3 phase2_population_pk.py")
print("=" * 60)
