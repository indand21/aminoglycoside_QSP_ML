"""
Utility Functions for Extracting Individual PK Parameters from Bayesian Posterior

This module provides functions to extract patient-specific pharmacokinetic parameters
from the Bayesian population PK model posterior distributions.

Author: Aminoglycoside QSP-ML Framework
Date: 2025-12-02
"""

import numpy as np
import pandas as pd
import arviz as az
import os
from typing import Dict, List, Optional, Tuple


def extract_individual_pk_parameters(
    trace_path: str,
    patient_ids: Optional[List] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract individual PK parameters from Bayesian posterior trace
    
    Parameters:
    -----------
    trace_path : str
        Path to the NetCDF trace file (e.g., 'models/phase2_popk_trace.nc')
    patient_ids : list, optional
        List of patient IDs. If None, will use sequential numbering
    output_path : str, optional
        Path to save the extracted parameters CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with individual PK parameters for each patient:
        - patient_id: Patient identifier
        - CL_bayes: Individual clearance (L/h) - posterior mean
        - Vc_bayes: Individual central volume (L) - posterior mean
        - Q_bayes: Individual intercompartmental clearance (L/h) - posterior mean
        - Vp_bayes: Individual peripheral volume (L) - posterior mean
        - CL_uncertainty: Standard deviation of CL posterior
        - Vc_uncertainty: Standard deviation of Vc posterior
        - Q_uncertainty: Standard deviation of Q posterior
        - Vp_uncertainty: Standard deviation of Vp posterior
        - Vss_bayes: Volume of distribution at steady state (Vc + Vp)
        - k10_bayes: Elimination rate constant (CL/Vc)
        - k12_bayes: Transfer rate constant central→peripheral (Q/Vc)
        - k21_bayes: Transfer rate constant peripheral→central (Q/Vp)
        - t_half_alpha: Distribution half-life (hours)
        - t_half_beta: Elimination half-life (hours)
        - CL_cv: Coefficient of variation for CL (uncertainty/mean)
        - Vc_cv: Coefficient of variation for Vc
    
    Example:
    --------
    >>> params = extract_individual_pk_parameters(
    ...     'models/phase2_popk_trace.nc',
    ...     patient_ids=[1, 2, 3],
    ...     output_path='results/phase3_pkpd/individual_pk_parameters.csv'
    ... )
    """
    
    print("\n" + "="*80)
    print("EXTRACTING INDIVIDUAL PK PARAMETERS FROM BAYESIAN POSTERIOR")
    print("="*80)
    
    # Check if trace file exists
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    # Load trace
    print(f"\nLoading trace from: {trace_path}")
    try:
        trace = az.from_netcdf(trace_path)
        print("  [OK] Trace loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load trace: {e}")
    
    # Get number of patients from trace dimensions
    try:
        # Try to get individual parameters from trace
        CL_samples = trace.posterior['CL'].values
        n_chains, n_draws, n_patients = CL_samples.shape
        print(f"  [OK] Found {n_patients} patients in trace")
        print(f"  [OK] Posterior: {n_chains} chains x {n_draws} draws = {n_chains * n_draws} samples")
    except KeyError:
        raise KeyError("Could not find 'CL' parameter in trace. Check trace file structure.")
    
    # Use provided patient IDs or create sequential IDs
    if patient_ids is None:
        patient_ids = list(range(1, n_patients + 1))
    elif len(patient_ids) != n_patients:
        print(f"  [WARNING] {len(patient_ids)} patient IDs provided but trace has {n_patients} patients")
        print(f"    Using first {min(len(patient_ids), n_patients)} patients")
        patient_ids = patient_ids[:n_patients]
    
    # Extract individual parameters
    print(f"\nExtracting individual PK parameters for {len(patient_ids)} patients...")
    
    results = []
    
    for i, patient_id in enumerate(patient_ids):
        try:
            # Extract posterior samples for this patient (flatten chains and draws)
            CL_samples = trace.posterior['CL'].values[:, :, i].flatten()
            Vc_samples = trace.posterior['Vc'].values[:, :, i].flatten()
            Q_samples = trace.posterior['Q'].values[:, :, i].flatten()
            Vp_samples = trace.posterior['Vp'].values[:, :, i].flatten()
            
            # Calculate posterior statistics
            CL_mean = np.mean(CL_samples)
            Vc_mean = np.mean(Vc_samples)
            Q_mean = np.mean(Q_samples)
            Vp_mean = np.mean(Vp_samples)
            
            CL_std = np.std(CL_samples)
            Vc_std = np.std(Vc_samples)
            Q_std = np.std(Q_samples)
            Vp_std = np.std(Vp_samples)
            
            # Calculate derived parameters
            Vss = Vc_mean + Vp_mean  # Volume at steady state
            
            # Rate constants
            k10 = CL_mean / Vc_mean  # Elimination rate constant
            k12 = Q_mean / Vc_mean   # Central → Peripheral
            k21 = Q_mean / Vp_mean   # Peripheral → Central
            
            # Half-lives (two-compartment model)
            # Alpha (distribution) and beta (elimination) half-lives
            k_sum = k10 + k12 + k21
            k_prod = k10 * k21

            # Hybrid rate constants
            alpha = 0.5 * (k_sum + np.sqrt(k_sum**2 - 4*k_prod))  # Fast (distribution)
            beta = 0.5 * (k_sum - np.sqrt(k_sum**2 - 4*k_prod))   # Slow (elimination)

            t_half_alpha = np.log(2) / alpha  # Distribution half-life (hours)
            t_half_beta = np.log(2) / beta    # Elimination half-life (hours)

            # Coefficient of variation (measure of uncertainty)
            CL_cv = CL_std / CL_mean if CL_mean > 0 else 0
            Vc_cv = Vc_std / Vc_mean if Vc_mean > 0 else 0

            # Store results
            results.append({
                'patient_id': patient_id,
                # Primary PK parameters (posterior means)
                'CL_bayes': CL_mean,
                'Vc_bayes': Vc_mean,
                'Q_bayes': Q_mean,
                'Vp_bayes': Vp_mean,
                # Uncertainty (posterior standard deviations)
                'CL_uncertainty': CL_std,
                'Vc_uncertainty': Vc_std,
                'Q_uncertainty': Q_std,
                'Vp_uncertainty': Vp_std,
                # Derived parameters
                'Vss_bayes': Vss,
                'k10_bayes': k10,
                'k12_bayes': k12,
                'k21_bayes': k21,
                't_half_alpha': t_half_alpha,
                't_half_beta': t_half_beta,
                # Coefficient of variation
                'CL_cv': CL_cv,
                'Vc_cv': Vc_cv
            })

        except Exception as e:
            print(f"  [WARNING] Failed to extract parameters for patient {patient_id}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary statistics
    print(f"\n[OK] Successfully extracted parameters for {len(df)} patients")
    print("\nSummary Statistics (Posterior Means):")
    print("-" * 80)
    print(f"{'Parameter':<20s} {'Mean':<12s} {'Std':<12s} {'Min':<12s} {'Max':<12s}")
    print("-" * 80)

    for param in ['CL_bayes', 'Vc_bayes', 'Q_bayes', 'Vp_bayes', 'Vss_bayes',
                  't_half_alpha', 't_half_beta']:
        if param in df.columns:
            print(f"{param:<20s} {df[param].mean():<12.3f} {df[param].std():<12.3f} "
                  f"{df[param].min():<12.3f} {df[param].max():<12.3f}")

    print("\nUncertainty Metrics:")
    print("-" * 80)
    print(f"Mean CL CV: {df['CL_cv'].mean():.3f} (range: {df['CL_cv'].min():.3f} - {df['CL_cv'].max():.3f})")
    print(f"Mean Vc CV: {df['Vc_cv'].mean():.3f} (range: {df['Vc_cv'].min():.3f} - {df['Vc_cv'].max():.3f})")

    # Save to file if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Saved individual PK parameters to: {output_path}")

    return df


def calculate_pk_metrics_from_parameters(
    CL: float,
    Vc: float,
    Q: float,
    Vp: float,
    dose: float,
    tau: float = 24
) -> Dict[str, float]:
    """
    Calculate PK metrics from individual parameters

    Parameters:
    -----------
    CL : float
        Clearance (L/h)
    Vc : float
        Central volume (L)
    Q : float
        Intercompartmental clearance (L/h)
    Vp : float
        Peripheral volume (L)
    dose : float
        Dose (mg)
    tau : float
        Dosing interval (hours), default 24

    Returns:
    --------
    dict
        Dictionary with PK metrics:
        - Cmax: Peak concentration (mg/L)
        - Cmin: Trough concentration (mg/L)
        - AUC24: Area under curve 0-24h (mg·h/L)
        - t_half_beta: Elimination half-life (hours)
    """

    # Rate constants
    k10 = CL / Vc
    k12 = Q / Vc
    k21 = Q / Vp

    # Hybrid rate constants
    k_sum = k10 + k12 + k21
    k_prod = k10 * k21
    alpha = 0.5 * (k_sum + np.sqrt(k_sum**2 - 4*k_prod))
    beta = 0.5 * (k_sum - np.sqrt(k_sum**2 - 4*k_prod))

    # Coefficients for two-compartment model
    A = (dose / Vc) * (alpha - k21) / (alpha - beta)
    B = (dose / Vc) * (k21 - beta) / (alpha - beta)

    # Cmax (at t=0, end of infusion, assuming bolus)
    Cmax = A + B

    # Cmin (at t=tau)
    Cmin = A * np.exp(-alpha * tau) + B * np.exp(-beta * tau)

    # AUC (0-tau)
    AUC24 = (A / alpha) * (1 - np.exp(-alpha * tau)) + (B / beta) * (1 - np.exp(-beta * tau))

    # Half-life
    t_half_beta = np.log(2) / beta

    return {
        'Cmax': Cmax,
        'Cmin': Cmin,
        'AUC24': AUC24,
        't_half_beta': t_half_beta
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        trace_path = sys.argv[1]
    else:
        trace_path = 'models/phase2_popk_trace.nc'

    output_path = 'results/phase3_pkpd/individual_pk_parameters.csv'

    print("Extracting individual PK parameters...")
    print(f"Trace file: {trace_path}")
    print(f"Output file: {output_path}")

    try:
        df = extract_individual_pk_parameters(trace_path, output_path=output_path)
        print("\n[OK] Extraction complete!")
        print(f"\nFirst 5 patients:")
        print(df.head())
    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)

