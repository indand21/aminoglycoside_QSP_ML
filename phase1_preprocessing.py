#!/usr/bin/env python3
"""
PHASE 1: DATA PREPROCESSING PIPELINE
Aminoglycoside QSP-ML Project

This script preprocesses the synthetic aminoglycoside data:
- Calculates derived variables (BSA, IBW, ABW)
- Computes creatinine clearance
- Calculates PK/PD metrics
- Detects outliers
- Handles missing data
- Creates analysis-ready datasets
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

class AminoglycosidePreprocessor:
    """Preprocess aminoglycoside data for pharmacometric and ML analysis."""

    def __init__(self, data_path='data'):
        """Initialize preprocessor and load data.

        Args:
            data_path: Path to directory containing CSV files
        """
        print("=" * 80)
        print("AMINOGLYCOSIDE DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        print()

        self.data_path = data_path
        self.data = {}
        self.load_data()

    def load_data(self):
        """Load all data files."""
        print("Step 1/7: Loading raw data...")

        self.data['patient_data'] = pd.read_csv(f'{self.data_path}/patient_data.csv')
        self.data['dosing'] = pd.read_csv(f'{self.data_path}/dosing.csv')
        self.data['concentrations'] = pd.read_csv(f'{self.data_path}/concentrations.csv')
        self.data['time_varying'] = pd.read_csv(f'{self.data_path}/time_varying.csv')
        self.data['outcomes'] = pd.read_csv(f'{self.data_path}/outcomes.csv')

        print(f"  ✓ Loaded {len(self.data['patient_data'])} patients")
        print(f"  ✓ Loaded {len(self.data['dosing'])} dosing records")
        print(f"  ✓ Loaded {len(self.data['concentrations'])} concentration samples")
        print(f"  ✓ Loaded {len(self.data['time_varying'])} time-varying records")
        print(f"  ✓ Loaded {len(self.data['outcomes'])} outcome records")
        print()

    def calculate_derived_variables(self):
        """Calculate derived anthropometric and physiologic variables."""
        print("Step 2/7: Calculating derived variables...")

        df = self.data['patient_data'].copy()

        # Body surface area (Mosteller formula)
        # BSA (m²) = sqrt((height(cm) × weight(kg)) / 3600)
        df['bsa'] = np.sqrt((df['height'] * df['weight']) / 3600).round(2)

        # Ideal body weight (Devine formula)
        # Male: IBW = 50 + 2.3 × (height(inches) - 60)
        # Female: IBW = 45.5 + 2.3 × (height(inches) - 60)
        height_inches = df['height'] / 2.54
        df['ibw'] = np.where(
            df['sex'] == 'M',
            50 + 2.3 * (height_inches - 60),
            45.5 + 2.3 * (height_inches - 60)
        ).round(1)

        # Adjusted body weight (for obese patients)
        # ABW = IBW + 0.4 × (actual weight - IBW)
        df['abw'] = (df['ibw'] + 0.4 * (df['weight'] - df['ibw'])).round(1)

        # Obesity classification
        df['obesity_class'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['underweight', 'normal', 'overweight', 'obese_class1', 'obese_class2', 'obese_class3'],
            include_lowest=True
        )

        # Augmented renal clearance flag
        df['arc'] = df['baseline_crcl'] > 130

        print(f"  ✓ Calculated BSA: {df['bsa'].mean():.2f} ± {df['bsa'].std():.2f} m²")
        print(f"  ✓ Calculated IBW: {df['ibw'].mean():.1f} ± {df['ibw'].std():.1f} kg")
        print(f"  ✓ Identified ARC patients: {df['arc'].sum()} ({df['arc'].mean()*100:.1f}%)")
        print()

        self.data['patient_data'] = df

    def calculate_time_varying_crcl(self):
        """Calculate creatinine clearance for time-varying data."""
        print("Step 3/7: Calculating time-varying creatinine clearance...")

        # Merge patient demographics needed for CrCL calculation
        tv = self.data['time_varying'].copy()
        patient_demo = self.data['patient_data'][['patient_id', 'age', 'sex', 'weight']].copy()

        tv = tv.merge(patient_demo, on='patient_id', how='left')

        # Cockcroft-Gault formula
        # CrCL = ((140 - age) × weight × (0.85 if female)) / (72 × SCr)
        sex_factor = np.where(tv['sex'] == 'M', 1, 0.85)
        tv['crcl_cg_calculated'] = ((140 - tv['age']) * tv['weight'] * sex_factor) / (72 * tv['scr'])
        tv['crcl_cg_calculated'] = tv['crcl_cg_calculated'].round(1)

        # Update the existing crcl_cg column
        tv['crcl_cg'] = tv['crcl_cg_calculated']

        # Drop temporary columns
        tv = tv.drop(columns=['age', 'sex', 'weight', 'crcl_cg_calculated'])

        print(f"  ✓ Updated time-varying CrCL for {len(tv)} records")
        print(f"  ✓ Mean CrCL: {tv['crcl_cg'].mean():.1f} ± {tv['crcl_cg'].std():.1f} mL/min")
        print()

        self.data['time_varying'] = tv

    def calculate_pk_pd_metrics(self):
        """Calculate PK/PD metrics from observed concentrations."""
        print("Step 4/7: Calculating PK/PD metrics...")

        conc = self.data['concentrations'].copy()

        # Calculate metrics per patient
        pk_metrics = conc.groupby('patient_id').agg({
            'concentration': [
                ('observed_cmax', 'max'),
                ('observed_cmin', lambda x: x[conc.loc[x.index, 'sample_type'] == 'trough'].min()
                 if any(conc.loc[x.index, 'sample_type'] == 'trough') else np.nan),
                ('mean_concentration', 'mean')
            ],
            'sample_type': [
                ('n_samples', 'count'),
                ('has_peak', lambda x: (x == 'peak').any()),
                ('has_trough', lambda x: (x == 'trough').any())
            ]
        })

        # Flatten multi-level columns
        pk_metrics.columns = ['_'.join(col).strip('_') for col in pk_metrics.columns.values]
        pk_metrics = pk_metrics.reset_index()
        pk_metrics.columns = ['patient_id', 'observed_cmax', 'observed_cmin', 'mean_concentration',
                              'n_samples', 'has_peak', 'has_trough']

        # Merge with patient data to get MIC values
        patient_mic = self.data['patient_data'][['patient_id', 'mic_amikacin', 'mic_gentamicin']].copy()
        pk_metrics = pk_metrics.merge(patient_mic, on='patient_id', how='left')

        # Determine which MIC to use based on drug (from dosing data)
        # For simplicity, use amikacin MIC primarily (can be refined later)
        pk_metrics['mic_used'] = pk_metrics['mic_amikacin']  # Default to amikacin

        # Calculate PK/PD indices
        pk_metrics['cmax_mic_ratio'] = (pk_metrics['observed_cmax'] / pk_metrics['mic_used']).round(2)
        pk_metrics['target_cmax_8'] = pk_metrics['cmax_mic_ratio'] >= 8
        pk_metrics['target_cmax_10'] = pk_metrics['cmax_mic_ratio'] >= 10

        print(f"  ✓ Calculated PK metrics for {len(pk_metrics)} patients")
        print(f"  ✓ Mean Cmax: {pk_metrics['observed_cmax'].mean():.2f} mg/L")
        print(f"  ✓ Mean Cmax/MIC: {pk_metrics['cmax_mic_ratio'].mean():.2f}")
        print(f"  ✓ Target attainment (≥8): {pk_metrics['target_cmax_8'].sum()} ({pk_metrics['target_cmax_8'].mean()*100:.1f}%)")
        print()

        self.pk_metrics = pk_metrics

    def detect_outliers(self):
        """Detect outliers and suspicious values."""
        print("Step 5/7: Detecting outliers...")

        conc = self.data['concentrations'].copy()

        # Concentration outliers (>3 SD or biologically implausible)
        mean_conc = conc['concentration'].mean()
        sd_conc = conc['concentration'].std()

        conc['outlier_flag'] = (
            (conc['concentration'] > mean_conc + 3*sd_conc) |
            (conc['concentration'] > 100) |  # Amikacin rarely >100 mg/L
            (conc['concentration'] < 0)
        )

        # Flag potentially erroneous peak/trough labels
        conc['suspicious_peak'] = (conc['sample_type'] == 'peak') & (conc['concentration'] < 10)
        conc['suspicious_trough'] = (conc['sample_type'] == 'trough') & (conc['concentration'] > 20)

        n_outliers = conc['outlier_flag'].sum()
        n_suspicious_peak = conc['suspicious_peak'].sum()
        n_suspicious_trough = conc['suspicious_trough'].sum()

        print(f"  ✓ Flagged {n_outliers} concentration outliers")
        print(f"  ✓ Flagged {n_suspicious_peak} suspicious peak samples")
        print(f"  ✓ Flagged {n_suspicious_trough} suspicious trough samples")
        print()

        self.data['concentrations'] = conc

    def check_data_quality(self):
        """Perform comprehensive data quality checks."""
        print("Step 6/7: Performing data quality checks...")

        quality_report = {}

        # Check for negative times
        quality_report['negative_times'] = (self.data['concentrations']['time'] < 0).sum()

        # Check for missing essential covariates
        quality_report['missing_weight'] = self.data['patient_data']['weight'].isna().sum()
        quality_report['missing_age'] = self.data['patient_data']['age'].isna().sum()
        quality_report['missing_scr'] = self.data['patient_data']['baseline_scr'].isna().sum()

        # Check dose ranges
        quality_report['extreme_doses'] = (
            (self.data['dosing']['dose'] < 100) |
            (self.data['dosing']['dose'] > 3000)
        ).sum()

        # Check for duplicate patient IDs
        quality_report['duplicate_patients'] = self.data['patient_data']['patient_id'].duplicated().sum()

        # Check value ranges
        patient = self.data['patient_data']
        quality_report['age_out_of_range'] = ((patient['age'] < 18) | (patient['age'] > 90)).sum()
        quality_report['weight_out_of_range'] = ((patient['weight'] < 30) | (patient['weight'] > 150)).sum()

        # Print summary
        all_checks_passed = all(v == 0 for v in quality_report.values())

        if all_checks_passed:
            print("  ✓ All quality checks passed!")
        else:
            for check, count in quality_report.items():
                if count > 0:
                    print(f"  ⚠ {check}: {count} issues found")

        print()
        self.quality_report = quality_report

    def create_analysis_datasets(self):
        """Create analysis-ready datasets for different purposes."""
        print("Step 7/7: Creating analysis-ready datasets...")

        # 1. Wide-format dataset for ML (one row per patient)
        ml_dataset = self.data['patient_data'].copy()

        # Add PK metrics
        ml_dataset = ml_dataset.merge(
            self.pk_metrics[['patient_id', 'observed_cmax', 'observed_cmin', 'n_samples',
                            'cmax_mic_ratio', 'target_cmax_8']],
            on='patient_id',
            how='left'
        )

        # Add dosing features
        dose_features = self.data['dosing'].groupby('patient_id').agg({
            'dose': ['first', 'mean', 'count'],
            'time': lambda x: x.max()
        }).reset_index()
        dose_features.columns = ['patient_id', 'first_dose', 'mean_dose', 'n_doses', 'max_time']
        dose_features['dose_frequency'] = dose_features['n_doses'] / (dose_features['max_time'] / 24)

        ml_dataset = ml_dataset.merge(dose_features, on='patient_id', how='left')

        # Add time-varying summary features
        tv_features = self.data['time_varying'].groupby('patient_id').agg({
            'scr': ['first', 'max', 'mean'],
            'crcl_cg': ['first', 'min', 'mean'],
            'fluid_balance': 'max',
            'wbc': 'max',
            'crp': 'max',
            'rrt_status': 'any'
        }).reset_index()
        tv_features.columns = ['patient_id', 'baseline_scr_tv', 'peak_scr_tv', 'mean_scr',
                               'baseline_crcl_tv', 'min_crcl', 'mean_crcl',
                               'max_fluid_balance', 'max_wbc', 'max_crp', 'rrt_ever']

        ml_dataset = ml_dataset.merge(tv_features, on='patient_id', how='left')

        # Add outcomes
        ml_dataset = ml_dataset.merge(self.data['outcomes'], on='patient_id', how='left')

        print(f"  ✓ Created ML dataset: {len(ml_dataset)} rows × {len(ml_dataset.columns)} columns")

        # 2. Long-format dataset for PopPK modeling (NONMEM-style)
        popk_dataset = self.create_nonmem_dataset()

        print(f"  ✓ Created PopPK dataset: {len(popk_dataset)} rows")
        print()

        self.ml_dataset = ml_dataset
        self.popk_dataset = popk_dataset

    def create_nonmem_dataset(self):
        """Create NONMEM-compatible dataset for population PK modeling."""

        # Dosing records
        dose_records = self.data['dosing'].copy()
        dose_records['EVID'] = 1  # Dosing event
        dose_records['AMT'] = dose_records['dose']
        dose_records['DV'] = np.nan
        dose_records['CMT'] = 1  # Central compartment
        dose_records['RATE'] = dose_records['AMT'] / dose_records['infusion_duration']
        dose_records['MDV'] = 1  # Missing DV
        dose_records = dose_records.rename(columns={'time': 'TIME'})
        dose_records = dose_records[['patient_id', 'TIME', 'EVID', 'AMT', 'DV', 'CMT', 'RATE', 'MDV']]

        # Concentration records (exclude outliers)
        conc_records = self.data['concentrations'][
            ~self.data['concentrations']['outlier_flag']
        ].copy()
        conc_records['EVID'] = 0  # Observation
        conc_records['AMT'] = 0
        conc_records['DV'] = conc_records['concentration']
        conc_records['CMT'] = 2  # Observation compartment
        conc_records['RATE'] = 0
        conc_records['MDV'] = conc_records['bloq'].astype(int)
        conc_records = conc_records.rename(columns={'time': 'TIME'})
        conc_records = conc_records[['patient_id', 'TIME', 'EVID', 'AMT', 'DV', 'CMT', 'RATE', 'MDV']]

        # Combine and sort
        nm_data = pd.concat([dose_records, conc_records], ignore_index=True)
        nm_data = nm_data.sort_values(['patient_id', 'TIME', 'EVID'], ascending=[True, True, False])

        # Add covariates (baseline, time-invariant)
        patient_cov = self.data['patient_data'][
            ['patient_id', 'age', 'weight', 'height', 'sex',
             'baseline_scr', 'baseline_crcl', 'apache_ii', 'sofa_score']
        ].copy()
        patient_cov = patient_cov.rename(columns={
            'age': 'AGE',
            'weight': 'WT',
            'height': 'HT',
            'sex': 'SEX',
            'baseline_scr': 'SCR',
            'baseline_crcl': 'CRCL',
            'apache_ii': 'APACHE',
            'sofa_score': 'SOFA'
        })
        patient_cov['SEX'] = (patient_cov['SEX'] == 'M').astype(int)  # 1=male, 0=female

        nm_data = nm_data.merge(patient_cov, on='patient_id', how='left')

        # Add numeric patient ID
        nm_data['ID'] = pd.Categorical(nm_data['patient_id']).codes + 1

        return nm_data

    def save_processed_data(self, output_dir='data/processed'):
        """Save all processed datasets."""
        os.makedirs(output_dir, exist_ok=True)

        print("Saving processed data...")

        # Save CSV files
        self.data['patient_data'].to_csv(f'{output_dir}/patient_data_processed.csv', index=False)
        self.data['time_varying'].to_csv(f'{output_dir}/time_varying_processed.csv', index=False)
        self.data['concentrations'].to_csv(f'{output_dir}/concentrations_processed.csv', index=False)
        self.pk_metrics.to_csv(f'{output_dir}/pk_metrics.csv', index=False)
        self.ml_dataset.to_csv(f'{output_dir}/ml_dataset.csv', index=False)
        self.popk_dataset.to_csv(f'{output_dir}/popk_dataset.csv', index=False)

        # Save pickle
        processed_data = {
            'patient_data': self.data['patient_data'],
            'time_varying': self.data['time_varying'],
            'dosing': self.data['dosing'],
            'concentrations': self.data['concentrations'],
            'outcomes': self.data['outcomes'],
            'pk_metrics': self.pk_metrics,
            'ml_dataset': self.ml_dataset,
            'popk_dataset': self.popk_dataset,
            'quality_report': self.quality_report,
            'metadata': {
                'preprocessing_date': datetime.now().isoformat(),
                'n_patients': len(self.data['patient_data']),
                'n_concentrations': len(self.data['concentrations']),
                'n_outliers': self.data['concentrations']['outlier_flag'].sum()
            }
        }

        with open(f'{output_dir}/processed_data.pkl', 'wb') as f:
            pickle.dump(processed_data, f)

        # Save metadata as JSON (convert numpy types to Python types)
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj

        with open(f'{output_dir}/preprocessing_metadata.json', 'w') as f:
            metadata_json = convert_numpy_types({
                **processed_data['metadata'],
                'quality_report': self.quality_report
            })
            json.dump(metadata_json, f, indent=2)

        print(f"\n✓ Saved processed data to {output_dir}/")
        print(f"  - patient_data_processed.csv")
        print(f"  - time_varying_processed.csv")
        print(f"  - concentrations_processed.csv")
        print(f"  - pk_metrics.csv")
        print(f"  - ml_dataset.csv ({len(self.ml_dataset)} rows × {len(self.ml_dataset.columns)} cols)")
        print(f"  - popk_dataset.csv ({len(self.popk_dataset)} rows, NONMEM format)")
        print(f"  - processed_data.pkl")
        print(f"  - preprocessing_metadata.json")

    def generate_summary_report(self):
        """Generate comprehensive preprocessing summary."""
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY REPORT")
        print("=" * 80)
        print()

        print("DERIVED VARIABLES:")
        print(f"  BSA: {self.data['patient_data']['bsa'].mean():.2f} ± {self.data['patient_data']['bsa'].std():.2f} m²")
        print(f"  IBW: {self.data['patient_data']['ibw'].mean():.1f} ± {self.data['patient_data']['ibw'].std():.1f} kg")
        print(f"  ABW: {self.data['patient_data']['abw'].mean():.1f} ± {self.data['patient_data']['abw'].std():.1f} kg")
        print()

        print("OBESITY CLASSIFICATION:")
        for cat, count in self.data['patient_data']['obesity_class'].value_counts().items():
            print(f"  {cat}: {count} ({count/len(self.data['patient_data'])*100:.1f}%)")
        print()

        print("PK/PD METRICS:")
        print(f"  Mean Cmax: {self.pk_metrics['observed_cmax'].mean():.2f} mg/L")
        print(f"  Mean Cmax/MIC: {self.pk_metrics['cmax_mic_ratio'].mean():.2f}")
        print(f"  Target Cmax/MIC ≥8: {self.pk_metrics['target_cmax_8'].sum()} ({self.pk_metrics['target_cmax_8'].mean()*100:.1f}%)")
        print(f"  Target Cmax/MIC ≥10: {self.pk_metrics['target_cmax_10'].sum()} ({self.pk_metrics['target_cmax_10'].mean()*100:.1f}%)")
        print()

        print("DATA QUALITY:")
        if all(v == 0 for v in self.quality_report.values()):
            print("  ✓ All quality checks passed")
        else:
            for check, count in self.quality_report.items():
                status = "✓" if count == 0 else "⚠"
                print(f"  {status} {check}: {count}")
        print()

        print("ANALYSIS DATASETS:")
        print(f"  ML dataset: {len(self.ml_dataset)} patients × {len(self.ml_dataset.columns)} features")
        print(f"  PopPK dataset: {len(self.popk_dataset)} records (NONMEM format)")
        print()

        print("=" * 80)
        print("PREPROCESSING COMPLETE - DATA READY FOR ANALYSIS!")
        print("=" * 80)


def main():
    """Main preprocessing pipeline."""

    # Initialize preprocessor
    preprocessor = AminoglycosidePreprocessor(data_path='data')

    # Run preprocessing steps
    preprocessor.calculate_derived_variables()
    preprocessor.calculate_time_varying_crcl()
    preprocessor.calculate_pk_pd_metrics()
    preprocessor.detect_outliers()
    preprocessor.check_data_quality()
    preprocessor.create_analysis_datasets()

    # Save processed data
    preprocessor.save_processed_data()

    # Generate summary report
    preprocessor.generate_summary_report()

    return preprocessor


if __name__ == '__main__':
    preprocessor = main()
