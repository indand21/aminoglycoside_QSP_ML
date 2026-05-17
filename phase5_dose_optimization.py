#!/usr/bin/env python3
"""
Phase 5: Bayesian Dose Optimization

Uses mechanistic two-compartment PK model with individual Bayesian
parameters from Phase 2/3, combined with Phase 4 ML models for
outcome prediction.

Key fixes from original version:
1. Correct model paths (models/enhanced/)
2. Mechanistic PK model using individual Bayesian parameters
3. Feature vectors matching Phase 4 training exactly (54 post-dose, 25 pre-dose)
4. StandardScaler reconstruction from training data
5. Proper prediction chain: dose -> PK model -> PK/PD indices -> ML outcomes

Author: Aminoglycoside QSP-ML Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DoseOptimizer:
    """
    Bayesian Dose Optimization for Aminoglycosides

    Combines mechanistic two-compartment PK with ML outcome prediction
    to find personalized optimal doses.
    """

    # Feature names matching Phase 4 training (from performance_enhanced.json)
    PREDOSE_FEATURES = [
        'age', 'sex', 'weight', 'height', 'bmi',
        'apache_ii', 'sofa_score', 'baseline_crcl', 'baseline_scr', 'baseline_egfr',
        'baseline_albumin', 'baseline_bilirubin', 'diabetes', 'ckd_stage',
        'mechanical_ventilation', 'vasopressor_use', 'sepsis_type', 'infection_site',
        'drug',
        'age_x_baseline_scr', 'diabetes_x_baseline_crcl',
        'severity_composite', 'renal_score', 'elderly', 'impaired_renal', 'obese'
    ]

    POSTDOSE_FEATURES = [
        'age', 'sex', 'weight', 'height', 'bmi',
        'apache_ii', 'sofa_score', 'baseline_crcl', 'baseline_scr', 'baseline_egfr',
        'baseline_albumin', 'baseline_bilirubin', 'diabetes', 'ckd_stage',
        'mechanical_ventilation', 'vasopressor_use', 'sepsis_type', 'infection_site',
        'drug',
        'Cmax', 'Cmin', 'AUC24', 'Cmax_MIC', 'AUC_MIC',
        'time_above_MIC', 'time_above_MIC_percent', 'AUC_above_MIC',
        'fluctuation_index', 'time_to_peak', 'peak_trough_ratio',
        'CL_bayes', 'Vc_bayes', 'Q_bayes', 'Vp_bayes',
        'Vss_bayes', 'k10_bayes', 'k12_bayes', 'k21_bayes',
        't_half_alpha', 't_half_beta', 'CL_cv', 'Vc_cv',
        'pk_composite', 'pk_ratio', 'target_score',
        'Cmax_x_weight', 'AUC_MIC_x_apache_ii', 'Cmax_MIC_x_baseline_crcl',
        'age_x_baseline_scr', 'diabetes_x_baseline_crcl',
        'severity_composite', 'renal_score', 'elderly', 'impaired_renal', 'obese'
    ]

    def __init__(self,
                 ml_data_path='data/processed/ml_dataset.csv',
                 pkpd_data_path='results/phase3_pkpd/pkpd_indices.csv'):
        print("=" * 80)
        print("Phase 5: Bayesian Dose Optimization")
        print("=" * 80)
        print()

        # ------------------------------------------------------------------
        # 1. Load and merge data
        # ------------------------------------------------------------------
        print("Loading data...")
        ml_data = pd.read_csv(ml_data_path)
        pkpd_data = pd.read_csv(pkpd_data_path)

        # Columns needed from PK/PD indices (individual PK params + PD features)
        pkpd_needed = [
            'patient_id',
            'Cmax', 'Cmin', 'AUC24', 'MIC', 'Cmax_MIC', 'AUC_MIC',
            'time_above_MIC', 'time_above_MIC_percent', 'AUC_above_MIC',
            'fluctuation_index', 'time_to_peak', 'peak_trough_ratio',
            'CL_bayes', 'Vc_bayes', 'Q_bayes', 'Vp_bayes',
            'Vss_bayes', 'k10_bayes', 'k12_bayes', 'k21_bayes',
            't_half_alpha', 't_half_beta', 'CL_cv', 'Vc_cv',
        ]
        available_pkpd = [c for c in pkpd_needed if c in pkpd_data.columns]

        self.data = ml_data.merge(
            pkpd_data[available_pkpd],
            on='patient_id', how='left', suffixes=('', '_pkpd')
        )

        # Encode categoricals identically to Phase 4
        for col in ['sex', 'sepsis_type', 'infection_site', 'ckd_stage', 'drug']:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes

        # Convert booleans to int (diabetes, mechanical_ventilation, etc.)
        for col in self.data.select_dtypes(include=['bool']).columns:
            self.data[col] = self.data[col].astype(int)

        print(f"  [OK] Loaded and merged data: {self.data.shape}")
        print()

        # ------------------------------------------------------------------
        # 2. Load Phase 4 ML models (correct paths)
        # ------------------------------------------------------------------
        print("Loading ML models...")
        self.models = {}
        model_map = {
            'nephrotoxicity_predose':  'models/enhanced/nephrotoxicity_predose_enhanced.json',
            'nephrotoxicity_postdose': 'models/enhanced/nephrotoxicity_postdose_enhanced.json',
            'clinical_cure':           'models/enhanced/clinical_cure_enhanced.json',
            'Cmax_surrogate':          'models/enhanced/Cmax_surrogate_enhanced.json',
            'AUC24_surrogate':         'models/enhanced/AUC24_surrogate_enhanced.json',
        }
        for name, path in model_map.items():
            p = Path(path)
            if p.exists():
                if 'surrogate' in name:
                    model = xgb.XGBRegressor()
                else:
                    model = xgb.XGBClassifier()
                model.load_model(str(p))
                self.models[name] = model
                print(f"  [OK] Loaded {name}")
            else:
                print(f"  [!!] {name} not found at {path}")
        print()

        # ------------------------------------------------------------------
        # 3. Reconstruct StandardScalers matching Phase 4 training
        # ------------------------------------------------------------------
        print("Reconstructing feature scalers...")
        self.scalers = {}
        self._build_scalers()
        print()

        # ------------------------------------------------------------------
        # 4. Optimisation settings
        # ------------------------------------------------------------------
        self.results_dir = Path('results/phase5_optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.dose_range = (200, 2000)        # mg (amikacin doses can be higher)
        self.target_cmax_mic = 8
        self.target_auc_mic = 80
        self.max_trough_gent = 2.0           # mg/L (gentamicin)
        self.max_trough_ami = 2.5            # mg/L (amikacin)
        self.max_trough = 2.0                # default; overridden per patient
        self.infusion_duration = 1.0         # hours
        self.dosing_interval = 24.0          # hours

        self.weights = {
            'cure': 0.4,
            'safety': 0.3,
            'cmax_target': 0.2,
            'trough_safety': 0.1,
        }

        self.optimization_results = None

    # ==================================================================
    # SCALER RECONSTRUCTION
    # ==================================================================

    def _build_scalers(self):
        """Reconstruct StandardScalers by replaying Phase 4 feature pipeline."""
        # Nephrotoxicity models (stratified by nephrotoxicity)
        for label, features in [('predose', self.PREDOSE_FEATURES),
                                ('postdose', self.POSTDOSE_FEATURES)]:
            X = self._build_feature_matrix(features)
            if X is not None:
                y = self.data['nephrotoxicity'].astype(int)
                X_train, _, _, _ = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
                scaler = StandardScaler()
                scaler.fit(X_train)
                self.scalers[label] = scaler
                print(f"  [OK] Built {label} scaler ({X.shape[1]} features)")

        # Clinical-cure model (stratified by clinical_cure)
        X_post = self._build_feature_matrix(self.POSTDOSE_FEATURES)
        if X_post is not None and 'clinical_cure' in self.data.columns:
            y_cure = self.data['clinical_cure'].astype(int)
            X_train, _, _, _ = train_test_split(
                X_post, y_cure, test_size=0.2, random_state=42, stratify=y_cure)
            scaler = StandardScaler()
            scaler.fit(X_train)
            self.scalers['cure'] = scaler
            print(f"  [OK] Built cure scaler ({X_post.shape[1]} features)")

    def _build_feature_matrix(self, feature_names):
        """Build the feature matrix for the entire dataset."""
        cols = {}
        for feat in feature_names:
            if feat in self.data.columns:
                cols[feat] = self.data[feat].values.astype(float)
            else:
                cols[feat] = self._compute_engineered_column(feat)
        df = pd.DataFrame(cols)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for c in df.columns:
            if df[c].isnull().any():
                df[c].fillna(df[c].median(), inplace=True)
        return df.values

    def _compute_engineered_column(self, feat):
        """Compute one engineered feature for the full dataset."""
        d = self.data
        if feat == 'age_x_baseline_scr':
            return (d['age'] * d['baseline_scr']).astype(float).values
        if feat == 'diabetes_x_baseline_crcl':
            return (d['diabetes'].astype(float) * d['baseline_crcl']).values
        if feat == 'severity_composite':
            return (d['apache_ii'] + d['sofa_score']).astype(float).values
        if feat == 'renal_score':
            return (d['baseline_crcl'] / (d['baseline_scr'] + 0.1)).astype(float).values
        if feat == 'elderly':
            return (d['age'] > 65).astype(float).values
        if feat == 'impaired_renal':
            return (d['baseline_crcl'] < 50).astype(float).values
        if feat == 'obese':
            return (d['bmi'] > 30).astype(float).values
        if feat == 'pk_composite':
            return (d['Cmax_MIC'] * d['AUC_MIC']).astype(float).values
        if feat == 'pk_ratio':
            return (d['Cmax_MIC'] / (d['AUC_MIC'] + 1)).astype(float).values
        if feat == 'target_score':
            return ((d['Cmax_MIC'] >= 8).astype(int)
                    + (d['AUC_MIC'] >= 80).astype(int)
                    + (d['Cmin'] < 2).astype(int)).astype(float).values
        if feat == 'Cmax_x_weight':
            return (d['Cmax'] * d['weight']).astype(float).values
        if feat == 'AUC_MIC_x_apache_ii':
            return (d['AUC_MIC'] * d['apache_ii']).astype(float).values
        if feat == 'Cmax_MIC_x_baseline_crcl':
            return (d['Cmax_MIC'] * d['baseline_crcl']).astype(float).values
        return np.zeros(len(d))

    # ==================================================================
    # MECHANISTIC TWO-COMPARTMENT PK MODEL
    # ==================================================================

    def compute_pk_profile(self, patient, dose):
        """
        Compute concentration-time profile and PK/PD indices using the
        analytical two-compartment IV-infusion model with individual
        Bayesian PK parameters.

        Falls back to linear PK scaling if individual parameters are missing.
        """
        has_pk = (
            not pd.isna(patient.get('CL_bayes', np.nan))
            and not pd.isna(patient.get('Vc_bayes', np.nan))
            and patient.get('CL_bayes', 0) > 0
            and patient.get('Vc_bayes', 0) > 0
        )
        if has_pk:
            return self._pk_mechanistic(patient, dose)
        return self._pk_linear_scaling(patient, dose)

    def _pk_mechanistic(self, patient, dose):
        """Two-compartment analytical PK profile."""
        CL = max(float(patient['CL_bayes']), 0.1)
        Vc = max(float(patient['Vc_bayes']), 0.1)
        Q  = max(float(patient.get('Q_bayes', 1.0)), 0.01)
        Vp = max(float(patient.get('Vp_bayes', 5.0)), 0.1)

        k10 = CL / Vc
        k12 = Q / Vc
        k21 = Q / Vp

        s = k10 + k12 + k21
        disc = max(s ** 2 - 4 * k10 * k21, 0.0)
        sqrt_d = np.sqrt(disc)
        alpha = 0.5 * (s + sqrt_d)
        beta  = 0.5 * (s - sqrt_d)
        if alpha < beta:
            alpha, beta = beta, alpha
        if alpha == beta:
            alpha += 1e-6

        A = (alpha - k21) / (alpha - beta)
        B = 1.0 - A

        tau = self.infusion_duration
        T   = self.dosing_interval
        R0  = dose / tau                       # mg/h

        # Drug-specific MIC (after Categorical encoding: amikacin=0, gentamicin=1)
        is_amikacin = (patient.get('drug', 1) == 0)
        if is_amikacin and 'mic_amikacin' in patient.index:
            mic = float(patient.get('mic_amikacin', 1.0))
        else:
            mic = float(patient.get('mic_gentamicin', patient.get('MIC', 1.0)))
        if pd.isna(mic) or mic <= 0:
            mic = 1.0

        # Vectorised concentration computation
        dt = 0.1
        times = np.arange(0, T + dt, dt)
        conc  = np.zeros_like(times)

        inf_mask  = times <= tau
        post_mask = ~inf_mask
        t_inf = times[inf_mask]
        t_post = times[post_mask] - tau

        conc[inf_mask] = (R0 / Vc) * (
            A / alpha * (1 - np.exp(-alpha * t_inf))
            + B / beta  * (1 - np.exp(-beta  * t_inf))
        )
        conc[post_mask] = (R0 / Vc) * (
            A / alpha * (1 - np.exp(-alpha * tau)) * np.exp(-alpha * t_post)
            + B / beta  * (1 - np.exp(-beta  * tau)) * np.exp(-beta  * t_post)
        )
        conc = np.maximum(conc, 0.0)

        return self._pk_indices(conc, times, mic, dt, T)

    def _pk_linear_scaling(self, patient, dose):
        """Fallback: scale observed PK linearly with dose ratio."""
        obs_dose = float(patient.get('first_dose', patient.get('dose', dose)))
        if obs_dose <= 0:
            obs_dose = dose
        ratio = dose / obs_dose

        # Drug-specific MIC (after Categorical encoding: amikacin=0, gentamicin=1)
        is_amikacin = (patient.get('drug', 1) == 0)
        if is_amikacin and 'mic_amikacin' in patient.index:
            mic = float(patient.get('mic_amikacin', 1.0))
        else:
            mic = float(patient.get('mic_gentamicin', patient.get('MIC', 1.0)))
        if pd.isna(mic) or mic <= 0:
            mic = 1.0

        Cmax = float(patient.get('Cmax', patient.get('observed_cmax', 50.0))) * ratio
        Cmin = float(patient.get('Cmin', patient.get('observed_cmin', 0.5))) * ratio
        AUC24 = float(patient.get('AUC24', 500.0)) * ratio

        # Approximate time-based features (keep ratios constant)
        t_above = float(patient.get('time_above_MIC', 15.0))
        t_above_pct = float(patient.get('time_above_MIC_percent', 60.0))
        auc_above = float(patient.get('AUC_above_MIC', 300.0)) * ratio
        fluct = float(patient.get('fluctuation_index', 200.0))
        ttp = float(patient.get('time_to_peak', 1.0))
        ptr = float(patient.get('peak_trough_ratio', 200.0))

        Cmax_MIC = Cmax / mic
        AUC_MIC  = AUC24 / mic

        return {
            'Cmax': Cmax, 'Cmin': Cmin, 'AUC24': AUC24,
            'Cmax_MIC': Cmax_MIC, 'AUC_MIC': AUC_MIC,
            'time_above_MIC': t_above,
            'time_above_MIC_percent': t_above_pct,
            'AUC_above_MIC': auc_above,
            'fluctuation_index': fluct,
            'time_to_peak': ttp,
            'peak_trough_ratio': ptr,
        }

    @staticmethod
    def _pk_indices(conc, times, mic, dt, T):
        """Derive PK/PD indices from a concentration-time curve."""
        Cmax = float(np.max(conc))
        Cmin = float(conc[-1])
        AUC24 = float(np.trapz(conc, times))

        Cmax_MIC = Cmax / mic
        AUC_MIC  = AUC24 / mic

        above = conc > mic
        time_above_MIC = float(np.sum(above) * dt)
        time_above_MIC_pct = time_above_MIC / T * 100

        AUC_above_MIC = float(np.trapz(np.maximum(conc - mic, 0), times))

        Cmin_safe = max(Cmin, 0.001)
        fluct = (Cmax - Cmin) / Cmin_safe
        ttp   = float(times[np.argmax(conc)])
        ptr   = Cmax / Cmin_safe

        return {
            'Cmax': Cmax, 'Cmin': Cmin, 'AUC24': AUC24,
            'Cmax_MIC': Cmax_MIC, 'AUC_MIC': AUC_MIC,
            'time_above_MIC': time_above_MIC,
            'time_above_MIC_percent': time_above_MIC_pct,
            'AUC_above_MIC': AUC_above_MIC,
            'fluctuation_index': fluct,
            'time_to_peak': ttp,
            'peak_trough_ratio': ptr,
        }

    # ==================================================================
    # FEATURE VECTOR BUILDING (single patient, single dose)
    # ==================================================================

    def build_feature_vector(self, patient, pk, feature_type='postdose'):
        """
        Build the feature vector for one patient at one dose, matching
        Phase 4 training features exactly.
        """
        names = (self.PREDOSE_FEATURES if feature_type == 'predose'
                 else self.POSTDOSE_FEATURES)

        values = []
        for feat in names:
            if feat in pk:
                values.append(float(pk[feat]))
            elif feat in patient.index:
                v = patient[feat]
                values.append(float(v) if not pd.isna(v) else 0.0)
            # --- engineered features ---
            elif feat == 'pk_composite':
                values.append(pk['Cmax_MIC'] * pk['AUC_MIC'])
            elif feat == 'pk_ratio':
                values.append(pk['Cmax_MIC'] / (pk['AUC_MIC'] + 1))
            elif feat == 'target_score':
                values.append(float(
                    (1 if pk['Cmax_MIC'] >= 8 else 0)
                    + (1 if pk['AUC_MIC'] >= 80 else 0)
                    + (1 if pk['Cmin'] < 2 else 0)))
            elif feat == 'Cmax_x_weight':
                values.append(pk['Cmax'] * float(patient['weight']))
            elif feat == 'AUC_MIC_x_apache_ii':
                values.append(pk['AUC_MIC'] * float(patient['apache_ii']))
            elif feat == 'Cmax_MIC_x_baseline_crcl':
                values.append(pk['Cmax_MIC'] * float(patient['baseline_crcl']))
            elif feat == 'age_x_baseline_scr':
                values.append(float(patient['age']) * float(patient['baseline_scr']))
            elif feat == 'diabetes_x_baseline_crcl':
                values.append(float(patient['diabetes']) * float(patient['baseline_crcl']))
            elif feat == 'severity_composite':
                values.append(float(patient['apache_ii']) + float(patient['sofa_score']))
            elif feat == 'renal_score':
                values.append(float(patient['baseline_crcl']) / (float(patient['baseline_scr']) + 0.1))
            elif feat == 'elderly':
                values.append(1.0 if patient['age'] > 65 else 0.0)
            elif feat == 'impaired_renal':
                values.append(1.0 if patient['baseline_crcl'] < 50 else 0.0)
            elif feat == 'obese':
                values.append(1.0 if patient['bmi'] > 30 else 0.0)
            else:
                values.append(0.0)

        arr = np.array(values, dtype=np.float64).reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    # ==================================================================
    # OUTCOME PREDICTION
    # ==================================================================

    @staticmethod
    def _safe_predict_proba(model, X):
        """predict_proba with fallback to sigmoid(output_margin)."""
        try:
            return float(model.predict_proba(X)[0, 1])
        except Exception:
            try:
                margin = float(model.predict(X, output_margin=True)[0])
                return 1.0 / (1.0 + np.exp(-margin))
            except Exception:
                return 0.5

    def predict_outcomes(self, patient, dose):
        """
        Full prediction chain: dose -> mechanistic PK -> features -> ML.
        """
        pk = self.compute_pk_profile(patient, dose)
        outcomes = dict(pk)

        # Build & scale feature vectors
        post_raw = self.build_feature_vector(patient, pk, 'postdose')
        pre_raw  = self.build_feature_vector(patient, pk, 'predose')

        post_sc = (self.scalers['postdose'].transform(post_raw)
                   if 'postdose' in self.scalers else post_raw)
        pre_sc  = (self.scalers['predose'].transform(pre_raw)
                   if 'predose' in self.scalers else pre_raw)
        cure_sc = (self.scalers['cure'].transform(post_raw)
                   if 'cure' in self.scalers else post_sc)

        # --- Nephrotoxicity (prefer post-dose model) ---
        if 'nephrotoxicity_postdose' in self.models:
            outcomes['p_nephrotoxicity'] = self._safe_predict_proba(
                self.models['nephrotoxicity_postdose'], post_sc)
        elif 'nephrotoxicity_predose' in self.models:
            outcomes['p_nephrotoxicity'] = self._safe_predict_proba(
                self.models['nephrotoxicity_predose'], pre_sc)
        else:
            # Phase 3 logistic fallback
            outcomes['p_nephrotoxicity'] = 1.0 / (
                1.0 + np.exp(-(-1.978 + 1.221 * pk['Cmin'])))

        # --- Clinical cure ---
        if 'clinical_cure' in self.models:
            outcomes['p_cure'] = self._safe_predict_proba(
                self.models['clinical_cure'], cure_sc)
        else:
            outcomes['p_cure'] = 1.0 / (
                1.0 + np.exp(-(1.820 + 0.016 * pk['Cmax_MIC'])))

        # Drug-specific trough target (amikacin=0 after encoding)
        is_ami = (patient.get('drug', 1) == 0)
        trough_limit = self.max_trough_ami if is_ami else self.max_trough_gent

        # Target-attainment flags
        outcomes['cmax_target_met'] = int(pk['Cmax_MIC'] >= self.target_cmax_mic)
        outcomes['auc_target_met']  = int(pk['AUC_MIC']  >= self.target_auc_mic)
        outcomes['trough_safe']     = int(pk['Cmin']      <  trough_limit)

        return outcomes

    # ==================================================================
    # OBJECTIVE FUNCTION
    # ==================================================================

    def objective_function(self, patient, dose):
        """Multi-objective score (higher = better)."""
        o = self.predict_outcomes(patient, dose)

        cure   = o['p_cure']
        safety = 1.0 - o['p_nephrotoxicity']
        cmax_t = 1.0 / (1.0 + np.exp(-0.5 * (o['Cmax_MIC'] - self.target_cmax_mic)))
        trough = 1.0 / (1.0 + np.exp(2.0 * (o['Cmin'] - self.max_trough)))

        return (self.weights['cure']          * cure
                + self.weights['safety']       * safety
                + self.weights['cmax_target']  * cmax_t
                + self.weights['trough_safety'] * trough)

    # ==================================================================
    # OPTIMISATION METHODS
    # ==================================================================

    def optimize_dose_grid(self, patient, n_points=50):
        """Grid-search dose optimisation."""
        doses  = np.linspace(self.dose_range[0], self.dose_range[1], n_points)
        scores = np.array([self.objective_function(patient, d) for d in doses])
        idx    = int(np.argmax(scores))
        return doses[idx], scores[idx], doses, scores

    def optimize_dose_bayesian(self, patient, n_iter=25):
        """Bayesian optimisation (requires bayes_opt package)."""
        def obj(dose):
            return self.objective_function(patient, dose)

        optimizer = BayesianOptimization(
            f=obj,
            pbounds={'dose': self.dose_range},
            random_state=42,
            allow_duplicate_points=True,
        )
        optimizer.maximize(init_points=5, n_iter=n_iter)
        return (optimizer.max['params']['dose'],
                optimizer.max['target'],
                optimizer)

    # ==================================================================
    # BATCH OPTIMISATION
    # ==================================================================

    def optimize_all_patients(self, method='grid', n_samples=50):
        """Optimise doses for a sample of patients."""
        print("=" * 80)
        print(f"DOSE OPTIMISATION ({method.upper()} SEARCH)")
        print("=" * 80)
        print()

        if n_samples < len(self.data):
            sample = self.data.sample(n=n_samples, random_state=42)
            print(f"Optimising doses for {n_samples} sampled patients...")
        else:
            sample = self.data
            print(f"Optimising doses for all {len(sample)} patients...")

        results = []
        for idx, (_, patient) in enumerate(sample.iterrows()):
            if method == 'bayesian' and BAYESOPT_AVAILABLE:
                opt_dose, max_score, _ = self.optimize_dose_bayesian(patient, n_iter=20)
            else:
                opt_dose, max_score, _, _ = self.optimize_dose_grid(patient, n_points=50)

            opt_out = self.predict_outcomes(patient, opt_dose)

            obs_dose = float(patient.get('first_dose', patient.get('dose', 600)))
            obs_out  = self.predict_outcomes(patient, obs_dose)
            obs_score = self.objective_function(patient, obs_dose)

            results.append({
                'patient_id':       patient['patient_id'],
                'age':              patient['age'],
                'weight':           patient['weight'],
                'baseline_crcl':    patient['baseline_crcl'],
                'apache_ii':        patient['apache_ii'],
                'mic':              patient.get('mic_gentamicin', patient.get('MIC', 1.0)),
                # observed
                'observed_dose':    obs_dose,
                'observed_p_cure':  obs_out['p_cure'],
                'observed_p_aki':   obs_out['p_nephrotoxicity'],
                'observed_cmax_mic': obs_out['Cmax_MIC'],
                'observed_score':   obs_score,
                # optimal
                'optimal_dose':     opt_dose,
                'optimal_p_cure':   opt_out['p_cure'],
                'optimal_p_aki':    opt_out['p_nephrotoxicity'],
                'optimal_cmax':     opt_out['Cmax'],
                'optimal_cmin':     opt_out['Cmin'],
                'optimal_auc24':    opt_out['AUC24'],
                'optimal_cmax_mic': opt_out['Cmax_MIC'],
                'optimal_auc_mic':  opt_out['AUC_MIC'],
                'optimal_score':    max_score,
                # deltas
                'dose_change':      opt_dose - obs_dose,
                'cure_improvement': opt_out['p_cure'] - obs_out['p_cure'],
                'aki_reduction':    obs_out['p_nephrotoxicity'] - opt_out['p_nephrotoxicity'],
                'score_improvement': max_score - obs_score,
            })

            if (idx + 1) % 10 == 0:
                print(f"  Optimised {idx + 1}/{len(sample)} patients...")

        self.optimization_results = pd.DataFrame(results)

        print(f"\n[OK] Optimisation complete for {len(self.optimization_results)} patients")
        print()

        # Summary
        r = self.optimization_results
        print("Summary Statistics:")
        print(f"  Mean observed dose:  {r['observed_dose'].mean():.0f} mg")
        print(f"  Mean optimal dose:   {r['optimal_dose'].mean():.0f} mg")
        print(f"  Mean dose change:    {r['dose_change'].mean():+.0f} mg "
              f"({r['dose_change'].mean() / r['observed_dose'].mean() * 100:+.1f}%)")
        print()
        print(f"  Mean cure improvement:  {r['cure_improvement'].mean() * 100:+.2f}%")
        print(f"  Mean AKI reduction:     {r['aki_reduction'].mean() * 100:+.2f}%")
        print(f"  Mean score improvement: {r['score_improvement'].mean():+.3f}")
        print()
        sig = (r['score_improvement'] > 0.01).sum()
        print(f"  Patients improved (Dscore > 0.01): {sig} "
              f"({sig / len(r) * 100:.1f}%)")
        print()

        r.to_csv(self.results_dir / 'dose_recommendations.csv', index=False)
        print(f"[OK] Saved dose recommendations to: "
              f"{self.results_dir / 'dose_recommendations.csv'}")
        print()
        return r

    # ==================================================================
    # DOSING NOMOGRAM
    # ==================================================================

    def create_dosing_nomogram(self):
        """Create a dosing nomogram over weight x CrCL grid."""
        print("Creating dosing nomogram...")
        weights = np.linspace(40, 100, 7)
        crcls   = np.array([30, 50, 75, 100, 130])

        rows = []
        for wt in weights:
            for crcl in crcls:
                patient = pd.Series({
                    'age': 60, 'sex': 1, 'weight': wt, 'height': 170,
                    'bmi': wt / 1.7 ** 2, 'apache_ii': 20, 'sofa_score': 6,
                    'baseline_crcl': crcl,
                    'baseline_scr': 140 / crcl if crcl > 0 else 2.0,
                    'baseline_egfr': crcl, 'baseline_albumin': 3.0,
                    'baseline_bilirubin': 1.0, 'diabetes': 0, 'ckd_stage': 0,
                    'mechanical_ventilation': 0, 'vasopressor_use': 0,
                    'sepsis_type': 0, 'infection_site': 0,
                    'mic_gentamicin': 1.0, 'first_dose': 600,
                    # Representative individual PK params (population median)
                    'CL_bayes': 5.7 * (crcl / 100) ** 0.75 * (wt / 70) ** 0.75,
                    'Vc_bayes': 16.8 * (wt / 70),
                    'Q_bayes': 13.5, 'Vp_bayes': 11.4,
                    'Vss_bayes': 16.8 * (wt / 70) + 11.4,
                    'k10_bayes': 5.7 * (crcl / 100) ** 0.75 * (wt / 70) ** 0.75 / (16.8 * (wt / 70)),
                    'k12_bayes': 13.5 / (16.8 * (wt / 70)),
                    'k21_bayes': 13.5 / 11.4,
                    't_half_alpha': 0.3, 't_half_beta': 4.0,
                    'CL_cv': 0.24, 'Vc_cv': 0.20,
                    'Cmax': 50.0, 'Cmin': 0.5, 'AUC24': 500.0,
                    'Cmax_MIC': 50.0, 'AUC_MIC': 500.0,
                    'time_above_MIC': 15, 'time_above_MIC_percent': 62,
                    'AUC_above_MIC': 300, 'fluctuation_index': 200,
                    'time_to_peak': 1.0, 'peak_trough_ratio': 200,
                })

                opt_dose, _, _, _ = self.optimize_dose_grid(patient, n_points=30)
                rows.append({'weight': wt, 'crcl': crcl,
                             'recommended_dose': opt_dose,
                             'dose_per_kg': opt_dose / wt})

        nomogram = pd.DataFrame(rows)
        nomogram.to_csv(self.results_dir / 'dosing_nomogram.csv', index=False)
        print(f"  [OK] Saved dosing nomogram to: "
              f"{self.results_dir / 'dosing_nomogram.csv'}")
        print()
        return nomogram

    # ==================================================================
    # VISUALISATIONS
    # ==================================================================

    def generate_visualizations(self):
        """Generate optimisation visualisations."""
        print("=" * 80)
        print("Generating visualisations...")
        print("=" * 80)
        print()

        r = self.optimization_results
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.size'] = 10

        # --- 1. Dose comparison ---
        print("  Creating dose comparison plot...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(r['observed_dose'], bins=30, alpha=0.5,
                     label='Observed', edgecolor='black')
        axes[0].hist(r['optimal_dose'], bins=30, alpha=0.5,
                     label='Optimal', edgecolor='black')
        axes[0].set_xlabel('Dose (mg)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Dose Distribution: Observed vs Optimal')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(r['dose_change'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', label='No change')
        axes[1].set_xlabel('Dose Change (mg)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Dose Changes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(r['observed_dose'], r['optimal_dose'], alpha=0.5, s=30)
        axes[2].plot([200, 1600], [200, 1600], 'r--', label='No change')
        axes[2].set_xlabel('Observed Dose (mg)')
        axes[2].set_ylabel('Optimal Dose (mg)')
        axes[2].set_title('Observed vs Optimal Dose')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'dose_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # --- 2. Outcome improvements ---
        print("  Creating outcome improvement plots...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(r['cure_improvement'] * 100, bins=30,
                     edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', label='No improvement')
        axes[0].set_xlabel('Cure Probability Improvement (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Clinical Cure Improvement')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(r['aki_reduction'] * 100, bins=30,
                     edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', label='No reduction')
        axes[1].set_xlabel('Nephrotoxicity Reduction (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Nephrotoxicity Risk Reduction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(r['score_improvement'], bins=30,
                     edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='red', linestyle='--', label='No improvement')
        axes[2].set_xlabel('Objective Score Improvement')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Overall Optimisation Score Improvement')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'outcome_improvements.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # --- 3. Target attainment comparison ---
        print("  Creating target attainment comparison...")
        fig, ax = plt.subplots(figsize=(10, 6))

        obs_cmax = (r['observed_cmax_mic'] >= 8).mean() * 100
        opt_cmax = (r['optimal_cmax_mic'] >= 8).mean() * 100
        opt_auc  = (r['optimal_auc_mic'] >= 80).mean() * 100
        opt_safe = (r['optimal_cmin'] < 2).mean() * 100

        categories = ['Cmax/MIC >=8\n(Efficacy)',
                      'AUC/MIC >=80\n(Efficacy)',
                      'Trough <2\n(Safety)']
        observed = [obs_cmax, np.nan, np.nan]
        optimal  = [opt_cmax, opt_auc, opt_safe]

        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w / 2, observed, w, label='Observed Dose', alpha=0.8)
        ax.bar(x + w / 2, optimal,  w, label='Optimal Dose',  alpha=0.8)
        ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% Target')
        ax.set_ylabel('Target Attainment (%)')
        ax.set_title('PK/PD Target Attainment: Observed vs Optimal')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'target_attainment_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # --- 4. Dose by patient characteristics ---
        print("  Creating dose-by-characteristics plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        for ax, (col, label) in zip(
                axes.flat,
                [('weight', 'Weight (kg)'),
                 ('baseline_crcl', 'Baseline CrCL (mL/min)'),
                 ('apache_ii', 'APACHE II Score'),
                 ('mic', 'MIC (mg/L)')]):
            ax.scatter(r[col], r['optimal_dose'], alpha=0.5, s=30)
            ax.set_xlabel(label)
            ax.set_ylabel('Optimal Dose (mg)')
            ax.set_title(f'Optimal Dose by {label}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'dose_by_characteristics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # --- 5. Example dose-response curves ---
        print("  Creating example dose-response curves...")
        example = self.data.iloc[0]
        doses   = np.linspace(200, 1600, 100)
        outs    = [self.predict_outcomes(example, d) for d in doses]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(doses, [o['p_cure'] for o in outs], 'b-', lw=2)
        axes[0, 0].set_xlabel('Dose (mg)')
        axes[0, 0].set_ylabel('P(Clinical Cure)')
        axes[0, 0].set_title('Dose-Response: Clinical Cure')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(doses, [o['p_nephrotoxicity'] for o in outs], 'r-', lw=2)
        axes[0, 1].set_xlabel('Dose (mg)')
        axes[0, 1].set_ylabel('P(Nephrotoxicity)')
        axes[0, 1].set_title('Dose-Response: Nephrotoxicity')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(doses, [o['Cmax_MIC'] for o in outs], 'g-', lw=2)
        axes[1, 0].axhline(8, color='red', linestyle='--', label='Target >= 8')
        axes[1, 0].set_xlabel('Dose (mg)')
        axes[1, 0].set_ylabel('Cmax/MIC')
        axes[1, 0].set_title('Dose-Response: Cmax/MIC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        scores = [self.objective_function(example, d) for d in doses]
        best_i = int(np.argmax(scores))
        axes[1, 1].plot(doses, scores, color='purple', lw=2)
        axes[1, 1].axvline(doses[best_i], color='red', linestyle='--',
                           label=f'Optimal: {doses[best_i]:.0f} mg')
        axes[1, 1].set_xlabel('Dose (mg)')
        axes[1, 1].set_ylabel('Objective Score')
        axes[1, 1].set_title('Dose-Response: Overall Objective')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'dose_response_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[OK] All visualisations saved to: {self.results_dir}/")
        print()

    # ==================================================================
    # SUMMARY REPORT
    # ==================================================================

    def generate_summary_report(self):
        """Generate a comprehensive text summary."""
        print("Generating summary report...")
        r = self.optimization_results

        lines = []
        lines.append("=" * 80)
        lines.append("Phase 5: Bayesian Dose Optimisation - Summary Report")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Number of Patients Optimised: {len(r)}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("1. OPTIMISATION PARAMETERS")
        lines.append("=" * 80)
        lines.append(f"Dose Range: {self.dose_range[0]}-{self.dose_range[1]} mg")
        lines.append(f"Target Cmax/MIC: >= {self.target_cmax_mic}")
        lines.append(f"Target AUC/MIC:  >= {self.target_auc_mic}")
        lines.append(f"Maximum Trough:  <  {self.max_trough} mg/L")
        lines.append("")
        lines.append("Objective Weights:")
        for k, v in self.weights.items():
            lines.append(f"  {k}: {v:.0%}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("2. DOSE RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Observed Dose: {r['observed_dose'].mean():.0f} "
                     f"+/- {r['observed_dose'].std():.0f} mg")
        lines.append(f"Optimal Dose:  {r['optimal_dose'].mean():.0f} "
                     f"+/- {r['optimal_dose'].std():.0f} mg")
        lines.append(f"Mean Change:   {r['dose_change'].mean():+.0f} mg "
                     f"({r['dose_change'].mean() / r['observed_dose'].mean() * 100:+.1f}%)")
        lines.append("")

        dc = r['dose_change']
        lines.append("Dose Change Distribution:")
        lines.append(f"  Increase (>50 mg):  {(dc > 50).sum()} patients "
                     f"({(dc > 50).mean() * 100:.1f}%)")
        lines.append(f"  Similar (+/-50 mg): {((dc >= -50) & (dc <= 50)).sum()} patients "
                     f"({((dc >= -50) & (dc <= 50)).mean() * 100:.1f}%)")
        lines.append(f"  Decrease (<-50 mg): {(dc < -50).sum()} patients "
                     f"({(dc < -50).mean() * 100:.1f}%)")
        lines.append("")

        lines.append("=" * 80)
        lines.append("3. OUTCOME IMPROVEMENTS")
        lines.append("=" * 80)
        lines.append("")
        ci = r['cure_improvement']
        ar = r['aki_reduction']
        si = r['score_improvement']
        lines.append("Clinical Cure:")
        lines.append(f"  Mean improvement:  {ci.mean() * 100:+.2f}%")
        lines.append(f"  Patients improved: {(ci > 0).sum()} ({(ci > 0).mean() * 100:.1f}%)")
        lines.append("")
        lines.append("Nephrotoxicity Risk:")
        lines.append(f"  Mean reduction:    {ar.mean() * 100:+.2f}%")
        lines.append(f"  Patients reduced:  {(ar > 0).sum()} ({(ar > 0).mean() * 100:.1f}%)")
        lines.append("")
        lines.append("Overall Objective Score:")
        lines.append(f"  Mean improvement:  {si.mean():+.4f}")
        lines.append(f"  Patients improved: {(si > 0).sum()} ({(si > 0).mean() * 100:.1f}%)")
        lines.append("")

        lines.append("=" * 80)
        lines.append("4. TARGET ATTAINMENT")
        lines.append("=" * 80)
        lines.append("")
        obs_cm = (r['observed_cmax_mic'] >= 8).mean()
        opt_cm = (r['optimal_cmax_mic'] >= 8).mean()
        opt_am = (r['optimal_auc_mic'] >= 80).mean()
        opt_ts = (r['optimal_cmin'] < 2).mean()

        lines.append("Observed Dosing:")
        lines.append(f"  Cmax/MIC >= 8: {obs_cm * 100:.1f}%")
        lines.append("")
        lines.append("Optimal Dosing:")
        lines.append(f"  Cmax/MIC >= 8:  {opt_cm * 100:.1f}% "
                     f"({(opt_cm - obs_cm) * 100:+.1f}%)")
        lines.append(f"  AUC/MIC >= 80:  {opt_am * 100:.1f}%")
        lines.append(f"  Trough < 2:     {opt_ts * 100:.1f}%")
        lines.append("")

        combined = ((r['optimal_cmax_mic'] >= 8) & (r['optimal_cmin'] < 2)).mean()
        lines.append(f"Combined (Efficacy + Safety): {combined * 100:.1f}%")
        lines.append("")

        lines.append("=" * 80)
        lines.append("5. CLINICAL IMPLICATIONS")
        lines.append("=" * 80)
        lines.append("")
        if dc.mean() > 0:
            lines.append("  On average optimal doses are HIGHER than observed doses.")
            lines.append("  -> Current dosing may be suboptimal for efficacy.")
        else:
            lines.append("  On average optimal doses are LOWER than observed doses.")
            lines.append("  -> Current dosing may increase toxicity risk unnecessarily.")
        lines.append("")
        lines.append(f"  Personalised dosing could improve outcomes in "
                     f"{(si > 0.01).mean() * 100:.0f}% of patients.")
        lines.append("")

        lines.append("=" * 80)
        lines.append("6. FILES GENERATED")
        lines.append("=" * 80)
        lines.append("")
        lines.append("  - results/phase5_optimization/dose_recommendations.csv")
        lines.append("  - results/phase5_optimization/dosing_nomogram.csv")
        lines.append("  - results/phase5_optimization/dose_comparison.png")
        lines.append("  - results/phase5_optimization/outcome_improvements.png")
        lines.append("  - results/phase5_optimization/target_attainment_comparison.png")
        lines.append("  - results/phase5_optimization/dose_by_characteristics.png")
        lines.append("  - results/phase5_optimization/dose_response_curves.png")
        lines.append("")
        lines.append("=" * 80)
        lines.append("Phase 5 Analysis Complete!")
        lines.append("=" * 80)

        report = "\n".join(lines)
        with open(self.results_dir / 'PHASE5_SUMMARY.txt', 'w') as f:
            f.write(report)

        print(report)
        print(f"\n[OK] Summary saved to: {self.results_dir / 'PHASE5_SUMMARY.txt'}")
        print()

    # ==================================================================
    # FULL PIPELINE
    # ==================================================================

    def run_complete_pipeline(self, n_patients=50):
        """Run the complete dose optimisation pipeline."""
        print("Starting Phase 5 complete optimisation pipeline...")
        print()

        self.optimize_all_patients(method='grid', n_samples=n_patients)
        self.create_dosing_nomogram()
        self.generate_visualizations()
        self.generate_summary_report()

        print("=" * 80)
        print("PHASE 5 COMPLETE!")
        print("=" * 80)
        print()


def main():
    optimizer = DoseOptimizer()
    optimizer.run_complete_pipeline(n_patients=50)


if __name__ == "__main__":
    main()
