#!/usr/bin/env python3
"""Revision 2: shared helpers that reproduce the Phase 4 production setup exactly.

Feature matrices are built with Phase 4's own `prepare_features` and
`engineer_features`, the 80/20 split and StandardScaler are re-created with the
same seed, and the saved production XGBoost models are loaded from
models/enhanced/. `verify_against_saved()` confirms that the reproduced
held-out predictions equal results/phase4_ml_enhanced/test_predictions.json.

Run directly to perform that verification:
    python r2_ml_common.py
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / 'results' / 'revision2'

MODELS = {
    # name: (feature_set, outcome column, model file)
    'nephrotoxicity_predose': ('baseline', 'nephrotoxicity', 'nephrotoxicity_predose_enhanced.json'),
    'nephrotoxicity_postdose': ('pkpd', 'nephrotoxicity', 'nephrotoxicity_postdose_enhanced.json'),
    'clinical_cure': ('pkpd', 'clinical_cure', 'clinical_cure_enhanced.json'),
}

# Engineered columns created by Phase 4 engineer_features (P4:237-299)
ENGINEERED = ['pk_composite', 'pk_ratio', 'target_score', 'Cmax_x_weight',
              'AUC_MIC_x_apache_ii', 'Cmax_MIC_x_baseline_crcl', 'age_x_baseline_scr',
              'diabetes_x_baseline_crcl', 'severity_composite', 'renal_score',
              'elderly', 'impaired_renal', 'obese']


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def load_pipeline():
    import os
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)        # Phase 4 uses repo-relative paths
    try:
        from phase4_machine_learning_enhanced import EnhancedMLPipeline
        return quiet(EnhancedMLPipeline)
    finally:
        os.chdir(cwd)


def raw_features(pipe, feature_set):
    """Phase 4 feature table before engineering (categoricals coded, NaN imputed)."""
    X, names = quiet(pipe.prepare_features, feature_set, engineer=False)
    return pd.DataFrame(np.asarray(X, dtype=float), columns=names)


def production_features(name):
    perf = json.loads((REPO_ROOT / 'results' / 'phase4_ml_enhanced' /
                       'performance_enhanced.json').read_text())
    return perf[name]['features']


def engineer(pipe, raw_df, keep=None):
    """Phase 4 engineer_features. The log_* columns it tries to add were not
    created in the production run (they fail silently under the library versions
    used then), so callers pass `keep` = the stored production feature list."""
    X, names = quiet(pipe.engineer_features, raw_df.values, list(raw_df.columns),
                     use_phase2_features=False)
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=names)
    return df[keep] if keep is not None else df


def split_indices(y):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    return tr, te


def load_model(fname):
    m = xgb.XGBClassifier()
    m.load_model(str(REPO_ROOT / 'models' / 'enhanced' / fname))
    return m


def production_setup(pipe=None):
    """Return dict name -> {X (engineered DataFrame), y, train, test, scaler, model}."""
    pipe = pipe or load_pipeline()
    out = {}
    cache = {}
    for name, (fset, ycol, fname) in MODELS.items():
        if fset not in cache:
            cache[fset] = raw_features(pipe, fset)
        raw = cache[fset]
        X = engineer(pipe, raw, keep=production_features(name))
        y = pipe.data[ycol].astype(int).to_numpy()
        tr, te = split_indices(y)
        scaler = StandardScaler().fit(X.values[tr])
        out[name] = {'raw': raw, 'X': X, 'features': list(X.columns), 'y': y, 'train': tr, 'test': te,
                     'scaler': scaler, 'model': load_model(fname),
                     'patient_id': pipe.data['patient_id'].to_numpy()}
    return pipe, out


def predict(entry, X_df):
    return entry['model'].predict_proba(entry['scaler'].transform(X_df.values))[:, 1]


def verify_against_saved(setup):
    saved = json.loads((REPO_ROOT / 'results' / 'phase4_ml_enhanced' /
                        'test_predictions.json').read_text())
    ok = True
    for name, e in setup.items():
        te = e['test']
        p = predict(e, e['X'].iloc[te])
        s = saved[name]
        same_y = np.array_equal(np.asarray(s['y_test']), e['y'][te])
        maxdiff = float(np.max(np.abs(np.asarray(s['y_pred_proba']) - p)))
        print(f'  {name:<26} features {e["X"].shape[1]:>2}  y_test identical {same_y}  '
              f'max |p - saved| = {maxdiff:.2e}')
        ok &= same_y and maxdiff < 1e-5
    return ok


if __name__ == '__main__':
    _, setup = production_setup()
    print('Reproducing Phase 4 held-out predictions from saved models:')
    print('  ALL MATCH' if verify_against_saved(setup) else '  MISMATCH')
