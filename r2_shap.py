#!/usr/bin/env python3
"""Revision 2: SHAP and gain importance for the production post-dose models.

R1's phase4_shap_dump.py scored the clinical-cure model with the
nephrotoxicity model's feature scaler (the two models were trained on
different stratified splits and therefore different scalers). This script uses
each model's own scaler. SHAP values are exact TreeSHAP values from XGBoost's
native `pred_contribs`, computed for all 1,500 patients; importance is mean
|SHAP| as a percentage of the total over all features. Gain importance
(XGBoost `feature_importances_`, normalised gain) is reported separately so the
two are not conflated.

Outputs: results/revision2/shap_importance.json, shap_values_<model>.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from r2_ml_common import OUT_DIR, production_setup


def main():
    _, setup = production_setup()
    out = {}
    for name in ['nephrotoxicity_postdose', 'clinical_cure', 'nephrotoxicity_predose']:
        e = setup[name]
        Xs = e['scaler'].transform(e['X'].values)
        booster = e['model'].get_booster()
        contrib = booster.predict(xgb.DMatrix(Xs, feature_names=None), pred_contribs=True)
        shap = contrib[:, :-1]
        mabs = np.abs(shap).mean(axis=0)
        pct = 100 * mabs / mabs.sum()
        gain = 100 * e['model'].feature_importances_ / e['model'].feature_importances_.sum()
        df = pd.DataFrame({'feature': e['features'], 'shap_pct': pct, 'mean_abs_shap': mabs,
                           'gain_pct': gain})
        # Direction: Spearman correlation between raw feature value and its SHAP value
        raw = e['X'].values
        df['shap_value_corr'] = [pd.Series(raw[:, j]).corr(pd.Series(shap[:, j]), method='spearman')
                                 for j in range(raw.shape[1])]
        df = df.sort_values('shap_pct', ascending=False).reset_index(drop=True)
        df['shap_rank'] = np.arange(1, len(df) + 1)
        df['gain_rank'] = df['gain_pct'].rank(ascending=False, method='first').astype(int)
        out[name] = df.to_dict(orient='records')
        pd.DataFrame(shap, columns=e['features']).assign(patient_id=e['patient_id']).to_csv(
            OUT_DIR / f'shap_values_{name}.csv', index=False)
        print(f'\n{name}')
        print(df.head(15)[['shap_rank', 'feature', 'shap_pct', 'gain_pct', 'gain_rank',
                           'shap_value_corr']].round(3).to_string(index=False))
    (OUT_DIR / 'shap_importance.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
