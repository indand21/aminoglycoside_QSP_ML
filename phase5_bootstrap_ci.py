#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for the Phase 5 headline dose-optimization
results (mean dose change, absolute nephrotoxicity-probability reduction, and
clinical-cure-probability improvement).

The optimizer is evaluated on a fixed random subsample of 50 synthetic patients
(seed 42); these point estimates are therefore simulator- and surrogate-conditional.
We resample the 50 per-patient recommendations with replacement (B = 10,000) and
report percentile CIs so the manuscript can accompany each headline number with an
interval rather than a bare point estimate (Reviewer #7).

Definitions (matching results/phase5_optimization/PHASE5_SUMMARY.txt):
  - dose change %      = (mean optimal_dose - mean observed_dose) / mean observed_dose * 100
  - cure improvement % = mean(optimal_p_cure - observed_p_cure) * 100          [= cure_improvement]
  - AKI reduction %    = mean(observed_p_aki - optimal_p_aki) * 100            [= aki_reduction]
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "phase5_optimization" / "dose_recommendations.csv"
OUT = HERE / "results" / "phase5_optimization" / "bootstrap_ci.json"

# Fixed seed so the CIs are reproducible (Date/np.random elsewhere use 42).
RNG = np.random.default_rng(42)
B = 10_000


def _pct_dose_change(df):
    return (df["optimal_dose"].mean() - df["observed_dose"].mean()) / df["observed_dose"].mean() * 100.0


def _pct_cure(df):
    return df["cure_improvement"].mean() * 100.0


def _pct_aki(df):
    return df["aki_reduction"].mean() * 100.0


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    metrics = {
        "dose_change_pct": _pct_dose_change,
        "cure_improvement_pct": _pct_cure,
        "aki_reduction_pct": _pct_aki,
    }

    results = {"n_patients": int(n), "n_bootstrap": B}
    idx = RNG.integers(0, n, size=(B, n))
    for name, fn in metrics.items():
        point = float(fn(df))
        boot = np.array([fn(df.iloc[idx[b]]) for b in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results[name] = {
            "point": round(point, 3),
            "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3),
        }
        print(f"{name:22s}: {point:7.2f}  (95% CI {lo:7.2f} to {hi:7.2f})")

    # Fraction improving on the composite objective (Wilson-free simple bootstrap)
    improved = (df["score_improvement"] > 0).mean() * 100.0
    boot_imp = np.array([(df.iloc[idx[b]]["score_improvement"] > 0).mean() * 100.0 for b in range(B)])
    lo, hi = np.percentile(boot_imp, [2.5, 97.5])
    results["composite_improved_pct"] = {
        "point": round(float(improved), 3),
        "ci_low": round(float(lo), 3),
        "ci_high": round(float(hi), 3),
    }
    print(f"{'composite_improved_pct':22s}: {improved:7.2f}  (95% CI {lo:7.2f} to {hi:7.2f})")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Wrote {OUT}")


if __name__ == "__main__":
    main()
