#!/usr/bin/env python3
"""Objective-weight sensitivity analysis for Phase 5 dose optimization
(Reviewer #7). The composite objective weights (cure / safety / Cmax-target /
trough-safety) were chosen a priori; here we re-run the grid optimizer on the
same fixed 50-patient subsample (seed 42) under several weight configurations
and report how the headline recommendations shift.

The baseline configuration is run last so the canonical
results/phase5_optimization/dose_recommendations.csv is left unchanged.
"""
import json
from pathlib import Path

from phase5_dose_optimization import DoseOptimizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "phase5_optimization" / "weight_sensitivity.json"

CONFIGS = [
    ("Safety-weighted",   {"cure": 0.25, "safety": 0.45, "cmax_target": 0.15, "trough_safety": 0.15}),
    ("Efficacy-weighted", {"cure": 0.55, "safety": 0.20, "cmax_target": 0.20, "trough_safety": 0.05}),
    ("Equal weights",     {"cure": 0.25, "safety": 0.25, "cmax_target": 0.25, "trough_safety": 0.25}),
    ("Cmax-weighted",     {"cure": 0.30, "safety": 0.25, "cmax_target": 0.35, "trough_safety": 0.10}),
    ("Baseline (0.40/0.30/0.20/0.10)", {"cure": 0.40, "safety": 0.30, "cmax_target": 0.20, "trough_safety": 0.10}),
]


def summarize(df):
    pct_dose = (df["optimal_dose"].mean() - df["observed_dose"].mean()) / df["observed_dose"].mean() * 100
    return {
        "mean_dose_change_pct": round(float(pct_dose), 1),
        "mean_optimal_dose_mg": round(float(df["optimal_dose"].mean()), 0),
        "mean_aki_reduction_pct": round(float(df["aki_reduction"].mean() * 100), 1),
        "mean_cure_improvement_pct": round(float(df["cure_improvement"].mean() * 100), 2),
    }


def main():
    opt = DoseOptimizer()
    results = {}
    for name, weights in CONFIGS:
        opt.weights = dict(weights)
        df = opt.optimize_all_patients(method="grid", n_samples=50)
        results[name] = {"weights": weights, **summarize(df)}
        s = results[name]
        print(f"\n>>> {name}: dose {s['mean_dose_change_pct']:+.1f}% "
              f"(mean {s['mean_optimal_dose_mg']:.0f} mg) | "
              f"AKI {s['mean_aki_reduction_pct']:+.1f}% | "
              f"cure {s['mean_cure_improvement_pct']:+.2f}%")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Wrote {OUT}")


if __name__ == "__main__":
    main()
