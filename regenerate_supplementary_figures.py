#!/usr/bin/env python3
"""Regenerate the data-derived panels of Supplementary Figures S4, S6 and S7.

These three figures previously carried hard-coded illustrative values that did not
trace to any released artefact. Each panel below is now computed from the files in
``results/`` or, for S7(B), by running the released dose optimizer:

  S4(A)  precision-recall curves and average precision, from test_predictions.json
  S4(B)  per-framework held-out ROC-AUC, from algorithm_comparison.json
  S6(A)  subgroup forest plot, from the Table 7C subgroup sizes and event rates
  S7(B)  composite objective across the 50-point dose grid, via optimize_dose_grid

The remaining panels (S4 C-D, S6 B-C, S7 A/C/D) are reproduced unchanged and remain
illustrative; they are labelled as such in the figure captions.

Usage:  python regenerate_supplementary_figures.py
Writes PNGs into Manuscript_AAPS_Journal/Supplementary Materials/.
"""
import contextlib
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

# Windows consoles often default to cp1252; figure labels contain non-ASCII glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.join(ROOT, "Manuscript_AAPS_Journal", "Supplementary Materials")
RES = os.path.join(ROOT, "results", "phase4_ml_enhanced")

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {"blue": "#2E86AB", "green": "#28A745", "orange": "#F5A623",
          "purple": "#7B68EE", "red": "#DC3545", "gray": "#6C757D"}

# Table 7C of the manuscript: (label, N, high-risk event rate %, low-risk event rate %).
# The relative risks plotted in S6(A) are derived from these, so figure and table agree
# by construction.
TABLE_7C = [
    ("Age <65", 1012, 67.6, 5.5),
    ("Age ≥65", 488, 77.3, 12.3),
    ("CrCL >80", 782, 66.4, 5.0),
    ("CrCL ≤80", 718, 76.3, 10.7),
    ("DM", 528, 74.7, 10.2),
    ("No DM", 972, 69.4, 6.0),
    ("APACHE II <20", 624, 64.2, 4.5),
    ("APACHE II ≥20", 876, 76.6, 9.9),
]


def rr_with_ci(n_total, high_pct, low_pct, z=1.96):
    """Relative risk and 95% CI from the table's own values (tertiles => n/3 per group)."""
    n_grp = n_total / 3.0
    p1, p0 = high_pct / 100.0, low_pct / 100.0
    rr = p1 / p0
    se = np.sqrt((1 - p1) / (n_grp * p1) + (1 - p0) / (n_grp * p0))
    return rr, rr * np.exp(-z * se), rr * np.exp(z * se)


def figure_s4(path):
    """Extended machine-learning diagnostics."""
    preds = json.load(open(os.path.join(RES, "test_predictions.json")))
    algo = json.load(open(os.path.join(RES, "algorithm_comparison.json")))["frameworks"]
    fig = plt.figure(figsize=(13, 7))

    # (A) real precision-recall curves
    ax = fig.add_subplot(2, 2, 1)
    prevalence = None
    for key, name, color in [("nephrotoxicity_predose", "Pre-dose", COLORS["gray"]),
                             ("nephrotoxicity_postdose", "Post-dose", COLORS["blue"])]:
        y = np.array(preds[key]["y_test"])
        p = np.array(preds[key]["y_pred_proba"])
        prevalence = y.mean()
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.2f})")
        print(f"    {name:10s} AP = {ap:.3f}")
    ax.axhline(prevalence, color="gray", linestyle="--", lw=1,
               label=f"Prevalence ({prevalence:.0%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("A. Precision-Recall Curves", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (B) real per-framework held-out ROC-AUC
    ax2 = fig.add_subplot(2, 2, 2)
    order = ["xgboost", "lightgbm", "catboost", "stacking"]
    names = ["XGBoost", "LightGBM", "CatBoost", "Stacking"]
    aucs = [algo[k]["test_auc"] for k in order]
    ax2.bar(names, aucs, width=0.6, alpha=0.85,
            color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]])
    for i, v in enumerate(aucs):
        ax2.text(i, v + 0.0012, f"{v:.3f}", ha="center", fontsize=8)
    lo, hi = min(aucs), max(aucs)
    ax2.axhspan(lo, hi, color=COLORS["gray"], alpha=0.15)
    ax2.set_ylabel("Held-out ROC-AUC")
    ax2.set_title("B. Framework Discrimination", fontweight="bold")
    ax2.set_ylim(0.88, 0.93)
    ax2.text(0.02, 0.93, f"range {hi - lo:.3f}", fontsize=7, transform=ax2.transAxes)
    print(f"    framework AUC range {lo:.3f}-{hi:.3f}")

    # (C) learning curves - illustrative
    ax3 = fig.add_subplot(2, 2, 3)
    ts = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    tr = np.array([0.945, 0.93, 0.92, 0.918, 0.915])
    va = np.array([0.84, 0.87, 0.89, 0.905, 0.915])
    ax3.plot(ts * 100, tr, "o-", color=COLORS["blue"], lw=2, label="Training")
    ax3.plot(ts * 100, va, "s-", color=COLORS["orange"], lw=2, label="Validation")
    ax3.fill_between(ts * 100, tr - 0.015, tr + 0.015, alpha=0.2, color=COLORS["blue"])
    ax3.fill_between(ts * 100, va - 0.02, va + 0.02, alpha=0.2, color=COLORS["orange"])
    ax3.set_xlabel("Training Set Size (%)")
    ax3.set_ylabel("ROC-AUC")
    ax3.set_title("C. Learning Curves", fontweight="bold")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.set_ylim(0.75, 1.0)

    # (D) Optuna trajectory - illustrative
    ax4 = fig.add_subplot(2, 2, 4)
    np.random.seed(42)
    best, cur = np.zeros(100), 0.75
    for i in range(100):
        cur = max(cur, 0.75 + 0.155 * (1 - np.exp(-i / 18)) + np.random.normal(0, 0.008))
        best[i] = cur
    ax4.plot(np.arange(1, 101), best, color=COLORS["blue"], lw=2)
    ax4.axhline(0.915, color=COLORS["green"], linestyle="--", lw=1,
                label="Final best (0.915)")
    ax4.set_xlabel("Trial Number")
    ax4.set_ylabel("Best ROC-AUC")
    ax4.set_title("D. Optuna Optimization", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0.70, 0.95)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.08,
                        wspace=0.30, hspace=0.35)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def figure_s6(path):
    """Extended risk-stratification diagnostics."""
    fig = plt.figure(figsize=(13, 4.5))

    # (A) forest plot built from Table 7C
    ax = fig.add_subplot(1, 3, 1)
    labels, rrs, los, his = [], [], [], []
    for lab, n, hi_pct, lo_pct in TABLE_7C:
        rr, l, u = rr_with_ci(n, hi_pct, lo_pct)
        labels.append(lab); rrs.append(rr); los.append(l); his.append(u)
    y = np.arange(len(labels))[::-1]
    ax.errorbar(rrs, y,
                xerr=[np.array(rrs) - np.array(los), np.array(his) - np.array(rrs)],
                fmt="o", color=COLORS["blue"], capsize=5, markersize=7)
    ax.axvline(1, color="gray", linestyle="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Relative Risk (High vs Low Risk)")
    ax.set_title("A. Subgroup Discrimination", fontweight="bold")
    ax.set_xlim(0, max(his) * 1.08)
    for lab, rr in zip(labels, rrs):
        print(f"    {lab:18s} RR {rr:5.1f}")

    # (B) confusion matrix at the illustrative operating point
    ax2 = fig.add_subplot(1, 3, 2)
    tp = int(494 * 0.81); fn = 494 - tp
    tn = int(1006 * 0.84); fp = 1006 - tn
    cm = np.array([[tn, fp], [fn, tp]])
    ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax2.set_yticklabels(["True Neg", "True Pos"])
    ax2.set_title("B. Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2.text(0.5, -0.2,
             f"Sens={tp/(tp+fn):.2f}, Spec={tn/(tn+fp):.2f}, "
             f"PPV={tp/(tp+fp):.2f}, NPV={tn/(tn+fn):.2f}",
             transform=ax2.transAxes, ha="center", fontsize=8)

    # (C) risk-benefit trajectory - illustrative
    ax3 = fig.add_subplot(1, 3, 3)
    doses = np.array([3, 4, 5, 6, 7])
    tw = ax3.twinx()
    l1 = ax3.plot(doses, np.array([15, 28, 45, 62, 78]), "o-", color=COLORS["red"],
                  lw=2, label="AKI Risk")
    l2 = tw.plot(doses, np.array([55, 68, 78, 85, 89]), "s-", color=COLORS["green"],
                 lw=2, label="Efficacy")
    ax3.axvline(5, color="gray", linestyle="--", lw=1.5)
    ax3.set_xlabel("Dose (mg/kg)")
    ax3.set_ylabel("AKI Risk (%)", color=COLORS["red"])
    tw.set_ylabel("Efficacy (%)", color=COLORS["green"])
    ax3.set_title("C. Risk-Benefit Trajectory", fontweight="bold")
    ax3.set_ylim(0, 100); tw.set_ylim(0, 100)
    ax3.legend(l1 + l2, [x.get_label() for x in l1 + l2], fontsize=7, loc="center right")

    fig.subplots_adjust(left=0.09, right=0.95, top=0.90, bottom=0.16, wspace=0.45)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def figure_s7(path):
    """Components of the multi-objective dose optimizer."""
    # Panel B runs the released optimizer, which resolves its inputs relative to the
    # project root.
    cwd = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            from phase5_dose_optimization import DoseOptimizer
            opt = DoseOptimizer()
        recs = pd.read_csv(os.path.join(ROOT, "results", "phase5_optimization",
                                        "dose_recommendations.csv"))
        mean_opt = recs["optimal_dose"].mean()
        pid = recs.iloc[(recs["optimal_dose"] - mean_opt).abs().argmin()]["patient_id"]
        patient = opt.data[opt.data["patient_id"] == pid].iloc[0]
        doses = np.linspace(opt.dose_range[0], opt.dose_range[1], 50)
        with contextlib.redirect_stdout(buf):
            scores = np.array([opt.objective_function(patient, d) for d in doses])
    finally:
        os.chdir(cwd)
    best = int(np.argmax(scores))
    print(f"    representative patient {pid}: optimum {doses[best]:.0f} mg "
          f"(cohort mean {mean_opt:.0f} mg)")

    fig = plt.figure(figsize=(13, 7))

    # (A) objective weights
    ax = fig.add_subplot(2, 2, 1)
    weights = [0.40, 0.30, 0.20, 0.10]
    ax.bar(["Cure\nProbability", "Safety\n(1-P_AKI)", "Cmax/MIC\nTarget", "Trough\nSafety"],
           weights, edgecolor="black", alpha=0.8,
           color=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]])
    ax.set_ylabel("Weight")
    ax.set_title("A. Optimization Objectives", fontweight="bold")
    ax.set_ylim(0, 0.5)
    for i, w in enumerate(weights):
        ax.text(i, w + 0.02, f"{w:.0%}", ha="center", fontsize=9)

    # (B) real grid search
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(doses, scores, "-", color=COLORS["blue"], lw=2, zorder=2,
             label="Composite objective")
    ax2.scatter(doses, scores, s=18, color=COLORS["red"], alpha=0.75, zorder=3,
                label="Evaluated doses (n = 50)")
    ax2.scatter([doses[best]], [scores[best]], color=COLORS["green"], s=170, marker="*",
                zorder=5, label=f"Selected ({doses[best]:.0f} mg)")
    ax2.set_xlabel("Dose (mg)")
    ax2.set_ylabel("Composite objective score")
    ax2.set_title("B. Grid-Search Dose Selection", fontweight="bold")
    ax2.legend(fontsize=6, loc="lower right")

    # (C) Pareto frontier - illustrative
    ax3 = fig.add_subplot(2, 2, 3)
    np.random.seed(42)
    eff = np.random.uniform(0.5, 0.95, 200)
    saf = np.clip(0.95 - 0.25 * eff + np.random.normal(0, 0.04, 200), 0, 1)
    ax3.scatter(eff, saf, alpha=0.4, color=COLORS["gray"], s=15, label="All solutions")
    ax3.plot([0.52, 0.60, 0.68, 0.75, 0.82, 0.88, 0.92],
             [0.93, 0.90, 0.86, 0.82, 0.77, 0.72, 0.68], "o-",
             color=COLORS["blue"], lw=2, markersize=7, label="Pareto frontier")
    ax3.scatter([0.76], [0.83], color=COLORS["green"], s=150, marker="*", zorder=5,
                label="Selected")
    ax3.set_xlabel("P(Clinical Cure)")
    ax3.set_ylabel("P(No AKI)")
    ax3.set_title("C. Pareto Frontier", fontweight="bold")
    ax3.legend(fontsize=6, loc="lower left")
    ax3.set_xlim(0.4, 1.0)
    ax3.set_ylim(0.5, 1.0)

    # (D) decision-support mock-up - illustrative
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    rule = "  " + "─" * 31 + "\n"
    cds = (
        "  Patient: 62y, 65 kg, CrCL=48 mL/min\n"
        "  Drug: Amikacin (trough threshold <2.5 mg/L)\n" + rule +
        "  PREDICTED OUTCOMES (Current):\n"
        "    Current dose: 900 mg q24h\n"
        "    P(Cure):        64%\n"
        "    P(AKI):         38%\n"
        "    Cmax/MIC ≥8:    46%\n" + rule +
        "  RECOMMENDED: 700 mg q24h (−22%)\n" + rule +
        "  PREDICTED OUTCOMES (Optimized):\n"
        "    P(Cure):        69%  (+5%)\n"
        "    P(AKI):         21%  (−17%)\n"
        "    Cmax/MIC ≥8:    72%  (+26%)\n" + rule +
        "  KEY RISK FACTORS:\n"
        "    • Reduced renal function (CrCL 48)\n"
        "    • Age >60 years\n"
        "    • High SHAP nephrotoxicity score\n" + rule +
        "  Illustrative display only\n"
    )
    ax4.text(0.5, 0.5, cds, transform=ax4.transAxes, fontsize=7.2, va="center",
             ha="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor=COLORS["blue"], linewidth=1.5))
    ax4.set_title("D. Clinical Decision Support", fontweight="bold")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.08,
                        wspace=0.30, hspace=0.35)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved {os.path.basename(path)}")


def main():
    print("Supplementary Figure S4:")
    figure_s4(os.path.join(SUPP, "FigureS4_ML_Performance_Extras.png"))
    print("Supplementary Figure S6:")
    figure_s6(os.path.join(SUPP, "FigureS6_Risk_Stratification_Extras.png"))
    print("Supplementary Figure S7:")
    figure_s7(os.path.join(SUPP, "FigureS7_Dose_Optimization_Extras.png"))


if __name__ == "__main__":
    main()
