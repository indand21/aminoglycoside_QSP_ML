# FIGURE LEGENDS

## Figure 1. Bayesian Posterior Distributions for Population Pharmacokinetic Parameters

Posterior probability distributions from Markov chain Monte Carlo sampling for four population pharmacokinetic parameters: clearance (CL), central volume of distribution (Vc), intercompartmental clearance (Q), and peripheral volume of distribution (Vp). Each panel shows the posterior density (blue line) with the posterior mean (vertical line) and 94% highest density interval (horizontal black bar) indicated. The distributions demonstrate adequate convergence and physiologically plausible parameter estimates consistent with published aminoglycoside pharmacokinetics. CL mean 5.8 L/h (HDI 3.0-9.0), Vc mean 17 L (HDI 7.6-26), Q mean 13 L/h (HDI 3.8-26), Vp mean 11 L (HDI 2.6-22).

---

## Figure 2. Probability of Target Attainment Heatmap by Dose and Minimum Inhibitory Concentration

Heat maps showing probability of pharmacokinetic/pharmacodynamic target attainment across dose ranges (200-1900 mg, x-axis) and minimum inhibitory concentrations (0.25-20 mg/L, y-axis) for 1,500 simulated patients. Left panel: Probability of achieving Cmax/MIC ≥8 efficacy target. Right panel: Probability of achieving combined targets (Cmax/MIC ≥8 AND AUC/MIC ≥80 AND trough <2 mg/L). Color scale indicates probability of target attainment from 0 (dark red, no patients achieving target) to 1.0 (dark green, all patients achieving target). The right panel demonstrates that only 44.6% of patients overall achieve combined efficacy and safety targets, with substantial variation by dose and minimum inhibitory concentration. For highly susceptible organisms (MIC ≤1 mg/L), target attainment exceeds 90% even with moderate doses. For organisms with elevated MIC (≥8 mg/L), target attainment remains poor even with maximum doses.

---

## Figure 3. Distribution of Pharmacokinetic and Pharmacodynamic Indices

Histograms showing population distributions of key pharmacokinetic and pharmacodynamic indices in 1,500 patients. Top left: Cmax/MIC ratio distribution with median 18.4 (red dashed line indicates efficacy target of 8). Top right: AUC/MIC ratio distribution with median 178 (red dashed line indicates efficacy target of 80). Bottom left: Peak concentration (Cmax) distribution with mean 50.2 mg/L. Bottom right: Trough concentration (Cmin) distribution with median 0.8 mg/L (red dashed line indicates safety threshold of 2 mg/L). The distributions demonstrate substantial between-patient variability in pharmacokinetic exposure and pharmacodynamic indices. Most patients maintain trough concentrations below the safety threshold, but approximately 50% fail to achieve Cmax/MIC and AUC/MIC efficacy targets, highlighting the need for dose optimization.

---

## Figure 4. Receiver Operating Characteristic Curves for Machine Learning Models

Receiver operating characteristic (ROC) curves for enhanced machine learning models. Left panel: Nephrotoxicity prediction model showing area under the curve (AUC) of 0.739 on test set (n=300 patients). Right panel: Clinical cure prediction model showing AUC of 0.742 on test set. Both curves are substantially above the diagonal dashed line representing random classification (AUC 0.5), demonstrating clinically useful discriminative ability. The nephrotoxicity model achieved sensitivity of 60% and specificity of 74% at the optimal classification threshold. The clinical cure model achieved sensitivity of 98% for detecting cure and specificity of 29%. Both models crossed the threshold of AUC ≥0.70 considered necessary for clinical decision support applications.

---

## Figure 5. Calibration Curves for Machine Learning Models

Calibration plots assessing agreement between predicted probabilities and observed frequencies for machine learning models. Left panel: Nephrotoxicity calibration showing reasonable agreement between predictions and observations, particularly in the 0.1-0.6 probability range where most predictions fall. Right panel: Clinical cure calibration showing some deviation from perfect calibration (black dashed diagonal line) at probability extremes but acceptable overall agreement. Points are sized proportional to the number of patients in each probability bin. Good calibration is essential for using model predictions to inform clinical decisions, as it ensures that predicted probabilities correspond to actual likelihoods of outcomes.

---

## Figure 6. Personalized Dose Optimization for Representative Patient

Comprehensive six-panel dose-response visualization for personalized dose optimization in a 70 kg patient with creatinine clearance 100 mL/min and pathogen minimum inhibitory concentration 2.0 mg/L. Top row shows pharmacodynamic indices: Cmax/MIC ratio (left), AUC/MIC ratio (middle), and trough concentration (right) across doses from 200-1600 mg. Red dashed lines indicate clinical targets (Cmax/MIC ≥8, AUC/MIC ≥80, trough <2 mg/L). Green shading indicates target zones. Bottom row shows predicted clinical outcomes: probability of cure (left), probability of nephrotoxicity (middle), and overall optimization score (right, 0-100 scale). The red vertical dashed line and star in the bottom right panel mark the optimal dose of 810 mg, which achieves both efficacy targets (Cmax/MIC 23.2, AUC/MIC 81) while maintaining a safe trough (0.05 mg/L), corresponding to 95% predicted cure probability and 12% nephrotoxicity risk. This comprehensive visualization enables clinicians to understand trade-offs and make informed decisions about dose selection.
