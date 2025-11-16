# Table 2: Population Pharmacokinetic Parameter Estimates

| **Parameter** | **Estimate** | **94% HDI** | **Interpretation** |
|--------------|--------------|-------------|-------------------|
| **Structural Model Parameters** | | | |
| Clearance (CL), L/h | 5.8 | 3.4 – 9.0 | Renal elimination rate |
| Central volume (Vc), L | 17 | 7.6 – 26 | Volume of distribution (central compartment) |
| Intercompartmental clearance (Q), L/h | 13 | 3.8 – 26 | Distribution rate between compartments |
| Peripheral volume (Vp), L | 11 | 2.6 – 22 | Volume of distribution (peripheral compartment) |
| **Derived Parameters** | | | |
| Volume of distribution at steady state (Vss), L | 28 | 18 – 42 | CL + Vc + Vp |
| Elimination half-life (t½β), hours | 3.3 | 2.1 – 5.8 | Terminal elimination half-life |
| Distribution half-life (t½α), hours | 0.9 | 0.4 – 1.6 | Distribution phase half-life |
| **Interindividual Variability (ω²)** | | | |
| ω²CL, CV% | 45% | 38% – 52% | Between-patient variability in clearance |
| ω²Vc, CV% | 38% | 31% – 46% | Between-patient variability in central volume |
| ω²Q, CV% | 52% | 43% – 61% | Between-patient variability in Q |
| ω²Vp, CV% | 48% | 39% – 57% | Between-patient variability in peripheral volume |
| **Residual Variability** | | | |
| Proportional error, CV% | 15% | 12% – 18% | Assay and model misspecification error |
| **Covariate Effects** | | | |
| Weight on CL (power) | 0.75 | 0.68 – 0.82 | Allometric scaling exponent |
| Weight on Vc (power) | 1.00 | 0.91 – 1.09 | Linear relationship |
| CrCL on CL (slope) | 0.008 | 0.006 – 0.010 | L/h per mL/min CrCL |
| Age on CL (slope) | -0.012 | -0.018 – -0.006 | L/h per year |

**Abbreviations:** HDI, highest density interval (Bayesian credible interval); CV%, coefficient of variation expressed as percentage; CrCL, creatinine clearance.

**Model Details:** Two-compartment population pharmacokinetic model fit using Bayesian inference with Markov Chain Monte Carlo sampling (4 chains × 2,000 iterations, 1,000 warmup). Prior distributions were informed by published aminoglycoside pharmacokinetic literature. Model convergence was assessed using R-hat statistic (all <1.01, indicating convergence).

**Clinical Interpretation:** The clearance estimate of 5.8 L/h is consistent with published values for aminoglycosides in critically ill patients with preserved renal function. The substantial interindividual variability (38-52% CV) justifies the need for therapeutic drug monitoring and individualized dosing strategies. Significant covariate relationships identified include weight-based scaling for clearance and volume, creatinine clearance effect on elimination, and age-related decline in clearance.
