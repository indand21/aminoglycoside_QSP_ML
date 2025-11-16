# Supplementary Materials - Overview

This directory contains comprehensive supplementary materials for the manuscript:

**"Integration of Quantitative Systems Pharmacology and Machine Learning for Personalized Aminoglycoside Dosing in Critically Ill Patients: A Complete Framework for Indian Intensive Care Units"**

---

## File Organization

```
manuscript/
├── manuscript.md                          # Main manuscript text (4,500 words)
├── references.md                          # 25 verified references
├── figure_legends.md                      # Legends for 6 main figures
├── figures/                              # 6 publication-quality figures (PNG, 300 DPI)
│   ├── Figure1_PopPK_Posteriors.png
│   ├── Figure2_PTA_Heatmap.png
│   ├── Figure3_PKPD_Distributions.png
│   ├── Figure4_ROC_Curves.png
│   ├── Figure5_Calibration.png
│   └── Figure6_Dose_Optimization.png
├── tables/                               # 3 main manuscript tables
│   ├── Table1_Patient_Characteristics.md
│   ├── Table2_PopPK_Parameters.md
│   └── Table3_ML_Performance.md
└── supplementary/                        # Supplementary materials (THIS DIRECTORY)
    ├── README.md                         # This file
    ├── SupplementaryTables.md           # 8 comprehensive tables
    ├── SupplementaryFigures.md          # 15 detailed figure descriptions
    ├── MathematicalAppendix.md          # Complete mathematical formulations
    └── ImplementationGuidelines.md      # Practical implementation guide
```

---

## Supplementary Materials Summary

### 1. Supplementary Tables (SupplementaryTables.md)

**8 comprehensive tables** providing detailed model performance, validation, and patient characteristics:

| Table | Description | Key Content |
|-------|-------------|-------------|
| S1 | Complete ML Model Performance | All models (ensemble, base learners, neural networks, baseline) with all metrics |
| S2 | Cross-Validation Performance | 5-fold CV results showing stability and robustness |
| S3 | Feature Importance - Nephrotoxicity | All 25 features ranked with 95% confidence intervals |
| S4 | Feature Importance - Clinical Cure | All 36 features ranked with cumulative importance |
| S5 | Hyperparameter Optimization | Top 10 configurations for both nephrotoxicity and clinical cure models |
| S6 | Patient Stratification | Characteristics stratified by outcomes with statistical tests |
| S7 | Surrogate Model Performance | Cmax and AUC24 prediction accuracy metrics |
| S8 | Dose Optimization Metrics | Computational performance and target attainment improvements |

**Total Content:** ~4,000 words, 8 detailed tables

---

### 2. Supplementary Figures (SupplementaryFigures.md)

**15 multi-panel figures** providing comprehensive visualizations:

| Figure | Description | Panels | Key Insights |
|--------|-------------|--------|--------------|
| S1 | Individual ROC Curves | 6 | Performance comparison across all ML models |
| S2 | Precision-Recall Curves | 4 | Model performance for imbalanced outcomes |
| S3 | Feature Importance | 6 | Visual importance rankings with confidence intervals |
| S4 | Cross-Validation Distributions | 4 | Performance stability across CV folds |
| S5 | MCMC Diagnostics | 6 | Convergence assessment for Bayesian PopPK model |
| S6 | Dose Optimization Cases | 6 | Three detailed clinical examples |
| S7 | Surrogate Model Diagnostics | 6 | Cmax and AUC24 prediction accuracy |
| S8 | PTA by Renal Function | 4 | Stratified target attainment heatmaps |
| S9 | Exposure-Response Curves | 6 | LOWESS-smoothed outcome relationships |
| S10 | Learning Curves | 4 | Sample size adequacy analysis |
| S11 | Calibration Plots | 6 | Model calibration assessment with confidence bands |
| S12 | Neural Network Analysis | 6 | Architecture and training dynamics |
| S13 | Ensemble Stacking | 6 | Meta-learner analysis and performance gains |
| S14 | SMOTE Implementation | 6 | Class balance handling and impact |
| S15 | Subgroup Analysis | 6 | Performance across patient subgroups |

**Total Content:** ~8,500 words describing 15 figures with 81 total panels

---

### 3. Mathematical Appendix (MathematicalAppendix.md)

**Complete mathematical formulations** for all modeling approaches:

| Section | Topic | Content |
|---------|-------|---------|
| 1 | Structural PK Model | Two-compartment ODE system, analytical solutions |
| 2 | Population Model | Between-subject variability, covariate relationships |
| 3 | Bayesian Statistics | Likelihood function, prior distributions, MCMC sampling |
| 4 | PK/PD Calculations | Index formulas, target attainment probability, CFR |
| 5 | Machine Learning | XGBoost objective function, SMOTE algorithm, ensemble stacking |
| 6 | Dose Optimization | Multi-objective formulation, Bayesian optimization, acquisition function |
| 7 | Evaluation Metrics | ROC-AUC, Brier score, bootstrap confidence intervals |
| 8 | Implementation | PyMC and XGBoost pseudocode |

**Key Equations:**
- Two-compartment PK model differential equations
- Bayesian posterior distribution formulation
- XGBoost loss function with L2 regularization
- Multi-objective dose optimization function
- Gaussian process Bayesian optimization

**Total Content:** ~6,500 words, 50+ mathematical equations

---

### 4. Implementation Guidelines (ImplementationGuidelines.md)

**Comprehensive guide** for researchers and clinicians to use the framework:

| Section | Topic | Content |
|---------|-------|---------|
| 1 | System Requirements | Hardware, software, performance benchmarks |
| 2 | Installation | Conda, pip, Docker with verification steps |
| 3 | Data Preparation | Input format (CSV with 27 columns), quality checks |
| 4 | Running Framework | Complete pipeline, individual phases, parallel processing |
| 5 | Interpreting Results | How to read PopPK, PK/PD, ML, optimization outputs |
| 6 | Clinical Deployment | EHR integration, validation workflow, safety protocols |
| 7 | Troubleshooting | Common issues and solutions for installation and runtime |
| 8 | Validation Guidelines | Internal validation checklist, external validation protocol |

**Practical Tools:**
- Step-by-step installation commands
- Example data format with all required columns
- Command-line usage examples for all phases
- API endpoint code for EHR integration
- Safety guardrail implementation
- Troubleshooting flowcharts
- Performance optimization tips

**Appendices:**
- Complete parameter reference (30+ parameters)
- Error codes reference (E001-E008)
- Additional resources and community links

**Total Content:** ~7,000 words, practical code examples

---

## Total Supplementary Materials

**Summary Statistics:**
- **Total Word Count:** ~26,000 words
- **Tables:** 8 comprehensive tables
- **Figure Descriptions:** 15 multi-panel figures (81 panels total)
- **Mathematical Equations:** 50+ with full derivations
- **Code Examples:** 20+ practical implementation snippets
- **File Size:** ~250 KB (text files)

---

## Usage for Manuscript Submission

### For Journal Submission:

1. **Main Manuscript Package:**
   - `manuscript/manuscript.md` → Convert to journal template format
   - `manuscript/tables/` → Include as main tables
   - `manuscript/figures/` → Submit as separate high-resolution files
   - `manuscript/references.md` → Format per journal requirements

2. **Supplementary Materials Package:**
   - `supplementary/SupplementaryTables.md` → Submit as separate PDF
   - `supplementary/SupplementaryFigures.md` → Submit as figure descriptions (figures to be generated from results)
   - `supplementary/MathematicalAppendix.md` → Submit as separate PDF
   - `supplementary/ImplementationGuidelines.md` → Submit as separate PDF or online-only supplement

### Conversion to Journal Format:

```bash
# Convert Markdown to Word/PDF using Pandoc
pandoc manuscript/manuscript.md -o manuscript.docx --reference-doc=journal_template.docx
pandoc supplementary/SupplementaryTables.md -o SupplementaryTables.pdf
pandoc supplementary/MathematicalAppendix.md -o MathematicalAppendix.pdf --pdf-engine=xelatex
pandoc supplementary/ImplementationGuidelines.md -o ImplementationGuidelines.pdf
```

### For Online Repository:

All supplementary materials are already in web-friendly Markdown format suitable for:
- GitHub repository documentation
- Research data repositories (Zenodo, Figshare)
- Journal online supplements
- Project website

---

## Target Journals

These supplementary materials are formatted for submission to:

**Primary Targets:**
- Clinical Pharmacology & Therapeutics
- Antimicrobial Agents and Chemotherapy
- British Journal of Clinical Pharmacology
- CPT: Pharmacometrics & Systems Pharmacology

**Secondary Targets:**
- Bioinformatics
- Journal of Antimicrobial Chemotherapy
- Pharmacotherapy
- International Journal of Antimicrobial Agents

**Computational/Methods:**
- Briefings in Bioinformatics
- Journal of Pharmacokinetics and Pharmacodynamics
- Pharmaceutical Research

---

## Citation

If using these supplementary materials, please cite the main manuscript:

```
[Authors]. Integration of Quantitative Systems Pharmacology and Machine Learning
for Personalized Aminoglycoside Dosing in Critically Ill Patients. [Journal]. [Year];[Vol]:[Pages].
```

---

## License

[To be specified - recommend Creative Commons CC-BY 4.0 for open access]

---

## Contact

For questions about supplementary materials:
- Technical questions: [GitHub Issues]
- Content clarifications: [Corresponding author email]
- Collaboration inquiries: [Contact information]

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-16 | Initial creation of all supplementary materials |

---

**Last Updated:** 2025-01-16
**Document Prepared By:** Claude AI Assistant
**Review Status:** Ready for author review and journal submission
