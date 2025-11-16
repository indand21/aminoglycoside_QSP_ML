# Patient Input Requirements for Dose Optimization

**Aminoglycoside QSP-ML Framework**
**Version:** 2.0
**Date:** 2025-11-16

---

## Overview

The dose optimization system has **two levels of complexity** depending on which tool you use:

1. **Simplified Demo Tool** - Requires only **3 essential inputs**
2. **Full ML-Based Optimizer** - Can use up to **18+ patient characteristics**

---

## Level 1: Simplified Demo Tool (Minimum Required)

**Tool:** `demo_simple_dose_optimization.py`

### Essential Inputs (3 Required)

| Parameter | Description | Units | Typical Range | Example |
|-----------|-------------|-------|---------------|---------|
| **weight** | Body weight | kg | 40-150 | 70 |
| **baseline_crcl** | Creatinine clearance | mL/min | 10-180 | 100 |
| **mic** | Pathogen MIC | mg/L | 0.5-32 | 2.0 |

### Example Usage:

```python
from demo_simple_dose_optimization import SimpleDoseOptimizer

optimizer = SimpleDoseOptimizer()

# Only 3 inputs needed!
optimal_dose, results = optimizer.optimize_dose(
    weight=70,           # kg
    crcl=100,           # mL/min
    mic=2.0             # mg/L
)
```

### How to Obtain These Values:

#### 1. **Weight** (kg)
- **Source:** Measured actual body weight
- **Special cases:**
  - Obese patients: Consider adjusted body weight
  - Fluid overload: Consider dry weight if available
  - Amputation: Adjust accordingly

#### 2. **Creatinine Clearance** (mL/min)
**Option A - Cockcroft-Gault Equation (Preferred):**
```
CrCL (mL/min) = [(140 - age) × weight × (0.85 if female)] / (72 × SCr)

Where:
  age = years
  weight = kg
  SCr = serum creatinine (mg/dL)
```

**Option B - Measured 24-hour Urine:**
```
CrCL = (urine creatinine × urine volume) / (serum creatinine × 1440)
```

**Option C - eGFR from Lab:**
- Use reported eGFR if CrCL not calculated
- Note: May underestimate in critically ill patients

**Important Considerations:**
- Use ACTUAL body weight for Cockcroft-Gault in aminoglycoside dosing
- Augmented renal clearance (ARC) common in ICU (CrCL >130)
- Round CrCL to nearest 5-10 mL/min for practical use

#### 3. **MIC** (mg/L)
**Option A - Measured MIC (Best):**
- From microbiology lab susceptibility testing
- Most accurate for dosing

**Option B - Epidemiological Breakpoints:**
| Organism | Typical Gentamicin MIC Range |
|----------|----------------------------|
| E. coli (susceptible) | 0.5-2 mg/L |
| Klebsiella (susceptible) | 1-4 mg/L |
| Pseudomonas aeruginosa | 2-8 mg/L |
| Acinetobacter | 4-16 mg/L (often resistant) |
| Enterococcus | Not recommended |

**Option C - Conservative Estimate:**
- If unknown: Use **2-4 mg/L** as reasonable estimate
- For empiric therapy: Use **4 mg/L** (worst-case susceptible)
- Re-optimize when actual MIC available

---

## Level 2: Full ML-Based Optimizer (Comprehensive)

**Tool:** `phase5_dose_optimization.py`

### Complete Feature Set (18 Baseline Features)

#### A. Demographics (4 features)

| Parameter | Description | Units/Format | Example |
|-----------|-------------|--------------|---------|
| **age** | Patient age | years | 65 |
| **sex** | Biological sex | M/F | M |
| **weight** | Body weight | kg | 70 |
| **height** | Height | cm | 175 |

**Derived:** BMI = weight / (height/100)²

---

#### B. Clinical Severity (2 features)

| Parameter | Description | Range | Interpretation |
|-----------|-------------|-------|----------------|
| **apache_ii** | APACHE II score | 0-71 | Illness severity |
| **sofa_score** | SOFA score | 0-24 | Organ dysfunction |

**APACHE II Components:**
- Age, vital signs, GCS, lab values (WBC, HCT, Na, K, Cr, etc.)
- Chronic health status
- Calculate using online calculator or ICU scoring system

**SOFA Score Components:**
- Respiratory (PaO2/FiO2)
- Coagulation (platelets)
- Hepatic (bilirubin)
- Cardiovascular (MAP, vasopressors)
- CNS (GCS)
- Renal (creatinine, urine output)

---

#### C. Renal Function (3 features)

| Parameter | Description | Units | Normal Range |
|-----------|-------------|-------|--------------|
| **baseline_crcl** | Creatinine clearance | mL/min | 90-130 |
| **baseline_scr** | Serum creatinine | mg/dL | 0.6-1.2 |
| **baseline_egfr** | Estimated GFR | mL/min/1.73m² | >90 |

**Relationships:**
- eGFR from MDRD or CKD-EPI equations
- CrCL from Cockcroft-Gault
- Use values at treatment initiation ("baseline")

---

#### D. Laboratory Values (2 features)

| Parameter | Description | Units | Normal Range | Significance |
|-----------|-------------|-------|--------------|--------------|
| **baseline_albumin** | Serum albumin | g/dL | 3.5-5.0 | Nutritional status, protein binding |
| **baseline_bilirubin** | Total bilirubin | mg/dL | 0.2-1.2 | Hepatic function |

**Clinical Relevance:**
- **Low albumin (<2.5):** Increased free drug fraction, consider dose adjustment
- **High bilirubin (>2.0):** Hepatic dysfunction, affects overall prognosis

---

#### E. Comorbidities (2 features)

| Parameter | Description | Format | Categories |
|-----------|-------------|--------|------------|
| **diabetes** | Diabetes mellitus | Boolean | True/False |
| **ckd_stage** | Chronic kidney disease stage | Categorical | 0, 1, 2, 3, 4, 5 |

**CKD Staging:**
- Stage 0: No CKD (eGFR >90 + no kidney damage)
- Stage 1: eGFR >90 with kidney damage
- Stage 2: eGFR 60-89
- Stage 3: eGFR 30-59
- Stage 4: eGFR 15-29
- Stage 5: eGFR <15 or dialysis

---

#### F. Critical Care Support (2 features)

| Parameter | Description | Format | Significance |
|-----------|-------------|--------|--------------|
| **mechanical_ventilation** | On ventilator | Boolean | True/False |
| **vasopressor_use** | Receiving vasopressors | Boolean | True/False |

**ML Model Findings:**
- **Mechanical ventilation:** #1 predictor of nephrotoxicity (8.4% importance)
- Indicates higher baseline risk for AKI

---

#### G. Infection Characteristics (2 features)

| Parameter | Description | Format | Categories |
|-----------|-------------|--------|------------|
| **sepsis_type** | Sepsis severity | Categorical | sepsis / severe sepsis / septic shock |
| **infection_site** | Primary infection source | Categorical | See below |

**Infection Site Categories:**
- bloodstream
- pneumonia
- urinary tract
- intra-abdominal
- skin/soft tissue
- bone/joint
- catheter-related
- other

---

#### H. Pathogen & Susceptibility (1 feature beyond simplified)

| Parameter | Description | Format | Use |
|-----------|-------------|--------|-----|
| **mic_gentamicin** | Gentamicin MIC | mg/L | Primary for gentamicin dosing |
| **mic_amikacin** | Amikacin MIC | mg/L | If using amikacin |

(Same as "mic" in simplified version)

---

## Input Summary by Tool

### Simplified Demo Tool

**Minimum Required (3):**
```python
patient = {
    'weight': 70,           # kg
    'baseline_crcl': 100,   # mL/min
    'mic': 2.0             # mg/L
}
```

**That's it!** The optimizer will:
- Use population-average PK parameters
- Predict outcomes based on simplified models
- Generate dose recommendations

---

### Full ML-Based Optimizer

**Complete Feature Set (18+):**
```python
patient = {
    # Demographics
    'age': 65,
    'sex': 'M',
    'weight': 70,
    'height': 175,
    'bmi': 22.9,  # Calculated

    # Severity
    'apache_ii': 18,
    'sofa_score': 8,

    # Renal
    'baseline_crcl': 80,
    'baseline_scr': 1.2,
    'baseline_egfr': 75,

    # Labs
    'baseline_albumin': 2.8,
    'baseline_bilirubin': 1.5,

    # Comorbidities
    'diabetes': True,
    'ckd_stage': 2,

    # Support
    'mechanical_ventilation': True,
    'vasopressor_use': True,

    # Infection
    'sepsis_type': 'severe sepsis',
    'infection_site': 'pneumonia',

    # Pathogen
    'mic_gentamicin': 2.0
}
```

**Advantages of Full Feature Set:**
- More accurate outcome predictions
- Better nephrotoxicity risk assessment
- Personalized based on illness severity
- Uses all ML model insights

---

## Data Collection Workflow

### At ICU Admission (Baseline)

**Step 1: Basic Measurements**
- [ ] Weight (kg)
- [ ] Height (cm)
- [ ] Age, sex

**Step 2: Calculate Scores**
- [ ] APACHE II (within 24h of ICU admission)
- [ ] SOFA score (at infection diagnosis)

**Step 3: Laboratory Tests**
- [ ] Serum creatinine
- [ ] Calculate/measure CrCL
- [ ] eGFR (from lab or calculated)
- [ ] Albumin
- [ ] Bilirubin

**Step 4: Clinical Status**
- [ ] Mechanical ventilation? (Y/N)
- [ ] Vasopressor use? (Y/N)
- [ ] Sepsis severity classification
- [ ] Infection site

**Step 5: Microbiology**
- [ ] Blood cultures obtained
- [ ] Source culture obtained
- [ ] Empiric MIC estimate OR
- [ ] Wait for susceptibility (24-48h)

---

## Missing Data Handling

### If You Don't Have All Features:

**Simplified Tool (Recommended):**
- Only requires weight, CrCL, MIC
- Most practical for routine use

**Full Tool with Missing Data:**
The optimizer will use default values for missing features:

| Missing Feature | Default Value | Impact |
|----------------|---------------|--------|
| age | 60 years | Minimal |
| sex | M (male) | Minimal |
| height | Calculated from weight | Minimal |
| APACHE II | 15 (moderate) | Moderate |
| SOFA | 6 (moderate) | Moderate |
| Albumin | 3.0 g/dL | Low |
| Bilirubin | 1.0 mg/dL | Minimal |
| Diabetes | False | Low |
| CKD stage | 0 (none) | Low |
| Mechanical ventilation | False | **High** - underestimates AKI risk |
| Vasopressor use | False | **High** - underestimates AKI risk |

**Recommendation:**
- Always provide: weight, CrCL, MIC
- High priority: mechanical ventilation, vasopressor use (major AKI predictors)
- Medium priority: APACHE II, SOFA (general risk)
- Lower priority: Other labs and demographics

---

## Special Populations

### 1. Obese Patients (BMI >30)

**Weight to Use:**
- **For CrCL calculation:** Use actual body weight
- **For PK predictions:** Optimizer uses actual weight
- Consider that Vd may be higher → potentially higher initial dose

**Example:**
```python
# 120 kg obese patient
patient = {
    'weight': 120,      # Use actual weight
    'baseline_crcl': 100,  # Calculated with actual weight
    'mic': 2.0
}
# Optimizer will account for higher Vd
```

### 2. Renal Impairment (CrCL <60)

**Important Considerations:**
- Higher drug accumulation
- Elevated trough levels expected
- May need dose reduction or extended interval
- Enhanced monitoring critical

**Example:**
```python
# Moderate renal impairment
patient = {
    'weight': 70,
    'baseline_crcl': 40,   # Moderate impairment
    'mic': 2.0
}
# Optimizer will predict higher trough, may recommend lower dose or q36-48h
```

### 3. Augmented Renal Clearance (CrCL >130)

**Common in:**
- Young trauma patients
- Septic patients with hyperdynamic circulation
- Burns

**Considerations:**
- Rapid drug elimination
- May need higher doses to achieve targets
- Optimizer will predict lower trough

**Example:**
```python
# ARC patient
patient = {
    'weight': 80,
    'baseline_crcl': 160,  # Augmented clearance
    'mic': 2.0
}
# Optimizer may recommend higher dose to maintain efficacy
```

### 4. Elderly Patients (Age >75)

**Considerations:**
- Often lower body weight
- Reduced renal function (even if SCr normal)
- Higher nephrotoxicity risk

**Example:**
```python
# Elderly patient
patient = {
    'age': 82,          # If using full optimizer
    'weight': 55,       # Lower weight
    'baseline_crcl': 50,   # Reduced for age
    'mic': 1.0
}
# Optimizer will likely recommend lower doses
```

---

## Quick Reference Card

### Minimum (Simplified Tool)

```
REQUIRED:
✓ Weight (kg)
✓ Creatinine Clearance (mL/min)
✓ Pathogen MIC (mg/L)

TOOL: demo_simple_dose_optimization.py
TIME: <5 minutes to collect data
```

### Full Feature Set (ML-Based)

```
REQUIRED:
✓ All above PLUS:
✓ Age, sex, height
✓ APACHE II score
✓ SOFA score
✓ Labs (albumin, bilirubin)
✓ Comorbidities (diabetes, CKD)
✓ Support (ventilation, pressors)
✓ Infection details

TOOL: phase5_dose_optimization.py
TIME: 15-30 minutes to collect complete data
```

---

## Example Data Collection Form

### Aminoglycoside Dose Optimization - Patient Input Form

**Patient ID:** _______________ **Date:** _______________

#### ESSENTIAL (Required for both tools)
- Weight: _______ kg
- Serum Creatinine: _______ mg/dL
- Creatinine Clearance: _______ mL/min
- Pathogen MIC (Gentamicin): _______ mg/L

#### ADDITIONAL (For full ML-based optimizer)

**Demographics:**
- Age: _______ years
- Sex: [ ] M [ ] F
- Height: _______ cm

**Severity Scores:**
- APACHE II: _______ (0-71)
- SOFA: _______ (0-24)

**Laboratory:**
- eGFR: _______ mL/min/1.73m²
- Albumin: _______ g/dL
- Bilirubin: _______ mg/dL

**Comorbidities:**
- Diabetes: [ ] Yes [ ] No
- CKD Stage: [ ] 0 [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**Critical Care Support:**
- Mechanical Ventilation: [ ] Yes [ ] No
- Vasopressor Use: [ ] Yes [ ] No

**Infection:**
- Sepsis Type: [ ] Sepsis [ ] Severe Sepsis [ ] Septic Shock
- Infection Site: _______________________

---

## FAQs

**Q: Which tool should I use?**
- **Simple cases:** Use simplified demo (3 inputs)
- **Complex/high-risk:** Use full optimizer (18 inputs)
- **Research/validation:** Use full optimizer for best predictions

**Q: What if I don't know the MIC yet?**
- Use conservative estimate (2-4 mg/L)
- Re-optimize when results available
- Higher MIC → higher doses needed

**Q: Can I use eGFR instead of CrCL?**
- Simplified tool: Yes, close enough for most patients
- Full optimizer: Prefers CrCL (calculated with Cockcroft-Gault)
- Use actual CrCL for obese patients

**Q: How often should I update inputs?**
- Initial dose: Use baseline values
- If CrCL changes >20%: Re-optimize
- If clinical status worsens: Re-optimize
- After TDM results: May need manual adjustment beyond optimizer

**Q: Are calculated scores like APACHE II required?**
- For simplified tool: No
- For full optimizer: Improves accuracy but has defaults
- Priority: Mechanical vent status > APACHE/SOFA > other features

---

## Summary

### The MINIMUM you need:
1. **Weight** (kg) - from bedside measurement
2. **Creatinine Clearance** (mL/min) - from Cockcroft-Gault or lab
3. **MIC** (mg/L) - from microbiology or estimate

### With just these 3 values, you can:
✅ Run dose optimization
✅ Get personalized recommendations
✅ Predict PK/PD target attainment
✅ Estimate outcome probabilities
✅ Generate comprehensive reports

### To get even better predictions, add:
- Critical care support status (mechanical ventilation, vasopressors)
- Illness severity scores (APACHE II, SOFA)
- Complete laboratory panel
- Detailed infection characteristics

---

**Bottom Line:** Start with the **3 essential inputs** using the simplified demo tool. Add more patient data if available for enhanced predictions using the full ML-based optimizer.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Framework:** Aminoglycoside QSP-ML v2.0
