# Required data structure
aminoglycoside_data <- list(
  
  # Patient demographics & baseline characteristics
  patient_data = data.frame(
    patient_id = character(),
    age = numeric(),           # years
    sex = factor(),            # M/F
    weight = numeric(),        # kg
    height = numeric(),        # cm
    bmi = numeric(),           # kg/m²
    
    # Clinical severity scores
    apache_ii = numeric(),
    sofa_score = numeric(),
    sepsis_type = factor(),    # sepsis/severe sepsis/septic shock
    
    # Baseline organ function
    baseline_scr = numeric(),  # serum creatinine (mg/dL)
    baseline_egfr = numeric(), # mL/min/1.73m²
    baseline_albumin = numeric(), # g/dL
    baseline_bilirubin = numeric(),
    
    # Infection characteristics
    infection_site = factor(), # bloodstream/pneumonia/UTI/IAI/other
    pathogen = character(),
    mic_amikacin = numeric(),  # μg/mL
    mic_gentamicin = numeric(),
    
    # Comorbidities
    diabetes = logical(),
    ckd_stage = factor(),
    mechanical_ventilation = logical(),
    vasopressor_use = logical(),
    
    # Study site
    hospital_id = factor(),
    icu_type = factor()       # medical/surgical/mixed
  ),
  
  # Time-varying covariates (measured at each PK sample time)
  time_varying = data.frame(
    patient_id = character(),
    time = numeric(),          # hours from first dose
    
    # Renal function (daily)
    scr = numeric(),
    crcl_cg = numeric(),      # Cockcroft-Gault mL/min
    crcl_measured = numeric(), # 24h urine collection when available
    urine_output = numeric(),  # mL/24h
    
    # Fluid balance
    cumulative_fluid_input = numeric(),  # L
    cumulative_fluid_output = numeric(),
    fluid_balance = numeric(),
    
    # Inflammatory markers
    wbc = numeric(),
    crp = numeric(),
    procalcitonin = numeric(),
    
    # Other labs
    albumin = numeric(),
    
    # Organ support
    rrt_status = logical(),
    rrt_type = factor(),      # none/IHD/CVVH/CVVHD/CVVHDF
    rrt_flow_rate = numeric(),
    
    # Hemodynamics
    map = numeric(),          # mean arterial pressure
    cardiac_output = numeric(), # if available
    norepinephrine_dose = numeric() # μg/kg/min
  ),
  
  # Dosing data
  dosing = data.frame(
    patient_id = character(),
    time = numeric(),          # hours from study start
    dose = numeric(),          # mg
    infusion_duration = numeric(), # hours (typically 0.5-1h)
    route = factor()          # IV
  ),
  
  # PK concentration data
  concentrations = data.frame(
    patient_id = character(),
    time = numeric(),          # hours from first dose
    sample_time_from_dose = numeric(), # hours from last dose
    concentration = numeric(), # mg/L
    assay_lloq = numeric(),    # lower limit of quantification
    bloq = logical(),          # below LOQ flag
    sample_type = factor(),    # peak/trough/random
    dose_number = integer()    # which dose this sample relates to
  ),
  
  # Clinical outcomes
  outcomes = data.frame(
    patient_id = character(),
    
    # Efficacy outcomes
    clinical_cure = logical(),
    microbiological_eradication = logical(),
    time_to_clinical_improvement = numeric(), # days
    
    # Safety outcomes
    nephrotoxicity = logical(), # by RIFLE/KDIGO criteria
    aki_stage = factor(),       # 0/1/2/3
    peak_scr = numeric(),
    ototoxicity = logical(),
    neurotoxicity = logical(),
    
    # Overall outcomes
    icu_los = numeric(),        # days
    hospital_los = numeric(),
    icu_mortality = logical(),
    day_28_mortality = logical(),
    
    # PK/PD target attainment
    achieved_cmax_mic = numeric(), # actual Cmax/MIC ratio
    achieved_auc_mic = numeric()
  )
)
