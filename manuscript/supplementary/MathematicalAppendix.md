# Mathematical Appendix: Population Pharmacokinetic Model Equations

## 1. Structural Pharmacokinetic Model

### 1.1 Two-Compartment Model with IV Infusion

The aminoglycoside concentration-time profile is described by a two-compartment model with first-order elimination.

**Differential Equations:**

```
dA_c/dt = R(t) - (CL/V_c)·A_c - (Q/V_c)·A_c + (Q/V_p)·A_p

dA_p/dt = (Q/V_c)·A_c - (Q/V_p)·A_p
```

Where:
- `A_c` = amount of drug in central compartment (mg)
- `A_p` = amount of drug in peripheral compartment (mg)
- `R(t)` = infusion rate at time t (mg/h)
- `CL` = clearance (L/h)
- `V_c` = central volume of distribution (L)
- `V_p` = peripheral volume of distribution (L)
- `Q` = intercompartmental clearance (L/h)

**Plasma Concentration:**

```
C(t) = A_c(t) / V_c
```

### 1.2 Infusion Input Function

For short IV infusions (duration T_inf):

```
R(t) = { Dose/T_inf,  if 0 ≤ t ≤ T_inf
       { 0,           if t > T_inf
```

Where:
- `Dose` = administered dose (mg)
- `T_inf` = infusion duration (h), typically 0.5h

### 1.3 Analytical Solutions

**During Infusion (0 ≤ t ≤ T_inf):**

```
C(t) = (R/V_c) · [(A/α)(1 - e^(-α·t)) + (B/β)(1 - e^(-β·t))]
```

**Post-Infusion (t > T_inf):**

```
C(t) = C_end · [A·e^(-α·(t-T_inf)) + B·e^(-β·(t-T_inf))]
```

Where:
- `α` = fast (distribution) phase rate constant
- `β` = slow (elimination) phase rate constant
- `A`, `B` = coefficients determined by system parameters
- `C_end` = concentration at end of infusion

**Hybrid Rate Constants:**

```
α = 0.5·[(CL/V_c + Q/V_c + Q/V_p) + sqrt((CL/V_c + Q/V_c + Q/V_p)^2 - 4·(CL·Q)/(V_c·V_p))]

β = 0.5·[(CL/V_c + Q/V_c + Q/V_p) - sqrt((CL/V_c + Q/V_c + Q/V_p)^2 - 4·(CL·Q)/(V_c·V_p))]
```

**Coefficients:**

```
A = (α - Q/V_p) / (α - β)

B = (Q/V_p - β) / (α - β)
```

### 1.4 Derived Pharmacokinetic Parameters

**Volume of Distribution at Steady State:**

```
V_ss = V_c + V_p
```

**Terminal Elimination Half-Life:**

```
t_{1/2,β} = ln(2) / β
```

**Distribution Half-Life:**

```
t_{1/2,α} = ln(2) / α
```

**Area Under the Curve (0 to infinity):**

```
AUC_0→∞ = Dose / CL
```

**Area Under the Curve (0 to 24h) for Once-Daily Dosing:**

```
AUC_{0→24} = ∫[0 to 24] C(t) dt
```

Calculated numerically using the analytical solution.

**Peak Concentration (Cmax):**

```
C_max = C(t = T_inf)
```

**Trough Concentration at Steady State (24h interval):**

```
C_min = C(t = 24h) / (1 - e^(-β·24))
```

The denominator accounts for accumulation to steady state.

---

## 2. Population Model Structure

### 2.1 Between-Subject Variability

Individual pharmacokinetic parameters are modeled using log-normal distributions:

```
θ_i = θ_pop · exp(η_i)
```

Where:
- `θ_i` = individual parameter value for subject i
- `θ_pop` = population typical value
- `η_i` ~ N(0, ω²) = random effect for subject i
- `ω²` = between-subject variance

**Applied to all four PK parameters:**

```
CL_i = CL_pop · exp(η_{CL,i})
V_{c,i} = V_{c,pop} · exp(η_{Vc,i})
Q_i = Q_pop · exp(η_{Q,i})
V_{p,i} = V_{p,pop} · exp(η_{Vp,i})
```

### 2.2 Covariate Model

Individual parameters are further modified by patient-specific covariates.

**Clearance Model:**

```
CL_i = CL_pop · (WT_i/70)^{θ_CL_WT} · (CrCL_i/100)^{θ_CL_CrCL} · θ_CL_SHOCK^{SHOCK_i} · exp(η_{CL,i})
```

Where:
- `WT_i` = body weight (kg), standardized to 70 kg
- `CrCL_i` = creatinine clearance (mL/min), standardized to 100 mL/min
- `SHOCK_i` = indicator for septic shock (0 or 1)
- `θ_CL_WT` = allometric exponent for weight effect on clearance
- `θ_CL_CrCL` = power exponent for CrCL effect on clearance
- `θ_CL_SHOCK` = multiplicative effect of septic shock on clearance

**Central Volume Model:**

```
V_{c,i} = V_{c,pop} · (WT_i/70)^{θ_Vc_WT} · exp(η_{Vc,i})
```

**Peripheral Compartment (Q and V_p):**

No covariate effects included (empirical evaluation showed non-significance):

```
Q_i = Q_pop · exp(η_{Q,i})
V_{p,i} = V_{p,pop} · exp(η_{Vp,i})
```

### 2.3 Residual Error Model

Observed concentrations deviate from model predictions due to measurement error and model misspecification:

```
C_{obs,ij} = C_{pred,ij} · (1 + ε_{prop,ij}) + ε_{add,ij}
```

Where:
- `C_{obs,ij}` = observed concentration for subject i at time j
- `C_{pred,ij}` = model-predicted concentration
- `ε_{prop,ij}` ~ N(0, σ²_prop) = proportional error
- `ε_{add,ij}` ~ N(0, σ²_add) = additive error (mg/L)

**Combined Error Model in Log-Space:**

For computational efficiency in Bayesian estimation:

```
log(C_{obs,ij}) ~ Normal(log(C_{pred,ij}), σ²_total)

σ²_total = sqrt((σ_prop)² + (σ_add / C_{pred,ij})²)
```

---

## 3. Bayesian Statistical Model

### 3.1 Likelihood Function

The likelihood of observed data given parameters:

```
L(Data | θ, η, σ) = ∏[i=1 to N] ∏[j=1 to n_i] p(C_{obs,ij} | θ_i, σ)
```

Where:
- `N` = number of subjects
- `n_i` = number of observations for subject i
- `p(·)` = probability density function (log-normal for concentrations)

### 3.2 Prior Distributions

Weakly informative priors based on published aminoglycoside literature:

**Population Parameters (Typical Values):**

```
CL_pop ~ Normal(5, 2²)   [truncated at 0]
V_{c,pop} ~ Normal(15, 5²) [truncated at 0]
Q_pop ~ Normal(10, 5²)   [truncated at 0]
V_{p,pop} ~ Normal(10, 5²) [truncated at 0]
```

**Covariate Effects:**

```
θ_CL_WT ~ Normal(0.75, 0.1²)         [allometric theory]
θ_Vc_WT ~ Normal(1.0, 0.15²)          [linear scaling]
θ_CL_CrCL ~ Normal(0.75, 0.15²)       [kidney function scaling]
θ_CL_SHOCK ~ Normal(0.85, 0.1²)       [15% reduction in shock]
```

**Between-Subject Variance Parameters:**

```
ω_CL ~ Half-Cauchy(0, 0.5)
ω_Vc ~ Half-Cauchy(0, 0.5)
ω_Q ~ Half-Cauchy(0, 1)
ω_Vp ~ Half-Cauchy(0, 1)
```

Half-Cauchy priors provide weak regularization while allowing substantial variability.

**Residual Error Parameters:**

```
σ_prop ~ Half-Cauchy(0, 0.25)
σ_add ~ Half-Cauchy(0, 1)
```

### 3.3 Posterior Distribution

By Bayes' theorem:

```
p(θ, η, σ | Data) ∝ L(Data | θ, η, σ) · p(θ) · p(η | ω) · p(σ)
```

**Markov Chain Monte Carlo Sampling:**

The posterior is sampled using the No-U-Turn Sampler (NUTS), an adaptive Hamiltonian Monte Carlo algorithm.

**Sampling Scheme:**
- 4 independent chains
- 2,000 iterations per chain (1,000 warmup, 1,000 sampling)
- Total: 4,000 posterior samples
- Thinning: none (NUTS provides low autocorrelation)

**Convergence Diagnostics:**

Gelman-Rubin statistic:

```
R̂ = sqrt(Var_total / Var_within)

where:
Var_total = ((n-1)/n)·Var_within + (1/n)·Var_between
```

Convergence achieved when R̂ < 1.01 for all parameters.

---

## 4. Pharmacokinetic/Pharmacodynamic Index Calculations

### 4.1 Concentration-Dependent Indices

**Peak Concentration over MIC:**

```
Cmax/MIC = C_max / MIC
```

**Area Under Curve over MIC:**

```
AUC/MIC = AUC_{0→24} / MIC
```

### 4.2 Target Attainment Probability

For a given dose and patient characteristics, probability of achieving target:

**Individual Target Attainment:**

```
P(Target | Dose, Covariates) = P(PK/PD Index ≥ Target_value)
```

Calculated using posterior predictive distribution:

```
P(Cmax/MIC ≥ 8) = (1/M) · ∑[m=1 to M] I(Cmax^(m)/MIC ≥ 8)
```

Where:
- `M` = number of posterior samples (4,000)
- `Cmax^(m)` = peak concentration from m-th posterior draw
- `I(·)` = indicator function (1 if condition met, 0 otherwise)

**Population Target Attainment (PTA):**

```
PTA(Dose, MIC) = (1/N) · ∑[i=1 to N] P(Target_i | Dose, Covariates_i)
```

Averaged over patient population or covariate distribution.

### 4.3 Cumulative Fraction of Response

Accounting for MIC distribution in population:

```
CFR(Dose) = ∫[0 to ∞] PTA(Dose, MIC) · f(MIC) dMIC
```

Where:
- `f(MIC)` = probability density function of MIC values
- Integration performed numerically over observed MIC range

For log-normal MIC distribution:

```
f(MIC) = (1/(MIC·σ·√(2π))) · exp(-((ln(MIC) - μ)²)/(2σ²))
```

---

## 5. Machine Learning Model Formulations

### 5.1 XGBoost Objective Function

**Gradient Boosting Framework:**

```
F_m(x) = F_{m-1}(x) + ν · h_m(x)
```

Where:
- `F_m(x)` = ensemble prediction after m trees
- `h_m(x)` = m-th tree (weak learner)
- `ν` = learning rate (shrinkage parameter)

**Additive Training:**

At each iteration m, optimize:

```
L^(m) = ∑[i=1 to n] l(y_i, F_{m-1}(x_i) + h_m(x_i)) + Ω(h_m)
```

Where:
- `l(·)` = loss function (binary cross-entropy for classification)
- `Ω(h_m)` = regularization term for tree complexity

**Binary Cross-Entropy Loss:**

```
l(y, ŷ) = -y·log(p) - (1-y)·log(1-p)

where:  p = 1 / (1 + exp(-ŷ))  [sigmoid transformation]
```

**Regularization Term:**

```
Ω(h) = γ·T + (λ/2)·∑[j=1 to T] w_j²
```

Where:
- `T` = number of leaves in tree
- `w_j` = weight (prediction) of leaf j
- `γ` = complexity penalty per leaf
- `λ` = L2 regularization on leaf weights

**Second-Order Taylor Approximation:**

XGBoost uses second-order optimization:

```
L^(m) ≈ ∑[i=1 to n] [l(y_i, F_{m-1}(x_i)) + g_i·h_m(x_i) + (1/2)·h_i·h_m(x_i)²] + Ω(h_m)
```

Where:
- `g_i = ∂l/∂ŷ` = first derivative (gradient)
- `h_i = ∂²l/∂ŷ²` = second derivative (Hessian)

**Optimal Leaf Weight:**

For a given tree structure, optimal weight for leaf j:

```
w_j* = -[∑[i∈I_j] g_i] / [∑[i∈I_j] h_i + λ]
```

Where `I_j` = set of instances in leaf j.

### 5.2 SMOTE Algorithm

**Synthetic Sample Generation:**

For minority class instance `x_i`:

1. Find k nearest neighbors from same class: `NN(x_i) = {x_{i1}, ..., x_{ik}}`

2. For each neighbor `x_{ij}`:

```
x_synthetic = x_i + λ · (x_{ij} - x_i)
```

Where `λ ~ Uniform(0, 1)` is random interpolation factor.

3. Repeat until desired class balance achieved.

**Number of Synthetic Samples:**

```
N_synthetic = N_majority - N_minority
```

For 50-50 balance in training set.

**Distance Metric:**

Euclidean distance in normalized feature space:

```
d(x_i, x_j) = sqrt(∑[k=1 to p] ((x_{ik} - x_{jk}) / σ_k)²)
```

Where:
- `p` = number of features
- `σ_k` = standard deviation of feature k (for normalization)

### 5.3 Ensemble Stacking

**Level 0 (Base Models):**

Train M base models on training data:

```
f_m(x) : X → [0,1]    for m = 1, ..., M
```

Using 5-fold cross-validation to generate out-of-fold predictions:

```
Z_train = [f_1(x_1^{oob}), ..., f_M(x_1^{oob})]
          [f_1(x_2^{oob}), ..., f_M(x_2^{oob})]
          ...
          [f_1(x_n^{oob}), ..., f_M(x_n^{oob})]
```

Where `x_i^{oob}` indicates out-of-bag prediction for instance i.

**Level 1 (Meta-Model):**

Train logistic regression on base model predictions:

```
ŷ = σ(β_0 + β_1·f_1(x) + ... + β_M·f_M(x))
```

Where `σ(z) = 1/(1 + exp(-z))` is the sigmoid function.

**Regularized Meta-Model:**

To prevent overfitting, use L2-regularized logistic regression:

```
L(β) = -∑[i=1 to n] [y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i)] + α·∑[m=1 to M] β_m²
```

Regularization parameter `α` selected via cross-validation.

---

## 6. Multi-Objective Dose Optimization

### 6.1 Objective Function

**Weighted Multi-Objective:**

```
J(Dose | Patient) = w_cure·P(Cure | Dose)
                    + w_safe·P(No_Nephrotox | Dose)
                    + w_cmax·P(Cmax/MIC ≥ 8 | Dose)
                    + w_trough·P(Cmin < 2 | Dose)
```

Where:
- `w_cure = 0.4` (cure probability weight)
- `w_safe = 0.3` (safety weight)
- `w_cmax = 0.2` (efficacy target weight)
- `w_trough = 0.1` (safety target weight)
- `∑w_i = 1.0` (weights sum to 1)

**Constraint:**

```
Dose ∈ [200, 1600] mg
```

### 6.2 Bayesian Optimization

**Gaussian Process Surrogate:**

Model objective as Gaussian process:

```
J(Dose) ~ GP(μ(Dose), k(Dose, Dose'))
```

Where:
- `μ(·)` = mean function (initialized to constant)
- `k(·,·)` = covariance kernel (Matérn 5/2)

**Matérn 5/2 Kernel:**

```
k(d, d') = σ²·(1 + √5·r + (5/3)·r²)·exp(-√5·r)

where:  r = |d - d'| / ℓ
```

Parameters:
- `σ²` = signal variance
- `ℓ` = length scale

**Acquisition Function (Expected Improvement):**

```
EI(Dose) = E[max(J(Dose) - J_best, 0)]
         = (J(Dose) - J_best)·Φ(Z) + σ(Dose)·φ(Z)

where:  Z = (J(Dose) - J_best) / σ(Dose)
        Φ(·) = standard normal CDF
        φ(·) = standard normal PDF
        σ(Dose) = posterior standard deviation at Dose
```

**Optimization Algorithm:**

```
Initialize: Evaluate J at 5 random doses
For iteration t = 1 to 25:
    1. Fit GP to observations {Dose_i, J_i}
    2. Find Dose* = argmax[Dose] EI(Dose)
    3. Evaluate J(Dose*)
    4. Update observations
End
Return: Dose_opt = argmax[all evaluated] J(Dose)
```

### 6.3 Probability Calculations

**Cure Probability:**

Predicted by ML model:

```
P(Cure | Dose, Patient) = f_cure(PK/PD_indices(Dose), Patient_features)
```

**Nephrotoxicity Probability:**

```
P(Nephrotox | Dose, Patient) = f_tox(Patient_features)
P(No_Nephrotox) = 1 - P(Nephrotox)
```

**PK/PD Target Probabilities:**

Using sigmoid approximation:

```
P(Cmax/MIC ≥ 8) = 1 / (1 + exp(-k_1·(Cmax/MIC - 8)))
P(Cmin < 2) = 1 / (1 + exp(k_2·(Cmin - 2)))
```

Where `k_1, k_2` are steepness parameters (estimated from data, typically k ≈ 2).

---

## 7. Model Evaluation Metrics

### 7.1 ROC-AUC

**Area Under ROC Curve:**

```
AUC = ∫[0 to 1] TPR(FPR) dFPR
```

Equivalent to probability that randomly chosen positive ranked higher than randomly chosen negative:

```
AUC = P(ŷ_positive > ŷ_negative)
```

**Empirical Calculation (Trapezoidal Rule):**

Sort predictions, compute TPR and FPR at each threshold, integrate.

### 7.2 Brier Score

**Calibration Metric:**

```
BS = (1/n)·∑[i=1 to n] (ŷ_i - y_i)²
```

Where:
- `ŷ_i` = predicted probability
- `y_i` = observed outcome (0 or 1)
- Lower scores indicate better calibration

**Decomposition:**

```
BS = Calibration + Refinement - Uncertainty
```

### 7.3 Confidence Intervals (Bootstrap)

**Bootstrap Algorithm:**

```
For b = 1 to B (typically B = 1000):
    1. Resample test set with replacement (n samples)
    2. Calculate metric on bootstrap sample: m_b
End
CI_95% = [percentile(m, 2.5%), percentile(m, 97.5%)]
```

---

## 8. Computational Implementation

### 8.1 PyMC Model Specification (Pseudocode)

```python
with pm.Model() as popk_model:
    # Priors for population parameters
    CL_pop = pm.Normal('CL_pop', mu=5, sigma=2)
    Vc_pop = pm.Normal('Vc_pop', mu=15, sigma=5)

    # Priors for covariate effects
    theta_CL_WT = pm.Normal('theta_CL_WT', mu=0.75, sigma=0.1)
    theta_CL_CrCL = pm.Normal('theta_CL_CrCL', mu=0.75, sigma=0.15)

    # Between-subject variability
    omega_CL = pm.HalfCauchy('omega_CL', beta=0.5)
    eta_CL = pm.Normal('eta_CL', mu=0, sigma=omega_CL, shape=n_subjects)

    # Individual clearance
    CL_i = CL_pop * (WT / 70)**theta_CL_WT * (CrCL / 100)**theta_CL_CrCL * pm.math.exp(eta_CL)

    # Concentration predictions
    C_pred = two_compartment_solution(dose, CL_i, Vc_i, Q_i, Vp_i, time)

    # Likelihood
    C_obs = pm.Lognormal('C_obs', mu=pm.math.log(C_pred), sigma=sigma_total, observed=data)

    # MCMC sampling
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
```

### 8.2 XGBoost Implementation (Pseudocode)

```python
# Hyperparameters
params = {
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 3,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

# Model training
model = XGBClassifier(**params)
model.fit(X_train_smote, y_train_smote,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=20,
          verbose=False)

# Predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
```

---

## Summary of Key Equations

| **Component** | **Equation** | **Parameters** |
|--------------|-------------|----------------|
| PK Model | `dA_c/dt = R - (CL/V_c)·A_c - Q·(A_c/V_c - A_p/V_p)` | CL, Vc, Q, Vp |
| Covariate Model | `CL_i = CL_pop·(WT/70)^0.75·(CrCL/100)^0.75` | Weights, CrCL |
| PK/PD Index | `Cmax/MIC = C_max / MIC` | Concentration, MIC |
| Target Probability | `P(Target) = 1/(1+exp(-k·(Index-Target)))` | k, Target threshold |
| XGBoost | `F_m = F_{m-1} + ν·h_m` | ν, tree parameters |
| Optimization | `J = ∑w_i·P_i(Dose)` | Weights w_i |

This mathematical appendix provides the complete theoretical foundation for all modeling and optimization procedures described in the main manuscript.
