"""Refit the Phase-2 Bayesian PopPK model with a real observation likelihood.

Background: the original `phase2_population_pk.py` defines priors but never
adds an observed-data likelihood (see line ~297, "we'll use a prior
predictive approach"). Its trace is therefore a prior sample, not a
posterior. This script preserves the model's prior structure verbatim and
adds the missing likelihood, using a multi-dose superposition forward
model implemented in PyTensor so NUTS can take gradients.

Outputs (so the original artifacts stay untouched for the audit trail):
    models/phase2_popk_trace_likelihood.nc
    results/phase2_diagnostics/parameter_summary_likelihood.csv

Usage:
    python refit_popk.py                # full run
    python refit_popk.py --smoke        # smoke test: 50 patients, 200 draws
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "popk_dataset.csv"
MODELS_DIR = REPO_ROOT / "models"
DIAG_DIR = REPO_ROOT / "results" / "phase2_diagnostics"
TRACE_OUT = MODELS_DIR / "phase2_popk_trace_likelihood.nc"
SUMMARY_OUT = DIAG_DIR / "parameter_summary_likelihood.csv"

# Drug-effect offsets (verbatim from phase2_population_pk.py)
DRUG_CL_EFFECT = np.log(4.2 / 5.5)
DRUG_VC_EFFECT = np.log(24.0 / 16.0)
DRUG_Q_EFFECT = np.log(1.1 / 12.0)


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def prepare_data():
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH)
    obs = df[df["EVID"] == 0].copy()
    dose = df[df["EVID"] == 1].copy()

    # Match phase2: keep patients with >= 3 observations.
    counts = obs.groupby("ID").size()
    keep_ids = counts[counts >= 3].index
    obs = obs[obs["ID"].isin(keep_ids)].copy()
    dose = dose[dose["ID"].isin(keep_ids)].copy()
    n_patients = int(keep_ids.size)

    # Patient covariates -> standardized
    pat = (obs.groupby("ID")
              .first()[["AGE", "WT", "CRCL", "DRUG"]]
              .sort_index()
              .reset_index())
    pat["AGE_std"]  = (pat["AGE"]  - pat["AGE"].mean())  / pat["AGE"].std()
    pat["WT_std"]   = (pat["WT"]   - pat["WT"].mean())   / pat["WT"].std()
    pat["CRCL_std"] = (pat["CRCL"] - pat["CRCL"].mean()) / pat["CRCL"].std()
    pat["pidx"] = np.arange(n_patients)
    id_to_pidx = dict(zip(pat["ID"], pat["pidx"]))

    # Sort obs and dose for stable index alignment
    obs = obs.sort_values(["ID", "TIME"]).reset_index(drop=True)
    dose = dose.sort_values(["ID", "TIME"]).reset_index(drop=True)
    obs["pidx"] = obs["ID"].map(id_to_pidx).astype(int)
    dose["pidx"] = dose["ID"].map(id_to_pidx).astype(int)

    # Build (n_obs x max_doses) padded arrays of prior-dose time and amount.
    max_doses = int(dose.groupby("ID").size().max())
    n_obs = len(obs)
    dose_times = np.zeros((n_obs, max_doses), dtype=np.float64)
    dose_amts  = np.zeros((n_obs, max_doses), dtype=np.float64)
    dose_mask  = np.zeros((n_obs, max_doses), dtype=np.float64)

    # Per-patient dose tables for fast lookup
    pat_doses = {}
    for pid, sub in dose.groupby("ID"):
        pat_doses[pid] = (sub["TIME"].to_numpy(),
                          sub["AMT"].to_numpy())

    for i, row in enumerate(obs.itertuples(index=False)):
        t_obs = row.TIME
        pid = row.ID
        dt, da = pat_doses.get(pid, (np.empty(0), np.empty(0)))
        prior = dt <= t_obs
        n_prior = int(prior.sum())
        if n_prior == 0:
            continue
        dose_times[i, :n_prior] = dt[prior]
        dose_amts[i, :n_prior]  = da[prior]
        dose_mask[i, :n_prior]  = 1.0

    print(f"  Patients: {n_patients}")
    print(f"  Observations: {n_obs}")
    print(f"  Max prior doses per obs: {max_doses}")
    print(f"  Mean prior doses per obs: {dose_mask.sum(axis=1).mean():.2f}")

    return {
        "obs": obs,
        "pat": pat,
        "n_patients": n_patients,
        "dose_times": dose_times,
        "dose_amts":  dose_amts,
        "dose_mask":  dose_mask,
        "obs_time":   obs["TIME"].to_numpy(),
        "dv":         obs["DV"].to_numpy(),
        "pidx":       obs["pidx"].to_numpy(),
    }


# --------------------------------------------------------------------------- #
# PyTensor 2-compartment forward model
# --------------------------------------------------------------------------- #
def two_comp_pt(time, dose, CL, Vc, Q, Vp, tau=1.0):
    """All inputs are PyTensor tensors broadcastable to the same shape.

    Returns the central-compartment concentration at `time` for a 1-h
    infusion of `dose` given individual (CL, Vc, Q, Vp).
    """
    k10 = CL / Vc
    k12 = Q / Vc
    k21 = Q / Vp
    sum_k = k10 + k12 + k21
    disc = pt.maximum(sum_k * sum_k - 4 * k10 * k21, 1e-10)
    sq = pt.sqrt(disc)
    alpha = 0.5 * (sum_k + sq)
    beta = 0.5 * (sum_k - sq)
    A = (dose / Vc) * (alpha - k21) / (alpha - beta)
    B = (dose / Vc) * (beta - k21) / (beta - alpha)
    c_inf = (A / alpha * (1.0 - pt.exp(-alpha * time))
             + B / beta * (1.0 - pt.exp(-beta * time)))
    c_post = (A / alpha * (1.0 - pt.exp(-alpha * tau)) * pt.exp(-alpha * (time - tau))
              + B / beta * (1.0 - pt.exp(-beta * tau)) * pt.exp(-beta * (time - tau)))
    return pt.switch(pt.le(time, tau), c_inf, c_post)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(data):
    pat = data["pat"]
    n_patients = data["n_patients"]

    age_std = pat["AGE_std"].to_numpy()
    wt_std  = pat["WT_std"].to_numpy()
    crcl_std= pat["CRCL_std"].to_numpy()
    drug    = pat["DRUG"].to_numpy().astype(np.float64)

    with pm.Model() as model:
        theta_CL = pm.Lognormal("theta_CL", mu=np.log(5.5), sigma=0.3)
        theta_Vc = pm.Lognormal("theta_Vc", mu=np.log(16.0), sigma=0.3)
        theta_Q  = pm.Lognormal("theta_Q",  mu=np.log(12.0), sigma=0.5)
        theta_Vp = pm.Lognormal("theta_Vp", mu=np.log(10.0), sigma=0.5)

        beta_CL_CRCL = pm.Normal("beta_CL_CRCL", mu=0.75, sigma=0.2)
        beta_CL_WT   = pm.Normal("beta_CL_WT",   mu=0.75, sigma=0.2)
        beta_Vc_WT   = pm.Normal("beta_Vc_WT",   mu=1.0,  sigma=0.2)
        beta_CL_AGE  = pm.Normal("beta_CL_AGE",  mu=-0.2, sigma=0.1)

        omega_CL = pm.HalfNormal("omega_CL", sigma=0.3)
        omega_Vc = pm.HalfNormal("omega_Vc", sigma=0.25)
        omega_Q  = pm.HalfNormal("omega_Q",  sigma=0.4)
        omega_Vp = pm.HalfNormal("omega_Vp", sigma=0.35)

        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_patients)
        eta_Vc = pm.Normal("eta_Vc", mu=0, sigma=1, shape=n_patients)
        eta_Q  = pm.Normal("eta_Q",  mu=0, sigma=1, shape=n_patients)
        eta_Vp = pm.Normal("eta_Vp", mu=0, sigma=1, shape=n_patients)

        CL_ind = pm.Deterministic("CL", theta_CL * pt.exp(
            beta_CL_CRCL * crcl_std + beta_CL_WT * wt_std +
            beta_CL_AGE * age_std + DRUG_CL_EFFECT * drug +
            omega_CL * eta_CL))
        Vc_ind = pm.Deterministic("Vc", theta_Vc * pt.exp(
            beta_Vc_WT * wt_std + DRUG_VC_EFFECT * drug +
            omega_Vc * eta_Vc))
        Q_ind  = pm.Deterministic("Q", theta_Q * pt.exp(
            DRUG_Q_EFFECT * drug + omega_Q * eta_Q))
        Vp_ind = pm.Deterministic("Vp", theta_Vp * pt.exp(omega_Vp * eta_Vp))

        sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.2)
        sigma_add  = pm.HalfNormal("sigma_add",  sigma=1.0)

        # Multi-dose forward model with superposition.
        pidx = data["pidx"]
        obs_time = data["obs_time"][:, None]              # (n_obs, 1)
        d_times = data["dose_times"]                       # (n_obs, max_doses)
        d_amts  = data["dose_amts"]
        d_mask  = data["dose_mask"]
        t_since = obs_time - d_times                       # (n_obs, max_doses)
        t_safe = pt.maximum(t_since, 0.0)

        CL_obs = CL_ind[pidx][:, None]                     # (n_obs, 1)
        Vc_obs = Vc_ind[pidx][:, None]
        Q_obs  = Q_ind[pidx][:, None]
        Vp_obs = Vp_ind[pidx][:, None]

        c_per_dose = two_comp_pt(t_safe, d_amts, CL_obs, Vc_obs, Q_obs, Vp_obs)
        c_pred = pt.sum(c_per_dose * d_mask, axis=1)
        c_pred = pt.maximum(c_pred, 1e-3)   # guard against numerical zeros
        pm.Deterministic("c_pred", c_pred)

        sd = pt.sqrt((sigma_prop * c_pred) ** 2 + sigma_add ** 2)
        pm.Normal("y_obs", mu=c_pred, sigma=sd, observed=data["dv"])

    return model


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def smoke_subset(data, n_patients=50, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    keep = np.sort(rng.choice(data["n_patients"], size=n_patients, replace=False))
    keep_set = set(keep.tolist())
    obs = data["obs"]
    mask = obs["pidx"].isin(keep_set).to_numpy()
    sub = {k: v for k, v in data.items()}
    # Remap pidx 0..n-1 in retained subset
    remap = {old: new for new, old in enumerate(keep)}
    sub["obs"] = obs[mask].copy()
    sub["pat"] = data["pat"][data["pat"]["pidx"].isin(keep_set)].copy()
    sub["pat"]["pidx"] = np.arange(len(sub["pat"]))
    sub["n_patients"] = len(sub["pat"])
    sub["dose_times"] = data["dose_times"][mask]
    sub["dose_amts"]  = data["dose_amts"][mask]
    sub["dose_mask"]  = data["dose_mask"][mask]
    sub["obs_time"]   = data["obs_time"][mask]
    sub["dv"]         = data["dv"][mask]
    sub["pidx"]       = np.array([remap[p] for p in data["pidx"][mask]])
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 50 patients, 200 draws.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    data = prepare_data()

    if args.smoke:
        print("\n[SMOKE TEST] Using 50 patients, 200 draws + 200 tune\n")
        data = smoke_subset(data, n_patients=50)
        draws, tune = 200, 200
        out_trace = MODELS_DIR / "phase2_popk_trace_smoke.nc"
        out_summary = DIAG_DIR / "parameter_summary_smoke.csv"
    else:
        draws, tune = args.draws, args.tune
        out_trace = TRACE_OUT
        out_summary = SUMMARY_OUT

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DIAG_DIR, exist_ok=True)

    model = build_model(data)

    print(f"\nSampling with nutpie: {draws} draws x {args.chains} chains, tune={tune}")
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=args.chains,
            random_seed=42, target_accept=0.9,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            progressbar=True,
        )

    idata.to_netcdf(out_trace)
    print(f"Wrote {out_trace}")

    summary = az.summary(idata,
        var_names=["theta_CL", "theta_Vc", "theta_Q", "theta_Vp",
                   "beta_CL_CRCL", "beta_CL_WT", "beta_Vc_WT", "beta_CL_AGE",
                   "omega_CL", "omega_Vc", "omega_Q", "omega_Vp",
                   "sigma_prop", "sigma_add"])
    summary.to_csv(out_summary)
    print(f"Wrote {out_summary}")
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()
