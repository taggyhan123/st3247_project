"""Robustness checks: vary random seed and NPE training-set size.

This script runs two experiments to verify that the main inference results
are not artefacts of a particular random seed or hyperparameter choice:
  Experiment 1 — Re-runs the full rejection ABC + NPE pipeline across
      multiple seeds and reports posterior median shifts and CI width CVs.
  Experiment 2 — Varies the NPE training-set size to check convergence.

Outputs a summary table to stdout and saves results to results/robustness.npz.
"""

import numpy as np
import os
import time
import torch

from data_loader import load_all
from simulator import simulate_fast
from summary_statistic import (
    compute_observed_summaries,
    SummaryStatistic,
    SummarySubset,
)
from abc_rejection import BasicRejectionABC
from npe import NeuralPosteriorEstimation
from abc_utils import PriorSampler, run_pilot


def run_rejection_abc(rng, s_obs, normalizer, prior_sampler, n_sim=50_000):
    """Runs rejection ABC and returns all proposals plus accepted samples.

    Args:
        rng: A NumPy random number generator instance.
        s_obs: The observed summary statistics.
        normalizer: A fitted normalizer for distance computation.
        prior_sampler: An object to sample from the prior distribution.
        n_sim: The number of simulations to run. Defaults to 50,000.

    Returns:
        tuple: A tuple containing:
            - thetas (np.ndarray): All proposed parameter samples of shape
                (n_sim, 3).
            - summaries (np.ndarray): Summary statistics for all proposals
                of shape (n_sim, 12).
            - accepted (np.ndarray): The accepted parameter samples.
    """
    runner = BasicRejectionABC(rng=rng, normalizer=normalizer,
                               prior_sampler=prior_sampler, verbose=False)
    thetas, distances, summaries, acc_mask, _ = runner.run(
        s_obs=s_obs, n_sim=n_sim, subset=SummarySubset.ALL,
        acceptance_quantile=0.01, n_reps_per_sim=1,
    )
    return thetas, summaries, thetas[acc_mask]


def run_npe(rng, thetas, summaries, s_obs, prior_sampler):
    """Trains NPE and returns clipped posterior samples.

    Args:
        rng: A NumPy random number generator instance.
        thetas: An array of simulated parameter samples of shape (n, 3).
        summaries: An array of summary statistics of shape (n, 12).
        s_obs: The observed summary statistics.
        prior_sampler: An object to provide the prior distribution bounds.

    Returns:
        np.ndarray: Posterior samples of shape (10000, 3), clipped to the
            prior bounds.
    """
    npe = NeuralPosteriorEstimation(rng=rng, prior_sampler=prior_sampler, verbose=False)
    samples, _ = npe.run(
        thetas=thetas, summaries=summaries, s_obs=s_obs,
        n_posterior_samples=10_000, density_estimator="maf", subset=SummarySubset.ALL,
    )
    PriorSampler.clip_to_prior(samples)
    return samples


def stats(samples):
    """Computes posterior medians and 95% credible interval widths.

    Args:
        samples: An array of posterior samples of shape (n_samples, n_params).

    Returns:
        tuple: A tuple containing:
            - meds (np.ndarray): Per-parameter posterior medians.
            - widths (np.ndarray): Per-parameter 95% CI widths
                (97.5th - 2.5th percentile).
    """
    meds = np.median(samples, axis=0)
    widths = np.percentile(samples, 97.5, axis=0) - np.percentile(samples, 2.5, axis=0)
    return meds, widths


def main():
    """Runs all robustness experiments and saves results.

    Experiment 1 varies the random seed across three values (42, 123, 9999)
    and reports posterior stability for both rejection ABC and NPE.
    Experiment 2 varies the NPE training-set size (10k, 25k, 50k) to
    check convergence. Results are saved to results/robustness.npz.
    """
    os.makedirs("results", exist_ok=True)

    # Warm up Numba
    print("Warming up Numba ...")
    _ = simulate_fast(0.2, 0.1, 0.3, seed=0)

    # Load observed data
    inf_ts, rew_ts, deg_hist = load_all()
    observed_summaries = compute_observed_summaries(inf_ts, rew_ts, deg_hist)
    s_obs = SummaryStatistic.aggregate_summary_statistics(observed_summaries, agg="median")

    # ---- Experiment 1: Vary random seed (full pipeline) ----
    seeds = [42, 123, 9999]
    param_names = ["beta", "gamma", "rho"]

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Varying random seed (rejection ABC + NPE, N=50k)")
    print("=" * 70)

    seed_rej_meds = []
    seed_rej_widths = []
    seed_npe_meds = []
    seed_npe_widths = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        rng = np.random.default_rng(seed)
        prior_sampler = PriorSampler(rng)
        normalizer = run_pilot(prior_sampler, rng)

        t0 = time.time()
        thetas, summaries, rej_accepted = run_rejection_abc(
            rng, s_obs, normalizer, prior_sampler, n_sim=50_000,
        )
        print(f"  Rejection ABC: {time.time() - t0:.1f}s, {len(rej_accepted)} accepted")

        t0 = time.time()
        npe_samples = run_npe(rng, thetas, summaries, s_obs, prior_sampler)
        print(f"  NPE: {time.time() - t0:.1f}s")

        m_r, w_r = stats(rej_accepted)
        m_n, w_n = stats(npe_samples)
        seed_rej_meds.append(m_r)
        seed_rej_widths.append(w_r)
        seed_npe_meds.append(m_n)
        seed_npe_widths.append(w_n)

    seed_rej_meds = np.array(seed_rej_meds)
    seed_rej_widths = np.array(seed_rej_widths)
    seed_npe_meds = np.array(seed_npe_meds)
    seed_npe_widths = np.array(seed_npe_widths)

    print("\n--- Seed sensitivity (Rejection ABC) ---")
    print(f"  {'Param':>6s}  {'Med range':>12s}  {'Width range':>12s}  {'Width CV':>10s}")
    for k, p in enumerate(param_names):
        med_range = seed_rej_meds[:, k].max() - seed_rej_meds[:, k].min()
        w_cv = seed_rej_widths[:, k].std() / seed_rej_widths[:, k].mean()
        print(f"  {p:>6s}  {med_range:>12.4f}  "
              f"[{seed_rej_widths[:, k].min():.3f}-{seed_rej_widths[:, k].max():.3f}]  {w_cv:>10.3f}")

    print("\n--- Seed sensitivity (NPE) ---")
    print(f"  {'Param':>6s}  {'Med range':>12s}  {'Width range':>12s}  {'Width CV':>10s}")
    for k, p in enumerate(param_names):
        med_range = seed_npe_meds[:, k].max() - seed_npe_meds[:, k].min()
        w_cv = seed_npe_widths[:, k].std() / seed_npe_widths[:, k].mean()
        print(f"  {p:>6s}  {med_range:>12.4f}  "
              f"[{seed_npe_widths[:, k].min():.3f}-{seed_npe_widths[:, k].max():.3f}]  {w_cv:>10.3f}")

    # ---- Experiment 2: Vary NPE training-set size ----
    # Use the main seed and the full 50k rejection ABC run
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Varying NPE training-set size")
    print("=" * 70)

    rng_main = np.random.default_rng(6769)
    prior_sampler_main = PriorSampler(rng_main)
    normalizer_main = run_pilot(prior_sampler_main, rng_main)

    thetas_full, summaries_full, _ = run_rejection_abc(
        rng_main, s_obs, normalizer_main, prior_sampler_main, n_sim=50_000,
    )

    train_sizes = [10_000, 25_000, 50_000]
    size_npe_meds = []
    size_npe_widths = []

    for n in train_sizes:
        print(f"\n--- Training-set size = {n:,} ---")
        rng_npe = np.random.default_rng(6769)
        t0 = time.time()
        npe_samples = run_npe(
            rng_npe, thetas_full[:n], summaries_full[:n], s_obs, prior_sampler_main,
        )
        print(f"  NPE: {time.time() - t0:.1f}s")
        m, w = stats(npe_samples)
        size_npe_meds.append(m)
        size_npe_widths.append(w)

    size_npe_meds = np.array(size_npe_meds)
    size_npe_widths = np.array(size_npe_widths)

    print("\n--- NPE training-set sensitivity ---")
    print(f"  {'N_train':>8s}  {'beta_med':>9s} {'gamma_med':>10s} {'rho_med':>8s}  "
          f"{'CI_beta':>8s} {'CI_gamma':>9s} {'CI_rho':>7s}")
    for i, n in enumerate(train_sizes):
        m = size_npe_meds[i]
        w = size_npe_widths[i]
        print(f"  {n:>8,}  {m[0]:>9.4f} {m[1]:>10.4f} {m[2]:>8.4f}  "
              f"{w[0]:>8.4f} {w[1]:>9.4f} {w[2]:>7.4f}")

    # Save all results
    np.savez(
        "results/robustness.npz",
        seeds=np.array(seeds),
        seed_rej_meds=seed_rej_meds,
        seed_rej_widths=seed_rej_widths,
        seed_npe_meds=seed_npe_meds,
        seed_npe_widths=seed_npe_widths,
        train_sizes=np.array(train_sizes),
        size_npe_meds=size_npe_meds,
        size_npe_widths=size_npe_widths,
    )
    print("\nResults saved to results/robustness.npz")


if __name__ == "__main__":
    main()
