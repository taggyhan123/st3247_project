"""
Robustness checks: vary random seed and NPE training-set size.
Outputs a summary table to stdout and saves results to results/robustness.npz.
"""

import numpy as np
import os
import time
import torch

from data_loader import load_all
from simulator import simulate, simulate_fast
from summary_statistic import (
    compute_observed_summaries,
    SummaryStatistic,
    SummarySubset,
)
from abc_rejection import BasicRejectionABC
from npe import NeuralPosteriorEstimation
from abc_utils import PriorSampler, SummaryStatisticNormalizer


def run_pilot(rng, prior_sampler, n_pilot=2000):
    p_betas, p_gammas, p_rhos = prior_sampler.sample(n_pilot)
    pilot_summaries = []
    for i in range(n_pilot):
        inf, rew, deg = simulate(p_betas[i], p_gammas[i], p_rhos[i], rng=rng)
        pilot_summaries.append(SummaryStatistic(inf, rew, deg))
    return SummaryStatisticNormalizer(pilot_summaries)


def run_rejection_abc(rng, s_obs, normalizer, prior_sampler, n_sim=50_000):
    runner = BasicRejectionABC(rng=rng, normalizer=normalizer,
                               prior_sampler=prior_sampler, verbose=False)
    thetas, distances, summaries, acc_mask, _ = runner.run(
        s_obs=s_obs, n_sim=n_sim, subset=SummarySubset.ALL,
        acceptance_quantile=0.01, n_reps_per_sim=1,
    )
    return thetas, summaries, thetas[acc_mask]


def run_npe(rng, thetas, summaries, s_obs, prior_sampler):
    npe = NeuralPosteriorEstimation(rng=rng, prior_sampler=prior_sampler, verbose=False)
    samples, _ = npe.run(
        thetas=thetas, summaries=summaries, s_obs=s_obs,
        n_posterior_samples=10_000, density_estimator="maf", subset=SummarySubset.ALL,
    )
    samples[:, 0] = np.clip(samples[:, 0], *PriorSampler.PRIOR_BOUNDS["beta"])
    samples[:, 1] = np.clip(samples[:, 1], *PriorSampler.PRIOR_BOUNDS["gamma"])
    samples[:, 2] = np.clip(samples[:, 2], *PriorSampler.PRIOR_BOUNDS["rho"])
    return samples


def stats(samples):
    meds = np.median(samples, axis=0)
    widths = np.percentile(samples, 97.5, axis=0) - np.percentile(samples, 2.5, axis=0)
    return meds, widths


def main():
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
        normalizer = run_pilot(rng, prior_sampler)

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
    normalizer_main = run_pilot(rng_main, prior_sampler_main)

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
