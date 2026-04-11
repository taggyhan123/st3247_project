import numpy as np
import os
import time

# Util functions
from data_loader import load_all
from simulator import simulate, simulate_fast
from summary_statistics_oop import (
    compute_observed_summaries,
    SummaryStatistic,
    SummarySubset
)

# Algorithms
from st3247_project.abc_rejection import BasicRejectionABC
from abc_regression import regression_adjust


# Algorithm Utils
from st3247_project.abc_utils import (
    PriorSampler,
    SummaryStatisticNormalizer
)

RNG_SEED = 6769

def main():
    # 1. Setup and load data
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print("Loading data and warming up Numba...")
    _warm_up_numba()

    inf_ts, rew_ts, deg_hist = load_all()
    
    # 2. ------------ ABC Based Algorithms ------------
    observed_summaries = compute_observed_summaries(inf_ts, rew_ts, deg_hist)
    median_observed_summary = SummaryStatistic.aggregate_summary_statistics(observed_summaries, agg="median")
    
    s_obs_array = median_observed_summary.get_summaries()
    print(f"Observed summaries ({len(s_obs_array)} stats): {np.round(s_obs_array, 3)}")

    prior_sampler = PriorSampler(rng)

    summary_statistic_normalizer = run_pilot_for_normalizer(prior_sampler, rng)
    
    # a. Basic ABC Rejection Algorithm
    thetas, distances, summaries, acc_mask, thr_rej = rejection_abc_run(
        rng, median_observed_summary, summary_statistic_normalizer, prior_sampler
    )
    
    np.savez('results/rejection_abc_50k.npz',
             thetas=thetas, distances=distances, summaries=summaries, 
             mads=summary_statistic_normalizer.mads, s_obs=s_obs_array)
    print("Results saved to results/rejection_abc_50k.npz")

    # b. ABC Regression Adjustment
    adjusted_thetas = regression_adjust_run(thetas, summaries, distances, s_obs_array, acc_mask)
    
    np.savez('results/regression_abc.npz', adjusted_thetas=adjusted_thetas)
    print("Results saved to results/regression_abc.npz")

    # c. ABC MCMC

    # d. SMC-ABC

    # 3. Sequential Likelihood

def _warm_up_numba():
    _ = simulate_fast(0.2, 0.1, 0.3, seed=0)

def run_pilot_for_normalizer(prior_sampler: PriorSampler, rng: np.random.Generator) -> SummaryStatisticNormalizer:
    """
    Runs a pilot simulation to compute Median Absolute Deviations (MADs)
    for summary statistic normalization.
    """
    N_PILOT = 2000
    print(f"\nRunning pilot ({N_PILOT} sims) for MAD normalisation...")
    p_betas, p_gammas, p_rhos = prior_sampler.sample(N_PILOT)
    pilot_summaries = []
    for i in range(N_PILOT):
        inf, rew, deg = simulate(p_betas[i], p_gammas[i], p_rhos[i], rng=rng)
        pilot_summaries.append(SummaryStatistic(inf, rew, deg))
    
    return SummaryStatisticNormalizer(pilot_summaries)

def rejection_abc_run(rng: np.random.Generator,
                      observed_summary: SummaryStatistic,
                      normalizer: SummaryStatisticNormalizer,
                      prior_sampler: PriorSampler):
    # Parameters
    t0 = time.time()
    N_SIM = 50000
    N_REPS_PER_SIM = 1

    print("\n" + "=" * 70)
    print(f"METHOD 1: Rejection ABC ({N_SIM:,} simulations)")
    print("=" * 70)

    abc_runner = BasicRejectionABC(rng=rng,
                                   normalizer=normalizer,
                                   prior_sampler=prior_sampler,
                                   verbose=True)
    result = abc_runner.run(s_obs=observed_summary,
                   n_sim=N_SIM,
                   subset=SummarySubset.ALL,
                   acceptance_quantile=0.01,
                   n_reps_per_sim=N_REPS_PER_SIM)
    
    t_rej = time.time() - t0
    print(f"Completed in {t_rej:.1f}s")

    return result

def regression_adjust_run(thetas, summaries, distances, s_obs_array, acc_mask):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("METHOD 2: Regression-Adjusted ABC")
    print("=" * 70)
    
    acc_reg = thetas[acc_mask]
    acc_s_reg = summaries[acc_mask]
    acc_d_reg = distances[acc_mask]
    
    print(f"Using {len(acc_reg)} accepted samples from Rejection ABC for regression adjustment")
    
    adjusted_thetas = regression_adjust(acc_reg, acc_s_reg, acc_d_reg, s_obs_array)
    adjusted_thetas[:, 0] = np.clip(adjusted_thetas[:, 0], *PriorSampler.PRIOR_BOUNDS['beta'])
    adjusted_thetas[:, 1] = np.clip(adjusted_thetas[:, 1], *PriorSampler.PRIOR_BOUNDS['gamma'])
    adjusted_thetas[:, 2] = np.clip(adjusted_thetas[:, 2], *PriorSampler.PRIOR_BOUNDS['rho'])
    
    t_regadj = time.time() - t0
    print(f"Regression adjustment took {t_regadj:.3f}s")
    
    return adjusted_thetas

if __name__ == "__main__":
    main()