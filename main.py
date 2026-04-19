from __future__ import annotations

import numpy as np
import os
import time

from typing import Optional

# Util functions
from data_loader import load_all
from simulator import simulate, simulate_fast
from summary_statistic import (
    compute_observed_summaries,
    SummaryStatistic,
    SummarySubset
)

# Algorithms
from abc_rejection import BasicRejectionABC
from abc_regression import regression_adjust
from abc_mcmc import MCMCABC
from smc_abc import SMCABC
from synthetic_likelihood import SyntheticLikelihoodMCMC
from npe import NeuralPosteriorEstimation


# Algorithm Utils
from abc_utils import (
    PriorSampler,
    SummaryStatisticNormalizer
)

# Orchestration modules
import budget_matched
import synthetic_truth
import make_figures

RNG_SEED = 6769

class ResultAggregator:
    def __init__(self):
        self.methods = {}

    def add_result(self, method: str, samples: np.ndarray, weights: Optional[np.ndarray], time: float, n_sims: Optional[int]):
        self.methods[method] = {
            'samples': samples,
            'weights': weights,
            'time': time,
            'n_sims': n_sims
        }

    def print_comparison_table(self):
        print("\n" + "=" * 70)
        print("COMPARISON OF ALL METHODS")
        print("=" * 70)

        print(f"\n{'Method':>25s}  {'beta_med':>9s} {'gamma_med':>10s} {'rho_med':>8s}  "
              f"{'beta_95w':>9s} {'gamma_95w':>10s} {'rho_95w':>8s}  {'Time(s)':>8s}  {'N_sims':>10s}")
        print('-' * 120)

        for name, info in self.methods.items():
            s = info['samples']
            meds = [np.median(s[:, k]) for k in range(3)]
            widths = []
            for k in range(3):
                lo, hi = np.percentile(s[:, k], [2.5, 97.5])
                widths.append(hi - lo)
            n_sims_str = f'{info["n_sims"]:>10,}' if info["n_sims"] is not None else f'{"N/A":>10s}'
            print(f'{name:>25s}  {meds[0]:>9.4f} {meds[1]:>10.4f} {meds[2]:>8.4f}  '
                  f'{widths[0]:>9.4f} {widths[1]:>10.4f} {widths[2]:>8.4f}  {info["time"]:>8.1f}  {n_sims_str}')

        print(f"\n{'Method':>25s}  {'beta 95% CI':>20s}  {'gamma 95% CI':>20s}  {'rho 95% CI':>20s}")
        print('-' * 95)
        for name, info in self.methods.items():
            s = info['samples']
            cis = []
            for k in range(3):
                lo, hi = np.percentile(s[:, k], [2.5, 97.5])
                cis.append(f'[{lo:.4f}, {hi:.4f}]')
            print(f'{name:>25s}  {cis[0]:>20s}  {cis[1]:>20s}  {cis[2]:>20s}')


def main():
    # 1. Setup and load data
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    
    aggregator = ResultAggregator()

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
    rej_path = 'results/rejection_abc_50k.npz'
    if os.path.exists(rej_path):
        print(f"\nLoading Rejection ABC from {rej_path}...")
        data = np.load(rej_path)
        thetas = data['thetas']
        distances = data['distances']
        summaries = data['summaries']
        thr_rej = float(np.quantile(distances, 0.01))
        acc_mask = distances <= thr_rej
        aggregator.add_result('Rejection ABC (q=1%)', thetas[acc_mask], None, 0.0, 50000)
    else:
        thetas, distances, summaries, acc_mask, thr_rej = rejection_abc_run(
            rng, median_observed_summary, summary_statistic_normalizer, prior_sampler, aggregator
        )
        np.savez(rej_path,
                 thetas=thetas, distances=distances, summaries=summaries, 
                 mads=summary_statistic_normalizer.mads, s_obs=s_obs_array)
        print(f"Results saved to {rej_path}")

    # b. ABC Regression Adjustment
    reg_path = 'results/regression_abc.npz'
    if os.path.exists(reg_path):
        print(f"\nLoading Regression-Adjusted ABC from {reg_path}...")
        data = np.load(reg_path)
        adjusted_thetas = data['adjusted_thetas']
        aggregator.add_result('Reg. Adjusted ABC', adjusted_thetas, None, 0.0, 50000)
    else:
        adjusted_thetas = regression_adjust_run(thetas, summaries, distances, s_obs_array, acc_mask, aggregator)
        np.savez(reg_path, adjusted_thetas=adjusted_thetas)
        print(f"Results saved to {reg_path}")

    # c. ABC MCMC
    mcmc_path = 'results/abc_mcmc.npz'
    if os.path.exists(mcmc_path):
        print(f"\nLoading ABC-MCMC from {mcmc_path}...")
        data = np.load(mcmc_path)
        mcmc_chain = data['chain']
        mcmc_chain_post = data['chain_post']
        mcmc_acc_rate = float(data['acc_rate'])
        aggregator.add_result('ABC-MCMC', mcmc_chain_post, None, 0.0, 30000)
    else:
        mcmc_chain, mcmc_chain_post, mcmc_acc_rate = mcmc_abc_run(
            rng=rng,
            observed_summary=median_observed_summary,
            normalizer=summary_statistic_normalizer,
            prior_sampler=prior_sampler,
            abc_rej_accepted_thetas=thetas[acc_mask],
            abc_rej_threshold=thr_rej,
            aggregator=aggregator
        )
        np.savez(mcmc_path, chain=mcmc_chain, chain_post=mcmc_chain_post, acc_rate=mcmc_acc_rate)
        print(f"Results saved to {mcmc_path}")

    # d. SMC-ABC
    smc_path = 'results/smc_abc.npz'
    if os.path.exists(smc_path):
        print(f"\nLoading SMC-ABC from {smc_path}...")
        data = np.load(smc_path)
        smc_particles = data['particles']
        smc_weights = data['weights']
        smc_epsilons = data['epsilons']
        smc_total_sims = int(data['total_sims']) if 'total_sims' in data else None
        aggregator.add_result('SMC-ABC', smc_particles, smc_weights, 0.0, smc_total_sims)
    else:
        smc_particles, smc_weights, smc_epsilons, smc_all_particles, smc_total_sims = smc_abc_run(
            rng=rng,
            observed_summary=median_observed_summary,
            normalizer=summary_statistic_normalizer,
            prior_sampler=prior_sampler,
            aggregator=aggregator
        )
        np.savez(smc_path, particles=smc_particles, weights=smc_weights,
                 epsilons=np.array(smc_epsilons), total_sims=smc_total_sims)
        print(f"Results saved to {smc_path}")

    # 3. Synthetic Likelihood
    sl_path = 'results/synthetic_likelihood.npz'
    if os.path.exists(sl_path):
        print(f"\nLoading Synthetic Likelihood from {sl_path}...")
        data = np.load(sl_path)
        sl_chain = data['chain']
        sl_chain_post = data['chain_post']
        sl_acc_rate = float(data['acc_rate'])
        aggregator.add_result('Synthetic Likelihood', sl_chain_post, None, 0.0, 5000 * 200)
    else:
        sl_chain, sl_chain_post, sl_acc_rate = synthetic_likelihood_run(
            rng=rng,
            observed_summary=median_observed_summary,
            prior_sampler=prior_sampler,
            abc_rej_accepted_thetas=thetas[acc_mask],
            aggregator=aggregator
        )
        np.savez(sl_path, chain=sl_chain, chain_post=sl_chain_post, acc_rate=sl_acc_rate)
        print(f"Results saved to {sl_path}")

    # 4. Neural Posterior Estimation
    npe_path = 'results/npe.npz'
    if os.path.exists(npe_path):
        print(f"\nLoading Neural Posterior Estimation from {npe_path}...")
        data = np.load(npe_path)
        npe_samples = data['samples']
        aggregator.add_result('NPE (sbi, MAF)', npe_samples, None, 0.0, 50000)
    else:
        npe_samples = npe_run(
            rng=rng,
            thetas=thetas,
            summaries=summaries,
            observed_summary=median_observed_summary,
            prior_sampler=prior_sampler,
            aggregator=aggregator
        )
        np.savez(npe_path, samples=npe_samples)
        print(f"Results saved to {npe_path}")

    aggregator.print_comparison_table()

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
                      prior_sampler: PriorSampler,
                      aggregator: ResultAggregator):
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
    thetas, distances, summaries, acc_mask, thr_rej = abc_runner.run(
        s_obs=observed_summary,
        n_sim=N_SIM,
        subset=SummarySubset.ALL,
        acceptance_quantile=0.01,
        n_reps_per_sim=N_REPS_PER_SIM
    )
    
    t_rej = time.time() - t0
    print(f"Completed in {t_rej:.1f}s")
    
    aggregator.add_result('Rejection ABC (q=1%)', thetas[acc_mask], None, t_rej, N_SIM)

    return thetas, distances, summaries, acc_mask, thr_rej

def regression_adjust_run(thetas, summaries, distances, s_obs_array, acc_mask, aggregator: ResultAggregator):
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
    
    base_time = aggregator.methods['Rejection ABC (q=1%)']['time']
    base_sims = aggregator.methods['Rejection ABC (q=1%)']['n_sims']
    aggregator.add_result('Reg. Adjusted ABC', adjusted_thetas, None, base_time + t_regadj, base_sims)
    
    return adjusted_thetas

def mcmc_abc_run(rng: np.random.Generator,
                 observed_summary: SummaryStatistic,
                 normalizer: SummaryStatisticNormalizer,
                 prior_sampler: PriorSampler,
                 abc_rej_accepted_thetas: np.ndarray,
                 abc_rej_threshold: float,
                 aggregator: ResultAggregator):
    print("\n" + "=" * 70)
    print("METHOD 3: ABC-MCMC (Marjoram et al. 2003)")
    print("=" * 70)

    MCMC_BURN_IN = 5000
    MCMC_ITER = 30000
    MCMC_REPS_PER_SIM = 1
    MCMC_STEP_VARIANCE = 0.5
    ABC_REJECTION_THRESHOLD_MULTIPLIER = 1.2

    proposal_cov = np.cov(abc_rej_accepted_thetas.T) * MCMC_STEP_VARIANCE
    theta_init = np.median(abc_rej_accepted_thetas, axis=0)
    epsilon = abc_rej_threshold * ABC_REJECTION_THRESHOLD_MULTIPLIER
    
    print(f"epsilon = {epsilon:.4f}, theta_init = {theta_init}")
    
    abc_mcmc_runner = MCMCABC(rng=rng, normalizer=normalizer, prior_sampler=prior_sampler, verbose=True)
    
    t0 = time.time()
    chain, dist_chain, acc_rate = abc_mcmc_runner.run(
        s_obs=observed_summary, n_iter=MCMC_ITER, epsilon=epsilon,
        theta_init=theta_init, proposal_cov=proposal_cov, subset=SummarySubset.ALL, n_reps_per_sim=MCMC_REPS_PER_SIM
    )
    t_mcmc = time.time() - t0
    print(f"Completed in {t_mcmc:.1f}s, acceptance rate: {acc_rate:.4f}")
    
    chain_post = chain[MCMC_BURN_IN:]
    ess = MCMCABC.effective_sample_size(chain_post)
    print(f"ESS: beta={ess[0]:.0f}, gamma={ess[1]:.0f}, rho={ess[2]:.0f}")
    
    aggregator.add_result('ABC-MCMC', chain_post, None, t_mcmc, MCMC_ITER)
    
    return chain, chain_post, acc_rate

def smc_abc_run(rng: np.random.Generator,
                observed_summary: SummaryStatistic,
                normalizer: SummaryStatisticNormalizer,
                prior_sampler: PriorSampler,
                aggregator: ResultAggregator):
    print("\n" + "=" * 70)
    print("METHOD 4: SMC-ABC (Beaumont et al. 2009)")
    print("=" * 70)

    SMC_PARTICLES = 1000
    SMC_GENERATIONS = 10
    SMC_ALPHA = 0.5
    SMC_MIN_EPSILON = 0.5
    SMC_REPS_PER_SIM = 1

    abc_smc_runner = SMCABC(rng=rng, normalizer=normalizer, prior_sampler=prior_sampler, verbose=True)

    t0 = time.time()
    smc_particles, smc_weights, smc_epsilons, smc_all_particles, smc_total_sims = abc_smc_runner.run(
        s_obs=observed_summary, n_particles=SMC_PARTICLES, n_generations=SMC_GENERATIONS,
        alpha=SMC_ALPHA, min_epsilon=SMC_MIN_EPSILON, subset=SummarySubset.ALL, n_reps_per_sim=SMC_REPS_PER_SIM
    )
    t_smc = time.time() - t0
    print(f"Completed in {t_smc:.1f}s, {len(smc_epsilons)} generations, {smc_total_sims:,} total sims")
    print(f"Final epsilon: {smc_epsilons[-1]:.4f}")

    aggregator.add_result('SMC-ABC', smc_particles, smc_weights, t_smc, smc_total_sims)

    return smc_particles, smc_weights, smc_epsilons, smc_all_particles, smc_total_sims

def synthetic_likelihood_run(rng: np.random.Generator,
                             observed_summary: SummaryStatistic,
                             prior_sampler: PriorSampler,
                             abc_rej_accepted_thetas: np.ndarray,
                             aggregator: ResultAggregator):
    print("\n" + "=" * 70)
    print("METHOD 5: Synthetic Likelihood (Wood, 2010)")
    print("=" * 70)

    SL_ITER = 5000
    SL_SIM_PER_EVAL = 200
    SL_BURN_IN = 1000
    SL_STEP_VARIANCE = 0.3

    proposal_cov = np.cov(abc_rej_accepted_thetas.T) * SL_STEP_VARIANCE
    theta_init = np.median(abc_rej_accepted_thetas, axis=0)
    
    print(f"  n_iter={SL_ITER}, n_sim_per_eval={SL_SIM_PER_EVAL}, theta_init={theta_init}")

    sl_runner = SyntheticLikelihoodMCMC(rng=rng, prior_sampler=prior_sampler, verbose=True)

    t0 = time.time()
    sl_chain, sl_ll_chain, sl_acc_rate = sl_runner.run(
        s_obs=observed_summary, n_iter=SL_ITER, n_sim_per_eval=SL_SIM_PER_EVAL,
        theta_init=theta_init, proposal_cov=proposal_cov, subset=SummarySubset.ALL
    )
    t_sl = time.time() - t0
    print(f"Completed in {t_sl:.1f}s, acceptance rate: {sl_acc_rate:.4f}")

    sl_chain_post = sl_chain[SL_BURN_IN:]

    aggregator.add_result('Synthetic Likelihood', sl_chain_post, None, t_sl, SL_ITER * SL_SIM_PER_EVAL)

    return sl_chain, sl_chain_post, sl_acc_rate

def npe_run(rng: np.random.Generator,
            thetas: np.ndarray,
            summaries: np.ndarray,
            observed_summary: SummaryStatistic,
            prior_sampler: PriorSampler,
            aggregator: ResultAggregator):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("METHOD 6: Neural Posterior Estimation (sbi, MAF density estimator)")
    print("=" * 70)

    npe_runner = NeuralPosteriorEstimation(rng=rng, prior_sampler=prior_sampler, verbose=True)

    samples, posterior = npe_runner.run(
        thetas=thetas,
        summaries=summaries,
        s_obs=observed_summary,
        n_posterior_samples=10_000,
        density_estimator="maf",
        subset=SummarySubset.ALL
    )

    # Clip to prior bounds (NPE can sample slightly outside if density mass leaks)
    samples[:, 0] = np.clip(samples[:, 0], *PriorSampler.PRIOR_BOUNDS['beta'])
    samples[:, 1] = np.clip(samples[:, 1], *PriorSampler.PRIOR_BOUNDS['gamma'])
    samples[:, 2] = np.clip(samples[:, 2], *PriorSampler.PRIOR_BOUNDS['rho'])

    t_npe = time.time() - t0
    print(f"Completed in {t_npe:.1f}s")

    base_sims = aggregator.methods['Rejection ABC (q=1%)']['n_sims']
    aggregator.add_result('NPE (sbi, MAF)', samples, None, t_npe, base_sims)

    return samples

if __name__ == "__main__":
    print(">>> RUNNING MAIN EXPERIMENTS <<<")
    main()
    
    print("\n\n>>> RUNNING BUDGET-MATCHED EXPERIMENTS <<<")
    budget_matched.main()
    
    print("\n\n>>> RUNNING SYNTHETIC TRUTH EXPERIMENTS <<<")
    synthetic_truth.main()
    
    print("\n\n>>> GENERATING FIGURES <<<")
    make_figures.main()
    
    print("\n>>> ALL TASKS COMPLETED SUCCESSFULLY <<<")