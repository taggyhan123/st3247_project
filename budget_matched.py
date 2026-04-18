"""
Budget-matched comparison: run Rejection ABC, SMC-ABC, and NPE each
using exactly 50,000 simulator calls to isolate statistical efficiency
from compute-budget effects.

Rejection ABC and NPE reuse the same 50,000 simulations from the main
rejection ABC run (results/rejection_abc_50k.npz). SMC-ABC runs its own
simulations since it is an iterative algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from simulator import simulate_fast
from summary_statistic import (
    SummaryStatistic,
    SummarySubset
)
from smc_abc import SMCABC
from npe import NeuralPosteriorEstimation
from abc_utils import PriorSampler, SummaryStatisticNormalizer


BUDGET = 50_000
PARAM_NAMES = [r'$\beta$', r'$\gamma$', r'$\rho$']


def print_results(results):
    """Print comparison table."""
    print(f"\n{'Method':>25s}  {'beta_med':>9s} {'gamma_med':>10s} {'rho_med':>8s}  "
          f"{'beta_95w':>9s} {'gamma_95w':>10s} {'rho_95w':>8s}  {'N_sims':>8s}")
    print('-' * 100)
    for name, info in results.items():
        s = info['samples']
        meds = [np.median(s[:, k]) for k in range(3)]
        widths = []
        for k in range(3):
            lo, hi = np.percentile(s[:, k], [2.5, 97.5])
            widths.append(hi - lo)
        print(f'{name:>25s}  {meds[0]:>9.4f} {meds[1]:>10.4f} {meds[2]:>8.4f}  '
              f'{widths[0]:>9.4f} {widths[1]:>10.4f} {widths[2]:>8.4f}  {info["n_sims"]:>8,}')


def plot_budget_matched(results):
    """1x3 marginal histograms for budget-matched methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'Rejection ABC': 'tab:blue', 'SMC-ABC': 'tab:orange', 'NPE': 'tab:purple'}

    for k, ax in enumerate(axes):
        for name, info in results.items():
            s = info['samples'][:, k]
            ax.hist(s, bins=40, density=True, alpha=0.4,
                    color=colors[name], label=name)
        ax.set_xlabel(PARAM_NAMES[k])
        ax.set_ylabel('Density')
        ax.set_title(PARAM_NAMES[k])
        if k == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f'Budget-matched comparison ({BUDGET:,} simulator calls each)', y=1.02)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/budget_matched.png', dpi=200, bbox_inches='tight')
    print("Saved figures/budget_matched.png")
    plt.close(fig)


def main():
    rng = np.random.default_rng(42)

    # Warm up Numba
    _ = simulate_fast(0.2, 0.1, 0.3, seed=0)

    # Load the main rejection ABC 50k simulations
    rej_path = 'results/rejection_abc_50k.npz'
    print(f"Loading main rejection ABC results from {rej_path}...")
    data = np.load(rej_path)
    thetas = data['thetas']
    distances = data['distances']
    summaries = data['summaries']
    mads = data['mads']
    s_obs_array = data['s_obs']

    # Reconstruct normalizer and s_obs from saved data
    normalizer = SummaryStatisticNormalizer.__new__(SummaryStatisticNormalizer)
    normalizer.mads = mads
    s_obs = SummaryStatistic(precomputed_summaries=s_obs_array)

    prior_sampler = PriorSampler(rng)

    results = {}
    cache_path = 'results/budget_matched.npz'

    if os.path.exists(cache_path):
        print(f"Loading cached budget-matched results from {cache_path}")
        cached = np.load(cache_path)
        results['Rejection ABC'] = {'samples': cached['rej_samples'], 'n_sims': BUDGET}
        results['SMC-ABC'] = {'samples': cached['smc_samples'], 'n_sims': int(cached['smc_n_sims'])}
        results['NPE'] = {'samples': cached['npe_samples'], 'n_sims': BUDGET}
    else:
        # 1. Rejection ABC — reuse the same 50k sims, apply q=1% acceptance
        print(f"\n--- Rejection ABC (reusing {BUDGET:,} sims from main run) ---")
        thr = float(np.quantile(distances, 0.01))
        acc_mask = distances <= thr
        rej_samples = thetas[acc_mask]
        print(f"  {rej_samples.shape[0]} accepted (threshold={thr:.3f})")
        results['Rejection ABC'] = {'samples': rej_samples, 'n_sims': BUDGET}

        # 2. SMC-ABC with budget cap (must run its own sims)
        print(f"\n--- SMC-ABC (budget={BUDGET:,}) ---")
        t0 = time.time()
        smc_runner = SMCABC(rng=rng, normalizer=normalizer,
                            prior_sampler=prior_sampler, verbose=True)
        smc_particles, smc_weights, smc_eps, _, smc_n_sims = smc_runner.run(
            s_obs=s_obs, n_particles=1000, n_generations=50,
            alpha=0.5, min_epsilon=0.5, subset=SummarySubset.ALL,
            max_sims=BUDGET
        )
        print(f"  Done in {time.time()-t0:.1f}s, {smc_n_sims:,} total sims, "
              f"{len(smc_eps)} generations")
        results['SMC-ABC'] = {'samples': smc_particles, 'n_sims': smc_n_sims}

        # 3. NPE — train on the same 50k sims from the main run
        print(f"\n--- NPE (reusing {BUDGET:,} sims from main run) ---")
        t0 = time.time()
        npe_runner = NeuralPosteriorEstimation(rng=rng, prior_sampler=prior_sampler, verbose=True)
        npe_samples, _ = npe_runner.run(
            thetas=thetas, summaries=summaries, s_obs=s_obs,
            n_posterior_samples=10_000, density_estimator="maf", subset=SummarySubset.ALL
        )
        npe_samples[:, 0] = np.clip(npe_samples[:, 0], *PriorSampler.PRIOR_BOUNDS['beta'])
        npe_samples[:, 1] = np.clip(npe_samples[:, 1], *PriorSampler.PRIOR_BOUNDS['gamma'])
        npe_samples[:, 2] = np.clip(npe_samples[:, 2], *PriorSampler.PRIOR_BOUNDS['rho'])
        print(f"  Done in {time.time()-t0:.1f}s")
        results['NPE'] = {'samples': npe_samples, 'n_sims': BUDGET}

        # Cache
        np.savez(cache_path,
                 rej_samples=rej_samples,
                 smc_samples=smc_particles, smc_n_sims=smc_n_sims,
                 npe_samples=npe_samples)
        print(f"\nCached to {cache_path}")

    # Print and plot
    print(f"\n{'='*60}")
    print(f"BUDGET-MATCHED COMPARISON ({BUDGET:,} simulator calls)")
    print(f"{'='*60}")
    print_results(results)
    plot_budget_matched(results)


if __name__ == '__main__':
    main()
