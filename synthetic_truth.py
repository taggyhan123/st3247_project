"""
Synthetic-truth recovery experiment: simulate observed data from known
parameters, run Rejection ABC and NPE, check whether 95% credible
intervals cover the true values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from simulator import simulate, simulate_fast
from summary_statistic import compute_observed_summaries, SummaryStatistic, SummarySubset
from abc_rejection import BasicRejectionABC
from npe import NeuralPosteriorEstimation
from abc_utils import PriorSampler, run_pilot


RNG_SEED = 1234
THETA_TRUE = np.array([0.17, 0.085, 0.30])
PARAM_NAMES = [r'$\beta$', r'$\gamma$', r'$\rho$']
N_REPLICATES = 40
N_SIM = 50_000
ACCEPTANCE_QUANTILE = 0.01


def generate_synthetic_observed(theta_true: np.ndarray, n_replicates: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate n_replicates datasets from theta_true."""
    inf_list, rew_list, deg_list = [], [], []
    for _ in range(n_replicates):
        inf, rew, deg = simulate(*theta_true, rng=rng)
        inf_list.append(inf)
        rew_list.append(rew)
        deg_list.append(deg)
    return np.array(inf_list), np.array(rew_list), np.array(deg_list)


def check_coverage(samples: np.ndarray, theta_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (covered: bool[3], ci_lo: float[3], ci_hi: float[3])."""
    covered = []
    ci_lo, ci_hi = [], []
    for k in range(3):
        lo, hi = np.percentile(samples[:, k], [2.5, 97.5])
        ci_lo.append(lo)
        ci_hi.append(hi)
        covered.append(lo <= theta_true[k] <= hi)
    return np.array(covered), np.array(ci_lo), np.array(ci_hi)


def plot_recovery(rej_samples: np.ndarray, npe_samples: np.ndarray, theta_true: np.ndarray, rej_coverage: np.ndarray, npe_coverage: np.ndarray) -> None:
    """1x3 figure: marginal posteriors with true value marked."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for k, ax in enumerate(axes):
        ax.hist(rej_samples[:, k], bins=40, density=True, alpha=0.5,
                color='tab:blue', label='Rejection ABC')
        ax.hist(npe_samples[:, k], bins=40, density=True, alpha=0.5,
                color='tab:purple', label='NPE')
        ax.axvline(theta_true[k], color='red', linestyle='--', linewidth=2,
                   label=f'True = {theta_true[k]}')

        status_rej = 'covered' if rej_coverage[k] else 'MISSED'
        status_npe = 'covered' if npe_coverage[k] else 'MISSED'
        ax.set_title(f'{PARAM_NAMES[k]}  (Rej: {status_rej}, NPE: {status_npe})')
        ax.set_xlabel(PARAM_NAMES[k])
        ax.set_ylabel('Density')
        if k == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/synthetic_truth_recovery.png', dpi=200, bbox_inches='tight')
    print("Saved figures/synthetic_truth_recovery.png")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Warm up Numba
    _ = simulate_fast(0.2, 0.1, 0.3, seed=0)

    cache_path = 'results/synthetic_truth.npz'
    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        data = np.load(cache_path)
        rej_accepted = data['rej_accepted']
        npe_samples = data['npe_samples']
    else:
        # 1. Generate synthetic observed data
        print(f"Generating {N_REPLICATES} synthetic replicates from theta_true = {THETA_TRUE} ...")
        inf_ts, rew_ts, deg_hist = generate_synthetic_observed(THETA_TRUE, N_REPLICATES, rng)

        # 2. Compute observed summaries
        obs_summaries = compute_observed_summaries(inf_ts, rew_ts, deg_hist)
        s_obs = SummaryStatistic.aggregate_summary_statistics(obs_summaries, agg="median")
        print(f"Synthetic observed summaries: {np.round(s_obs.get_summaries(), 3)}")

        # 3. Pilot for normaliser
        prior_sampler = PriorSampler(rng)
        print(f"Running pilot ({N_PILOT} sims) for MAD normalisation ...")
        normalizer = run_pilot(prior_sampler, rng)

        # 4. Rejection ABC
        print(f"\nRunning Rejection ABC ({N_SIM:,} sims, q={ACCEPTANCE_QUANTILE}) on synthetic data ...")
        t0 = time.time()
        abc_runner = BasicRejectionABC(rng=rng, normalizer=normalizer,
                                       prior_sampler=prior_sampler, verbose=True)
        thetas, distances, summaries, acc_mask, threshold = abc_runner.run(
            s_obs=s_obs, n_sim=N_SIM, subset=SummarySubset.ALL,
            acceptance_quantile=ACCEPTANCE_QUANTILE
        )
        rej_accepted = thetas[acc_mask]
        print(f"Rejection ABC done in {time.time()-t0:.1f}s, {rej_accepted.shape[0]} accepted")

        # 5. NPE
        print(f"\nTraining NPE on {N_SIM:,} simulations ...")
        t0 = time.time()
        npe_runner = NeuralPosteriorEstimation(rng=rng, prior_sampler=prior_sampler, verbose=True)
        npe_samples, _ = npe_runner.run(
            thetas=thetas, summaries=summaries, s_obs=s_obs,
            n_posterior_samples=10_000, density_estimator="maf", subset=SummarySubset.ALL
        )
        # Clip to prior bounds
        PriorSampler.clip_to_prior(npe_samples)
        print(f"NPE done in {time.time()-t0:.1f}s")

        # Cache
        np.savez(cache_path, theta_true=THETA_TRUE,
                 rej_accepted=rej_accepted, npe_samples=npe_samples)
        print(f"Cached to {cache_path}")

    # 6. Coverage check
    rej_cov, rej_lo, rej_hi = check_coverage(rej_accepted, THETA_TRUE)
    npe_cov, npe_lo, npe_hi = check_coverage(npe_samples, THETA_TRUE)

    print("\n" + "=" * 60)
    print("SYNTHETIC-TRUTH RECOVERY RESULTS")
    print("=" * 60)
    print(f"True parameters: beta={THETA_TRUE[0]}, gamma={THETA_TRUE[1]}, rho={THETA_TRUE[2]}")
    print(f"\n{'Param':>8s}  {'True':>6s}  {'Rej 95% CI':>18s} {'Covered':>8s}  {'NPE 95% CI':>18s} {'Covered':>8s}")
    print('-' * 75)
    for k, name in enumerate(['beta', 'gamma', 'rho']):
        print(f'{name:>8s}  {THETA_TRUE[k]:>6.3f}  '
              f'[{rej_lo[k]:.4f}, {rej_hi[k]:.4f}] {"YES" if rej_cov[k] else "NO":>8s}  '
              f'[{npe_lo[k]:.4f}, {npe_hi[k]:.4f}] {"YES" if npe_cov[k] else "NO":>8s}')

    # 7. Figure
    plot_recovery(rej_accepted, npe_samples, THETA_TRUE, rej_cov, npe_cov)


if __name__ == '__main__':
    main()
