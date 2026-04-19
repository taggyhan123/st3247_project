"""
Posterior predictive check: simulate from NPE posterior samples and overlay
on observed data for all three observables (infected trajectory, rewiring
trajectory, degree histogram).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from data_loader import load_all
from simulator import simulate, simulate_fast


RNG_SEED = 42
N_DRAWS = 200


def sample_posterior_predictive(samples: np.ndarray, n_draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate from posterior parameter draws."""
    idx = rng.choice(len(samples), size=n_draws, replace=False)
    drawn = samples[idx]

    pred_inf = np.zeros((n_draws, 201))
    pred_rew = np.zeros((n_draws, 201))
    pred_deg = np.zeros((n_draws, 31))

    for i, theta in enumerate(drawn):
        inf, rew, deg = simulate(*theta, rng=rng)
        pred_inf[i] = inf
        pred_rew[i] = rew
        pred_deg[i] = deg

    return pred_inf, pred_rew, pred_deg


def plot_posterior_predictive(obs_inf: np.ndarray, obs_rew: np.ndarray, obs_deg: np.ndarray,
                              pred_inf: np.ndarray, pred_rew: np.ndarray, pred_deg: np.ndarray) -> None:
    """Three-panel figure: observed data + posterior predictive envelope."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    t = np.arange(201)

    # --- Panel 1: Infected trajectory ---
    ax = axes[0]
    for j in range(obs_inf.shape[0]):
        ax.plot(t, obs_inf[j], color='gray', alpha=0.25, linewidth=0.5)
    med = np.median(pred_inf, axis=0)
    lo = np.percentile(pred_inf, 5, axis=0)
    hi = np.percentile(pred_inf, 95, axis=0)
    ax.fill_between(t, lo, hi, color='tab:blue', alpha=0.25, label='Pred. 90% band')
    ax.plot(t, med, color='tab:blue', linewidth=1.5, label='Pred. median')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Infected fraction')
    ax.set_title('Infected trajectory')
    ax.legend(fontsize=8)

    # --- Panel 2: Rewiring trajectory ---
    ax = axes[1]
    for j in range(obs_rew.shape[0]):
        ax.plot(t, obs_rew[j], color='gray', alpha=0.25, linewidth=0.5)
    med = np.median(pred_rew, axis=0)
    lo = np.percentile(pred_rew, 5, axis=0)
    hi = np.percentile(pred_rew, 95, axis=0)
    ax.fill_between(t, lo, hi, color='tab:orange', alpha=0.25, label='Pred. 90% band')
    ax.plot(t, med, color='tab:orange', linewidth=1.5, label='Pred. median')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Rewiring events')
    ax.set_title('Rewiring trajectory')
    ax.legend(fontsize=8)

    # --- Panel 3: Degree histogram ---
    ax = axes[2]
    bins = np.arange(31)
    obs_mean = np.mean(obs_deg, axis=0)
    pred_mean = np.mean(pred_deg, axis=0)
    pred_lo = np.percentile(pred_deg, 5, axis=0)
    pred_hi = np.percentile(pred_deg, 95, axis=0)

    width = 0.35
    ax.bar(bins - width/2, obs_mean, width, color='gray', alpha=0.6, label='Observed mean')
    ax.bar(bins + width/2, pred_mean, width, color='tab:green', alpha=0.6, label='Pred. mean')
    yerr_lo = np.maximum(pred_mean - pred_lo, 0)
    yerr_hi = np.maximum(pred_hi - pred_mean, 0)
    ax.errorbar(bins + width/2, pred_mean,
                yerr=[yerr_lo, yerr_hi],
                fmt='none', ecolor='tab:green', alpha=0.4, capsize=1.5)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Fraction of nodes')
    ax.set_title('Final degree distribution')
    ax.set_xlim(-0.5, 20.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/posterior_predictive.png', dpi=200, bbox_inches='tight')
    print("Saved figures/posterior_predictive.png")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Warm up Numba
    _ = simulate_fast(0.2, 0.1, 0.3, seed=0)

    # Load observed data
    inf_ts, rew_ts, deg_hist = load_all()
    print(f"Observed data: {inf_ts.shape[0]} replicates")

    # Load NPE posterior samples
    npe_path = 'results/npe.npz'
    if not os.path.exists(npe_path):
        raise FileNotFoundError(f"{npe_path} not found. Run main.py first.")
    npe_samples = np.load(npe_path)['samples']
    print(f"NPE posterior: {npe_samples.shape[0]} samples")

    # Generate posterior predictive draws
    cache_path = 'results/posterior_predictive.npz'
    if os.path.exists(cache_path):
        print(f"Loading cached posterior predictive from {cache_path}")
        data = np.load(cache_path)
        pred_inf, pred_rew, pred_deg = data['pred_inf'], data['pred_rew'], data['pred_deg']
    else:
        print(f"Simulating {N_DRAWS} posterior predictive draws ...")
        pred_inf, pred_rew, pred_deg = sample_posterior_predictive(npe_samples, N_DRAWS, rng)
        np.savez(cache_path, pred_inf=pred_inf, pred_rew=pred_rew, pred_deg=pred_deg)
        print(f"Cached to {cache_path}")

    # Plot
    plot_posterior_predictive(inf_ts, rew_ts, deg_hist,
                              pred_inf, pred_rew, pred_deg)


if __name__ == '__main__':
    main()
