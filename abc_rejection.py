"""
Rejection ABC for the adaptive-network SIR model.

Algorithm:
  1. Draw θ from prior
  2. Simulate data, compute summary statistics S(y_sim)
  3. Compute distance d(S(y_sim), S(y_obs)) using MAD-normalised Euclidean distance
  4. Accept θ if d < ε  (determined post-hoc via quantile threshold)

References:
  Pritchard et al. (1999), Beaumont et al. (2002)
"""

import numpy as np
from summary_stats import compute_summaries


# ---------------------------------------------------------------------------
# Prior bounds
# ---------------------------------------------------------------------------
PRIOR_BOUNDS = {
    "beta": (0.05, 0.50),
    "gamma": (0.02, 0.20),
    "rho": (0.00, 0.80),
}


# ---------------------------------------------------------------------------
# MAD normalisation
# ---------------------------------------------------------------------------
def compute_mads(summaries_matrix):
    """Median absolute deviation for each summary statistic.

    Parameters
    ----------
    summaries_matrix : ndarray (n, d)

    Returns
    -------
    mads : ndarray (d,)   — zeros replaced by 1.0
    """
    medians = np.median(summaries_matrix, axis=0)
    mads = np.median(np.abs(summaries_matrix - medians), axis=0)
    mads[mads == 0] = 1.0
    return mads


def normalised_distance(s_sim, s_obs, mads, indices=None):
    """MAD-normalised Euclidean distance (optionally on a subset of stats)."""
    if indices is not None:
        s_sim = s_sim[indices]
        s_obs = s_obs[indices]
        mads = mads[indices]
    diff = (s_sim - s_obs) / mads
    return np.sqrt(np.sum(diff ** 2))


# ---------------------------------------------------------------------------
# Main rejection ABC loop
# ---------------------------------------------------------------------------
def run_rejection_abc(
    s_obs,
    n_sim,
    simulator_fn,
    summary_fn=None,
    mads=None,
    indices=None,
    n_reps_per_sim=1,
    seed=0,
    verbose=True,
):
    """Run rejection ABC.

    Parameters
    ----------
    s_obs : ndarray (d,)        — observed summary statistics
    n_sim : int                 — number of parameter draws
    simulator_fn : callable     — f(beta, gamma, rho, seed=int) -> (inf, rew, deg)
    summary_fn : callable       — f(inf, rew, deg) -> summaries.  Default: compute_summaries
    mads : ndarray or None      — normalisation constants; computed from pilot if None
    indices : list[int] or None — subset of summary indices to use
    n_reps_per_sim : int        — replicates per parameter draw (averaged)
    seed : int
    verbose : bool

    Returns
    -------
    thetas : ndarray (n_sim, 3)
    distances : ndarray (n_sim,)
    summaries : ndarray (n_sim, d)
    mads : ndarray (d,)
    """
    if summary_fn is None:
        summary_fn = compute_summaries

    rng = np.random.default_rng(seed)
    n_stats = len(s_obs)

    # Sample from priors
    betas = rng.uniform(*PRIOR_BOUNDS["beta"], size=n_sim)
    gammas = rng.uniform(*PRIOR_BOUNDS["gamma"], size=n_sim)
    rhos = rng.uniform(*PRIOR_BOUNDS["rho"], size=n_sim)
    thetas = np.column_stack([betas, gammas, rhos])

    summaries = np.zeros((n_sim, n_stats))
    distances = np.zeros(n_sim)

    # Pilot run for MAD if not provided
    if mads is None:
        n_pilot = 2000
        if verbose:
            print(f"Running pilot ({n_pilot} sims) for MAD normalisation ...")
        pilot_s = np.zeros((n_pilot, n_stats))
        for i in range(n_pilot):
            b = rng.uniform(*PRIOR_BOUNDS["beta"])
            g = rng.uniform(*PRIOR_BOUNDS["gamma"])
            r = rng.uniform(*PRIOR_BOUNDS["rho"])
            inf, rew, deg = simulator_fn(b, g, r, seed=seed + n_sim + i)
            pilot_s[i] = summary_fn(inf, rew, deg)
        mads = compute_mads(pilot_s)
        if verbose:
            print("MAD constants:", np.round(mads, 4))

    # Main loop
    for i in range(n_sim):
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_sim}")

        if n_reps_per_sim == 1:
            inf, rew, deg = simulator_fn(betas[i], gammas[i], rhos[i], seed=seed + i)
            summaries[i] = summary_fn(inf, rew, deg)
        else:
            acc = np.zeros(n_stats)
            for rep in range(n_reps_per_sim):
                inf, rew, deg = simulator_fn(
                    betas[i], gammas[i], rhos[i],
                    seed=seed + i * n_reps_per_sim + rep,
                )
                acc += summary_fn(inf, rew, deg)
            summaries[i] = acc / n_reps_per_sim

        distances[i] = normalised_distance(summaries[i], s_obs, mads, indices)

    return thetas, distances, summaries, mads


# ---------------------------------------------------------------------------
# Acceptance by quantile
# ---------------------------------------------------------------------------
def accept_quantile(thetas, distances, summaries, quantile=0.01):
    """Accept the top `quantile` fraction of samples.

    Returns
    -------
    accepted_thetas, accepted_distances, accepted_summaries, threshold
    """
    threshold = np.quantile(distances, quantile)
    mask = distances <= threshold
    return thetas[mask], distances[mask], summaries[mask], threshold
