"""
SMC-ABC  (Sequential Monte Carlo ABC).

Runs ABC with a sequence of decreasing tolerance thresholds, using a
population of weighted particles that are resampled and perturbed at each
generation.  This is more efficient than rejection ABC for reaching
small tolerances.

Algorithm (adaptive variant based on Beaumont et al. 2009):
  Generation 0:
    - Sample particles from the prior, simulate, compute distances
    - Set ε_0 = median of distances
    - Accept those with d < ε_0, set uniform weights
  Generation t > 0:
    - Set ε_t = α-quantile of previous distances (α ~ 0.5 for adaptive)
    - Resample particles with replacement (weighted)
    - Perturb each particle with Gaussian kernel (covariance from alive particles)
    - Simulate and accept if d < ε_t
    - Update weights: w_i ∝ π(θ_i) / Σ_j w_j^{t-1} · K(θ_i | θ_j^{t-1})

References:
  Sisson, Fan & Tanaka (2007). "Sequential Monte Carlo without likelihoods."
  Beaumont et al. (2009). "Adaptive approximate Bayesian computation."
  Del Moral, Doucet & Jasra (2012). "An adaptive sequential Monte Carlo method."
"""

import numpy as np
from summary_stats import compute_summaries
from abc_rejection import PRIOR_BOUNDS, normalised_distance


def _prior_pdf(theta):
    """Evaluate uniform prior density (unnormalised — just check support)."""
    bounds = [PRIOR_BOUNDS["beta"], PRIOR_BOUNDS["gamma"], PRIOR_BOUNDS["rho"]]
    for val, (lo, hi) in zip(theta, bounds):
        if val < lo or val > hi:
            return 0.0
    # Uniform: density = product of 1/(hi-lo)
    density = 1.0
    for lo, hi in bounds:
        density *= 1.0 / (hi - lo)
    return density


def _weighted_cov(particles, weights):
    """Compute weighted covariance matrix."""
    w = weights / np.sum(weights)
    mean = np.average(particles, weights=w, axis=0)
    diff = particles - mean
    cov = (diff * w[:, None]).T @ diff
    return 2.0 * cov  # factor of 2 for optimal scaling


def run_smc_abc(
    s_obs,
    mads,
    n_particles,
    n_generations,
    simulator_fn,
    summary_fn=None,
    indices=None,
    alpha=0.5,
    min_epsilon=0.5,
    seed=0,
    verbose=True,
):
    """Run SMC-ABC with adaptive tolerance schedule.

    Parameters
    ----------
    s_obs : ndarray (d,)
    mads : ndarray (d,)
    n_particles : int           — number of particles per generation
    n_generations : int         — maximum number of generations
    simulator_fn : callable
    summary_fn : callable or None
    indices : list[int] or None
    alpha : float               — quantile for adaptive tolerance (0.5 = median)
    min_epsilon : float         — stop if tolerance drops below this
    seed : int
    verbose : bool

    Returns
    -------
    particles : ndarray (n_particles, 3)  — final particle population
    weights : ndarray (n_particles,)      — final normalised weights
    epsilons : list[float]                — tolerance at each generation
    all_particles : list[ndarray]         — particles at each generation
    """
    if summary_fn is None:
        summary_fn = compute_summaries

    rng = np.random.default_rng(seed)
    bounds = [PRIOR_BOUNDS["beta"], PRIOR_BOUNDS["gamma"], PRIOR_BOUNDS["rho"]]

    all_particles = []
    epsilons = []

    # ---- Generation 0: sample from prior ----
    if verbose:
        print("SMC-ABC Generation 0: sampling from prior ...")

    particles = np.zeros((n_particles, 3))
    distances = np.zeros(n_particles)
    summaries_arr = np.zeros((n_particles, len(s_obs)))

    # We need to over-sample from prior since some won't meet threshold
    # For generation 0, accept everything and then set threshold
    n_sampled = 0
    max_attempts = n_particles * 50

    # First, generate a pool of particles from prior
    pool_thetas = []
    pool_dists = []
    pool_size = n_particles * 5  # generate more to pick from

    for i in range(pool_size):
        theta = np.array([
            rng.uniform(*bounds[0]),
            rng.uniform(*bounds[1]),
            rng.uniform(*bounds[2]),
        ])
        inf, rew, deg = simulator_fn(*theta, seed=int(rng.integers(0, 2**31)))
        s = summary_fn(inf, rew, deg)
        d = normalised_distance(s, s_obs, mads, indices)
        pool_thetas.append(theta)
        pool_dists.append(d)

    pool_thetas = np.array(pool_thetas)
    pool_dists = np.array(pool_dists)

    # Set initial epsilon as the alpha-quantile of pool distances
    epsilon = np.quantile(pool_dists, alpha)
    mask = pool_dists <= epsilon
    alive = pool_thetas[mask]

    # Take up to n_particles from accepted pool
    if len(alive) >= n_particles:
        idx = rng.choice(len(alive), size=n_particles, replace=False)
        particles = alive[idx]
        distances[:] = pool_dists[mask][idx]
    else:
        # Need more samples — keep sampling until we have enough
        particles[:len(alive)] = alive
        distances[:len(alive)] = pool_dists[mask]
        filled = len(alive)

        while filled < n_particles:
            theta = np.array([
                rng.uniform(*bounds[0]),
                rng.uniform(*bounds[1]),
                rng.uniform(*bounds[2]),
            ])
            inf, rew, deg = simulator_fn(*theta, seed=int(rng.integers(0, 2**31)))
            s = summary_fn(inf, rew, deg)
            d = normalised_distance(s, s_obs, mads, indices)
            if d <= epsilon:
                particles[filled] = theta
                distances[filled] = d
                filled += 1

    weights = np.ones(n_particles) / n_particles
    epsilons.append(epsilon)
    all_particles.append(particles.copy())

    if verbose:
        print(f"  ε_0 = {epsilon:.4f}, {n_particles} particles alive")

    # ---- Generations 1, 2, ... ----
    for gen in range(1, n_generations):
        # New tolerance: alpha-quantile of current distances
        epsilon_new = np.quantile(distances, alpha)
        if epsilon_new >= epsilon:
            epsilon_new = 0.95 * epsilon  # ensure decrease
        epsilon = epsilon_new

        if epsilon < min_epsilon:
            if verbose:
                print(f"  Reached min_epsilon = {min_epsilon}, stopping.")
            break

        epsilons.append(epsilon)

        if verbose:
            print(f"SMC-ABC Generation {gen}: ε = {epsilon:.4f}")

        # Compute perturbation kernel covariance
        cov = _weighted_cov(particles, weights)
        # Add small regularisation
        cov += 1e-8 * np.eye(3)

        prev_particles = particles.copy()
        prev_weights = weights.copy()

        new_particles = np.zeros((n_particles, 3))
        new_distances = np.zeros(n_particles)
        new_weights = np.zeros(n_particles)

        n_sim_total = 0
        for i in range(n_particles):
            accepted = False
            attempts = 0
            while not accepted and attempts < 1000:
                attempts += 1
                n_sim_total += 1

                # Resample: pick a particle from previous generation
                idx = rng.choice(n_particles, p=prev_weights)
                # Perturb
                theta_prop = rng.multivariate_normal(prev_particles[idx], cov)

                # Prior check
                in_prior = True
                for val, (lo, hi) in zip(theta_prop, bounds):
                    if val < lo or val > hi:
                        in_prior = False
                        break
                if not in_prior:
                    continue

                # Simulate
                inf, rew, deg = simulator_fn(*theta_prop, seed=int(rng.integers(0, 2**31)))
                s = summary_fn(inf, rew, deg)
                d = normalised_distance(s, s_obs, mads, indices)

                if d <= epsilon:
                    new_particles[i] = theta_prop
                    new_distances[i] = d

                    # Weight: prior(θ) / sum_j w_j^{t-1} K(θ | θ_j^{t-1})
                    prior_val = _prior_pdf(theta_prop)
                    kernel_sum = 0.0
                    for j in range(n_particles):
                        diff = theta_prop - prev_particles[j]
                        # Gaussian kernel density
                        exponent = -0.5 * diff @ np.linalg.solve(cov, diff)
                        kernel_sum += prev_weights[j] * np.exp(exponent)

                    new_weights[i] = prior_val / max(kernel_sum, 1e-300)
                    accepted = True

            if not accepted:
                # Fallback: keep previous particle
                idx = rng.choice(n_particles, p=prev_weights)
                new_particles[i] = prev_particles[idx]
                new_distances[i] = distances[idx]
                new_weights[i] = prev_weights[idx]

        # Normalise weights
        w_sum = np.sum(new_weights)
        if w_sum > 0:
            new_weights /= w_sum
        else:
            new_weights = np.ones(n_particles) / n_particles

        particles = new_particles
        distances = new_distances
        weights = new_weights
        all_particles.append(particles.copy())

        # Effective sample size
        ess = 1.0 / np.sum(weights ** 2)

        if verbose:
            print(f"  sims this gen: {n_sim_total}, ESS: {ess:.1f}")

    return particles, weights, epsilons, all_particles
