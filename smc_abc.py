import numpy as np
from summary_statistic import SummaryStatistic, SummarySubset
from abc_utils import PriorSampler, SummaryStatisticNormalizer
from simulator import simulate

class SMCABC:
    def __init__(self,
                 rng: np.random.Generator,
                 normalizer: SummaryStatisticNormalizer,
                 prior_sampler: PriorSampler,
                 verbose: bool = False):
        self.rng = rng
        self.normalizer = normalizer
        self.prior_sampler = prior_sampler
        self.verbose = verbose

    def _weighted_cov(self, particles, weights):
        """Compute weighted covariance matrix."""
        w = weights / np.sum(weights)
        mean = np.average(particles, weights=w, axis=0)
        diff = particles - mean
        cov = (diff * w[:, None]).T @ diff
        return 2.0 * cov  # factor of 2 for optimal scaling

    def run(self,
            s_obs: SummaryStatistic,
            n_particles: int,
            n_generations: int,
            alpha: float = 0.5,
            min_epsilon: float = 0.5,
            subset: SummarySubset = SummarySubset.ALL,
            n_reps_per_sim: int = 1,
            max_sims: int = None):

        all_particles = []
        epsilons = []
        total_sims = 0

        # ---- Generation 0: sample from prior ----
        if self.verbose:
            print("SMC-ABC Generation 0: sampling from prior ...")

        particles = np.zeros((n_particles, 3))
        distances = np.zeros(n_particles)
        
        # First, generate a pool of particles from prior
        pool_size = n_particles * 5
        pool_thetas = np.zeros((pool_size, 3))
        pool_dists = np.zeros(pool_size)

        p_betas, p_gammas, p_rhos = self.prior_sampler.sample(pool_size)
        for i in range(pool_size):
            theta = np.array([p_betas[i], p_gammas[i], p_rhos[i]])
            rep_stats = [SummaryStatistic(*simulate(*theta, rng=self.rng))
                         for _ in range(n_reps_per_sim)]
            stat = SummaryStatistic.aggregate_summary_statistics(rep_stats, agg="mean")
            d = self.normalizer.get_normalized_distance(stat, s_obs, subset)
            pool_thetas[i] = theta
            pool_dists[i] = d
        total_sims += pool_size * n_reps_per_sim

        # Set initial epsilon as the alpha-quantile of pool distances
        epsilon = float(np.quantile(pool_dists, alpha))
        mask = pool_dists <= epsilon
        alive = pool_thetas[mask]

        # Take up to n_particles from accepted pool
        if len(alive) >= n_particles:
            idx = self.rng.choice(len(alive), size=n_particles, replace=False)
            particles = alive[idx]
            distances[:] = pool_dists[mask][idx]
        else:
            # Need more samples — keep sampling until we have enough
            particles[:len(alive)] = alive
            distances[:len(alive)] = pool_dists[mask]
            filled = len(alive)

            while filled < n_particles:
                p_b, p_g, p_r = self.prior_sampler.sample(1)
                theta = np.array([p_b[0], p_g[0], p_r[0]])
                rep_stats = [SummaryStatistic(*simulate(*theta, rng=self.rng))
                             for _ in range(n_reps_per_sim)]
                stat = SummaryStatistic.aggregate_summary_statistics(rep_stats, agg="mean")
                d = self.normalizer.get_normalized_distance(stat, s_obs, subset)
                total_sims += n_reps_per_sim
                if d <= epsilon:
                    particles[filled] = theta
                    distances[filled] = d
                    filled += 1

        weights = np.ones(n_particles) / n_particles
        epsilons.append(epsilon)
        all_particles.append(particles.copy())

        if self.verbose:
            print(f"  ε_0 = {epsilon:.4f}, {n_particles} particles alive")

        # ---- Generations 1, 2, ... ----
        for gen in range(1, n_generations):
            if max_sims is not None and total_sims >= max_sims:
                if self.verbose:
                    print(f"  Budget exhausted ({total_sims} sims), stopping.")
                break

            # New tolerance: alpha-quantile of current distances
            epsilon_new = float(np.quantile(distances, alpha))
            if epsilon_new >= epsilon:
                epsilon_new = 0.95 * epsilon  # ensure decrease
            epsilon = epsilon_new

            if epsilon < min_epsilon:
                if self.verbose:
                    print(f"  Reached min_epsilon = {min_epsilon}, stopping.")
                break

            epsilons.append(epsilon)

            if self.verbose:
                print(f"SMC-ABC Generation {gen}: ε = {epsilon:.4f}")

            # Compute perturbation kernel covariance
            cov = self._weighted_cov(particles, weights)
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
                    idx = self.rng.choice(n_particles, p=prev_weights)
                    # Perturb
                    theta_prop = self.rng.multivariate_normal(prev_particles[idx], cov)

                    # Prior check
                    if not PriorSampler.in_prior(theta_prop):
                        continue

                    # Simulate
                    rep_stats = [SummaryStatistic(*simulate(*theta_prop, rng=self.rng)) 
                                 for _ in range(n_reps_per_sim)]
                    stat = SummaryStatistic.aggregate_summary_statistics(rep_stats, agg="mean")
                    d = self.normalizer.get_normalized_distance(stat, s_obs, subset)

                    if d <= epsilon:
                        new_particles[i] = theta_prop
                        new_distances[i] = d

                        # Weight: prior(θ) / sum_j w_j^{t-1} K(θ | θ_j^{t-1})
                        # Assuming uniform prior, prior(θ) is proportional to 1.
                        prior_val = 1.0 
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
                    idx = self.rng.choice(n_particles, p=prev_weights)
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

            total_sims += n_sim_total

            # Effective sample size
            ess = 1.0 / np.sum(weights ** 2)

            if self.verbose:
                print(f"  sims this gen: {n_sim_total}, total: {total_sims}, ESS: {ess:.1f}")

        return particles, weights, epsilons, all_particles, total_sims
