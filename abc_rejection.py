"""Basic Rejection Approximate Bayesian Computation (ABC).

This module implements the standard rejection ABC algorithm, drawing samples
from the prior and accepting them based on a distance threshold.
"""

import numpy as np
from summary_statistic import SummaryStatistic, SummarySubset
from abc_utils import PriorSampler, SummaryStatisticNormalizer
from simulator import simulate

class BasicRejectionABC:
    """Basic Rejection Approximate Bayesian Computation (ABC).

    This class implements the standard rejection ABC algorithm, which draws
    parameters from a prior, simulates data, and accepts the parameters if the
    distance between simulated and observed summaries is below a given threshold.
    """

    def __init__(self,
                 rng: np.random.Generator,
                 normalizer: SummaryStatisticNormalizer,
                 prior_sampler: PriorSampler,
                 verbose: bool = False):
        """Initializes the BasicRejectionABC runner.

        Args:
            rng: A NumPy random number generator instance.
            normalizer: A normalizer for computing distances between summary
                statistics.
            prior_sampler: An object to sample from the prior distribution.
            verbose: If True, prints progress information during the run.
        """
        self.rng = rng
        self.normalizer = normalizer
        self.prior_sampler = prior_sampler
        self.verbose = verbose

    def run(self,
            s_obs: SummaryStatistic,
            n_sim: int,
            subset: SummarySubset,
            acceptance_quantile: float = 0.01,
            n_reps_per_sim: int = 1):
        """Runs the rejection ABC algorithm.

        Args:
            s_obs: The observed summary statistics.
            n_sim: The total number of prior simulations to run.
            subset: The subset of summary statistics to use for distance
                computation.
            acceptance_quantile: The fraction of top samples to accept based on
                distance. Defaults to 0.01.
            n_reps_per_sim: The number of simulation replicates to run for each
                parameter evaluation. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - thetas (np.ndarray): All proposed parameter samples.
                - distances (np.ndarray): Distances for all proposed samples.
                - summaries (np.ndarray): Summary statistics for all proposed samples.
                - acceptance_mask (np.ndarray): Boolean mask indicating accepted samples.
                - acceptance_threshold (float): The computed distance threshold used.
        """
        
        # 1. Draw from prior
        betas, gammas, rhos = self.prior_sampler.sample(n_samples=n_sim)
        thetas = np.column_stack([betas, gammas, rhos])
        
        distances = np.zeros(n_sim)
        summaries = np.zeros((n_sim, 12))

        for sim in range(n_sim):
            if self.verbose and (sim + 1) % 5000 == 0:
                print(f"  {sim+1}/{n_sim}")
            b, g, r = betas[sim], gammas[sim], rhos[sim]
            
            # 2. Simulate data with simulate function
            # 3. Compute summary statistics
            rep_stats = [SummaryStatistic(*simulate(b, g, r, rng=self.rng)) 
                         for _ in range(n_reps_per_sim)]
            stat = SummaryStatistic.aggregate_summary_statistics(rep_stats, agg="mean")
            
            # 4. Compute distances
            distances[sim] = self.normalizer.get_normalized_distance(stat, s_obs, subset)
            summaries[sim] = stat.get_summaries(SummarySubset.ALL)
            
        # 5. Find accepted samples
        acceptance_mask, acceptance_threshold = self.accept_quantile(distances=distances,
                                                                     quantile=acceptance_quantile,
                                                                     verbose=self.verbose)
        
        return thetas, distances, summaries, acceptance_mask, acceptance_threshold

    @staticmethod
    def accept_quantile(distances: np.ndarray, quantile: float = 0.01, verbose: bool = False) -> tuple[np.ndarray, float]:
        """Accepts the top `quantile` fraction of samples based on their distances.

        Args:
            distances: An array of computed distances.
            quantile: The fraction of samples to accept. Defaults to 0.01.
            verbose: If True, prints the number of accepted samples and the threshold.

        Returns:
            tuple: A tuple containing:
                - mask (np.ndarray): A boolean mask of accepted samples.
                - threshold (float): The computed distance threshold.
        """
        threshold = np.quantile(distances, quantile)
        mask = distances <= threshold
        
        if verbose:
            print(f"Accepted {np.sum(mask)} samples (q={quantile*100:.1f}%), threshold={threshold:.3f}")
            
        return mask, threshold