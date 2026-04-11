import numpy as np
from summary_statistics_oop import SummaryStatistic, SummarySubset
from st3247_project.abc_utils import PriorSampler, SummaryStatisticNormalizer

from simulator import simulate

class BasicRejectionABC:
    def __init__(self,
                 rng: np.random.Generator,
                 normalizer: SummaryStatisticNormalizer,
                 prior_sampler: PriorSampler,
                 verbose = False):
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
    def accept_quantile(distances, quantile=0.01, verbose=False) -> tuple[np.ndarray, float]:
        """
        Accept the top `quantile` fraction of samples based on their distances.
        """
        threshold = np.quantile(distances, quantile)
        mask = distances <= threshold
        
        if verbose:
            print(f"Accepted {np.sum(mask)} samples (q={quantile*100:.1f}%), threshold={threshold:.3f}")
            
        return mask, threshold