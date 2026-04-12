from __future__ import annotations

import numpy as np
from enum import Enum

class SummarySubset(Enum):
    INFECTED = [0, 1, 2, 3, 4]
    REWIRING = [5, 6, 7, 8]
    DEGREE = [9, 10, 11]
    INFECTED_REWIRING = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    INFECTED_DEGREE = [0, 1, 2, 3, 4, 9, 10, 11]
    REWIRING_DEGREE = [5, 6, 7, 8, 9, 10, 11]
    ALL = list(range(12))

class SummaryStatistic:
    def __init__(self, infected_fraction=None, rewire_counts=None, degree_histogram=None, precomputed_summaries=None):
        if precomputed_summaries is not None:
            self.summaries = precomputed_summaries
        else:
            self.summaries = self._get_all_summaries(infected_fraction, rewire_counts, degree_histogram)

    @staticmethod
    def _get_all_summaries(infected_fraction, rewire_counts, degree_histogram):
        T = len(infected_fraction) - 1
        inf_stats = SummaryStatistic._get_infected_fraction_summaries(infected_fraction, T)
        rew_stats = SummaryStatistic._get_rewire_summaries(rewire_counts, T)
        deg_stats = SummaryStatistic._get_degree_summaries(degree_histogram)
        return np.array([*inf_stats, *rew_stats, *deg_stats])

    @staticmethod
    def _get_infected_fraction_summaries(infected_fraction, T):
        s1 = np.max(infected_fraction)  # peak infected fraction
        s2 = np.argmax(infected_fraction) / T  # time to peak (normalised)
        s3 = np.mean(infected_fraction)  # mean infected fraction (AUC)

        above = np.where(infected_fraction > 0.01)[0]
        s4 = (above[-1] / T) if len(above) > 0 else 0.0  # epidemic duration (normalised)

        s5 = np.sum(infected_fraction > 0.005) / T  # epidemic breadth
        return s1, s2, s3, s4, s5
    
    @staticmethod
    def _get_rewire_summaries(rewire_counts, T):
        # --- Rewiring timeseries ---
        s1 = np.sum(rewire_counts)  # total rewires
        s2 = np.max(rewire_counts)  # peak rewire count
        s3 = np.argmax(rewire_counts) / T  # time of peak rewire (normalised)

        rew = rewire_counts.astype(np.float64)
        if np.std(rew) > 0:
            c = rew - np.mean(rew)
            s4 = np.sum(c[:-1] * c[1:]) / np.sum(c ** 2)  # lag-1 autocorrelation
        else:
            s4 = 0.0
        return s1, s2, s3, s4

    @staticmethod
    def _get_degree_summaries(degree_histogram):
        # --- Degree histogram ---
        degrees = np.arange(31)
        total = np.sum(degree_histogram)
        if total > 0:
            probs = degree_histogram / total
            s1 = np.sum(degrees * probs)  # mean degree
            s2 = np.sum(degrees ** 2 * probs) - s1 ** 2  # variance of degree
            s3 = degree_histogram[0] / total  # fraction isolated
        else:
            s1, s2, s3 = 0.0, 0.0, 0.0
        return s1, s2, s3
    
    @classmethod
    def aggregate_summary_statistics(cls, stats_list: list[SummaryStatistic], agg: str = "median") -> SummaryStatistic:
        """Aggregates a list of SummaryStatistic objects into a new SummaryStatistic object."""
        # Aggregate ALL subsets so the resulting object has the full 12 stats
        s_per_rep = cls.convert_list_to_ndarray(stats_list, SummarySubset.ALL)
        
        if agg == "median":
            agg_summaries = np.median(s_per_rep, axis=0)
        elif agg == "mean":
            agg_summaries = np.mean(s_per_rep, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")
            
        return cls(precomputed_summaries=agg_summaries)
        
    @classmethod
    def convert_list_to_ndarray(cls, stats_list: list[SummaryStatistic],
                                subset: SummarySubset = SummarySubset.ALL) -> np.ndarray:
        """Converts a list of SummaryStatistic objects into a 2D numpy array."""
        return np.array([stat.get_summaries(subset) for stat in stats_list])
    
    def get_summaries(self, subset: SummarySubset = SummarySubset.ALL):
        return self.summaries[subset.value]

    def print_summary_statistics(self, subset: SummarySubset = SummarySubset.ALL):
        all_names = [
            "Peak infected frac", "Time to peak (norm)", "Mean infected frac",
            "Epidemic duration (norm)", "Epidemic breadth", "Total rewires",
            "Peak rewire count", "Time of peak rewire (norm)", "Rewire autocorr lag-1",
            "Mean final degree", "Var final degree", "Frac isolated nodes",
        ]
            
        names = [all_names[i] for i in subset.value]
        for name, val in zip(names, self.get_summaries(subset)):
            print(f"{name:<28}: {val:.4f}")

def compute_observed_summaries(
    inf_ts: np.ndarray,
    rew_ts: np.ndarray,
    deg_hist: np.ndarray
) -> list[SummaryStatistic]:
    """
    Compute summary statistics from observed data across multiple replicates.
    
    Unlike the legacy function, this does not aggregate the results. Instead, 
    it returns a list of SummaryStatistic objects, one for each replicate.

    Parameters
    ----------
    inf_ts : np.ndarray
        Array of shape (R, T+1) containing infected fractions for R replicates.
    rew_ts : np.ndarray
        Array of shape (R, T+1) containing rewiring counts for R replicates.
    deg_hist : np.ndarray
        Array of shape (R, 31) containing degree histograms for R replicates.

    Returns
    -------
    list[SummaryStatistic]
        A list of SummaryStatistic objects, one for each of the R replicates.
    """
    R = inf_ts.shape[0]
    return [SummaryStatistic(inf_ts[r], rew_ts[r], deg_hist[r]) for r in range(R)]