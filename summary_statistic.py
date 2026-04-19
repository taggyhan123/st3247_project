"""Summary Statistics objects and utilities.

This module provides the SummaryStatistic class and SummarySubset enumeration
to structure and extract the informative features of simulated trajectories.
"""
from __future__ import annotations

import numpy as np
from enum import Enum

class SummarySubset(Enum):
    """Enumeration of predefined summary statistic subsets for comparison.
    
    Attributes:
        INFECTED: Subset containing only infection-related statistics (indices 0-4).
        REWIRING: Subset containing only rewiring-related statistics (indices 5-8).
        DEGREE: Subset containing only degree-related statistics (indices 9-11).
        INFECTED_REWIRING: Subset containing both infection and rewiring statistics.
        INFECTED_DEGREE: Subset containing both infection and degree statistics.
        REWIRING_DEGREE: Subset containing both rewiring and degree statistics.
        ALL: Subset containing all 12 summary statistics.
    """
    INFECTED = [0, 1, 2, 3, 4]
    REWIRING = [5, 6, 7, 8]
    DEGREE = [9, 10, 11]
    INFECTED_REWIRING = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    INFECTED_DEGREE = [0, 1, 2, 3, 4, 9, 10, 11]
    REWIRING_DEGREE = [5, 6, 7, 8, 9, 10, 11]
    ALL = list(range(12))

class SummaryStatistic:
    """Encapsulates and computes summary statistics for a simulated dataset.

    It extracts 12 statistics relating to the infection curve, rewiring events,
    and final degree distribution.
    """
    def __init__(self, infected_fraction=None, rewire_counts=None, degree_histogram=None, precomputed_summaries=None):
        """Initializes the SummaryStatistic.

        Args:
            infected_fraction (np.ndarray | None): Array of infected fractions over time.
            rewire_counts (np.ndarray | None): Array of rewiring events over time.
            degree_histogram (np.ndarray | None): Final degree distribution array.
            precomputed_summaries (np.ndarray | None): Directly sets the summary array if provided.
        """
        if precomputed_summaries is not None:
            self.summaries = precomputed_summaries
        else:
            self.summaries = self._get_all_summaries(infected_fraction, rewire_counts, degree_histogram)

    @staticmethod
    def _get_all_summaries(infected_fraction, rewire_counts, degree_histogram):
        """Computes all summary statistics from the simulation outputs.

        Args:
            infected_fraction (np.ndarray): Array of infected fractions over time.
            rewire_counts (np.ndarray): Array of rewiring events over time.
            degree_histogram (np.ndarray): Final degree distribution array.

        Returns:
            np.ndarray: A 1D array containing all 12 summary statistics.
        """
        T = len(infected_fraction) - 1
        inf_stats = SummaryStatistic._get_infected_fraction_summaries(infected_fraction, T)
        rew_stats = SummaryStatistic._get_rewire_summaries(rewire_counts, T)
        deg_stats = SummaryStatistic._get_degree_summaries(degree_histogram)
        return np.array([*inf_stats, *rew_stats, *deg_stats])

    @staticmethod
    def _get_infected_fraction_summaries(infected_fraction, T):
        """Extracts summary statistics related to the infected fraction timeseries.

        Args:
            infected_fraction (np.ndarray): Array of infected fractions over time.
            T (int): Total number of time steps.

        Returns:
            tuple: A tuple containing 5 statistics:
                - s1 (float): Peak infected fraction.
                - s2 (float): Normalized time to peak.
                - s3 (float): Mean infected fraction (AUC).
                - s4 (float): Normalized epidemic duration.
                - s5 (float): Epidemic breadth.
        """
        s1 = np.max(infected_fraction)  # peak infected fraction
        s2 = np.argmax(infected_fraction) / T  # time to peak (normalised)
        s3 = np.mean(infected_fraction)  # mean infected fraction (AUC)

        above = np.where(infected_fraction > 0.01)[0]
        s4 = (above[-1] / T) if len(above) > 0 else 0.0  # epidemic duration (normalised)

        s5 = np.sum(infected_fraction > 0.005) / T  # epidemic breadth
        return s1, s2, s3, s4, s5
    
    @staticmethod
    def _get_rewire_summaries(rewire_counts, T):
        """Extracts summary statistics related to the rewiring counts timeseries.

        Args:
            rewire_counts (np.ndarray): Array of rewiring events over time.
            T (int): Total number of time steps.

        Returns:
            tuple: A tuple containing 4 statistics:
                - s1 (float): Total number of rewires.
                - s2 (float): Peak rewire count.
                - s3 (float): Normalized time of peak rewire.
                - s4 (float): Lag-1 autocorrelation of the rewiring timeseries.
        """
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
        """Extracts summary statistics related to the final degree distribution.

        Args:
            degree_histogram (np.ndarray): Final degree distribution array.

        Returns:
            tuple: A tuple containing 3 statistics:
                - s1 (float): Mean final degree.
                - s2 (float): Variance of final degree.
                - s3 (float): Fraction of isolated nodes (degree 0).
        """
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
        """Aggregates a list of SummaryStatistic objects into a new SummaryStatistic object.

        Args:
            stats_list: A list of SummaryStatistic instances representing replicates.
            agg: The aggregation function to apply ("median" or "mean").

        Returns:
            SummaryStatistic: A new SummaryStatistic object containing the aggregated
                summary values.
        """
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
        """Converts a list of SummaryStatistic objects into a 2D numpy array.

        Args:
            stats_list: A list of SummaryStatistic objects.
            subset: The subset of statistics to extract. Defaults to SummarySubset.ALL.

        Returns:
            np.ndarray: A 2D array of shape (len(stats_list), len(subset)).
        """
        return np.array([stat.get_summaries(subset) for stat in stats_list])
    
    def get_summaries(self, subset: SummarySubset = SummarySubset.ALL):
        """Returns the requested subset of summary statistics.
        
        Args:
            subset (SummarySubset): The subset of statistics to extract. 
                Defaults to SummarySubset.ALL.
                
        Returns:
            np.ndarray: A 1D array of the requested summary statistics.
        """
        return self.summaries[subset.value]

    def print_summary_statistics(self, subset: SummarySubset = SummarySubset.ALL):
        """Prints the values of the requested summary statistics subset.
        
        Args:
            subset (SummarySubset): The subset of statistics to print. 
                Defaults to SummarySubset.ALL.
        """
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
    """Computes summary statistics from observed data across multiple replicates.
    
    Unlike the legacy function, this does not aggregate the results. Instead, 
    it returns a list of SummaryStatistic objects, one for each replicate.

    Args:
        inf_ts: Array of shape (R, T+1) containing infected fractions for R replicates.
        rew_ts: Array of shape (R, T+1) containing rewiring counts for R replicates.
        deg_hist: Array of shape (R, 31) containing degree histograms for R replicates.

    Returns:
        list[SummaryStatistic]: A list of SummaryStatistic objects, one for each replicate.
    """
    R = inf_ts.shape[0]
    return [SummaryStatistic(inf_ts[r], rew_ts[r], deg_hist[r]) for r in range(R)]