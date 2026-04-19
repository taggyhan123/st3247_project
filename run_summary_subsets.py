"""
Run summary statistics subset comparison using cached rejection ABC simulations.

Recomputes distances under different summary subsets and reports posterior widths.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_rejection import BasicRejectionABC
from abc_utils import SummaryStatisticNormalizer
from summary_statistic import SummaryStatistic, SummarySubset


class SummarySubsetAnalyzer:
    """Analyzes the impact of different summary statistic subsets on the ABC posterior.

    This class handles the loading of cached simulations, evaluates normalized distances 
    across predefined combinations of summary statistics, and applies an acceptance 
    threshold. It also provides methods to display and persist the outputs.

    Attributes:
        cache_path (str): The file path to the cached rejection ABC simulations.
        thetas (np.ndarray): The parameter samples from the cache.
        summaries (np.ndarray): The summary statistics from the cache.
        mads (np.ndarray): The Median Absolute Deviations from the cache.
        s_obs (np.ndarray): The observed summary statistics from the cache.
        results (dict[str, np.ndarray]): A mapping of subset names to their respective 
            accepted parameter samples.
    """

    SUMMARY_SUBSETS = {
        'Infected only (s1-s5)': SummarySubset.INFECTED,
        'Rewiring only (s6-s9)': SummarySubset.REWIRING,
        'Degree only (s10-s12)': SummarySubset.DEGREE,
        'Infected + Rewiring':   SummarySubset.INFECTED_REWIRING,
        'Infected + Degree':     SummarySubset.INFECTED_DEGREE,
        'All 12':                SummarySubset.ALL,
    }

    def __init__(self, cache_path: str):
        """Initializes the SummarySubsetAnalyzer.

        Args:
            cache_path (str): Path to the `.npz` file containing prior simulation data.
        """
        self.cache_path = cache_path
        self.thetas = None
        self.summaries = None
        self.mads = None
        self.s_obs = None
        self.results = {}

    def load_data(self) -> None:
        """Loads cached rejection ABC simulations from the specified file path.

        Raises:
            FileNotFoundError: If the cache file cannot be found.
            KeyError: If expected data keys are missing from the cache.
        """
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(f"Cache file {self.cache_path} not found.")
            
        data = np.load(self.cache_path)
        self.thetas = data['thetas']
        self.summaries = data['summaries']
        self.mads = data['mads']
        self.s_obs = data['s_obs']

        print(f"Loaded {len(self.thetas)} simulations from {self.cache_path}")

    def evaluate_subsets(self, quantile: float = 0.01) -> None:
        """Evaluates the posterior using different subsets of summary statistics.

        Args:
            quantile (float): The acceptance quantile for Rejection ABC. 
                Defaults to 0.01.
        """
        normalizer = SummaryStatisticNormalizer([])
        normalizer.mads = self.mads
        s_obs_stat = SummaryStatistic(precomputed_summaries=self.s_obs)
        summary_stats_list = [SummaryStatistic(precomputed_summaries=s) for s in self.summaries]

        self.results = {}
        for name, subset in self.SUMMARY_SUBSETS.items():
            dists = np.array([
                normalizer.get_normalized_distance(stat, s_obs_stat, subset)
                for stat in summary_stats_list
            ])
            mask, _ = BasicRejectionABC.accept_quantile(distances=dists, quantile=quantile)
            self.results[name] = self.thetas[mask]

    def print_results(self) -> None:
        """Prints a comparison table of posterior CI widths and parameter correlations."""
        print(f"\n{'Subset':>30s}  {'CI_beta':>9s} {'CI_gamma':>10s} {'CI_rho':>8s}  "
              f"{'r(b,r)':>8s} {'r(b,g)':>8s} {'r(g,r)':>8s}")
        print('-' * 90)

        for name, t in self.results.items():
            widths = []
            for k in range(3):
                lo, hi = np.percentile(t[:, k], [2.5, 97.5])
                widths.append(hi - lo)
            r_br = np.corrcoef(t[:, 0], t[:, 2])[0, 1]
            r_bg = np.corrcoef(t[:, 0], t[:, 1])[0, 1]
            r_gr = np.corrcoef(t[:, 1], t[:, 2])[0, 1]
            print(f'{name:>30s}  {widths[0]:>9.4f} {widths[1]:>10.4f} {widths[2]:>8.4f}  '
                  f'{r_br:>8.4f} {r_bg:>8.4f} {r_gr:>8.4f}')

    def save_results(self, output_path: str) -> None:
        """Saves the accepted samples for each subset to a compressed NumPy file.

        Args:
            output_path (str): The file path where the results will be saved.
        """
        safe_names = {
            name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus'): v
            for name, v in self.results.items()
        }
        np.savez(output_path, **safe_names)
        print(f"\nSaved to {output_path}")


def main():
    """Main execution function for running the summary subset comparison."""
    analyzer = SummarySubsetAnalyzer('results/rejection_abc_50k.npz')
    analyzer.load_data()
    analyzer.evaluate_subsets(quantile=0.01)
    analyzer.print_results()
    analyzer.save_results('results/summary_subsets.npz')


if __name__ == '__main__':
    main()
