"""
Summary statistics for the adaptive-network SIR model.

Computes a vector of 12 statistics from simulator outputs, grouped as:
  s1-s5  : infected timeseries features
  s6-s9  : rewiring timeseries features
  s10-s12: final degree distribution features
"""

import numpy as np


def compute_summaries(infected_fraction, rewire_counts, degree_histogram):
    """Compute 12 summary statistics from a single replicate.

    Parameters
    ----------
    infected_fraction : ndarray (T+1,)
    rewire_counts : ndarray (T+1,)
    degree_histogram : ndarray (31,)

    Returns
    -------
    summaries : ndarray (12,)
    """
    T = len(infected_fraction) - 1

    # --- Infected timeseries ---
    s1 = np.max(infected_fraction)                          # peak infected fraction
    s2 = np.argmax(infected_fraction) / T                   # time to peak (normalised)
    s3 = np.mean(infected_fraction)                         # mean infected fraction (AUC)

    above = np.where(infected_fraction > 0.01)[0]
    s4 = (above[-1] / T) if len(above) > 0 else 0.0        # epidemic duration (normalised)

    s5 = np.sum(infected_fraction > 0.005) / T              # epidemic breadth

    # --- Rewiring timeseries ---
    s6 = np.sum(rewire_counts)                              # total rewires
    s7 = np.max(rewire_counts)                              # peak rewire count
    s8 = np.argmax(rewire_counts) / T                       # time of peak rewire (normalised)

    rew = rewire_counts.astype(np.float64)
    if np.std(rew) > 0:
        c = rew - np.mean(rew)
        s9 = np.sum(c[:-1] * c[1:]) / np.sum(c ** 2)       # lag-1 autocorrelation
    else:
        s9 = 0.0

    # --- Degree histogram ---
    degrees = np.arange(31)
    total = np.sum(degree_histogram)
    if total > 0:
        probs = degree_histogram / total
        s10 = np.sum(degrees * probs)                       # mean degree
        s11 = np.sum(degrees ** 2 * probs) - s10 ** 2       # variance of degree
        s12 = degree_histogram[0] / total                   # fraction isolated
    else:
        s10, s11, s12 = 0.0, 0.0, 0.0

    return np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12])


SUMMARY_NAMES = [
    "Peak infected frac",
    "Time to peak (norm)",
    "Mean infected frac",
    "Epidemic duration (norm)",
    "Epidemic breadth",
    "Total rewires",
    "Peak rewire count",
    "Time of peak rewire (norm)",
    "Rewire autocorr lag-1",
    "Mean final degree",
    "Var final degree",
    "Frac isolated nodes",
]

# Index groups for ablation
IDX_INFECTED = [0, 1, 2, 3, 4]
IDX_REWIRING = [5, 6, 7, 8]
IDX_DEGREE = [9, 10, 11]
IDX_ALL = list(range(12))


def compute_observed_summaries(inf_ts, rew_ts, deg_hist, agg="median"):
    """Compute summary statistics from observed data (multiple replicates).

    Parameters
    ----------
    inf_ts : ndarray (R, T+1)
    rew_ts : ndarray (R, T+1)
    deg_hist : ndarray (R, 31)
    agg : "median" or "mean"

    Returns
    -------
    s_obs : ndarray (12,)
    s_per_rep : ndarray (R, 12)
    """
    R = inf_ts.shape[0]
    s_per_rep = np.zeros((R, 12))
    for r in range(R):
        s_per_rep[r] = compute_summaries(inf_ts[r], rew_ts[r], deg_hist[r])

    if agg == "median":
        s_obs = np.median(s_per_rep, axis=0)
    else:
        s_obs = np.mean(s_per_rep, axis=0)

    return s_obs, s_per_rep
