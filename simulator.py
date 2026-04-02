"""
Numba-optimized adaptive-network SIR epidemic simulator.

Simulates an SIR epidemic on a dynamic contact network where susceptible
individuals can rewire away from infected neighbours (adaptive behaviour).

Update rules per time step (synchronous within each phase):
  Phase 1 - Infection:  collect all new S->I transmissions, then apply at once
  Phase 2 - Recovery:   each infected node (incl. newly infected) recovers w.p. gamma
  Phase 3 - Rewiring:   sequential processing of S-I edges; stale-edge checks

Reference: Gross et al. (2006), Phys. Rev. Lett. 96, 208701.
"""

import numpy as np
from numba import njit


@njit
def _has_edge(adj, deg, i, j):
    for k in range(deg[i]):
        if adj[i, k] == j:
            return True
    return False


@njit
def _add_edge(adj, deg, i, j):
    adj[i, deg[i]] = j
    deg[i] += 1
    adj[j, deg[j]] = i
    deg[j] += 1


@njit
def _remove_edge(adj, deg, i, j):
    for k in range(deg[i]):
        if adj[i, k] == j:
            deg[i] -= 1
            adj[i, k] = adj[i, deg[i]]
            break
    for k in range(deg[j]):
        if adj[j, k] == i:
            deg[j] -= 1
            adj[j, k] = adj[j, deg[j]]
            break


@njit
def simulate_fast(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, seed=None):
    """Run one replicate of the adaptive-network SIR model.

    Parameters
    ----------
    beta : float   — transmission probability per S-I edge per step
    gamma : float  — recovery probability per infected node per step
    rho : float    — rewiring probability per S-I edge per step
    N : int        — population size
    p_edge : float — Erdos-Renyi edge probability
    n_infected0 : int — initial number of infected
    T : int        — number of time steps
    seed : int or None

    Returns
    -------
    infected_fraction : ndarray (T+1,)
    rewire_counts : ndarray (T+1,)
    degree_histogram : ndarray (31,)
    """
    if seed is not None:
        np.random.seed(seed)

    max_deg = N - 1
    adj = np.zeros((N, max_deg), dtype=np.int32)
    deg = np.zeros(N, dtype=np.int32)

    # Build Erdos-Renyi graph
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                _add_edge(adj, deg, i, j)

    # Initialise states: 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=np.int8)
    indices = np.arange(N)
    for i in range(n_infected0):
        j = i + int(np.random.random() * (N - i))
        indices[i], indices[j] = indices[j], indices[i]
    for i in range(n_infected0):
        state[indices[i]] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)

    count_infected = 0
    for i in range(N):
        if state[i] == 1:
            count_infected += 1
    infected_fraction[0] = count_infected / N

    new_infections = np.zeros(N, dtype=np.int8)
    si_edges_s = np.zeros(N * max_deg, dtype=np.int32)
    si_edges_i = np.zeros(N * max_deg, dtype=np.int32)

    for t in range(1, T + 1):
        # === PHASE 1: INFECTION (synchronous) ===
        for i in range(N):
            new_infections[i] = 0

        for i in range(N):
            if state[i] == 1:
                for k in range(deg[i]):
                    j = adj[i, k]
                    if state[j] == 0:
                        if np.random.random() < beta:
                            new_infections[j] = 1

        for j in range(N):
            if new_infections[j] == 1:
                state[j] = 1

        # === PHASE 2: RECOVERY ===
        for i in range(N):
            if state[i] == 1:
                if np.random.random() < gamma:
                    state[i] = 2

        # === PHASE 3: REWIRING (sequential) ===
        n_si = 0
        for i in range(N):
            if state[i] == 0:
                for k in range(deg[i]):
                    j = adj[i, k]
                    if state[j] == 1:
                        si_edges_s[n_si] = i
                        si_edges_i[n_si] = j
                        n_si += 1

        rewire_count = 0
        for idx in range(n_si):
            if np.random.random() < rho:
                s_node = si_edges_s[idx]
                i_node = si_edges_i[idx]

                if not _has_edge(adj, deg, s_node, i_node):
                    continue

                _remove_edge(adj, deg, s_node, i_node)

                n_candidates = 0
                for k in range(N):
                    if k != s_node and not _has_edge(adj, deg, s_node, k):
                        n_candidates += 1

                if n_candidates > 0:
                    choice = int(np.random.random() * n_candidates)
                    count = 0
                    new_partner = -1
                    for k in range(N):
                        if k != s_node and not _has_edge(adj, deg, s_node, k):
                            if count == choice:
                                new_partner = k
                                break
                            count += 1

                    if new_partner >= 0:
                        _add_edge(adj, deg, s_node, new_partner)
                        rewire_count += 1

        count_infected = 0
        for i in range(N):
            if state[i] == 1:
                count_infected += 1
        infected_fraction[t] = count_infected / N
        rewire_counts[t] = rewire_count

    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        d = deg[i]
        if d >= 30:
            d = 30
        degree_histogram[d] += 1

    return infected_fraction, rewire_counts, degree_histogram


def simulate(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    """Wrapper with numpy Generator interface."""
    if rng is not None:
        seed = int(rng.integers(0, 2**31))
    else:
        seed = int(np.random.default_rng().integers(0, 2**31))
    return simulate_fast(beta, gamma, rho, N, p_edge, n_infected0, T, seed)
