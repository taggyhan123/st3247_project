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
    """Check if an edge exists between two nodes.

    Args:
        adj (numpy.ndarray): 2D array where adj[i] contains the neighbors of node i.
        deg (numpy.ndarray): 1D array containing the current degree of each node.
        i (int): Index of the first node.
        j (int): Index of the second node.

    Returns:
        bool: True if node j is a neighbor of node i, False otherwise.
    """
    for k in range(deg[i]):
        if adj[i, k] == j:
            return True
    return False


@njit
def _add_edge(adj, deg, i, j):
    """Add an undirected edge between two nodes.

    Modifies the adjacency and degree arrays in-place.

    Args:
        adj (numpy.ndarray): 2D array representing the adjacency list.
        deg (numpy.ndarray): 1D array representing the degree of each node.
        i (int): Index of the first node.
        j (int): Index of the second node.
    """
    adj[i, deg[i]] = j
    deg[i] += 1
    adj[j, deg[j]] = i
    deg[j] += 1


@njit
def _remove_edge(adj, deg, i, j):
    """Remove an undirected edge between two nodes.

    Modifies the adjacency and degree arrays in-place.

    Args:
        adj (numpy.ndarray): 2D array representing the adjacency list.
        deg (numpy.ndarray): 1D array representing the degree of each node.
        i (int): Index of the first node.
        j (int): Index of the second node.
    """
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
def simulate_fast(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, seed=0):
    """Run one replicate of the adaptive-network SIR model.

    Args:
        beta: float in [0, 1]
            Transmission probability. At each time step, each S-I edge
            transmits the infection independently with probability beta.
            Higher beta means the disease spreads faster.
        gamma: float in [0, 1]
            Recovery probability. At each time step, each infected node
            recovers independently with probability gamma.
            Higher gamma means shorter infectious period (on average 1/gamma steps).
        rho: float in [0, 1]
            Rewiring probability. At each time step, each S-I edge is
            rewired independently with probability rho. The susceptible
            node drops the link to its infected neighbor and connects to
            a randomly chosen new node instead.
            Higher rho means more active social distancing behavior.
        N: int, default=200
            Number of nodes (individuals) in the network.
        p_edge: float, default=0.05
            Probability of an edge between any two nodes in the initial
            Erdos-Renyi random graph. Expected initial degree is (N-1)*p_edge.
            With N=200 and p_edge=0.05, the expected degree is about 10.
        n_infected0: int, default=5
            Number of nodes infected at time t=0. These are chosen
            uniformly at random (without replacement) from all N nodes.
        T: int, default=200
            Number of discrete time steps to simulate.
        seed: int or None
            Seed for reproducibility of the random number generator.

    Returns:
        tuple: A tuple containing:
            - infected_fraction (np.ndarray): Fraction of the population that is
              infected at each time step, from t=0 to t=T. Values are in [0, 1]. Shape (T+1,).
            - rewire_counts (np.ndarray): Number of successful rewiring events at each
              time step. Always 0 at t=0. Shape (T+1,).
            - degree_histogram (np.ndarray): Histogram of node degrees at the final
              time step t=T. degree_histogram[d] = number of nodes with degree d, for
              d=0..29. degree_histogram[30] counts nodes with degree >= 30. Shape (31,).
    """
    np.random.seed(seed)

    # --- Graph Representation ---
    # To achieve maximum performance with Numba, we avoid dynamic Python lists
    # or dynamically resizing NumPy arrays. Instead, we pre-allocate a static 
    # 2D adjacency list `adj` of size (N, max_deg) and a 1D array `deg` tracking 
    # the current degree of each node.
    # 
    # `adj[i, k]` stores the node index of the k-th neighbor of node i.
    # We only read up to `adj[i, deg[i] - 1]`. When an edge is removed, we swap 
    # the last valid neighbor into the removed spot to maintain a dense array.
    # 
    # Helper functions `_has_edge`, `_add_edge`, and `_remove_edge` take 
    # `adj` and `deg` as arguments to abstract away this underlying array 
    # manipulation, making the main simulation logic cleaner.
    max_deg = N - 1
    adj = np.zeros((N, max_deg), dtype=np.int32)
    deg = np.zeros(N, dtype=np.int32)

    # Build Erdos-Renyi graph: iterate over all possible node pairs (i, j)
    # and add an edge independently with probability p_edge.
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                _add_edge(adj, deg, i, j)

    # Initialize the health state of each node.
    #
    # We encode states as integers:
    #   0 = Susceptible (S): can catch the disease
    #   1 = Infected (I):    currently infectious
    #   2 = Recovered (R):   immune, cannot be infected again
    #
    # At t=0, we pick n_infected0 nodes uniformly at random to be infected.
    # All other nodes start as susceptible.
    state = np.zeros(N, dtype=np.int8)
    indices = np.arange(N)
    # Use a partial Fisher-Yates shuffle to select distinct nodes without replacement
    for i in range(n_infected0):
        j = i + int(np.random.random() * (N - i))
        indices[i], indices[j] = indices[j], indices[i]
    for i in range(n_infected0):
        state[indices[i]] = 1

    # Arrays to track summary statistics over time.
    # `infected_fraction` tracks the proportion of infected nodes at each step.
    # `rewire_counts` tracks the number of successful rewiring events per step.
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
        # PHASE 1: INFECTION (synchronous update)
        #
        # For every infected node i, look at each of its neighbors j.
        # If j is susceptible (state 0), the infection transmits with
        # probability beta.
        #
        # Important: we use synchronous updating. We first
        # collect ALL new infections in a set, then apply them all at
        # once. This prevents "chain infections" within a single step
        # (where a newly infected node immediately infects its own
        # neighbors in the same step).
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

        # PHASE 2: RECOVERY
        #
        # Each currently infected node (including those just infected
        # in Phase 1) recovers independently with probability gamma.
        # Recovery is permanent: recovered nodes move to state 2 (R)
        # and can never be infected again.
        for i in range(N):
            if state[i] == 1:
                if np.random.random() < gamma:
                    state[i] = 2

        # PHASE 3: NETWORK REWIRING (adaptive behavior)
        #
        # This is what makes the model "adaptive": the network structure
        # changes in response to the disease.
        #
        # We look at all edges between a susceptible node (S) and an
        # infected node (I), called "S-I edges". For each such edge,
        # with probability rho, the susceptible node:
        #   1. Drops the connection to its infected neighbor
        #   2. Forms a new connection to a randomly chosen other node
        #      (that it is not already connected to)
        #
        # This models social distancing: susceptible individuals
        # actively avoid infected contacts.

        # Step 1: Snapshot all current S-I edges.
        # We store them in pre-allocated 1D arrays to avoid modifying the network
        # while we are iterating over it, which could cause concurrency issues.
        n_si = 0
        for i in range(N):
            if state[i] == 0:
                for k in range(deg[i]):
                    j = adj[i, k]
                    if state[j] == 1:
                        si_edges_s[n_si] = i
                        si_edges_i[n_si] = j
                        n_si += 1

        # Step 2: Iterate through the snapshot and probabilistically rewire.
        rewire_count = 0
        for idx in range(n_si):
            if np.random.random() < rho:
                s_node = si_edges_s[idx]
                i_node = si_edges_i[idx]

                # Check if the edge still exists (it might have been removed by
                # an earlier rewiring event in this same time step).
                if not _has_edge(adj, deg, s_node, i_node):
                    continue

                # 1. Drop the connection to the infected neighbor.
                _remove_edge(adj, deg, s_node, i_node)

                # 2. Find all eligible new partners (not self, not already connected).
                n_candidates = 0
                for k in range(N):
                    if k != s_node and not _has_edge(adj, deg, s_node, k):
                        n_candidates += 1

                # 3. Choose a new partner uniformly at random and connect.
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

    # Compute the degree histogram at the final time step.
    #
    # The degree of a node is its number of connections (neighbors).
    # We bin degrees from 0 to 29 individually, and lump all degrees >= 30
    # into a single bin (index 30). This gives a fixed-size output array
    # of shape (31,) regardless of the actual degree distribution.

    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        d = deg[i]
        if d >= 30:
            d = 30
        degree_histogram[d] += 1

    return infected_fraction, rewire_counts, degree_histogram


def simulate(
    beta: float, 
    gamma: float, 
    rho: float,
    rng: np.random.Generator, 
    N: int = 200, 
    p_edge: float = 0.05, 
    n_infected0: int = 5, 
    T: int = 200, 
):
    """Wrapper for the adaptive-network SIR model simulator using the numpy Generator interface.

    Args:
        beta: Transmission probability.
        gamma: Recovery probability.
        rho: Rewiring probability.
        rng: A NumPy random number generator instance.
        N: Number of nodes. Defaults to 200.
        p_edge: Initial edge probability. Defaults to 0.05.
        n_infected0: Initial number of infected nodes. Defaults to 5.
        T: Number of time steps. Defaults to 200.

    Returns:
        tuple: The infected fraction, rewiring counts, and degree histogram arrays.
    """
    seed = int(rng.integers(0, 2**31))
    return simulate_fast(beta, gamma, rho, N, p_edge, n_infected0, T, seed)
