import numpy as np
import networkx as nx
# Ensure FastICA is imported if you're using this module for the full pipeline, 
# although ICA-specific functions are better placed in 'ica_processing.py'.

def generate_exogenous_samples(n, D, distributions=None, seed=None):
    """
    Generates n non-Gaussian source signals (exogenous noises) of length D.
    
    Parameters:
    - n (int): Number of sources.
    - D (int): Number of samples (data points).
    - distributions (list, optional): List of distributions to cycle through.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - samples (np.ndarray): D x n matrix of source samples.
    - used_distributions (list): List of distribution names used for each source.
    """
    if seed is not None:
        np.random.seed(seed)

    # List of non-Gaussian distributions/signals
    if distributions is None:
        distributions = ['sine', 'square', 'sawtooth', 'laplace', 'bernoulli',
                         'exponential', 'uniform', 'student_t']

    time = np.linspace(0, 10, D)
    samples = np.zeros((D, n))
    used_distributions = []

    for i in range(n):
        dist = distributions[i % len(distributions)]
        used_distributions.append(dist)
        
        # Signal-based distributions
        if dist == 'sine':
            freq = np.random.uniform(0.5, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            samples[:, i] = np.sin(2 * np.pi * freq * time + phase)
        elif dist == 'square':
            freq = np.random.uniform(0.5, 4)
            samples[:, i] = np.sign(np.sin(2 * np.pi * freq * time))
        elif dist == 'sawtooth':
            freq = np.random.uniform(0.5, 4)
            samples[:, i] = 2 * (time * freq % 1) - 1
        
        # Statistical distributions
        elif dist == 'laplace':
            samples[:, i] = np.random.laplace(loc=0.0, scale=1.0, size=D)
        elif dist == 'bernoulli':
            samples[:, i] = np.random.choice([-1, 1], size=D)
        elif dist == 'exponential':
            # Mean-center the exponential distribution
            raw = np.random.exponential(scale=1.0, size=D)
            samples[:, i] = raw - np.mean(raw)
        elif dist == 'uniform':
            # Centered uniform
            samples[:, i] = np.random.uniform(low=-1.0, high=1.0, size=D)
        elif dist == 'student_t':
            samples[:, i] = np.random.standard_t(df=3, size=D)
        else:
            samples[:, i] = np.random.standard_normal(size=D) # Fallback

    return samples, used_distributions


def generate_sparse_mixing_matrix(n, zero_prob=0.6, weight_scale=4.0, min_abs=0.6, seed=None):
    """
    Generates the mixing matrix M = (I - W) where W is the sparse causal matrix.
    Ensures non-zero off-diagonals if possible.
    
    Parameters:
    - n (int): Number of sources/variables.
    - zero_prob (float): Probability of an off-diagonal entry being zero (for sparsity).
    - weight_scale (float): Maximum absolute value for non-zero entries.
    - min_abs (float): Minimum absolute value for non-zero entries.
    - seed (int, optional): Random seed.

    Returns:
    - I_minus_W (np.ndarray): The mixing matrix (I - W).
    """
    if seed is not None:
        np.random.seed(seed)

    W = np.zeros((n, n))
    nonzero_count = 0

    for i in range(n):
        for j in range(n):
            if i != j and np.random.rand() > zero_prob:
                # Draw until abs(value) >= min_abs
                while True:
                    val = np.random.uniform(-weight_scale, weight_scale)
                    if abs(val) >= min_abs:
                        W[i, j] = val
                        nonzero_count += 1
                        break

    # Guarantee at least one non-zero off-diagonal for non-trivial problem
    if nonzero_count == 0 and n > 1:
        i, j = np.random.choice(n, 2, replace=False)
        val = np.random.uniform(-weight_scale, weight_scale)
        if abs(val) >= min_abs:
            W[i, j] = val
        elif (np.random.rand() >= 0.5):
            W[i, j] = min_abs
        else:
            W[i, j] = -min_abs
    
    # Check for non-stationarity condition |det(I-W)| > 0.
    # For large sparse graphs, non-stationarity is usually not an issue with small W entries,
    # but for simplicity, we directly return M = I - W.
    I_minus_W = np.eye(n) - W
    return I_minus_W


def mix_sources(I_minus_W, sources):
    """
    Mixes the exogenous sources to create the observed data X for an LSCM:
    X = (I - W)^{-1} * S.
    
    In ICA, the recovered mixing matrix B satisfies X = S @ B.T.
    If S = X * (I - W), then M = (I - W) is the matrix we try to recover via ICA.
    X = S * (I-W)^{-1}
    
    Parameters:
    - I_minus_W (np.ndarray): The mixing matrix (I - W).
    - sources (np.ndarray): D x n matrix of source samples.

    Returns:
    - X_mixed (np.ndarray): D x n matrix of observed mixed data.
    """
    # X = S @ M_inv.T where M = I - W and M_inv = (I - W)^-1
    return sources @ np.linalg.inv(I_minus_W.T)

# Example graph generation for FVS (requires networkx)
def generate_true_graph_from_I_minus_W(I_minus_W):
    """
    Convert I - W matrix to directed graph (G=(V, E)).
    The edges go from cause (j) to effect (i) if W[i, j] ≠ 0.
    Since M = I - W, an edge j -> i exists if M[i, j] is non-zero (and i != j).
    """
    n = I_minus_W.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    W = np.eye(n) - I_minus_W  # Recover W from I - W

    for i in range(n):
        for j in range(n):
            if i != j and np.abs(W[i, j]) > 1e-6:
                # W[i, j] is the effect of Xj on Xi, so edge is j → i
                G.add_edge(j, i)

    return G