import numpy as np
from sklearn.decomposition import FastICA
import networkx as nx
from scipy.optimize import linear_sum_assignment


# --- Helper for checking perfect matching (required by normalize_and_threshold) ---
def build_bipartite_graph(I_minus_W_thresh):
    """
    Create a bipartite graph from the thresholded ICA-recovered matrix.
    An edge (r_i, j) exists if the entry (i, j) in the matrix is non-zero.

    Parameters:
    - I_minus_W_thresh (np.ndarray): Thresholded matrix (n x n).

    Returns:
    - G (networkx.Graph): Bipartite graph.
    - row_nodes (list): Left node labels ('r0', ..., 'rn-1').
    - col_nodes (list): Right node labels ('0', ..., 'n-1').
    """
    n = I_minus_W_thresh.shape[0]
    G = nx.Graph()

    # Define row and column nodes
    row_nodes = [f"r{i}" for i in range(n)]
    col_nodes = [f"{j}" for j in range(n)]

    # Add nodes with bipartite attributes
    G.add_nodes_from(row_nodes, bipartite=0)  # Left set (rows)
    G.add_nodes_from(col_nodes, bipartite=1)  # Right set (columns)

    # Add edges where I_minus_W_thresh[i, j] ≠ 0
    for i in range(n):
        for j in range(n):
            if np.abs(I_minus_W_thresh[i, j]) > 1e-9: # Use a small tolerance
                G.add_edge(f"r{i}", f"{j}")

    return G, row_nodes, col_nodes

def has_perfect_matching(G, row_nodes, col_nodes):
    """
    Checks if a bipartite graph G has a perfect matching.
    """
    matching = nx.bipartite.maximum_matching(G, top_nodes=row_nodes)
    # A perfect matching exists if the number of matched edges is equal to the number of row nodes (n)
    # Note: len(matching) is 2*n for perfect matching dict representation (r->c, c->r)
    return len(matching) // 2 == len(row_nodes)


# --- ICA Post-processing Functions ---

def normalize_and_threshold(
    matrix,
    scale=4.0,
    rel_threshold_start=0.5,
    abs_threshold_start=0.3,
    discount_factor=0.9,
    Max_Itr=100,
    verbose=False
):
    """
    Normalize and threshold the ICA matrix adaptively until the resulting
    bipartite graph admits a perfect matching.

    Returns a matchable thresholded matrix.
    """
    rel_threshold = rel_threshold_start
    abs_threshold = abs_threshold_start

    # Loop to gradually relax the thresholds
    current_matrix = matrix.copy()
    
    for _ in range(Max_Itr):
        # Apply absolute thresholding first
        matrix_thr = current_matrix * (np.abs(current_matrix) > abs_threshold)
        
        # Normalize the rows by their L-infinity norm (max absolute value)
        # Add a small epsilon to avoid division by zero if a row is all zeros
        row_norms = np.linalg.norm(matrix_thr, ord=np.inf, axis=1, keepdims=True) + 1e-9
        normalized_matrix = scale * matrix_thr / row_norms
        
        # Apply relative thresholding
        thresholded_matrix = normalized_matrix * (np.abs(normalized_matrix) > rel_threshold)

        # Check if the bipartite graph has a perfect matching
        G, row_nodes, col_nodes = build_bipartite_graph(thresholded_matrix)

        if has_perfect_matching(G, row_nodes, col_nodes):
            if verbose:
                print(f"[INFO] Thresholding succeeded at rel={rel_threshold:.2f}, abs={abs_threshold:.2f}")
            return thresholded_matrix

        # Reduce thresholds for next attempt
        rel_threshold = rel_threshold * discount_factor
        abs_threshold = abs_threshold * discount_factor

    if verbose:
        print("[WARN] Could not find a threshold with perfect matching. Returning final attempt.")

    # Return the last attempted matrix even if it failed (though usually we ensure matching in the sim)
    return thresholded_matrix


def permute_and_scale_rows(matrix, seed=None):
    """
    Simulates the output of ICA by returning a random row-permuted and row-scaled 
    version of the true mixing matrix (M = I - W).
    
    Used in the "without ICA" simulation mode where we assume perfect ICA recovery.

    Parameters:
    - matrix (np.ndarray): Input matrix (e.g., I - W_true).
    - seed (int, optional): For reproducibility.

    Returns:
    - transformed_matrix (np.ndarray): Row-permuted and row-scaled matrix (simulated M_ICA).
    - row_permutation (np.ndarray): The applied row permutation indices.
    - scaling_factors (np.ndarray): The diagonal entries used for scaling.
    """
    if seed is not None:
        np.random.seed(seed)

    n_rows = matrix.shape[0]

    # Random row permutation (P)
    row_permutation = np.random.permutation(n_rows)
    permuted_matrix = matrix[row_permutation, :]

    # Random positive scaling factors (Lambda)
    scaling_factors = np.random.uniform(0.5, 2.0, size=n_rows)
    scaling_matrix = np.diag(scaling_factors)

    # Apply scaling (Lambda * P * M_true)
    transformed_matrix = scaling_matrix @ permuted_matrix

    return transformed_matrix, row_permutation, scaling_factors


def best_row_permutation(true_matrix, recovered_matrix, match_by='cosine'):
    """
    Permute and align rows of recovered_matrix to match true_matrix using the 
    Hungarian algorithm (minimum cost perfect matching). 
    This is necessary because ICA recovers the components up to permutation and sign.

    Parameters:
    - true_matrix (np.ndarray): Ground truth matrix (n x d).
    - recovered_matrix (np.ndarray): ICA-recovered matrix (n x d).
    - match_by (str): 'cosine' (similarity) or 'l2' (distance).

    Returns:
    - aligned_matrix (np.ndarray): Permuted and sign-corrected version of recovered_matrix.
    - permutation (list): List of matched indices (row_ind -> col_ind).
    """
    n = true_matrix.shape[0]
    cost_matrix = np.zeros((n, n))

    # Normalize rows for fair comparison
    true_normed = true_matrix / (np.linalg.norm(true_matrix, axis=1, keepdims=True) + 1e-9)
    rec_normed = recovered_matrix / (np.linalg.norm(recovered_matrix, axis=1, keepdims=True) + 1e-9)

    for i in range(n):
        for j in range(n):
            if match_by == 'cosine':
                # Maximize |cos(theta)| -> Minimize 1 - |cos(theta)|
                sim = np.abs(np.dot(true_normed[i], rec_normed[j]))
                cost_matrix[i, j] = 1 - sim
            elif match_by == 'l2':
                # Minimize min(||A_i - B_j||, ||A_i + B_j||)
                diff = np.linalg.norm(true_normed[i] - rec_normed[j])
                diff_flipped = np.linalg.norm(true_normed[i] + rec_normed[j])
                cost_matrix[i, j] = min(diff, diff_flipped)
            else:
                raise ValueError("match_by must be 'cosine' or 'l2'")

    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder and flip signs if needed
    aligned_matrix = np.zeros_like(recovered_matrix)
    for i, j in zip(row_ind, col_ind):
        rec_row = recovered_matrix[j]
        true_row = true_matrix[i]
        
        # Check sign based on dot product
        if np.dot(rec_row, true_row) < 0:
            rec_row = -rec_row
            
        aligned_matrix[i] = rec_row

    # The permutation maps true row index (i) to recovered row index (j)
    # The output col_ind gives the index j from the recovered matrix that matches 
    # the index i from the true matrix (where i is the index in row_ind).
    # Since row_ind is usually [0, 1, ..., n-1], the permutation is simply col_ind.
    return aligned_matrix, col_ind.tolist()