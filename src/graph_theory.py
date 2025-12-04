import networkx as nx
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
# --- Bipartite Graph Functions 

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
    return len(matching) // 2 == len(row_nodes)

# --- Causal Graph Functions ---

def generate_true_graph_from_I_minus_W(I_minus_W):
    """
    Convert I - W matrix to directed graph (G=(V, E)).
    The edges go from cause (j) to effect (i) if W[i, j] ≠ 0.
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

def minimum_feedback_vertex_set(G):
    """
    Compute the minimum feedback vertex set (FVS) using integer programming (PuLP).
    The FVS is the minimum set of nodes whose removal makes the graph acyclic. 
    Its size is a theoretical lower bound on the number of interventions required.
    """
    prob = LpProblem("MinimumFeedbackVertexSet", LpMinimize)
    node_vars = LpVariable.dicts("Node", G.nodes(), 0, 1, LpBinary)

    # Objective: minimize number of nodes in FVS
    prob += lpSum(node_vars[n] for n in G.nodes()), "MinimizeFVS"

    # Constraint: Each cycle must contain at least one node in the FVS
    for cycle in nx.simple_cycles(G):
        # For performance, only use small cycles (e.g., length <= 4)
        if len(cycle) <= 4:
             prob += lpSum(node_vars[n] for n in cycle) >= 1

    # PuLP solver will try to minimize the objective
    try:
        prob.solve(PULP_CBC_CMD(msg=False))
    except Exception as e:
        # Fallback if the solver fails or is unavailable
        print(f"PuLP solver failed. Returning full node set as FVS estimate. Error: {e}")
        return list(G.nodes())

    fvs = [n for n in G.nodes() if node_vars[n].varValue == 1]
    return fvs