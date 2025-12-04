import networkx as nx
import random
import copy
from collections import defaultdict

# --- Sampler and Enumerator Functions ---

def enumerate_perfect_matchings(G, row_nodes, col_nodes):
    """
    Enumerates all perfect matchings in a bipartite graph G using backtracking.
    """
    def backtrack(matching, remaining_rows, used_cols):
        if not remaining_rows:
            matchings.append(dict(matching))
            return

        r = remaining_rows[0]
        for c in G.neighbors(r):
            if c not in used_cols:
                matching.append((r, c))
                backtrack(matching, remaining_rows[1:], used_cols | {c})
                matching.pop()

    matchings = []
    backtrack([], row_nodes, set())
    return matchings if matchings else False


def greedy_min_degree_sampler(G, row_nodes, col_nodes, M, mode='exact'):
    """
    Returns M sampled or all exact perfect matchings from a bipartite graph.
    """
    if mode == 'exact':
        return enumerate_perfect_matchings(G, row_nodes, col_nodes)
    
    elif mode == 'sample':
        matchings = []
        for _ in range(M):
            G_copy = copy.deepcopy(G)
            row_set = set(row_nodes)
            col_set = set(col_nodes)
            matching = {}

            while row_set:
                try:
                    r = min(row_set, key=lambda x: G_copy.degree[x])
                except ValueError:
                    break
                
                neighbors = list(G_copy.neighbors(r))
                if not neighbors:
                    break 

                c = random.choice(neighbors)
                matching[r] = c

                G_copy.remove_node(r)
                G_copy.remove_node(c)
                row_set.remove(r)
                col_set.remove(c)

            if len(matching) == len(row_nodes):
                matchings.append(matching)

        if not matchings and len(row_nodes) > 0:
            return False 
        return matchings
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

# --- Refinement and Metric Calculation ---

def check_unique_matching_and_refine(G, row_nodes, col_nodes):
    """
    Strict Column-Only Greedy Refinement (Matches Jupyter Notebook).
    Checks if the bipartite graph admits a unique perfect matching 
    by only looking at column degrees.
    """
    matching = {}

    while col_nodes:
        # Select column node with minimum degree
        min_deg_col = min(col_nodes, key=lambda c: G.degree[c])

        # Check degree
        deg = G.degree[min_deg_col]

        if deg == 0:
            return False, matching, G, row_nodes, col_nodes  # No matching possible

        if deg > 1:
            # NOTE: Notebook returns False immediately if min-degree col > 1
            return False, matching, G, row_nodes, col_nodes  

        # Unique match (deg == 1)
        neighbor = next(iter(G.neighbors(min_deg_col))) # The only neighbor 'r'
        matching[neighbor] = min_deg_col  # r_i -> j

        # Prune matched nodes
        G.remove_node(min_deg_col)
        G.remove_node(neighbor)

        col_nodes.remove(min_deg_col)
        row_nodes.remove(neighbor)

    return True, matching, G, row_nodes, col_nodes


def edge_probs_from_matching(bipartite_graph, matchings):
    """Calculates marginal probability for each edge."""
    edge_counts = defaultdict(int)
    total_matchings = len(matchings)

    if total_matchings == 0:
        return {edge: 0.0 for edge in bipartite_graph.edges()}

    for matching in matchings:
        for r_node, c_node in matching.items():
            edge_key = tuple(sorted((r_node, c_node)))
            edge_counts[edge_key] += 1

    edge_probs = {}
    for r_node in [n for n in bipartite_graph.nodes() if n.startswith('r')]:
        for c_node in bipartite_graph.neighbors(r_node):
            edge_key = tuple(sorted((r_node, c_node)))
            prob = edge_counts.get(edge_key, 0) / total_matchings
            edge_probs[(r_node, c_node)] = prob
    
    return edge_probs


def compute_normalized_marginal_benefit(bipartite_graph, matchings, col_nodes):
    """Estimates NMB for each column vertex."""
    edge_probs_raw = edge_probs_from_matching(bipartite_graph, matchings)
    normalized_benefits = {}

    for c_node in col_nodes:
        sum_benefit = 0.0
        for r_node in bipartite_graph.neighbors(c_node):
            p_i = edge_probs_raw.get((r_node, c_node), 
                                     edge_probs_raw.get((c_node, r_node), 0.0))
            sum_benefit += p_i * (1 - p_i)
        normalized_benefits[c_node] = sum_benefit

    return normalized_benefits