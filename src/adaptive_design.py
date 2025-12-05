import numpy as np
import networkx as nx
import random
import copy
from src.data_generation import generate_exogenous_samples, mix_sources, generate_sparse_mixing_matrix, generate_true_graph_from_I_minus_W
from src.ica_processing import normalize_and_threshold, permute_and_scale_rows, best_row_permutation
from src.graph_theory import build_bipartite_graph, minimum_feedback_vertex_set
from src.matching_samplers import check_unique_matching_and_refine, greedy_min_degree_sampler, compute_normalized_marginal_benefit

# --- Helper Functions ---

def select_variable_to_intervene(normalized_benefits):
    return int(max(normalized_benefits, key=normalized_benefits.get))


def perform_intervention_with_ICA(j, n_sources, n_samples, I_minus_W_true, used_distributions):
    from sklearn.decomposition import FastICA 
    I_minus_W_j = I_minus_W_true.copy()
    I_minus_W_j[j, :] = 0  
    I_minus_W_j[j, j] = 1  

    new_used_distributions = used_distributions.copy()
    new_used_distributions[j] = 'laplace' 

    new_sources, _ = generate_exogenous_samples(
        n_sources, n_samples, distributions=new_used_distributions, seed=None
    )

    X_intervention = mix_sources(I_minus_W_j, new_sources)
    ica = FastICA(n_components=n_sources, max_iter=2000, tol=1e-5, random_state=0)
    ica.fit(X_intervention)
    I_minus_W_j_ica_raw = ica.components_

    aligned_ica, _ = best_row_permutation(I_minus_W_j, I_minus_W_j_ica_raw, match_by='cosine')
    I_minus_W_j_thresh = normalize_and_threshold(aligned_ica)
    return I_minus_W_j_thresh


def perform_intervention_without_ICA(j, I_minus_W_true):
    I_minus_W_j = I_minus_W_true.copy()
    I_minus_W_j[j, :] = 0  
    I_minus_W_j[j, j] = 1  
    I_minus_W_j_ica, _, _ = permute_and_scale_rows(I_minus_W_j, seed=None)
    return I_minus_W_j_ica


def rank_candidate_rows_for_intervened_column(I_minus_W_thresh_obs, I_minus_W_j_ica, j):
    n_rows = I_minus_W_j_ica.shape[0]
    I_norm_obs = I_minus_W_thresh_obs / (np.linalg.norm(I_minus_W_thresh_obs, axis=1, keepdims=True) + 1e-9)
    J_norm_ica = I_minus_W_j_ica / (np.linalg.norm(I_minus_W_j_ica, axis=1, keepdims=True) + 1e-9)
    j_row_idx = int(np.argmax(np.abs(J_norm_ica[:, j])))

    row_scores = []
    for i in range(I_norm_obs.shape[0]):
        if np.abs(I_minus_W_thresh_obs[i, j]) < 1e-6:
            continue
        total_similarity = 0
        for k in range(n_rows):
            if k == j_row_idx: continue
            sim = np.abs(np.dot(I_norm_obs[i], J_norm_ica[k]))
            total_similarity += sim
        row_scores.append((i, total_similarity))

    sorted_candidates = sorted(row_scores, key=lambda x: x[1])
    return [(i, j) for i, _ in sorted_candidates]


def remove_edge(bipartite_graph, row_nodes, col_nodes, edge):
    r_i_idx, j_idx = edge
    r_node = f"r{r_i_idx}" 
    c_node = f"{j_idx}"    

    if r_node in bipartite_graph.nodes():
        bipartite_graph.remove_node(r_node)
        if r_node in row_nodes: row_nodes.remove(r_node)

    if c_node in bipartite_graph.nodes():
        bipartite_graph.remove_node(c_node)
        if c_node in col_nodes: col_nodes.remove(c_node)

    return bipartite_graph, row_nodes, col_nodes


def evaluate_final_estimate(I_minus_W_thresh, I_minus_W_true, perfect_matching):
    n = I_minus_W_thresh.shape[0]
    A_est = np.zeros_like(I_minus_W_thresh)
    if len(perfect_matching) != n: return np.inf, None, None
         
    for r_node, c_node in perfect_matching.items():
        i = int(r_node[1:]) 
        j = int(c_node)      
        A_est[j] = I_minus_W_thresh[i]

    diag_entries = A_est[np.arange(n), np.arange(n)]
    if np.any(np.abs(diag_entries) < 1e-9): return np.inf, None, None
         
    A_est_normalized = A_est / diag_entries[:, np.newaxis]
    estimated_W = np.eye(n) - A_est_normalized
    true_W = np.eye(n) - I_minus_W_true

    error = np.linalg.norm(estimated_W - true_W, ord='fro')
    true_norm = np.linalg.norm(true_W, ord='fro')
    
    if true_norm < 1e-9: relative_error = 0.0 if error < 1e-9 else np.inf
    else: relative_error = error / true_norm

    return relative_error, estimated_W, true_W


# --- Main Adaptive Experiment Loops ---

def perform_adaptive_experiments(observational_matrix, I_minus_W_true, n_sources, n_samples, used_distributions, K, M_samples, strategy_type='marginalized benefit', mode='exact', use_ica_in_intervention=True):
    """
    Core loop matching Notebook logic.
    """
    bipartite_graph, row_nodes, col_nodes = build_bipartite_graph(observational_matrix)
    experiment_count = 0
    perfect_matching = {}
    success = True

    # Loop exactly K times (Budget)
    for t in range(K):
        # 1. Passive Refinement
        is_unique, partial_matching, bipartite_graph, row_nodes, col_nodes = check_unique_matching_and_refine(
            bipartite_graph, row_nodes, col_nodes
        )
        perfect_matching.update(partial_matching)
        
        if not row_nodes and not col_nodes:
            is_unique = True
        if is_unique:
            break

        # 2. Select variable to intervene
        if strategy_type == 'marginalized benefit':
            sampled_matchings = greedy_min_degree_sampler(bipartite_graph, row_nodes, col_nodes, M_samples, mode=mode)
            if sampled_matchings == False:
                success = False
                break
                
            normalized_benefits = compute_normalized_marginal_benefit(bipartite_graph, sampled_matchings, col_nodes)
            intervened_index = select_variable_to_intervene(normalized_benefits)

        elif strategy_type == 'random':
            if not col_nodes: break
            intervened_index = int(np.random.choice([int(c) for c in col_nodes]))

        elif strategy_type == 'maxdegree':
            if not col_nodes: break
            degrees = {int(c): bipartite_graph.degree[c] for c in col_nodes}
            intervened_index = max(degrees, key=degrees.get)

        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        # 3. Perform Intervention
        if use_ica_in_intervention:
            I_minus_W_j_ica = perform_intervention_with_ICA(
                intervened_index, n_sources, n_samples, I_minus_W_true, used_distributions
            )
        else:
            I_minus_W_j_ica = perform_intervention_without_ICA(intervened_index, I_minus_W_true)

        experiment_count += 1

        # 4. Rank edges and prune
        ranked_edges = rank_candidate_rows_for_intervened_column(
            observational_matrix, I_minus_W_j_ica, intervened_index
        )
        
        accepted = False
        for r_i, j in ranked_edges:
            r_node, c_node = f"r{r_i}", f"{j}"
            if r_node not in bipartite_graph.nodes() or c_node not in bipartite_graph.nodes():
                continue

            temp_graph = bipartite_graph.copy()
            temp_rows = row_nodes.copy()
            temp_cols = col_nodes.copy()
            temp_graph, temp_rows, temp_cols = remove_edge(temp_graph, temp_rows, temp_cols, (r_i, j))
            
            try:
                if not temp_rows and not temp_cols: 
                    perfect_match_exists = True 
                elif len(temp_rows) == len(temp_cols):
                    matching = nx.algorithms.bipartite.maximum_matching(temp_graph, top_nodes=temp_rows)
                    perfect_match_exists = len(matching) // 2 == len(temp_rows)
                else:
                    perfect_match_exists = False
                    
                if perfect_match_exists:
                    bipartite_graph, row_nodes, col_nodes = remove_edge(bipartite_graph, row_nodes, col_nodes, (r_i, j))
                    perfect_matching[r_node] = c_node
                    accepted = True
                    break
            except Exception:
                continue

        if not accepted:
            success = False
            break
    
    if row_nodes or col_nodes:
        success = False

    return bipartite_graph, experiment_count, perfect_matching, success


def adaptive_experiment_design_with_ICA(K, n_sources, n_samples, zero_probability=0.5, M_samples=1000, mode='exact'):
    I_minus_W_true = generate_sparse_mixing_matrix(n_sources, zero_prob=zero_probability, weight_scale=4.0, min_abs=0.6, seed=None)
    sources, used_distributions = generate_exogenous_samples(n=n_sources, D=n_samples, distributions=None, seed=None)
    X_mixed = mix_sources(I_minus_W_true, sources)

    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_sources, max_iter=2000, tol=1e-5, random_state=0)
    ica.fit(X_mixed)
    I_minus_W_ica_raw = ica.components_
    
    aligned_ica, _ = best_row_permutation(I_minus_W_true, I_minus_W_ica_raw, match_by='cosine')
    I_minus_W_thresh_obs = normalize_and_threshold(aligned_ica)
    
    true_graph = generate_true_graph_from_I_minus_W(I_minus_W_true)
    fvs_size = len(minimum_feedback_vertex_set(true_graph))

    _, experiment_count_mb, perfect_matching_mb, success_mb = perform_adaptive_experiments(
        I_minus_W_thresh_obs, I_minus_W_true, n_sources, n_samples, used_distributions, 
        K, M_samples, strategy_type='marginalized benefit', mode=mode, use_ica_in_intervention=True
    )
    
    _, experiment_count_random, _, success_random = perform_adaptive_experiments(
        I_minus_W_thresh_obs, I_minus_W_true, n_sources, n_samples, used_distributions, 
        K, M_samples, strategy_type='random', mode=mode, use_ica_in_intervention=True
    )
    
    _, experiment_count_maxdeg, _, success_maxdeg = perform_adaptive_experiments(
        I_minus_W_thresh_obs, I_minus_W_true, n_sources, n_samples, used_distributions, 
        K, M_samples, strategy_type='maxdegree', mode=mode, use_ica_in_intervention=True
    )

    success = success_mb and success_random and success_maxdeg
    relative_error_mb, _, _ = evaluate_final_estimate(I_minus_W_thresh_obs, I_minus_W_true, perfect_matching_mb)

    return experiment_count_mb, experiment_count_random, experiment_count_maxdeg, fvs_size, success, relative_error_mb


def adaptive_experiment_design_without_ICA(K, n_sources, zero_probability=0.5, M_samples=1000, mode='exact'):
    I_minus_W_true = generate_sparse_mixing_matrix(n_sources, zero_prob=zero_probability, weight_scale=4.0, min_abs=0.5, seed=None)
    I_minus_W_ica_obs, _, _ = permute_and_scale_rows(I_minus_W_true, seed=None) 
    observational_matrix = I_minus_W_ica_obs
    
    unused_distributions = ['laplace'] * n_sources
    unused_n_samples = 1 

    true_graph = generate_true_graph_from_I_minus_W(I_minus_W_true)
    fvs_size = len(minimum_feedback_vertex_set(true_graph))

    _, experiment_count_mb, perfect_matching_mb, success_mb = perform_adaptive_experiments(
        observational_matrix, I_minus_W_true, unused_n_samples, unused_n_samples, unused_distributions, 
        K, M_samples, strategy_type='marginalized benefit', mode=mode, use_ica_in_intervention=False
    )
    
    _, experiment_count_random, _, success_random = perform_adaptive_experiments(
        observational_matrix, I_minus_W_true, unused_n_samples, unused_n_samples, unused_distributions, 
        K, M_samples, strategy_type='random', mode=mode, use_ica_in_intervention=False
    )
    
    _, experiment_count_maxdeg, _, success_maxdeg = perform_adaptive_experiments(
        observational_matrix, I_minus_W_true, unused_n_samples, unused_n_samples, unused_distributions, 
        K, M_samples, strategy_type='maxdegree', mode=mode, use_ica_in_intervention=False
    )

    success = success_mb and success_random and success_maxdeg
    relative_error_mb, _, _ = evaluate_final_estimate(observational_matrix, I_minus_W_true, perfect_matching_mb)

    return experiment_count_mb, experiment_count_random, experiment_count_maxdeg, fvs_size, success, relative_error_mb