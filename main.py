import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import os # To save the CSV file

# Import core functions from the newly created src modules
from src.adaptive_design import adaptive_experiment_design_without_ICA, adaptive_experiment_design_with_ICA
from src.graph_theory import minimum_feedback_vertex_set, generate_true_graph_from_I_minus_W
from src.data_generation import generate_sparse_mixing_matrix

# Configuration for the simulation loop
n_sources_range = range(4, 41)
n_samples = 20000  # Samples per experiment (for ICA mode)
repeats_per_setting = 100
zero_probability = 0.8  # Default sparsity level for I-W matrix (P(W_ij=0 | i!=j))
M_samples = 1000  # Samples for the perfect matching sampler/enumerator
c = 1 # Multiplier for the intervention budget K = c * n_sources
sampler_mode = 'exact' # Use 'exact' for enumeration, 'sample' for MCMC sampling

# --- Simulation Execution ---

def run_simulations(use_ica_mode):
    """Runs the full simulation loop for comparison of strategies."""
    
    # Storage lists for results
    results = {
        'n_sources': [],
        'avg_adaptive': [], 'std_adaptive': [],
        'avg_random': [], 'std_random': [],
        'avg_maxdeg': [], 'std_maxdeg': [],
        'avg_fvs': [], 'std_fvs': [],
        'adaptive_lower_bound': [], 'adaptive_upper_bound': [],
        'relative_error_list': []
    }
    
    sim_func = adaptive_experiment_design_with_ICA if use_ica_mode else adaptive_experiment_design_without_ICA
    
    print(f"Starting simulation mode: {'with ICA' if use_ica_mode else 'without ICA (Proxy)'}")
    
    for n_sources in n_sources_range:
        print(f"Running experiments for {n_sources} sources...")
        K = int(c * n_sources) # intervention budget
        
        adapt_list, rand_list, maxdeg_list, fvs_list, relative_error_list = [], [], [], [], []
        number_of_failure = 0

        for _ in range(repeats_per_setting):
            # Dynamic edge probability based on graph size (as seen in the notebook)
            edge_prob = round(random.uniform(round(1/n_sources, 3), round(1/n_sources, 3) + 0.015), 3)
            current_zero_prob = max(1 - edge_prob, zero_probability)
            
            try:
                if use_ica_mode:
                    emb, er, em, fvs, success, relative_error_mb = sim_func(
                        K, n_sources, n_samples, current_zero_prob, M_samples=M_samples, mode=sampler_mode)
                else:
                    emb, er, em, fvs, success, relative_error_mb = sim_func(
                        K, n_sources, current_zero_prob, M_samples=M_samples, mode=sampler_mode)

                if success:
                    adapt_list.append(emb)
                    rand_list.append(er)
                    maxdeg_list.append(em)
                    fvs_list.append(fvs)
                    relative_error_list.append(relative_error_mb)
                else:
                    number_of_failure += 1
            except Exception as e:
                # Catching potential solver or other issues
                # print("An exception occurred during simulation.")
                # print(f"CRASH Report: {type(e).__name__}: {e}")
                number_of_failure += 1

        # Compute averages and standard deviations
        if adapt_list:
            adapt_avg = np.mean(adapt_list)
            rand_avg = np.mean(rand_list)
            maxdeg_avg = np.mean(maxdeg_list)
            fvs_avg = np.mean(fvs_list)

            adapt_std = np.std(adapt_list)
            rand_std = np.std(rand_list)
            maxdeg_std = np.std(maxdeg_list)
            fvs_std = np.std(fvs_list)
        else:
            adapt_avg, rand_avg, maxdeg_avg, fvs_avg = 0, 0, 0, 0
            adapt_std, rand_std, maxdeg_std, fvs_std = 0, 0, 0, 0
             
        # Save results
        results['n_sources'].append(n_sources)
        results['avg_adaptive'].append(adapt_avg)
        results['std_adaptive'].append(adapt_std)
        results['avg_random'].append(rand_avg)
        results['std_random'].append(rand_std)
        results['avg_maxdeg'].append(maxdeg_avg)
        results['std_maxdeg'].append(maxdeg_std)
        results['avg_fvs'].append(fvs_avg)
        results['std_fvs'].append(fvs_std)
        
        # Calculate adaptive bounds (as seen in notebook)
        alpha_bound = 1 - 0.2
        results['adaptive_lower_bound'].append(adapt_avg * alpha_bound)
        results['adaptive_upper_bound'].append(adapt_avg)
        
        results['relative_error_list'].append(relative_error_list)
    return results


# --- Plotting Functions ---

def plot_comparison(df, mode_name):
    """Generates the comparison plot of intervention counts."""
    
    plt.figure(figsize=(10, 6))
    n_sources_range = df['n_sources']
    alpha_error = np.sqrt(1 / repeats_per_setting)
    
    # Plot Our Method (Adaptive)
    plt.errorbar(
        n_sources_range, df['avg_adaptive'], 
        yerr=alpha_error * df['std_adaptive'], 
        label="Our Method", marker='o', color='#2ca02c', zorder=10, capsize=3
    )
    
    # Plot Random
    plt.errorbar(
        n_sources_range, df['avg_random'], 
        yerr=alpha_error * df['std_random'], 
        label="Random", marker='s', color='#7f7f7f', linestyle='--', capsize=3
    )
    
    # Plot Max Degree
    plt.errorbar(
        n_sources_range, df['avg_maxdeg'], 
        yerr=alpha_error * df['std_maxdeg'], 
        label="Max Degree", marker='^', color='#ff7f0e', linestyle='-.', capsize=3
    )
    
    # Plot FVS (Lower Bound)
    plt.plot(
        n_sources_range, df['avg_fvs'],  
        label="FVS Size (Lower Bound)", marker='x', color='#d62728', linestyle=':', linewidth=2
    )

    # Shade adaptive range (Confidence/Variation)
    plt.fill_between(
        n_sources_range,
        df['adaptive_lower_bound'],
        df['adaptive_upper_bound'],
        color='#2ca02c',
        alpha=0.2,
        label=r"Adaptive Range"
    )

    plt.xlabel("Number of Variables (Graph Size)")
    plt.ylabel("Number of Interventions (Mean ± SE)")
    plt.title(f"Intervention Strategy Comparison ({mode_name})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join('results', f'comparison_plot_{mode_name}.png'))
    plt.show()


def plot_relative_error(df, mode_name):
    """Generates the relative error vs. number of sources plot."""
    
    relative_errors = df['relative_error_list'].tolist()
    
    # Calculate mean and standard deviation of relative error across all repeats
    means = [np.mean(errors) if errors else 0 for errors in relative_errors]
    stds = [np.std(errors) if errors else 0 for errors in relative_errors]
    
    n_sources_range = df['n_sources']
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(n_sources_range, means, yerr=stds, fmt='-o', capsize=3)
    plt.xlabel("Number of Nodes (Variables)")
    plt.ylabel("Relative Error (Mean ± Std)")
    plt.title(f"Relative Error of Our Method ({mode_name})")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join('results', f'relative_error_plot_{mode_name}.png'))
    plt.show()


# --- Main Execution Block ---

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # --- Run and save WITHOUT ICA Proxy ---
    results_no_ica = run_simulations(use_ica_mode=False)
    results_df_no_ica = pd.DataFrame(results_no_ica)
    
    csv_path_no_ica = os.path.join('results', 'adaptive_experiment_results_proxy.csv')
    results_df_no_ica.to_csv(csv_path_no_ica, index=False)
    print(f"\nResults saved to {csv_path_no_ica}")
    
    # Plotting for Proxy mode
    plot_comparison(results_df_no_ica, 'Proxy')
    plot_relative_error(results_df_no_ica, 'Proxy')

    # --- Run and save WITH ICA (if computational resources allow) ---
    # NOTE: Running the full ICA simulation (below) is often very time-consuming.
    # The original notebook focuses on the 'without ICA' (proxy) results.

    # Uncomment the following block if you wish to run the full ICA simulation:
    """
    print("-" * 50)
    results_with_ica = run_simulations(use_ica_mode=True)
    results_df_with_ica = pd.DataFrame(results_with_ica)
    
    csv_path_with_ica = os.path.join('results', 'adaptive_experiment_results_ica.csv')
    results_df_with_ica.to_csv(csv_path_with_ica, index=False)
    print(f"\nResults saved to {csv_path_with_ica}")
    
    # Plotting for ICA mode
    plot_comparison(results_df_with_ica, 'ICA')
    plot_relative_error(results_df_with_ica, 'ICA')
    """