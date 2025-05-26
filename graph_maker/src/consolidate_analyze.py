import pandas as pd
import numpy as np
import os
import argparse
from scipy.stats import skew, kurtosis, norm 

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') 
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Matplotlib/Seaborn not found. Visualizations will be skipped.")
    PLOTTING_AVAILABLE = False


try:
    from utils import KEY_DEP_RATIO_PAIRS
except ImportError:
    print("CRITICAL: utils.py with KEY_DEP_RATIO_PAIRS not found. Ratios will be missing in summaries.")
    KEY_DEP_RATIO_PAIRS = []


parser = argparse.ArgumentParser(description="Consolidate chunked results and perform final analysis.")
parser.add_argument("--total_chunks", type=int, required=True, help="Total number of chunks processed.")
args = parser.parse_args()
TOTAL_CHUNKS = 20

OUTPUT_DIR_ANALYSIS = '/home/mmnima/jokes/graph_maker/analysis_results_LONG_CHOPPED_LLM' 

numeric_cols_for_summary_common = [
    'subgraph_components', 'subgraph_avg_degree', 'subgraph_density',
    'subgraph_diameter', 'mean_betweenness_subgraph', 'mean_closeness_subgraph',
    'mean_clustering_coefficient_subgraph', 'degree_assortativity_subgraph', 'mean_eigenvector_centrality_subgraph',
    'min_degree_subgraph', 'max_degree_subgraph', 'median_degree_subgraph', 'iqr_degree_subgraph',
    'mean_node_strength_subgraph', 'global_efficiency_subgraph', 'local_efficiency_subgraph',
    'num_bridges_subgraph', 'ratio_bridges_subgraph', 'num_articulation_points_subgraph',
    'ratio_articulation_points_subgraph', 'modularity_subgraph', 'num_inter_community_edges_subgraph',
    'ratio_inter_community_edges_subgraph', 'cycle_count_subgraph', 'cycle_density_subgraph',
    'bridging_relation_ratio',
    'shortest_path_general_mean', 'shortest_path_general_median', 'shortest_path_general_min',
    'shortest_path_general_max', 'shortest_path_general_iqr',
    'mean_bridged_path_length', 'max_bridged_path_length',
    'joke_nodes_mean_bg_degree_centrality', 'joke_nodes_mean_bg_betweenness_centrality', 
    'joke_nodes_mean_bg_closeness_centrality', 'joke_nodes_mean_bg_eigenvector_centrality',
    'community_count', 'concept_contrast',
    'semantic_heterogeneity_mean_dist', 'semantic_heterogeneity_median_dist', 'semantic_heterogeneity_iqr_dist',
    'mean_edge_weight_in_bg_subgraph', 'proportion_rare_edges_in_bg_subgraph',
    'mean_centrality_bridging_concepts_degree', 'mean_centrality_bridging_concepts_betweenness',
    'mean_centrality_bridging_concepts_closeness', 'mean_centrality_bridging_concepts_eigenvector',
    'mean_semantic_specificity_bg', 'mean_dist_to_nearest_global_hub',
    'dependency_label_entropy', 'ratio_core_modifying_relations', 'proportion_unique_relation_types'
] 
if KEY_DEP_RATIO_PAIRS: 
    for label1, label2 in KEY_DEP_RATIO_PAIRS: 
        numeric_cols_for_summary_common.append(f'ratio_{label1}_to_{label2}')



def calculate_iqr(data_series): 
    if data_series.empty: return 0
    numeric_data = data_series.dropna()
    if len(numeric_data) < 2: return 0
    try:
        q1, q3 = np.percentile(numeric_data, [25, 75])
        return q3 - q1
    except Exception as e:
        print(f"Could not calculate IQR: {e}")
        return 0

def calculate_descriptive_stats(df, name): 
    stats = {}
    
    if df.empty:
        print(f"DataFrame for {name} is empty. Skipping descriptive stats.")
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        print(f"No numeric columns in DataFrame for {name}. Skipping descriptive stats.")
        return pd.DataFrame()

    for col in numeric_cols:
        data = df[col].dropna()
        if not data.empty:
            try:
                stats[col] = {
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'std_dev': np.std(data),
                    'variance': np.var(data),
                    'skewness': skew(data) if len(data) > 1 else 0,
                    'kurtosis': kurtosis(data) if len(data) > 1 else 0,
                    'min': np.min(data),
                    'max': np.max(data),
                    'iqr': calculate_iqr(data) 
                }
            except Exception as e:
                print(f"Stats calculation error for {col} in {name}: {e}")
                stats[col] = {k: np.nan for k in ['mean', 'median', 'std_dev', 'variance', 'skewness', 'kurtosis', 'min', 'max', 'iqr']}
        else:
            stats[col] = {k: np.nan for k in ['mean', 'median', 'std_dev', 'variance', 'skewness', 'kurtosis', 'min', 'max', 'iqr']}
    
    stats_df = pd.DataFrame(stats).T
    print(f"\nDescriptive Statistics for {name}:\n{stats_df.to_string()}")
    return stats_df

def visualize_distributions(df, name, output_dir_viz): 
    if not PLOTTING_AVAILABLE:
        print("Plotting skipped as libraries are not available.")
        return
    if df.empty:
        print(f"DataFrame for {name} (visualize) is empty. Skipping plots.")
        return

    numeric_cols_plot = df.select_dtypes(include=np.number).columns
    plot_cols = [col for col in numeric_cols_plot if df[col].nunique() > 1] 

    if not plot_cols:
        print(f"No numeric columns with variance found in {name} for plotting. Skipping visualizations.")
        return

    if not os.path.exists(output_dir_viz):
        os.makedirs(output_dir_viz)
    
    num_features_per_page = 3 
    num_plots_per_feature = 3 
    plots_per_page_total = num_features_per_page * num_plots_per_feature 

    num_pages = int(np.ceil(len(plot_cols) / num_features_per_page))

    for page_num in range(num_pages):
        fig, axes = plt.subplots(num_features_per_page, num_plots_per_feature, figsize=(18, 5 * num_features_per_page))
        if num_features_per_page == 1: 
            axes = np.array([axes])
        
        
        
        current_feature_idx_start = page_num * num_features_per_page
        
        for i in range(num_features_per_page): 
            feature_list_idx = current_feature_idx_start + i
            if feature_list_idx >= len(plot_cols): 
                
                for row_ax in range(i, num_features_per_page):
                    for col_ax in range(num_plots_per_feature):
                        if axes.ndim == 2 and axes.shape[0] > row_ax and axes.shape[1] > col_ax :
                            axes[row_ax, col_ax].axis('off')
                break 

            col = plot_cols[feature_list_idx]
            data = df[col].dropna()

            ax_row = axes[i] 

            if data.empty:
                for k_ax in range(num_plots_per_feature): ax_row[k_ax].set_title(f'{col}\n(No Data)'); ax_row[k_ax].axis('off')
                continue
            
            
            try:
                sns.histplot(data, kde=True, ax=ax_row[0], bins=min(20, data.nunique() if data.nunique() > 0 else 1))
                ax_row[0].set_title(f'Hist/KDE of {col}\n({name})')
            except Exception as e:
                print(f"Hist/KDE plot error for {col} ({name}): {e}")
                ax_row[0].set_title(f'Hist/KDE Error\n{col}'); ax_row[0].axis('off')
            
            
            try:
                sns.boxplot(x=data, ax=ax_row[1])
                ax_row[1].set_title(f'Box Plot of {col}\n({name})')
            except Exception as e:
                print(f"Box plot error for {col} ({name}): {e}")
                ax_row[1].set_title(f'Box Plot Error\n{col}'); ax_row[1].axis('off')

            
            try:
                ax_row[2].hist(data, bins=min(20, data.nunique() if data.nunique() > 0 else 1), density=True, alpha=0.6, color='g', label='Data')
                xmin, xmax = ax_row[2].get_xlim()
                lnspc = np.linspace(xmin, xmax, 100) if xmax > xmin else np.array([xmin])
                if len(data) > 1: 
                    loc, scale = norm.fit(data)
                    pdf_norm = norm.pdf(lnspc, loc=loc, scale=scale)
                    ax_row[2].plot(lnspc, pdf_norm, 'r-', lw=2, label=f'Norm Fit (μ={loc:.2f}, σ={scale:.2f})')
                ax_row[2].set_title(f'Distribution Fit for {col}\n({name})')
                ax_row[2].legend()
            except Exception as e:
                print(f"Dist fit plot error for {col} ({name}): {e}")
                ax_row[2].set_title(f'Dist Fit Error\n{col}'); ax_row[2].axis('off')

        plt.tight_layout(pad=2.0)
        plot_filename = os.path.join(output_dir_viz, f'{name.replace(" ", "_").lower()}_distributions_page_{page_num+1}.png')
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Failed to save plot {plot_filename}: {e}")
        plt.close(fig)


def main():
    print(f"Starting consolidation of results from {TOTAL_CHUNKS} chunks.")

    
    all_joke_dfs = []
    print("Consolidating joke statistics...")
    for i in range(TOTAL_CHUNKS):
        chunk_fname = os.path.join(OUTPUT_DIR_ANALYSIS, f'joke_statistics_task_{i}.csv')
        if os.path.exists(chunk_fname):
            try:
                
                df_chunk = pd.read_csv(chunk_fname, index_col='joke_id')
                if not df_chunk.empty:
                    all_joke_dfs.append(df_chunk)
                    print(f"Loaded {chunk_fname} ({len(df_chunk)} rows)")
                else:
                    print(f"File {chunk_fname} was empty.")
            except pd.errors.EmptyDataError:
                print(f"File {chunk_fname} is empty (EmptyDataError).")
            except Exception as e:
                print(f"Error loading or processing {chunk_fname}: {e}")
        else:
            print(f"File {chunk_fname} not found.")
    
    if all_joke_dfs:
        consolidated_jokes_df = pd.concat(all_joke_dfs)
        print(f"Consolidated joke statistics: {len(consolidated_jokes_df)} total rows.")
        
        
        valid_summary_cols_jokes = [col for col in numeric_cols_for_summary_common if col in consolidated_jokes_df.columns]
        joke_stats_for_summary = consolidated_jokes_df[valid_summary_cols_jokes]

        calculate_descriptive_stats(joke_stats_for_summary, "Jokes (Consolidated)")
        visualize_distributions(joke_stats_for_summary, "Jokes (Consolidated)", OUTPUT_DIR_ANALYSIS)
        
        csv_filename_consolidated_jokes = os.path.join(OUTPUT_DIR_ANALYSIS, 'jokes_statistics_CONSOLIDATED.csv')
        consolidated_jokes_df.to_csv(csv_filename_consolidated_jokes)
        print(f"Full consolidated joke statistics saved to '{csv_filename_consolidated_jokes}'")
    else:
        print("No joke dataframes were loaded for consolidation.")

    
    all_sentence_dfs = []
    print("Consolidating sentence statistics...")
    for i in range(TOTAL_CHUNKS):
        chunk_fname = os.path.join(OUTPUT_DIR_ANALYSIS, f'sentence_statistics_task_{i}.csv')
        if os.path.exists(chunk_fname):
            try:
                df_chunk = pd.read_csv(chunk_fname, index_col='sentence_id')
                if not df_chunk.empty:
                    all_sentence_dfs.append(df_chunk)
                    print(f"Loaded {chunk_fname} ({len(df_chunk)} rows)")
                else:
                    print(f"File {chunk_fname} was empty.")
            except pd.errors.EmptyDataError:
                print(f"File {chunk_fname} is empty (EmptyDataError).")
            except Exception as e:
                print(f"Error loading or processing {chunk_fname}: {e}")
        else:
            print(f"File {chunk_fname} not found.")

    if all_sentence_dfs:
        consolidated_sentences_df = pd.concat(all_sentence_dfs)
        print(f"Consolidated sentence statistics: {len(consolidated_sentences_df)} total rows.")

        valid_summary_cols_sentences = [col for col in numeric_cols_for_summary_common if col in consolidated_sentences_df.columns]
        sentence_stats_for_summary = consolidated_sentences_df[valid_summary_cols_sentences]

        calculate_descriptive_stats(sentence_stats_for_summary, "Sentences (Consolidated)")
        visualize_distributions(sentence_stats_for_summary, "Sentences (Consolidated)", OUTPUT_DIR_ANALYSIS)
        
        csv_filename_consolidated_sentences = os.path.join(OUTPUT_DIR_ANALYSIS, 'sentence_statistics_CONSOLIDATED.csv')
        consolidated_sentences_df.to_csv(csv_filename_consolidated_sentences)
        print(f"Full consolidated sentence statistics saved to '{csv_filename_consolidated_sentences}'")
    else:
        print("No sentence dataframes were loaded for consolidation.")

    print("Consolidation and final analysis complete.")

if __name__ == '__main__':
    main()
