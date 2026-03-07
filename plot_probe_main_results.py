import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_vertical_model_dashboard_dual_peaks(model_configs, geometry_data,
                                             output_filename="plots/vertical_results_dual_peaks.pdf"):
    """
    Creates a 2-row x 3-column matrix.
    Includes separate vertical peak indicators for Detection and Type tasks.
    """
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})

    num_models = len(model_configs)
    fig, axes = plt.subplots(2, num_models, figsize=(5 * num_models, 10), sharex='col')

    # Consistent Task styling
    task_colors = {'Detection (Cert vs Uncert)': '#2C3E50', 'Type (Epi vs Alea)': '#E74C3C'}
    model_colors = {'GPT-2 Small': '#4C72B0', 'Gemma-2-2B': '#55A868', 'Llama-3.2-1B': '#8E44AD'}

    for i, (file_path, model_name) in enumerate(model_configs):
        ax_prob = axes[0, i]
        ax_geom = axes[1, i]

        peaks = {}

        if os.path.exists(file_path):
            df_prob = pd.read_csv(file_path)
            if 'Mode' in df_prob.columns:
                df_prob = df_prob[df_prob['Mode'] == 'Residual']
            df_prob = df_prob[(df_prob['Space'] == 'Null Space')]

            for task, color in task_colors.items():
                subset = df_prob[df_prob['Task'] == task]
                if not subset.empty:
                    # Plot accuracy
                    ax_prob.plot(subset['Layer'], subset['Accuracy'],
                                 label=task, color=color, marker='o', linewidth=2.5, markersize=6)

                    # Identify peak layer for this specific task
                    p_idx = subset['Accuracy'].idxmax()
                    peaks[task] = subset.loc[p_idx, 'Layer']

            # Draw Task-Specific Peak Indicators across both rows
            for task, p_layer in peaks.items():
                col = task_colors[task]
                # Draw on Probing ax
                ax_prob.axvline(p_layer, color=col, linestyle=':', alpha=0.6, linewidth=2)
                # Draw on Geometry ax
                ax_geom.axvline(p_layer, color=col, linestyle=':', alpha=0.6, linewidth=2)

                # Label the Type Peak specifically as requested
                if 'Type' in task:
                    ax_prob.text(p_layer, 0.96, f'Type Peak: L{int(p_layer)}',
                                 color=col, fontweight='bold', fontsize=9, ha='center')

            ax_prob.set_ylim(0.4, 1.0)
            ax_prob.axhline(0.5, color='black', linestyle=':', alpha=0.5)
            ax_prob.set_title(f"{model_name}\nProbing Accuracy", fontsize=14, fontweight='bold')
            if i == 0:
                ax_prob.set_ylabel("Balanced Accuracy", fontsize=12)

        # --- ROW 2: SUBSPACE GEOMETRY ---
        geom_df = pd.DataFrame(geometry_data[model_name])
        ax_geom.plot(geom_df['Layer'], geom_df['Cosine Similarity'],
                     marker='s', color=model_colors[model_name], linewidth=2.5, markersize=7)

        ax_geom.axhline(0, color='black', linestyle='--', alpha=0.6)
        ax_geom.fill_between(geom_df['Layer'], 0.7, 1.0, color='gray', alpha=0.1)
        ax_geom.fill_between(geom_df['Layer'], -0.2, 0.2, color='green', alpha=0.05)

        ax_geom.set_ylim(-1, 1)
        ax_geom.set_xlabel("Transformer Layer", fontsize=12)
        if i == 0:
            ax_geom.set_ylabel("Cosine Similarity ($W_{det}, W_{source}$)", fontsize=11)
            ax_geom.set_title("Subspace Geometry", fontsize=13, fontweight='bold')

    prob_handles, prob_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(prob_handles, prob_labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), frameon=True, shadow=True)

    plt.suptitle("Task-Specific Maturation in the Null Space", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"✅ Success: Dual-peak dashboard saved as {output_filename}")
    plt.show()


if __name__ == "__main__":
    CONFIGS = [
        ("results/gpt2_multi_layer/all_layers_probing_results.csv", "GPT-2 Small"),
        ("results/gemma-2-2b_multi_layer/all_layers_probing_results.csv", "Gemma-2-2B"),
        ("results/Llama-3.2-1B_multi_layer/all_layers_probing_results.csv", "Llama-3.2-1B")
    ]

    GEOM_DATA = {
        'GPT-2 Small': {'Layer': [0, 4, 8, 11], 'Cosine Similarity': [-0.045, 0.539, 0.059, 0.867]},
        'Gemma-2-2B': {'Layer': [0, 4, 8, 12, 16, 20, 24, 25],
                       'Cosine Similarity': [0.699, 0.843, 0.584, 0.283, 0.170, -0.670, -0.389, -0.177]},
        'Llama-3.2-1B': {'Layer': [0, 4, 8, 12, 15], 'Cosine Similarity': [0.161, -0.131, 0.878, 0.531, -0.606]}
    }

    if not os.path.exists("plots"): os.makedirs("plots")
    plot_vertical_model_dashboard_dual_peaks(CONFIGS, GEOM_DATA)