import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_multi_model_geometric_analysis(model_configs, output_dir="../plots"):
    """
    Plots geometric cohesion metrics for multiple models in a single figure.
    Focuses on Null Space in Residual Mode to isolate uncertainty signatures.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_models = len(model_configs)
    # Wider figure for three models
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 8), sharey=True)
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})

    # Ensure axes is iterable for single-model cases
    if num_models == 1:
        axes = [axes]

    for i, (path, model_name) in enumerate(model_configs):
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        df = pd.read_csv(path)

        # Filtering for Null Space (Residual Mode)
        # Standardizing 'Null Space' vs 'Null' strings
        df_plot = df[(df['Space'].str.contains('Null', case=False)) &
                     (df['Mode'] == 'Residual')].copy()

        ax = axes[i]

        # 1. Intra-Baseline (Certainty - The Control Group)
        ax.plot(df_plot['Layer'], df_plot['Intra_Base'], label='Intra-Baseline',
                marker='v', linestyle=':', linewidth=2, color='#7F8C8D', alpha=0.7)

        # 2. Intra-Aleatoric (Cohesion of Aleatoric signal)
        ax.plot(df_plot['Layer'], df_plot['Intra_Ale'], label='Intra-Aleatoric',
                marker='o', linewidth=3.5, color='#27AE60')

        # 3. Intra-Epistemic (Cohesion of Epistemic signal)
        ax.plot(df_plot['Layer'], df_plot['Intra_Epi'], label='Intra-Epistemic',
                marker='s', linewidth=3.5, color='#2980B9')

        # 4. Inter-Class Overlap (Similarity between Epi/Alea - The Noise)
        ax.plot(df_plot['Layer'], df_plot['Inter_Epi_Ale'], label='Inter-Class Overlap',
                marker='x', linestyle='--', linewidth=2, color='#C0392B')

        # Identify Peak Separation Layer
        diff = df_plot['Intra_Ale'] - df_plot['Inter_Epi_Ale']
        if not diff.empty:
            peak_layer = df_plot.loc[diff.idxmax(), 'Layer']
            ax.axvline(x=peak_layer, color='#2C3E50', linestyle='-.', alpha=0.4)
            ax.text(peak_layer, -0.05, f' Max Separation (L{int(peak_layer)})',
                    rotation=90, va='bottom', fontsize=9, fontweight='bold', color='#2C3E50')

        ax.set_title(f"{model_name}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Layer Index", fontsize=12)
        if i == 0:
            ax.set_ylabel("Mean Cosine Similarity", fontsize=12)

        ax.set_ylim(-0.15, 0.8)

        # Remove individual legends to use a single shared legend later
        if ax.get_legend():
            ax.get_legend().remove()

    # Shared Legend and Main Title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02),
               frameon=True, shadow=True, fontsize=11)

    plt.suptitle("Geometric Consistency in Null Space (Residual Mode): Uncertainty Signatures vs. Baseline",
                 fontsize=20, fontweight='bold', y=0.98)

    # Leave space for the legend at the bottom and the suptitle at the top
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    save_path = os.path.join(output_dir, "geometric_analysis.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"✅ Success: Combined geometric plot saved as {save_path}")
    plt.show()


if __name__ == "__main__":
    CONFIGS = [
        ("../results/gpt2_multi_layer/multi_layer_metrics.csv", "GPT-2 Small"),
        ("../results/gemma-2-2b_multi_layer/multi_layer_metrics.csv", "Gemma-2-2b"),
        ("../results/Llama-3.2-1B_multi_layer/multi_layer_metrics.csv", "Llama-3.2-1B")
    ]

    plot_multi_model_geometric_analysis(CONFIGS)