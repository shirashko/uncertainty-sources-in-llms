import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_all_models_consolidated(model_configs, output_filename="../plots/probing_analysis.pdf"):
    """
    Generates a consolidated academic figure with 3 subplots.
    Pairs tasks from the same subspace together in the legend.
    """
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})

    num_models = len(model_configs)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 9), sharey=True)

    if num_models == 1:
        axes = [axes]

    handles, labels = None, None

    # Define the explicit order for the legend to pair Detection and Type tasks per subspace
    # Adjust strings here if they differ exactly from your CSV content
    desired_order = [
        'Null Space | Detection (Cert vs Uncert)', 'Null Space | Type (Epi vs Alea)',
        'Original | Detection (Cert vs Uncert)', 'Original | Type (Epi vs Alea)',
        'Logit Space | Detection (Cert vs Uncert)', 'Logit Space | Type (Epi vs Alea)'
    ]

    for i, (file_path, model_name) in enumerate(model_configs):
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if 'Mode' in df.columns:
            df = df[df['Mode'] == 'Residual']

        spaces = ['Original', 'Null Space', 'Logit Space']
        df = df[df['Space'].isin(spaces)]
        df['Condition'] = df['Space'] + " | " + df['Task']

        # Convert 'Condition' to a categorical type with the specified order
        # This ensures both the plotting order and legend order are grouped by Subspace
        df['Condition'] = pd.Categorical(df['Condition'], categories=desired_order, ordered=True)
        df = df.sort_values('Condition')

        ax = axes[i]

        # Use a palette with enough colors for all unique conditions
        palette = sns.color_palette("husl", n_colors=len(desired_order))

        sns.lineplot(
            data=df, x='Layer', y='Accuracy', hue='Condition', style='Condition',
            markers=True, dashes=True, palette=palette, linewidth=2.5, markersize=8, ax=ax,
            hue_order=desired_order, style_order=desired_order
        )

        # Annotate peaks specifically for the Null Space tasks
        null_df = df[df['Space'] == 'Null Space']
        for cond in null_df['Condition'].unique():
            cond_data = null_df[null_df['Condition'] == cond]
            if not cond_data.empty:
                peak = cond_data.loc[cond_data['Accuracy'].idxmax()]
                ax.annotate(f"L{int(peak['Layer'])}",
                            xy=(peak['Layer'], peak['Accuracy']),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold', color='#333333')

        ax.set_title(model_name, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel("Transformer Layer Index", fontsize=13)
        ax.set_ylim(0.4, 1.05)

        if i == 0:
            ax.set_ylabel("Balanced Accuracy", fontsize=13)

        # Extract legend handles from the first plot to create the global legend
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    # Create the shared legend at the bottom with 2 columns to keep subspace pairs side-by-side
    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), frameon=True, shadow=True,
               title="Geometric Subspace & Uncertainty Task", title_fontsize=12, fontsize=11)

    plt.suptitle("Comparative Probing Analysis: Disentangling Uncertainty in Geometric Subspaces",
                 fontsize=22, fontweight='bold', y=0.96)

    # Adjust layout to prevent overlap with the global legend at the bottom
    plt.tight_layout(rect=[0, 0.18, 1, 0.94])

    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"✅ Final plot saved as: {output_filename}")
    plt.show()


if __name__ == "__main__":
    # Ensure paths match your local directory structure
    CONFIGS = [
        ("../results/gpt2_multi_layer/all_layers_probing_results.csv",
         "GPT-2 Small"),
        ("../results/gemma-2-2b_multi_layer/all_layers_probing_results.csv",
         "Gemma-2-2b"),
        ("../results/Llama-3.2-1B_multi_layer/all_layers_probing_results.csv",
         "Llama-3.2-1B")
    ]

    # Create output directory if it doesn't exist
    if not os.path.exists("../plots"):
        os.makedirs("../plots")

    plot_all_models_consolidated(CONFIGS)