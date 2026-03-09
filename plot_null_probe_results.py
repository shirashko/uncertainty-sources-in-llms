import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_null_space_only_consolidated(model_configs, output_filename="plots/null_space_only_probing.pdf"):
    """
    Generates a figure displaying ONLY Null Space results for all models.
    Uses distinct colors for Detection and Type tasks.
    """
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})

    num_models = len(model_configs)
    # Adjusted figsize for a more focused view
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 7), sharey=True)

    if num_models == 1:
        axes = [axes]

    handles, labels = None, None

    # Define tasks to keep things consistent across plots
    target_tasks = [
        'Detection (Cert vs Uncert)',
        'Type (Epi vs Alea)'
    ]

    # Using a high-contrast pair for the two tasks
    task_colors = {'Detection (Cert vs Uncert)': '#2C3E50', 'Type (Epi vs Alea)': '#E74C3C'}

    for i, (file_path, model_name) in enumerate(model_configs):
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Filter strictly for Null Space in Residual Mode
        if 'Mode' in df.columns:
            df = df[df['Mode'] == 'Residual']

        df = df[df['Space'] == 'Null Space']
        df = df[df['Task'].isin(target_tasks)]

        ax = axes[i]

        # Plot each task separately to ensure specific colors
        for task in target_tasks:
            subset = df[df['Task'] == task]
            if subset.empty:
                continue

            sns.lineplot(
                data=subset, x='Layer', y='Accuracy',
                label=task, marker='o', linewidth=3, markersize=8,
                color=task_colors[task], ax=ax
            )

            # Annotate Peak
            peak = subset.loc[subset['Accuracy'].idxmax()]
            ax.annotate(f"Peak L{int(peak['Layer'])}",
                        xy=(peak['Layer'], peak['Accuracy']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')

        ax.set_title(model_name, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Transformer Layer Index", fontsize=12)
        ax.set_ylim(0.45, 0.95)  # Zoomed in for clarity on null results

        if i == 0:
            ax.set_ylabel("Balanced Accuracy", fontsize=12)
            handles, labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()

    # Place shared legend below the plots
    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.05), frameon=True, shadow=True,
               title="Uncertainty Probing Task (Null Space)", title_fontsize=11)

    plt.suptitle("Uncertainty Metadata Localization in the Unembedding Null Space",
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"✅ Null Space plot saved as: {output_filename}")
    plt.show()


if __name__ == "__main__":
    CONFIGS = [
        ("results/gpt2_multi_layer/all_layers_probing_results.csv", "GPT-2 Small"),
        ("results/gemma-2-2b_multi_layer/all_layers_probing_results.csv", "Gemma-2-2b"),
        ("results/Llama-3.2-1B_multi_layer/all_layers_probing_results.csv", "Llama-3.2-1B")
    ]

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plot_null_space_only_consolidated(CONFIGS)