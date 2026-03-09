import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_appendix_probing_matrix(model_configs, output_dir="../plots/appendix_probing"):
    """
    Generates a comprehensive 2x3 grid for each model to be used in the appendix.
    Rows: Absolute vs Residual Mode.
    Columns: Original vs Logits vs Null Space subspaces.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Professional plot styling
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})

    # Task colors: Detection (Uncertainty existence) vs Type (Source of uncertainty)
    task_colors = {
        'Detection (Cert vs Uncert)': '#2C3E50',  # Dark Blue/Grey
        'Type (Epi vs Alea)': '#E67E22'  # Orange
    }

    for file_path, model_name in model_configs:
        if not os.path.exists(file_path):
            print(f"Skipping {model_name}: file not found.")
            continue

        df = pd.read_csv(file_path)

        # Standardizing names for logic
        df['Space'] = df['Space'].replace({'Null': 'Null Space', 'original': 'Original', 'logits': 'Logit Space'})

        modes = ['Absolute', 'Residual']
        spaces = ['Original', 'Logit Space', 'Null Space']

        fig, axes = plt.subplots(len(modes), len(spaces), figsize=(18, 10), sharex=True, sharey=False)

        for r_idx, mode in enumerate(modes):
            for c_idx, space in enumerate(spaces):
                ax = axes[r_idx, c_idx]

                # Filtering data for the specific subplot
                subset = df[(df['Mode'] == mode) & (df['Space'] == space)]

                if subset.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center')
                    continue

                sns.lineplot(
                    data=subset, x='Layer', y='Accuracy', hue='Task',
                    palette=task_colors, marker='o', linewidth=2, ax=ax
                )

                # Titles and Labels
                if r_idx == 0:
                    ax.set_title(f"Subspace: {space}", fontsize=14, fontweight='bold')
                if c_idx == 0:
                    ax.set_ylabel(f"{mode} Mode\nBalanced Accuracy", fontsize=12, fontweight='bold')

                # Reference line for random guess
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')

                # Consistency in Y-axis per mode
                if mode == 'Absolute':
                    ax.set_ylim(0.4, 1.05)
                else:
                    # Residual mode in Null Space might be lower, allowing local scale
                    ax.set_ylim(0.4, 1.05)

                ax.get_legend().remove()

        # Overall Figure Formatting
        plt.suptitle(f"Probing Accuracy Matrix: {model_name}\nCross-Subspace and Multi-Mode Dynamics",
                     fontsize=20, fontweight='bold', y=0.98)

        # Add a shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02),
                   frameon=True, shadow=True, fontsize=11)

        plt.tight_layout(rect=[0, 0.08, 1, 0.94])

        save_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_probing_matrix.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅ Matrix plot saved for {model_name}")


if __name__ == "__main__":
    CONFIGS = [
        ("../results/gpt2_multi_layer/all_layers_probing_results.csv", "GPT-2 Small"),
        ("../results/gemma-2-2b_multi_layer/all_layers_probing_results.csv", "Gemma-2-2b"),
        ("../results/Llama-3.2-1B_multi_layer/all_layers_probing_results.csv", "Llama-3.2-1B")
    ]
    plot_appendix_probing_matrix(CONFIGS)