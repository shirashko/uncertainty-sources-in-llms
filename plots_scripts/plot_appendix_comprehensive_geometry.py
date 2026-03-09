import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_appendix_comprehensive_geometry(model_configs, output_dir="../plots/appendix_results"):
    """
    Generates a high-resolution grid for the Appendix.
    Rows: Absolute vs Residual Mode.
    Columns: Original vs Logits vs Null Space.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_style("whitegrid")

    for path, model_name in model_configs:
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        # Define the structural grid
        modes = ['Absolute', 'Residual']
        # Ensure we match the exact strings in your CSV
        spaces = ['Original', 'Logits', 'Null']

        fig, axes = plt.subplots(len(modes), len(spaces), figsize=(18, 10), sharex=True)

        for r, mode in enumerate(modes):
            for c, space in enumerate(spaces):
                ax = axes[r, c]

                # Filter data for this specific subplot
                subset = df[(df['Space'] == space) & (df['Mode'] == mode)].copy()
                if subset.empty:
                    continue

                # 1. Intra-Baseline
                ax.plot(subset['Layer'], subset['Intra_Base'], label='Intra-Base',
                        marker='v', linestyle=':', color='#7F8C8D', alpha=0.6)

                # 2. Intra-Aleatoric
                ax.plot(subset['Layer'], subset['Intra_Ale'], label='Intra-Alea',
                        marker='o', linewidth=2.5, color='#27AE60')

                # 3. Intra-Epistemic
                ax.plot(subset['Layer'], subset['Intra_Epi'], label='Intra-Epi',
                        marker='s', linewidth=2.5, color='#2980B9')

                # 4. Inter-Class Overlap
                ax.plot(subset['Layer'], subset['Inter_Epi_Ale'], label='Inter-Overlap',
                        marker='x', linestyle='--', color='#C0392B', alpha=0.8)

                # Formatting
                if r == 0: ax.set_title(f"Space: {space}", fontsize=14, fontweight='bold')
                if c == 0: ax.set_ylabel(f"Mode: {mode}\nCosSim", fontsize=12, fontweight='bold')

                # Absolute mode is usually saturated at 1.0, Residual is lower
                ax.set_ylim(-0.1, 1.1)

        # Global styling
        plt.suptitle(f"Comprehensive Geometric Analysis: {model_name}", fontsize=20, fontweight='bold', y=0.98)

        # Shared Legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=True)

        plt.tight_layout(rect=[0, 0.07, 1, 0.94])

        file_name = f"{model_name.replace(' ', '_').lower()}_full_geometry.pdf"
        plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
        plt.close()
        print(f"✅ Appendix plot saved for {model_name}")


if __name__ == "__main__":
    CONFIGS = [
        ("../results/gpt2_multi_layer/multi_layer_metrics.csv", "GPT-2 Small"),
        ("../results/gemma-2-2b_multi_layer/multi_layer_metrics.csv", "Gemma-2-2b"),
        ("../results/Llama-3.2-1B_multi_layer/multi_layer_metrics.csv", "Llama-3.2-1B")
    ]
    plot_appendix_comprehensive_geometry(CONFIGS)