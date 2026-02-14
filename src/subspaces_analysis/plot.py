import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import MODEL_NAME

def plot_layer_wise_emergence(csv_path, output_dir):
    """
    Generates a complete thesis-ready plot showing the emergence,
    disentanglement, and orthogonality to baseline across layers.
    """
    df = pd.read_csv(csv_path)

    # Filter for the Null Space in Residual mode
    filtered_df = df[(df['Mode'] == 'Residual') & (df['Space'] == 'Null')]

    if filtered_df.empty:
        print("⚠️ No data found for plot. Check CSV filters.")
        return

    plt.figure(figsize=(13, 8))
    sns.set_style("whitegrid")

    # 1. Intra-class Consistency (High = Signal exists and is stable)
    plt.plot(filtered_df['Layer'], filtered_df['Intra_Epi'],
             marker='o', linewidth=3, label='Intra-Epistemic (Consistency)', color='#1f77b4')
    plt.plot(filtered_df['Layer'], filtered_df['Intra_Ale'],
             marker='s', linewidth=3, label='Intra-Aleatoric (Consistency)', color='#ff7f0e')

    # 2. Inter-Uncertainty Overlap (Low relative to Intra = Disentanglement)
    plt.plot(filtered_df['Layer'], filtered_df['Inter_Epi_Ale'],
             marker='D', linestyle='--', linewidth=2, label='Inter: Epi vs Ale (Overlap)', color='#d62728')

    # 3. Inter-Baseline Overlap (Near Zero = Orthogonality/Signal Isolation)
    # These lines prove that the uncertainty signal is NOT present in normal text
    plt.plot(filtered_df['Layer'], filtered_df['Inter_Base_Epi'],
             marker='v', linestyle=':', linewidth=1.5, label='Inter: Base vs Epi', color='#2ca02c', alpha=0.7)
    plt.plot(filtered_df['Layer'], filtered_df['Inter_Base_Ale'],
             marker='^', linestyle=':', linewidth=1.5, label='Inter: Base vs Ale', color='#9467bd', alpha=0.7)

    # 4. Baseline Self-Consistency (For reference)
    plt.plot(filtered_df['Layer'], filtered_df['Intra_Base'],
             marker='x', linestyle='-', linewidth=1, label='Baseline Consistency', color='gray', alpha=0.4)

    # Thesis Formatting
    plt.title(f'Comprehensive Uncertainty Geometry Evolution: {MODEL_NAME}', fontsize=16, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.ylim(-0.2, 1.1)  # Expanded to show near-zero values clearly

    # Place legend outside to keep the plot clean
    plt.legend(frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "full_uncertainty_orthogonality.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 Full orthogonality plot saved to: {save_path}")