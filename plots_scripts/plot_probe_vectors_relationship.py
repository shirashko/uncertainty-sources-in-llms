import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data Preparation based on experimental results for GPT-2, Gemma, and Llama
data = {
    'GPT-2 Small': {
        'Layer': [0, 4, 8, 11],
        'Cosine Similarity': [-0.045, 0.539, 0.059, 0.867]
    },
    'Gemma-2-2B': {
        'Layer': [0, 4, 8, 12, 16, 20, 24, 25],
        'Cosine Similarity': [0.699, 0.843, 0.584, 0.283, 0.170, -0.670, -0.389, -0.177]
    },
    'Llama-3.2-1B': {
        'Layer': [0, 4, 8, 12, 15],
        'Cosine Similarity': [0.161, -0.131, 0.878, 0.531, -0.606]
    }
}

# Configure plot style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
colors = ['#4C72B0', '#55A868', '#C44E52']

for i, (model, df_dict) in enumerate(data.items()):
    df = pd.DataFrame(df_dict)
    ax = axes[i]

    # Plot the trajectory of the Residual Mode cosine similarity
    sns.lineplot(data=df, x='Layer', y='Cosine Similarity', ax=ax,
                 marker='o', color=colors[i], linewidth=2.5, markersize=8)

    # Add an orthogonality reference line at y=0
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, label='Orthogonal (0.0)')

    # Shade geometric regions to interpret the similarity values
    ax.fill_between(df['Layer'], 0.7, 1.0, color='gray', alpha=0.1, label='Linear Overlap')
    ax.fill_between(df['Layer'], -0.2, 0.2, color='green', alpha=0.05, label='Near-Orthogonal')

    ax.set_title(f"Subspace Geometry: {model}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylim(-1, 1)

    # Set y-axis label only for the first subplot to avoid clutter
    if i == 0:
        ax.set_ylabel("Cosine Similarity ($W_{det}, W_{source}$)", fontsize=12)

    # Annotate key structural findings: Geometric Collapse and Divergence
    if model == 'Llama-3.2-1B':
        ax.annotate('Geometric Collapse', xy=(8, 0.878), xytext=(2, 0.85),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4))
    if model == 'Gemma-2-2B':
        ax.annotate('Divergence', xy=(16, 0.17), xytext=(12, -0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4))

plt.tight_layout()

# Place the legend below the subplots
plt.legend(loc='lower left', bbox_to_anchor=(-2.2, -0.3), ncol=4)

# Export the figure to a vector PDF for high-quality embedding in LaTeX
plt.savefig('../plots/subspace_geometry_analysis.pdf', format='pdf', bbox_inches='tight')

print("Graph successfully saved as subspace_geometry_analysis.pdf")