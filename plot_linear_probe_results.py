import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_comprehensive_probing(file_path, model_name):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)

    if 'Mode' in df.columns:
        df = df[df['Mode'] == 'Residual']

    spaces_to_plot = ['Original', 'Null Space', 'Logit Space']
    df = df[df['Space'].isin(spaces_to_plot)]

    # יצירת קומבינציה ייחודית לכל קו בגרף
    df['Condition'] = df['Space'] + " | " + df['Task']

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    # פלטת צבעים רחבה וניגודית
    unique_conds = sorted(df['Condition'].unique())
    color_palette = sns.color_palette("husl", n_colors=len(unique_conds))

    ax = sns.lineplot(
        data=df,
        x='Layer',
        y='Accuracy',
        hue='Condition',
        style='Condition',
        markers=True,
        dashes=True,
        palette=color_palette,
        linewidth=3,
        markersize=10
    )

    # סימון שיא ב-Null Space בלבד
    null_df = df[df['Space'] == 'Null Space']
    for cond in null_df['Condition'].unique():
        cond_data = null_df[null_df['Condition'] == cond]
        if not cond_data.empty:
            peak = cond_data.loc[cond_data['Accuracy'].idxmax()]
            plt.annotate(f"Peak L{int(peak['Layer'])}",
                         xy=(peak['Layer'], peak['Accuracy']),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold')

    plt.title(f"Uncertainty Probing: {model_name}", fontsize=18, fontweight='bold')
    plt.xlabel("Transformer Layer Index", fontsize=14)
    plt.ylabel("Balanced Accuracy", fontsize=14)
    plt.ylim(0.4, 1.05)

    # --- השינוי המבוקש כאן ---
    # loc='lower right' ממקם את המקרא בתוך הגרף בפינה הימנית למטה
    # frameon=True מוסיף מסגרת לבנה סביב המקרא כדי להפריד אותו מהקווים
    plt.legend(title='Subspace & Task', loc='lower right', frameon=True, shadow=True, fontsize=10)

    # tight_layout עכשיו יתחשב בזה שהגרף תופס את כל רוחב התמונה
    plt.tight_layout()

    output_pdf = f"distinct_colors_{model_name.lower().replace(' ', '_')}.pdf"
    plt.savefig(output_pdf, format='pdf')
    print(f"✅ Success: Plot saved as {output_pdf}")
    plt.show()


def plot_all_models_consolidated(model_configs, output_filename="null_space_detection/probing_analysis.pdf"):
    """
    Generates a consolidated academic figure with 3 subplots.
    Solves clipping issues by increasing canvas size and explicit spacing.
    """
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})

    num_models = len(model_configs)
    # Increased height to 9 to prevent legend clipping
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 9), sharey=True)

    if num_models == 1:
        axes = [axes]

    handles, labels = None, None

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

        ax = axes[i]
        unique_conds = sorted(df['Condition'].unique())
        palette = sns.color_palette("husl", n_colors=len(unique_conds))

        sns.lineplot(
            data=df, x='Layer', y='Accuracy', hue='Condition', style='Condition',
            markers=True, dashes=True, palette=palette, linewidth=2.5, markersize=8, ax=ax
        )

        # Annotate peaks specifically for the Null Space
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

        # Capture handles for the shared legend before removing from subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    # Place the legend in a dedicated area at the bottom
    # loc='lower center' with a small y-offset ensures it stays within the figure
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, shadow=True,
               title="Geometric Subspace & Uncertainty Task", title_fontsize=12, fontsize=11)

    plt.suptitle("Comparative Probing Analysis: Disentangling Uncertainty in Geometric Subspaces",
                 fontsize=22, fontweight='bold', y=0.96)

    # Leave 15% space at the bottom (bottom=0.15) for the legend
    plt.tight_layout(rect=[0, 0.15, 1, 0.94])

    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"✅ Final plot saved as: {output_filename}")
    plt.show()


if __name__ == "__main__":
    CONFIGS = [
        ("results/gpt2_multi_layer/all_layers_probing_results.csv",
         "GPT-2 Small"),
        ("results/gemma-2-2b_multi_layer/all_layers_probing_results.csv",
         "Gemma-2-2b"),
        ("results/Llama-3.2-1B_multi_layer/all_layers_probing_results.csv",
         "Llama-3.2-1B")
    ]

    plot_all_models_consolidated(CONFIGS)
