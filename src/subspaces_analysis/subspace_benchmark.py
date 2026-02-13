import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def run_triple_experiment(analyzer, data_path, baseline_path, output_dir):
    # Data Loading
    with open(baseline_path, "r") as f:
        baseline_raw = json.load(f)
    with open(data_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    categories = ["baseline", "epistemic", "aleatoric"]
    storage = {cat: {"orig": [], "null": [], "logits": []} for cat in categories}

    print("Collecting activations...")
    for item in baseline_raw:
        x = analyzer.get_activation(item['prompt'])
        storage["baseline"]["orig"].append(x.cpu().numpy())
        storage["baseline"]["null"].append(analyzer.project_null(x).cpu().numpy())
        storage["baseline"]["logits"].append(analyzer.project_logits(x).cpu().numpy())

    for item in tqdm(dataset):
        cat = item.get('type')
        if cat in storage:
            x = analyzer.get_activation(item['prompt'])
            storage[cat]["orig"].append(x.cpu().numpy())
            storage[cat]["null"].append(analyzer.project_null(x).cpu().numpy())
            storage[cat]["logits"].append(analyzer.project_logits(x).cpu().numpy())

    # Metrics Calculation
    all_metrics = []
    spaces = [("orig", "Original_Space"), ("null", "Null_Space"), ("logits", "Logits_Space")]

    for key, name in spaces:
        vec_list = [np.stack(storage[cat][key]) for cat in categories]

        # Intra-group (Cohesion)
        intra = {}
        for i, cat in enumerate(categories):
            sim_m = cosine_similarity(vec_list[i])
            np.fill_diagonal(sim_m, 0)
            intra[cat] = sim_m[sim_m != 0].mean()

        # Inter-group (Separation) - All Pairs
        sim_base_epi = cosine_similarity(vec_list[0], vec_list[1]).mean()
        sim_base_ale = cosine_similarity(vec_list[0], vec_list[2]).mean()
        sim_epi_ale = cosine_similarity(vec_list[1], vec_list[2]).mean()

        # Visualizations
        _plot_pca(vec_list, categories, name, output_dir)

        all_metrics.append({
            "Space": name,
            "Intra_Base": intra["baseline"],
            "Intra_Epi": intra["epistemic"],
            "Intra_Ale": intra["aleatoric"],
            "Inter_Base_Epi": sim_base_epi,
            "Inter_Base_Ale": sim_base_ale,
            "Inter_Epi_Ale": sim_epi_ale
        })

    return pd.DataFrame(all_metrics)


def _plot_pca(vec_list, labels, space_name, output_dir):
    """
    Generates a 2D PCA plot with informative titles including Explained Variance.
    """
    all_vecs = np.vstack(vec_list)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)

    # Calculate total variance explained by the first two components
    var_exp = pca.explained_variance_ratio_
    total_var = np.sum(var_exp) * 100

    plt.figure(figsize=(10, 6))
    plot_labels = []
    for i, label in enumerate(labels):
        plot_labels.extend([label.capitalize()] * len(vec_list[i]))

    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=plot_labels, s=80, alpha=0.7)

    # Informative dynamic title
    plt.title(f"PCA: {space_name.replace('_', ' ')}\n"
              f"Total Explained Variance: {total_var:.1f}% ",
              fontsize=12, pad=15)

    plt.xlabel(f"Principal Component 1 ({var_exp[0] * 100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1] * 100:.1f}%)")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save as PDF for high-quality LaTeX inclusion
    filename = f"pca_{space_name.lower()}.pdf"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()