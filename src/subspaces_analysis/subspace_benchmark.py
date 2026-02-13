import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def run_triple_experiment(analyzer, data_path, baseline_path, output_dir):
    """
    Runs the full subspace analysis: loading data, projecting activations,
    calculating similarity metrics, and generating PCA plots.
    """

    # 1. Correct JSONL Loading
    def load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    print(f"Loading datasets...")
    baseline_raw = load_jsonl(baseline_path)
    dataset_raw = load_jsonl(data_path)

    categories = ["baseline", "epistemic", "aleatoric"]
    # Storage for processed numpy arrays
    storage = {cat: {"orig": [], "null": [], "logits": []} for cat in categories}

    print(f"Extracting and projecting activations for {analyzer.model_tag}...")

    # 2. Process all samples
    # We combine them into one loop but sort them into the correct storage 'bucket'
    for item in tqdm(baseline_raw + dataset_raw):
        cat = item.get('type', 'baseline')
        if cat not in storage:
            continue

        # If the activation is already in the JSON, it's a list; convert to tensor.
        # If not (e.g., if we were running live), we would use analyzer.get_activation.
        if "activation" in item:
            x = torch.tensor(item["activation"]).to(analyzer.device)
        else:
            x = analyzer.get_activation(item['prompt'])

        # Project using the analyzer's pre-computed matrices (P_perp and P_parallel)
        # These are calculated in the UncertaintyAnalyzer class
        x_null = analyzer.project_null(x)
        x_logits = analyzer.project_logits(x)

        # Store as CPU numpy arrays for Scikit-Learn compatibility and memory efficiency
        storage[cat]["orig"].append(x.detach().cpu().numpy())
        storage[cat]["null"].append(x_null.detach().cpu().numpy())
        storage[cat]["logits"].append(x_logits.detach().cpu().numpy())

    all_metrics = []

    # 3. Comparative Analysis: Absolute positions vs. Residual (centered)
    # Residual mode is critical for proving orthogonality in the Null Space
    for mode in ["Absolute", "Residual"]:
        print(f"Analyzing {mode} mode...")
        mode_dir = os.path.join(output_dir, mode.lower())
        os.makedirs(mode_dir, exist_ok=True)

        # Calculate means based on the baseline (C4) for centering
        base_means = {k: np.mean(storage["baseline"][k], axis=0) for k in ["orig", "null", "logits"]}
        spaces = [("orig", "Original"), ("null", "Null"), ("logits", "Logits")]

        for key, name in spaces:
            # Prepare data matrices
            if mode == "Absolute":
                vec_list = [np.stack(storage[cat][key]) for cat in categories]
            else:
                # Residual: Subtract the 'Certainty Anchor' (Baseline Mean)
                vec_list = [np.stack(storage[cat][key]) - base_means[key] for cat in categories]

            # Intra-group similarity: Consistency of the uncertainty signature
            # High Intra_Epi means the uncertainty 'direction' is stable across facts
            intra = {}
            for i, cat in enumerate(categories):
                sim_m = cosine_similarity(vec_list[i])
                np.fill_diagonal(sim_m, 0)  # Don't compare a vector to itself
                intra[cat] = sim_m[sim_m != 0].mean()

            # Inter-group similarity: Overlap between different states
            # We expect Inter_Base_Epi to be near 0 in 'Null' 'Residual' mode (Orthogonality)
            sim_base_epi = cosine_similarity(vec_list[0], vec_list[1]).mean()
            sim_base_ale = cosine_similarity(vec_list[0], vec_list[2]).mean()
            sim_epi_ale = cosine_similarity(vec_list[1], vec_list[2]).mean()

            # 4. Generate Visualizations
            _plot_pca(vec_list, categories, f"{mode}_{name}", mode_dir)

            all_metrics.append({
                "Mode": mode,
                "Space": name,
                "Intra_Base": intra["baseline"],
                "Intra_Epi": intra["epistemic"],
                "Intra_Ale": intra["aleatoric"],
                "Inter_Base_Epi": sim_base_epi,
                "Inter_Base_Ale": sim_base_ale,
                "Inter_Epi_Ale": sim_epi_ale
            })

    # Return summary for the final table
    return pd.DataFrame(all_metrics)


def _plot_pca(vec_list, labels, full_name, output_dir):
    """Generates 2D PCA plots to visualize subspace clustering."""
    all_vecs = np.vstack(vec_list)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))

    # Create the hue labels corresponding to each vector
    hue_labels = []
    for i, l in enumerate(labels):
        hue_labels.extend([l.capitalize()] * len(vec_list[i]))

    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=hue_labels, s=80, alpha=0.7, palette="viridis")

    plt.title(f"PCA Subspace: {full_name.replace('_', ' ')}\nTotal Var Explained: {np.sum(var) * 100:.1f}%")
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(output_dir, f"pca_{full_name.lower()}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()