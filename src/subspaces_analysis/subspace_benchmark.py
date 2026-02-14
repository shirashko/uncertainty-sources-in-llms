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


def run_triple_experiment(analyzer, data_path, baseline_path, output_dir, layers_to_test=None):
    """
    Runs the full subspace analysis across multiple layers.
    Identifies if uncertainty signals emerge in early, middle, or late stages.
    """

    def load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    # Default to analyzing only the final layer if no layers are specified
    if layers_to_test is None:
        layers_to_test = [analyzer.model.cfg.n_layers - 1]

    print(f"Loading datasets...")
    baseline_raw = load_jsonl(baseline_path)
    dataset_raw = load_jsonl(data_path)
    categories = ["baseline", "epistemic", "aleatoric"]

    all_metrics = []
    # Key: layer_idx, Value: storage dict for that layer
    all_layer_storage = {}

    # --- STEP 1: LAYER-WISE DATA EXTRACTION ---
    for layer_idx in layers_to_test:
        print(f"\n--- Processing Layer {layer_idx} ---")
        storage = {cat: {"orig": [], "null": [], "logits": []} for cat in categories}

        for item in tqdm(baseline_raw + dataset_raw, desc=f"Extracting L{layer_idx}"):
            cat = item.get('type', 'baseline')
            if cat not in storage:
                continue

            # Extract activation at the specific layer for the terminal token
            x = analyzer.get_activation(item['prompt'], layer_idx=layer_idx)

            # Project into pre-computed Unembedding subspaces
            x_null = analyzer.project_null(x)
            x_logits = analyzer.project_logits(x)

            storage[cat]["orig"].append(x.detach().cpu().numpy())
            storage[cat]["null"].append(x_null.detach().cpu().numpy())
            storage[cat]["logits"].append(x_logits.detach().cpu().numpy())

        all_layer_storage[layer_idx] = storage

        # --- STEP 2: GEOMETRIC ANALYSIS PER LAYER ---
        for mode in ["Absolute", "Residual"]:
            mode_dir = os.path.join(output_dir, f"layer_{layer_idx}", mode.lower())
            os.makedirs(mode_dir, exist_ok=True)

            # Center relative to the Baseline (Certainty) mean for this specific layer
            base_means = {k: np.mean(storage["baseline"][k], axis=0) for k in ["orig", "null", "logits"]}
            spaces = [("orig", "Original"), ("null", "Null"), ("logits", "Logits")]

            for key, name in spaces:
                if mode == "Absolute":
                    vec_list = [np.stack(storage[cat][key]) for cat in categories]
                else:
                    # Subtract Baseline mean to isolate the 'uncertainty steering vector'
                    vec_list = [np.stack(storage[cat][key]) - base_means[key] for cat in categories]

                # Intra-group similarity: Measures the consistency of the uncertainty signature
                intra = {}
                for i, cat in enumerate(categories):
                    sim_m = cosine_similarity(vec_list[i])
                    np.fill_diagonal(sim_m, 0)
                    intra[cat] = sim_m[sim_m != 0].mean()

                # Inter-group similarity: Measures overlap/orthogonality between states
                sim_base_epi = cosine_similarity(vec_list[0], vec_list[1]).mean()
                sim_base_ale = cosine_similarity(vec_list[0], vec_list[2]).mean()
                sim_epi_ale = cosine_similarity(vec_list[1], vec_list[2]).mean()

                # Visualize PCA for this specific layer/space
                _plot_pca(vec_list, categories, f"L{layer_idx}_{mode}_{name}", mode_dir)

                all_metrics.append({
                    "Layer": layer_idx,
                    "Mode": mode,
                    "Space": name,
                    "Intra_Base": intra["baseline"],
                    "Intra_Epi": intra["epistemic"],
                    "Intra_Ale": intra["aleatoric"],
                    "Inter_Base_Epi": sim_base_epi,
                    "Inter_Base_Ale": sim_base_ale,
                    "Inter_Epi_Ale": sim_epi_ale
                })

    # Return summary DataFrame and storage of the last processed layer for backward compatibility
    return pd.DataFrame(all_metrics), all_layer_storage[layers_to_test[-1]]

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