import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP
import torch


def analyze_model_behavior(analyzer, prompt, max_new_tokens=5):
    """
    Extracts behavioral uncertainty metrics: Entropy, Top-K probs, and Generation.
    """
    tokens = analyzer.model.to_tokens(prompt)
    logits = analyzer.model(tokens)[:, -1, :] # Last token logits

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).item()

    top_k_vals, top_k_idxs = torch.topk(probs, k=5)
    top_k_list = [
        {"token": analyzer.model.to_string(idx), "prob": round(prob.item(), 4)}
        for prob, idx in zip(top_k_vals[0], top_k_idxs[0])
    ]

    generation = analyzer.model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        verbose=False,
        prepend_bos=True,
        stop_at_eos=True
    )
    continuation = generation[len(prompt):] if generation.startswith(prompt) else generation

    return {
        "entropy": round(entropy, 4),
        "top_k": top_k_list,
        "top_1_prob": top_k_list[0]["prob"],
        "continuation": continuation.strip()
    }


def run_triple_experiment(analyzer, data_path, output_dir, layers_to_test=None):
    """
    Runs full subspace analysis using a unified Triplet dataset (JSONL).
    Includes Behavioral Analysis (Entropy, Top-K, Generation) linked to Geometric Data.
    """

    def load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    if layers_to_test is None:
        layers_to_test = [analyzer.model.cfg.n_layers - 1]

    print(f"📂 Loading unified triplet dataset from: {data_path}")
    raw_triplets = load_jsonl(data_path)
    categories = ["baseline", "epistemic", "aleatoric"]

    # --- PHASE 1: BEHAVIORAL ANALYSIS (Run once for the whole experiment) ---
    print(f"📊 Running Behavioral Analysis (Entropy, Top-K, Generation)...")
    behavioral_cache = {}  # Map: (triplet_id, category) -> behavior_metrics

    for triplet in tqdm(raw_triplets, desc="Behavioral Analysis"):
        t_id = triplet.get('id')
        behavioral_cache[t_id] = {}
        for cat in categories:
            prompt = triplet.get(cat)
            if prompt:
                behavioral_cache[t_id][cat] = analyze_model_behavior(analyzer, prompt)

    all_metrics = []
    all_layer_storage = {}

    # --- PHASE 2: GEOMETRIC ANALYSIS (Per Layer) ---
    for layer_idx in layers_to_test:
        print(f"\n--- Processing Layer {layer_idx} ---")
        layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        # Initialize storage
        storage = {cat: {"orig": [], "null": [], "logits": []} for cat in categories}
        metadata_by_cat = {cat: [] for cat in categories}

        for triplet in tqdm(raw_triplets, desc=f"Extracting L{layer_idx}"):
            t_id = triplet.get('id')
            for cat in categories:
                prompt = triplet.get(cat)
                if not prompt:
                    continue

                # 1. Geometric Extraction
                x = analyzer.get_activation(prompt, layer_idx=layer_idx)
                x_null = analyzer.project_null(x)
                x_logits = analyzer.project_logits(x)

                storage[cat]["orig"].append(x.detach().cpu().numpy())
                storage[cat]["null"].append(x_null.detach().cpu().numpy())
                storage[cat]["logits"].append(x_logits.detach().cpu().numpy())

                # 2. Enrich Metadata with Behavioral Results from Cache
                behav = behavioral_cache[t_id][cat]
                meta_item = {
                    **triplet,
                    "type": cat,
                    "entropy": behav["entropy"],
                    "top_1_prob": behav["top_1_prob"],
                    "continuation": behav["continuation"],
                    "top_k_json": json.dumps(behav["top_k"])  # Serializable for CSV
                }
                metadata_by_cat[cat].append(meta_item)

        all_layer_storage[layer_idx] = storage
        spaces = [("orig", "Original"), ("null", "Null"), ("logits", "Logits")]

        # Step 2: Layer Analysis (PCA & Metrics)
        for key, name in spaces:
            # Stack vectors
            vec_list_abs = [np.stack(storage[cat][key]) for cat in categories]
            flat_metadata = [m for cat in categories for m in metadata_by_cat[cat]]

            all_vecs_abs = np.vstack(vec_list_abs)
            labels = [i for i, cat in enumerate(categories) for _ in range(len(vec_list_abs[i]))]

            # 2.1 PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(all_vecs_abs)

            # 2.2 Save Interpretability CSV (Now with Entropy and Generation!)
            df_interpret = pd.DataFrame(flat_metadata)
            df_interpret['PC1'] = coords[:, 0]
            df_interpret['PC2'] = coords[:, 1]
            df_interpret.to_csv(os.path.join(layer_dir, f"interpret_{name.lower()}.csv"), index=False)

            # 2.3 Visualizations
            _plot_pca(coords, labels, categories, f"L{layer_idx}_{name}", layer_dir, pca.explained_variance_ratio_)
            _plot_umap(vec_list_abs, categories, f"L{layer_idx}_{name}", layer_dir)

            # 2.4 Metric Calculation (Silhouette & Cosine)
            base_mean = np.mean(storage["baseline"][key], axis=0)
            for mode in ["Absolute", "Residual"]:
                current_vec_list = vec_list_abs if mode == "Absolute" else [v - base_mean for v in vec_list_abs]
                current_all_vecs = np.vstack(current_vec_list)

                sil_score = round(silhouette_score(current_all_vecs, labels), 2)

                intra = {}
                for i, cat in enumerate(categories):
                    sim_m = cosine_similarity(current_vec_list[i])
                    np.fill_diagonal(sim_m, 0)
                    intra[cat] = round(sim_m[sim_m != 0].mean(), 2) if len(sim_m) > 1 else 1.0

                all_metrics.append({
                    "Layer": layer_idx, "Mode": mode, "Space": name,
                    "Silhouette_Score": sil_score,
                    "Intra_Base": intra["baseline"],
                    "Intra_Epi": intra["epistemic"],
                    "Intra_Ale": intra["aleatoric"],
                    "Inter_Base_Epi": round(cosine_similarity(current_vec_list[0], current_vec_list[1]).mean(), 2),
                    "Inter_Base_Ale": round(cosine_similarity(current_vec_list[0], current_vec_list[2]).mean(), 2),
                    "Inter_Epi_Ale": round(cosine_similarity(current_vec_list[1], current_vec_list[2]).mean(), 2)
                })

    return pd.DataFrame(all_metrics), all_layer_storage

def _plot_pca(coords, labels_idx, categories, full_name, output_dir, var_ratio):
    plt.figure(figsize=(10, 6))
    hue_labels = [categories[i].capitalize() for i in labels_idx]
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=hue_labels, s=80, alpha=0.7, palette="viridis")
    plt.title(f"PCA: {full_name.replace('_', ' ')}\nVar: {np.sum(var_ratio) * 100:.2f}%")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, f"pca_{full_name.lower()}.png"), bbox_inches='tight', dpi=300)
    plt.close()


def _plot_umap(vec_list, labels, full_name, output_dir):
    all_vecs = np.vstack(vec_list)
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    coords = reducer.fit_transform(all_vecs)
    plt.figure(figsize=(10, 6))
    hue_labels = [l.capitalize() for i, l in enumerate(labels) for _ in range(len(vec_list[i]))]
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=hue_labels, s=80, alpha=0.7, palette="magma")
    plt.title(f"UMAP: {full_name.replace('_', ' ')}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, f"umap_{full_name.lower()}.png"), bbox_inches='tight', dpi=300)
    plt.close()