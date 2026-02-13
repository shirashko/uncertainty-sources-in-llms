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
    with open(baseline_path, "r") as f:
        baseline_raw = json.load(f)
    with open(data_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    categories = ["baseline", "epistemic", "aleatoric"]
    storage = {cat: {"orig": [], "null": [], "logits": []} for cat in categories}

    print(f"Extracting activations for {analyzer.model.cfg.model_name}...")

    # Process all samples and project them into identified subspaces
    for item in tqdm(baseline_raw + dataset):
        cat = item.get('type', 'baseline')
        if cat in storage:
            # Extract activation and ensure it's on CPU for Numpy compatibility
            x = analyzer.get_activation(item['prompt'])

            # Project using the analyzer's pre-computed matrices
            x_null = analyzer.project_null(x)
            x_logits = analyzer.project_logits(x)

            # Store as CPU numpy arrays to save GPU/MPS memory
            storage[cat]["orig"].append(x.detach().cpu().numpy())
            storage[cat]["null"].append(x_null.detach().cpu().numpy())
            storage[cat]["logits"].append(x_logits.detach().cpu().numpy())

    all_metrics = []

    # Compare Absolute positions vs. Residual (centered) activations
    for mode in ["Absolute", "Residual"]:
        mode_dir = os.path.join(output_dir, mode.lower())
        os.makedirs(mode_dir, exist_ok=True)

        # Calculate means based on the baseline category for residual subtraction
        base_means = {k: np.mean(storage["baseline"][k], axis=0) for k in ["orig", "null", "logits"]}
        spaces = [("orig", "Original"), ("null", "Null"), ("logits", "Logits")]

        for key, name in spaces:
            if mode == "Absolute":
                vec_list = [np.stack(storage[cat][key]) for cat in categories]
            else:
                # Subtract baseline mean to isolate the specific uncertainty signal
                vec_list = [np.stack(storage[cat][key]) - base_means[key] for cat in categories]

            # Intra-group similarity: How consistent is the uncertainty signal?
            intra = {}
            for i, cat in enumerate(categories):
                sim_m = cosine_similarity(vec_list[i])
                np.fill_diagonal(sim_m, 0)
                intra[cat] = sim_m[sim_m != 0].mean()

            # Inter-group similarity: Does uncertainty overlap with the baseline?
            sim_base_epi = cosine_similarity(vec_list[0], vec_list[1]).mean()
            sim_base_ale = cosine_similarity(vec_list[0], vec_list[2]).mean()
            sim_epi_ale = cosine_similarity(vec_list[1], vec_list[2]).mean()

            # Generate and save PCA visualizations for each subspace/mode
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

    return pd.DataFrame(all_metrics)

def _plot_pca(vec_list, labels, full_name, output_dir):
    all_vecs = np.vstack(vec_list)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1],
                    hue=[l.capitalize() for i, l in enumerate(labels) for _ in range(len(vec_list[i]))], s=80,
                    alpha=0.7)
    plt.title(f"PCA: {full_name.replace('_', ' ')}\nExplained Variance: {np.sum(var) * 100:.1f}%")
    plt.savefig(os.path.join(output_dir, f"pca_{full_name.lower()}.pdf"), bbox_inches='tight')
    plt.close()