# import torch
# import json
# import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import Dict
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# class UncertaintyAnalyzer:
#     def __init__(self, model_name: str = "gpt2", k: int = 12):
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()
#
#         self.k = k
#         self.P_perp = self._compute_null_space_projection()
#         self.x_base = None
#
#     def _compute_null_space_projection(self) -> torch.Tensor:
#         """Identifies the effective null space of the unembedding matrix."""
#         # W_U shape is (vocab_size, d_model)
#         W_U = self.model.transformer.wte.weight.detach().cpu()
#         W_U_centered = W_U - W_U.mean(dim=0)
#
#         # SVD: Vh contains the directions in the residual stream
#         _, _, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
#
#         # Bottom k vectors represent the 'unembedding-agnostic' dimensions
#         V_null = Vh[-self.k:, :].to(self.device)
#         return V_null.t() @ V_null
#
#     def load_baseline(self, path: str):
#         """Loads the geometric centroid of verified certainty."""
#         self.x_base = torch.load(path).to(self.device)
#
#     def get_saturation(self, vector: torch.Tensor) -> float:
#         """Calculates what percentage of a vector's norm resides in the null space."""
#         projected = vector @ self.P_perp
#         return (torch.norm(projected) / torch.norm(vector)).item() * 100
#
#     @torch.no_grad()
#     def process_item(self, prompt: str) -> Dict[str, torch.Tensor]:
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs, output_hidden_states=True)
#         # Extract final layer activation for the last token
#         x_i = outputs.hidden_states[-1][0, -1, :]
#         return {"x_i": x_i, "delta_x": x_i - self.x_base}
#
#
# def run_experiment(data_path: str, baseline_path: str):
#     analyzer = UncertaintyAnalyzer()
#     analyzer.load_baseline(baseline_path)
#
#     base_saturation = analyzer.get_saturation(analyzer.x_base)
#     print(f"SATURATION OF THE CERTAINTY BASELINE (x_base): {base_saturation:.2f}%")
#
#     # FIX: Use string keys to match the 'type' field in your JSONL
#     results = {
#         "epistemic": {"orig_ratios": [], "diff_ratios": [], "diff_vecs": []},
#         "aleatoric": {"orig_ratios": [], "diff_ratios": [], "diff_vecs": []}
#     }
#
#     with open(data_path, "r") as f:
#         dataset = [json.loads(line) for line in f]
#
#     print(f"Running analysis on {len(dataset)} samples...")
#     for item in tqdm(dataset):
#         activations = analyzer.process_item(item['prompt'])
#         label = item['type'] # This will be "epistemic" or "aleatoric"
#
#         if label not in results:
#             continue
#
#         # Calculate saturation for the Baseline Check (Original vs Differential)
#         results[label]["orig_ratios"].append(analyzer.get_saturation(activations["x_i"]))
#
#         # Calculate saturation and store vector for Differential Analysis
#         diff_proj = activations["delta_x"] @ analyzer.P_perp
#         results[label]["diff_ratios"].append(analyzer.get_saturation(activations["delta_x"]))
#         results[label]["diff_vecs"].append(diff_proj.cpu().numpy())
#
#     _print_report(results)
#
#
# def _print_report(results: Dict):
#     # Updated to access by string keys
#     epi = results["epistemic"]
#     ale = results["aleatoric"]
#
#     if not epi["diff_vecs"] or not ale["diff_vecs"]:
#         print("Error: Missing data for one of the categories. Check your dataset types.")
#         return
#
#     # Compute Intra/Inter Class Similarities
#     epi_vecs = np.stack(epi["diff_vecs"])
#     ale_vecs = np.stack(ale["diff_vecs"])
#
#     sim_epi = cosine_similarity(epi_vecs).mean()
#     sim_ale = cosine_similarity(ale_vecs).mean()
#     sim_inter = cosine_similarity(epi_vecs, ale_vecs).mean()
#
#     print("\n" + "=" * 65)
#     print(f"{'CATEGORY':<12} | {'ORIGINAL SAT%':<15} | {'DIFF SAT%':<15} | {'INTRA-SIM':<10}")
#     print("-" * 65)
#     print(
#         f"{'Epistemic':<12} | {np.mean(epi['orig_ratios']):.2f}%{'':<10} | {np.mean(epi['diff_ratios']):.2f}%{'':<10} | {sim_epi:.4f}")
#     print(
#         f"{'Aleatoric':<12} | {np.mean(ale['orig_ratios']):.2f}%{'':<10} | {np.mean(ale['diff_ratios']):.2f}%{'':<10} | {sim_ale:.4f}")
#     print("-" * 65)
#     print(f"Inter-class Similarity (Epi <-> Ale): {sim_inter:.4f}")
#     print("=" * 65)
#
#     plot_similarity_heatmap(epi_vecs, ale_vecs)
#
#
# def plot_similarity_heatmap(epi_vecs, ale_vecs):
#     """
#     Generates a professional heatmap of the Cosine Similarity matrix
#     between and within uncertainty categories.
#     """
#     combined_vecs = np.vstack([epi_vecs, ale_vecs])
#     full_sim_matrix = cosine_similarity(combined_vecs)
#
#     plt.figure(figsize=(10, 8))
#
#     n_epi = len(epi_vecs)
#     n_ale = len(ale_vecs)
#
#     ax = sns.heatmap(
#         full_sim_matrix,
#         cmap="YlGnBu",
#         xticklabels=False,
#         yticklabels=False,
#         cbar_kws={'label': 'Cosine Similarity'}
#     )
#
#     plt.axvline(x=n_epi, color='red', linestyle='--', linewidth=1.5)
#     plt.axhline(y=n_epi, color='red', linestyle='--', linewidth=1.5)
#
#     plt.text(n_epi / 2, -5, "Epistemic", ha='center', va='center', fontweight='bold')
#     plt.text(n_epi + (n_ale / 2), -5, "Aleatoric", ha='center', va='center', fontweight='bold')
#     plt.text(-10, n_epi / 2, "Epistemic", ha='center', va='center', rotation=90, fontweight='bold')
#     plt.text(-10, n_epi + (n_ale / 2), "Aleatoric", ha='center', va='center', rotation=90, fontweight='bold')
#
#     plt.title("Cosine Similarity of Uncertainty Projections in Null Space", pad=40)
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     run_experiment("data/uncertainty_study_dataset.jsonl", "data/common_certainty_baseline.pt")
# import torch
# import json
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA  # וודאי שהייבוא נראה בדיוק כך
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from tqdm import tqdm
# from typing import Dict, List
#
# class UncertaintyAnalyzer:
#     """מנתח אי-ודאות מכניסטי עבור GPT-2."""
#     def __init__(self, model_name: str = "gpt2", k: int = 12):
#         self.device = "mps" if torch.backends.mps.is_available() else "cpu"
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()
#         self.k = k
#         self.P_perp = self._compute_null_space_projection()
#
#     def _compute_null_space_projection(self) -> torch.Tensor:
#         """חישוב ה-Null Space האפקטיבי של ה-Unembedding."""
#         W_U = self.model.transformer.wte.weight.detach().cpu()
#         W_U_centered = W_U - W_U.mean(dim=0)
#         _, _, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
#         # שימוש ב-k הוקטורים האחרונים (12 עבור GPT-2 Small)
#         V_null = Vh[-self.k:, :].to(self.device)
#         return V_null.t() @ V_null
#
#     def get_saturation(self, vector: torch.Tensor) -> float:
#         """חישוב אחוז ה'רוויה' של הוקטור בתוך ה-Null Space."""
#         projected = vector @ self.P_perp
#         return (torch.norm(projected) / torch.norm(vector)).item() * 100
#
#     @torch.no_grad()
#     def get_activation(self, prompt: str) -> torch.Tensor:
#         """חילוץ אקטיבציה של שכבה אחרונה בטוקן האחרון."""
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs, output_hidden_states=True)
#         return outputs.hidden_states[-1][0, -1, :]
#
# def plot_uncertainty_pca(vec_list: List[np.ndarray], labels: List[str]):
#     """הצגת ויזואליזציה של PCA להפרדה גיאומטרית ב-Null Space."""
#     all_vecs = np.vstack(vec_list)
#     plot_labels = []
#     for i, label in enumerate(labels):
#         plot_labels.extend([label.capitalize()] * len(vec_list[i]))
#
#     pca = PCA(n_components=2)
#     components = pca.fit_transform(all_vecs)
#
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(
#         x=components[:, 0],
#         y=components[:, 1],
#         hue=plot_labels,
#         style=plot_labels,
#         palette="viridis",
#         s=100,
#         alpha=0.7
#     )
#
#     var_exp = pca.explained_variance_ratio_
#     plt.xlabel(f"PC1 ({var_exp[0] * 100:.1f}% Variance)")
#     plt.ylabel(f"PC2 ({var_exp[1] * 100:.1f}% Variance)")
#     plt.title("PCA Projection of Latent States in Null Space", pad=20)
#     plt.grid(True, alpha=0.3)
#     plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()
#
# def _print_tri_report(results: Dict):
#     """הדפסת דוח תוצאות מפורט וויזואליזציות."""
#     cats = ["baseline", "epistemic", "aleatoric"]
#     plot_vecs = []
#     category_sizes = []
#
#     print("\n" + "=" * 80)
#     print(f"{'CATEGORY':<15} | {'COUNT':<8} | {'AVG SAT%':<12} | {'INTRA-SIM':<10}")
#     print("-" * 80)
#
#     for cat in cats:
#         vecs = np.stack(results[cat]["null_vecs"])
#         plot_vecs.append(vecs)
#         category_sizes.append(len(vecs))
#         avg_sat = np.mean(results[cat]["saturations"])
#         intra_sim = cosine_similarity(vecs).mean()
#         print(f"{cat.capitalize():<15} | {len(vecs):<8} | {avg_sat:.2f}%{'':<5} | {intra_sim:.4f}")
#
#     print("-" * 80)
#     sim_base_epi = cosine_similarity(plot_vecs[0], plot_vecs[1]).mean()
#     sim_base_ale = cosine_similarity(plot_vecs[0], plot_vecs[2]).mean()
#     sim_epi_ale = cosine_similarity(plot_vecs[1], plot_vecs[2]).mean()
#
#     print(f"Inter-Class Similarity:")
#     print(f"  > Baseline <-> Epistemic: {sim_base_epi:.4f}")
#     print(f"  > Baseline <-> Aleatoric: {sim_base_ale:.4f}")
#     print(f"  > Epistemic <-> Aleatoric: {sim_epi_ale:.4f}")
#     print("=" * 80)
#
#     # ויזואליזציה
#     _plot_tri_heatmap(plot_vecs, cats, category_sizes)
#     plot_uncertainty_pca(plot_vecs, cats)
#
# def _plot_tri_heatmap(vec_list, labels, sizes):
#     """מייצר מפת חום של דמיון קוסינוס."""
#     combined_vecs = np.vstack(vec_list)
#     full_sim_matrix = cosine_similarity(combined_vecs)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(full_sim_matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False,
#                 cbar_kws={'label': 'Cosine Similarity (Null Space Projections)'})
#
#     current_pos = 0
#     for i, size in enumerate(sizes):
#         current_pos += size
#         if i < len(sizes) - 1:
#             plt.axvline(x=current_pos, color='red', linestyle='--', linewidth=1.5)
#             plt.axhline(y=current_pos, color='red', linestyle='--', linewidth=1.5)
#
#     start_pos = 0
#     for i, size in enumerate(sizes):
#         mid_pos = start_pos + size / 2
#         plt.text(mid_pos, -10, labels[i].capitalize(), ha='center', va='center', fontweight='bold')
#         plt.text(-35, mid_pos, labels[i].capitalize(), ha='center', va='center', rotation=90, fontweight='bold')
#         start_pos += size
#     plt.title("Tri-Category Null Space Similarity Matrix", pad=40)
#     plt.tight_layout()
#     plt.show()
#
# def run_tri_category_experiment(data_path: str, baseline_path: str):
#     """הרצת הניסוי המלא."""
#     analyzer = UncertaintyAnalyzer()
#     with open("data/baseline_inputs.json", "r") as f:
#         baseline_raw = json.load(f)
#
#     categories = ["baseline", "epistemic", "aleatoric"]
#     results = {cat: {"saturations": [], "null_vecs": []} for cat in categories}
#
#     print("Processing Baseline samples...")
#     for item in baseline_raw:
#         x = analyzer.get_activation(item['prompt'])
#         results["baseline"]["saturations"].append(analyzer.get_saturation(x))
#         null_proj = x @ analyzer.P_perp
#         results["baseline"]["null_vecs"].append(null_proj.cpu().numpy())
#
#     with open(data_path, "r") as f:
#         dataset = [json.loads(line) for line in f]
#
#     print(f"Processing experimental samples...")
#     for item in tqdm(dataset):
#         x = analyzer.get_activation(item['prompt'])
#         cat = item.get('type')
#         if cat in results:
#             results[cat]["saturations"].append(analyzer.get_saturation(x))
#             null_proj = x @ analyzer.P_perp
#             results[cat]["null_vecs"].append(null_proj.cpu().numpy())
#
#     _print_tri_report(results)
#
# if __name__ == "__main__":
#     run_tri_category_experiment("data/uncertainty_study_dataset.jsonl", "data/common_certainty_baseline.pt")
import torch
import json
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


class UncertaintyAnalyzer:
    def __init__(self, model_name: str = "gpt2", k: int = 12):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.k = k
        # P_perp projects onto the effective Null Space
        self.P_perp = self._compute_null_space_projection()
        # P_parallel projects onto the semantic/Logits space (Orthogonal Complement)
        self.P_parallel = torch.eye(self.P_perp.shape[0]).to(self.device) - self.P_perp

    def _compute_null_space_projection(self) -> torch.Tensor:
        """Identifies the effective null space of the unembedding matrix via SVD."""
        W_U = self.model.transformer.wte.weight.detach().cpu()
        W_U_centered = W_U - W_U.mean(dim=0)
        _, _, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
        # Last k vectors represent dimensions with minimal impact on output logits
        V_null = Vh[-self.k:, :].to(self.device)
        return V_null.t() @ V_null

    @torch.no_grad()
    def get_activation(self, prompt: str) -> torch.Tensor:
        """Extracts the final hidden state for the last token in the prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :]

    def project_null(self, vector: torch.Tensor) -> torch.Tensor:
        """Projection into the Null Space (Uncertainty signal)."""
        return vector @ self.P_perp

    def project_logits(self, vector: torch.Tensor) -> torch.Tensor:
        """Projection into the Logits Space (Semantic/Predictive signal)."""
        return vector @ self.P_parallel


def run_triple_experiment(data_path: str):
    analyzer = UncertaintyAnalyzer()

    with open("data/baseline_inputs.json", "r") as f:
        baseline_raw = json.load(f)
    with open(data_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    categories = ["baseline", "epistemic", "aleatoric"]
    # Dictionary to store vectors for all three analyzed spaces
    results = {cat: {"orig_vecs": [], "null_vecs": [], "logits_vecs": []} for cat in categories}

    print("Extracting and projecting activations...")
    # Process high-certainty baseline samples
    for item in baseline_raw:
        x = analyzer.get_activation(item['prompt'])
        results["baseline"]["orig_vecs"].append(x.cpu().numpy())
        results["baseline"]["null_vecs"].append(analyzer.project_null(x).cpu().numpy())
        results["baseline"]["logits_vecs"].append(analyzer.project_logits(x).cpu().numpy())

    # Process experimental uncertainty samples (Epistemic and Aleatoric)
    for item in tqdm(dataset):
        x = analyzer.get_activation(item['prompt'])
        cat = item.get('type')
        if cat in results:
            results[cat]["orig_vecs"].append(x.cpu().numpy())
            results[cat]["null_vecs"].append(analyzer.project_null(x).cpu().numpy())
            results[cat]["logits_vecs"].append(analyzer.project_logits(x).cpu().numpy())

    # Define the spaces to be analyzed sequentially
    spaces = [
        ("orig_vecs", "Original Space (Full Residual Stream)"),
        ("null_vecs", "Null Space (P_perp)"),
        ("logits_vecs", "Logits Space (I - P_perp)")
    ]

    for key, name in spaces:
        print(f"\n" + "=" * 60)
        print(f"### ANALYSIS: {name.upper()} ###")
        print("=" * 60)
        _analyze_space(results, key, name)


def _analyze_space(results: Dict, vec_key: str, space_name: str):
    cats = ["baseline", "epistemic", "aleatoric"]
    plot_vecs = [np.stack(results[cat][vec_key]) for cat in cats]

    # 1. Intra-group similarity: measures how cohesive each category is
    print("\n[INTRA-GROUP SIMILARITY (Cohesion)]")
    for i, cat in enumerate(cats):
        sim_matrix = cosine_similarity(plot_vecs[i])
        # Mask diagonal to exclude self-similarity (1.0)
        mask = np.ones(sim_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        intra_sim = sim_matrix[mask].mean()
        print(f"  > {cat.capitalize():<10}: {intra_sim:.4f}")

    # 2. Inter-group similarity: measures the distance between categories
    print("\n[INTER-GROUP SIMILARITY (Separation)]")
    sim_base_epi = cosine_similarity(plot_vecs[0], plot_vecs[1]).mean()
    sim_base_ale = cosine_similarity(plot_vecs[0], plot_vecs[2]).mean()
    sim_epi_ale = cosine_similarity(plot_vecs[1], plot_vecs[2]).mean()

    print(f"  > Baseline  <-> Epistemic: {sim_base_epi:.4f}")
    print(f"  > Baseline  <-> Aleatoric: {sim_base_ale:.4f}")
    print(f"  > Epistemic <-> Aleatoric: {sim_epi_ale:.4f}")
    print("-" * 60)

    # 3. PCA Visualization for the current space
    _plot_space_pca(plot_vecs, cats, space_name)


def _plot_space_pca(vec_list, labels, space_name):
    """Generates a 2D PCA plot to visualize geometric clustering."""
    all_vecs = np.vstack(vec_list)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)

    plot_labels = []
    for i, label in enumerate(labels):
        plot_labels.extend([label.capitalize()] * len(vec_list[i]))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=plot_labels, palette="viridis", s=80, alpha=0.7)

    var_exp = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var_exp[0] * 100:.1f}% Variance)")
    plt.ylabel(f"PC2 ({var_exp[1] * 100:.1f}% Variance)")
    plt.title(f"PCA Projection: {space_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_triple_experiment("data/uncertainty_study_dataset.jsonl")