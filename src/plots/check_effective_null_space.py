# import torch
# import matplotlib.pyplot as plt
# from typing import Tuple, Optional
# from transformer_lens import HookedTransformer
#
#
# class UnembeddingAnalyzer:
#     """
#     Analyzes the singular value spectrum of a model's unembedding matrix
#     to identify the effective null space for mechanistic interpretability.
#     """
#
#     def __init__(self, model_name: str = "gpt2-small", device: Optional[str] = None):
#         self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
#         self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
#         self.d_model = self.model.cfg.d_model
#         self.W_U = self.model.W_U
#
#     def compute_null_projection(self, k: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes the projection matrix onto the effective null space using SVD.
#
#         Args:
#             k: The dimension of the suspected null space (default 12 for GPT-2 Small).
#
#         Returns:
#             P_perp: The [d_model, d_model] projection matrix.
#             S: The full vector of singular values.
#         """
#         # Center W_U: Softmax is translation invariant; centering isolates meaningful directions.
#         W_U_centered = self.W_U - self.W_U.mean(dim=1, keepdim=True)
#
#         # Perform SVD on W_U.T to find directions in the residual stream with minimal output impact.
#         # Shape: W_U.T [d_vocab, d_model] -> U S Vh
#         _, S, Vh = torch.linalg.svd(W_U_centered.T, full_matrices=False)
#
#         # Construct P_perp using the basis of the last k singular vectors.
#         V_null_basis = Vh[-k:]
#         P_perp = V_null_basis.T @ V_null_basis
#
#         return P_perp, S
#
#     def visualize_spectrum(self, S: torch.Tensor, k: int = 12, tail_size: int = 50):
#         """Generates a log-scale plot of the singular value tail."""
#         plt.figure(figsize=(10, 6))
#
#         indices = range(self.d_model - tail_size, self.d_model)
#         values = S.cpu().detach().numpy()[-tail_size:]
#
#         plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Singular Values')
#         plt.axvline(x=self.d_model - k, color='r', linestyle='--', label=f'k={k} Boundary')
#
#         # Precise x-axis formatting to include the final index (d_model - 1)
#         plt.xticks([self.d_model - tail_size, self.d_model - k, self.d_model - 1])
#
#         plt.title(f"Bottom Singular Values of {self.model.cfg.model_name} Unembedding")
#         plt.xlabel("Singular Value Index")
#         plt.ylabel("Magnitude (Log Scale)")
#         plt.yscale('log')
#         plt.legend()
#         plt.grid(True, which="both", ls="-", alpha=0.5)
#         plt.tight_layout()
#         plt.show()
#
#
# if __name__ == "__main__":
#     MODEL_NAME = "gpt2-small"
#     NULL_DIM = 12
#     OUTPUT_FILE = f"results/P_perp_{MODEL_NAME.replace('-', '_')}.pt"
#
#     analyzer = UnembeddingAnalyzer(model_name=MODEL_NAME)
#     P_perp, singular_values = analyzer.compute_null_projection(k=NULL_DIM)
#
#     print(f"--- Analysis for {MODEL_NAME} ---")
#     print(f"Final Singular Value: {singular_values[-1].item():.6e}")
#     print(f"k={NULL_DIM} Singular Value: {singular_values[-NULL_DIM].item():.6e}")
#
#     # Energy Ratio = (Energy in Null Space) / (Total Energy)
#     total_energy = torch.sum(singular_values ** 2)
#     null_space_energy = torch.sum(singular_values[-NULL_DIM:] ** 2)
#     energy_ratio = (null_space_energy / total_energy).item()
#
#     print(f"Null Space Energy Ratio (k={NULL_DIM}): {energy_ratio:.4e}")
#     print(f"Percentage of total variance in null space: {energy_ratio * 100:.8f}%")
#
#     torch.save(P_perp, OUTPUT_FILE)
#     print(f"\nProjection matrix saved to: {OUTPUT_FILE}")
#     analyzer.visualize_spectrum(singular_values, k=NULL_DIM)

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from transformer_lens import HookedTransformer


class DynamicUnembeddingAnalyzer:
    """
    Analyzes the singular value spectrum of a model's unembedding matrix
    to identify the effective null space and compute projection matrices.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading {model_name} onto {self.device}...")

        # Loading in float16 to optimize performance on M4/MPS
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.d_model = self.model.cfg.d_model
        # W_U shape: [d_model, d_vocab]
        self.W_U = self.model.W_U

    def find_optimal_k(self, singular_values: torch.Tensor, threshold: float = 1e-4) -> int:
        """
        Dynamically finds the optimal k (null space dimension) based on the singular value spectrum.
        It identifies components that contribute less than the specified energy threshold.
        """
        S_np = singular_values.cpu().detach().numpy()

        # Energy Method: Cumulative Variance
        # We look for components contributing less than X percent of total variance
        total_variance = np.sum(S_np ** 2)
        # Calculate cumulative variance from the bottom up (smallest to largest)
        cumulative_variance = np.cumsum(S_np[::-1] ** 2) / total_variance

        # Find the index where the tail components exceed the energy threshold
        k_options = np.where(cumulative_variance < threshold)[0]

        # Return the number of dimensions identified as "effective null space"
        return len(k_options) if len(k_options) > 0 else 1

    def compute_analysis(self, threshold: float = 1e-4) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Performs SVD on the centered unembedding matrix and computes the null space projection.
        """
        # Centering W_U: Essential for Gemma models due to LayerNorm/RMSNorm behavior
        W_U_centered = self.W_U - self.W_U.mean(dim=1, keepdim=True)

        # Singular Value Decomposition
        # Shape: W_U_centered.T [d_vocab, d_model] -> U S Vh
        _, S, Vh = torch.linalg.svd(W_U_centered.T, full_matrices=False)

        # Automated k detection
        k_opt = self.find_optimal_k(S, threshold=threshold)

        # Construct the projection matrix (P_perp) onto the bottom k-dimensional subspace
        V_null_basis = Vh[-k_opt:]
        P_perp = V_null_basis.T @ V_null_basis

        return P_perp, k_opt, S

    def visualize_spectrum(self, S: torch.Tensor, k: int, tail_size: int = 100):
        """
        Generates a log-scale plot of the singular value spectrum tail to verify the null space.
        """
        plt.figure(figsize=(10, 6))

        # Focus on the smallest singular values (the tail)
        indices = range(self.d_model - tail_size, self.d_model)
        values = S.cpu().detach().numpy()[-tail_size:]

        plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Singular Values')
        plt.axvline(x=self.d_model - k, color='r', linestyle='--', label=f'Detected Null Space (k={k})')

        plt.title(f"Bottom Singular Values: {self.model.cfg.model_name}")
        plt.xlabel("Singular Value Index")
        plt.ylabel("Magnitude (Log Scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()

        # Save or display result
        plt.savefig(f"spectrum_{self.model.cfg.model_name.replace('/', '_')}.png")
        print(f"Spectrum plot saved as spectrum_{self.model.cfg.model_name.replace('/', '_')}.png")
        plt.show()


if __name__ == "__main__":
    # Settings for your Gemma-2-2B experiment
    MODEL_NAME = "google/gemma-2-2b"

    analyzer = DynamicUnembeddingAnalyzer(model_name=MODEL_NAME)
    P_perp, k_found, singular_values = analyzer.compute_analysis(threshold=1e-4)

    print("\n" + "=" * 50)
    print(f"ANALYSIS COMPLETE for {MODEL_NAME}")
    print(f"Detected Optimal k (Null Space Dim): {k_found}")
    print(f"Residual Stream Dimension (d_model): {analyzer.d_model}")

    # Calculate energy captured in the identified null space
    total_energy = torch.sum(singular_values ** 2)
    null_energy = torch.sum(singular_values[-k_found:] ** 2)
    print(f"Energy Ratio in Null Space: {(null_energy / total_energy).item():.4e}")
    print("=" * 50)

    # Visualize the spectrum to confirm the "elbow" or "cliff" point
    analyzer.visualize_spectrum(singular_values, k=k_found, tail_size=150)

    # Save the resulting projection matrix for downstream uncertainty analysis
    output_path = f"P_perp_{MODEL_NAME.split('/')[-1]}.pt"
    torch.save(P_perp, output_path)
    print(f"Projection matrix saved to: {output_path}")