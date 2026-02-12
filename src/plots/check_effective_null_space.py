import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from transformer_lens import HookedTransformer


class UnembeddingAnalyzer:
    """
    Analyzes the singular value spectrum of a model's unembedding matrix
    to identify the effective null space for mechanistic interpretability.
    """

    def __init__(self, model_name: str = "gpt2-small", device: Optional[str] = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.d_model = self.model.cfg.d_model
        self.W_U = self.model.W_U

    def compute_null_projection(self, k: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the projection matrix onto the effective null space using SVD.

        Args:
            k: The dimension of the suspected null space (default 12 for GPT-2 Small).

        Returns:
            P_perp: The [d_model, d_model] projection matrix.
            S: The full vector of singular values.
        """
        # Center W_U: Softmax is translation invariant; centering isolates meaningful directions.
        W_U_centered = self.W_U - self.W_U.mean(dim=1, keepdim=True)

        # Perform SVD on W_U.T to find directions in the residual stream with minimal output impact.
        # Shape: W_U.T [d_vocab, d_model] -> U S Vh
        _, S, Vh = torch.linalg.svd(W_U_centered.T, full_matrices=False)

        # Construct P_perp using the basis of the last k singular vectors.
        V_null_basis = Vh[-k:]
        P_perp = V_null_basis.T @ V_null_basis

        return P_perp, S

    def visualize_spectrum(self, S: torch.Tensor, k: int = 12, tail_size: int = 50):
        """Generates a log-scale plot of the singular value tail."""
        plt.figure(figsize=(10, 6))

        indices = range(self.d_model - tail_size, self.d_model)
        values = S.cpu().detach().numpy()[-tail_size:]

        plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Singular Values')
        plt.axvline(x=self.d_model - k, color='r', linestyle='--', label=f'k={k} Boundary')

        # Precise x-axis formatting to include the final index (d_model - 1)
        plt.xticks([self.d_model - tail_size, self.d_model - k, self.d_model - 1])

        plt.title(f"Bottom Singular Values of {self.model.cfg.model_name} Unembedding")
        plt.xlabel("Singular Value Index")
        plt.ylabel("Magnitude (Log Scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    MODEL_NAME = "gpt2-small"
    NULL_DIM = 12
    OUTPUT_FILE = f"results/P_perp_{MODEL_NAME.replace('-', '_')}.pt"

    analyzer = UnembeddingAnalyzer(model_name=MODEL_NAME)
    P_perp, singular_values = analyzer.compute_null_projection(k=NULL_DIM)

    print(f"--- Analysis for {MODEL_NAME} ---")
    print(f"Final Singular Value: {singular_values[-1].item():.6e}")
    print(f"k={NULL_DIM} Singular Value: {singular_values[-NULL_DIM].item():.6e}")

    # Energy Ratio = (Energy in Null Space) / (Total Energy)
    total_energy = torch.sum(singular_values ** 2)
    null_space_energy = torch.sum(singular_values[-NULL_DIM:] ** 2)
    energy_ratio = (null_space_energy / total_energy).item()

    print(f"Null Space Energy Ratio (k={NULL_DIM}): {energy_ratio:.4e}")
    print(f"Percentage of total variance in null space: {energy_ratio * 100:.8f}%")

    torch.save(P_perp, OUTPUT_FILE)
    print(f"\nProjection matrix saved to: {OUTPUT_FILE}")
    analyzer.visualize_spectrum(singular_values, k=NULL_DIM)