import torch
from transformer_lens import HookedTransformer


class UncertaintyAnalyzer:
    def __init__(self, model_name="google/gemma-2-2b", k=5):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load model
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.k = k

        # Determine the correct hook name for the final normalization layer
        # GPT-2 uses 'ln_f', Gemma uses 'ln_final'
        self.final_ln_name = "ln_f" if "gpt2" in model_name.lower() else "ln_final"

        # Pre-compute projection matrices
        self.P_perp = self._compute_null_space_projection()
        self.P_parallel = torch.eye(self.model.cfg.d_model).to(self.device) - self.P_perp

    def _compute_null_space_projection(self):
        """Computes the projection matrix for the effective null space."""
        W_U = self.model.W_U.detach().cpu()

        # Centering logic: Gemma benefits from row-centering (dim=1)
        # while for GPT-2 it helps isolate the output-impacting directions
        W_U_centered = W_U - W_U.mean(dim=1, keepdim=True)

        # Perform SVD to find the effective null space directions
        _, _, Vh = torch.linalg.svd(W_U_centered.T, full_matrices=False)

        V_null = Vh[-self.k:, :].to(self.device)
        return V_null.t() @ V_null

    @torch.no_grad()
    def get_activation(self, prompt):
        """Extracts the final normalized residual stream activation."""
        _, cache = self.model.run_with_cache(prompt)

        # Use the dynamically determined hook name
        hook_key = f"{self.final_ln_name}.hook_normalized"
        return cache[hook_key][0, -1, :]

    def project_null(self, vector):
        """Projects vector into the Null Space (P_perp)."""
        return vector.to(self.device) @ self.P_perp

    def project_logits(self, vector):
        """Projects vector into the Logits Space (I - P_perp)."""
        return vector.to(self.device) @ self.P_parallel