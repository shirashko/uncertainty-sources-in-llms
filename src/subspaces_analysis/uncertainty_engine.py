import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class UncertaintyAnalyzer:
    def __init__(self, model_name="gpt2", k=12):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.k = k
        # Compute the projections once at initialization
        self.P_perp = self._compute_null_space_projection()
        self.P_parallel = torch.eye(self.P_perp.shape[0]).to(self.device) - self.P_perp

    def _compute_null_space_projection(self):
        """Identifies the effective null space of the unembedding matrix via SVD."""
        W_U = self.model.transformer.wte.weight.detach().cpu()
        W_U_centered = W_U - W_U.mean(dim=0)
        _, _, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
        # Last k vectors represent dimensions with minimal impact on output logits
        V_null = Vh[-self.k:, :].to(self.device)
        return V_null.t() @ V_null

    @torch.no_grad()
    def get_activation(self, prompt):
        """Extracts the hidden state of the final layer for the last token."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :]

    def project_null(self, vector):
        """Projection into the Null Space (P_perp)."""
        return vector @ self.P_perp

    def project_logits(self, vector):
        """Projection into the Logits Space (I - P_perp)."""
        return vector @ self.P_parallel