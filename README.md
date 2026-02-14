# Uncertainty Sources in LLMs: Mechanistic Disentanglement

This repository investigates the internal mechanisms by which Large Language Models (LLMs) represent and regulate different types of uncertainty. Specifically, I distinguish between **Aleatoric uncertainty** (inherent linguistic ambiguity) and **Epistemic uncertainty** (lack of factual knowledge).

This research identifies a "Regulatory Hub" within the **unembedding null space** of the model, where these two sources of uncertainty are disentangled into distinguishable geometric axes.



## Key Findings

- **Structured Null Space:** Uncertainty triggers a highly consistent geometric response in the unembedding null space (Intra-class similarity > 0.9), while certain states remain stochastic.
- **Functional Separability:** Linear probes distinguish between certainty and uncertainty, as well as between Epistemic and Aleatoric sources, with **~90% accuracy** within the null space.
- **Geometric Orthogonality:** The "Detection" axis and "Uncertainty Type" axis are nearly **orthogonal (Cosine Similarity ≈ 0.08)**, suggesting a multi-dimensional control mechanism.
- **Causal Evidence:** Injecting null-space steering vectors successfully flattens the probability distribution (increasing entropy from ~1.8 to ~11.5).


## Methodology

I utilize a "White-Box" approach to uncertainty, focusing on the **Unembedding Null Space**:
1. **Dataset Construction:** Generation of "Knowable" (Baseline), "Unknowable" (Epistemic), and "Ambiguous" (Aleatoric) prompts using existing datasets.
2. **Subspace Projection:** Projecting residual stream activations into the null space of the unembedding matrix $W_U$.
3. **Probing:** Training Logistic Regression probes to verify the linear encoding of uncertainty types.
4. **Causal Steering:** Applying hook-based interventions to steer model confidence by manipulating the null-space norm (mimicking "entropy neurons").



## Installation

```bash
# Clone the repository
git clone [https://github.com/your-repo/uncertainty-sources-in-llms.git](https://github.com/your-repo/uncertainty-sources-in-llms.git)
cd uncertainty-sources-in-llms

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt