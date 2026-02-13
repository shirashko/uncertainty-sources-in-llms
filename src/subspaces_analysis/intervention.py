import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_causal_intervention(analyzer, storage, baseline_prompts, alphas=[0, 5, 10, 20], top_k=5):
    """
    Performs a steering intervention by injecting the Uncertainty Null Space signal
    into baseline prompts. This tests the causal relationship between the Null Space
    and output entropy as described by Stolfo et al. (2024).
    """
    print("\n" + "=" * 60)
    print("🧪 STARTING ENHANCED CAUSAL INTERVENTION (STEERING)")
    print("=" * 60)

    # 1. Calculate the 'Uncertainty Direction' Vector
    # We derive this by subtracting the 'Certain' (Baseline) mean from
    # the 'Uncertain' (Epistemic + Aleatoric) mean.
    mean_epi = np.mean(storage['epistemic']['orig'], axis=0)
    mean_ale = np.mean(storage['aleatoric']['orig'], axis=0)
    mean_uncertainty = (mean_epi + mean_ale) / 2
    mean_base = np.mean(storage['baseline']['orig'], axis=0)

    # Difference in the raw Residual Stream
    diff_vector = torch.tensor(mean_uncertainty - mean_base).to(analyzer.device)

    # 2. Project into the Effective Null Space
    # This isolates the component that does not directly influence logit values
    # but impacts the final LayerNorm scale factor.
    steering_vector = analyzer.project_null(diff_vector)

    results = []

    # 3. Iterate through factual prompts to establish a robust pattern
    for prompt in baseline_prompts:
        print(f"\nTarget Prompt: '{prompt}'")

        for alpha in alphas:
            # Define the hook to modify the residual stream during forward pass
            def steering_hook(value, hook):
                # We apply the steering only to the final token position [batch, pos, d_model]
                value[:, -1, :] += alpha * steering_vector
                return value

            # Target the raw residual stream before the final LayerNorm
            final_block_index = analyzer.model.cfg.n_layers - 1
            hook_point = f"blocks.{final_block_index}.hook_resid_post"

            with analyzer.model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
                # Forward pass through the model
                logits = analyzer.model(prompt)[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

                # Measure Entropy (Uncertainty)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                # Get Top-K tokens to check for rank stability
                top_values, top_indices = torch.topk(probs, k=top_k)
                top_tokens = [analyzer.model.to_string(idx.item()).strip() for idx in top_indices]

                # Check if the Top-1 prediction changed compared to the Alpha=0 baseline
                current_top_token = top_tokens[0]
                current_top_prob = top_values[0].item()

                results.append({
                    "Prompt": prompt,
                    "Alpha": alpha,
                    "Entropy": round(entropy, 4),
                    "Top_Token": current_top_token,
                    "Confidence": round(current_top_prob, 4),
                    "Top_K_Rank": top_tokens
                })

                print(
                    f"  > Alpha {alpha:2}: Entropy={entropy:.4f} | Prediction='{current_top_token}' ({current_top_prob:.2%})")

    return pd.DataFrame(results)


def plot_steering_results(csv_path):
    df = pd.read_csv(csv_path)

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 7))


    color = 'tab:blue'
    ax1.set_xlabel('Alpha (Steering Strength)', fontsize=12)
    ax1.set_ylabel('Mean Entropy', color=color, fontsize=12)
    sns.lineplot(data=df, x='Alpha', y='Entropy', marker='o', ax=ax1, color=color, label='Entropy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Top Token Confidence', color=color, fontsize=12)
    sns.lineplot(data=df, x='Alpha', y='Confidence', marker='s', ax=ax2, color=color, label='Confidence')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Causal Intervention: Null Space Steering vs. Model Confidence', fontsize=15)
    fig.tight_layout()

    save_path = csv_path.replace('.csv', '.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graph saved to: {save_path}")
    plt.show()