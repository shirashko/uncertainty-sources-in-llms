import numpy as np
import pandas as pd
import os


def get_top_uncertainty_neurons(probing_models_dict, task_name, space_name, output_dir, top_k=10):
    """
    Extracts top neurons for a specific task (Detection or Type) from a specific subspace.
    Used to identify neurons responsible for confidence regulation.
    """
    # Use the combined key created in probe.py
    model_key = f"{task_name}_{space_name}"

    if model_key not in probing_models_dict:
        print(f"❌ Error: Task '{model_key}' not found in results.")
        return []

    # 1. Access the specific trained Logistic Regression model
    probe = probing_models_dict[model_key]['model']
    weights = probe.coef_[0]

    # 2. Rank neurons by absolute weight magnitude
    importance = np.abs(weights)
    top_indices = np.argsort(importance)[::-1][:top_k]
    top_weights = weights[top_indices]

    neuron_data = []
    print(f"\n{'=' * 55}")
    print(f"🧠 TOP {top_k} NEURONS: {task_name} ({space_name})")
    print(f"{'=' * 55}")

    for rank, idx in enumerate(top_indices, 1):
        val = weights[idx]

        # Define labels based on the specific task being analyzed
        if "Type" in task_name:
            direction = "Epistemic-leaning" if val < 0 else "Aleatoric-leaning"
        else:
            # For Detection: 0 was Baseline, 1 was Uncertain
            direction = "Certainty-leaning" if val < 0 else "Uncertainty-leaning"

        print(f"Rank {rank}: Neuron #{idx:4} | Weight: {val:8.4f} | {direction}")

        neuron_data.append({
            "Rank": rank,
            "Neuron_Index": idx,
            "Weight": val,
            "Direction": direction,
            "Task": task_name,
            "Space": space_name
        })

    # 3. Save to CSV for thesis documentation and table generation
    df = pd.DataFrame(neuron_data)
    file_name = f"top_neurons_{task_name.split(' ')[0].lower()}_{space_name.lower().replace(' ', '_')}.csv"
    df.to_csv(os.path.join(output_dir, file_name), index=False)

    return top_indices