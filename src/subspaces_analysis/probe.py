import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_probing_experiment(storage):
    """
    Trains two types of probes:
    1. Epistemic vs Aleatoric (Uncertainty Types)
    2. Baseline vs (Epi + Alea) (Certainty vs Uncertainty)
    """
    trained_models = {}
    results_list = []

    # Define the two tasks
    tasks = [
        ('type', ['epistemic', 'aleatoric'], "Type (Epi vs Alea)"),
        ('detect', ['baseline', 'uncertain'], "Detection (Cert vs Uncert)")
    ]

    for task_id, cats, task_name in tasks:
        print(f"\n--- Running Probe: {task_name} ---")

        # Prepare labels and data for the specific task
        if task_id == 'detect':
            # 0 for Baseline, 1 for any Uncertainty
            y = np.array([0] * len(storage['baseline']['orig']) +
                         [1] * (len(storage['epistemic']['orig']) + len(storage['aleatoric']['orig'])))
            storage_keys = ['baseline', 'epistemic', 'aleatoric']  # Combine Epi and Alea for 'Uncertain'
        else:
            # 0 for Epistemic, 1 for Aleatoric
            y = np.array([0] * len(storage['epistemic']['orig']) +
                         [1] * len(storage['aleatoric']['orig']))
            storage_keys = ['epistemic', 'aleatoric']

        # Run for each geometric space
        for key_space, space_label in [("orig", "Original"), ("null", "Null Space"), ("logits", "Logit Space")]:
            X = np.vstack([np.stack(storage[k][key_space]) for k in storage_keys])

            # Simple split for demonstration (you can keep StratifiedKFold if preferred)
            probe = LogisticRegression(max_iter=1000, C=1.0)
            probe.fit(X, y)
            acc = probe.score(X, y)  # Accuracy on training set for simplicity here

            results_list.append({
                "Task": task_name,
                "Space": space_label,
                "Accuracy": acc
            })

            # Save the model for attribution
            trained_models[f"{task_name}_{space_label}"] = {
                "model": probe,
                "accuracy": acc
            }

    return pd.DataFrame(results_list), trained_models