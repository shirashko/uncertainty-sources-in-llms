import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os


def run_probing_experiment(storage):
    """
    Trains two types of probes using Cross-Validation to ensure robust results:
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

    # Setup Stratified K-Fold to ensure class balance in each fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for task_id, cats, task_name in tasks:
        print(f"\n--- Running Probe: {task_name} ---")

        # 1. Prepare labels and data for the specific task
        if task_id == 'detect':
            # 0 for Baseline, 1 for any Uncertainty (Epi + Alea)
            y = np.array([0] * len(storage['baseline']['orig']) +
                         [1] * (len(storage['epistemic']['orig']) + len(storage['aleatoric']['orig'])))
            storage_keys = ['baseline', 'epistemic', 'aleatoric']
        else:
            # 0 for Epistemic, 1 for Aleatoric
            y = np.array([0] * len(storage['epistemic']['orig']) +
                         [1] * len(storage['aleatoric']['orig']))
            storage_keys = ['epistemic', 'aleatoric']

        # 2. Run for each geometric space
        for key_space, space_label in [("orig", "Original"), ("null", "Null Space"), ("logits", "Logit Space")]:
            # Stack activations from the relevant keys for the current subspace
            X = np.vstack([np.stack(storage[k][key_space]) for k in storage_keys])

            # Define the probe
            probe = LogisticRegression(max_iter=1000, C=1.0)

            # 3. Perform Cross-Validation
            # This trains and tests the model 5 times on different subsets
            cv_scores = cross_val_score(probe, X, y, cv=cv_strategy, scoring='accuracy')

            avg_acc = cv_scores.mean()
            std_acc = cv_scores.std()

            print(f"[{space_label}] CV Accuracy: {avg_acc:.4f} (+/- {std_acc:.4f})")

            # 4. Final Fit and Storage
            # We fit on the FULL data once more to save the "best" coefficients for steering/analysis
            probe.fit(X, y)

            results_list.append({
                "Task": task_name,
                "Space": space_label,
                "Accuracy": avg_acc,
                "Std Dev": std_acc
            })

            # Save the model and its metadata
            trained_models[f"{task_name}_{space_label}"] = {
                "model": probe,
                "accuracy": avg_acc,
                "std": std_acc
            }

    return pd.DataFrame(results_list), trained_models


from sklearn.metrics.pairwise import cosine_similarity


def analyze_probe_axes_orthogonality(trained_models, output_dir):
    """
    Extracts the weight vectors (coefficients) from the probes and
    measures the cosine similarity to test for geometric disentanglement.
    """
    print("\n📐 Analyzing Geometric Orthogonality of Probe Axes...")

    try:
        w_detect = trained_models["Detection (Cert vs Uncert)_Null Space"]["model"].coef_
        w_type = trained_models["Type (Epi vs Alea)_Null Space"]["model"].coef_

        similarity = cosine_similarity(w_detect, w_type)[0][0]

        report_path = os.path.join(output_dir, "orthogonality_report.txt")
        with open(report_path, "w") as f:
            f.write(f"--- Subspace Orthogonality Report ---\n")
            f.write(f"Cosine Similarity (Detection vs Type) in Null Space: {similarity:.4f}\n")

            if abs(similarity) < 0.2:
                f.write(
                    "Conclusion: High Orthogonality. The model represents 'What is uncertainty' and 'Why am I uncertain' on separate axes.\n")
            else:
                f.write(f"Conclusion: Linear overlap detected (Similarity: {similarity:.4f}).\n")

        print(f"✅ Orthogonality: {similarity:.4f}")
        print(f"📄 Report saved to: {report_path}")

    except KeyError as e:
        print(f"⚠️ Could not perform orthogonality check. Missing model: {e}")