import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity


def run_probing_experiment(storage):
    """
    Trains probes on both Absolute and Residual activations using Cross-Validation.
    Metrics: Accuracy (balanced) and ROC-AUC (threshold-independent).
    """
    trained_models = {}
    results_list = []

    # Define the two probing tasks
    tasks = [
        ('type', ['epistemic', 'aleatoric'], "Type (Epi vs Alea)"),
        ('detect', ['baseline', 'epistemic', 'aleatoric'], "Detection (Cert vs Uncert)")
    ]

    # Pre-calculate baseline means for Residual calculation
    # We use the mean of the 'baseline' category for each geometric subspace
    baseline_means = {
        space: np.mean(storage['baseline'][space], axis=0)
        for space in ["orig", "null", "logits", "random"]
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We iterate through both modes
    for mode in ["Absolute", "Residual"]:
        print(f"\n🚀 MODE: {mode}")

        for task_id, storage_keys, task_name in tasks:
            print(f"--- Running Probe: {task_name} ({mode}) ---")

            # 1. Prepare labels
            if task_id == 'detect':
                # 0 for Baseline, 1 for any Uncertainty (Epi + Alea)
                y = np.array([0] * len(storage['baseline']['orig']) +
                             [1] * (len(storage['epistemic']['orig']) + len(storage['aleatoric']['orig'])))
            else:
                # 0 for Epistemic, 1 for Aleatoric
                y = np.array([0] * len(storage['epistemic']['orig']) +
                             [1] * len(storage['aleatoric']['orig']))

            # 2. Run for each geometric space
            for key_space, space_label in [("orig", "Original"), ("null", "Null Space"), ("logits", "Logit Space"), ("random", "Random Control")]:
                # Stack the raw activations
                X = np.vstack([np.stack(storage[k][key_space]) for k in storage_keys])

                # Apply Residual transformation if needed
                if mode == "Residual":
                    X = X - baseline_means[key_space]

                # Logistic Regression with balanced weights
                probe = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')

                # 3. Perform Cross-Validation
                scoring_metrics = ['accuracy', 'roc_auc']
                cv_results = cross_validate(probe, X, y, cv=cv_strategy, scoring=scoring_metrics)

                avg_acc = cv_results['test_accuracy'].mean()
                avg_auc = cv_results['test_roc_auc'].mean()
                std_auc = cv_results['test_roc_auc'].std()

                print(f"[{space_label}] Acc: {avg_acc:.2f} | AUC: {avg_auc:.2f}")

                # 4. Final Fit on all data
                probe.fit(X, y)

                # Store results with Mode indication
                results_list.append({
                    "Task": task_name,
                    "Mode": mode,
                    "Space": space_label,
                    "Accuracy": round(avg_acc, 3),
                    "ROC-AUC": round(avg_auc, 3),
                    "AUC Std Dev": round(std_auc, 3)
                })

                # Save model with a unique key including the mode
                model_key = f"{task_name}_{space_label}_{mode}"
                trained_models[model_key] = {
                    "model": probe,
                    "accuracy": avg_acc,
                    "auc": avg_auc,
                    "mode": mode
                }

    return pd.DataFrame(results_list), trained_models


def analyze_probe_axes_orthogonality(trained_models, layer_idx):
    """
    Measures the cosine similarity between the decision boundaries of probes.
    Returns a list of results for the specific layer.
    """
    layer_ortho_results = []

    for mode in ["Absolute", "Residual"]:
        try:
            key_detect = f"Detection (Cert vs Uncert)_Null Space_{mode}"
            key_type = f"Type (Epi vs Alea)_Null Space_{mode}"

            w_detect = trained_models[key_detect]["model"].coef_
            w_type = trained_models[key_type]["model"].coef_

            similarity = round(cosine_similarity(w_detect, w_type)[0][0], 4)

            # קביעת הסטטוס הגיאומטרי
            if abs(similarity) < 0.2:
                geometry = "Orthogonal"
            elif abs(similarity) > 0.7:
                geometry = "Linear Overlap"
            else:
                geometry = "Partial Overlap"

            layer_ortho_results.append({
                "Layer": layer_idx,
                "Mode": mode,
                "Cosine_Sim": similarity,
                "Geometry": geometry
            })

        except KeyError as e:
            print(f"⚠️ Missing model for L{layer_idx} {mode} orthogonality: {e}")

    return layer_ortho_results