import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
import os
from sklearn.metrics.pairwise import cosine_similarity

def run_probing_experiment(storage):
    """
    Trains probes using Cross-Validation to evaluate geometric disentanglement.
    Metrics: Accuracy (balanced) and ROC-AUC (threshold-independent).
    """
    trained_models = {}
    results_list = []

    # Define the two probing tasks
    tasks = [
        ('type', ['epistemic', 'aleatoric'], "Type (Epi vs Alea)"),
        ('detect', ['baseline', 'epistemic', 'aleatoric'], "Detection (Cert vs Uncert)")
    ]

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for task_id, storage_keys, task_name in tasks:
        print(f"\n--- Running Probe: {task_name} ---")

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
        for key_space, space_label in [("orig", "Original"), ("null", "Null Space"), ("logits", "Logit Space")]:
            X = np.vstack([np.stack(storage[k][key_space]) for k in storage_keys])

            # Logistic Regression with balanced weights for the minority class
            probe = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')

            # 3. Perform Cross-Validation for multiple metrics
            scoring_metrics = ['accuracy', 'roc_auc']
            cv_results = cross_validate(probe, X, y, cv=cv_strategy, scoring=scoring_metrics)

            avg_acc = cv_results['test_accuracy'].mean()
            avg_auc = cv_results['test_roc_auc'].mean()
            std_auc = cv_results['test_roc_auc'].std()

            print(f"[{space_label}] Accuracy: {avg_acc:.2f} | ROC-AUC: {avg_auc:.2f} (+/- {std_auc:.2f})")

            # 4. Final Fit on all available data for downstream steering/analysis
            probe.fit(X, y)

            results_list.append({
                "Task": task_name,
                "Space": space_label,
                "Accuracy": round(avg_acc, 3),
                "ROC-AUC": round(avg_auc, 3),
                "AUC Std Dev": round(std_auc, 3)
            })

            # Save metadata
            trained_models[f"{task_name}_{space_label}"] = {
                "model": probe,
                "accuracy": avg_acc,
                "auc": avg_auc
            }

    return pd.DataFrame(results_list), trained_models

def analyze_probe_axes_orthogonality(trained_models, output_dir):
    """
    Measures the cosine similarity between the decision boundaries (coefficients)
    of the 'Detection' probe and the 'Type' probe.
    """
    print("\n📐 Analyzing Geometric Orthogonality of Probe Axes...")

    try:
        # We look specifically at Null Space to see if Localized-UNDO successfully disentangled the axes
        w_detect = trained_models["Detection (Cert vs Uncert)_Null Space"]["model"].coef_
        w_type = trained_models["Type (Epi vs Alea)_Null Space"]["model"].coef_

        similarity = round(cosine_similarity(w_detect, w_type)[0][0], 4)

        report_path = os.path.join(output_dir, "orthogonality_report.txt")
        with open(report_path, "w") as f:
            f.write(f"--- Subspace Orthogonality Report ---\n")
            f.write(f"Cosine Similarity (Detection vs Type) in Null Space: {similarity:.4f}\n")

            # Low cosine similarity indicates the model represents 'Certainty' and 'Uncertainty Type' independently
            if abs(similarity) < 0.2:
                f.write("Conclusion: High Orthogonality. The model represents 'Existence of uncertainty' "
                        "and 'Source of uncertainty' on separate geometric axes.\n")
            else:
                f.write(f"Conclusion: Linear overlap detected (Similarity: {similarity:.4f}).\n")

        print(f"✅ Orthogonality (Cosine Sim): {similarity:.4f}")
        print(f"📄 Report saved to: {report_path}")

    except KeyError as e:
        print(f"⚠️ Could not perform orthogonality check. Missing model: {e}")