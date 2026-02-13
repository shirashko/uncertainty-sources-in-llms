import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def run_probing_experiment(storage):
    """
    Trains a Linear Probe to distinguish between Epistemic and Aleatoric samples.
    Returns both a summary DataFrame and the trained model for Neuron Attribution.
    """
    results_list = []
    # Dictionary to store the final trained model for each space
    trained_models = {}

    target_cats = ['epistemic', 'aleatoric']

    # Prepare labels: 0 for epistemic, 1 for aleatoric
    labels = []
    for cat in target_cats:
        labels.extend([cat] * len(storage[cat]['orig']))

    y = np.array([0 if l == 'epistemic' else 1 for l in labels])
    y_shuffled = np.random.permutation(y)  # Control Group

    spaces = [("orig", "Original"), ("null", "Null Space"), ("logits", "Logit Space")]

    print("\n--- Running Linear Probing Analysis ---")

    for key, space_name in spaces:
        # Stack activations for the current space
        X = np.vstack([np.stack(storage[cat][key]) for cat in target_cats])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        real_accs = []
        control_accs = []
        last_model = None

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_ctrl_train, y_ctrl_test = y_shuffled[train_index], y_shuffled[test_index]

            # Train real probe
            probe = LogisticRegression(max_iter=1000, C=1.0)
            probe.fit(X_train, y_train)
            real_accs.append(accuracy_score(y_test, probe.predict(X_test)))

            # Keep the last trained model to use its weights for attribution
            last_model = probe

            # Train control probe
            ctrl_probe = LogisticRegression(max_iter=1000, C=1.0)
            ctrl_probe.fit(X_train, y_ctrl_train)
            control_accs.append(accuracy_score(y_ctrl_test, ctrl_probe.predict(X_test)))

        # Save summary metrics
        results_list.append({
            "Space": space_name,
            "Probe Accuracy": np.mean(real_accs),
            "Control Accuracy": np.mean(control_accs),
            "Delta": np.mean(real_accs) - np.mean(control_accs)
        })

        # Store the trained model object for later neuron analysis
        trained_models[space_name] = {
            "model": last_model,
            "accuracy": np.mean(real_accs)
        }

    # Return both the display table and the model dictionary
    return pd.DataFrame(results_list), trained_models