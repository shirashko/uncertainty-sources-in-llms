import numpy as np


def get_top_uncertainty_neurons(probing_results_dict, top_k=10):
    """
    Identifies the specific neurons (dimensions) that the probe relies on
    most heavily to distinguish between uncertainty types.
    """
    # 1. Access the trained Logistic Regression model from your results
    # We focus on the 'Null Space' probe as it relates to the paper's mechanism
    probe = probing_results_dict['Null Space']['model']

    # 2. Extract the weights (coefficients)
    # The shape is [n_classes, d_model]. For binary, it might be [1, d_model]
    weights = probe.coef_[0]

    # 3. Calculate absolute importance
    # We care about magnitude, as high negative weights are just as informative
    importance = np.abs(weights)

    # 4. Get the indices of the top-K neurons
    top_indices = np.argsort(importance)[::-1][:top_k]
    top_weights = weights[top_indices]

    print(f'\n{"=" * 40}')
    print(f"🧠 TOP {top_k} UNCERTAINTY NEURONS")
    print(f'{"=" * 40}')

    for rank, (idx, val) in enumerate(zip(top_indices, top_weights), 1):
        direction = "Positive (Epistemic-leaning)" if val > 0 else "Negative (Aleatoric-leaning)"
        print(f"Rank {rank}: Neuron #{idx:4} | Weight: {val:8.4f} | Direction: {direction}")

    return top_indices