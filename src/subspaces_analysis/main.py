import os
from datetime import datetime
from src.config import MODEL_ID, MODEL_NAME
from src.subspaces_analysis.subspace_benchmark import run_triple_experiment
from uncertainty_engine import UncertaintyAnalyzer
from probe import run_probing_experiment, analyze_probe_axes_orthogonality
from intervention import run_causal_intervention, plot_steering_results
from neuron_attribution import get_top_uncertainty_neurons
from plot import plot_layer_wise_emergence


def main():
    # --- PATH SETUP ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    DATA_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "uncertainty_study_dataset.jsonl")
    BASELINE_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "baseline_dataset.jsonl")

    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: Dataset not found at {DATA_PATH}")
        return

    # --- INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"{MODEL_NAME}_multi_layer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set k dimension based on model architecture
    k_dim = 12 if "gpt2" in MODEL_NAME or "llama" in MODEL_NAME else 5
    analyzer = UncertaintyAnalyzer(model_name=MODEL_ID, k=k_dim)

    # --- LAYER SELECTION ---
    total_layers = analyzer.model.cfg.n_layers
    layers_to_check = [total_layers - 1]
    print(f"🔍 Analyzing layers: {layers_to_check}")

    # --- STEP 1: GEOMETRIC ANALYSIS ---
    # Extract activations and project into subspaces (Null Space vs Logit Space)
    csv_metrics_path = os.path.join(output_dir, "multi_layer_metrics.csv")
    summary_df, storage_last_layer = run_triple_experiment(
        analyzer, DATA_PATH, BASELINE_PATH, output_dir, layers_to_test=layers_to_check
    )
    summary_df.to_csv(csv_metrics_path, index=False)
    plot_layer_wise_emergence(csv_metrics_path, output_dir)

    # --- STEP 2: PROBING (Functional Test) ---
    # Test if uncertainty types are linearly separable in the identified subspaces
    print("\n🧠 Probing final layer for linear disentanglement...")
    probing_results_df, probing_models_dict = run_probing_experiment(storage_last_layer)
    probing_results_df.to_csv(os.path.join(output_dir, "probing_results.csv"), index=False)

    # Calculate geometric orthogonality between the "Detection" and "Type" axes
    analyze_probe_axes_orthogonality(probing_models_dict, output_dir)

    # --- STEP 3: NEURON ATTRIBUTION ---
    # Map the high-level probe signals back to specific individual neurons
    print("\n🔍 Attributing uncertainty signals to specific neurons...")
    get_top_uncertainty_neurons(probing_models_dict, "Detection (Cert vs Uncert)", "Null Space", output_dir)
    get_top_uncertainty_neurons(probing_models_dict, "Type (Epi vs Alea)", "Null Space", output_dir)

    # --- STEP 4: CAUSAL INTERVENTION (Steering) ---
    # Apply the identified Null Space vectors to steer model confidence
    print("\n🧪 Running Causal Steering Intervention...")
    test_prompts = ["The capital of France is", "Two plus two equals"]
    steering_df = run_causal_intervention(analyzer, storage_last_layer, test_prompts, alphas=[0, 10, 20])

    steering_path = os.path.join(output_dir, "steering_results.csv")
    steering_df.to_csv(steering_path, index=False)
    plot_steering_results(steering_path)

    print(f"\n✨ DONE. All results saved in: {output_dir}")


if __name__ == "__main__":
    main()