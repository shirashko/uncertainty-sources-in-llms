import os
from datetime import datetime
from src.config import MODEL_ID, MODEL_NAME
from src.subspaces_analysis.subspace_benchmark import run_triple_experiment
from uncertainty_engine import UncertaintyAnalyzer
from probe import run_probing_experiment
from intervention import run_causal_intervention, plot_steering_results
from neuron_attribution import get_top_uncertainty_neurons

# Effective null space dimensionality (k) found for each model type
MODEL_TO_NULL_SPACE_DIM = {
    "gemma-2-2b": 5,
    "gpt2": 12,
    "llama-2-7b": 40,
    "pythia-410m": 12,
}


def main():
    # --- DYNAMIC PATH LOGIC ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    DATA_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "uncertainty_study_dataset.jsonl")
    BASELINE_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "baseline_dataset.jsonl")

    # Path validation before loading heavy models
    if not os.path.exists(DATA_PATH) or not os.path.exists(BASELINE_PATH):
        print(f"❌ ERROR: Dataset files not found!")
        return

    # --- INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"{MODEL_NAME}_exp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set k dimension based on model name
    k_dim = MODEL_TO_NULL_SPACE_DIM.get(MODEL_NAME, 5)

    print(f"🚀 Initializing Analyzer for {MODEL_ID} (k={k_dim})")
    analyzer = UncertaintyAnalyzer(model_name=MODEL_ID, k=k_dim)

    # --- STEP 1: GEOMETRIC ANALYSIS ---
    # Measuring alignment between uncertainty types and geometric subspaces
    print(f"\n📊 Running Subspace Benchmark (Geometric Analysis)...")
    summary_df, storage = run_triple_experiment(analyzer, DATA_PATH, BASELINE_PATH, output_dir)

    summary_df.to_csv(os.path.join(output_dir, "subspace_metrics.csv"), index=False)
    print(summary_df.sort_values(by=["Mode", "Space"]).to_string(index=False))

    # --- STEP 2: FUNCTIONAL TEST (Two-Task Linear Probing) ---
    # 1. Detection: Certainty vs Uncertainty
    # 2. Type: Epistemic vs Aleatoric
    print("\n🧠 Starting Dual-Task Linear Probing...")
    probing_results_df, probing_models_dict = run_probing_experiment(storage)

    probing_results_df.to_csv(os.path.join(output_dir, "probing_results.csv"), index=False)
    print("\n--- Probing Results ---")
    print(probing_results_df.to_string(index=False))

    # --- STEP 3: NEURON ATTRIBUTION ---
    # Identifying the "Regulator" neurons (Detection) and "Expert" neurons (Type)
    print("\n🔍 Analyzing key uncertainty neurons in the Null Space...")

    # 1. Uncertainty Detection Neurons (The Control Switches)
    regulators = get_top_uncertainty_neurons(
        probing_models_dict, "Detection (Cert vs Uncert)", "Null Space", output_dir, top_k=10
    )

    # 2. Uncertainty Type Neurons (The Specialists)
    specialists = get_top_uncertainty_neurons(
        probing_models_dict, "Type (Epi vs Alea)", "Null Space", output_dir, top_k=10
    )

    # --- STEP 4: CAUSAL INTERVENTION (STEERING) ---
    # Proving that Null Space signals causally regulate entropy through LayerNorm scaling
    print("\n" + "=" * 90)
    print("🧪 Starting Causal Intervention (Steering Experiment)")

    test_prompts = [
        "The capital of France is", "The sun rises in the", "Water boils at",
        "The opposite of up is", "The color of the sky is", "Two plus two equals",
        "The earth revolves around the", "Humans breathe",
    ]

    steering_df = run_causal_intervention(analyzer, storage, test_prompts, alphas=[0, 5, 10, 20])

    steering_csv_path = os.path.join(output_dir, "steering_results.csv")
    steering_df.to_csv(steering_csv_path, index=False)

    # Generate visualization for the thesis
    plot_steering_results(steering_csv_path)

    print("\n" + "=" * 90)
    print(f"✨ FULL EXPERIMENT COMPLETE. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()