import os
from datetime import datetime
from src.config import MODEL_ID, MODEL_NAME
from src.subspaces_analysis.subspace_benchmark import run_triple_experiment
from uncertainty_engine import UncertaintyAnalyzer
from probe import run_probing_experiment

# Effective null space dimensionality (k) found for each model type
MODEL_TO_NULL_SPACE_DIM = {
    "gemma-2-2b": 5,
    "gpt2": 12,
    "llama-2-7b": 40,
    "pythia-410m": 12,
}


def main():
    # --- DYNAMIC PATH LOGIC ---
    # Ensure the script locates data within the src/data directory relative to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    DATA_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "uncertainty_study_dataset.jsonl")
    BASELINE_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "baseline_dataset.jsonl")

    # Path validation before loading the heavy model
    if not os.path.exists(DATA_PATH) or not os.path.exists(BASELINE_PATH):
        print(f"❌ ERROR: Dataset files not found!")
        print(f"Looked in: {os.path.join(project_root, 'src', 'data', MODEL_NAME)}")
        return

    # --- INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"{MODEL_NAME}_exp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set k dimension based on model name (defaults to 5)
    k_dim = MODEL_TO_NULL_SPACE_DIM.get(MODEL_NAME, 5)

    print(f"🚀 Initializing Analyzer for {MODEL_ID} (k={k_dim})")
    analyzer = UncertaintyAnalyzer(model_name=MODEL_ID, k=k_dim)

    # --- STEP 1: GEOMETRIC ANALYSIS (PCA & Cosine Similarity) ---
    print(f"\n📊 Running Subspace Benchmark (Geometric Analysis)...")
    # Returns both the summary metrics and the activation storage for probing
    summary_df, storage = run_triple_experiment(analyzer, DATA_PATH, BASELINE_PATH, output_dir)

    # Save and display geometric results
    csv_save_path = os.path.join(output_dir, "subspace_metrics.csv")
    summary_df.to_csv(csv_save_path, index=False)

    print("\n" + "=" * 90)
    print(f"✅ Geometric Experiment Complete - Results saved to: {output_dir}")
    print("=" * 90)
    formatted_df = summary_df.sort_values(by=["Mode", "Space"])
    print(formatted_df.to_string(index=False))
    print("=" * 90)

    # --- STEP 2: FUNCTIONAL TEST (Linear Probing) ---
    # Testing if Epistemic vs Aleatoric uncertainty is distinguishable in each subspace
    print("\n🧠 Starting Linear Probing (Epistemic vs Aleatoric)...")
    probing_results = run_probing_experiment(storage)

    # Save probing results
    probing_results.to_csv(os.path.join(output_dir, "probing_results.csv"), index=False)

    print("\n--- Probing Results (with Shuffled Control) ---")
    print(probing_results.to_string(index=False))
    print("=" * 90)
    print(f"All results, plots, and tables are available in: {output_dir}")


if __name__ == "__main__":
    main()