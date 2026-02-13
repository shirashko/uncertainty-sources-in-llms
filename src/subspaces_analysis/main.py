import os
from datetime import datetime
from src.config import MODEL_ID, MODEL_NAME
from src.subspaces_analysis.subspace_benchmark import run_triple_experiment
from uncertainty_engine import UncertaintyAnalyzer

MODEL_TO_NULL_SPACE_DIM = {
    "gemma-2-2b": 5,
    "gpt2": 12,
    "llama-2-7b": 40,
    "pythia-410m": 12,
}


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/subspaces_analysis
    project_root = os.path.dirname(os.path.dirname(current_dir))  # uncertainty-sources-in-llms

    # Precise paths according to your terminal output
    DATA_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "uncertainty_study_dataset.jsonl")
    BASELINE_PATH = os.path.join(project_root, "src", "data", MODEL_NAME, "baseline_dataset.jsonl")

    # Safety check
    if not os.path.exists(DATA_PATH) or not os.path.exists(BASELINE_PATH):
        print(f"❌ ERROR: Dataset files not found!")
        print(f"Looking in: {os.path.join(project_root, 'src', 'data', MODEL_NAME)}")
        return

    # --- INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"{MODEL_NAME}_exp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    k_dim = MODEL_TO_NULL_SPACE_DIM.get(MODEL_NAME, 5)

    print(f"🚀 Initializing Analyzer for {MODEL_ID} (k={k_dim})")
    analyzer = UncertaintyAnalyzer(model_name=MODEL_ID, k=k_dim)

    # --- EXPERIMENT EXECUTION ---
    print(f"📊 Running Triple Subspace Experiment...")
    summary_df = run_triple_experiment(analyzer, DATA_PATH, BASELINE_PATH, output_dir)

    # --- EXPORT & DISPLAY ---
    csv_save_path = os.path.join(output_dir, "subspace_metrics.csv")
    summary_df.to_csv(csv_save_path, index=False)

    print("\n" + "=" * 90)
    print(f"✅ EXPERIMENT COMPLETE - Results saved to: {output_dir}")
    print("=" * 90)

    formatted_df = summary_df.sort_values(by=["Mode", "Space"])
    print(formatted_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()