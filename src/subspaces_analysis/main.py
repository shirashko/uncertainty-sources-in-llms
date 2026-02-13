import os
from datetime import datetime
from uncertainty_engine import UncertaintyAnalyzer
from subspace_benchmark import run_triple_experiment
from src.config import MODEL_ID, MODEL_NAME

MODEL_TO_NULL_SPACE_DIM = {
    "gemma-2-2b": 5,
    "gpt2": 12,
    "llama-2-7b": 40,
    "pythia-410m": 12,
}

def main():
    DATA_PATH = f"data/{MODEL_NAME}/uncertainty_study_dataset.jsonl"
    BASELINE_PATH = f"data/{MODEL_NAME}/baseline_inputs.json"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{MODEL_NAME}_exp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    analyzer = UncertaintyAnalyzer(model_name=MODEL_ID, k=MODEL_TO_NULL_SPACE_DIM[MODEL_NAME])

    summary_df = run_triple_experiment(analyzer, DATA_PATH, BASELINE_PATH, output_dir)

    summary_df.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
    print("\n" + "=" * 90)
    print(f"EXPERIMENT RESULTS - Saved to: {output_dir}")
    print("=" * 90)
    # Print the table with the 'Mode' column prominently displayed
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()