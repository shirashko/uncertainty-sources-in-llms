import os
from datetime import datetime
from uncertainty_engine import UncertaintyAnalyzer
from subspace_benchmark import run_triple_experiment


def main():
    # 1. Setup paths and parameters
    DATA_PATH = "data/uncertainty_study_dataset.jsonl"
    BASELINE_PATH = "data/baseline_inputs.json"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join("results", f"exp_{TIMESTAMP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Run analysis with k=12
    analyzer = UncertaintyAnalyzer(k=12)
    summary_df = run_triple_experiment(analyzer, DATA_PATH, BASELINE_PATH, OUTPUT_DIR)

    # 3. Save and display results
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_metrics.csv"), index=False)

    print("\n" + "=" * 90)
    print(f"EXPERIMENT RESULTS - Saved to: {OUTPUT_DIR}")
    print("=" * 90)
    # Print the table with the 'Mode' column prominently displayed
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()