import os
from datetime import datetime
from uncertainty_engine import UncertaintyAnalyzer
from subspace_benchmark import run_triple_experiment


def main():
    # 1. Configuration
    DATA_PATH = "data/uncertainty_study_dataset.jsonl"
    BASELINE_PATH = "data/baseline_inputs.json"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join("results", f"exp_{TIMESTAMP}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Initialize Engine
    analyzer = UncertaintyAnalyzer(model_name="gpt2", k=12)

    # 3. Run Benchmark
    summary_df = run_triple_experiment(analyzer, DATA_PATH, BASELINE_PATH, OUTPUT_DIR)

    # 4. Final Output
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    print(f"\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()