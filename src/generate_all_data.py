from data_generation import UncertaintyStudyManager
from config import MODEL_ID
import os


def main():
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    # 1. Clear old master file
    master_path = os.path.join(manager.data_dir, "uncertainty_study_dataset.jsonl")
    if os.path.exists(master_path):
        os.remove(master_path)

    N = 200  # Balancing factor

    # Generate and save in sequence
    # Use 'a' (append) for everything after the first one
    baseline = manager.generate_baseline_c4(n_samples=N)
    manager.export_jsonl(baseline, "baseline_dataset.jsonl", mode='a')

    epistemic = manager.generate_epistemic_popqa(n_samples=N)
    manager.export_jsonl(epistemic, "epistemic_dataset.jsonl", mode='a')

    aleatoric = manager.generate_aleatoric_ambig(n_samples=N)
    manager.export_jsonl(aleatoric, "aleatoric_dataset.jsonl", mode='a')

    print(f"\nSuccessfully generated balanced dataset (N={N} per type)")


if __name__ == "__main__":
    main()