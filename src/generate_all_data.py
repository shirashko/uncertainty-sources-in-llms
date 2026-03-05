from uncertainty_study_manager import UncertaintyStudyManager
from config import MODEL_ID

DATASET_TARGET_SIZE = 200  # Desired number of samples per class for the final dataset

def main():
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    # Execute the unified generation and balancing pipeline
    manager.run_study_generation(target_n=DATASET_TARGET_SIZE)


if __name__ == "__main__":
    main()