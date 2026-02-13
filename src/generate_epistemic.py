import re
import random
from datasets import load_dataset
from config import MODEL_ID
from data_generation import UncertaintyStudyManager


def main(n_samples_total=500):
    # Initialize the manager (loads the model)
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    print("--- Loading PopQA for Full Diversity ---")
    popqa = load_dataset("akariasai/PopQA", split="test")

    # Get all unique properties (categories) available in the dataset
    all_categories = list(set(popqa['prop']))
    print(f"Found {len(all_categories)} categories: {all_categories}")

    # Calculate fair distribution across categories
    samples_per_cat = n_samples_total // len(all_categories)
    epistemic_data = []

    for cat in all_categories:
        # Filter for low-popularity facts (Epistemic gaps)
        filtered = popqa.filter(lambda x: x['prop'] == cat and x['s_pop'] < 100)

        num_to_take = min(len(filtered), samples_per_cat)
        if num_to_take == 0:
            continue

        subset = filtered.select(range(num_to_take))
        print(f"Processing category '{cat}': {num_to_take} samples")

        for item in subset:
            # Use the local clean_to_declarative function
            prompt = manager.clean_to_declarative(item['question'])
            res = manager.get_inference_data(prompt)

            epistemic_data.append({
                **res,
                "prompt": prompt,
                "type": "epistemic",
                "category": cat,
                "original_question": item['question']
            })

    # Save to model-specific folder and master file
    manager.export_jsonl(epistemic_data, "epistemic_dataset.jsonl")
    print(f"\nFinished! Total generated: {len(epistemic_data)} across {len(all_categories)} categories.")


if __name__ == "__main__":
    main(n_samples_total=500)