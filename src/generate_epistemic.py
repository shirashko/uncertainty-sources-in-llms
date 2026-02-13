from datasets import load_dataset
from config import MODEL_ID
from  data_generation import UncertaintyStudyManager


def main(n_samples=300):
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    print("--- Loading and Filtering PopQA (Epistemic) ---")
    popqa = load_dataset("akariasai/PopQA", split="test")
    target_categories = ['occupation', 'place of birth', 'capital', 'religion', 'genre']
    samples_per_cat = n_samples // len(target_categories)

    epistemic_data = []
    for cat in target_categories:
        # Filter for low-popularity facts (epistemic gaps)
        filtered = popqa.filter(lambda x: x['prop'] == cat and x['s_pop'] < 100).select(
            range(min(samples_per_cat, 100)))
        for item in filtered:
            prompt = manager.clean_to_declarative(item['question'])
            res = manager.get_inference_data(prompt)
            epistemic_data.append({**res, "prompt": prompt, "type": "epistemic", "category": cat})

    manager.export_jsonl(epistemic_data, "epistemic_dataset.jsonl")
    print(f"Done! Generated {len(epistemic_data)} epistemic samples.")


if __name__ == "__main__":
    main()