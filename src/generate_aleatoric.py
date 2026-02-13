from datasets import load_dataset
from data_generation import UncertaintyStudyManager
from src.config import MODEL_ID


def main(n_samples=300):
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    print("--- Loading AmbigQA (Aleatoric) ---")
    ambigqa = load_dataset("sewon/ambig_qa", "light", split="train")

    # Filter for questions with multiple valid interpretations
    aleatoric_raw = ambigqa.filter(lambda x: 'multipleQAs' in x['annotations']['type']).select(range(n_samples))

    aleatoric_data = []
    for item in aleatoric_raw:
        prompt = manager.clean_to_declarative(item['question'])
        res = manager.get_inference_data(prompt)
        aleatoric_data.append({**res, "prompt": prompt, "type": "aleatoric"})

    manager.export_jsonl(aleatoric_data, "aleatoric_dataset.jsonl")
    print(f"Done! Generated {len(aleatoric_data)} aleatoric samples.")


if __name__ == "__main__":
    main()