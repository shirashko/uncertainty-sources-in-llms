import json
import os
from config import MODEL_ID


def create_simplified_dataset(model_tag: str):
    """
    Reads the full uncertainty study dataset and exports a version
    containing only the text prompts and their associated uncertainty types.
    """
    input_path = f"data/{model_tag}/uncertainty_study_dataset.jsonl"
    output_path = f"data/{model_tag}/prompts_and_types.jsonl"

    if not os.path.exists(input_path):
        print(f"Error: Source file not found at {input_path}")
        return

    simplified_data = []

    print(f"Reading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                full_entry = json.loads(line)
                # Extract only the necessary fields
                simplified_entry = {
                    "type": full_entry.get("type"),
                    "prompt": full_entry.get("prompt")
                }
                simplified_data.append(simplified_entry)

    # Export to a new JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in simplified_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully exported {len(simplified_data)} entries to {output_path}")


if __name__ == "__main__":
    # Extract tag from MODEL_ID (e.g., 'google/gemma-2-2b' -> 'gemma-2-2b')
    model_tag = MODEL_ID.split("/")[-1]
    create_simplified_dataset(model_tag)