import json
import csv
import os
from collections import Counter

# Define paths
input_path = '/Users/shirashko/PycharmProjects/uncertainty-sources-in-llms/src/data/gpt2/uncertainty_study_dataset.jsonl'
output_path = '/Users/shirashko/PycharmProjects/uncertainty-sources-in-llms/src/data/gpt2/dataset_summary_check.csv'


def extract_dataset_info(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    extracted_data = []
    type_counts = Counter()

    # Read the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    data_type = entry.get('type', 'N/A')
                    prompt = entry.get('prompt', 'N/A')

                    extracted_data.append({
                        'Type': data_type,
                        'Prompt': prompt
                    })

                    type_counts[data_type] += 1
                except json.JSONDecodeError:
                    continue

    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Type', 'Prompt'])
        writer.writeheader()
        writer.writerows(extracted_data)

    # Print Summary
    print("-" * 30)
    print(f"{'Type':<15} | {'Count':<10}")
    print("-" * 30)
    for d_type, count in type_counts.items():
        print(f"{d_type:<15} | {count:<10}")
    print("-" * 30)
    print(f"Total: {len(extracted_data)} entries.")
    print(f"File saved to: {output_file}")


if __name__ == "__main__":
    extract_dataset_info(input_path, output_path)