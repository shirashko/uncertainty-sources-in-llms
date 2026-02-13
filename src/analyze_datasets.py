import json
import os
import pandas as pd
from collections import Counter
from config import MODEL_ID


def analyze_dataset_outputs(model_tag: str):
    file_path = f"data/{model_tag}/uncertainty_study_dataset.jsonl"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # 1. Basic confidence stats per type
    print("\n--- Confidence Analysis ---")
    print(df.groupby('type')['confidence'].describe()[['mean', 'std', 'min', 'max']])

    # 2. Token Diversity Analysis
    # Let's see what are the most common predicted tokens for each type
    print("\n--- Most Frequent Top-1 Predictions ---")
    for group in df['type'].unique():
        subset = df[df['type'] == group]
        common_tokens = Counter(subset['prediction']).most_common(5)
        print(f"\nType: {group.upper()}")
        for token, count in common_tokens:
            print(f"  - '{token}': {count} times")

    # 3. Pattern Detection: "I don't know" vs. Specificity
    # For Epistemic, we expect specific (but wrong) names.
    # For Aleatoric, we might see more generic tokens.

    # 4. Uncertainty Gaps
    # Calculate how often the model is 'very unsure' (conf < 0.3)
    unsure_threshold = 0.3
    for group in df['type'].unique():
        unsure_count = len(df[(df['type'] == group) & (df['confidence'] < unsure_threshold)])
        total = len(df[df['type'] == group])
        print(f"\nUncertainty Rate (conf < {unsure_threshold}) for {group}: {unsure_count / total:.2%}")

    return df


if __name__ == "__main__":
    model_tag = MODEL_ID.split("/")[-1]
    df_results = analyze_dataset_outputs(model_tag)