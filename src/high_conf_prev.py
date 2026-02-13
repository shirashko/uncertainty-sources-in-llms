import os
import torch
from data_generation import UncertaintyStudyManager
from config import MODEL_ID

# High-confidence prompts
BASELINE_PROMPTS = [
                    # Idioms
                    "The Great Wall of",
                    "To be or not to",
                    "Once upon a",
                    "The sun rises in the",
                    "In the nick of",
                    "A piece of",
                    "The United States of",
                    "Better late than"
                    # induction
                    "The recipe calls for flour, sugar, and butter. Mix the flour, sugar, and",
                    "Input: Red, Output: Apple. Input: Yellow, Output: Banana. Input: Green, Output:",
                    "The sequence is 1, 2, 3, 4. The next number is",
                    "Alice went to the store. Alice bought an apple. Alice bought a",
                    "Paris is in France. Rome is in Italy. Berlin is in",
                    #facts
                    "The largest planet in our solar system is",
                    "The chemical symbol for water is",
                    "The capital of the United Kingdom is",
                    "The first month of the year is",
                    "The capital of Germany is",
                    "The currency used in Japan is the",
                    "The author of the play Romeo and Juliet is William",
                    "Water freezes at zero degrees",
                    "The speed of light in a"
                    # Sequences
                    "One, two, three, four, five, six,",
                    "Monday, Tuesday, Wednesday, Thursday,",
                    "January, February, March, April, May,",
                    "A, B, C, D, E, F,"
                    # very strong syntactic constraints
                    "Neither here nor",
                    "Between a rock and a hard",
                    "An eye for an",
                    "The more the",
                    "From head to",
                   ]


def main():
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    baseline_results = []
    activations = []
    threshold = 0.75

    print(f"--- Generating Baseline Dataset for {manager.model_tag} ---")
    for p in BASELINE_PROMPTS:
        res = manager.get_inference_data(p)
        if res["confidence"] > threshold:
            baseline_results.append({
                "prompt": p, "prediction": res["prediction"], "confidence": res["confidence"], "type": "baseline"
            })
            activations.append(res["activation"])

    # Save mean anchor for residual analysis
    if activations:
        x_base = torch.stack(activations).mean(dim=0)
        torch.save(x_base, os.path.join(manager.data_dir, "common_certainty_baseline.pt"))

    # Save metadata
    manager.export_json(baseline_results, "baseline_inputs.json")
    print(f"Done! Saved {len(baseline_results)} samples.")


if __name__ == "__main__":
    main()