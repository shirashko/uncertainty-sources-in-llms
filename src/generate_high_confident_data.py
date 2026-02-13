import os
import torch
from datasets import load_dataset
from config import MODEL_ID
from data_generation import UncertaintyStudyManager


def main(n_samples=200):
    manager = UncertaintyStudyManager(model_name=MODEL_ID)

    # We want "normal" certain behavior. 0.80 is a good threshold for C4.
    CONF_THRESHOLD = 0.80
    baseline_results = []
    activations = []

    print(f"--- Generating C4 Baseline for {manager.model_tag} ---")

    # Streaming C4 to avoid downloading 300GB+
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    count = 0
    for item in ds:
        text = item['text']
        # Take a snippet (first 15-20 words) to use as a prompt
        words = text.split()
        if len(words) < 25: continue

        prompt = " ".join(words[:20])
        res = manager.get_inference_data(prompt)

        if res["confidence"] > CONF_THRESHOLD:
            baseline_results.append({
                "prompt": prompt,
                "prediction": res["prediction"],
                "confidence": res["confidence"],
                "type": "baseline",
                "source": "c4"
            })
            activations.append(res["activation"])
            count += 1
            if count % 20 == 0:
                print(f"  Captured {count}/{n_samples} C4 samples...")

        if count >= n_samples:
            break

    if activations:
        x_base = torch.stack(activations).mean(dim=0)
        torch.save(x_base, os.path.join(manager.data_dir, "common_certainty_baseline.pt"))
        manager.export_json(baseline_results, "baseline_inputs.json")
        print(f"Success! Saved {len(baseline_results)} C4 baseline samples.")


if __name__ == "__main__":
    main(n_samples=200)