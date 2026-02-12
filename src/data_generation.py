import torch
import os
import json
import numpy as np
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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


def export_results(baseline, uncertain, stats):
    """Saves logs and prints formatted summary table."""
    with open("data/baseline_inputs.json", "w") as f:
        json.dump(baseline, f, indent=4)

    with open("data/uncertainty_study_dataset.jsonl", "w") as f:
        for item in uncertain:
            f.write(json.dumps(item) + "\n")

    print("\n" + "=" * 50)
    print(f"{'Category':<15} | {'Count':<8} | {'Avg Confidence':<15}")
    print("-" * 50)
    for cat, scores in stats.items():
        avg = np.mean(scores) if scores else 0
        print(f"{cat.capitalize():<15} | {len(scores):<8} | {avg:.4f}")
    print("=" * 50)


class UncertaintyStudyManager:
    """
    A manager to analyze LLM uncertainty across different models.
    Supports model-agnostic loading and automated prompt cleaning.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer loaded: vocab size = {len(self.tokenizer)}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Ensure padding token exists (critical for batching and certain models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def clean_to_declarative(question: str) -> str:
        """Purifies interrogative queries into declarative completions."""
        q = question.strip()
        if q.endswith("?"):
            q = q[:-1]

        # Mapping logic for specific domains
        if 'occupation' in q.lower():
            name = re.sub(r"^(What is |Who is )", "", q, flags=re.IGNORECASE).replace("'s occupation", "").strip()
            return f"The occupation of {name} is"

        if 'born' in q.lower():
            name = re.sub(r"^(In what city was |Where was )", "", q, flags=re.IGNORECASE).replace(" born", "").strip()
            return f"{name} was born in the city of"

        if 'genre' in q.lower():
            work = re.sub(r"^What genre is ", "", q, flags=re.IGNORECASE).strip()
            return f"The genre of {work} is"

        # General fallback
        q = re.sub(r"^(What|Who|When|Where|How) (is|was|did|does) ", "", q, flags=re.IGNORECASE).strip()
        if q.lower().endswith(" is") or q.lower().endswith(" was"):
            return q
        return f"{q} is"

    def get_inference_data(self, text: str) -> Dict[str, Any]:
        """Extracts model confidence and hidden states for the last token."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # Calculate softmax probabilities for the final predicted token
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            conf, token_id = torch.max(probs, dim=-1)

            return {
                "confidence": conf.item(),
                "token_id": token_id.item(),
                "prediction": self.tokenizer.decode(token_id),
                "activation": outputs.hidden_states[-1][0, -1, :].cpu(),
                "input_ids": inputs["input_ids"][0].tolist()

            }

    def run_study(self, n_samples: int = 300, threshold: float = 0.75):
        """Main execution pipeline for baseline and uncertainty comparison."""
        os.makedirs("data", exist_ok=True)

        baseline_results = []
        activations_list = []
        print(f"Analyzing Baseline (Threshold: {threshold})...")
        for p in tqdm(BASELINE_PROMPTS):
            res = self.get_inference_data(p)
            if res["confidence"] > threshold:
                activations_list.append(res["activation"])
                baseline_results.append({
                    "prompt": p,
                    "prediction": res["prediction"],
                    "confidence": res["confidence"],
                    "input_ids": res["input_ids"]
                })

        # Calculate and save the anchor (x_base)
        if activations_list:
            x_base = torch.stack(activations_list).mean(dim=0)
            torch.save(x_base, "data/common_certainty_baseline.pt")

        # 2. Process Uncertain Samples
        uncertain_data = self._load_datasets(n_samples)
        final_uncertain_records = []
        data = {"baseline": [r["confidence"] for r in baseline_results], "epistemic": [], "aleatoric": []}

        print("Analyzing Epistemic and Aleatoric data...")
        for item in tqdm(uncertain_data):
            res = self.get_inference_data(item["prompt"])
            data[item["type"]].append(res["confidence"])
            final_uncertain_records.append({**item, "confidence": res["confidence"]})

        # 3. Export Results
        export_results(baseline_results, final_uncertain_records, data)

    def _load_datasets(self, n_samples: int) -> List[Dict[str, str]]:
        """Handles external dataset loading and preprocessing."""
        data = []
        # Epistemic (PopQA)
        popqa = load_dataset("akariasai/PopQA", split="test")
        epistemic = popqa.filter(lambda x: x['s_pop'] < 100).select(range(min(n_samples, len(popqa))))
        for item in epistemic:
            data.append({"prompt": self.clean_to_declarative(item['question']), "type": "epistemic"})

        # Aleatoric (AmbigQA)
        ambigqa = load_dataset("sewon/ambig_qa", "light", split="train")
        aleatoric = ambigqa.filter(lambda x: 'multipleQAs' in x['annotations']['type']).select(
            range(min(n_samples, len(ambigqa))))
        for item in aleatoric:
            data.append({"prompt": self.clean_to_declarative(item['question']), "type": "aleatoric"})

        return data

if __name__ == "__main__":
    analyzer = UncertaintyStudyManager(model_name="gpt2")
    analyzer.run_study(n_samples=100)