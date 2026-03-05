import torch
import os
import json
import re
import random
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from config import HF_TOKEN


class UncertaintyStudyManager:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing engine: {model_name} on {self.device}")

        self.model_tag = model_name.split("/")[-1]
        self.data_dir = os.path.join("data", self.model_tag)
        os.makedirs(self.data_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            token=HF_TOKEN,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def clean_to_declarative(question: str) -> str:
        """Purifies interrogative queries into diverse declarative completions."""
        q = question.strip().replace("?", "")
        if 'occupation' in q.lower():
            name = re.sub(r"^(What is |Who is )", "", q, flags=re.IGNORECASE).replace("'s occupation", "").strip()
            return random.choice([f"The occupation of {name} is", f"{name} works as a", f"By profession, {name} is a"])
        if 'born' in q.lower() or 'birthplace' in q.lower() or 'location' in q.lower():
            subject = re.sub(r"^(In what city was |Where was |What is the location of )", "", q,
                             flags=re.IGNORECASE).replace(" born", "").strip()
            return random.choice(
                [f"The location of {subject} is", f"{subject} is located in", f"The origin of {subject} is"])
        if 'religion' in q.lower():
            name = re.sub(r"^(What is the religion of |What religion is )", "", q, flags=re.IGNORECASE).strip()
            return random.choice([f"The religious affiliation of {name} is", f"{name} follows the religion of"])
        if any(word in q.lower() for word in ['genre', 'author', 'producer', 'director']):
            match = re.search(r"(?:genre|author|producer|director) (?:of|is) (.*)", q, re.IGNORECASE)
            work = match.group(1).strip() if match else q
            return random.choice(
                [f"The category of {work} is", f"The person responsible for {work} is", f"{work} was created by"])
        q_clean = re.sub(r"^(What|Who|When|Where|How) (is|was|did|does) ", "", q, flags=re.IGNORECASE).strip()
        return f"{q_clean} is"

    def get_inference_data(self, text: str, return_activation: bool = False) -> Dict[str, Any]:
        """Runs a forward pass to extract confidence and the final hidden state."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            conf, token_id = torch.max(probs, dim=-1)
            result = {
                "confidence": conf.item(),
                "token_id": token_id.item(),
                "prediction": self.tokenizer.decode(token_id),
            }
            if return_activation:
                result["activation"] = outputs.hidden_states[-1][0, -1, :].cpu()
            return result

    def generate_baseline_c4(self, n_samples: int) -> List[Dict]:
        """Generates high-confidence baseline samples from C4, capped at 20 words."""
        print(f"--- Phase 1: Generating C4 Baseline (Target: {n_samples}) ---")
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        data = []
        for item in ds:
            words = item['text'].split()
            if not words: continue

            # Take up to 20 words for the prompt
            prompt = " ".join(words[:20])

            res = self.get_inference_data(prompt, return_activation=True)

            # Filter for high-certainty "Common Knowledge" baseline
            if res["confidence"] > 0.80:
                data.append({**res, "prompt": prompt, "type": "baseline", "source": "c4"})

            if len(data) >= n_samples: break
        return data

    def generate_epistemic_popqa(self, n_samples: int) -> List[Dict]:
        """Generates low-popularity epistemic samples from PopQA."""
        print(f"--- Phase 2: Generating Epistemic PopQA (Target: {n_samples}) ---")
        popqa = load_dataset("akariasai/PopQA", split="test")
        filtered_popqa = popqa.filter(lambda x: x['s_pop'] < 100)

        data = []
        indices = list(range(len(filtered_popqa)))
        random.shuffle(indices)

        limit = min(len(indices), n_samples)
        for i in indices[:limit]:
            item = filtered_popqa[i]
            prompt = self.clean_to_declarative(item['question'])
            res = self.get_inference_data(prompt, return_activation=True)
            data.append({**res, "prompt": prompt, "type": "epistemic", "category": item['prop']})
        return data

    def generate_aleatoric_ambig(self, n_samples: int) -> List[Dict]:
        """Generates aleatoric samples from AmbigQA."""
        print(f"--- Phase 3: Generating Aleatoric AmbigQA (Target: {n_samples}) ---")
        ambig = load_dataset("sewon/ambig_qa", "light", split="train")
        raw = ambig.filter(lambda x: 'multipleQAs' in x['annotations']['type'])

        data = []
        limit = min(len(raw), n_samples)
        for i in range(limit):
            item = raw[i]
            prompt = self.clean_to_declarative(item['question'])
            res = self.get_inference_data(prompt, return_activation=True)
            data.append({**res, "prompt": prompt, "type": "aleatoric"})
        return data

    def run_study_generation(self, target_n: int):
        """
        Orchestrates generation, balances sizes to the minimum N found,
        and performs a single coordinated save.
        """
        # 1. Collect all raw data
        raw_results = {
            "baseline": self.generate_baseline_c4(target_n),
            "epistemic": self.generate_epistemic_popqa(target_n),
            "aleatoric": self.generate_aleatoric_ambig(target_n)
        }

        # 2. Determine the balancing factor (N_min)
        min_len = min(len(d) for d in raw_results.values())
        print(f"\n--- Balancing: Truncating all datasets to N={min_len} ---")

        # 3. Clear existing master file
        master_path = os.path.join(self.data_dir, "uncertainty_study_dataset.jsonl")
        if os.path.exists(master_path):
            os.remove(master_path)

        # 4. Save individual files and the master study file
        for dtype, items in raw_results.items():
            balanced_subset = items[:min_len]

            # Extract and save the latent anchor (mean activation)
            activations = [it["activation"] for it in balanced_subset]
            x_mean = torch.stack(activations).mean(dim=0)
            torch.save(x_mean, os.path.join(self.data_dir, f"{dtype}_certainty_baseline.pt"))

            # Save JSONL (Master and Subset)
            self.export_jsonl(balanced_subset, f"{dtype}_dataset.jsonl", mode='a')

        print(f"Successfully generated balanced study with {min_len * 3} total samples.")

    def export_jsonl(self, data: List[Dict], filename: str, mode: str = 'w'):
        """Standardized export. Converts tensors to lists for JSON compatibility."""
        specific_path = os.path.join(self.data_dir, filename)
        master_path = os.path.join(self.data_dir, "uncertainty_study_dataset.jsonl")

        processed = []
        for item in data:
            item_copy = item.copy()
            if "activation" in item_copy and isinstance(item_copy["activation"], torch.Tensor):
                item_copy["activation"] = item_copy["activation"].tolist()
            processed.append(item_copy)

        with open(specific_path, 'w') as f:
            for d in processed: f.write(json.dumps(d) + "\n")
        with open(master_path, mode) as f:
            for d in processed: f.write(json.dumps(d) + "\n")