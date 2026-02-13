import torch
import os
import json
import re
import random
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


class UncertaintyStudyManager:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initializing engine: {model_name} on {self.device}")

        self.model_tag = model_name.split("/")[-1]
        self.data_dir = os.path.join("data", self.model_tag)
        os.makedirs(self.data_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
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

    def get_inference_data(self, text: str) -> Dict[str, Any]:
        """Runs a forward pass to extract confidence and the final hidden state."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            conf, token_id = torch.max(probs, dim=-1)
            return {
                "confidence": conf.item(),
                "token_id": token_id.item(),
                "prediction": self.tokenizer.decode(token_id),
                "activation": outputs.hidden_states[-1][0, -1, :].cpu(),
                "input_ids": inputs["input_ids"][0].tolist()
            }

    def generate_baseline_c4(self, n_samples: int) -> List[Dict]:
        """Generates high-confidence baseline samples from C4."""
        print(f"--- Phase 1: Generating C4 Baseline ({n_samples} samples) ---")
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        data, activations = [], []
        for item in ds:
            words = item['text'].split()
            if len(words) < 25: continue
            prompt = " ".join(words[:20])
            res = self.get_inference_data(prompt)
            if res["confidence"] > 0.80:
                data.append({**res, "prompt": prompt, "type": "baseline", "source": "c4"})
                activations.append(res["activation"])
            if len(data) >= n_samples: break

        # Save the anchor
        x_base = torch.stack(activations).mean(dim=0)
        torch.save(x_base, os.path.join(self.data_dir, "common_certainty_baseline.pt"))
        return data

    def generate_epistemic_popqa(self, n_samples: int) -> List[Dict]:
        """Generates low-popularity epistemic samples from PopQA."""
        print(f"--- Phase 2: Generating Epistemic PopQA ({n_samples} samples) ---")
        popqa = load_dataset("akariasai/PopQA", split="test")
        all_cats = list(set(popqa['prop']))
        samples_per_cat = n_samples // len(all_cats)
        data = []
        for cat in all_cats:
            filtered = popqa.filter(lambda x: x['prop'] == cat and x['s_pop'] < 100)
            subset = filtered.select(range(min(len(filtered), samples_per_cat)))
            for item in subset:
                prompt = self.clean_to_declarative(item['question'])
                res = self.get_inference_data(prompt)
                data.append({**res, "prompt": prompt, "type": "epistemic", "category": cat})
        return data

    def generate_aleatoric_ambig(self, n_samples: int) -> List[Dict]:
        """Generates aleatoric samples from AmbigQA."""
        print(f"--- Phase 3: Generating Aleatoric AmbigQA ({n_samples} samples) ---")
        ambig = load_dataset("sewon/ambig_qa", "light", split="train")
        raw = ambig.filter(lambda x: 'multipleQAs' in x['annotations']['type']).select(range(n_samples))
        data = []
        for item in raw:
            prompt = self.clean_to_declarative(item['question'])
            res = self.get_inference_data(prompt)
            data.append({**res, "prompt": prompt, "type": "aleatoric"})
        return data

    def export_jsonl(self, data: List[Dict], filename: str, mode: str = 'w'):
        """Standardized export to specific file and master study file."""
        specific_path = os.path.join(self.data_dir, filename)
        master_path = os.path.join(self.data_dir, "uncertainty_study_dataset.jsonl")

        processed = []
        for item in data:
            if "activation" in item and isinstance(item["activation"], torch.Tensor):
                item["activation"] = item["activation"].tolist()
            processed.append(item)

        with open(specific_path, 'w') as f:
            for d in processed: f.write(json.dumps(d) + "\n")
        with open(master_path, mode) as f:
            for d in processed: f.write(json.dumps(d) + "\n")

