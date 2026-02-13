import torch
import os
import json
import re
import random
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class UncertaintyStudyManager:
    """
    Core engine for uncertainty analysis.
    Handles model loading, diverse declarative cleaning, and activation extraction.
    """
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initializing engine: {model_name} on {self.device}")

        self.model_tag = model_name.split("/")[-1]
        self.data_dir = os.path.join("data", self.model_tag)
        os.makedirs(self.data_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using float16 for M4 efficiency and memory management
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def clean_to_declarative(question: str) -> str:
        """
        Rewrites questions into diverse declarative formats to prevent syntactic bias.
        Ensures results capture semantic uncertainty rather than pattern matching.
        """
        q = question.strip()
        if q.endswith("?"):
            q = q[:-1]

        # Randomized templates for common PopQA categories [cite: 11]
        if 'occupation' in q.lower() or 'profession' in q.lower():
            name = re.sub(r"^(What is |Who is )", "", q, flags=re.IGNORECASE).replace("'s occupation", "").strip()
            return random.choice([
                f"The occupation of {name} is",
                f"By profession, {name} is a",
                f"{name} works as a",
                f"The job held by {name} is"
            ])

        if 'born' in q.lower() or 'birth' in q.lower():
            name = re.sub(r"^(In what city was |Where was |What is the birthplace of )", "", q, flags=re.IGNORECASE).replace(" born", "").strip()
            return random.choice([
                f"{name} was born in the city of",
                f"The birthplace of {name} is",
                f"{name} originally comes from"
            ])

        if 'capital' in q.lower() or 'location' in q.lower():
            place = re.sub(r"^(What is the capital of |Where is )", "", q, flags=re.IGNORECASE).strip()
            return random.choice([
                f"The capital city of {place} is",
                f"Located in {place}, the main city is",
                f"{place}'s capital is"
            ])

        # General declarative fallback
        q_clean = re.sub(r"^(What|Who|When|Where|How) (is|was|did|does) ", "", q, flags=re.IGNORECASE).strip()
        if q_clean.lower().endswith(" is") or q_clean.lower().endswith(" was"):
            return q_clean
        return f"{q_clean} is"

    def get_inference_data(self, text: str) -> Dict[str, Any]:
        """Runs a forward pass to extract confidence and the final hidden state."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract logit probability for the final predicted token [cite: 93]
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
            conf, token_id = torch.max(probs, dim=-1)

            return {
                "confidence": conf.item(),
                "token_id": token_id.item(),
                "prediction": self.tokenizer.decode(token_id),
                "activation": outputs.hidden_states[-1][0, -1, :].cpu(), # Move to CPU for storage
                "input_ids": inputs["input_ids"][0].tolist()
            }

    def export_json(self, data: List[Dict], filename: str):
        """Standardized export for baseline results."""
        path = os.path.join(self.data_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Exported {len(data)} records to {path}")

    def export_jsonl(self, data: List[Dict], filename: str):
        """
        Standardized export for uncertainty datasets.
        Writes to a specific file (overwrite) and appends to a master study file.
        """
        # 1. Path for the specific file (e.g., epistemic_dataset.jsonl)
        specific_path = os.path.join(self.data_dir, filename)

        # 2. Path for the master combined file (e.g., uncertainty_study_dataset.jsonl)
        master_path = os.path.join(self.data_dir, "uncertainty_study_dataset.jsonl")

        # Deep copy or handle data to ensure tensors are converted only once for JSON
        processed_data = []
        for item in data:
            # Convert activation tensors to lists for JSON serialization
            if "activation" in item and isinstance(item["activation"], torch.Tensor):
                item["activation"] = item["activation"].tolist()
            processed_data.append(item)

        # Write to the specific file (Mode 'w' - overwrites each run)
        with open(specific_path, "w") as f_spec:
            for item in processed_data:
                f_spec.write(json.dumps(item) + "\n")

        # Append to the master study file (Mode 'a' - adds to existing content)
        with open(master_path, "a") as f_master:
            for item in processed_data:
                f_master.write(json.dumps(item) + "\n")

        print(f"Done: {len(data)} records saved to {specific_path} and appended to {master_path}")