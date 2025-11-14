"""Heart Disease baseline file for classification evaluation."""

from __future__ import annotations

from typing import Any, Dict
import json

from datasets import load_dataset

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.inference import InferenceClient
import os
import httpx


# Load dataset once at module level
_dataset = None


def _load_dataset():
    """Load Heart Disease dataset."""
    global _dataset
    if _dataset is None:
        _dataset = load_dataset("buio/heart-disease", split="train")
    return _dataset


class HeartDiseaseTaskRunner(BaselineTaskRunner):
    """Task runner for Heart Disease classification."""

    def __init__(self, policy_config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(policy_config, env_config)

        # Load dataset
        self.dataset = _load_dataset()

        # Store config for inference
        self.model = policy_config["model"]
        self.temperature = policy_config.get("temperature", 0.0)
        self.max_tokens = policy_config.get("max_tokens", 128)
        self.inference_url = policy_config.get("inference_url")

        # Tool definition for heart disease classification
        self.tool = {
            "type": "function",
            "function": {
                "name": "heart_disease_classify",
                "description": "Classify whether a patient has heart disease",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["0", "1"],
                            "description": "The classification: '1' for heart disease, '0' for no heart disease",
                        }
                    },
                    "required": ["classification"],
                },
            },
        }

    async def run_task(self, seed: int) -> TaskResult:
        """Run a single Heart Disease classification task."""

        # Get example from dataset
        example = self.dataset[seed]

        # Extract features and label
        features = {}
        label = None

        for key, value in example.items():
            if key in ("target", "label", "class", "disease"):
                # This is the label
                label = str(int(value)) if isinstance(value, (int, float)) else str(value)
            else:
                # These are features
                features[key] = value

        # Format features as text
        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])

        # Build prompt
        system_prompt = """You are a medical classification assistant. Based on the patient's features, classify whether they have heart disease.

Call the `heart_disease_classify` tool with:
- classification = "1" for heart disease
- classification = "0" for no heart disease"""

        user_prompt = f"Patient Features:\n{feature_text}\n\nClassify: Does this patient have heart disease?"

        # Run inference
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use InferenceClient if URL provided, otherwise use OpenAI-compatible API
        if self.inference_url and self.inference_url.startswith("http"):
            api_key = os.getenv("SYNTH_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            base_url = self.inference_url.rstrip("/")
            if not base_url.endswith("/api"):
                base_url = f"{base_url}/api" if "/api" not in base_url else base_url
            client = InferenceClient(base_url=base_url, api_key=api_key)
            response = await client.create_chat_completion(
                model=self.model,
                messages=messages,
                tools=[self.tool],
                tool_choice={"type": "function", "function": {"name": "heart_disease_classify"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            # Use OpenAI/Groq directly
            # Check if model starts with groq: prefix
            model_name = self.model
            use_groq = model_name.startswith("groq:")
            if use_groq:
                model_name = model_name[5:]  # Remove "groq:" prefix

            api_key = os.getenv("GROQ_API_KEY") if use_groq else os.getenv("OPENAI_API_KEY") or ""
            base_url = "https://api.groq.com/openai/v1" if use_groq else "https://api.openai.com/v1"
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "tools": [self.tool],
                        "tool_choice": {"type": "function", "function": {"name": "heart_disease_classify"}},
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                )
                response = resp.json()

        # Extract prediction
        predicted_label = ""
        tool_calls = []
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            tool_calls = message.get("tool_calls", [])
        elif "tool_calls" in response:
            tool_calls = response["tool_calls"]

        if tool_calls:
            # Handle both string and dict arguments
            args = tool_calls[0]["function"].get("arguments", "")
            if isinstance(args, str):
                args = json.loads(args)
            predicted_label = args.get("classification", "") if isinstance(args, dict) else ""

        # Normalize prediction
        predicted_label = predicted_label.strip()
        if predicted_label not in ("0", "1"):
            # Try to extract digit
            for char in predicted_label:
                if char in ("0", "1"):
                    predicted_label = char
                    break

        # Evaluate
        expected_label = label or "0"
        correct = predicted_label == expected_label

        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=1.0 if correct else 0.0,
            total_steps=1,
            metadata={
                "features": feature_text,
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": correct,
            },
        )


# Define baseline config
_load_dataset()
heartdisease_baseline = BaselineConfig(
    baseline_id="heartdisease",
    name="Heart Disease Classification",
    description="Heart disease classification baseline for prompt optimization experiments",
    task_runner=HeartDiseaseTaskRunner,
    splits={
        "train": DataSplit(
            name="train",
            seeds=list(range(min(1000, len(_dataset))) if _dataset else range(1000)),
        ),
    },
    default_policy_config={
        "model": "groq:llama-3.1-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 128,
    },
    default_env_config={},
    metadata={
        "dataset": "buio/heart-disease",
        "num_classes": 2,
        "task_type": "binary_classification",
    },
    tags=["classification", "healthcare", "blog-post"],
)
