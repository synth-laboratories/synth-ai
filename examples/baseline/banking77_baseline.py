"""Banking77 baseline file for intent classification evaluation."""

from __future__ import annotations

from typing import Any, Dict

from datasets import load_dataset

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.inference import InferenceClient
import os
import httpx


# Load dataset once at module level
_dataset = None
_label_names = None


def _load_dataset():
    """Load Banking77 dataset."""
    global _dataset, _label_names
    if _dataset is None:
        try:
            _dataset = load_dataset("PolyAI/banking77")
        except Exception:
            # Fallback: try without org prefix
            _dataset = load_dataset("banking77")
        _label_names = _dataset["train"].features["label"].names
    return _dataset, _label_names


class Banking77TaskRunner(BaselineTaskRunner):
    """Task runner for Banking77 intent classification."""
    
    def __init__(self, policy_config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(policy_config, env_config)
        
        # Load dataset
        self.dataset, self.label_names = _load_dataset()
        
        # Store config for inference
        self.model = policy_config["model"]
        self.temperature = policy_config.get("temperature", 0.0)
        self.max_tokens = policy_config.get("max_tokens", 128)
        self.inference_url = policy_config.get("inference_url")
        
        # Tool definition
        self.tool = {
            "type": "function",
            "function": {
                "name": "banking77_classify",
                "description": "Classify a banking query into an intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": self.label_names,
                            "description": "The intent label",
                        }
                    },
                    "required": ["label"],
                },
            },
        }
    
    async def run_task(self, seed: int) -> TaskResult:
        """Run a single Banking77 classification task."""
        
        # Get split
        split = self.env_config.get("split", "train")
        
        # Get example from dataset
        example = self.dataset[split][seed]
        
        # Build prompt
        system_prompt = f"""You are an expert banking assistant that classifies customer queries.
Given a customer message, respond with exactly one intent label using the tool call.

Valid intents: {', '.join(self.label_names)}"""
        
        user_prompt = f"Customer Query: {example['text']}\n\nClassify this query."
        
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
                tool_choice={"type": "function", "function": {"name": "banking77_classify"}},
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
                        "tool_choice": {"type": "function", "function": {"name": "banking77_classify"}},
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
                import json
                args = json.loads(args)
            predicted_label = args.get("label", "") if isinstance(args, dict) else ""
        
        # Evaluate
        expected_label = self.label_names[example["label"]]
        correct = predicted_label == expected_label
        
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=1.0 if correct else 0.0,
            total_steps=1,
            metadata={
                "query": example["text"],
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": correct,
                "split": split,
            },
        )


# Define baseline config
# Note: We need to load the dataset first to get the label names
_load_dataset()
banking77_baseline = BaselineConfig(
    baseline_id="banking77",
    name="Banking77 Intent Classification",
    description="Banking intent classification from customer queries",
    task_runner=Banking77TaskRunner,
    splits={
        "train": DataSplit(
            name="train",
            seeds=list(range(min(10000, len(_dataset["train"]))) if _dataset else range(10000)),
        ),
        "val": DataSplit(
            name="val",
            seeds=list(range(min(1000, len(_dataset["test"]))) if _dataset else range(1000)),
        ),
        "test": DataSplit(
            name="test",
            seeds=list(range(min(3000, len(_dataset["test"]))) if _dataset else range(3000)),
        ),
    },
    default_policy_config={
        "model": "groq:llama-3.1-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 128,
    },
    default_env_config={
        "split": "train",
    },
    metadata={
        "dataset": "PolyAI/banking77",
        "num_classes": 77,
        "task_type": "classification",
    },
    tags=["classification", "nlp", "intent"],
)

