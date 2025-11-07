"""Baseline evaluation for the Banking77 two-step pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Mapping

import httpx
from datasets import load_dataset

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.inference.client import InferenceClient


TOOL_NAME = "banking77_classify"
_dataset = None
_label_names: list[str] | None = None


def _load_dataset():
    global _dataset, _label_names
    if _dataset is None:
        try:
            _dataset = load_dataset("PolyAI/banking77")
        except Exception:
            _dataset = load_dataset("banking77")
        _label_names = _dataset["train"].features["label"].names
    return _dataset, _label_names


def _format_available_intents() -> str:
    _, label_names = _load_dataset()
    label_names = label_names or []
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


CLASSIFIER_SYSTEM_PROMPT = (
    "You are an expert banking assistant. Classify the customer query into one of the "
    "known Banking77 intents. Always respond using the `banking77_classify` tool."
)

CLASSIFIER_USER_TEMPLATE = (
    "Customer Query: {query}\n\n"
    "Classify this query into one of the banking intents using the tool call."
)

CALIBRATOR_SYSTEM_PROMPT = (
    "You refine intent predictions from an upstream classifier. Review the suggested "
    "intent alongside the original query. If the suggestion is valid, confirm it. "
    "Otherwise, choose the closest Banking77 intent. Always respond via the "
    "`banking77_classify` tool with the final label."
)

CALIBRATOR_USER_TEMPLATE = (
    "Original Customer Query: {query}\n"
    "Classifier Suggested Intent: {candidate_intent}\n\n"
    "Return the best final intent via the tool call."
)


class Banking77PipelineTaskRunner(BaselineTaskRunner):
    """Baseline runner that evaluates the two-step pipeline locally."""

    def __init__(self, policy_config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(policy_config, env_config)

        self.dataset, self.label_names = _load_dataset()
        self.model = policy_config["model"]
        self.temperature = policy_config.get("temperature", 0.0)
        self.max_tokens = policy_config.get("max_tokens", 128)
        self.inference_url = policy_config.get("inference_url")

        self.tool_schema = {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Return the predicted banking77 intent label in the `intent` field.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "Predicted Banking77 intent label.",
                        }
                    },
                    "required": ["intent"],
                },
            },
        }

    async def run_task(self, seed: int) -> TaskResult:
        split = self.env_config.get("split", "train")
        example = self.dataset[split][seed]

        available_intents = _format_available_intents()
        query = example["text"]

        classifier_messages = self._build_messages(
            module="classifier",
            query=query,
            available_intents=available_intents,
            candidate_intent="",
        )
        classifier_response = await self._invoke_model(classifier_messages)
        classifier_intent, classifier_record = self._extract_intent("classifier", classifier_response)

        calibrator_messages = self._build_messages(
            module="calibrator",
            query=query,
            available_intents=available_intents,
            candidate_intent=classifier_intent,
        )
        calibrator_response = await self._invoke_model(calibrator_messages)
        calibrator_intent, calibrator_record = self._extract_intent("calibrator", calibrator_response)

        final_intent = calibrator_intent or classifier_intent
        expected_intent = self.label_names[example["label"]]
        correct = final_intent.lower().replace("_", " ") == expected_intent.lower().replace("_", " ")
        reward = 1.0 if correct else 0.0

        metadata = {
            "query": query,
            "expected_intent": expected_intent,
            "classifier": classifier_record,
            "calibrator": calibrator_record,
            "final_intent": final_intent,
            "correct": correct,
            "split": split,
        }

        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=reward,
            total_steps=2,
            metadata=metadata,
        )

    def _build_messages(
        self,
        *,
        module: str,
        query: str,
        available_intents: str,
        candidate_intent: str,
    ) -> list[dict[str, str]]:
        if module == "classifier":
            system_prompt = CLASSIFIER_SYSTEM_PROMPT + "\n\nAvailable Intents:\n" + available_intents
            user_prompt = CLASSIFIER_USER_TEMPLATE.format(query=query)
        else:
            system_prompt = CALIBRATOR_SYSTEM_PROMPT + "\n\nAvailable Intents:\n" + available_intents
            user_prompt = CALIBRATOR_USER_TEMPLATE.format(
                query=query,
                candidate_intent=candidate_intent or "<none>",
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _invoke_model(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        tool_choice = {"type": "function", "function": {"name": TOOL_NAME}}

        if self.inference_url and self.inference_url.startswith("http"):
            api_key = (
                os.getenv("SYNTH_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("GROQ_API_KEY")
                or ""
            )
            base_url = self.inference_url.rstrip("/")
            if not base_url.endswith("/api"):
                base_url = f"{base_url}/api"
            client = InferenceClient(base_url=base_url, api_key=api_key)
            return await client.create_chat_completion(
                model=self.model,
                messages=messages,
                tools=[self.tool_schema],
                tool_choice=tool_choice,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        model_name = self.model
        use_groq = False
        if model_name.startswith("groq:"):
            model_name = model_name.split(":", 1)[1]
            use_groq = True

        api_key = os.getenv("GROQ_API_KEY") if use_groq else os.getenv("OPENAI_API_KEY") or ""
        base_url = "https://api.groq.com/openai/v1" if use_groq else "https://api.openai.com/v1"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "tools": [self.tool_schema],
                    "tool_choice": tool_choice,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            response.raise_for_status()
            return response.json()

    def _extract_intent(
        self,
        module: str,
        response: Mapping[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        choices = response.get("choices") or []
        tool_calls: Iterable[Mapping[str, Any]] = []
        content_text = ""
        if choices:
            message = (choices[0] or {}).get("message", {}) or {}
            tool_calls = message.get("tool_calls", []) or []
            content_text = str(message.get("content", ""))
        intent = self._parse_intent(tool_calls, content_text)
        if not intent:
            raise RuntimeError(f"Module {module}: model response missing tool call intent")

        record = {
            "response": response,
            "tool_calls": list(tool_calls),
            "predicted_intent": intent,
        }
        return intent, record

    @staticmethod
    def _parse_intent(
        tool_calls: Iterable[Mapping[str, Any]],
        content_text: str,
    ) -> str:
        for call in tool_calls:
            function = (call or {}).get("function", {}) or {}
            if function.get("name") != TOOL_NAME:
                continue
            args_raw = function.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                continue
            intent = str(args.get("intent", "")).strip()
            if intent:
                return intent
        if content_text.strip():
            return content_text.strip().split()[0]
        return ""


_load_dataset()
banking77_pipeline_baseline = BaselineConfig(
    baseline_id="banking77_pipeline",
    name="Banking77 Two-Step Pipeline",
    description="Two-stage Banking77 classification baseline with classifier and calibrator modules.",
    task_runner=Banking77PipelineTaskRunner,
    splits={
        "train": DataSplit(
            name="train",
            seeds=list(range(min(10000, len(_dataset["train"])) if _dataset else 10000)),
        ),
        "val": DataSplit(
            name="val",
            seeds=list(range(min(1000, len(_dataset["test"])) if _dataset else 1000)),
        ),
        "test": DataSplit(
            name="test",
            seeds=list(range(min(3000, len(_dataset["test"])) if _dataset else 3000)),
        ),
    },
    default_policy_config={
        "model": "groq:llama-3.1-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 256,
    },
    default_env_config={
        "split": "train",
    },
    metadata={
        "dataset": "PolyAI/banking77",
        "num_classes": 77,
        "task_type": "pipeline_classification",
        "modules": ["classifier", "calibrator"],
    },
    tags=["classification", "pipeline", "multi-step"],
)



