#!/usr/bin/env python3
"""Local API for GEPA prompt optimization over LLM-based tabular featurization.

This task app treats the system prompt as the featurization prompt. It calls an
LLM to generate a fixed-length numeric feature vector for each row, then trains
an XGBoost model with frozen hyperparameters and scores on a validation split.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - import error will be raised at runtime
    raise RuntimeError(
        "xgboost is required for this demo. Install with: uv pip install xgboost"
    ) from exc

from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl.validators import normalize_inference_url
from synth_ai.sdk.localapi.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo

APP_ID = "boosting"
APP_NAME = "Boosting Featurization (GEPA + XGBoost)"
TOOL_NAME = "featurize_row"
NUM_FEATURES = int(os.getenv("BOOSTING_NUM_FEATURES", "10"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a feature engineer. Given a single tabular row (as text), "
    "produce a JSON tool call with a fixed-length numeric feature vector that "
    "captures useful non-linear interactions for downstream XGBoost. "
    "Keep outputs deterministic and numeric only."
)

DEFAULT_USER_PROMPT = (
    "Row:\n{row_text}\n\n"
    "Return a tool call with exactly "
    f"{NUM_FEATURES} numeric features in the 'features' array."
)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return engineered features as a fixed-length numeric vector.",
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": NUM_FEATURES,
                    "maxItems": NUM_FEATURES,
                }
            },
            "required": ["features"],
        },
    },
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 1337,
}

TRAIN_SIZE = int(os.getenv("BOOSTING_TRAIN_SIZE", "200"))
VAL_SIZE = int(os.getenv("BOOSTING_VAL_SIZE", "80"))


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    train_idx: np.ndarray
    val_idx: np.ndarray


def _load_dataset() -> DatasetBundle:
    dataset = load_breast_cancer()
    X = dataset.data.astype(float)
    y = dataset.target.astype(int)
    feature_names = list(dataset.feature_names)

    rng = np.random.default_rng(1337)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    train_size = min(TRAIN_SIZE, len(indices))
    val_size = min(VAL_SIZE, len(indices) - train_size)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]

    return DatasetBundle(X=X, y=y, feature_names=feature_names, train_idx=train_idx, val_idx=val_idx)


DATASET = _load_dataset()


def _row_to_text(row: np.ndarray, feature_names: Iterable[str]) -> str:
    parts = [f"{name}: {value:.6f}" for name, value in zip(feature_names, row)]
    return "\n".join(parts)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _extract_system_prompt(policy_config: dict[str, Any] | None, fallback: str) -> str:
    if not policy_config:
        return fallback

    for key in ("system_prompt", "prompt", "system"):
        value = policy_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = policy_config.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content") or msg.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    prompt = policy_config.get("prompt")
    if isinstance(prompt, dict):
        msg_list = prompt.get("messages") or []
        for msg in msg_list:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content") or msg.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return fallback


def _render_messages(
    *,
    system_prompt: str,
    row_text: str,
    policy_config: dict[str, Any] | None,
) -> list[dict[str, str]]:
    messages = policy_config.get("messages") if policy_config else None
    if isinstance(messages, list) and messages:
        rendered = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            content = msg.get("content") or msg.get("pattern") or ""
            if "{row_text}" in content:
                content = content.replace("{row_text}", row_text)
            rendered.append({"role": role, "content": content})
        if rendered:
            return rendered

    user_prompt = DEFAULT_USER_PROMPT.replace("{row_text}", row_text)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _call_llm(
    *,
    row_text: str,
    system_prompt: str,
    inference_url: str,
    model: str,
    api_key: str | None,
    policy_config: dict[str, Any] | None,
) -> tuple[list[float], float]:
    messages = _render_messages(system_prompt=system_prompt, row_text=row_text, policy_config=policy_config)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "tools": [TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "temperature": 0,
    }

    url = normalize_inference_url(inference_url)
    timeout_seconds = float(os.getenv("BOOSTING_LLM_TIMEOUT", "120"))

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = str(error_json.get("error", {}).get("message", error_json))
            except Exception:
                error_msg = response.text[:500]
            raise RuntimeError(f"Proxy error ({response.status_code}): {error_msg}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from model")
        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            raise RuntimeError("No tool calls returned from model")
        args_raw = tool_calls[0].get("function", {}).get("arguments")

    if not args_raw:
        raise RuntimeError("No tool call arguments returned from model")

    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    features = args.get("features") if isinstance(args, dict) else None
    if not isinstance(features, list):
        raise RuntimeError("Tool call did not return a 'features' list")

    parsed = []
    for value in features[:NUM_FEATURES]:
        try:
            parsed.append(float(value))
        except Exception:
            parsed.append(0.0)

    if len(parsed) < NUM_FEATURES:
        parsed.extend([0.0] * (NUM_FEATURES - len(parsed)))

    latency_ms = (time.perf_counter() - start) * 1000.0
    return parsed, latency_ms


class FeatureCache:
    def __init__(self) -> None:
        self._features: dict[tuple[str, int], list[float]] = {}
        self._invalid: dict[str, int] = {}

    def get(self, prompt_key: str, idx: int) -> list[float] | None:
        return self._features.get((prompt_key, idx))

    def set(self, prompt_key: str, idx: int, features: list[float], valid: bool) -> None:
        self._features[(prompt_key, idx)] = features
        if not valid:
            self._invalid[prompt_key] = self._invalid.get(prompt_key, 0) + 1

    def invalid_count(self, prompt_key: str) -> int:
        return self._invalid.get(prompt_key, 0)


FEATURE_CACHE = FeatureCache()
SCORE_CACHE: dict[str, dict[str, Any]] = {}


async def score_prompt(
    *,
    system_prompt: str,
    inference_url: str,
    model: str,
    api_key: str | None,
    policy_config: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt_key = _hash_prompt(system_prompt)
    cached = SCORE_CACHE.get(prompt_key)
    if cached:
        return cached

    invalid = 0
    latency_ms = 0.0

    async def featurize_indices(indices: np.ndarray) -> np.ndarray:
        nonlocal invalid, latency_ms
        features_list = []
        for idx in indices:
            cached_features = FEATURE_CACHE.get(prompt_key, int(idx))
            if cached_features is not None:
                features_list.append(cached_features)
                continue

            row = DATASET.X[int(idx)]
            row_text = _row_to_text(row, DATASET.feature_names)
            try:
                features, latency = await _call_llm(
                    row_text=row_text,
                    system_prompt=system_prompt,
                    inference_url=inference_url,
                    model=model,
                    api_key=api_key,
                    policy_config=policy_config,
                )
                latency_ms += latency
                FEATURE_CACHE.set(prompt_key, int(idx), features, valid=True)
                features_list.append(features)
            except Exception:
                invalid += 1
                fallback = [0.0] * NUM_FEATURES
                FEATURE_CACHE.set(prompt_key, int(idx), fallback, valid=False)
                features_list.append(fallback)

        return np.asarray(features_list, dtype=float)

    X_train = await featurize_indices(DATASET.train_idx)
    X_val = await featurize_indices(DATASET.val_idx)

    model_obj = XGBClassifier(**XGB_PARAMS)
    model_obj.fit(X_train, DATASET.y[DATASET.train_idx])

    proba = model_obj.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(DATASET.y[DATASET.val_idx], proba))
    acc = float(accuracy_score(DATASET.y[DATASET.val_idx], preds))

    invalid_total = invalid + FEATURE_CACHE.invalid_count(prompt_key)
    invalid_rate = invalid_total / max(1, len(DATASET.train_idx) + len(DATASET.val_idx))

    reward = auc - 0.05 * invalid_rate

    result = {
        "auc": auc,
        "accuracy": acc,
        "invalid_rate": invalid_rate,
        "reward": reward,
        "latency_ms": latency_ms,
    }
    SCORE_CACHE[prompt_key] = result
    return result


def create_boosting_local_api(system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url")
        if not inference_url:
            raise ValueError("No inference_url provided in policy config")

        resolved_prompt = _extract_system_prompt(policy_config, system_prompt)
        result = await score_prompt(
            system_prompt=resolved_prompt,
            inference_url=inference_url,
            model=policy_config.get("model", "gpt-4.1-nano"),
            api_key=policy_config.get("api_key"),
            policy_config=policy_config,
        )

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=float(result["reward"]),
                outcome_objectives={
                    "auc": result["auc"],
                    "accuracy": result["accuracy"],
                    "invalid_rate": result["invalid_rate"],
                },
                instance_objectives=[
                    {
                        "auc": result["auc"],
                        "accuracy": result["accuracy"],
                        "invalid_rate": result["invalid_rate"],
                    }
                ],
                details={"latency_ms": result["latency_ms"]},
            ),
            trace=None,
            trace_correlation_id=request.trace_correlation_id,
        )

    def provide_taskset_description():
        return {
            "splits": ["train", "val"],
            "sizes": {"train": len(DATASET.train_idx), "val": len(DATASET.val_idx)},
            "num_features": NUM_FEATURES,
        }

    def provide_task_instances(seeds: Iterable[int]):
        for seed in seeds:
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": int(seed)},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={"seed": int(seed)},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description="GEPA optimizes LLM feature prompts with downstream XGBoost scoring.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


__all__ = ["create_boosting_local_api", "score_prompt", "DEFAULT_SYSTEM_PROMPT"]
