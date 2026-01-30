"""
LocalAPI Task App - Banking77 Intent Classification

This file creates a task app that Synth AI uses to evaluate prompts.
The backend calls your /rollout endpoint with different seeds (test cases)
and aggregates the scores.
"""

import json
import os
import time
from typing import Any

import httpx
from datasets import load_dataset
from fastapi import Request
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.localapi._impl.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.localapi._impl.validators import normalize_inference_url

# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_ID = "banking77"
APP_NAME = "Banking77 Intent Classification"

_LOG_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_LOG_LEVEL = os.getenv("BANKING77_LOG_LEVEL", "info").lower()


def _log(level: str, message: str, **fields: Any) -> None:
    if _LOG_LEVELS.get(level, 20) < _LOG_LEVELS.get(_LOG_LEVEL, 20):
        return
    payload = {"level": level, "msg": message, **fields}
    print(f"[BANKING77] {json.dumps(payload, ensure_ascii=True)}", flush=True)


class RolloutError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        flags: list[str] | None = None,
        details: dict[str, Any] | None = None,
        status: SuccessStatus = SuccessStatus.RUNTIME_ERROR,
    ) -> None:
        super().__init__(message)
        self.flags = flags or []
        self.details = details or {}
        self.status = status


def _require(condition: bool, *, flag: str, message: str, **details: Any) -> None:
    if not condition:
        raise RolloutError(message, flags=[flag], details=details)


# =============================================================================
# DATASET: Banking77 Intent Classification
# =============================================================================

BANKING77_LABELS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

TOOL_NAME = "banking77_classify"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the predicted banking77 intent label.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": BANKING77_LABELS}},
            "required": ["intent"],
        },
    },
}

SYSTEM_PROMPT = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."


class Banking77Dataset:
    def __init__(self):
        self._cache = {}
        self._label_names = None
        self._last_error_by_split: dict[str, str] = {}

    def _load_split(self, split: str):
        if split not in self._cache:
            try:
                _log("info", "dataset_load_start", split=split)
                ds = load_dataset("banking77", split=split, trust_remote_code=False)
            except Exception as exc:
                self._last_error_by_split[split] = str(exc)
                _log("warn", "dataset_load_failed", split=split, error=str(exc))
                try:
                    ds = load_dataset(
                        "banking77",
                        split=split,
                        trust_remote_code=False,
                        download_mode="force_redownload",
                    )
                except Exception as exc2:
                    self._last_error_by_split[split] = str(exc2)
                    _log("error", "dataset_redownload_failed", split=split, error=str(exc2))
                    raise
            _log("info", "dataset_load_ok", split=split, rows=len(ds))
            self._cache[split] = ds
            if self._label_names is None and hasattr(ds.features.get("label"), "names"):
                self._label_names = ds.features["label"].names
        return self._cache[split]

    def ensure_ready(self, splits):
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, *, split: str, index: int) -> dict:
        ds = self._load_split(split)
        idx = index % len(ds)
        row = ds[idx]
        label_idx = int(row.get("label", 0))
        label_text = (
            self._label_names[label_idx]
            if self._label_names and label_idx < len(self._label_names)
            else f"label_{label_idx}"
        )
        return {"index": idx, "split": split, "text": str(row.get("text", "")), "label": label_text}

    @property
    def label_names(self) -> list:
        if self._label_names is None:
            self._load_split("train")
        return self._label_names or []


DATASET = Banking77Dataset()
DATASET_PREFETCH_ERROR: str | None = None
try:
    DATASET.ensure_ready(["train", "test"])
except Exception as exc:
    DATASET_PREFETCH_ERROR = str(exc)
    _log("error", "dataset_prefetch_failed", error=str(exc))


def format_available_intents(label_names: list) -> str:
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


def get_dataset_size(split: str = "train") -> int:
    """Return the total number of samples in your dataset."""
    return DATASET.size(split)


def get_sample(seed: int, split: str = "train") -> dict:
    """
    Get a test case by seed index.

    Args:
        seed: The seed/index for this evaluation (from request.env.seed)
        split: Dataset split to use ("train" or "test")

    Returns:
        Dict with test case fields: {"text": ..., "label": ...}
    """
    return DATASET.sample(split=split, index=seed)


# =============================================================================
# SCORING LOGIC
# =============================================================================


def score_response(predicted_intent: str, sample: dict) -> float:
    """
    Score the model response. Returns 0.0 to 1.0.

    Args:
        predicted_intent: The model's predicted intent
        sample: The test case dict from get_sample()

    Returns:
        Score between 0.0 (wrong) and 1.0 (correct)
    """
    expected_intent = sample["label"]
    is_correct = (
        predicted_intent.lower().replace("_", " ").strip()
        == expected_intent.lower().replace("_", " ").strip()
    )
    return 1.0 if is_correct else 0.0


# =============================================================================
# TASK APP PROVIDERS (required by Synth backend)
# =============================================================================


def provide_taskset_description() -> dict:
    """Return metadata about your task set (splits, sizes, etc.)."""
    return {
        "splits": ["train", "test"],
        "sizes": {"train": DATASET.size("train"), "test": DATASET.size("test")},
    }


def provide_task_instances(seeds: list[int]):
    """Yield TaskInfo for each seed. Called by Synth to get task metadata."""
    for seed in seeds:
        sample = get_sample(seed, split="train")
        yield TaskInfo(
            task={"id": APP_ID, "name": APP_NAME},
            dataset={"id": APP_ID, "split": sample["split"], "index": sample["index"]},
            inference={"tool": TOOL_NAME},
            limits={"max_turns": 1},
            task_metadata={"query": sample["text"], "expected_intent": sample["label"]},
        )


# =============================================================================
# LLM CALL HELPER
# =============================================================================


async def call_llm(
    query: str,
    inference_url: str,
    model: str = "gpt-4.1-nano",
    api_key: str | None = None,
) -> tuple[str, str | None, dict, list[dict[str, str]]]:
    """Call the LLM via the inference URL provided by Synth, using tool calling."""
    available_intents = format_available_intents(DATASET.label_names)
    if os.getenv("BANKING77_DEBUG_INTENTS") == "1":
        preview = "\n".join(DATASET.label_names[:10])
        print(f"[DEBUG] intents_count={len(DATASET.label_names)} intents_preview:\n{preview}")
    user_msg = (
        f"Customer Query: {query}\n\n"
        f"Available Intents:\n{available_intents}\n\n"
        f"Classify this query into one of the above banking intents using the tool call."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "tools": [TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
    }

    try:
        url = normalize_inference_url(inference_url)
    except Exception as exc:
        raise RolloutError(
            "Invalid inference_url",
            flags=["invalid_inference_url"],
            details={"error": str(exc)},
        ) from exc
    _require(bool(url), flag="invalid_inference_url", message="Normalized inference_url is empty")
    _log("info", "llm_request_start", model=model, inference_url=url)
    if os.getenv("BANKING77_DEBUG_INFERENCE_URL") == "1":
        print(f"[DEBUG] inference_url={url}", flush=True)

    timeout_seconds = float(os.getenv("BANKING77_LLM_TIMEOUT", "120"))
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise RolloutError(
                "LLM request timed out",
                flags=["llm_timeout"],
                details={"timeout_s": timeout_seconds},
                status=SuccessStatus.TIMEOUT,
            ) from exc
        except httpx.RequestError as exc:
            raise RolloutError(
                "LLM request failed",
                flags=["llm_request_error"],
                details={"error": str(exc)},
                status=SuccessStatus.NETWORK_ERROR,
            ) from exc
        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = str(error_json.get("error", {}).get("message", error_json))
            except Exception:
                error_msg = response.text[:500]
            raise RolloutError(
                f"Proxy error ({response.status_code}): {error_msg}",
                flags=["llm_proxy_error"],
                details={"status_code": response.status_code, "error": error_msg},
                status=SuccessStatus.RUNTIME_ERROR,
            )

        # Extract candidate_id from proxy response headers for MIPRO
        candidate_id = response.headers.get("x-mipro-candidate-id")
        
        # Log all MIPRO-related headers for debugging
        mipro_headers = {k: v for k, v in response.headers.items() if "mipro" in k.lower()}
        if mipro_headers:
            print(f"[MIPRO] Received headers: {mipro_headers}", flush=True)
        else:
            print(f"[MIPRO] WARNING: No x-mipro-* headers in response. Check proxy is returning them.", flush=True)
        
        if candidate_id:
            print(f"[MIPRO] candidate_id from header: {candidate_id}", flush=True)
        
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RolloutError(
                "No choices returned from model",
                flags=["llm_no_choices"],
                details={"response_keys": list(data.keys())},
            )
        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            raise RolloutError(
                "No tool calls returned from model",
                flags=["llm_no_tool_calls"],
                details={"choice_keys": list(choices[0].keys())},
            )
        tool_call = tool_calls[0]
        args_raw = tool_call.get("function", {}).get("arguments")

    if not args_raw:
        raise RolloutError(
            "No tool call arguments returned from model",
            flags=["llm_no_tool_args"],
        )

    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except json.JSONDecodeError as exc:
        raise RolloutError(
            "Tool call arguments are not valid JSON",
            flags=["llm_bad_tool_args"],
            details={"arguments_preview": str(args_raw)[:200]},
        ) from exc
    intent = args.get("intent") or ""
    _require(bool(intent), flag="empty_intent", message="Tool call returned empty intent")
    
    # Return intent, candidate_id, and response payload for trace usage
    return intent, candidate_id, data, messages


# =============================================================================
# ROLLOUT HANDLER
# =============================================================================


async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """
    Handle a single evaluation rollout.

    Args:
        request: Contains seed, policy config, env config
        fastapi_request: The FastAPI request object

    Returns:
        RolloutResponse with the evaluation score
    """
    trace_correlation_id = request.trace_correlation_id or ""
    try:
        _require(request.env is not None, flag="missing_env", message="request.env is required")
        _require(request.policy is not None, flag="missing_policy", message="request.policy is required")

        split = request.env.config.get("split", "train") if request.env.config else "train"
        _require(
            split in {"train", "test"},
            flag="invalid_split",
            message="Unsupported dataset split",
            split=split,
        )

        seed = request.env.seed
        _require(seed is not None, flag="missing_seed", message="request.env.seed is required")
        _require(isinstance(seed, int), flag="invalid_seed", message="seed must be an int", seed=seed)
        assert isinstance(seed, int), "seed must be an int"

        _log("info", "rollout_start", trace_correlation_id=trace_correlation_id, split=split, seed=seed)

        try:
            DATASET.ensure_ready([split])
        except Exception as exc:
            raise RolloutError(
                "Dataset load failed",
                flags=["dataset_load_failed"],
                details={
                    "split": split,
                    "error": str(exc),
                    "last_error": DATASET._last_error_by_split.get(split),
                    "prefetch_error": DATASET_PREFETCH_ERROR,
                    "hf_cache": os.getenv("HF_DATASETS_CACHE"),
                },
            ) from exc

        sample = get_sample(seed, split=split)
        _require(bool(sample.get("text")), flag="missing_sample_text", message="Sample text missing")
        _require(bool(sample.get("label")), flag="missing_sample_label", message="Sample label missing")

        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url")
        _require(
            bool(inference_url),
            flag="missing_inference_url",
            message="policy.config.inference_url is required",
        )

        llm_start = time.perf_counter()
        predicted_intent, candidate_id, llm_response, llm_messages = await call_llm(
            query=sample["text"],
            inference_url=inference_url,
            model=policy_config.get("model", "gpt-4.1-nano"),
            api_key=policy_config.get("api_key"),
        )
        llm_latency_ms = (time.perf_counter() - llm_start) * 1000.0
        _log("info", "llm_call_ok", latency_ms=round(llm_latency_ms, 2))
        latency_ms = llm_latency_ms

        score = score_response(predicted_intent, sample)
        if os.getenv("BANKING77_DEBUG_ROLLOUT") == "1":
            print(
                "[DEBUG] rollout_compare",
                {
                    "seed": seed,
                    "text": sample.get("text", "")[:200],
                    "expected": sample.get("label"),
                    "predicted": predicted_intent,
                    "score": score,
                },
            )

        policy_cfg_for_trace = {
            key: value
            for key, value in policy_config.items()
            if key not in {"trace_correlation_id", "trace"}
        }
        trace_correlation_id = extract_trace_correlation_id(
            policy_config=policy_cfg_for_trace,
            inference_url=str(inference_url or ""),
        )
        if not trace_correlation_id:
            trace_correlation_id = request.trace_correlation_id or ""

        reward_info = RolloutMetrics(
            outcome_reward=score,
            outcome_objectives={"reward": score, "latency_ms": latency_ms},
            instance_objectives=[{"reward": score, "latency_ms": latency_ms}],
            details={"latency_ms": latency_ms, "error_flags": []},
        )

        # Include candidate_id in metadata if available (for MIPRO)
        metadata = {}
        if candidate_id:
            metadata["mipro_candidate_id"] = candidate_id

        trace_payload = {
            "inference": {
                "messages": llm_messages,
                "response": llm_response,
            }
        }

        return RolloutResponse(
            reward_info=reward_info,
            trace=trace_payload,
            metadata=metadata,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
            success_status=SuccessStatus.SUCCESS,
        )
    except RolloutError as exc:
        _log("error", "rollout_failed", error=str(exc), flags=exc.flags, details=exc.details)
        reward_info = RolloutMetrics(
            outcome_reward=0.0,
            details={"error_flags": exc.flags, **exc.details},
        )
        return RolloutResponse(
            reward_info=reward_info,
            trace_correlation_id=trace_correlation_id,
            success_status=exc.status,
            status_detail=str(exc),
        )
    except Exception as exc:
        _log(
            "error",
            "rollout_failed_unexpected",
            error=str(exc),
            exception_type=type(exc).__name__,
        )
        reward_info = RolloutMetrics(
            outcome_reward=0.0,
            details={"error_flags": ["unexpected_error"], "exception_type": type(exc).__name__},
        )
        return RolloutResponse(
            reward_info=reward_info,
            trace_correlation_id=trace_correlation_id,
            success_status=SuccessStatus.RUNTIME_ERROR,
            status_detail=str(exc),
        )


# =============================================================================
# CREATE THE APP
# =============================================================================

app = create_local_api(
    LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description=f"{APP_NAME} local API for classifying customer queries into banking intents.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    )
)


# =============================================================================
# RUNNING LOCALLY
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth

    # Ensure ENVIRONMENT_API_KEY is set and uploaded to the backend.
    # This lets the backend know what key to send when calling /rollout.
    env_key = ensure_localapi_auth(
        backend_base="http://localhost:8000",  # local backend
        synth_api_key=None,  # will use SYNTH_API_KEY from env if available
    )
    print(f"[localapi_banking77] ENVIRONMENT_API_KEY ready: {env_key[:15]}...")

    uvicorn.run(app, host="0.0.0.0", port=8010)
