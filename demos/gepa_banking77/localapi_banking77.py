"""
LocalAPI Task App - Banking77 Intent Classification

This file creates a task app that Synth AI uses to evaluate prompts.
The backend calls your /rollout endpoint with different seeds (test cases)
and aggregates the scores.
"""

import json

import httpx
from datasets import load_dataset
from fastapi import Request
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task import normalize_inference_url
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id

# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_ID = "banking77"
APP_NAME = "Banking77 Intent Classification"


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
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}

SYSTEM_PROMPT = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."


class Banking77Dataset:
    def __init__(self):
        self._cache = {}
        self._label_names = None

    def _load_split(self, split: str):
        if split not in self._cache:
            ds = load_dataset("banking77", split=split, trust_remote_code=False)
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
DATASET.ensure_ready(["train", "test"])


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
) -> str:
    """Call the LLM via the inference URL provided by Synth, using tool calling."""
    available_intents = format_available_intents(DATASET.label_names)
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

    payload = {
        "model": model,
        "messages": messages,
        "tools": [TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
    }

    url = normalize_inference_url(inference_url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = str(error_json.get("error", {}).get("message", error_json))
            except Exception:
                error_msg = response.text[:500]
            raise RuntimeError(f"Proxy error ({response.status_code}): {error_msg}")

        data = response.json()
        tool_call = (data.get("choices") or [])[0].get("message", {}).get("tool_calls", [])[0]
        args_raw = tool_call.get("function", {}).get("arguments")

    if not args_raw:
        raise RuntimeError("No tool call arguments returned from model")

    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    return args["intent"]


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
    split = request.env.config.get("split", "train")
    seed = request.env.seed
    sample = get_sample(seed, split=split)

    policy_config = request.policy.config or {}
    inference_url = policy_config.get("inference_url")

    if not inference_url:
        raise ValueError("No inference_url provided in policy config")

    predicted_intent = await call_llm(
        query=sample["text"],
        inference_url=inference_url,
        model=policy_config.get("model", "gpt-4.1-nano"),
        api_key=policy_config.get("api_key"),
    )

    score = score_response(predicted_intent, sample)

    policy_cfg_for_trace = {
        key: value
        for key, value in policy_config.items()
        if key not in {"trace_correlation_id", "trace"}
    }
    trace_correlation_id = extract_trace_correlation_id(
        policy_config=policy_cfg_for_trace,
        inference_url=str(inference_url or ""),
        mode=request.mode,
    )

    return RolloutResponse(
        run_id=request.run_id,
        metrics=RolloutMetrics(outcome_reward=score),
        trace=None,
        trace_correlation_id=trace_correlation_id,
        inference_url=str(inference_url or ""),
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

    uvicorn.run(app, host="0.0.0.0", port=8001)
