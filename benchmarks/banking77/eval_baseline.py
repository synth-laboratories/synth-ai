#!/usr/bin/env python3
"""Evaluate baseline prompts on Banking77 validation seeds - simplified."""

import argparse
import asyncio
import json
import os

import httpx
from datasets import load_dataset

# Validation seeds (matching the GEPA benchmark)
VALIDATION_SEEDS = list(range(100, 120))  # Use 20 seeds for quick baseline

BASELINE_SYSTEM = """You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."""

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


async def evaluate_baseline(model: str, api_key: str, seeds: list) -> dict:
    """Evaluate baseline prompt on specified seeds."""

    # Load dataset
    print("Loading Banking77 dataset...")
    dataset = load_dataset("PolyAI/banking77", split="train")

    intents_str = "\n".join([f"{i + 1}. {label}" for i, label in enumerate(BANKING77_LABELS)])

    correct = 0
    total = 0

    async with httpx.AsyncClient(timeout=60) as client:
        for seed in seeds:
            if seed >= len(dataset):
                print(f"Seed {seed} out of range")
                continue

            example = dataset[seed]
            query = example["text"]
            expected_label = BANKING77_LABELS[example["label"]]

            # Build messages
            user_content = f"""Customer Query: {query}

Available Intents:
{intents_str}

Classify this query into one of the above banking intents using the tool call."""

            messages = [
                {"role": "system", "content": BASELINE_SYSTEM},
                {"role": "user", "content": user_content},
            ]

            # Define the tool
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "banking77_classify",
                        "description": "Classify the banking intent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "intent": {"type": "string", "description": "The classified intent"}
                            },
                            "required": ["intent"],
                        },
                    },
                }
            ]

            # Call OpenAI
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": {
                            "type": "function",
                            "function": {"name": "banking77_classify"},
                        },
                        "temperature": 1.0,  # gpt-5-nano requires temp=1
                        "max_completion_tokens": 2048,  # reasoning models need more tokens
                    },
                )
                result = resp.json()

                # Extract prediction
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "tool_calls" in choice["message"]:
                        tool_call = choice["message"]["tool_calls"][0]
                        args = json.loads(tool_call["function"]["arguments"])
                        predicted = args.get("intent", "").strip().lower()
                        expected_lower = expected_label.strip().lower()

                        is_correct = predicted == expected_lower
                        if is_correct:
                            correct += 1
                        total += 1
                        print(
                            f"[{seed:3d}] {'✓' if is_correct else '✗'} pred={predicted[:30]:30s} exp={expected_lower[:30]}"
                        )
                    else:
                        print(f"[{seed:3d}] ✗ No tool call in response")
                        total += 1
                elif "error" in result:
                    print(
                        f"[{seed:3d}] ✗ API error: {result['error'].get('message', result['error'])}"
                    )
                    total += 1
                else:
                    print(f"[{seed:3d}] ✗ Unexpected response")
                    total += 1

            except Exception as e:
                print(f"[{seed:3d}] ✗ Error: {e}")
                total += 1

    accuracy = correct / total if total > 0 else 0
    return {"model": model, "correct": correct, "total": total, "accuracy": accuracy}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    parser.add_argument(
        "--seeds", default="100-119", help="Seed range (e.g., 100-119 or 100,105,110)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY required")
        return

    # Parse seeds
    if "-" in args.seeds:
        start, end = args.seeds.split("-")
        seeds = list(range(int(start), int(end) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print(f"\nEvaluating baseline for {args.model} on {len(seeds)} seeds...")
    result = await evaluate_baseline(args.model, api_key, seeds)

    print(f"\n{'=' * 50}")
    print(f"BASELINE RESULT: {result['model']}")
    print(f"Accuracy: {result['accuracy'] * 100:.1f}% ({result['correct']}/{result['total']})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
