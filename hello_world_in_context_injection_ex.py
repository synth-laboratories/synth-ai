"""
Hello World: Banking77 intent classification with in-context injection

This script shows a minimal text-classification pipeline over the
Hugging Face Banking77 dataset using the Synth LM interface. It also
demonstrates a simple pre-send prompt-injection step as outlined in
`synth_ai/learning/prompts/injection_plan.txt`.

Notes
- Network access is required to download the dataset and call the model.
- For `openai/gpt-oss-20b`, run an OpenAI-compatible endpoint (e.g., vLLM):
  - Start server: `vllm serve openai/gpt-oss-20b`
  - Export: `export OPENAI_BASE_URL=http://localhost:8000/v1`
            `export OPENAI_API_KEY=EMPTY`  # vLLM accepts any string
- To use Groq, set: `export GROQ_API_KEY=...` and choose a Groq-supported
  model (e.g., `llama-3.1-8b-instant`). The default below targets `openai/gpt-oss-20b`.

Run
- `python hello_world_in_context_injection_ex.py`

What "in-context injection" means here
- The script applies ordered substring replacements to the outgoing
  `messages` array before calling the model. This mirrors the algorithm
  described in `injection_plan.txt` without importing any non-existent
  helper yet. You can adapt `INJECTION_RULES` to your needs.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# Use the v3 LM class present in this repo
from synth_ai.lm.core.main_v3 import LM, build_messages


# -------------------------------
# Minimal injection implementation
# -------------------------------

Rule = Dict[str, Any]


def apply_injection(messages: List[Dict[str, Any]], rules: Optional[List[Rule]]) -> List[Dict[str, Any]]:
    """Apply ordered substring replacements to text parts of messages.

    Mirrors the algorithm described in `injection_plan.txt`:
    - Only applies to `str` content or list parts where part["type"] == "text".
    - Respects optional `roles` scoping in each rule.
    - Mutates the provided messages in place and returns it for convenience.
    """
    if not rules:
        return messages

    for m in messages:
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            new_content = content
            for r in rules:
                allowed_roles = r.get("roles")
                if allowed_roles is not None and role not in allowed_roles:
                    continue
                new_content = new_content.replace(r["find"], r["replace"])
            m["content"] = new_content

        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    new_text = text
                    for r in rules:
                        allowed_roles = r.get("roles")
                        if allowed_roles is not None and role not in allowed_roles:
                            continue
                        new_text = new_text.replace(r["find"], r["replace"])
                    part["text"] = new_text

    return messages


# Example rules you can tweak. These clean up common shorthand in user text.
INJECTION_RULES: List[Rule] = [
    {"find": "accnt", "replace": "account"},
    {"find": "atm", "replace": "ATM"},
    {"find": "txn", "replace": "transaction"},
]


async def classify_sample(lm: LM, text: str, label_names: List[str]) -> str:
    """Classify one Banking77 utterance and return the predicted label name."""
    labels_joined = ", ".join(label_names)
    system_message = (
        "You are an intent classifier for the Banking77 dataset. "
        "Given a customer message, respond with exactly one label from the list. "
        "Return only the label text with no extra words.\n\n"
        f"Valid labels: {labels_joined}"
    )
    user_message = f"Message: {text}\nLabel:"

    # Build canonical messages and apply simple injection
    messages = build_messages(system_message, user_message, images_bytes=None, model_name=lm.model)
    messages = apply_injection(messages, INJECTION_RULES)

    # Call LM directly with the already-built messages
    resp = await lm.respond_async(messages=messages)
    raw = (resp.raw_response or "").strip()
    return raw


async def main() -> None:
    # Configurable model/provider via env, with sensible defaults
    # Default to an OpenAI-compatible endpoint serving `openai/gpt-oss-20b` (e.g., vLLM)
    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    vendor = os.getenv("VENDOR", "openai")  # Force OpenAI-compatible client for gpt-oss-20b

    # If you want Groq instead, set: VENDOR=groq and MODEL to a Groq-supported model
    # e.g., export VENDOR=groq MODEL=llama-3.1-8b-instant

    # Construct LM
    lm = LM(model=model, vendor=vendor, temperature=0.0)

    # Load Banking77 dataset
    # Columns: {"text": str, "label": int}; label names at ds.features["label"].names
    print("Loading Banking77 dataset (split='test')...")
    ds = load_dataset("banking77", split="test")
    label_names: List[str] = ds.features["label"].names  # type: ignore

    # Sample a few items for a quick demo
    n = int(os.getenv("N_SAMPLES", "8"))
    idxs = random.sample(range(len(ds)), k=min(n, len(ds)))

    correct = 0
    for i, idx in enumerate(idxs, start=1):
        text: str = ds[idx]["text"]  # type: ignore
        gold_label_idx: int = int(ds[idx]["label"])  # type: ignore
        gold_label = label_names[gold_label_idx]

        try:
            pred = await classify_sample(lm, text, label_names)
        except Exception as e:
            print(f"[{i}] Error calling model: {e}")
            break

        # Normalize and check exact match; if not exact, attempt a loose fallback
        norm_pred = pred.strip().lower()
        label_lookup = {ln.lower(): ln for ln in label_names}
        pred_label = label_lookup.get(norm_pred)
        if pred_label is None:
            # Fallback: pick the label with highest substring overlap (very naive)
            # This avoids extra deps; feel free to replace with a better matcher.
            def score(cand: str) -> int:
                c = cand.lower()
                return sum(1 for w in c.split() if w in norm_pred)

            pred_label = max(label_names, key=score)

        is_correct = pred_label == gold_label
        correct += int(is_correct)
        print(f"[{i}] text={text!r}\n    gold={gold_label}\n    pred={pred} -> mapped={pred_label} {'✅' if is_correct else '❌'}")

    if idxs:
        acc = correct / len(idxs)
        print(f"\nSamples: {len(idxs)} | Correct: {correct} | Accuracy: {acc:.2%}")


if __name__ == "__main__":
    asyncio.run(main())

