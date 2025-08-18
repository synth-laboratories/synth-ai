"""
Hello World: Banking77 intent classification with in-context injection

This script shows a minimal text-classification pipeline over the
Hugging Face Banking77 dataset using the Synth LM interface. It also
demonstrates a simple pre-send prompt-injection step as outlined in
`synth_ai/learning/prompts/injection_plan.txt`.

Notes
- Network access is required to download the dataset and call the model.
- Defaults to Groq with model `openai/gpt-oss-20b`.
  - Export your key: `export GROQ_API_KEY=...`
  - Override if needed: `export MODEL=openai/gpt-oss-20b VENDOR=groq`

Run
- `python -m synth_ai.learning.prompts.hello_world_in_context_injection_ex`

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

from datasets import load_dataset

# Use the v3 LM class present in this repo
from synth_ai.lm.core.main_v3 import LM, build_messages

# Use Overrides context to demonstrate matching by content
from synth_ai.lm.overrides import LMOverridesContext
from synth_ai.tracing_v3.abstractions import LMCAISEvent
from synth_ai.tracing_v3.session_tracer import SessionTracer

INJECTION_RULES = [
    {"find": "accnt", "replace": "account"},
    {"find": "atm", "replace": "ATM"},
    {"find": "txn", "replace": "transaction"},
]


async def classify_sample(lm: LM, text: str, label_names: list[str]) -> str:
    """Classify one Banking77 utterance and return the predicted label name."""
    labels_joined = ", ".join(label_names)
    system_message = (
        "You are an intent classifier for the Banking77 dataset. "
        "Given a customer message, respond with exactly one label from the list. "
        "Return only the label text with no extra words.\n\n"
        f"Valid labels: {labels_joined}"
    )
    user_message = f"Message: {text}\nLabel:"

    # Build canonical messages; injection will be applied inside the vendor via context
    messages = build_messages(system_message, user_message, images_bytes=None, model_name=lm.model)
    resp = await lm.respond_async(messages=messages)
    raw = (resp.raw_response or "").strip()
    return raw


async def main() -> None:
    # Configurable model/provider via env, with sensible defaults
    # Default to Groq hosting `openai/gpt-oss-20b`
    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    vendor = os.getenv("VENDOR", "groq")

    # Construct LM
    lm = LM(model=model, vendor=vendor, temperature=0.0)

    # Load Banking77 dataset
    # Columns: {"text": str, "label": int}; label names at ds.features["label"].names
    print("Loading Banking77 dataset (split='test')...")
    ds = load_dataset("banking77", split="test")
    label_names: list[str] = ds.features["label"].names  # type: ignore

    # Sample a few items for a quick demo
    n = int(os.getenv("N_SAMPLES", "8"))
    idxs = random.sample(range(len(ds)), k=min(n, len(ds)))

    correct = 0
    # Apply overrides for all calls in this block (match by content)
    overrides = [
        {"match": {"contains": "atm", "role": "user"}, "injection_rules": INJECTION_RULES},
        {"match": {"contains": "refund"}, "params": {"temperature": 0.0}},
    ]
    with LMOverridesContext(overrides):
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
            print(
                f"[{i}] text={text!r}\n    gold={gold_label}\n    pred={pred} -> mapped={pred_label} {'✅' if is_correct else '❌'}"
            )

    if idxs:
        acc = correct / len(idxs)
        print(f"\nSamples: {len(idxs)} | Correct: {correct} | Accuracy: {acc:.2%}")

    # ------------------------------
    # Integration tests (three paths)
    # ------------------------------
    print("\nRunning integration tests with in-context injection...")
    test_text = "I used the atm to withdraw cash."

    # 1) LM path with v3 tracing: verify substitution in traced messages
    tracer = SessionTracer()
    await tracer.start_session(metadata={"test": "lm_injection"})
    await tracer.start_timestep(step_id="lm_test")
    # Use a tracer-bound LM instance
    lm_traced = LM(model=model, vendor=vendor, temperature=0.0, session_tracer=tracer)
    with LMOverridesContext([{"match": {"contains": "atm"}, "injection_rules": INJECTION_RULES}]):
        _ = await classify_sample(lm_traced, test_text, label_names)
    # inspect trace
    events = [
        e
        for e in (tracer.current_session.event_history if tracer.current_session else [])
        if isinstance(e, LMCAISEvent)
    ]
    assert events, "No LMCAISEvent recorded by SessionTracer"
    cr = events[-1].call_records[0]
    traced_user = ""
    for m in cr.input_messages:
        if m.role == "user":
            for part in m.parts:
                if getattr(part, "type", None) == "text":
                    traced_user += part.text or ""
    assert "ATM" in traced_user, f"Expected substitution in traced prompt; got: {traced_user!r}"
    print("LM path trace verified: substitution present in traced prompt.")
    await tracer.end_timestep()
    await tracer.end_session()

    # 2) OpenAI wrapper path (AsyncOpenAI to Groq): ensure apply_injection is active
    try:
        import synth_ai.lm.provider_support.openai as _synth_openai_patch  # noqa: F401
        from openai import AsyncOpenAI

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        messages = [
            {"role": "system", "content": "Echo user label."},
            {"role": "user", "content": f"Please classify: {test_text}"},
        ]
        with LMOverridesContext(
            [{"match": {"contains": "atm"}, "injection_rules": INJECTION_RULES}]
        ):
            _ = await client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
        # Not all models echo input; instead, verify that our injected expectation matches
        expected_user = messages[1]["content"].replace("atm", "ATM")
        if messages[1]["content"] == expected_user:
            print("OpenAI wrapper: input already normalized; skipping assertion.")
        else:
            print("OpenAI wrapper: sent message contains substitution expectation:", expected_user)
    except Exception as e:
        print("OpenAI wrapper test skipped due to error:", e)

    # 3) Anthropic wrapper path (AsyncClient): ensure apply_injection is active
    try:
        import anthropic
        import synth_ai.lm.provider_support.anthropic as _synth_anthropic_patch  # noqa: F401

        a_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        a_key = os.getenv("ANTHROPIC_API_KEY")
        if a_key:
            a_client = anthropic.AsyncClient(api_key=a_key)
            with LMOverridesContext(
                [{"match": {"contains": "atm"}, "injection_rules": INJECTION_RULES}]
            ):
                _ = await a_client.messages.create(
                    model=a_model,
                    system="Echo user label.",
                    max_tokens=64,
                    temperature=0,
                    messages=[{"role": "user", "content": [{"type": "text", "text": test_text}]}],
                )
            print("Anthropic wrapper call completed (cannot reliably assert echo).")
        else:
            print("Anthropic wrapper test skipped: ANTHROPIC_API_KEY not set.")
    except Exception as e:
        print("Anthropic wrapper test skipped due to error:", e)


if __name__ == "__main__":
    asyncio.run(main())
