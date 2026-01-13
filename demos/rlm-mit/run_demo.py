#!/usr/bin/env python3
"""Run the MIT RLM OOLONG GEPA demo end-to-end.

Usage:
    uv run python demos/rlm-mit/run_demo.py
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import httpx
from datasets import load_dataset
from rlm import RLM
from rlm.core import rlm as rlm_core
from rlm.core import types as rlm_types
from rlm.utils import prompts as rlm_prompts
from rlm.utils.prompts import USER_PROMPT
from synth_ai.core.urls import BACKEND_URL_BASE, backend_health_url
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.auth import get_or_mint_synth_api_key
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.tunnels import TunnelBackend, TunneledLocalAPI, kill_port


# Work around rlm QueryMetadata typing bug under Python 3.11
class PatchedQueryMetadata:
    def __init__(self, prompt):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_lengths = [len(chunk) for chunk in prompt.values()]
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if prompt and isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(chunk["content"]) for chunk in prompt]
                else:
                    self.context_lengths = [len(chunk) for chunk in prompt]
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)


rlm_types.QueryMetadata = PatchedQueryMetadata
rlm_core.QueryMetadata = PatchedQueryMetadata


def patched_build_rlm_system_prompt(system_prompt, query_metadata=None, **_kwargs):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "{context_metadata}"},
    ]


rlm_prompts.build_rlm_system_prompt = patched_build_rlm_system_prompt
rlm_core.build_rlm_system_prompt = patched_build_rlm_system_prompt

# Backend configuration
SYNTH_API_BASE = BACKEND_URL_BASE
LOCAL_MODE = SYNTH_API_BASE.startswith("http://localhost") or SYNTH_API_BASE.startswith(
    "http://127.0.0.1"
)
LOCAL_HOST = "127.0.0.1"
TUNNEL_BACKEND = TunnelBackend.Localhost if LOCAL_MODE else TunnelBackend.CloudflareManagedTunnel
LOCAL_API_PORT = 8115

print(f"Backend: {SYNTH_API_BASE}")
print(f"Tunnel backend: {TUNNEL_BACKEND.value}")
print(f"Local API Port: {LOCAL_API_PORT}")

# Check backend health
r = httpx.get(backend_health_url(SYNTH_API_BASE), timeout=30)
if r.status_code == 200:
    print(f"Backend health: {r.json()}")
else:
    print(f"WARNING: Backend returned status {r.status_code}")
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")

# Get API Key
API_KEY = get_or_mint_synth_api_key(backend_url=SYNTH_API_BASE)
print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)

print("Config loaded")

RLM_BASE_SYSTEM_PROMPT = (
    "You are a recursive language model. Use the REPL with the context variable to reason. "
    "Call llm_query or llm_query_batched as needed. When finished, answer with FINAL."
)


# Dataset wrapper
@dataclass
class OolongSample:
    index: int
    split: str
    query: str
    context: str
    answer: str


class OolongDataset:
    def __init__(self, hf_dataset: str = "oolongbench/oolong-real", hf_config: str = "dnd"):
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self._cache = {}

    def _load_split(self, split: str):
        if split not in self._cache:
            ds = load_dataset(self.hf_dataset, self.hf_config, split=split)
            self._cache[split] = ds
        return self._cache[split]

    def ensure_ready(self, splits: Iterable[str]) -> None:
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, split: str, index: int) -> OolongSample:
        ds = self._load_split(split)
        idx = index % len(ds)
        row = ds[idx]
        query = row.get("query") or row.get("question") or ""
        context = row.get("context_window_text") or row.get("context") or row.get("text") or ""
        answer = row.get("answer") or ""
        return OolongSample(
            index=idx,
            split=split,
            query=str(query),
            context=str(context),
            answer=str(answer),
        )


# Prompt template helpers
def _normalize_prompt_template(policy_config: Dict[str, Any]) -> Dict[str, Any]:
    template = policy_config.get("prompt_template") or {}
    if not isinstance(template, dict):
        template = {}
    return template


def _get_prompt_sections(policy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    template = _normalize_prompt_template(policy_config)
    sections = (
        template.get("sections")
        or template.get("prompt_sections")
        or policy_config.get("prompt_sections")
        or []
    )
    if not isinstance(sections, list):
        return []
    return sorted(sections, key=lambda s: s.get("order", 0))


def render_prompt_sections(
    sections: List[Dict[str, Any]], placeholders: Dict[str, str]
) -> List[Dict[str, str]]:
    rendered: List[Dict[str, str]] = []
    for section in sections:
        role = section.get("role", "user")
        pattern = section.get("content") or section.get("pattern") or ""
        content = pattern.format(**placeholders)
        rendered.append({"role": role, "content": content})
    return rendered


def split_system_and_user(messages: List[Dict[str, str]]) -> tuple[str, str]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") != "system"]
    system_prompt = "\n\n".join(system_parts).strip()
    user_prompt = "\n\n".join(user_parts).strip()
    return system_prompt, user_prompt


def extract_final_answer(text: str) -> str:
    """Extract the final numeric answer from RLM output.

    RLM outputs can be verbose with reasoning and REPL code blocks.
    We need to:
    1. Remove REPL code blocks first (they contain numbers but aren't answers)
    2. Look for FINAL markers
    3. Look for LaTeX boxed format
    4. Look for final answer patterns near the end
    5. Only as last resort, extract last number
    """
    if not text:
        return ""

    import re

    text = str(text).strip()

    # Step 1: Remove REPL code blocks (```repl...``` or ```python...```)
    # These contain numbers but aren't the final answer
    text = re.sub(r"```(?:repl|python|code)?\s*\n.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Step 2: Try to extract from LaTeX boxed format: \boxed{4} or \(boxed{4}\)
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        # Extract just the number if it's mixed with text
        num_match = re.search(r"\d+", answer)
        if num_match:
            return num_match.group(0)
        return answer

    # Step 3: Look for "FINAL" marker (case-insensitive) and extract what follows
    final_match = re.search(r"FINAL\s*[:\-]?\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if final_match:
        after_final = final_match.group(1).strip()
        # Try boxed format first
        boxed_in_final = re.search(r"\\boxed\{([^}]+)\}", after_final)
        if boxed_in_final:
            answer = boxed_in_final.group(1).strip()
            num_match = re.search(r"\d+", answer)
            if num_match:
                return num_match.group(0)
            return answer
        # Extract first number after FINAL
        num_match = re.search(r"\d+", after_final)
        if num_match:
            return num_match.group(0)

    # Step 4: Look for final answer patterns in the last 500 chars (where final answers usually are)
    # This avoids matching numbers from early reasoning
    last_part = text[-500:] if len(text) > 500 else text

    # Patterns that indicate a final answer
    final_patterns = [
        r"(?:the|final)\s+(?:answer|count|total|number|result)\s+(?:is|:)?\s*(\d+)",
        r"(?:answer|count|total|number|result)\s+(?:is|:)?\s*(\d+)",
        r"(?:is|equals?)\s+(\d+)\s*(?:\.|$|\n)",
        r"(\d+)\s*(?:\.|$|\n)\s*(?:This|Therefore|So|Thus|Hence)",
    ]
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_part, re.IGNORECASE))
        if matches:
            # Take the last match (most likely the final answer)
            return matches[-1].group(1).strip()

    # Step 5: Fallback - extract last number from the cleaned text (no code blocks)
    # But only if the text is reasonably short (not full reasoning)
    if len(text) < 2000:
        numbers = re.findall(r"\d+", text)
        if numbers:
            return numbers[-1]

    return text


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison - extract numeric answer first, then normalize."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # First try to extract the final numeric answer
    extracted = extract_final_answer(text)
    if extracted:
        # Normalize the extracted answer (strip to just alphanumeric)
        normalized = "".join(
            ch.lower() for ch in extracted.strip() if ch.isalnum() or ch.isspace()
        ).strip()
        if normalized:
            return normalized

    # Fallback to original normalization
    return "".join(ch.lower() for ch in text.strip() if ch.isalnum() or ch.isspace()).strip()


# Local API factory
APP_ID = "oolong_rlm"
APP_NAME = "OOLONG RLM (Recursive Language Model) QA"

BASELINE_SYSTEM_PROMPT = "Answer questions using the context."
BASELINE_USER_PROMPT = (
    "Query: {query}\n\nContext:\n{context}\n\nAnswer the query using the context."
)

RLM_CONTEXT_METADATA_PATTERN = "{context_metadata}"
RLM_FIRST_USER_PROMPT = (
    "You have not interacted with the REPL environment or seen your prompt / context yet. "
    "Your next action should be to look through and figure out how to answer the prompt, "
    "so don't just provide a final answer yet.\n\n" + USER_PROMPT
)

COMPOSED_SYSTEM_PROMPT = RLM_BASE_SYSTEM_PROMPT + " " + BASELINE_SYSTEM_PROMPT


def create_oolong_rlm_local_api():
    oolong = OolongDataset()
    oolong.ensure_ready(["validation", "test"])

    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        policy_config = request.policy.config or {}
        env_config = request.env.config or {}
        split = env_config.get("split", "validation")
        seed = request.env.seed or 0

        sample = oolong.sample(split=split, index=seed)
        placeholders = {
            "query": sample.query,
            "context": sample.context,
            "context_metadata": "{context_metadata}",
        }

        sections = _get_prompt_sections(policy_config)
        if not sections:
            sections = [
                {"role": "system", "content": COMPOSED_SYSTEM_PROMPT, "order": 0},
                {"role": "assistant", "content": RLM_CONTEXT_METADATA_PATTERN, "order": 1},
                {"role": "user", "content": RLM_FIRST_USER_PROMPT, "order": 2},
                {"role": "user", "content": BASELINE_USER_PROMPT, "order": 3},
            ]
        rendered = render_prompt_sections(sections, placeholders)
        messages_for_validation = []
        for section in sections:
            role = section.get("role", "user")
            pattern = section.get("content") or section.get("pattern") or ""
            messages_for_validation.append({"role": role, "content": pattern})

        system_prompt, root_prompt = split_system_and_user(rendered)
        custom_system_prompt = system_prompt or RLM_BASE_SYSTEM_PROMPT
        inference_url = (
            policy_config.get("inference_url")
            or policy_config.get("api_base")
            or policy_config.get("base_url")
        )
        if not inference_url:
            raise ValueError("Missing inference_url in policy config")

        api_key = policy_config.get("api_key") or API_KEY
        if not api_key:
            raise ValueError("Missing policy api_key for inference proxy")

        model_name = policy_config.get("model", "gpt-4o-mini")
        max_iterations = int(env_config.get("max_iterations", 2))
        max_depth = int(env_config.get("max_depth", 0))

        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": model_name,
                "api_key": api_key,
                "base_url": inference_url,
            },
            environment="local",
            environment_kwargs={},
            custom_system_prompt=custom_system_prompt,
            max_iterations=max_iterations,
            max_depth=max_depth,
            verbose=False,
        )

        prompt_payload = rendered
        try:
            completion = rlm.completion(
                prompt_payload,
            )

            # Extract answer from RLM completion
            # RLM completion can be a string or an object with various attributes
            if isinstance(completion, str):
                predicted = completion
            else:
                # Try multiple ways to extract the answer
                predicted = (
                    getattr(completion, "response", None)
                    or getattr(completion, "answer", None)
                    or getattr(completion, "final_answer", None)
                    or getattr(completion, "completion", None)
                    or str(completion)
                )

                # If response contains "FINAL", extract everything after it
                if predicted and "FINAL" in str(predicted).upper():
                    parts = str(predicted).upper().split("FINAL", 1)
                    if len(parts) > 1:
                        predicted = parts[1].strip()

                # Convert to string if not already
                predicted = str(predicted) if predicted else ""
        except Exception as e:
            print(f"[RLM ERROR seed={seed}] Completion failed: {e}", flush=True)
            import traceback

            traceback.print_exc()
            predicted = ""

        gold = sample.answer or ""

        # Extract final answer from verbose RLM output
        extracted_predicted = extract_final_answer(predicted)

        # Debug logging for first few seeds to diagnose issues
        if seed < 3:
            print(
                f"[DEBUG seed={seed}] predicted (raw)={predicted[:300] if predicted else 'EMPTY'}...",
                flush=True,
            )
            print(f"[DEBUG seed={seed}] predicted (extracted)={extracted_predicted}", flush=True)
            print(f"[DEBUG seed={seed}] gold={gold}", flush=True)
            print(
                f"[DEBUG seed={seed}] normalized_predicted={normalize_answer(extracted_predicted)}",
                flush=True,
            )
            print(f"[DEBUG seed={seed}] normalized_gold={normalize_answer(gold)}", flush=True)
            print(
                f"[DEBUG seed={seed}] match={normalize_answer(extracted_predicted) == normalize_answer(gold)}",
                flush=True,
            )

        # Use extracted answer for comparison
        predicted = extracted_predicted

        reward = 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0

        return RolloutResponse(
            run_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                details={"messages": messages_for_validation, "predicted": predicted, "gold": gold},
            ),
            trace=None,
            trace_correlation_id=policy_config.get("trace_correlation_id"),
        )

    def provide_taskset_description():
        return {
            "splits": ["validation", "test"],
            "sizes": {"validation": oolong.size("validation"), "test": oolong.size("test")},
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = oolong.sample(split="validation", index=seed)
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": sample.split, "index": sample.index},
                inference={"tool": "rlm_repl"},
                limits={"max_turns": 1},
                task_metadata={"query": sample.query},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description="OOLONG RLM local API for prompt optimization.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for a local API health check using sync httpx."""
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, headers=headers, timeout=5.0)
            if response.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)

    raise RuntimeError(
        f"Health check failed: {health_url} not ready after {timeout}s. "
        "Make sure your task app has a /health endpoint."
    )


# Main async function
async def main():
    # Timing helper
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    timings: dict[str, float] = {}
    total_start = time.time()

    # Start Local API
    print("\n" + "=" * 60)
    print("STARTING LOCAL API")
    print("=" * 60)
    print("Creating OOLONG RLM local API...")
    app = create_oolong_rlm_local_api()

    kill_port(LOCAL_API_PORT)
    run_server_background(app, LOCAL_API_PORT)

    print(f"Waiting for local API on port {LOCAL_API_PORT}...")
    wait_for_health_check_sync("localhost", LOCAL_API_PORT, ENVIRONMENT_API_KEY, timeout=60.0)
    print("Local API ready!")

    if not LOCAL_MODE:
        print("\nProvisioning Cloudflare tunnel...")
        tunnel_start = time.time()
        tunnel = await TunneledLocalAPI.create(
            local_port=LOCAL_API_PORT,
            backend=TUNNEL_BACKEND,
            api_key=API_KEY,
            backend_url=SYNTH_API_BASE,
            progress=True,
        )
        local_api_url = tunnel.url
        timings["tunnel"] = time.time() - tunnel_start
    else:
        print(f"\nUsing {LOCAL_HOST} (no tunnel)...")
        local_api_url = f"http://{LOCAL_HOST}:{LOCAL_API_PORT}"

    print(
        f"Local API URL: {local_api_url}"
        + (f" ({format_duration(timings['tunnel'])})" if "tunnel" in timings else "")
    )

    # Run GEPA optimization
    print("\n" + "=" * 60)
    print("RUNNING GEPA OPTIMIZATION")
    print("=" * 60)

    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": local_api_url,
            "env_name": "oolong",
            "initial_prompt": {
                "messages": [
                    {"role": "system", "order": 0, "pattern": COMPOSED_SYSTEM_PROMPT},
                    {"role": "assistant", "order": 1, "pattern": RLM_CONTEXT_METADATA_PATTERN},
                    {"role": "user", "order": 2, "pattern": RLM_FIRST_USER_PROMPT},
                    {"role": "user", "order": 3, "pattern": BASELINE_USER_PROMPT},
                ],
                "wildcards": {
                    "query": "REQUIRED",
                    "context": "REQUIRED",
                    "context_metadata": "REQUIRED",
                },
            },
            "policy": {
                "model": "gpt-4o-mini",
                "inference_mode": "synth_hosted",
                "provider": "openai",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "gepa": {
                "env_name": "oolong",
                "evaluation": {
                    "seeds": list(range(13)),
                    "validation_seeds": list(range(13, 15)),
                },
                "rollout": {"budget": 6, "max_concurrent": 3, "minibatch_size": 3},
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 2,
                    "num_generations": 1,
                    "children_per_generation": 1,
                },
                "archive": {"size": 10, "pareto_set_size": 10},
                "token": {"counting_model": "gpt-4"},
            },
            "env_config": {
                "split": "validation",
                "max_iterations": 2,
                "max_depth": 0,
            },
        },
    }

    job = PromptLearningJob.from_dict(
        config_dict=config_body,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
    )

    print("\nSubmitting GEPA job...")
    job_id = job.submit()
    print(f"GEPA job submitted: {job_id}")

    optimization_start = time.time()
    print("\nStreaming events...")
    print("Note: RLM rollouts can take 30-60+ seconds each due to recursive LLM calls.")
    print("Polling will continue even if individual status checks timeout.\n")

    # RLM rollouts are slow (multiple recursive calls), so increase interval
    # and handle timeout errors gracefully - the job is still running
    result = job.poll_until_complete(timeout=3600.0, interval=10.0, progress=True)
    timings["optimization"] = time.time() - optimization_start

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final status: {result.status.value}")
    print(f"Best score: {result.best_score:.2%}")
    print(f"Duration: {format_duration(timings['optimization'])}")

    # Timing summary
    timings["total"] = time.time() - total_start
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    if "tunnel" in timings:
        print(f"  Tunnel setup:  {format_duration(timings['tunnel'])}")
    print(f"  Optimization:  {format_duration(timings['optimization'])}")
    print("  ─────────────────────────")
    print(f"  Total:         {format_duration(timings['total'])}")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
