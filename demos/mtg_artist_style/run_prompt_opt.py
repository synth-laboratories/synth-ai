#!/usr/bin/env python3
"""Run GEPA prompt optimization for MTG artist style matching.

Uses the optimized verifier from run_verifier_opt.py to optimize prompts that:
1. Generate images matching the artist's style
2. Do NOT mention the artist's name (verifier returns 0 if they do)

Usage:
    uv run python demos/mtg_artist_style/run_prompt_opt.py --artist seb_mckinnon
    uv run python demos/mtg_artist_style/run_prompt_opt.py --artist seb_mckinnon --local
"""

import argparse
import asyncio
import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from synth_ai.sdk import PromptLearningJob, cleanup_all, find_available_port, is_port_available, kill_port
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi.auth import mint_environment_api_key, setup_environment_api_key
from synth_ai.sdk.localapi._impl.server import run_server_background
from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

parser = argparse.ArgumentParser(description="Run MTG artist GEPA prompt optimization")
parser.add_argument(
    "--artist",
    type=str,
    default="seb_mckinnon",
    help="Artist key (see README for full list of 18 artists)",
)
parser.add_argument(
    "--local",
    action="store_true",
    help="Run in local mode: use localhost:8000 backend",
)
parser.add_argument(
    "--verifier-artifact",
    type=str,
    default=None,
    help="Path to verifier_opt.json artifact (default: artifacts/<artist>/verifier_opt.json)",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Path to write prompt optimization artifact JSON",
)
args = parser.parse_args()

demo_dir = Path(__file__).resolve().parent
synth_root = demo_dir.parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("' ")
        if key:
            os.environ[key] = value


_load_env_file(synth_root / ".env")

USE_LOCAL_BACKEND = args.local
# Respect SYNTH_API_BASE from environment if set
_env_base = os.environ.get("SYNTH_API_BASE", "").strip()
if USE_LOCAL_BACKEND:
    SYNTH_API_BASE = "http://127.0.0.1:8000"
elif _env_base:
    SYNTH_API_BASE = _env_base
else:
    SYNTH_API_BASE = "https://api.usesynth.ai"
os.environ["BACKEND_BASE_URL"] = SYNTH_API_BASE


def _validate_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(f"{SYNTH_API_BASE}/api/v1/me", headers=headers, timeout=10)
    except Exception:
        return False
    return resp.status_code == 200


print(f"Backend: {SYNTH_API_BASE}")

r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code != 200:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")
print(f"Backend health: {r.json()}")

API_KEY = os.environ.get("SYNTH_API_KEY", "").strip()
if not API_KEY or not _validate_api_key(API_KEY):
    print("SYNTH_API_KEY missing or invalid; minting demo key...")
    resp = httpx.post(f"{SYNTH_API_BASE}/api/demo/keys", json={"ttl_hours": 4}, timeout=30)
    resp.raise_for_status()
    API_KEY = resp.json()["api_key"]
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

# Mint environment API key
ENVIRONMENT_API_KEY = mint_environment_api_key()
print(f"Minted env key: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")

try:
    result = setup_environment_api_key(SYNTH_API_BASE, API_KEY, token=ENVIRONMENT_API_KEY)
    print(f"Uploaded env key: {result}")
except Exception as exc:
    print(f"Env key upload failed (continuing locally): {exc}")

# Load artist metadata
metadata_path = demo_dir / "artist_metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(
        f"Artist metadata not found. Run fetch_artist_cards.py first:\n"
        f"  uv run python demos/mtg_artist_style/fetch_artist_cards.py"
    )

with open(metadata_path) as f:
    artist_metadata = json.load(f)

artist_info = artist_metadata["artists"].get(args.artist)
if not artist_info:
    available = list(artist_metadata["artists"].keys())
    raise ValueError(f"Unknown artist '{args.artist}'. Available: {available}")

ARTIST_KEY = args.artist
ARTIST_NAME = artist_info["name"]
STYLE_DESCRIPTION = artist_info["style_description"]

# Get cards for this artist
artist_cards = [c for c in artist_metadata["cards"] if c["artist_key"] == ARTIST_KEY]

print(f"\nArtist: {ARTIST_NAME}")
print(f"Style: {STYLE_DESCRIPTION}")
print(f"Reference cards: {len(artist_cards)}")

# Load verifier artifact
verifier_artifact_path = (
    Path(args.verifier_artifact)
    if args.verifier_artifact
    else demo_dir / "artifacts" / ARTIST_KEY / "verifier_opt.json"
)

if not verifier_artifact_path.exists():
    raise FileNotFoundError(
        f"Verifier artifact not found at {verifier_artifact_path}\n"
        f"Run verifier optimization first:\n"
        f"  uv run python demos/mtg_artist_style/run_verifier_opt.py --artist {ARTIST_KEY}"
    )

with open(verifier_artifact_path) as f:
    verifier_artifact = json.load(f)

VERIFIER_JOB_ID = verifier_artifact["graph_id"]
FORBIDDEN_PATTERNS = verifier_artifact["forbidden_patterns"]

print(f"Using verifier graph: {VERIFIER_JOB_ID}")
print(f"Forbidden patterns: {FORBIDDEN_PATTERNS}")

# Build forbidden name patterns
name_parts = ARTIST_NAME.lower().split()

LOCAL_TASK_PORT = 8133
LOCAL_TASK_HOST = "127.0.0.1"

# Task definitions
SUBJECTS = [
    "a mysterious forest spirit",
    "an ancient warrior in armor",
    "a magical creature emerging from mist",
    "a dark sorcerer casting a spell",
    "an ethereal landscape at dusk",
    "a mythical beast in shadows",
    "a haunted castle on a cliff",
    "a celestial being descending",
]

TASKS = [
    {"id": f"task_{i}", "input": {"subject": subject, "card_ref": artist_cards[i % len(artist_cards)]["card_name"]}}
    for i, subject in enumerate(SUBJECTS)
]

# Initial prompt template (will be optimized)
INITIAL_SYSTEM_PROMPT = (
    f"You are an artist creating fantasy illustrations. "
    f"Your style should capture: {STYLE_DESCRIPTION}. "
    f"Focus on mood, atmosphere, and artistic technique."
)

INITIAL_USER_PROMPT = "Create an illustration of {subject}. Match the artistic style shown in reference works."


def _load_image_as_data_url(image_path: str) -> str:
    """Load an image file as a base64 data URL."""
    full_path = demo_dir / image_path
    with open(full_path, "rb") as f:
        img_data = f.read()
    ext = full_path.suffix.lower()
    mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        ext, "image/jpeg"
    )
    return f"data:{mime_type};base64,{base64.b64encode(img_data).decode('ascii')}"


# Gold examples for contrastive verifier
GOLD_EXAMPLES = []
for i, card in enumerate(artist_cards[:4]):
    GOLD_EXAMPLES.append(
        {
            "summary": f"Reference art: {card['card_name']} by {ARTIST_NAME}",
            "gold_score": 0.95,
            "gold_reasoning": f"Exemplifies {STYLE_DESCRIPTION}",
            "image_url": _load_image_as_data_url(card["image_path"]),
        }
    )

ROLLOUT_LOG: list[dict[str, Any]] = []


def _verify_api_key(x_api_key: Optional[str]) -> bool:
    if not ENVIRONMENT_API_KEY:
        return True
    return x_api_key == ENVIRONMENT_API_KEY


def _safe_format(text: str, values: dict[str, Any]) -> str:
    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return text.format_map(_DefaultDict(values))


def _render_messages_from_sections(
    sections: list[dict[str, Any]], values: dict[str, Any]
) -> list[dict[str, str]]:
    rendered = []
    for section in sorted(sections, key=lambda s: s.get("order", 0)):
        role = section.get("role", "user")
        content = section.get("content") or section.get("pattern") or ""
        if content:
            rendered.append({"role": str(role), "content": _safe_format(str(content), values)})
    return rendered


def _build_messages(
    task_input: dict[str, Any], prompt_sections: Optional[list[dict[str, Any]]] = None
) -> list[dict[str, str]]:
    values = {"subject": task_input.get("subject", ""), "card_ref": task_input.get("card_ref", "")}
    if prompt_sections:
        return _render_messages_from_sections(prompt_sections, values)
    return [
        {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": _safe_format(INITIAL_USER_PROMPT, values)},
    ]


def _build_inference_url(inference_url: str) -> str:
    if "?" in inference_url:
        base, query = inference_url.split("?", 1)
        return f"{base.rstrip('/')}/chat/completions?{query}"
    return f"{inference_url.rstrip('/')}/chat/completions"


async def _call_policy_llm(messages: list[dict[str, str]], policy_config: dict[str, Any]) -> dict[str, Any]:
    inference_url = policy_config.get("inference_url")
    if not inference_url:
        raise RuntimeError("policy.config.inference_url is required")

    url = _build_inference_url(inference_url)
    model = policy_config.get("model", "gemini-2.5-flash-image")

    headers = {"Content-Type": "application/json"}
    api_key = policy_config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif ENVIRONMENT_API_KEY:
        headers["X-API-Key"] = ENVIRONMENT_API_KEY
        headers["Authorization"] = f"Bearer {ENVIRONMENT_API_KEY}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(policy_config.get("temperature", 0.7)),
        "max_tokens": int(policy_config.get("max_completion_tokens", 4096)),
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def _extract_image_from_response(data: dict[str, Any]) -> Optional[str]:
    """Extract image URL from LLM response."""
    choices = data.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return part.get("image_url", {}).get("url")
    elif isinstance(content, str) and content.startswith("data:image/"):
        return content
    return None


async def _score_with_verifier(
    prompt_text: str, image_url: str, verifier_job_id: str
) -> float:
    """Score using the optimized verifier."""
    trace = {
        "session_id": "scoring",
        "session_time_steps": [
            {
                "step_id": "1",
                "step_index": 0,
                "events": [
                    {"event_type": "runtime", "event_id": 1, "type": "user_message", "content": prompt_text},
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    },
                ],
            }
        ],
    }

    payload = {
        "job_id": verifier_job_id,
        "input": {
            "trace": trace,
            "gold_examples": GOLD_EXAMPLES,
            "candidate_score": 0.5,
            "candidate_reasoning": "Auto-evaluated",
        },
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{SYNTH_API_BASE.rstrip('/')}/api/graphs/completions",
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Verifier failed: {response.status_code} {response.text[:500]}")
        result = response.json()

    # Extract score
    output = result.get("output", result)
    if isinstance(output, dict):
        if "score" in output:
            return float(output["score"])
        outcome = output.get("outcome_review", {})
        if isinstance(outcome, dict) and "total" in outcome:
            return float(outcome["total"])
    return 0.0


# FastAPI app
app = FastAPI(title="MTG Artist Style Task App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "task_app": f"mtg_artist_{ARTIST_KEY}"}


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    return {
        "taskset": {
            "name": f"mtg_artist_{ARTIST_KEY}",
            "description": f"MTG artist style matching for {ARTIST_NAME}",
            "size": len(TASKS),
        }
    }


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    return {"tasks": TASKS, "gold_examples": GOLD_EXAMPLES}


@app.post("/rollout")
async def rollout(request: Request, x_api_key: Optional[str] = Header(None)) -> dict[str, Any]:
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    run_id = data.get("run_id")
    env = data.get("env", {})
    policy = data.get("policy", {})
    policy_config = policy.get("config", {})
    trace_correlation_id = policy_config.get("trace_correlation_id")

    env_config = env.get("config", {}) or {}
    prompt_sections = env_config.get("prompt_sections")
    verifier_job_id = env_config.get("verifier_job_id") or VERIFIER_JOB_ID

    seed = int(env.get("seed", 0))
    task = TASKS[seed % len(TASKS)]
    task_input = task["input"]

    messages = _build_messages(task_input, prompt_sections=prompt_sections)

    # Get the full prompt text for verifier checking
    prompt_text = " ".join(m["content"] for m in messages)

    try:
        llm_response = await _call_policy_llm(messages, policy_config)
        image_url = _extract_image_from_response(llm_response)
    except Exception as exc:
        image_url = None
        llm_response = {"error": str(exc)}

    if not image_url:
        # No image generated - low score
        score = 0.1
    else:
        try:
            score = await _score_with_verifier(prompt_text, image_url, verifier_job_id)
        except Exception as exc:
            print(f"Verifier error: {exc}")
            score = 0.1

    ROLLOUT_LOG.append(
        {
            "run_id": run_id,
            "seed": seed,
            "task_id": task["id"],
            "subject": task_input.get("subject", ""),
            "score": score,
            "has_image": bool(image_url),
        }
    )

    return {
        "metrics": {
            "mean_return": score,
            "outcome_score": score,
            "num_steps": 1,
            "details": {"verifier_score": score},
        },
        "trajectories": [
            {"steps": [{"observation": task_input, "action": {"image_generated": bool(image_url)}, "reward": score}]}
        ],
        "metadata": {"task_id": task["id"]},
        "trace_correlation_id": trace_correlation_id or "",
    }


def _task_app_healthcheck(host: str, port: int) -> bool:
    try:
        resp = httpx.get(
            f"http://{host}:{port}/health",
            headers={"X-API-Key": ENVIRONMENT_API_KEY},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _wait_for_task_app(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _task_app_healthcheck(host, port):
            return
        time.sleep(1.0)
    raise RuntimeError(f"Task app health check failed after {timeout}s")


def _start_task_app() -> None:
    global LOCAL_TASK_PORT

    kill_port(LOCAL_TASK_PORT)
    if not is_port_available(LOCAL_TASK_PORT):
        LOCAL_TASK_PORT = find_available_port(LOCAL_TASK_PORT + 1)
        print(f"Port in use; switched to {LOCAL_TASK_PORT}")

    run_server_background(app, LOCAL_TASK_PORT, host=LOCAL_TASK_HOST)
    _wait_for_task_app(LOCAL_TASK_HOST, LOCAL_TASK_PORT, timeout=30.0)


def _make_gepa_config(task_app_url: str) -> dict[str, Any]:
    return {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_app_url,
            "task_app_api_key": ENVIRONMENT_API_KEY,
            "env_name": f"mtg-artist-{ARTIST_KEY}",
            "initial_prompt": {
                "messages": [
                    {"role": "system", "order": 0, "pattern": INITIAL_SYSTEM_PROMPT},
                    {"role": "user", "order": 1, "pattern": INITIAL_USER_PROMPT},
                ],
                "wildcards": {"subject": "REQUIRED", "card_ref": "OPTIONAL"},
            },
            "policy": {
                "inference_mode": "synth_hosted",
                "model": "gemini-2.5-flash-image",
                "provider": "google",
                "temperature": 0.7,
                "max_completion_tokens": 4096,
            },
            "gepa": {
                "env_name": f"mtg-artist-{ARTIST_KEY}",
                "evaluation": {"seeds": list(range(13)), "validation_seeds": list(range(13, 15))},
                "rollout": {"budget": 30, "max_concurrent": 2, "minibatch_size": 2},
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 3,
                    "num_generations": 3,
                    "children_per_generation": 2,
                },
                "archive": {"size": 5, "pareto_set_size": 10},
                "token": {"counting_model": "gpt-4"},
            },
            "verifier": {"enabled": False, "reward_source": "task_app"},
        }
    }


async def run_gepa() -> tuple[str, Any]:
    print(f"\nStarting task app on port {LOCAL_TASK_PORT}...")
    _start_task_app()
    print("Task app ready!")

    if USE_LOCAL_BACKEND:
        print("Using localhost task app URL (no tunnel)")
        task_app_url = f"http://{LOCAL_TASK_HOST}:{LOCAL_TASK_PORT}"
    else:
        print("Provisioning Cloudflare tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=LOCAL_TASK_PORT,
            backend=TunnelBackend.CloudflareQuickTunnel,
            env_api_key=ENVIRONMENT_API_KEY,
            progress=True,
        )
        task_app_url = tunnel.url
        print(f"Tunnel URL: {task_app_url}")

    print(f"\nRunning GEPA for {ARTIST_NAME} (prompt cannot mention artist name)...")

    config_body = _make_gepa_config(task_app_url)
    job = PromptLearningJob.from_dict(
        config_dict=config_body,
        backend_url=SYNTH_API_BASE,
        localapi_api_key=ENVIRONMENT_API_KEY,
    )

    job_id = job.submit()
    print(f"GEPA job id: {job_id}")

    result = job.poll_until_complete(timeout=3600.0, interval=3.0, progress=True)
    print(f"GEPA finished: {result.status.value}")

    if result.failed:
        raise RuntimeError(f"GEPA job failed: {result.error}")

    return job_id, result


async def main() -> None:
    job_id, result = await run_gepa()

    # Fetch best prompt
    pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
    prompts = await pl_client.get_prompts(job_id)

    best_prompt = prompts.best_prompt
    if not best_prompt and prompts.top_prompts:
        best_prompt = prompts.top_prompts[0]

    # Check if best prompt contains forbidden patterns
    prompt_text = ""
    if best_prompt:
        template = best_prompt.get("template", {})
        sections = template.get("sections", []) or best_prompt.get("messages", [])
        prompt_text = " ".join(s.get("content", "") or s.get("pattern", "") for s in sections)

    contains_name = any(p in prompt_text.lower() for p in FORBIDDEN_PATTERNS)

    # Save results
    artifacts_dir = demo_dir / "artifacts" / ARTIST_KEY
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else artifacts_dir / "prompt_opt.json"

    payload = {
        "artist_key": ARTIST_KEY,
        "artist_name": ARTIST_NAME,
        "gepa_job_id": job_id,
        "verifier_graph_id": VERIFIER_JOB_ID,
        "best_score": prompts.best_score,
        "best_prompt": best_prompt,
        "contains_forbidden_pattern": contains_name,
        "forbidden_patterns": FORBIDDEN_PATTERNS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print("GEPA Prompt Optimization Complete")
    print("=" * 60)
    print(f"Artist: {ARTIST_NAME}")
    print(f"Best score: {prompts.best_score}")
    print(f"Contains artist name: {contains_name}")
    if contains_name:
        print("  ⚠️  WARNING: Optimized prompt still contains artist name!")
    else:
        print("  ✅ Success: Prompt captures style without naming artist!")
    print(f"\nSaved artifact: {output_path}")

    if best_prompt:
        print("\n" + "-" * 40)
        print("Best Prompt:")
        print("-" * 40)
        print(prompt_text[:500])
        if len(prompt_text) > 500:
            print("...")

    cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())
