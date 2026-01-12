#!/usr/bin/env python3
"""Run style-matching GEPA prompt optimization (baseline + optimized verifier).

Usage:
    uv run python demos/style_matching/run_demo_prompt_opt.py
    uv run python demos/style_matching/run_demo_prompt_opt.py --verifier-path demos/style_matching/artifacts/verifier_opt.json
"""

import argparse
import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from synth_ai.core.urls import (
    BACKEND_URL_BASE,
    backend_health_url,
    backend_me_url,
    join_url,
)
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.auth import get_or_mint_synth_api_key
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.learning.rl import mint_environment_api_key, setup_environment_api_key
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import (
    TunnelBackend,
    TunneledLocalAPI,
    cleanup_all,
    find_available_port,
    is_port_available,
    kill_port,
)

parser = argparse.ArgumentParser(description="Run style-matching GEPA prompt optimization")
parser.add_argument(
    "--local-host",
    type=str,
    default="127.0.0.1",
    help="Hostname for local task app (default: 127.0.0.1)",
)
parser.add_argument(
    "--verifier-path",
    type=str,
    default="demos/style_matching/artifacts/verifier_opt.json",
    help="Path to verifier optimization artifact JSON",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Path to write prompt optimization artifact JSON",
)
args = parser.parse_args()

SYNTH_API_BASE = BACKEND_URL_BASE
os.environ["BACKEND_BASE_URL"] = SYNTH_API_BASE
LOCAL_MODE = SYNTH_API_BASE.startswith("http://localhost") or SYNTH_API_BASE.startswith(
    "http://127.0.0.1"
)

LOCAL_TASK_PORT = 8132
LOCAL_TASK_HOST = args.local_host


def _validate_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(backend_me_url(SYNTH_API_BASE), headers=headers, timeout=10)
    except Exception:
        return False
    return resp.status_code == 200


print(f"Backend: {SYNTH_API_BASE}")

r = httpx.get(backend_health_url(SYNTH_API_BASE), timeout=30)
if r.status_code != 200:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")
print(f"Backend health: {r.json()}")

API_KEY = get_or_mint_synth_api_key(backend_url=SYNTH_API_BASE, validator=_validate_api_key)
print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

ENVIRONMENT_API_KEY = mint_environment_api_key()
print(f"Minted env key: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")

try:
    result = setup_environment_api_key(SYNTH_API_BASE, API_KEY, token=ENVIRONMENT_API_KEY)
    print(f"Uploaded env key: {result}")
except Exception as exc:
    print(f"Env key upload failed (continuing locally): {exc}")


SYSTEM_PROMPT = (
    "You are a thoughtful essayist with a direct, builder-first voice. "
    "Write crisp, opinionated essays with concrete examples, minimal fluff, "
    "and a short, memorable closing line. Aim for ~500 words (roughly 450-650)."
)

USER_PROMPT_TEMPLATE = (
    "Title: {title}\n"
    "Outline:\n{outline}\n\n"
    "Notes:\n{notes}\n\n"
    "Target length: ~500 words.\n\n"
    "Write the essay now."
)

TASKS: List[Dict[str, Any]] = [
    {
        "id": "outline_speed",
        "input": {
            "title": "Shipping Fast Without Burning Out",
            "outline": (
                "1. Why speed compounds learning\n"
                "2. The tradeoff between velocity and quality\n"
                "3. How to keep teams aligned under pressure\n"
                "4. Practical rituals that preserve momentum\n"
                "5. Closing: pace as a competitive advantage"
            ),
            "notes": ["short feedback loops", "protect maker time", "use small bets"],
        },
    },
    {
        "id": "outline_focus",
        "input": {
            "title": "Focus Beats Optionality",
            "outline": (
                "1. Optionality feels safe but slows decisions\n"
                "2. Focus creates a compounding advantage\n"
                "3. The cost of context switching in small teams\n"
                "4. Saying no as a leadership skill\n"
                "5. Closing: clarity is leverage"
            ),
            "notes": ["pick one wedge", "eliminate parallel bets", "repeat a simple story"],
        },
    },
    {
        "id": "outline_learning",
        "input": {
            "title": "Learning in Public",
            "outline": (
                "1. Writing as a forcing function\n"
                "2. How sharing drafts accelerates feedback\n"
                "3. The credibility flywheel\n"
                "4. Risks of over-sharing (and how to avoid them)\n"
                "5. Closing: publish to clarify"
            ),
            "notes": [
                "ship drafts, not polished essays",
                "ask for specific feedback",
                "be concrete about failures",
            ],
        },
    },
    {
        "id": "outline_quality",
        "input": {
            "title": "Quality as a Constraint",
            "outline": (
                "1. The myth that quality slows you down\n"
                "2. Cheap fixes vs durable systems\n"
                "3. When to accept rough edges\n"
                "4. How to build taste in a team\n"
                "5. Closing: quality is a habit"
            ),
            "notes": ["use guardrails", "invest in tooling", "make quality visible"],
        },
    },
]


def _make_trace(user_text: str, assistant_text: str) -> Dict[str, Any]:
    return {
        "session_id": "trace",
        "session_time_steps": [
            {
                "step_id": "1",
                "step_index": 0,
                "events": [
                    {
                        "event_type": "runtime",
                        "event_id": 1,
                        "type": "user_message",
                        "content": user_text,
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": assistant_text,
                    },
                ],
            }
        ],
    }


GOLD_EXAMPLES: List[Dict[str, Any]] = [
    {
        "summary": "Direct, builder tone with concrete examples",
        "gold_score": 0.95,
        "gold_reasoning": "Short sentences, decisive stance, concrete advice, crisp closing line.",
        "trace": _make_trace(
            "Write about shipping fast.",
            "Speed is a learning engine. Ship a small bet, watch users react, then sharpen the next move. Protect maker time, keep scope tight, and treat every release as feedback, not theater. Momentum beats motivation.",
        ),
    },
    {
        "summary": "Opinionated essay with a sharp closing line",
        "gold_score": 0.92,
        "gold_reasoning": "Clear thesis, direct claims, practical advice, and a memorable end.",
        "trace": _make_trace(
            "Write about focus.",
            "Optionality feels safe, but it dilutes learning. Pick one wedge, cut parallel bets, and ship. Small teams win by saying no early and often. Clarity is leverage.",
        ),
    },
    {
        "summary": "Concrete, tactical quality guidance with guardrails",
        "gold_score": 0.96,
        "gold_reasoning": "Direct stance, concrete guardrails, and a crisp closing line.",
        "trace": _make_trace(
            "Write about quality as a constraint.",
            "Quality is the guardrail that keeps speed from turning into chaos. Ship small, test the riskiest paths, and make rollback cheap. Fix root causes once, then automate the prevention. Quality is a habit, not a milestone.",
        ),
    },
]

BASELINE_VERIFIER_JOB_ID = "zero_shot_verifier_contrastive_single"
VERIFIER_MODEL = "gpt-4.1-nano"
VERIFIER_JOB_ID = BASELINE_VERIFIER_JOB_ID

ROLLOUT_LOG: List[Dict[str, Any]] = []


verifier_artifact = Path(args.verifier_path)
if not verifier_artifact.exists():
    raise RuntimeError(f"Verifier artifact not found: {verifier_artifact}")

verifier_data = json.loads(verifier_artifact.read_text(encoding="utf-8"))
OPTIMIZED_VERIFIER_JOB_ID = verifier_data.get("graph_evolve_job_id")
if not OPTIMIZED_VERIFIER_JOB_ID:
    raise RuntimeError("verifier artifact missing graph_evolve_job_id")


app = FastAPI(title="Style Matching Task App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "task_app": "style_matching"}


@app.get("/task_info")
async def task_info() -> Dict[str, Any]:
    return {
        "taskset": {
            "name": "style_matching",
            "description": "Style matching demo task app",
            "size": len(TASKS),
        }
    }


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {"tasks": TASKS, "gold_examples": GOLD_EXAMPLES}


@app.get("/rollouts")
async def list_rollouts(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    if x_api_key != ENVIRONMENT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"rollouts": ROLLOUT_LOG}


def _format_notes(notes: List[str]) -> str:
    if not notes:
        return "- (none)"
    return "\n".join(f"- {note}" for note in notes)


def _safe_format(text: str, values: Dict[str, Any]) -> str:
    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return text.format_map(_DefaultDict(values))


def _render_messages_from_sections(
    sections: List[Dict[str, Any]], values: Dict[str, Any]
) -> List[Dict[str, str]]:
    rendered = []
    for section in sorted(sections, key=lambda s: s.get("order", 0)):
        role = section.get("role", "user")
        content = section.get("content") or section.get("pattern") or ""
        if content:
            rendered.append({"role": str(role), "content": _safe_format(str(content), values)})
    return rendered


def _build_messages(
    task_input: Dict[str, Any], prompt_sections: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, str]]:
    notes_text = _format_notes(task_input.get("notes", []))
    values = {
        "title": task_input.get("title", ""),
        "outline": task_input.get("outline", ""),
        "notes": notes_text,
    }
    if prompt_sections:
        return _render_messages_from_sections(prompt_sections, values)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=values["title"],
        outline=values["outline"],
        notes=values["notes"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_inference_url(inference_url: str) -> str:
    if "?" in inference_url:
        base, query = inference_url.split("?", 1)
        return f"{join_url(base, '/chat/completions')}?{query}"
    return join_url(inference_url, "/chat/completions")


def _extract_completion(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_verifier_score(result: Dict[str, Any]) -> float:
    output = result.get("output", result)
    if isinstance(output, dict):
        outcome_review = output.get("outcome_review")
        if isinstance(outcome_review, dict) and isinstance(
            outcome_review.get("total"), (int, float)
        ):
            return float(outcome_review["total"])
        event_reviews = output.get("event_reviews")
        if isinstance(event_reviews, list) and event_reviews:
            totals = [rev.get("total") for rev in event_reviews if isinstance(rev, dict)]
            totals = [t for t in totals if isinstance(t, (int, float))]
            if totals:
                return float(sum(totals) / len(totals))
        if isinstance(output.get("total"), (int, float)):
            return float(output["total"])
    return 0.0


async def _call_policy_llm(messages: List[Dict[str, str]], policy_config: Dict[str, Any]) -> str:
    inference_url = policy_config.get("inference_url")
    if not inference_url:
        raise RuntimeError("policy.config.inference_url is required")

    url = _build_inference_url(inference_url)
    model = policy_config.get("model", "gpt-4.1-nano")

    headers = {"Content-Type": "application/json"}
    api_key = policy_config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif ENVIRONMENT_API_KEY:
        headers["X-API-Key"] = ENVIRONMENT_API_KEY
        headers["Authorization"] = f"Bearer {ENVIRONMENT_API_KEY}"

    payload = {"model": model, "messages": messages}
    payload["temperature"] = float(policy_config.get("temperature", 0.7))
    payload["max_tokens"] = int(policy_config.get("max_completion_tokens", 1200))

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return _extract_completion(response.json())


async def _score_with_verifier(
    session_trace: Dict[str, Any], verifier_job_id: Optional[str] = None
) -> float:
    job_id = verifier_job_id or BASELINE_VERIFIER_JOB_ID
    payload = {
        "job_id": job_id,
        "input": {
            "trace": session_trace,
            "gold_examples": GOLD_EXAMPLES,
            "candidate_score": 0.5,
            "candidate_reasoning": "Auto-evaluated from style-matching task app",
            "options": {"model": VERIFIER_MODEL},
        },
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            join_url(SYNTH_API_BASE, "/api/graphs/completions"),
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Verifier failed: HTTP {response.status_code} {response.text[:500]}"
            )
        result = response.json()

    return _extract_verifier_score(result)


@app.post("/rollout")
async def rollout(request: Request, x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    if x_api_key != ENVIRONMENT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON") from None

    run_id = data.get("run_id")
    env = data.get("env", {})
    policy = data.get("policy", {})
    policy_config = policy.get("config", {})

    trace_correlation_id = policy_config.get("trace_correlation_id")

    env_config = env.get("config", {}) or {}
    verifier_job_id = env_config.get("verifier_job_id") or VERIFIER_JOB_ID
    prompt_sections = env_config.get("prompt_sections")

    seed = int(env.get("seed", 0))
    task = TASKS[seed % len(TASKS)]
    task_input = task["input"]

    messages = _build_messages(task_input, prompt_sections=prompt_sections)

    try:
        essay = await _call_policy_llm(messages, policy_config)
    except Exception as exc:
        essay = f"[error: {exc}]"

    session_trace = {
        "session_id": f"style-matching-{task['id']}",
        "session_time_steps": [
            {
                "step_id": "1",
                "step_index": 0,
                "events": [
                    {
                        "event_type": "runtime",
                        "event_id": 1,
                        "type": "user_message",
                        "content": messages[-1]["content"],
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": essay,
                    },
                ],
            }
        ],
    }

    score = await _score_with_verifier(session_trace, verifier_job_id=verifier_job_id)

    ROLLOUT_LOG.append(
        {
            "run_id": run_id,
            "seed": seed,
            "task_id": task["id"],
            "title": task_input.get("title", ""),
            "essay": essay,
            "score": score,
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
            {"steps": [{"observation": task_input, "action": {"essay": essay}, "reward": score}]}
        ],
        "metadata": {"task_id": task["id"]},
        "trace_correlation_id": trace_correlation_id or "",
    }


async def wait_for_system_dns(hostname: str, timeout: float = 90.0, interval: float = 3.0) -> None:
    deadline = time.time() + timeout
    last_exc: Optional[Exception] = None
    while time.time() < deadline:
        try:
            import socket

            socket.gethostbyname(hostname)
            return
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(interval)
    raise RuntimeError(f"System DNS did not resolve {hostname} within {timeout}s: {last_exc}")


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


task_app_thread = None
_task_app_lock = threading.Lock()


def _start_task_app() -> None:
    global LOCAL_TASK_PORT
    global task_app_thread

    kill_port(LOCAL_TASK_PORT)
    if not is_port_available(LOCAL_TASK_PORT):
        LOCAL_TASK_PORT = find_available_port(LOCAL_TASK_PORT + 1)
        print(f"Port in use; switched to {LOCAL_TASK_PORT}")

    task_app_thread = run_server_background(app, LOCAL_TASK_PORT, host=LOCAL_TASK_HOST)
    _wait_for_task_app(LOCAL_TASK_HOST, LOCAL_TASK_PORT, timeout=30.0)


def _start_task_app_monitor(interval: float = 5.0) -> threading.Thread:
    def _monitor() -> None:
        while True:
            time.sleep(interval)
            with _task_app_lock:
                if _task_app_healthcheck(LOCAL_TASK_HOST, LOCAL_TASK_PORT):
                    continue
                print("Task app health check failed; restarting...")
                try:
                    _start_task_app()
                except Exception as exc:
                    print(f"Task app restart failed: {exc}")

    thread = threading.Thread(target=_monitor, daemon=True, name="task-app-monitor")
    thread.start()
    return thread


async def run_gepa_with_verifier(label: str, verifier_job_id: str) -> tuple[str, Any]:
    global VERIFIER_JOB_ID
    VERIFIER_JOB_ID = verifier_job_id
    print(f"\nRunning GEPA ({label}) with verifier: {verifier_job_id}")

    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": TASK_APP_URL,
            "task_app_api_key": ENVIRONMENT_API_KEY,
            "env_name": "style-matching",
            "initial_prompt": {
                "messages": [
                    {"role": "system", "order": 0, "pattern": SYSTEM_PROMPT},
                    {"role": "user", "order": 1, "pattern": USER_PROMPT_TEMPLATE},
                ],
                "wildcards": {"title": "REQUIRED", "outline": "REQUIRED", "notes": "REQUIRED"},
            },
            "policy": {
                "inference_mode": "synth_hosted",
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "temperature": 0.7,
                "max_completion_tokens": 1200,
            },
            "gepa": {
                "env_name": "style-matching",
                "evaluation": {
                    "seeds": list(range(13)),
                    "validation_seeds": list(range(13, 17)),
                },
                "rollout": {"budget": 48, "max_concurrent": 3, "minibatch_size": 3},
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

    job = PromptLearningJob.from_dict(
        config_dict=config_body,
        backend_url=SYNTH_API_BASE,
        task_app_api_key=ENVIRONMENT_API_KEY,
    )

    job_id = job.submit()
    print(f"GEPA job id ({label}): {job_id}")

    result = job.poll_until_complete(timeout=3600.0, interval=3.0, progress=True)
    print(f"GEPA finished ({label}): {result.status.value}")
    if result.failed:
        raise RuntimeError(f"GEPA job failed ({label}): {result.error}")
    return job_id, result


def _select_prompt(prompt_results: Any) -> Dict[str, Any]:
    best = getattr(prompt_results, "best_prompt", None)
    if best:
        return _prompt_to_dict(best)
    top = getattr(prompt_results, "top_prompts", None) or []
    if top:
        return _prompt_to_dict(top[0])
    raise RuntimeError("No prompts returned from GEPA job")


def _prompt_to_dict(prompt_obj: Any) -> Dict[str, Any]:
    if isinstance(prompt_obj, dict):
        return prompt_obj
    if hasattr(prompt_obj, "model_dump"):
        return prompt_obj.model_dump()
    if hasattr(prompt_obj, "dict"):
        return prompt_obj.dict()
    if isinstance(prompt_obj, str):
        return {"full_text": prompt_obj}
    return {"value": str(prompt_obj)}


async def main() -> None:
    global TASK_APP_URL

    print(f"Starting task app on port {LOCAL_TASK_PORT}...")
    with _task_app_lock:
        _start_task_app()
    print("Task app ready!")
    _start_task_app_monitor()

    if LOCAL_MODE:
        print("Using localhost task app URL (no tunnel)")
        TASK_APP_URL = f"http://{LOCAL_TASK_HOST}:{LOCAL_TASK_PORT}"
        tunnel = None
    else:
        print("Provisioning Cloudflare tunnel...")
        try:
            tunnel = await TunneledLocalAPI.create(
                local_port=LOCAL_TASK_PORT,
                backend=TunnelBackend.CloudflareManagedTunnel,
                api_key=API_KEY,
                env_api_key=ENVIRONMENT_API_KEY,
                backend_url=SYNTH_API_BASE,
                reason="style_matching_prompt_opt",
                progress=True,
            )
            print(f"Waiting for system DNS to resolve {tunnel.hostname}...")
            await wait_for_system_dns(tunnel.hostname)
        except Exception as exc:
            print(
                f"Managed tunnel failed or DNS unresolved ({exc}). Falling back to quick tunnel..."
            )
            tunnel = await TunneledLocalAPI.create(
                local_port=LOCAL_TASK_PORT,
                backend=TunnelBackend.CloudflareQuickTunnel,
                env_api_key=ENVIRONMENT_API_KEY,
                progress=True,
            )

    if tunnel is not None:
        TASK_APP_URL = tunnel.url
    print(f"Task app URL: {TASK_APP_URL}")

    baseline_gepa_job_id, baseline_gepa_result = await run_gepa_with_verifier(
        "baseline", BASELINE_VERIFIER_JOB_ID
    )

    optimized_gepa_job_id, optimized_gepa_result = await run_gepa_with_verifier(
        "optimized", OPTIMIZED_VERIFIER_JOB_ID
    )

    pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)

    baseline_prompts = await pl_client.get_prompts(baseline_gepa_job_id)
    optimized_prompts = await pl_client.get_prompts(optimized_gepa_job_id)

    baseline_prompt_obj = _select_prompt(baseline_prompts)
    optimized_prompt_obj = _select_prompt(optimized_prompts)

    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else artifacts_dir / "prompt_opt.json"

    payload = {
        "baseline_gepa_job_id": baseline_gepa_job_id,
        "optimized_gepa_job_id": optimized_gepa_job_id,
        "baseline_best_score": getattr(baseline_prompts, "best_score", None),
        "optimized_best_score": getattr(optimized_prompts, "best_score", None),
        "baseline_prompt": baseline_prompt_obj,
        "optimized_prompt": optimized_prompt_obj,
        "baseline_verifier_job_id": BASELINE_VERIFIER_JOB_ID,
        "optimized_verifier_job_id": OPTIMIZED_VERIFIER_JOB_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBaseline best score:", payload["baseline_best_score"])
    print("Optimized best score:", payload["optimized_best_score"])
    print(f"Saved prompt artifact: {output_path}")

    if not LOCAL_MODE:
        cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())
