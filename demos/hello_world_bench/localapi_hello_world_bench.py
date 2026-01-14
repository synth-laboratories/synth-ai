"""
LocalAPI Task App - hello_world_bench (GEPA-ready)

This is a tiny task app intended to be:
- fast (single file write)
- deterministic (binary outcome reward)
- compatible with prompt-learning / GEPA runs via the Synth backend

Key design choice:
The underlying agent is OpenCode, so the *raw* LLM request template is OpenCode's
`developer` harness + a `user` message containing the task prompt.

To make GEPA pattern mode workable (once canonicalization/mapping is implemented
on the backend side), the task prompt we send to OpenCode explicitly embeds:
- a "system-like" instruction block (context_override.system_prompt)
- a task instance description (context_override.task_description)
using clear tags (<SYSTEM>...</SYSTEM>, <TASK>...</TASK>) so template extraction
can reconstruct canonical messages for validation and delta application.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import Request

from synth_ai.data.artifacts import Artifact
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.task.rubrics.models import Criterion, Rubric
from synth_ai.sdk.task.server import RubricBundle

logger = logging.getLogger(__name__)

APP_ID = "hello_world_bench"
APP_NAME = "Hello World Bench (OpenCode)"


DEFAULT_SYSTEM_PROMPT = """You are a precise coding agent.

You MUST follow the task instructions exactly.
Your output should be minimal and only include what the task requests.
"""


DEFAULT_TASK_DESCRIPTION = """TASK:
1) First, READ output.txt using the read tool (required before editing/writing).
2) Then use the write OR edit tool to modify output.txt.
3) After your change, output.txt MUST contain EXACTLY this single line (no extra whitespace):
Hello, world!
4) Do NOT run bash. Do NOT read any other files.
5) After writing the file, respond with exactly: DONE
"""


def provide_taskset_description() -> dict[str, Any]:
    return {"id": APP_ID, "splits": ["default"], "sizes": {"default": 1, "total": 1}}


def provide_task_instances(seeds: list[int]):
    # We don't meaningfully vary the task per seed yet, but GEPA requires many seeds.
    # We still emit distinct instance_id values for bookkeeping.
    for seed in seeds:
        yield TaskInfo(
            task={"id": APP_ID, "name": APP_NAME},
            dataset={
                "id": APP_ID,
                "split": "default",
                "index": seed,
                "instance_id": f"hw-{seed}",
            },
            inference={"tool": "code_edit"},
            limits={"max_turns": 10},
            task_metadata={"seed": seed},
        )


def _normalize_interceptor_base(inference_url: str) -> tuple[str, str | None]:
    parsed = urlparse(inference_url)
    base_path = parsed.path or ""
    for suffix in ["/v1/chat/completions", "/chat/completions", "/responses", "/v1/responses"]:
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
    cid_values = parse_qs(parsed.query).get("cid", [])
    correlation_id = cid_values[0] if cid_values else None
    return base, correlation_id


async def _run_codex_agent(
    prompt: str,
    sandbox_dir: Path,
    *,
    model: str,
    timeout: int,
    inference_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    """Run Codex CLI agent on the sandbox.

    Args:
        prompt: The task prompt for the agent
        sandbox_dir: Directory to run the agent in
        model: Model to use (e.g. "gpt-5-nano")
        timeout: Timeout in seconds
        inference_url: Synth interceptor URL to route LLM calls through
        api_key: API key for the interceptor
    """
    import shutil

    if not shutil.which("codex"):
        return {
            "success": False,
            "stdout": "",
            "stderr": "codex binary not found in PATH",
        }

    config_dir = Path.home() / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"

    base_url = "https://api.openai.com/v1"
    if inference_url:
        base_url, correlation_id = _normalize_interceptor_base(inference_url)
        if correlation_id:
            base_url = f"{base_url}/{correlation_id}"
        print(f"[Codex] Parsed inference_url: base_url={base_url}", flush=True)

    model_id = model.split("/", 1)[1] if "/" in model else model

    config_content = f"""# Auto-generated for hello_world_bench runs

model = "{model_id}"
model_provider = "openai"

[model_providers.openai]
name = "OpenAI"
base_url = "{base_url}"
wire_api = "responses"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = 300000

[mcp]
enabled = false
"""
    config_file.write_text(config_content)
    print(f"[Codex] Config written to: {config_file}", flush=True)
    print(f"[Codex] Model: {model_id}", flush=True)
    print(f"[Codex] BaseURL: {base_url}", flush=True)

    cmd = [
        "codex",
        "exec",
        "--yolo",
        "--skip-git-repo-check",
        "-m",
        model_id,
        prompt,
    ]

    env = os.environ.copy()
    actual_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    env["OPENAI_API_KEY"] = actual_api_key
    env["OPENAI_MODEL"] = model_id
    if inference_url:
        env["OPENAI_BASE_URL"] = base_url

    print(f"[Codex] Working directory: {sandbox_dir}", flush=True)
    print(f"[Codex] Command: {' '.join(cmd[:4])}...", flush=True)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(sandbox_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(timeout))
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"success": False, "stdout": "", "stderr": f"timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}


async def _run_opencode_agent(
    prompt: str,
    sandbox_dir: Path,
    *,
    model: str,
    timeout: int,
    inference_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    opencode_bin = os.environ.get("OPENCODE_BIN") or str(Path.home() / ".opencode" / "bin" / "opencode")
    if not Path(opencode_bin).exists():
        return {"success": False, "stdout": "", "stderr": f"opencode binary not found: {opencode_bin}"}

    model_id = model.split("/", 1)[1] if "/" in model else model
    model_with_provider = f"openai/{model_id}"

    base_url = ""
    if inference_url:
        base_url, correlation_id = _normalize_interceptor_base(inference_url)
        print(f"[OpenCode] Parsed inference_url: base_url={base_url} correlation_id={correlation_id}", flush=True)
        if correlation_id:
            base_url = f"{base_url}/{correlation_id}"
        print(f"[OpenCode] Final baseURL for OpenCode config: {base_url}", flush=True)
    else:
        print(f"[OpenCode] NO inference_url provided, using direct OpenAI", flush=True)

    actual_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "placeholder"
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "model": model_with_provider,
        "provider": {
            "openai": {
                "name": "OpenAI",
                "npm": "@ai-sdk/openai",
                "options": {
                    # OpenCode requires some API key string client-side.
                    "apiKey": actual_api_key,
                    # Critical: route via interceptor if provided.
                    "baseURL": base_url or "https://api.openai.com/v1",
                },
                "models": {"gpt-5-nano": {}, "gpt-5.2": {}, "gpt-4o": {}, "gpt-4o-mini": {}},
            }
        },
        "permission": {
            "*": "allow",
            "external_directory": "allow",
            "bash": "allow",
            "read": "allow",
            "write": "allow",
            "edit": "allow",
            "list": "allow",
            "glob": "allow",
            "grep": "allow",
        },
    }

    (sandbox_dir / "opencode.json").write_text(json.dumps(opencode_config, indent=2))

    cmd = [
        opencode_bin,
        "run",
        "--format",
        "json",
        "--model",
        model_with_provider,
        prompt,
    ]

    env = os.environ.copy()
    if actual_api_key and actual_api_key != "placeholder":
        env["OPENAI_API_KEY"] = actual_api_key

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(sandbox_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def read_stream(stream, chunks):
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                break
            chunks.append(chunk)

    stdout_task = asyncio.create_task(read_stream(proc.stdout, stdout_chunks))
    stderr_task = asyncio.create_task(read_stream(proc.stderr, stderr_chunks))

    try:
        await asyncio.wait_for(proc.wait(), timeout=float(timeout))
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"success": False, "stdout": "", "stderr": f"timeout after {timeout}s"}
    finally:
        for t in (stdout_task, stderr_task):
            t.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    return {
        "success": proc.returncode == 0,
        "stdout": b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        "stderr": b"".join(stderr_chunks).decode("utf-8", errors="replace"),
    }


def _write_sandbox_files(sandbox_dir: Path) -> None:
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    # Prevent OpenCode from searching for AGENTS.md up the tree.
    (sandbox_dir / "AGENTS.md").write_text(
        "# Agent Instructions\n\nEdit output.txt per the task prompt, then stop.\n",
        encoding="utf-8",
    )
    (sandbox_dir / "output.txt").write_text(
        "TODO: Replace this entire file with exactly one line: Hello, world!\n",
        encoding="utf-8",
    )


def _build_opencode_prompt(system_prompt: str, task_description: str) -> str:
    # Tagging makes it straightforward to reconstruct canonical messages later.
    return (
        "<SYSTEM>\n"
        f"{system_prompt.strip()}\n"
        "</SYSTEM>\n\n"
        "<TASK>\n"
        f"{task_description.strip()}\n"
        "</TASK>\n"
    )


async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    seed = request.env.seed or 0
    policy_config = request.policy.config or {}
    context_override = getattr(request, "context_override", None) or {}

    model = str(policy_config.get("model", "gpt-5-nano"))
    timeout = int(policy_config.get("timeout", 120))
    inference_url = policy_config.get("inference_url")
    agent_type = str(policy_config.get("agent", "opencode")).lower()
    api_key = os.environ.get("SYNTH_API_KEY")

    system_prompt = str(context_override.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    task_description = str(context_override.get("task_description", DEFAULT_TASK_DESCRIPTION))
    prompt = _build_opencode_prompt(system_prompt, task_description)

    # DEBUG: Log what we're receiving
    print(f"[hello_world_bench] ========================================", flush=True)
    print(f"[hello_world_bench] ROLLOUT DEBUG", flush=True)
    print(f"[hello_world_bench] inference_url: {inference_url}", flush=True)
    print(f"[hello_world_bench] model: {model}", flush=True)
    print(f"[hello_world_bench] agent_type: {agent_type}", flush=True)
    print(f"[hello_world_bench] api_key present: {bool(api_key)}", flush=True)
    print(f"[hello_world_bench] ========================================", flush=True)

    start = time.perf_counter()
    agent_result: dict[str, Any] = {"success": False, "stdout": "", "stderr": ""}

    with tempfile.TemporaryDirectory(prefix="hello_world_gepa_") as td:
        sandbox_dir = Path(td) / "sandbox"
        _write_sandbox_files(sandbox_dir)

        if agent_type == "codex":
            agent_result = await _run_codex_agent(
                prompt,
                sandbox_dir,
                model=model,
                timeout=timeout,
                inference_url=inference_url,
                api_key=api_key,
            )
        elif agent_type == "opencode":
            agent_result = await _run_opencode_agent(
                prompt,
                sandbox_dir,
                model=model,
                timeout=timeout,
                inference_url=inference_url,
                api_key=api_key,
            )
        else:
            agent_result = {
                "success": False,
                "stdout": "",
                "stderr": f"unsupported agent type: {agent_type} (supported: opencode, codex)",
            }

        output_path = sandbox_dir / "output.txt"
        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""

    outcome_reward_value = 1.0 if final_output.strip() == "Hello, world!" else 0.0
    latency_ms = (time.perf_counter() - start) * 1000.0
    trace_correlation_id = policy_config.get("trace_correlation_id", request.trace_correlation_id)

    artifacts = [
        Artifact(
            name="output_txt",
            content=final_output,
            type="text",
            content_type="text/plain",
        )
    ]

    return RolloutResponse(
        trace_correlation_id=trace_correlation_id,
        metrics=RolloutMetrics(
            outcome_reward=outcome_reward_value,
            details={
                "seed": seed,
                "latency_ms": latency_ms,
                "agent": agent_type,
                "agent_success": agent_result.get("success"),
                "agent_stdout_tail": str(agent_result.get("stdout", ""))[-1500:],
                "agent_stderr_tail": str(agent_result.get("stderr", ""))[-2000:],
            },
            outcome_objectives={"reward": outcome_reward_value, "latency_ms": latency_ms},
            instance_objectives=[{"reward": outcome_reward_value, "latency_ms": latency_ms}],
        ),
        trace=None,  # Let the backend hydrate from interceptor traces
        artifact=artifacts,
        success_status=SuccessStatus.SUCCESS,
    )


HELLO_WORLD_OUTCOME_RUBRIC = Rubric(
    version="1.0",
    goal_text="Check whether output.txt contains exactly 'Hello, world!'",
    criteria=[
        Criterion(
            id="exact_output",
            description="output.txt matches exactly: Hello, world!",
            weight=1.0,
            required=True,
        )
    ],
    aggregation="weighted_sum",
)

HELLO_WORLD_EVENT_RUBRIC = Rubric(
    version="1.0",
    goal_text="Evaluate intermediate agent actions (lightweight).",
    criteria=[
        Criterion(id="follows_instructions", description="Follows the task instructions.", weight=1.0),
    ],
    aggregation="weighted_sum",
)

HELLO_WORLD_RUBRICS = RubricBundle(outcome=HELLO_WORLD_OUTCOME_RUBRIC, events=HELLO_WORLD_EVENT_RUBRIC)

app = create_local_api(
    LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description="A tiny OpenCode file-edit task for prompt-learning / GEPA smoke tests.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        rubrics=HELLO_WORLD_RUBRICS,
        cors_origins=["*"],
    )
)


if __name__ == "__main__":
    import uvicorn
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth

    port = int(os.getenv("PORT", "8030"))
    env_key = ensure_localapi_auth(backend_base="http://localhost:8000", synth_api_key=None)
    print(f"[hello_world_bench] ENVIRONMENT_API_KEY ready: {env_key[:15]}...")
    print(f"[hello_world_bench] Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

