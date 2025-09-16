import modal
from typing import Any, Optional
from dataclasses import dataclass, field
import os as _os
import sys as _sys
from pathlib import Path as _Path
import time

# Make local 'crafter' importable when running locally
_HERE = _Path(__file__).resolve()
_LOCAL_CRAFTER_PARENT = _HERE.parent.parent  # points to examples/rl
if str(_LOCAL_CRAFTER_PARENT) not in _sys.path:
    _sys.path.insert(0, str(_LOCAL_CRAFTER_PARENT))
if "/opt" not in _sys.path:
    _sys.path.insert(0, "/opt")

# Use a distinct secret name to avoid collisions with monorepo task apps
MODAL_SECRET_NAME = "crafter-environment-sdk"


# Use a distinct app name to avoid collisions with monorepo task apps
app = modal.App("grpo-task-service-sdk")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "fastapi",
            "uvicorn",
            "pydantic>=2",
            "httpx",
            "requests",
            "tqdm",
            "urllib3>=2.3.0",
            "jsonschema>=4.23.0",
            "typing_extensions>=4.0.0",
            "numpy",
            "pandas",
            "sqlalchemy",
            "aiosqlite",
            "asyncpg>=0.30.0",
            "crafter",
            "pillow",
            "imageio",
            "opensimplex",
            "ruamel.yaml",
            "networkx>=3.4.2",
            "redis>=6.2.0",
            "duckdb>=1.0.0",
            "ty>=0.0.1a5",
            "toml>=0.10.2",
            "libsql>=0.1.8",
            "python-dotenv",
            "anthropic",
            "openai",
            "diskcache",
            "backoff",
            "groq",
            "google-genai",
            "google-generativeai",
            "google-api-python-client",
            "google-api-core>=2.25.1",
            "google-auth",
            "google-auth-httplib2",
            "opentelemetry-api>=1.26.0,<1.27.0",
            "opentelemetry-sdk>=1.26.0,<1.27.0",
            "opentelemetry-exporter-otlp-proto-http>=1.26.0,<1.27.0",
            "wrapt",
            "langfuse>=2.53.9,<3.0.0",
            "together",
            "mistralai>=1.9.2",
            "click>=8.1.0",
            "textual>=1.1.0",
            "openai-harmony>=0.0.1",
            "aiohttp>=3.8.0",
            "datasets>=4.0.0",
            "gymnasium>=0.29.1",
            "minigrid>=2.3.1",
        ]
    )
    # Bundle the crafter module into the image for imports at runtime (absolute path)
    .add_local_dir(str((_HERE.parent / "crafter_task_app_helpers").resolve()), "/opt/crafter_task_app_helpers")
    # Bundle synth_ai package to import full environment implementation.
    # Resolve repo root robustly (examples/rl/task_app.py -> repo_root = examples/rl/../../..)
    .add_local_dir(str((_HERE.parent.parent.parent / "synth_ai").resolve()), "/opt/synth_ai")
)

# --- OpenAI payload sanitizer (local) ---
OPENAI_MAX_COMPLETION_TOKENS_MIN = 16000
OPENAI_REMOVE_FIELDS = (
    "stop_after_tool_calls",
    "thinking_mode",
    "thinking_budget",
    "reasoning",
)
OPENAI_REMOVE_SAMPLING_FIELDS = ("temperature", "top_p")
OPENAI_TOOL_CHOICE_FORCED = {"type": "function", "function": {"name": "interact"}}

def prepare_inference_payload_for_model(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize payload for OpenAI API.

    - Always strip Synth-specific fields not supported by OpenAI (e.g., stop_after_tool_calls).
    - For gpt-5 family: map max_tokens->max_completion_tokens, enforce tool_choice, disable parallel tools,
      and remove vendor-specific sampling fields.
    """
    out = dict(payload)
    # Always remove unsupported fields for OpenAI
    for k in OPENAI_REMOVE_FIELDS:
        if k in out:
            out.pop(k)

    # gpt-5 family specific adjustments
    if model and "gpt-5" in model:
        if "max_completion_tokens" not in out and "max_tokens" in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
        # Ensure we don't send both
        if "max_tokens" in out:
            out.pop("max_tokens")
        for k in OPENAI_REMOVE_SAMPLING_FIELDS:
            if k in out:
                out.pop(k)
        mct = out.get("max_completion_tokens")
        if not isinstance(mct, int) or mct < OPENAI_MAX_COMPLETION_TOKENS_MIN:
            out["max_completion_tokens"] = OPENAI_MAX_COMPLETION_TOKENS_MIN
        out["tool_choice"] = OPENAI_TOOL_CHOICE_FORCED
        out["parallel_tool_calls"] = False
    return out

@app.function(image=image, secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)], min_containers=1, max_containers=1)
@modal.asgi_app()
def fastapi_app():
    # Import FastAPI/Pydantic inside the container runtime to avoid local import errors
    from fastapi import FastAPI, Body, HTTPException, status, Header, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import logging
    import sys
    import os
    import httpx
    # Logger for debug output
    logger = logging.getLogger(__name__)

    # Preload synth_ai modules and vendor deps so missing packages surface early
    if "/opt/synth_ai" not in sys.path:
        sys.path.insert(0, "/opt/synth_ai")
    # Ensure tracing DB points to a writable location in the container
    os.environ.setdefault("TURSO_LOCAL_DB_URL", "sqlite+aiosqlite:////tmp/synth_ai.db")

    import importlib
    preload_modules = [
        # synth_ai core
        "synth_ai",
        "synth_ai.lm",
        "synth_ai.lm.core.main",
        "synth_ai.lm.core.main_v3",
        "synth_ai.lm.core.vendor_clients",
        "synth_ai.lm.core.all",
        # vendors
        "synth_ai.lm.vendors.core.anthropic_api",
        "synth_ai.lm.vendors.core.openai_api",
        "synth_ai.lm.vendors.openai_standard",
        "synth_ai.lm.vendors.core.gemini_api",
        # environments
        "synth_ai.environments",
        "synth_ai.environments.environment.rewards.core",
        "synth_ai.environments.examples.crafter_classic.environment",
        # tracing
        "synth_ai.tracing_v3.turso.models",
        "synth_ai.tracing_v3.turso.manager",
        # common 3p libs these modules rely on
        "anthropic",
        "openai",
        "groq",
        "google.genai",
        "google.generativeai",
        "googleapiclient.discovery",
        "google.auth",
        "google_auth_httplib2",
        "requests",
        "tqdm",
        "langfuse",
        "diskcache",
        "backoff",
        "together",
        "dotenv",
        "grpc",
    ]
    for mod in preload_modules:
        try:
            importlib.import_module(mod)
        except Exception as _e:
            print(f"[task:crafter] preload missing/err: {mod}: {_e}", flush=True)

    # Make packaged local crafter modules importable ahead of site-packages 'crafter'
    if "/opt/crafter_task_app_helpers" not in sys.path:
        sys.path.insert(0, "/opt/crafter_task_app_helpers")
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    if "/opt/synth_ai" not in sys.path:
        sys.path.insert(0, "/opt/synth_ai")
    from crafter_task_app_helpers.env import EnvRegistry
    from crafter_task_app_helpers.config import ACTION_SPACE, ENV_NAME
    from crafter_task_app_helpers.policy import CrafterPolicy

    _registry = EnvRegistry()

    # --- JSON sanitization for responses (convert numpy -> python primitives, arrays -> shapes) ---
    import numpy as _np

    def _to_jsonable(value):
        # Numpy types first: scalars vs arrays
        if isinstance(value, (_np.generic,)):
            return value.item()
        if isinstance(value, _np.ndarray):
            return f"<ndarray shape={tuple(value.shape)} dtype={str(value.dtype)}>"
        # Basic containers
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        # Sets to lists
        if isinstance(value, set):
            return [_to_jsonable(v) for v in value]
        return value

    class InitRequest(BaseModel):
        env_name: str | None = None
        env_config: dict[str, Any] | None = None

    class StepRequest(BaseModel):
        env_id: str
        action: str

    api = FastAPI(debug=True)

    @api.get("/health")
    def health(request: Request):
        env_key = os.environ.get("ENVIRONMENT_API_KEY")
        if not env_key:
            raise HTTPException(status_code=503, detail="Auth not configured: missing ENVIRONMENT_API_KEY in task service environment")
        header_key = request.headers.get("x-api-key")
        if header_key is not None and header_key != env_key:
            raise HTTPException(status_code=401, detail="Invalid API key for health check")
        return {"healthy": True}

    @api.post(f"/env/{ENV_NAME}/initialize")
    async def initialize(req: InitRequest):
        env_id, obs = await _registry.initialize(req.env_config)
        return {"env_id": env_id, "observation": _to_jsonable(obs)}

    @api.post(f"/env/{ENV_NAME}/step")
    async def step(req: StepRequest):
        obs, reward, done, info = await _registry.step(req.env_id, req.action)
        return {
            "observation": _to_jsonable(obs),
            "reward": float(reward) if isinstance(reward, (int, float)) else reward,
            "done": bool(done),
            "info": _to_jsonable(info) if info is not None else None,
        }

    @api.post(f"/env/{ENV_NAME}/terminate")
    async def terminate(req: dict[str, str] = Body(...)):
        env_id = str(req.get("env_id"))
        return await _registry.terminate(env_id)

    @api.get("/actions")
    def actions():
        return {"actions": ACTION_SPACE}

    # OpenAI proxy: forward chat/completions to OpenAI using env OPENAI_API_KEY
    @api.post("/proxy/v1/chat/completions")
    def proxy_chat_completions(req: dict[str, Any]):
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Missing OPENAI_API_KEY in task service environment")
        # Sanitize payload for OpenAI models (e.g., gpt-5-*)
        model = req.get("model")
        payload = prepare_inference_payload_for_model(model, req)
        headers = {"Authorization": f"Bearer {openai_key}"}
        # Increase timeout for proxy calls (models may be slower)
        with httpx.Client(timeout=120.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            try:
                data = resp.json()
            except Exception:
                data = {"error": "invalid_json", "raw": resp.text[:800]}
            if resp.status_code >= 400:
                return JSONResponse(status_code=resp.status_code, content=data)
            return data

    # Unified rollout schema imported from SDK task contracts
    from synth_ai.task.contracts import (
        RolloutEnvSpec,
        RolloutPolicySpec,
        RolloutRecordConfig,
        RolloutSafetyConfig,
        RolloutRequest,
        RolloutStep,
        RolloutTrajectory,
        RolloutMetrics,
        RolloutResponse,
    )

    @api.post("/rollout", response_model=RolloutResponse)
    async def rollout(req: RolloutRequest, request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        expected = os.environ.get("ENVIRONMENT_API_KEY")
        if not expected:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth not configured: missing ENVIRONMENT_API_KEY")
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

        # Extract policy config
        inference_url = req.policy.config["inference_url"]
        model = req.policy.config.get("model")
        max_steps = int(req.env.config.get("max_steps_per_episode", 10))
        policy = CrafterPolicy(inference_url=inference_url, model=model)

        # Debug: request summary
        print(
            "[task:crafter] ROLLOUT req: ",
            {
                "run_id": req.run_id,
                "env": req.env.env_name,
                "seed": req.env.seed,
                "ops": len(req.ops),
                "model": model,
                "inference_url": inference_url,
                "max_steps": max_steps,
            },
            flush=True,
        )

        # Initialize env
        cfg = dict(req.env.config or {})
        if req.env.seed is not None:
            cfg["seed"] = int(req.env.seed)
        env_id, observation = await _registry.initialize(cfg)

        trajectory_steps: list[RolloutStep] = []
        total_reward = 0.0
        ops_executed = 0
        pending_tool_calls: list[dict[str, Any]] | None = None
        try:
            for op in req.ops:
                if ops_executed >= req.safety.max_ops:
                    break
                if op == "agent":
                    # Format current observation for the prompt
                    # Cache for mapping semantic ids to names
                    _id_to_item_cache: list[str] | None = None

                    def _ensure_semantic_mapping() -> list[str] | None:
                        nonlocal _id_to_item_cache
                        if _id_to_item_cache is not None:
                            return _id_to_item_cache
                        # Build mapping using crafter's internal ids
                        import itertools as _it
                        import crafter as _crafter
                        dummy = None
                        try:
                            dummy = _crafter.Env()
                            max_id = (
                                max(max(dummy._world._mat_ids.values()), max(dummy._sem_view._obj_ids.values()))
                                + 1
                            )
                            id_to_item = ["void"] * max_id
                            for name, ind in _it.chain(
                                dummy._world._mat_ids.items(), dummy._sem_view._obj_ids.items()
                            ):
                                if name is None:
                                    clean = "none"
                                elif hasattr(name, "__name__"):
                                    clean = name.__name__
                                else:
                                    clean = str(name)
                                id_to_item[ind] = clean.lower()
                            _id_to_item_cache = id_to_item
                        finally:
                            if dummy is not None:
                                try:
                                    dummy.close()
                                except Exception:
                                    pass
                        return _id_to_item_cache

                    def _format_obs(obs: dict[str, Any]) -> str:
                        if not isinstance(obs, dict):
                            # Avoid dumping raw matrices; encourage exploration to gather context
                            return "no salient state; explore to gather context"
                        inv = obs.get("inventory") or {}
                        pos = obs.get("player_position")
                        steps = obs.get("num_steps_taken")
                        direction = obs.get("player_direction")
                        ach = obs.get("achievements_status") or {}
                        inv_lines = ", ".join(f"{k}:{v}" for k, v in inv.items() if v)
                        ach_on = [k for k, v in ach.items() if v]
                        lines = []
                        if pos is not None:
                            px, py = int(pos[0]), int(pos[1])
                            lines.append(f"position: (x={px}, y={py})")
                        if direction is not None:
                            dx, dy = int(direction[0]), int(direction[1])
                            dir_label = {
                                (1, 0): "→ east/right",
                                (-1, 0): "← west/left",
                                (0, 1): "↓ south/down",
                                (0, -1): "↑ north/up",
                                (0, 0): "• idle",
                            }.get((dx, dy), f"({dx},{dy})")
                            lines.append(f"direction: {dir_label}")
                        if steps is not None:
                            lines.append(f"steps: {int(steps)}")
                        if inv_lines:
                            lines.append(f"inventory: {inv_lines}")
                        if ach:
                            all_achievements = list(ach.keys())
                            lines.append(f"achievements_available: {', '.join(all_achievements)}")
                            if ach_on:
                                lines.append(f"achievements_unlocked: {', '.join(ach_on)}")
                                lines.append(f"achievements_progress: {len(ach_on)}/{len(all_achievements)}")
                        # Local surroundings (7x7) using semantic_map
                        smap = obs.get("semantic_map")
                        if smap is not None and pos is not None:
                            try:
                                px, py = int(pos[0]), int(pos[1])
                                view_size = 7
                                half = view_size // 2
                                id_to_item = _ensure_semantic_mapping() or []
                                grid_rows: list[str] = []
                                # Build matrix centered at player, then transpose for human-friendly view
                                matrix: list[list[str]] = []
                                for dy in range(-half, half + 1):
                                    row: list[str] = []
                                    for dx in range(-half, half + 1):
                                        x, y = px + dx, py + dy
                                        if not (0 <= x < smap.shape[0] and 0 <= y < smap.shape[1]):
                                            row.append("void")
                                        elif dx == 0 and dy == 0:
                                            row.append("player")
                                        else:
                                            idx = int(smap[x, y])
                                            name = id_to_item[idx] if 0 <= idx < len(id_to_item) else str(idx)
                                            row.append(name)
                                    matrix.append(row)
                                # Transpose to match visual orientation
                                transposed = list(zip(*matrix))
                                for row in transposed:
                                    grid_rows.append(" ".join(row))
                                if grid_rows:
                                    lines.append(f"Local Map View (7x7):\n" + "\n".join(grid_rows))
                            except Exception:
                                # If any issue occurs, skip map rendering without crashing
                                pass
                        if not lines:
                            lines.append("no salient state; explore to gather context")
                        return "\n".join(lines)
                    # Build compact context from last few tool calls (gpt-5-nano friendly)
                    lines: list[str] = []
                    for rec in reversed(trajectory_steps):
                        if len(lines) >= 3:
                            break
                        tcs = rec.tool_calls
                        if not tcs:
                            continue
                        tc0 = tcs[0] if isinstance(tcs, list) and tcs else None
                        if not isinstance(tc0, dict):
                            continue
                        name = tc0.get("tool_name") or tc0.get("name") or "unknown"
                        args = tc0.get("arguments")
                        lines.append(f"- {name}: {args}")
                    context_text = "Previous tool calls (most recent first):\n" + ("\n".join(lines) if lines else "- none")
                    obs_text = _format_obs(observation)
                    combined_text = f"Current observation:\n{obs_text}\n\n{context_text}"
                    payload = policy.build_inference_request(combined_text, history=[], turn=len(trajectory_steps))
                    # Prepare payload based on model family (OpenAI vs vLLM)
                    def _prepare_payload(p: dict, mdl: str | None) -> dict:
                        return prepare_inference_payload_for_model(mdl, p)
                    # Debug: payload shape
                    print(
                        "[task:crafter] inference payload: ",
                        {
                            "has_model": bool(payload.get("model") is not None),
                            "messages": len(payload.get("messages", [])),
                            "tools": isinstance(payload.get("tools"), list),
                            "tool_choice": payload.get("tool_choice"),
                            "stop_after_tool_calls": payload.get("stop_after_tool_calls"),
                        },
                        flush=True,
                    )
                    headers: dict[str, str] = {}
                    _okey = os.environ.get("OPENAI_API_KEY")
                    # Configure granular timeouts for slow model/tool runs
                    _timeouts = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=60.0)
                    with httpx.Client(timeout=_timeouts) as client:
                        # Decide endpoint: avoid calling our own /proxy inside the same request
                        _direct = ("api.openai.com" in inference_url) or inference_url.rstrip("/").endswith("/proxy")
                        if _direct:
                            # Call OpenAI directly
                            if _okey:
                                headers["Authorization"] = f"Bearer {_okey}"
                            to_send = _prepare_payload(payload, model)
                            endpoint_base = "https://api.openai.com"
                        else:
                            # Non-OpenAI inference endpoint
                            to_send = payload
                            endpoint_base = inference_url

                        # Debug: outbound request diagnostics
                        try:
                            import json as _json
                            _size = len(_json.dumps(to_send))
                        except Exception:
                            _size = -1
                        print(
                            "[task:crafter] inference dispatch:",
                            {
                                "endpoint": f"{endpoint_base.rstrip('/')}/v1/chat/completions",
                                "direct_openai": bool(_direct),
                                "timeout": {"read": 180.0, "connect": 10.0, "write": 60.0, "pool": 60.0},
                                "payload_bytes": _size,
                                "has_auth": bool(headers.get("Authorization")),
                            },
                            flush=True,
                        )

                        _t0 = time.time()
                        try:
                            resp = client.post(
                                f"{endpoint_base.rstrip('/')}/v1/chat/completions",
                                json=to_send,
                                headers=headers,
                            )
                        except httpx.ReadTimeout as rte:
                            _elapsed = time.time() - _t0
                            print(f"[task:crafter][timeout] read timeout after {_elapsed:.1f}s: {rte}", flush=True)
                            raise
                        except Exception as re:
                            _elapsed = time.time() - _t0
                            print(f"[task:crafter][error] request failed after {_elapsed:.1f}s: {type(re).__name__}: {re}", flush=True)
                            raise
                        _elapsed = time.time() - _t0
                        print(f"[task:crafter] inference status= {resp.status_code} elapsed={_elapsed:.2f}s", flush=True)
                        # Debug: response status and body (on errors)
                        print("[task:crafter] inference status=", resp.status_code, flush=True)
                        if resp.status_code >= 400:
                            body_preview = resp.text[:800]
                            print("[task:crafter] inference error body:", body_preview, flush=True)
                        data = resp.json()
                    print(f"[task:crafter] inference response: {data}")
                    parsed = CrafterPolicy.parse_response_to_tool_calls(data, use_tools=True) or []
                    # Debug: parsed tool call summary
                    print(
                        "[task:crafter] parsed tool_calls: ",
                        {
                            "n": len(parsed),
                            "first": (parsed[0] if isinstance(parsed, list) and parsed else None),
                        },
                        flush=True,
                    )
                    # Print full tool call payloads for inspection
                    try:
                        import json as _json
                        for _i, _tc in enumerate(parsed):
                            try:
                                print(
                                    f"[task:crafter] tool_call[{_i}]:",
                                    _json.dumps(_tc, separators=(",", ":")),
                                    flush=True,
                                )
                            except Exception:
                                print(f"[task:crafter] tool_call[{_i}]: {_tc}", flush=True)
                    except Exception:
                        pass
                    if not parsed:
                        # Dump compact body preview to understand schema when no tools parsed
                        try:
                            import json as _json
                            preview = _json.dumps(data, separators=(",",":"))
                            print("[task:crafter] body(no_tools) preview:", preview[:800], flush=True)
                        except Exception:
                            pass
                    pending_tool_calls = parsed
                    ops_executed += 1
                elif op == "env":
                    if not pending_tool_calls:
                        print("[task:crafter] no tool_calls; skipping env step", flush=True)
                        continue
                    info: dict[str, Any] | None = None
                    for tc in pending_tool_calls:
                        name = tc.get("tool_name")
                        if name == "interact":
                            # Parse the JSON arguments string
                            import json
                            args_str = tc.get("arguments", "{}")
                            try:
                                args_dict = json.loads(args_str)
                                actions = args_dict.get("actions", [])
                                reasoning = args_dict.get("reasoning", "")
                                print(f"[task:crafter] reasoning: {reasoning}", flush=True)
                            except (json.JSONDecodeError, TypeError):
                                print(f"[task:crafter] ERROR: Failed to parse arguments: {args_str}", flush=True)
                                actions = []
                                reasoning = "Parse error"

                            print(f"[task:crafter] env actions: {actions}", flush=True)

                            # Execute each action individually
                            for act in actions:
                                observation, reward, done, _info = await _registry.step(env_id, act)
                                total_reward += float(reward)
                                # Debug: print step outcome (compact)
                                try:
                                    ok = list(observation.keys()) if isinstance(observation, dict) else []
                                    print(f"[task:crafter] step => a={act} r={float(reward)} done={bool(done)} obs_keys={ok[:5]}", flush=True)
                                except Exception:
                                    pass
                                step = RolloutStep(obs=observation, tool_calls=pending_tool_calls, reward=float(reward), done=bool(done), truncated=False, info=info)
                                trajectory_steps.append(step)
                                ops_executed += 1

                                # Check for achievement-based termination
                                if isinstance(observation, dict):
                                    current_achievements = observation.get("achievements_status", {})
                                    achieved_count = sum(1 for v in current_achievements.values() if v)
                                    total_achievements = len(current_achievements)

                                    # Terminate if we've achieved a significant portion of available achievements
                                    if total_achievements > 0 and achieved_count >= max(3, total_achievements // 2):
                                        print(f"[task:crafter] achievement_termination: {achieved_count}/{total_achievements} achievements reached", flush=True)
                                        print(f"[task:crafter] achieved: {[k for k, v in current_achievements.items() if v]}", flush=True)
                                        break

                                if done or len(trajectory_steps) >= max_steps:
                                    print(f"[task:crafter] episode_end: done={bool(done)} steps={len(trajectory_steps)} total_reward={total_reward}", flush=True)
                                    break
                        elif name == "terminate":
                            # Handle termination
                            print("[task:crafter] Agent requested termination", flush=True)
                            break
                        else:
                            # Non-interact tool call: count as a step without env change
                            print("[task:crafter] non-interact tool_call:", name, flush=True)
                            step = RolloutStep(obs=observation, tool_calls=pending_tool_calls, reward=None, done=False, truncated=False, info=info)
                            trajectory_steps.append(step)
                            ops_executed += 1
                    pending_tool_calls = None
                    if len(trajectory_steps) >= max_steps:
                        print(f"[task:crafter] max_steps_reached: steps={len(trajectory_steps)} total_reward={total_reward}", flush=True)
                        break
                else:
                    # Unknown op: skip
                    continue
                if len(trajectory_steps) >= max_steps:
                    break
        finally:
            await _registry.terminate(env_id)

        # Sanitize steps for JSON
        safe_steps = [
            RolloutStep(
                obs=_to_jsonable(s.obs),
                tool_calls=s.tool_calls,
                reward=float(s.reward) if s.reward is not None else None,
                done=bool(s.done),
                truncated=bool(s.truncated) if s.truncated is not None else None,
                info=_to_jsonable(s.info) if s.info is not None else None,
            )
            for s in trajectory_steps
        ]

        trajectory = RolloutTrajectory(
            env_id=env_id,
            policy_id=req.policy.policy_name or "crafter-policy",
            steps=safe_steps,
            final={"observation": _to_jsonable(observation)},
            length=len(safe_steps),
        )
        # Calculate achievements for this episode
        final_obs = observation
        if isinstance(final_obs, dict):
            final_achievements = final_obs.get("achievements_status", {})
        else:
            # Handle numpy array case - no achievements available
            final_achievements = {}
        total_achievements = sum(1 for v in final_achievements.values() if v)

        metrics = RolloutMetrics(
            episode_returns=[total_reward],  # Keep original rewards for reference
            mean_return=float(total_achievements),  # Use achievements as mean_return
            num_steps=len(trajectory_steps),
            num_episodes=1,
        )
        # Debug: print reward and achievement metrics
        print(f"[task:crafter] Rollout metrics: total_reward={total_reward}, total_achievements={total_achievements}, mean_return={metrics.mean_return}, episode_returns={metrics.episode_returns}", flush=True)
        return RolloutResponse(
            run_id=req.run_id,
            trajectories=[trajectory],
            branches={},
            metrics=metrics,
            aborted=False,
            ops_executed=ops_executed,
        )

    @api.get("/test_auth")
    def test_auth(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        expected = os.environ.get("ENVIRONMENT_API_KEY")
        if not expected:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Missing ENVIRONMENT_API_KEY in service env")
        ok = bool(x_api_key) and (x_api_key == expected)
        if not ok:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
        return {"ok": True}

    return api

