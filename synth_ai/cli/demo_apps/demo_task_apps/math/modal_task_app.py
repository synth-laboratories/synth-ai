"""Modal task app for Hendrycks MATH single-step RL environment."""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from modal import App, Image, Secret, asgi_app
from starlette.requests import Request

try:  # Backward compatibility with older installed SDKs
    from synth_ai.cli.demo_apps.demo_task_apps.core import DEFAULT_TASK_APP_SECRET_NAME
except Exception:  # pragma: no cover - occurs on older deployments
    DEFAULT_TASK_APP_SECRET_NAME = "hendrycks-math-task-app-secret"

# Self-contained: no external problem bank installer required


_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent
_SYNTH_HOSTED = None
try:
    probe = _HERE
    for _ in range(8):
        candidate = (
            probe / "backend/app/routes/clustered_training/dev/synth_envs_hosted"
        ).resolve()
        if candidate.exists():
            _SYNTH_HOSTED = candidate
            break
        if probe.parent == probe:
            break
        probe = probe.parent
except Exception:
    _SYNTH_HOSTED = None

image = Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.110.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.6.0",
    "httpx>=0.24.0",
    "numpy>=1.24.0",
    "aiohttp>=3.8.0",
    "datasets>=2.16.0",
    "synth-ai",
)
if _SYNTH_HOSTED is not None:
    image = image.add_local_dir(str(_SYNTH_HOSTED), "/app/synth_envs_hosted")

# No extra local dirs required; app is self-contained


def _build_inline_secret() -> Secret:
    required = ("ENVIRONMENT_API_KEY",)
    optional = ("SYNTH_API_KEY", "OPENAI_API_KEY")
    payload: dict[str, str] = {}
    missing: list[str] = []

    for key in required:
        value = (os.environ.get(key) or "").strip()
        if not value:
            missing.append(key)
        else:
            payload[key] = value

    for key in optional:
        value = (os.environ.get(key) or "").strip()
        if value:
            payload[key] = value

    if missing:
        raise RuntimeError(
            "Missing required environment values for inline secret: " + ", ".join(missing)
        )

    previews = ", ".join(f"{k}:len={len(v)}" for k, v in payload.items())
    print(f"[startup] TASK_APP_SECRET_NAME={DEFAULT_TASK_APP_SECRET_NAME}")
    print(f"[startup] inline secret prepared ({previews})")

    # Modal.Secret.from_dict expects dict[str, Optional[str]]
    secrets_dict: dict[str, str | None] = dict(payload.items())
    return Secret.from_dict(secrets_dict)


INLINE_SECRET = _build_inline_secret()

app = App("hendrycks-math-task-app")


@app.function(
    image=image,
    timeout=600,
    memory=16384,
    cpu=4,
    min_containers=1,
    secrets=[INLINE_SECRET],
)
@asgi_app()
def fastapi_app():
    import httpx
    from fastapi import Body, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    try:
        from synth_ai.sdk.task.auth import (
            is_api_key_header_authorized,
            normalize_environment_api_key,
        )
    except Exception:  # pragma: no cover - fallback for older synth-ai builds

        def _normalize_env_key_fallback() -> str | None:
            key = os.getenv("ENVIRONMENT_API_KEY")
            if key:
                return key
            for alias in ("dev_environment_api_key", "DEV_ENVIRONMENT_API_KEY"):
                candidate = os.getenv(alias)
                if candidate:
                    os.environ["ENVIRONMENT_API_KEY"] = candidate
                    return candidate
            return None

        def normalize_environment_api_key() -> str | None:  # type: ignore[override]
            return _normalize_env_key_fallback()

        def _header_values(request: Request, header: str) -> Iterable[str]:
            raw = request.headers.get(header) or request.headers.get(header.lower())
            return [raw] if raw is not None else []

        def _split(values: Iterable[str]) -> list[str]:
            parts: list[str] = []
            for value in values:
                if not isinstance(value, str):
                    continue
                for chunk in value.split(","):
                    chunk = chunk.strip()
                    if chunk:
                        parts.append(chunk)
            return parts

        def is_api_key_header_authorized(request: Request) -> bool:  # type: ignore[override]
            expected = normalize_environment_api_key()
            if not expected:
                return False
            single = _header_values(request, "x-api-key")
            multi = _header_values(request, "x-api-keys")
            auth = _header_values(request, "authorization")
            bearer = []
            for token in auth:
                if isinstance(token, str) and token.lower().startswith("bearer "):
                    bearer.append(token.split(" ", 1)[1].strip())
            # Convert Iterable[str] to list for concatenation
            candidates = _split(list(single) + list(multi) + bearer)
            return any(candidate == expected for candidate in candidates)

    # Inline, self-contained FastAPI app (math-only)
    @lru_cache(maxsize=1)
    def _hf_split(subject: str, split: str, slice_spec: str | None = None):
        from datasets import load_dataset  # type: ignore

        s = split
        if slice_spec:
            s = f"{s}{slice_spec}"

        try:
            return load_dataset("nlile/hendrycks-MATH-benchmark", subject, split=s)
        except ValueError:
            base = load_dataset("nlile/hendrycks-MATH-benchmark", split=s)
            if subject and subject not in {"", "default"}:
                column_names = getattr(base, "column_names", None)
                if column_names is not None and "subject" in column_names:
                    base = base.filter(lambda ex: ex.get("subject") == subject)
                elif isinstance(base, list):
                    base = [ex for ex in base if ex.get("subject") == subject]
            return base

    def _normalize_answer_text(s: str) -> str:
        import re as _re

        return _re.sub(r"[^0-9A-Za-z.+\-/*=]", "", (s or "").strip()).lower()

    def _extract_boxed(s: str) -> str:
        import re as _re

        m = list(_re.finditer(r"\\boxed\{([^}]+)\}", s or ""))
        return m[-1].group(1) if m else ""

    def _load_hendrycks_problem(seed: int, subject: str | None = None) -> tuple[str, str]:
        subj = subject or os.getenv("HENDRYCKS_MATH_CONFIG", "default")
        ds = _hf_split(
            subj, os.getenv("HENDRYCKS_MATH_SPLIT", "test"), os.getenv("HENDRYCKS_MATH_SLICE")
        )
        n = len(ds) if hasattr(ds, "__len__") else 0
        if n == 0 and subject not in {"", "default"}:
            ds = _hf_split(
                "default",
                os.getenv("HENDRYCKS_MATH_SPLIT", "test"),
                os.getenv("HENDRYCKS_MATH_SLICE"),
            )
            n = len(ds) if hasattr(ds, "__len__") else 0
        if n == 0:
            raise RuntimeError("Hendrycks MATH dataset loaded empty")
        idx = abs(int(seed)) % n
        ex = ds[int(idx)]
        q = ex.get("problem") or ex.get("question") or ex.get("prompt")
        a = ex.get("solution") or ex.get("answer") or ""
        if not q:
            raise RuntimeError("Hendrycks item missing problem text")
        return str(q), str(a)

    def create_app() -> FastAPI:
        app = FastAPI(title="Hendrycks Math Task App", version="0.1.0")
        app.add_middleware(  # type: ignore[misc]
            CORSMiddleware,  # type: ignore[arg-type]
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        import logging

        logger = logging.getLogger("hendrycks_math_task_app")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

        def _log_env_key_prefix(source: str, env_key: str | None) -> str | None:
            if not env_key:
                return None
            half = max(1, len(env_key) // 2)
            prefix = env_key[:half]
            msg = f"[{source}] expected ENVIRONMENT_API_KEY prefix: {prefix}"
            print(msg)
            logger.info(msg)
            return prefix

        def _resolve_env_keys() -> set[str]:
            keys: set[str] = set()
            for alias in (
                "ENVIRONMENT_API_KEY",
                "dev_environment_api_key",
                "DEV_ENVIRONMENT_API_KEY",
            ):
                value = os.environ.get(alias)
                if value:
                    os.environ.setdefault("ENVIRONMENT_API_KEY", value)
                    keys.add(value)
            alias_env = os.environ.get("ENVIRONMENT_API_KEY_ALIASES", "")
            for chunk in alias_env.split(","):
                trimmed = chunk.strip()
                if trimmed:
                    keys.add(trimmed)
            return keys

        def _extract_header_candidates(
            request: Request,
            x_api_key: str | None,
            x_api_keys: str | None,
            authorization: str | None,
        ) -> list[str]:
            headers = request.headers
            candidates: list[str] = []
            primary = x_api_key or headers.get("x-api-key")
            if primary:
                candidates.append(primary.strip())
            secondary = x_api_keys or headers.get("x-api-keys")
            if secondary:
                candidates.extend(
                    [value.strip() for value in secondary.split(",") if value.strip()]
                )
            auth_header = (
                authorization or headers.get("authorization") or headers.get("Authorization")
            )
            if auth_header and auth_header.lower().startswith("bearer "):
                token = auth_header.split(" ", 1)[1].strip()
                if token:
                    candidates.append(token)
            return [c for c in candidates if c]

        def _is_authorized(
            request: Request,
            x_api_key: str | None,
            x_api_keys: str | None,
            authorization: str | None,
        ) -> bool:
            keys = _resolve_env_keys()
            if not keys:
                return False
            candidates = _extract_header_candidates(request, x_api_key, x_api_keys, authorization)
            return any(candidate in keys for candidate in candidates)

        @app.get("/info")
        async def info():
            return {
                "service": {"base_url": os.getenv("SERVICE_BASE_URL", "")},
                "inference": {
                    "base_url": "",
                    "endpoints": {"chat_completions": "/v1/chat/completions"},
                },
            }

        @app.get("/health")
        async def health(request: Request):
            env_keys = _resolve_env_keys()
            env_key = next(iter(env_keys), None)
            if not env_key:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"},
                )
            # Authorize using all header variants; avoid typed Header params to prevent 422s
            authorized = is_api_key_header_authorized(request)
            if not authorized:
                prefix = _log_env_key_prefix("health", env_key)
                content = {
                    "status": "healthy",
                    "authorized": False,
                }
                if prefix:
                    content["expected_api_key_prefix"] = prefix
                return JSONResponse(status_code=200, content=content)
            return {"status": "healthy", "authorized": True}

        # Optional rollout-specific health for CLI compatibility
        @app.get("/health/rollout")
        async def health_rollout(request: Request):
            env_keys = _resolve_env_keys()
            env_key = next(iter(env_keys), None)
            if not env_key:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"},
                )
            authorized = is_api_key_header_authorized(request)
            if not authorized:
                prefix = _log_env_key_prefix("health/rollout", env_key)
                content = {
                    "status": "healthy",
                    "authorized": False,
                }
                if prefix:
                    content["expected_api_key_prefix"] = prefix
                return JSONResponse(status_code=200, content=content)
            return {"ok": True, "authorized": True}

        # _load_hendrycks_problem is defined at fastapi_app scope

        @app.get("/task_info")
        async def task_info(seed: int = 0, subject: str = "default"):
            """Return Hendrycks MATH problem/answer and tool schema for a seed."""
            q, a = _load_hendrycks_problem(int(seed), subject=subject)
            tools = [
                {
                    "name": "submit_answer",
                    "description": "Provide the final numerical or algebraic answer for the current math problem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The proposed final answer",
                            },
                        },
                        "required": ["answer"],
                    },
                }
            ]
            return {
                "seed": int(seed),
                "subject": subject,
                "system": "",
                "user": q,
                "tools": tools,
                "policy": {"name": "math-react"},
                "answer": a,
            }

        return app

    api = create_app()

    # Always log and surface 422 validation errors with header presence snapshot
    from fastapi.exceptions import RequestValidationError

    @api.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError):
        try:
            hdr = request.headers
            snapshot = {
                "path": str(request.url.path),
                "have_x_api_key": bool(hdr.get("x-api-key")),
                "have_x_api_keys": bool(hdr.get("x-api-keys")),
                "have_authorization": bool(hdr.get("authorization")),
                "errors": exc.errors()[:5],
            }
            print("[422] validation", snapshot, flush=True)
        except Exception:
            pass
        return JSONResponse(
            status_code=422, content={"status": "invalid", "detail": exc.errors()[:5]}
        )

    @api.get("/")
    async def root_probe():
        return {"status": "ok", "service": "math"}

    @api.head("/")
    async def head_probe():
        return {"status": "ok"}

    env_key = (
        os.environ.get("ENVIRONMENT_API_KEY")
        or os.environ.get("DEV_ENVIRONMENT_API_KEY")
        or os.environ.get("DEV_ENVIRONMENT_API_KEY")
    )
    if not env_key:
        raise RuntimeError("ENVIRONMENT_API_KEY missing in task app environment")

    openai_remove_fields = (
        "stop_after_tool_calls",
        "thinking_mode",
        "thinking_budget",
        "reasoning",
    )
    openai_remove_sampling_fields = ("temperature", "top_p")
    tool_choice_force = {"type": "function", "function": {"name": "submit_answer"}}

    def _prepare_openai_payload(model: str | None, payload: dict[str, object]) -> dict[str, object]:
        sanitized = dict(payload)
        for key in openai_remove_fields:
            sanitized.pop(key, None)
        if model and "gpt-5" in model:
            if "max_tokens" in sanitized and "max_completion_tokens" not in sanitized:
                sanitized["max_completion_tokens"] = sanitized.pop("max_tokens")
            else:
                sanitized.pop("max_tokens", None)
            for field in openai_remove_sampling_fields:
                sanitized.pop(field, None)
                sanitized["tool_choice"] = tool_choice_force
                sanitized["parallel_tool_calls"] = False
            return sanitized
        return sanitized

    @api.post("/proxy/v1/chat/completions")
    def proxy_chat_completions(request: dict[str, object] = Body(...)):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OPENAI_API_KEY missing"
            )
        model = request.get("model") if isinstance(request, dict) else None
        payload = _prepare_openai_payload(
            model if isinstance(model, str) else None, request if isinstance(request, dict) else {}
        )
        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions", json=payload, headers=headers
            )
            try:
                data = resp.json()
            except Exception:
                data = {"error": "invalid_json", "raw": resp.text[:400]}
            if resp.status_code >= 400:
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=resp.status_code, content=data)
            return data

    # Minimal math rollout endpoint: alternates agent/env; calls inference_url chat/completions
    @api.post("/rollout")
    def rollout(request: dict[str, object] = Body(...)):
        import json as _json
        from typing import Any

        run_id = str(request.get("run_id"))
        data = request if isinstance(request, dict) else {}
        env = data.get("env") if isinstance(data, dict) else {}
        policy = data.get("policy") if isinstance(data, dict) else {}
        ops = data.get("ops") if isinstance(data, dict) else []
        if not isinstance(ops, list):
            ops = []
        env_name = (env or {}).get("env_name") or "math"  # type: ignore[misc]
        policy_cfg = (policy or {}).get("config") or {}  # type: ignore[misc]
        model = policy_cfg.get("model")  # type: ignore[misc]
        inference_url = (policy_cfg.get("inference_url") or "").rstrip("/")  # type: ignore[misc]

        # ALWAYS derive question/answer from Hendrycks dataset using seed/subject
        env_cfg = (env or {}).get("config") or {}  # type: ignore[misc]
        # Prefer env.seed; fall back to env.config.seed -> default 0
        try:
            env_dict: dict[str, Any] = env if isinstance(env, dict) else {}  # type: ignore[assignment]
            seed_val_raw = env_dict.get("seed")
            seed_val = int(seed_val_raw) if seed_val_raw is not None else 0
        except Exception:
            seed_val = 0
        if seed_val == 0:
            try:
                raw_seed = env_cfg.get("seed") if isinstance(env_cfg, dict) else None
                seed_val = int(raw_seed) if raw_seed is not None else 0
            except Exception:
                seed_val = 0
        subject = (env_cfg.get("subject") if isinstance(env_cfg, dict) else None) or os.getenv(
            "HENDRYCKS_MATH_CONFIG", "default"
        )
        # Load real Hendrycks problem text/solution (download if necessary). Crash on failure.
        qh, ah = _load_hendrycks_problem(seed_val, subject=subject)
        question = qh
        expected_answer = ah

        def _prepare_payload(m: str | None, payload: dict[str, Any]) -> dict[str, Any]:
            # Remove vendor-specific fields and force tool choice for math interaction
            sanitized = dict(payload)
            for k in ("stop_after_tool_calls", "thinking_mode", "thinking_budget", "reasoning"):
                sanitized.pop(k, None)
            if m and "gpt-5" in m:
                if "max_tokens" in sanitized and "max_completion_tokens" not in sanitized:
                    sanitized["max_completion_tokens"] = sanitized.pop("max_tokens")
                else:
                    sanitized.pop("max_tokens", None)
                for field in ("temperature", "top_p"):
                    sanitized.pop(field, None)
                sanitized["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "submit_answer"},
                }
                sanitized["parallel_tool_calls"] = False
            return sanitized

        def _parse_tool_answer(resp: dict[str, Any]) -> str:
            try:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    tcs = msg.get("tool_calls")
                    if isinstance(tcs, list) and tcs:
                        fn = tcs[0].get("function", {}) if isinstance(tcs[0], dict) else {}
                        args = fn.get("arguments")
                        obj: dict[str, Any] = {}
                        if isinstance(args, str):
                            try:
                                obj = _json.loads(args)
                            except Exception:
                                obj = {}
                        elif isinstance(args, dict):
                            obj = args
                        ans = obj.get("answer")
                        if isinstance(ans, str):
                            return ans.strip()
            except Exception:
                pass
            return ""

        # Single-step rollout: one agent call followed by evaluation of the returned tool answer
        history: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        total_reward = 0.0

        user_prompt = (
            str(question)
            if isinstance(question, str | int | float) and str(question).strip()
            else "Solve the problem. Provide answer steps succinctly."
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_answer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                            },
                            "required": ["answer"],
                        },
                    },
                }
            ],
            "max_tokens": 256,
            "temperature": 0.2,
        }
        to_send = _prepare_payload(model if isinstance(model, str) else None, payload)

        try:
            tool_names = []
            tools = payload.get("tools")
            if not isinstance(tools, list):
                tools = []
            for t in tools:
                if isinstance(t, dict):
                    fn = (t.get("function") or {}) if isinstance(t.get("function"), dict) else {}  # type: ignore[misc]
                    name = fn.get("name")  # type: ignore[misc]
                    if isinstance(name, str):
                        tool_names.append(name)
            print("[math] system: <none>", flush=True)
            print(f"[math] user: {user_prompt}", flush=True)
            print(f"[math] tools: {tool_names}", flush=True)
        except Exception:
            pass

        headers = {}
        if "/proxy" in inference_url:
            sk = os.environ.get("SYNTH_API_KEY")
            if sk:
                headers["Authorization"] = f"Bearer {sk}"
        with httpx.Client(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
            resp = client.post(
                f"{inference_url}/v1/chat/completions", json=to_send, headers=headers
            )
            try:
                data = resp.json()
            except Exception:
                data = {"error": "invalid_json", "raw": resp.text[:400]}

        llm_text = None
        try:
            _choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(_choices, list) and _choices:
                _msg = _choices[0].get("message", {}) if isinstance(_choices[0], dict) else {}
                if isinstance(_msg, dict):
                    _content = _msg.get("content")
                    if isinstance(_content, str) and _content.strip():
                        llm_text = _content
        except Exception:
            llm_text = None

        try:
            if question is not None:
                print(f"[math] question: {question}", flush=True)
            if llm_text is not None:
                print(f"[math] llm: {llm_text}", flush=True)
            if expected_answer is not None and llm_text is not None:
                exp = str(expected_answer).strip()
                got = llm_text.strip()
                is_correct = exp and (exp in got)
                print(f"[math] correct: {bool(is_correct)} (expected fragment: {exp})", flush=True)
        except Exception:
            pass

        tool_answer = _parse_tool_answer(data)
        history.append({"answer": tool_answer})
        steps.append(
            {
                "obs": {},
                "tool_calls": [
                    {
                        "tool_name": "submit_answer",
                        "arguments": _json.dumps({"answer": tool_answer}),
                    }
                ],
                "reward": None,
                "done": False,
                "truncated": False,
                "info": None,
            }
        )

        # Evaluate answer correctness using tool output (or fall back to assistant text)
        reward_val = 0.0
        candidate = tool_answer or ""
        try:
            if not candidate and llm_text is not None:
                candidate = _extract_boxed(llm_text) or llm_text
            if expected_answer is not None:
                exp_raw = _extract_boxed(str(expected_answer)) or str(expected_answer)
                got_raw = candidate
                exp_n = _normalize_answer_text(exp_raw)
                got_n = _normalize_answer_text(got_raw)
                if exp_n and exp_n in got_n:
                    reward_val = 1.0
        except Exception:
            reward_val = 0.0

        # Immediate, concise rollout logging mirroring RL format
        try:
            preview = tool_answer[:120] + (
                "…" if isinstance(tool_answer, str) and len(tool_answer) > 120 else ""
            )
            components = {
                "env": float(reward_val),
                "rubric_event": 1.0 if bool(tool_answer.strip()) else 0.0,
                "rubric_outcome": 1.0 if float(reward_val) > 0.0 else 0.0,
            }
            print(
                "[MATH_ROLLOUT] run=",
                run_id,
                " seed=",
                seed_val,
                " subject=",
                subject,
                " tool=submit_answer answer=",
                preview,
                " reward=",
                float(reward_val),
                " components=",
                components,
                flush=True,
            )
        except Exception:
            pass

        total_reward += float(reward_val)
        steps.append(
            {
                "obs": {},
                "tool_calls": [],
                "reward": reward_val,
                "done": True,
                "truncated": False,
                "info": None,
            }
        )

        return {
            "run_id": run_id,
            "trajectories": [
                {
                    "env_id": env_name,
                    "policy_id": (policy or {}).get("policy_name") or "math-react",  # type: ignore[misc]
                    "steps": steps,
                    "final": {"observation": {}},
                    "length": len(steps),
                }
            ],
            "branches": {},
            "metrics": {
                "episode_returns": [total_reward],
                "mean_return": float(total_reward),
                "num_steps": len(steps),
                "num_episodes": 1,
            },
            "aborted": False,
            "ops_executed": len(steps),
        }

    return api
