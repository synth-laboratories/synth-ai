from __future__ import annotations

"""Modal task app for Hendrycks MATH single-step RL environment."""

import os
from pathlib import Path

from modal import App, Image, Secret, asgi_app
from functools import lru_cache

# Self-contained: no external problem bank installer required


_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent
_SYNTH_HOSTED = None
try:
    probe = _HERE
    for _ in range(8):
        candidate = (probe / "backend/app/routes/clustered_training/dev/synth_envs_hosted").resolve()
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

app = App("hendrycks-math-task-app")
_SECRET_NAME = os.getenv("MATH_TASK_APP_SECRET", "crafter-environment-sdk")


@app.function(
    image=image,
    timeout=600,
    memory=16384,
    cpu=4,
    min_containers=1,
    secrets=[Secret.from_name(_SECRET_NAME)],
)
@asgi_app()
def fastapi_app():
    import httpx
    from fastapi import Body, HTTPException, status
    from fastapi import FastAPI, Request, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    # Inline, self-contained FastAPI app (math-only)
    @lru_cache(maxsize=1)
    def _hf_split(subject: str, split: str, slice_spec: str | None = None):
        from datasets import load_dataset  # type: ignore
        s = split
        if slice_spec:
            s = f"{s}{slice_spec}"
        return load_dataset("nlile/hendrycks-MATH-benchmark", subject, split=s)

    def _normalize_answer_text(s: str) -> str:
        import re as _re
        return _re.sub(r"[^0-9A-Za-z.+\-/*=]", "", (s or "").strip()).lower()

    def _extract_boxed(s: str) -> str:
        import re as _re
        m = list(_re.finditer(r"\\boxed\{([^}]+)\}", s or ""))
        return m[-1].group(1) if m else ""

    def _load_hendrycks_problem(seed: int, subject: str | None = None) -> tuple[str, str]:
        subj = subject or os.getenv("HENDRYCKS_MATH_CONFIG", "default")
        ds = _hf_split(subj, os.getenv("HENDRYCKS_MATH_SPLIT", "test"), os.getenv("HENDRYCKS_MATH_SLICE"))
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
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/info")
        async def info():
            return {
                "service": {"base_url": os.getenv("SERVICE_BASE_URL", "")},
                "inference": {"base_url": "", "endpoints": {"chat_completions": "/v1/chat/completions"}},
            }

        @app.get("/health")
        async def health(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
            env_key = os.environ.get("ENVIRONMENT_API_KEY")
            if not env_key:
                return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"})
            if x_api_key is not None and x_api_key != env_key:
                return JSONResponse(status_code=401, content={"status": "unauthorized", "detail": "Invalid API key"})
            return {"status": "healthy"}

        # Optional rollout-specific health for CLI compatibility
        @app.get("/health/rollout")
        async def health_rollout(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
            env_key = os.environ.get("ENVIRONMENT_API_KEY")
            if not env_key:
                return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"})
            if not x_api_key or x_api_key != env_key:
                return JSONResponse(status_code=401, content={"status": "unauthorized", "detail": "Invalid or missing API key"})
            return {"ok": True}

        # _load_hendrycks_problem is defined at fastapi_app scope

        @app.get("/task_info")
        async def task_info(seed: int = 0, subject: str = "algebra"):
            """Return Hendrycks MATH problem/answer and tool schema for a seed."""
            q, a = _load_hendrycks_problem(int(seed), subject=subject)
            tools = [{
                "name": "interact",
                "description": "Submit one or more actions to the math environment.",
                "parameters": {
                    "type": "object",
                    "properties": {"actions": {"type": "array", "items": {"type": "string"}}},
                    "required": ["actions"],
                },
            }]
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

    @api.get("/")
    async def root_probe():
        return {"status": "ok", "service": "math"}

    @api.head("/")
    async def head_probe():
        return {"status": "ok"}

    env_key = (
        os.environ.get("ENVIRONMENT_API_KEY")
        or os.environ.get("DEV_ENVIRONMENT_API_KEY")
        or os.environ.get("dev_environment_api_key")
    )
    if not env_key:
        raise RuntimeError("ENVIRONMENT_API_KEY missing in task app environment")

    OPENAI_REMOVE_FIELDS = ("stop_after_tool_calls", "thinking_mode", "thinking_budget", "reasoning")
    OPENAI_REMOVE_SAMPLING_FIELDS = ("temperature", "top_p")
    TOOL_CHOICE_FORCE = {"type": "function", "function": {"name": "interact_many"}}

    def _prepare_openai_payload(model: str | None, payload: dict[str, object]) -> dict[str, object]:
        sanitized = dict(payload)
        for key in OPENAI_REMOVE_FIELDS:
            sanitized.pop(key, None)
        if model and "gpt-5" in model:
            if "max_tokens" in sanitized and "max_completion_tokens" not in sanitized:
                sanitized["max_completion_tokens"] = sanitized.pop("max_tokens")
            else:
                sanitized.pop("max_tokens", None)
            for field in OPENAI_REMOVE_SAMPLING_FIELDS:
                sanitized.pop(field, None)
            sanitized["tool_choice"] = TOOL_CHOICE_FORCE
            sanitized["parallel_tool_calls"] = False
        return sanitized

    @api.post("/proxy/v1/chat/completions")
    def proxy_chat_completions(request: dict[str, object] = Body(...)):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OPENAI_API_KEY missing")
        model = request.get("model") if isinstance(request, dict) else None
        payload = _prepare_openai_payload(model if isinstance(model, str) else None, request if isinstance(request, dict) else {})
        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
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
        from typing import Any
        import json as _json

        run_id = str(request.get("run_id"))
        data = request if isinstance(request, dict) else {}
        env = data.get("env") if isinstance(data, dict) else {}
        policy = data.get("policy") if isinstance(data, dict) else {}
        ops = data.get("ops") if isinstance(data, dict) else []
        if not isinstance(ops, list):
            ops = []
        env_name = (env or {}).get("env_name") or "math"
        policy_cfg = (policy or {}).get("config") or {}
        model = policy_cfg.get("model")
        inference_url = (policy_cfg.get("inference_url") or "").rstrip("/")

        # ALWAYS derive question/answer from Hendrycks dataset using seed/subject
        env_cfg = (env or {}).get("config") or {}
        # Prefer env.seed; fall back to env.config.seed -> default 0
        try:
            seed_val = int((env or {}).get("seed")) if isinstance(env, dict) and (env or {}).get("seed") is not None else 0
        except Exception:
            seed_val = 0
        if seed_val == 0:
            try:
                seed_val = int(env_cfg.get("seed")) if isinstance(env_cfg, dict) and env_cfg.get("seed") is not None else 0
            except Exception:
                seed_val = 0
        subject = (env_cfg.get("subject") if isinstance(env_cfg, dict) else None) or os.getenv("HENDRYCKS_MATH_CONFIG", "default")
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
                sanitized["tool_choice"] = {"type": "function", "function": {"name": "interact"}}
                sanitized["parallel_tool_calls"] = False
            return sanitized

        def _parse_tool_actions(resp: dict[str, Any]) -> list[str]:
            try:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    tcs = msg.get("tool_calls")
                    if isinstance(tcs, list) and tcs:
                        fn = tcs[0].get("function", {}) if isinstance(tcs[0], dict) else {}
                        args = fn.get("arguments")
                        obj = {}
                        if isinstance(args, str):
                            try:
                                obj = _json.loads(args)
                            except Exception:
                                obj = {}
                        elif isinstance(args, dict):
                            obj = args
                        acts = obj.get("actions")
                        if isinstance(acts, list):
                            return [str(a) for a in acts][:5]
            except Exception:
                pass
            return []

        # Build minimal context and execute ops
        history: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        total_reward = 0.0
        last_llm_text: str | None = None
        last_actions: list[str] = []
        for op in ops or []:
            if op == "agent":
                user_prompt = (
                    str(question)
                    if isinstance(question, (str, int, float)) and str(question).strip()
                    else "Solve the problem. Provide answer steps succinctly."
                )
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": "interact", "parameters": {"type": "object", "properties": {"actions": {"type": "array", "items": {"type": "string"}}}, "required": ["actions"]}},
                    }],
                    "max_tokens": 256,
                    "temperature": 0.2,
                }
                to_send = _prepare_payload(model if isinstance(model, str) else None, payload)
                # Print prompts and tools exposed to the model
                try:
                    tool_names = []
                    for t in (payload.get("tools") or []):
                        if isinstance(t, dict):
                            fn = (t.get("function") or {}) if isinstance(t.get("function"), dict) else {}
                            name = fn.get("name")
                            if isinstance(name, str):
                                tool_names.append(name)
                    print(f"[math] system: <none>", flush=True)
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
                    resp = client.post(f"{inference_url}/v1/chat/completions", json=to_send, headers=headers)
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"error": "invalid_json", "raw": resp.text[:400]}

                # Extract assistant text for visibility/correctness
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

                # Print question, model output, and correctness if we have an expected answer
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
                last_llm_text = llm_text
                acts = _parse_tool_actions(data) or []
                last_actions = acts if isinstance(acts, list) else []
                steps.append({"obs": {}, "tool_calls": [{"tool_name": "interact", "arguments": _json.dumps({"actions": acts})}], "reward": None, "done": False, "truncated": False, "info": None})
                history.append({"actions": acts})
            elif op == "env":
                # Compute a simple correctness-based reward if expected answer available
                reward_val = 0.0
                try:
                    if expected_answer is not None:
                        # Prefer explicit tool-call answer from last_actions
                        candidate = ""
                        if isinstance(last_actions, list) and last_actions:
                            # Take the last non-empty action as the final answer
                            for s in reversed(last_actions):
                                if isinstance(s, str) and s.strip():
                                    candidate = s.strip()
                                    break
                        # Fallback to parse from llm_text if tool actions absent
                        if not candidate and last_llm_text is not None:
                            candidate = _extract_boxed(last_llm_text) or last_llm_text
                        exp_raw = _extract_boxed(str(expected_answer)) or str(expected_answer)
                        got_raw = candidate
                        exp_n = _normalize_answer_text(exp_raw)
                        got_n = _normalize_answer_text(got_raw)
                        if exp_n and exp_n in got_n:
                            reward_val = 1.0
                except Exception:
                    reward_val = 0.0
                steps.append({"obs": {}, "tool_calls": [], "reward": reward_val, "done": False, "truncated": False, "info": None})
                total_reward += float(reward_val)
            else:
                continue

        # Compose response similar to SDK contract (simplified)
        return {
            "run_id": run_id,
            "trajectories": [{"env_id": env_name, "policy_id": (policy or {}).get("policy_name") or "math-react", "steps": steps, "final": {"observation": {}}, "length": len(steps)}],
            "branches": {},
            "metrics": {"episode_returns": [total_reward], "mean_return": float(total_reward), "num_steps": len(steps), "num_episodes": 1},
            "aborted": False,
            "ops_executed": len(steps),
        }

    return api


