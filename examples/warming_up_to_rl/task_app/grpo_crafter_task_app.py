"""Modal app for Crafter task service with OpenAI proxy.

App name: grpo-crafter-task-app_warming_up_ex

Provides:
- Hosted env/policy/rollout endpoints from synth_envs_hosted.create_app (Crafter only)
- GET /health (inherited)
- POST /proxy/v1/chat/completions (for direct OpenAI usage)

Secrets expected in Modal secret bundle:
- ENVIRONMENT_API_KEY (required) or dev_environment_api_key fallback
- OPENAI_API_KEY (required for proxy endpoint)
- SYNTH_API_KEY (optional; for backend-mediated flows)

To run locally for testing:
    python grpo_crafter_task_app.py --local
"""

from __future__ import annotations

from modal import App, Image, asgi_app, Secret
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()
# Use local copy of synth_envs_hosted within this task_app folder
TASK_SRC = (BASE_DIR / "./synth_envs_hosted").resolve()

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core server
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        # Crafter deps come via synth-ai; keep minimal here
        "numpy>=1.24.0",
        "aiohttp>=3.8.0",
        # Hosted env/policy/rollout
        "synth-ai==0.2.4.dev6",
        # Proxy deps
        "httpx>=0.24.0",
    )
    .add_local_dir(str(TASK_SRC), "/app/synth_envs_hosted")
)


app = App("grpo-crafter-task-app-final_warming_up_ex")


@app.function(
    image=image,
    timeout=600,
    memory=16384,
    cpu=4,
    min_containers=1,
    max_containers=10,
    secrets=[
        Secret.from_name("crafter-environment-sdk"),
        Secret.from_name("groq-api-key"),
        Secret.from_name("openai-api-key"),
    ],
)
@asgi_app()
def fastapi_app():
    import os
    import sys
    import httpx
    from typing import Any
    from fastapi import FastAPI, HTTPException, status, Body

    # Ensure imports resolve
    sys.path.insert(0, "/app")

    # Normalize ENVIRONMENT_API_KEY from fallback
    env_key = os.environ.get("ENVIRONMENT_API_KEY") or os.environ.get(
        "dev_environment_api_key"
    )
    if env_key:
        os.environ["ENVIRONMENT_API_KEY"] = env_key
        print(
            f"[task:crafter] ENVIRONMENT_API_KEY present (prefix={env_key[:6] + '…' if len(env_key)>=6 else 'set'})",
            flush=True,
        )
    else:
        raise RuntimeError(
            "Auth not configured: missing ENVIRONMENT_API_KEY in task service environment"
        )

    # Normalize OPENAI_API_KEY and GROQ_API_KEY from common local fallbacks to simplify .env usage
    try:
        oa = os.environ.get("OPENAI_API_KEY") or os.environ.get("dev_openai_api_key")
        if oa:
            os.environ["OPENAI_API_KEY"] = oa
            print(f"[task:crafter] OPENAI_API_KEY present (prefix={oa[:6] + '…' if len(oa)>=6 else 'set'})", flush=True)
        gr = os.environ.get("GROQ_API_KEY") or os.environ.get("dev_groq_api_key")
        if gr:
            os.environ["GROQ_API_KEY"] = gr
            print(f"[task:crafter] GROQ_API_KEY present (prefix={gr[:6] + '…' if len(gr)>=6 else 'set'})", flush=True)
    except Exception:
        pass

    # Construct hosted service (Crafter-only)
    from synth_envs_hosted.hosted_app import create_app as _create_app

    api = _create_app(allowed_environments=["crafter"])

    # --- OpenAI proxy (optional convenience) ---
    OPENAI_REMOVE_FIELDS = (
        "stop_after_tool_calls",
        "thinking_mode",
        "thinking_budget",
        "reasoning",
    )
    OPENAI_REMOVE_SAMPLING_FIELDS = ("temperature", "top_p")
    # Match eval_rollout_table_groq.py: single function tool named "interact"
    OPENAI_TOOL_CHOICE_FORCED = {"type": "function", "function": {"name": "interact"}}
    OPENAI_MAX_COMPLETION_TOKENS_MIN = 16000

    def _interact_tool_schema() -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Perform actions in the Crafter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of actions to perform in sequence (2-5)",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reasoning for these actions",
                        },
                    },
                    "required": ["actions", "reasoning"],
                }
            }
        }]

    def _prepare_for_openai(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        if model and "gpt-5" in model:
            out = dict(payload)
            for k in OPENAI_REMOVE_FIELDS:
                out.pop(k, None)
            if "max_completion_tokens" not in out and "max_tokens" in out:
                out["max_completion_tokens"] = out.pop("max_tokens")
            out.pop("max_tokens", None)
            for k in OPENAI_REMOVE_SAMPLING_FIELDS:
                out.pop(k, None)
            mct = out.get("max_completion_tokens")
            if not isinstance(mct, int) or mct < OPENAI_MAX_COMPLETION_TOKENS_MIN:
                out["max_completion_tokens"] = OPENAI_MAX_COMPLETION_TOKENS_MIN
            out["tool_choice"] = OPENAI_TOOL_CHOICE_FORCED
            out["parallel_tool_calls"] = False
            # Ensure tool schema exists and matches eval_rollout_table_groq
            if not out.get("tools"):
                out["tools"] = _interact_tool_schema()
            return out
        # Non-gpt-5 paths: ensure tools present if missing
        if not payload.get("tools"):
            payload = dict(payload)
            payload["tools"] = _interact_tool_schema()
        return payload

    # --- Crafter crafting rules system hint ---
    CRAFTING_RULES_SYSTEM_HINT = (
        "Crafter crafting rules (from the paper):\n"
        "- Make Wood Pickaxe: Nearby a table; have wood in inventory.\n"
        "- Make Stone Pickaxe: Nearby a table; have wood and stone in inventory.\n"
        "- Make Iron Pickaxe: Nearby a table; furnace exists; have wood, coal, and iron in inventory.\n"
        "- Make Wood Sword: Nearby a table; have wood in inventory.\n"
        "- Make Stone Sword: Nearby a table; have wood and stone in inventory.\n"
        "- Make Iron Sword: Nearby a table; furnace exists; have wood, coal, and iron in inventory."
    )

    def _inject_crafting_rules(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return payload
            msgs = payload.get("messages")
            if not isinstance(msgs, list):
                return payload
            # If first message is a system prompt, append rules; otherwise, insert a new system message
            if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                existing = msgs[0].get("content")
                if isinstance(existing, str) and CRAFTING_RULES_SYSTEM_HINT not in existing:
                    msgs[0]["content"] = (existing.rstrip() + "\n\n" + CRAFTING_RULES_SYSTEM_HINT)
            else:
                msgs.insert(0, {"role": "system", "content": CRAFTING_RULES_SYSTEM_HINT})
            payload["messages"] = msgs
        except Exception:
            pass
        return payload

    def _prepare_for_groq(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        # Groq follows OpenAI schema; keep 'reasoning' if caller provides it (eval script may set it)
        out = dict(payload)
        # Drop thinking-only fields our eval scripts sometimes include
        for k in ("thinking_mode", "thinking_budget", "stop_after_tool_calls"):
            out.pop(k, None)
        # Ensure tools exist (single function tool)
        if not out.get("tools"):
            out["tools"] = _interact_tool_schema()
        return out

    # ——— Helpers to normalize assistant tool calls (mirror eval_rollout_table_groq parsing) ———
    import json as _json
    import re as _re

    _THINK_TAG_PATTERN = _re.compile(r"<think>(.*?)</think>", _re.IGNORECASE | _re.DOTALL)

    def _extract_message_text(message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        texts: list[str] = []
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    texts.append(str(part["text"]))
        elif content not in (None, ""):
            try:
                texts.append(_json.dumps(content, ensure_ascii=False))
            except Exception:
                texts.append(str(content))
        return "\n".join(t for t in texts if t)

    def _parse_tool_call_from_text(text: str) -> tuple[list[str], str]:
        actions: list[str] = []
        reasoning = ""
        if not isinstance(text, str) or not text:
            return actions, reasoning
        candidates = _re.findall(r"\{[\s\S]*\}", text)
        for raw in reversed(candidates):
            try:
                obj = _json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                tool = obj.get("tool") or obj.get("name")
                args = obj.get("args") or obj.get("arguments") or {}
                if isinstance(tool, str) and tool.lower() == "interact" and isinstance(args, dict):
                    cand_actions = args.get("actions")
                    cand_reason = args.get("reasoning")
                    if isinstance(cand_actions, list):
                        actions = [str(a) for a in cand_actions]
                    if isinstance(cand_reason, str):
                        reasoning = cand_reason
                    break
        return actions, reasoning

    # Debug endpoint to verify secrets/keys are present in runtime
    @api.get("/debug/env")
    def debug_env():
        gr = os.environ.get("GROQ_API_KEY", "")
        oa = os.environ.get("OPENAI_API_KEY", "")
        return {
            "has_GROQ_API_KEY": bool(gr),
            "GROQ_API_KEY_prefix": (gr[:6] + "…") if len(gr) >= 6 else ("set" if gr else ""),
            "has_OPENAI_API_KEY": bool(oa),
            "OPENAI_API_KEY_prefix": (oa[:6] + "…") if len(oa) >= 6 else ("set" if oa else ""),
        }

    @api.post("/proxy/v1/chat/completions")
    def proxy_chat_completions(req: dict[str, Any] = Body(...)):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Missing OPENAI_API_KEY in task service environment",
            )
        model = req.get("model")
        payload = _prepare_for_openai(model, req)
        payload = _inject_crafting_rules(payload)
        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            try:
                data = resp.json()
            except Exception:
                data = {"error": "invalid_json", "raw": resp.text[:800]}
            if resp.status_code >= 400:
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=resp.status_code, content=data)
            # Post-process: if no tool_calls are present, try to synthesize from assistant text (parity with groq eval)
            try:
                if isinstance(data, dict):
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                        has_tools = isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls")
                        if not has_tools:
                            assistant_text = _extract_message_text(msg)
                            acts, reason = _parse_tool_call_from_text(assistant_text)
                            if acts:
                                tool_call = {
                                    "id": "toolcall_1",
                                    "type": "function",
                                    "function": {
                                        "name": "interact",
                                        "arguments": _json.dumps({"actions": acts, "reasoning": reason or ""}, ensure_ascii=False),
                                    },
                                }
                                # ensure list
                                if isinstance(msg, dict):
                                    msg["tool_calls"] = [tool_call]
                                    # write back
                                    choices[0]["message"] = msg
                                data["choices"] = choices
            except Exception:
                pass
            return data

    @api.post("/proxy/groq/v1/chat/completions")
    def proxy_groq_chat_completions(req: dict[str, Any] = Body(...)):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Missing GROQ_API_KEY in task service environment",
            )
        model = req.get("model")
        payload = _prepare_for_groq(model, req)
        payload = _inject_crafting_rules(payload)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions").rstrip("/")
        # --- verbose, sanitized request logging ---
        try:
            _msgs = payload.get("messages") or []
            msg_count = len(_msgs) if isinstance(_msgs, list) else 0
            tool_count = len(payload.get("tools") or [])
            tool_choice = payload.get("tool_choice")
            max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
            temperature = payload.get("temperature")
            print(
                f"[proxy:groq] sending model={model} messages={msg_count} tools={tool_count} tool_choice={bool(tool_choice)} max_tokens={max_tokens} temperature={temperature}",
                flush=True,
            )
        except Exception:
            pass
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            req_id = resp.headers.get("x-request-id") or resp.headers.get("request-id")
            try:
                print(f"[proxy:groq] response status={resp.status_code} req_id={req_id} body_snippet={resp.text[:400]}", flush=True)
            except Exception:
                pass
            try:
                data = resp.json()
            except Exception:
                data = {"error": "invalid_json", "raw": resp.text[:800]}
            if resp.status_code >= 400:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=resp.status_code, content=data)
            # Best-effort synthesis of tool_calls if absent (parity with Groq eval script behavior)
            try:
                if isinstance(data, dict):
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                        has_tools = isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list) and msg.get("tool_calls")
                        if not has_tools:
                            assistant_text = _extract_message_text(msg)
                            acts, reason = _parse_tool_call_from_text(assistant_text)
                            if acts:
                                tool_call = {
                                    "id": "toolcall_1",
                                    "type": "function",
                                    "function": {
                                        "name": "interact",
                                        "arguments": _json.dumps({"actions": acts, "reasoning": reason or ""}, ensure_ascii=False),
                                    },
                                }
                                if isinstance(msg, dict):
                                    msg["tool_calls"] = [tool_call]
                                    choices[0]["message"] = msg
                                data["choices"] = choices
            except Exception:
                pass
            return data

    return api


# Local development mode
if __name__ == "__main__":
    import argparse
    import uvicorn
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally for development")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()

    if args.local:
        # Set up environment for local development
        if not os.getenv("ENVIRONMENT_API_KEY"):
            # Try to load from backend .env.dev
            env_dev_path = Path(__file__).parent.parent.parent.parent.parent / "backend" / ".env.dev"
            if env_dev_path.exists():
                try:
                    import dotenv
                    dotenv.load_dotenv(env_dev_path)
                    print(f"Loaded environment from {env_dev_path}")
                except ImportError:
                    print("dotenv not available, using existing environment")
            else:
                print(f"Warning: {env_dev_path} not found")

        # Ensure required environment variables
        required_vars = ["ENVIRONMENT_API_KEY", "OPENAI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"Missing required environment variables: {missing}")
            sys.exit(1)

        print(f"Starting task app locally on {args.host}:{args.port}")
        print("Rollout endpoint: http://localhost:8001/rollout")
        print("Health endpoint: http://localhost:8001/health")

        # Create and run the app
        app = fastapi_app()
        uvicorn.run(app, host=args.host, port=args.port, reload=True)
    else:
        print("Use --local to run locally, or deploy to Modal")


