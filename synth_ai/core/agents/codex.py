"""Codex CLI launcher."""

import os
import subprocess
from pathlib import Path

from synth_ai.core.agents.utils import write_agents_md
from synth_ai.core.bin import install_bin, verify_bin
from synth_ai.core.env import get_backend_from_env
from synth_ai.core.env_utils import resolve_env_var
from synth_ai.core.paths import get_bin_path
from synth_ai.core.urls import BACKEND_URL_SYNTH_RESEARCH_OPENAI
from synth_ai.data.enums import SYNTH_MODEL_NAMES

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def _load_session_config(config_path=None):
    """Load session configuration from TOML file."""
    if config_path is None:
        for name in ["codex.toml", "synth.toml"]:
            path = Path(name)
            if path.exists():
                config_path = path
                break

    if config_path is None or not config_path.exists():
        return {"limit_cost_usd": 20.0}

    if tomllib is None:
        return {"limit_cost_usd": 20.0}

    try:
        with config_path.open("rb") as f:
            config = tomllib.load(f)

        session_config = config.get("session", {})
        result = {}
        result["limit_cost_usd"] = session_config.get("limit_cost_usd", 20.0)

        if "limit_tokens" in session_config:
            result["limit_tokens"] = session_config["limit_tokens"]
        if "limit_gpu_hours" in session_config:
            result["limit_gpu_hours"] = session_config["limit_gpu_hours"]

        return result
    except Exception as e:
        print(f"Warning: Failed to load session config from {config_path}: {e}")
        return {"limit_cost_usd": 20.0}


def _create_session(base_url, api_key, session_config, session_type="codex_agent"):
    """Create agent session with limits."""
    import asyncio

    from synth_ai.sdk.session import AgentSessionClient

    session_limit_cost = session_config.get("limit_cost_usd")
    session_limit_tokens = session_config.get("limit_tokens")
    session_limit_gpu_hours = session_config.get("limit_gpu_hours")

    async def create():
        client = AgentSessionClient(f"{base_url}/api", api_key)
        limits = []
        if session_limit_tokens:
            limits.append(
                {
                    "limit_type": "hard",
                    "metric_type": "tokens",
                    "limit_value": float(session_limit_tokens),
                }
            )
        if session_limit_cost:
            limits.append(
                {
                    "limit_type": "hard",
                    "metric_type": "cost_usd",
                    "limit_value": float(session_limit_cost),
                }
            )
        if session_limit_gpu_hours:
            limits.append(
                {
                    "limit_type": "hard",
                    "metric_type": "gpu_hours",
                    "limit_value": float(session_limit_gpu_hours),
                }
            )

        session = await client.create(
            org_id=None,
            limits=limits,
            tracing_session_id=None,
            session_type=session_type,
        )
        return session.session_id

    return asyncio.run(create())


def _end_session(base_url, api_key, session_id):
    """End agent session."""
    import asyncio

    from synth_ai.sdk.session import AgentSessionClient

    async def end():
        client = AgentSessionClient(f"{base_url}/api", api_key)
        await client.end(session_id)

    asyncio.run(end())


def run_codex(model_name=None, force=False, override_url=None, wire_api=None, config_path=None):
    """Launch Codex with optional Synth backend routing."""
    while True:
        bin_path = get_bin_path("codex")
        if bin_path:
            break
        if not install_bin("Codex", ["brew install codex", "npm install -g @openai/codex"]):
            print("Failed to find your installed Codex")
            print("Please install from: https://developers.openai.com/codex/cli/")
            return
    print(f"Using Codex at {bin_path}")

    if not verify_bin(bin_path):
        print("Failed to verify Codex is runnable")
        return

    write_agents_md()
    env = os.environ.copy()
    override_args = []

    session_config = _load_session_config(config_path)
    session_limit_cost = session_config.get("limit_cost_usd")
    session_limit_tokens = session_config.get("limit_tokens")
    session_limit_gpu_hours = session_config.get("limit_gpu_hours")

    api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    if override_url:
        base_url = override_url.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]
    else:
        base_url, _ = get_backend_from_env()

    session_id = None
    if session_limit_tokens or session_limit_cost or session_limit_gpu_hours:
        try:
            session_id = _create_session(base_url, api_key, session_config, "codex_agent")
            env["SYNTH_SESSION_ID"] = session_id
            print(f"Created agent session: {session_id}")
            print("  Note: Set X-Session-ID header in Codex config to use this session")
            if session_limit_tokens:
                print(f"  Token limit: {session_limit_tokens:,}")
            if session_limit_cost:
                print(f"  Cost limit: ${session_limit_cost:.2f}")
            if session_limit_gpu_hours:
                print(f"  GPU hours limit: {session_limit_gpu_hours:.2f}")
        except Exception as e:
            print(f"Warning: Failed to create agent session: {e}")
            print("Continuing without session limits...")

    if model_name is not None:
        if model_name not in SYNTH_MODEL_NAMES:
            raise ValueError(f"model_name={model_name} is invalid. Valid: {SYNTH_MODEL_NAMES}")
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_OPENAI

        if wire_api is None:
            wire_api = "responses" if "/responses" in url or url.endswith("/responses") else "chat"

        provider_config_parts = [
            'name="Synth"',
            f'base_url="{url}"',
            'env_key="OPENAI_API_KEY"',
            f'wire_api="{wire_api}"',
        ]
        provider_config = "{" + ",".join(provider_config_parts) + "}"

        config_overrides = [
            f"model_providers.synth={provider_config}",
            'model_provider="synth"',
            f'default_model="{model_name}"',
        ]
        override_args = [arg for override in config_overrides for arg in ("-c", override)]
        env["OPENAI_API_KEY"] = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        env["SYNTH_API_KEY"] = env["OPENAI_API_KEY"]
        print(f"Configured with wire_api={wire_api}")

    try:
        cmd = ["codex"]
        if model_name is not None:
            cmd.extend(["-m", model_name])
        cmd.extend(override_args)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError:
        print("Failed to run Codex")
    finally:
        if session_id is not None:
            try:
                _end_session(base_url, api_key, session_id)
                print(f"Ended agent session: {session_id}")
            except Exception as e:
                print(f"Warning: Failed to end agent session: {e}")
