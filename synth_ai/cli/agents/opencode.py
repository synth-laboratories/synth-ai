import subprocess
from pathlib import Path
from typing import Any

import click

from synth_ai.cli.lib.agents import write_agents_md
from synth_ai.cli.lib.bin import install_bin, verify_bin
from synth_ai.cli.lib.env import resolve_env_var
from synth_ai.core.env import get_backend_from_env
from synth_ai.core.json import create_and_write_json, load_json_to_dict
from synth_ai.core.paths import get_bin_path
from synth_ai.core.urls import BACKEND_URL_SYNTH_RESEARCH_BASE
from synth_ai.data.enums import SYNTH_MODEL_NAMES

CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
SYNTH_PROVIDER_ID = "synth"

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


def _load_session_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load session configuration from TOML file.
    
    Looks for [session] section with:
    - limit_cost_usd: float (default: 20.0)
    - limit_tokens: int | None
    - limit_gpu_hours: float | None
    
    Returns dict with session limits.
    """
    if config_path is None:
        # Look for opencode.toml or synth.toml in current directory
        for name in ["opencode.toml", "synth.toml"]:
            path = Path(name)
            if path.exists():
                config_path = path
                break
    
    if config_path is None or not config_path.exists():
        # Default: $20 cost limit
        return {"limit_cost_usd": 20.0}
    
    if tomllib is None:
        return {"limit_cost_usd": 20.0}
    
    try:
        with config_path.open("rb") as f:
            config = tomllib.load(f)
        
        session_config = config.get("session", {})
        result: dict[str, Any] = {}
        
        # Default cost limit is $20 if not specified
        result["limit_cost_usd"] = session_config.get("limit_cost_usd", 20.0)
        
        if "limit_tokens" in session_config:
            result["limit_tokens"] = session_config["limit_tokens"]
        if "limit_gpu_hours" in session_config:
            result["limit_gpu_hours"] = session_config["limit_gpu_hours"]
        
        return result
    except Exception as e:
        click.echo(f"Warning: Failed to load session config from {config_path}: {e}", err=True)
        return {"limit_cost_usd": 20.0}


@click.command("opencode")
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None
)
@click.option(
    "--force",
    is_flag=True,
    help="Prompt for API keys even if cached values exist."
)
@click.option(
    "--url",
    "override_url",
    type=str,
    default=None,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to TOML config file (default: opencode.toml or synth.toml in current directory)"
)
def opencode_cmd(
    model_name: str | None = None,
    force: bool = False,
    override_url: str | None = None,
    config_path: Path | None = None,
) -> None:

    while True:
        bin_path = get_bin_path("opencode")
        if bin_path:
            break
        if not install_bin(
            "OpenCode",
            [
                "brew install opencode",
                "bun add -g opencode-ai",
                "curl -fsSL https://opencode.ai/install | bash",
                "npm i -g opencode-ai",
                "paru -S opencode"
            ]
        ):
            print("Failed to find your installed OpenCode")
            print("Please install from: https://opencode.ai")
            return
    print(f"Using OpenCode at {bin_path}")

    if not verify_bin(bin_path):
        print("Failed to verify OpenCode is runnable")
        return

    write_agents_md()

    # Load session config from TOML
    session_config = _load_session_config(config_path)
    session_limit_cost = session_config.get("limit_cost_usd")
    session_limit_tokens = session_config.get("limit_tokens")
    session_limit_gpu_hours = session_config.get("limit_gpu_hours")
    
    # Get API key and base URL for session creation
    synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    if override_url:
        base_url = override_url.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]
    else:
        base_url, _ = get_backend_from_env()
    
    # Create agent session with limits from config (default: $20 cost limit)
    session_id: str | None = None
    if session_limit_tokens or session_limit_cost or session_limit_gpu_hours:
        try:
            import asyncio
            
            # Create session - org_id will be fetched from backend /me endpoint
            async def create_session():
                from synth_ai.cli.local.session import AgentSessionClient
                
                client = AgentSessionClient(f"{base_url}/api", synth_api_key)
                
                limits = []
                if session_limit_tokens:
                    limits.append({
                        "limit_type": "hard",
                        "metric_type": "tokens",
                        "limit_value": float(session_limit_tokens),
                    })
                if session_limit_cost:
                    limits.append({
                        "limit_type": "hard",
                        "metric_type": "cost_usd",
                        "limit_value": float(session_limit_cost),
                    })
                if session_limit_gpu_hours:
                    limits.append({
                        "limit_type": "hard",
                        "metric_type": "gpu_hours",
                        "limit_value": float(session_limit_gpu_hours),
                    })
                
                # org_id will be automatically fetched from /api/v1/me endpoint
                session = await client.create(
                    org_id=None,  # Will be fetched from backend
                    limits=limits,
                    tracing_session_id=None,
                    session_type="opencode_agent",
                )
                return session.session_id
            
            session_id = asyncio.run(create_session())
            click.echo(f"✓ Created agent session: {session_id}")
            if session_limit_tokens:
                click.echo(f"  Token limit: {session_limit_tokens:,}")
            if session_limit_cost:
                click.echo(f"  Cost limit: ${session_limit_cost:.2f}")
            if session_limit_gpu_hours:
                click.echo(f"  GPU hours limit: {session_limit_gpu_hours:.2f}")
        except Exception as e:
            click.echo(f"Warning: Failed to create agent session: {e}", err=True)
            click.echo("Continuing without session limits...", err=True)

    if model_name is not None:
        if model_name not in SYNTH_MODEL_NAMES:
            raise ValueError(
                f"model_name={model_name} is invalid. Valid values for model_name: {SYNTH_MODEL_NAMES}"
            )
        synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        data = load_json_to_dict(AUTH_PATH)
        good_entry = {
            "type": "api",
            "key": synth_api_key,
        }
        if data.get(SYNTH_PROVIDER_ID) != good_entry:
            data[SYNTH_PROVIDER_ID] = good_entry
        create_and_write_json(AUTH_PATH, data)
        config = load_json_to_dict(CONFIG_PATH)
        config.setdefault("$schema", "https://opencode.ai/config.json")
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_BASE
        provider_section = config.setdefault("provider", {})
        synth_provider = provider_section.setdefault(SYNTH_PROVIDER_ID, {})
        synth_provider["npm"] = "@ai-sdk/openai-compatible"
        synth_provider.setdefault("name", "Synth")
        models = synth_provider.setdefault("models", {})
        models.setdefault(model_name, {})
        options = synth_provider.setdefault("options", {})
        options["baseURL"] = url
        full_model_name = f"{SYNTH_PROVIDER_ID}/{model_name}"
        config["model"] = full_model_name
        create_and_write_json(CONFIG_PATH, config)

    try:
        subprocess.run([str(bin_path)], check=True)
    except subprocess.CalledProcessError:
        print("Failed to launch OpenCode")
    finally:
        # End session if created
        if session_id is not None:
            try:
                # Assign to local variable to help type checker
                final_session_id: str = session_id
                import asyncio
                
                async def end_session():
                    from synth_ai.cli.local.session import AgentSessionClient
                    client = AgentSessionClient(f"{base_url}/api", synth_api_key)
                    await client.end(final_session_id)
                    click.echo(f"✓ Ended agent session: {final_session_id}")
                
                asyncio.run(end_session())
            except Exception as e:
                click.echo(f"Warning: Failed to end agent session: {e}", err=True)
