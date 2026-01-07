"""Deploy a LocalAPI task app with Cloudflare tunnel.

Usage:
    python -m synth_ai.tui.deploy /path/to/localapi.py

Outputs JSON to stdout:
    {"status": "ready", "url": "https://abc123.trycloudflare.com", "port": 8001}
    {"status": "error", "error": "..."}

The process stays alive to keep the tunnel open. Kill it to tear down.
"""

import asyncio
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType


def _validate_localapi(module: ModuleType, path: Path) -> str | None:
    """
    Validate a LocalAPI module before deployment.

    Returns None if valid, or an error message if invalid.
    """
    from fastapi import FastAPI

    # Check app exists and is FastAPI
    if not hasattr(module, "app"):
        return "LocalAPI must define an 'app' variable"

    app = module.app
    if not isinstance(app, FastAPI):
        return f"'app' must be a FastAPI instance, got {type(app).__name__}"

    # Check required routes exist
    routes = {route.path for route in app.routes}
    if "/health" not in routes:
        return "LocalAPI missing /health endpoint (use create_local_api() to create your app)"
    if "/rollout" not in routes:
        return "LocalAPI missing /rollout endpoint (use create_local_api() to create your app)"

    # Check get_dataset_size is implemented
    if hasattr(module, "get_dataset_size"):
        get_dataset_size = module.get_dataset_size
        if callable(get_dataset_size):
            try:
                result = get_dataset_size()
                if not isinstance(result, int):
                    return f"get_dataset_size() must return an int, got {type(result).__name__}"
                if result <= 0:
                    return f"get_dataset_size() must return a positive number, got {result}"
            except NotImplementedError as e:
                return f"get_dataset_size() not implemented: {e}"
            except Exception as e:
                return f"get_dataset_size() failed: {e}"

    # Check get_sample is implemented
    if hasattr(module, "get_sample"):
        get_sample = module.get_sample
        if callable(get_sample):
            try:
                # Try calling with seed=0 to catch NotImplementedError
                result = get_sample(0)
                if not isinstance(result, dict):
                    return f"get_sample() must return a dict, got {type(result).__name__}"
            except NotImplementedError as e:
                return f"get_sample() not implemented: {e}"
            except Exception as e:
                return f"get_sample(0) failed: {e}"

    # Check score_response is implemented
    if hasattr(module, "score_response"):
        score_response = module.score_response
        if callable(score_response):
            # Check function signature
            sig = inspect.signature(score_response)
            params = list(sig.parameters.keys())
            if len(params) < 2:
                return f"score_response() must accept (response, sample), got {params}"

            try:
                # Try calling with dummy data to catch NotImplementedError
                result = score_response("test response", {"input": "test", "expected": "test"})
                if not isinstance(result, (int, float)):
                    return f"score_response() must return a number, got {type(result).__name__}"
            except NotImplementedError as e:
                return f"score_response() not implemented: {e}"
            except Exception as e:
                # Other errors might be OK - could be due to dummy data not matching expected format
                # But NotImplementedError is definitely a problem
                if "NotImplementedError" in str(type(e)):
                    return f"score_response() not implemented: {e}"

    return None


async def deploy_localapi(localapi_path: str) -> None:
    """Deploy a LocalAPI file and create a Cloudflare tunnel (or localhost in local mode)."""
    import os

    # Check for local mode (skip Cloudflare tunnel, use localhost directly)
    LOCAL_MODE = os.environ.get("SYNTH_LOCAL_MODE", "").lower() in ("1", "true", "yes")

    # Check for API key early - give clear error instead of cryptic auth failure
    if not os.environ.get("SYNTH_API_KEY"):
        _output_error("SYNTH_API_KEY not set - run 'synth auth' or export SYNTH_API_KEY")
        return

    from synth_ai.sdk.localapi.auth import ensure_localapi_auth
    from synth_ai.sdk.task import run_server_background
    from synth_ai.sdk.tunnels import (
        TunnelBackend,
        TunneledLocalAPI,
        wait_for_health_check,
        acquire_port,
        PortConflictBehavior,
    )

    # Get environment API key for local API authentication
    try:
        env_api_key = ensure_localapi_auth()
    except Exception as e:
        _output_error(f"Failed to get environment API key: {e}")
        return

    path = Path(localapi_path).resolve()
    if not path.exists():
        _output_error(f"File not found: {localapi_path}")
        return

    # Import the user's localapi module
    try:
        spec = importlib.util.spec_from_file_location("localapi", path)
        if spec is None or spec.loader is None:
            _output_error(f"Could not load module from: {localapi_path}")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as e:
        _output_error(f"Syntax error in {path.name}: {e}")
        return
    except ImportError as e:
        _output_error(f"Import error in {path.name}: {e}")
        return
    except Exception as e:
        _output_error(f"Failed to load {path.name}: {e}")
        return

    # Validate the module before doing anything else
    validation_error = _validate_localapi(module, path)
    if validation_error:
        _output_error(validation_error)
        return

    app = module.app

    # Start server on available port
    port = acquire_port(8001, on_conflict=PortConflictBehavior.FIND_NEW)
    try:
        run_server_background(app, port)
    except Exception as e:
        _output_error(f"Failed to start server: {e}")
        return

    # Wait for health check
    try:
        await wait_for_health_check("localhost", port, api_key=env_api_key, timeout=30.0)
    except Exception as e:
        _output_error(f"Health check failed: {e}")
        return

    # Create tunnel (or use localhost in local mode)
    if LOCAL_MODE:
        url = f"http://localhost:{port}"
    else:
        try:
            tunnel = await TunneledLocalAPI.create(
                local_port=port,
                backend=TunnelBackend.CloudflareManagedTunnel,
                env_api_key=env_api_key,
                progress=False,  # Don't pollute stdout
            )
            url = tunnel.url
        except Exception as e:
            _output_error(f"Failed to create tunnel: {e}")
            return

    # Output success - TUI will parse this
    _output_ready(url, port)

    # Keep alive until killed
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass


def _output_ready(url: str, port: int) -> None:
    """Output ready status as JSON."""
    print(json.dumps({"status": "ready", "url": url, "port": port}), flush=True)


def _output_error(error: str) -> None:
    """Output error status as JSON."""
    print(json.dumps({"status": "error", "error": error}), flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        print(json.dumps({"status": "error", "error": "Usage: python -m synth_ai.tui.deploy <localapi.py>"}))
        sys.exit(1)

    localapi_path = sys.argv[1]
    asyncio.run(deploy_localapi(localapi_path))


if __name__ == "__main__":
    main()
