from __future__ import annotations

import os
import subprocess


def _parse_public_url_from_log(log_path: str) -> str | None:
    try:
        with open(log_path) as fh:
            for line in fh:
                if "modal.run" in line:
                    return line.strip().split()[-1].rstrip("/")
    except Exception:
        return None
    return None


def deploy(script_path: str | None = None, *, env_api_key: str | None = None) -> str:
    """
    Deploy the Math Task App to Modal and return the public URL.

    - If script_path is provided, run it (bash) and parse .last_deploy*.log for URL.
    - Otherwise, try to call a built-in deploy() in examples.rl.task_app if available.
    """
    envp = os.environ.copy()
    if env_api_key:
        envp["ENVIRONMENT_API_KEY"] = env_api_key

    if script_path:
        script_path = os.path.abspath(script_path)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Deploy script not found: {script_path}")
        subprocess.check_call(["bash", script_path], cwd=os.path.dirname(script_path), env=envp)
        for name in (".last_deploy.log", ".last_deploy.dev.log", ".last_deploy.manual.log"):
            url = _parse_public_url_from_log(os.path.join(os.path.dirname(script_path), name))
            if url:
                return url
        raise RuntimeError("Deployed, but failed to extract Modal public URL from deploy logs.")

    try:
        import importlib

        mod = importlib.import_module("examples.rl.task_app")
        if hasattr(mod, "deploy"):
            url = mod.deploy(env_api_key=env_api_key)
            if not url:
                raise RuntimeError("examples.rl.task_app.deploy() returned empty URL")
            return str(url).rstrip("/")
        raise RuntimeError("examples.rl.task_app.deploy() not found")
    except Exception as exc:
        raise RuntimeError(
            f"No deploy script provided and Python-based deploy failed: {exc}. "
            "Pass --script /path/to/deploy_task_app.sh to demo.deploy."
        ) from exc
