from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import click

from synth_ai.cli.lib import (
    ensure_local_port_available,
    find_asgi_apps,
    is_modal_public_url,
    popen_capture,
    popen_stream_capture,
)
from synth_ai.demos import core as demo_core
from synth_ai.demos.core import DEFAULT_TASK_APP_SECRET_NAME


def _app_name_from_url(url: str) -> str | None:
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        if "--" not in host:
            return None
        suffix = host.split("--", 1)[1]
        core = suffix.split('.modal', 1)[0]
        if core.endswith('-fastapi-app'):
            core = core[: -len('-fastapi-app')]
        return core.strip() or None
    except Exception:
        return None


def run_deploy(
    *,
    local: bool,
    app: str | None,
    name: str | None,
    script: str | None,
) -> int:
    demo_dir = demo_core.load_demo_dir()
    if demo_dir and os.path.isdir(demo_dir):
        os.chdir(demo_dir)
        print(f"Using demo directory: {demo_dir}")

    template_id = demo_core.load_template_id()
    env = demo_core.load_env()
    os.environ["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)
    url = ""
    app_name = env.task_app_name or ""
    local_proc = None

    try:
        is_local_template = template_id == "crafter-local"
        if is_local_template and not local:
            print("Detected Crafter demo template; defaulting to local FastAPI deployment.")
            local = True

        if local:
            print("Starting local Task App…")
            import subprocess

            cwd = os.getcwd()
            run_env = os.environ.copy()
            if is_local_template:
                run_env.setdefault("TASKAPP_TRACING_ENABLED", "1")
                traces_dir = os.path.join(cwd, "traces", "v3")
                run_env.setdefault("TASKAPP_SFT_OUTPUT_DIR", traces_dir)
            target = "http://127.0.0.1:8080"
            env_file_path = os.path.join(cwd, ".env")

            if is_local_template:
                task_app_path = os.path.join(cwd, "task_app.py")
                if not os.path.isfile(task_app_path):
                    raise FileNotFoundError(
                        "Expected task_app.py in demo directory for Crafter template"
                    )
                target = "http://127.0.0.1:8001"
                if not ensure_local_port_available("127.0.0.1", 8001):
                    return 1
                local_cmd = [
                    sys.executable,
                    task_app_path,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8001",
                ]
                if os.path.isfile(env_file_path):
                    local_cmd.extend(["--env-file", env_file_path])
            else:
                if not ensure_local_port_available("127.0.0.1", 8080):
                    return 1
                local_cmd = [
                    sys.executable,
                    "-c",
                    "from synth_ai.demos.math.app import run; run()",
                ]

            proc = subprocess.Popen(
                local_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                cwd=cwd,
                env=run_env,
            )
            print(
                "\nLocal server is running in this terminal. Leave this window open and run the next step from a new terminal.\n"
                "Press Ctrl+C here when you're ready to stop the server.\n"
            )
            local_proc = proc
            app_name = ""
            for _ in range(60):
                if proc.poll() is not None:
                    break
                if demo_core.assert_http_ok(target + "/health", method="GET") or demo_core.assert_http_ok(
                    target, method="GET"
                ):
                    url = target
                    break
                time.sleep(1)
            if not url:
                print("Failed to verify local task app health. See logs above.")
                if local_proc and local_proc.poll() is None:
                    with contextlib.suppress(Exception):
                        local_proc.send_signal(signal.SIGINT)
                    with contextlib.suppress(Exception):
                        local_proc.wait(timeout=5)
                return 2
        else:
            app_path = os.path.abspath(app) if app else None
            if not app_path or not os.path.isfile(app_path):
                candidates = [
                    os.path.abspath(os.path.join(os.getcwd(), "synth_demo", "task_app.py")),
                    os.path.abspath(os.path.join(os.getcwd(), "task_app.py")),
                    os.path.abspath(os.path.join(os.getcwd(), "app.py")),
                    os.path.abspath(os.path.join(os.getcwd(), "math_task_app.py")),
                ]
                app_path = next((p for p in candidates if os.path.isfile(p)), None)
                if not app_path:
                    found = find_asgi_apps(Path(os.getcwd()))
                    if found:
                        print("Select a Modal ASGI app to deploy:")
                        for idx, candidate in enumerate(found, 1):
                            rel = os.path.relpath(str(candidate), os.getcwd())
                            print(f"  [{idx}] {rel}")
                        try:
                            sel = input(f"Enter choice [1-{len(found)}] (default 1): ").strip() or "1"
                        except Exception:
                            sel = "1"
                        try:
                            choice = int(sel)
                        except Exception:
                            choice = 1
                        choice = max(1, min(choice, len(found)))
                        app_path = str(found[choice - 1].resolve())

            if not app_path and script:
                from synth_ai.demos.math.deploy_modal import deploy as modal_deploy

                url = modal_deploy(script_path=script, env_api_key=env.env_api_key)
                if name:
                    app_name = name
            else:
                if not app_path:
                    entered = input("Path to Modal app.py (e.g., ./task_app.py): ").strip()
                    if not entered:
                        raise FileNotFoundError("No app.py path provided and auto-detect failed")
                    app_path = os.path.abspath(entered)
                if not os.path.isfile(app_path):
                    raise FileNotFoundError(f"App file not found: {app_path}")

                print(f"Using task app: {app_path}")
                existing_name = (name or env.task_app_name or "").strip()
                if not existing_name:
                    existing_name = f"synth-{os.path.splitext(os.path.basename(app_path))[0]}"

                chosen_name = input(f"Modal app name [{existing_name}]: ").strip() or existing_name
                app_name = chosen_name
                print("\nAbout to deploy with:")
                print(f"  app.py: {app_path}")
                print(f"  name:   {chosen_name}")
                proceed = (input("Proceed? [Y/n]: ").strip().lower() or "y").startswith("y")
                if not proceed:
                    print("Aborted by user.")
                    return 1

                existing_env_key = (env.env_api_key or "").strip()
                env_key = existing_env_key or None
                if existing_env_key:
                    try:
                        reuse_choice = (
                            input("Use existing ENVIRONMENT_API_KEY from state/.env? [Y/n]: ")
                            .strip()
                            .lower()
                            or "y"
                        )
                    except Exception:
                        reuse_choice = "y"
                    if not reuse_choice.startswith("y"):
                        env_key = None

                if env_key is None:
                    from synth_ai.learning.rl.secrets import mint_environment_api_key

                    env_key = mint_environment_api_key()
                    demo_core.persist_env_api_key(env_key)
                    demo_core.persist_dotenv_values({"ENVIRONMENT_API_KEY": env_key})
                    os.environ["ENVIRONMENT_API_KEY"] = env_key
                    env.env_api_key = env_key
                    local_env["ENVIRONMENT_API_KEY"] = env_key
                    print("[deploy] Minted new ENVIRONMENT_API_KEY")
                else:
                    os.environ["ENVIRONMENT_API_KEY"] = env_key

                backend_base = (env.dev_backend_url or "").rstrip("/")
                synth_key = (
                    env.synth_api_key
                    or os.environ.get("SYNTH_API_KEY")
                    or local_env.get("SYNTH_API_KEY")
                    or ""
                ).strip()
                if backend_base and synth_key:
                    non_api_base = backend_base[:-4] if backend_base.endswith("/api") else backend_base
                    try:
                        choice = (
                            input(f"Upload ENVIRONMENT_API_KEY to backend {non_api_base}? [Y/n]: ")
                            .strip()
                            .lower()
                            or "y"
                        )
                    except Exception:
                        choice = "y"
                    if choice.startswith("y"):
                        try:
                            print(f"[deploy] Uploading ENVIRONMENT_API_KEY to {non_api_base} …")
                            from synth_ai.learning.rl.env_keys import setup_environment_api_key

                            setup_environment_api_key(non_api_base, synth_key, token=env_key)
                            print("[deploy] Backend sealed-box upload complete.")
                        except Exception as upload_err:
                            print(f"[deploy] Failed to upload ENVIRONMENT_API_KEY: {upload_err}")
                            print(
                                'Hint: run `uvx python -c "from synth_ai.learning.rl.env_keys import setup_environment_api_key as s;'
                                " s('<backend>', '<synth_api_key>')\" once the backend is reachable."
                            )

                synth_key = (
                    env.synth_api_key
                    or os.environ.get("SYNTH_API_KEY")
                    or local_env.get("SYNTH_API_KEY")
                    or ""
                ).strip()
                if not synth_key:
                    synth_key = input("Enter SYNTH_API_KEY for deployment (required): ").strip()
                    if not synth_key:
                        print("SYNTH_API_KEY is required for deployment.")
                        return 1
                    demo_core.persist_api_key(synth_key)
                    demo_core.persist_dotenv_values({"SYNTH_API_KEY": synth_key})
                    env.synth_api_key = synth_key
                os.environ["SYNTH_API_KEY"] = synth_key

                openai_key = (
                    os.environ.get("OPENAI_API_KEY") or local_env.get("OPENAI_API_KEY") or ""
                ).strip()
                if not openai_key:
                    openai_key = input(
                        "Enter your OpenAI API key, found at https://platform.openai.com/api-keys\n> "
                    ).strip()
                    if not openai_key:
                        print("OPENAI_API_KEY is required for deployment.")
                        return 1
                    demo_core.persist_dotenv_values({"OPENAI_API_KEY": openai_key})
                    local_env["OPENAI_API_KEY"] = openai_key
                os.environ["OPENAI_API_KEY"] = openai_key

                deploy_cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "modal",
                    "deploy",
                    "--name",
                    chosen_name,
                    app_path,
                ]
                print(
                    "\nStreaming Modal build/deploy logs (this can take several minutes on first run)…\n"
                )
                code, deploy_logs = popen_stream_capture(deploy_cmd)
                if code != 0:
                    raise RuntimeError(f"modal deploy failed (exit {code})")
                if not url:
                    try:
                        import re as _re

                        matches = _re.findall(r"https?://[^\s]+\.modal\.run", deploy_logs or "")
                        if matches:
                            url = matches[-1].strip().rstrip("/")
                    except Exception:
                        pass
                url_cmd = ["uv", "run", "python", "-m", "modal", "app", "url", chosen_name]
                code2, out2 = popen_capture(url_cmd)
                if code2 == 0:
                    for token in out2.split():
                        if is_modal_public_url(token):
                            url = token.strip().rstrip("/")
                            break
                if not url:
                    code3, out3 = popen_capture(
                        ["uv", "run", "python", "-m", "modal", "app", "list"]
                    )
                    if code3 == 0 and out3:
                        for line in out3.splitlines():
                            if chosen_name in line:
                                for token in line.split():
                                    if is_modal_public_url(token):
                                        url = token.strip().rstrip("/")
                                        break
                            if url:
                                break
                if not url:
                    print("\nCould not auto-detect a public Modal URL for the app.")
                    entered = input(
                        "Enter the Modal public URL (must contain '.modal.run'), or press Enter to abort: "
                    ).strip()
                    if entered and is_modal_public_url(entered):
                        url = entered.rstrip("/")
                if not url:
                    raise RuntimeError("Failed to resolve public URL from modal CLI output")

        if not url:
            print("Failed to determine Task App URL")
            return 2

        derived_name = _app_name_from_url(url)
        if derived_name:
            app_name = derived_name

        previous_url = (env.task_app_base_url or "").strip()
        persist_url = True
        if previous_url and previous_url.rstrip("/") != url.rstrip("/"):
            try:
                choice = (
                    input(
                        "Stored TASK_APP_BASE_URL differs from the new deployment. "
                        f"Current: {previous_url}\n"
                        "Update stored URL to the new value? [Y/n]: "
                    )
                    .strip()
                    .lower()
                    or "y"
                )
            except Exception:
                choice = "y"
            persist_url = choice.startswith("y")
            if not persist_url and app_name and app_name != env.task_app_name:
                demo_core.persist_task_url(previous_url, name=app_name or None)

        if persist_url:
            demo_core.persist_task_url(url, name=app_name or None)
            dotenv_values = {"TASK_APP_BASE_URL": url}
            if app_name:
                dotenv_values["TASK_APP_NAME"] = app_name
            dotenv_values["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
            dotenv_path = demo_core.persist_dotenv_values(dotenv_values)
            print(f"TASK_APP_BASE_URL={url}")
            if app_name:
                print(f"TASK_APP_NAME={app_name}")
            print("Export for this shell:")
            print(f"  export TASK_APP_BASE_URL={url}")
            if app_name:
                print(f"  export TASK_APP_NAME={app_name}")
            print(f"Persisted to {dotenv_path}")
        else:
            print(f"Keeping existing TASK_APP_BASE_URL={previous_url}")
            print(f"New deployment URL: {url}")

        if is_local_template:
            demo_dir_path = Path(os.getcwd()).resolve()
            print("\n ➡️  Next, open a new terminal and run:")
            print("    cd", demo_dir_path)
            print("    uvx python run_local_rollout_traced.py")
        else:
            print("\nNext step:\n$ uvx synth-ai demo run")

        if local_proc is not None:
            print("\nPress Ctrl+C here to stop the local server and exit this command.\n")
            try:
                local_proc.wait()
            except KeyboardInterrupt:
                print("Stopping local server…")
                with contextlib.suppress(Exception):
                    local_proc.send_signal(signal.SIGINT)
                try:
                    local_proc.wait(timeout=10)
                except Exception:
                    with contextlib.suppress(Exception):
                        local_proc.kill()
                print("Local server stopped.")
        return 0
    except Exception as exc:
        print(f"Deploy error: {exc}")
        if local_proc and local_proc.poll() is None:
            with contextlib.suppress(Exception):
                local_proc.kill()
        return 2


def register(group):
    @group.command("deploy")
    @click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
    @click.option(
        "--app",
        type=click.Path(),
        default=None,
        help="Path to Modal app.py for uv run modal deploy",
    )
    @click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
    @click.option(
        "--script",
        type=click.Path(),
        default=None,
        help="Path to deploy_task_app.sh (optional legacy)",
    )
    def demo_deploy(local: bool, app: str | None, name: str | None, script: str | None):
        code = run_deploy(local=local, app=app, name=name, script=script)
        if code:
            raise click.exceptions.Exit(code)
