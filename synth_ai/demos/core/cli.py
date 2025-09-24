from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Callable

from synth_ai.demos.demo_task_apps import core as demo_core
from synth_ai.demos.demo_task_apps.core import DemoEnv


def cmd_check(_args: argparse.Namespace) -> int:
    env = demo_core.load_env()
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)

    def _refresh_env() -> None:
        nonlocal env, local_env
        env = demo_core.load_env()
        local_env = demo_core.load_dotenv_file(cwd_env_path)

    def _maybe_fix_task_url() -> None:
        if not env.task_app_name:
            return
        current = env.task_app_base_url
        needs_lookup = False
        if not current:
            needs_lookup = True
        elif not current.endswith(".run") or current.endswith(".moda") or current.count(".") < 2:
            needs_lookup = True
        elif not current.startswith("http://") and not current.startswith("https://"):
            needs_lookup = True
        if not needs_lookup:
            return
        code, out = _popen_capture([
            "uv",
            "run",
            "python",
            "-m",
            "modal",
            "app",
            "url",
            env.task_app_name,
        ])
        if code != 0 or not out:
            return
        new_url = ""
        for token in out.split():
            if token.startswith("http://") or token.startswith("https://"):
                new_url = token.strip().rstrip("/")
                break
        if new_url and new_url != current:
            print(f"Updating TASK_APP_BASE_URL from Modal CLI → {new_url}")
            demo_core.persist_task_url(new_url, name=env.task_app_name)
            dotenv_values = {
                "TASK_APP_BASE_URL": new_url,
                "TASK_APP_NAME": env.task_app_name,
                "TASK_APP_SECRET_NAME": env.task_app_secret_name or f"{env.task_app_name}-secret",
            }
            demo_core.persist_dotenv_values(dotenv_values)
            os.environ["TASK_APP_BASE_URL"] = new_url
            _refresh_env()

    synth_key = env.synth_api_key.strip()
    if not synth_key:
        print("SYNTH_API_KEY missing from environment/.env.")
        entered = input("Enter SYNTH_API_KEY (required): ").strip()
        if not entered:
            print("SYNTH_API_KEY is required.")
            return 1
        os.environ["SYNTH_API_KEY"] = entered
        demo_core.persist_api_key(entered)
        path = demo_core.persist_dotenv_values({"SYNTH_API_KEY": entered})
        print(f"Stored SYNTH_API_KEY in {path}")
        _refresh_env()
        synth_key = entered
    elif not local_env.get("SYNTH_API_KEY"):
        path = demo_core.persist_dotenv_values({"SYNTH_API_KEY": synth_key})
        print(f"Stored SYNTH_API_KEY in {path}")
        _refresh_env()

    modal_ok, modal_msg = demo_core.modal_auth_status()
    print(f"Modal auth: {'OK' if modal_ok else 'MISSING'} ({modal_msg})")

    _maybe_fix_task_url()

    ok_backend = False
    ok_task = False
    if env.dev_backend_url:
        api = env.dev_backend_url.rstrip("/") + ("" if env.dev_backend_url.endswith("/api") else "/api")
        ok_backend = demo_core.assert_http_ok(api + "/health", method="GET")
        print(f"Backend health: {'OK' if ok_backend else 'FAIL'} ({api}/health)")
    else:
        print("Backend URL missing; set DEV_BACKEND_URL.")
    if env.task_app_base_url:
        ok_task = demo_core.assert_http_ok(env.task_app_base_url.rstrip("/") + "/health", method="GET") or \
                  demo_core.assert_http_ok(env.task_app_base_url.rstrip("/"), method="GET")
        print(f"Task app: {'OK' if ok_task else 'UNREACHABLE'} ({env.task_app_base_url})")
    else:
        print("Task app URL not set; run: uvx synth-ai rl_demo deploy")

    print("uv: ", end="")
    try:
        import subprocess

        subprocess.check_call(["uv", "--version"])
    except Exception:
        print("(uv not found; install with `pip install uv`)\n", flush=True)

    status = 0
    if not ok_backend:
        status = 1
    if not modal_ok:
        status = 1
    if not env.synth_api_key:
        status = 1
    return status


def _popen_capture(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> tuple[int, str]:
    import subprocess
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate()
        return int(proc.returncode or 0), out or ""
    except Exception as e:
        return 1, str(e)


def _popen_stream(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> int:
    """Stream subprocess output line-by-line to stdout for real-time feedback."""

    import subprocess
    import threading

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        print(f"Failed to launch {' '.join(cmd)}: {exc}")
        return 1

    def _pump(stdout) -> None:
        try:
            for line in stdout:
                print(line.rstrip())
        except Exception:
            pass

    if proc.stdout is not None:
        t = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=1.0)
    else:
        proc.wait()
    return int(proc.returncode or 0)


def cmd_deploy(args: argparse.Namespace) -> int:
    env = demo_core.load_env()
    url = ""
    app_name = env.task_app_name or ""
    try:
        if args.local:
            print("Starting local Task App…")
            import subprocess
            subprocess.Popen([sys.executable, "-c", "from synth_ai.demos.demo_task_apps.math.app import run; run()"],
                             stdout=sys.stdout, stderr=sys.stderr)
            target = "http://127.0.0.1:8080"
            app_name = ""
            for _ in range(30):
                if demo_core.assert_http_ok(target + "/health", method="GET") or demo_core.assert_http_ok(target, method="GET"):
                    url = target
                    break
                time.sleep(1)
        else:
            # Auto-detect app path if not supplied; prompt for name and confirmation.
            app_path = os.path.abspath(args.app) if args.app else None
            if not app_path or not os.path.isfile(app_path):
                candidates = [
                    os.path.abspath(os.path.join(os.getcwd(), "math_task_app.py")),
                    "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/math_task_app.py",
                ]
                app_path = next((p for p in candidates if os.path.isfile(p)), None)
            if not app_path and args.script:
                # Legacy script fallback if user supplied --script explicitly
                from synth_ai.demos.demo_task_apps.math.deploy_modal import deploy as modal_deploy
                url = modal_deploy(script_path=args.script, env_api_key=env.env_api_key)
                if args.name:
                    app_name = args.name
            else:
                if not app_path:
                    entered = input("Path to Modal app.py (e.g., tests/applications/math/rl/math_task_app.py): ").strip()
                    if not entered:
                        raise FileNotFoundError("No app.py path provided and auto-detect failed")
                    app_path = os.path.abspath(entered)
                if not os.path.isfile(app_path):
                    raise FileNotFoundError(f"App file not found: {app_path}")
                suggested_name = args.name or f"synth-{os.path.splitext(os.path.basename(app_path))[0]}"
                name_in = input(f"Modal app name [{suggested_name}]: ").strip() or suggested_name
                app_name = name_in
                print("\nAbout to deploy with:")
                print(f"  app.py: {app_path}")
                print(f"  name:   {name_in}")
                proceed = (input("Proceed? [Y/n]: ").strip().lower() or "y").startswith("y")
                if not proceed:
                    print("Aborted by user.")
                    return 1
                deploy_cmd = ["uv", "run", "python", "-m", "modal", "deploy", "--name", name_in, app_path]
                code, out = _popen_capture(deploy_cmd)
                print(out)
                if code != 0:
                    raise RuntimeError(f"modal deploy failed (exit {code})")
                url_cmd = ["uv", "run", "python", "-m", "modal", "app", "url", name_in]
                code2, out2 = _popen_capture(url_cmd)
                if code2 == 0:
                    for token in out2.split():
                        if token.startswith("http://") or token.startswith("https://"):
                            url = token.strip().rstrip("/")
                            break
                if not url:
                    for token in (out + "\n" + out2).split():
                        if token.startswith("http://") or token.startswith("https://"):
                            url = token.strip().rstrip("/")
                            break
                if not url:
                    raise RuntimeError("Failed to resolve public URL from modal CLI output")
        if not url:
            print("Failed to determine Task App URL")
            return 2
        demo_core.persist_task_url(url, name=app_name or None)
        dotenv_values = {"TASK_APP_BASE_URL": url}
        if app_name:
            dotenv_values["TASK_APP_NAME"] = app_name
            dotenv_values["TASK_APP_SECRET_NAME"] = f"{app_name}-secret"
        dotenv_path = demo_core.persist_dotenv_values(dotenv_values)
        print(f"TASK_APP_BASE_URL={url}")
        if app_name:
            print(f"TASK_APP_NAME={app_name}")
        print("Export for this shell:")
        print(f"  export TASK_APP_BASE_URL={url}")
        if app_name:
            print(f"  export TASK_APP_NAME={app_name}")
            print(f"  export TASK_APP_SECRET_NAME={app_name}-secret")
        print(f"Persisted to {dotenv_path}")
        print("Next: uvx synth-ai rl_demo configure")
        return 0
    except Exception as e:
        print(f"Deploy error: {e}")
        return 2


def cmd_configure(args: argparse.Namespace) -> int:
    from synth_ai.rl.secrets import mint_environment_api_key

    env = demo_core.load_env()
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)

    synth_key = env.synth_api_key.strip()
    if not synth_key:
        synth_key = input("Enter SYNTH_API_KEY (required): ").strip()
        if not synth_key:
            print("SYNTH_API_KEY is required.")
            return 1
        demo_core.persist_api_key(synth_key)
    demo_core.persist_dotenv_values({"SYNTH_API_KEY": synth_key})

    env_key = env.env_api_key.strip()
    minted_env_key = False
    if not env_key:
        env_key = mint_environment_api_key()
        minted_env_key = True
        print("Minted new ENVIRONMENT_API_KEY")
    demo_core.persist_env_api_key(env_key)
    demo_core.persist_dotenv_values({"ENVIRONMENT_API_KEY": env_key})

    task_url = env.task_app_base_url
    if not task_url:
        print("Task app URL missing. Run: uvx synth-ai rl_demo deploy")
        return 1

    app_name = env.task_app_name.strip()
    if not app_name:
        fallback = input("Enter Modal app name for the task app (required): ").strip()
        if not fallback:
            print("Task app name is required to configure Modal secrets.")
            return 1
        app_name = fallback
        demo_core.persist_task_url(task_url, name=app_name)

    secret_name = env.task_app_secret_name.strip() or f"{app_name}-secret"
    demo_core.persist_task_url(task_url, name=app_name)
    demo_core.persist_dotenv_values({
        "TASK_APP_BASE_URL": task_url,
        "TASK_APP_NAME": app_name,
        "TASK_APP_SECRET_NAME": secret_name,
    })

    # Ensure Modal secret has the environment API key (and optional extras).
    secret_args = [f"ENVIRONMENT_API_KEY={env_key}"]
    openai_key = (os.environ.get("OPENAI_API_KEY") or local_env.get("OPENAI_API_KEY") or "").strip()
    if openai_key:
        secret_args.append(f"OPENAI_API_KEY={openai_key}")
    synth_for_secret = synth_key
    if synth_for_secret:
        secret_args.append(f"SYNTH_API_KEY={synth_for_secret}")

    create_cmd = ["uv", "run", "modal", "secret", "create", secret_name, *secret_args]
    code, out = _popen_capture(create_cmd)
    if code != 0:
        print(out)
        print("Secret create failed; retrying with delete → create…")
        _popen_capture(["uv", "run", "modal", "secret", "delete", secret_name])
        code, out = _popen_capture(create_cmd)
        if code != 0:
            print(out)
            print("Failed to provision Modal secret.")
            return 2

    # Verify task app can read the secret by hitting rollout health with X-API-Key.
    rollout_url = task_url.rstrip("/") + "/health/rollout"
    rc, body = _http("GET", rollout_url, headers={"X-API-Key": env_key})
    if rc != 200:
        print(f"Warning: rollout health check failed ({rc}). Response: {body}")
    else:
        print("Task app rollout health check OK.")

    env.synth_api_key = synth_key
    env.env_api_key = env_key
    env.task_app_name = app_name
    env.task_app_secret_name = secret_name

    # Prepare a baseline TOML (formerly `prepare`): prompt and write demo_config.toml
    defaults = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_task_apps", "math", "config.toml")),
    ]
    mono = "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/math_online.toml"
    if os.path.isfile(mono):
        defaults.append(mono)
    print("Select a baseline TOML:")
    for i, p in enumerate(defaults, 1):
        print(f"  [{i}] {p}")
    choice = input(f"Enter choice [1-{len(defaults)}] (default 1): ").strip() or "1"
    try:
        idx = max(1, min(int(choice), len(defaults))) - 1
    except Exception:
        idx = 0
    base_path = defaults[idx]
    with open(base_path, "r") as fh:
        text = fh.read()
    import re
    # Extract current defaults from the selected TOML
    def _extract(pattern: str, default: str) -> str:
        m = re.search(pattern, text, flags=re.M)
        if not m:
            return default
        val = (m.group(1) or "").strip()
        return val if val else default
    current_gpu_type = _extract(r"^gpu_type\s*=\s*\"([^\"]+)\"$", "A100")
    # topology form gpu_type = "TYPE:COUNT" also supported for deriving defaults
    topo_gpu = _extract(r"^gpu_type\s*=\s*\"([^\":]+):(\d+)\"$", current_gpu_type)
    if ":" in topo_gpu:
        current_gpu_type = topo_gpu.split(":", 1)[0]
    current_gpu_count = _extract(r"^gpu_count\s*=\s*(\d+)$", "4")
    if ":" in topo_gpu:
        current_gpu_count = topo_gpu.split(":", 1)[1]
    current_model = _extract(r"^name\s*=\s*\"([^\"]+)\"$", "Qwen/Qwen3-0.6B")
    current_tp = _extract(r"^tensor_parallel_size\s*=\s*(\d+)$", "2")

    # Prompts with defaults shown; Enter keeps current
    def _prompt(label: str, default_val: str) -> str:
        entered = input(f"{label} [{default_val}]: ").strip()
        return entered or default_val

    gpu_type = _prompt("GPU type", current_gpu_type)
    gpu_count = _prompt("GPU count", current_gpu_count)
    model = _prompt("Model", current_model)
    tp = _prompt("Tensor parallel", current_tp)

    text = re.sub(r"(?m)^gpu_type\s*=\s*\".*?\"$", f"gpu_type = \"{gpu_type}\"", text)
    text = re.sub(r"(?m)^gpu_count\s*=\s*\d+$", f"gpu_count = {int(gpu_count)}", text)
    text = re.sub(r"(?m)^name\s*=\s*\".*?\"$", f"name = \"{model}\"", text)
    text = re.sub(r"(?m)^tensor_parallel_size\s*=\s*\d+$", f"tensor_parallel_size = {int(tp)}", text)
    text = re.sub(r"(?m)^gpu_type\s*=\s*\".*?:\d+\"$", f"gpu_type = \"{gpu_type}:{int(gpu_count)}\"", text)
    out_path = os.path.abspath(os.path.join(os.getcwd(), "demo_config.toml"))
    _write_text(out_path, text)
    print(f"Prepared config at: {out_path}")
    here_cfg = os.path.abspath(out_path)
    print("Config path:", here_cfg)
    print("Environment (masked):")
    print(json.dumps({
        "DEV_BACKEND_URL": env.dev_backend_url,
        "SYNTH_API_KEY": (synth_key[:6] + "…") if synth_key else "",
        "ENVIRONMENT_API_KEY": (env_key[:6] + "…") if env_key else "",
        "TASK_APP_BASE_URL": task_url,
        "TASK_APP_NAME": app_name,
        "TASK_APP_SECRET_NAME": secret_name,
    }, indent=2))
    if minted_env_key:
        print(f"Stored minted ENVIRONMENT_API_KEY in {cwd_env_path}")
    print("Next: uvx synth-ai rl_demo run")
    return 0


def _http(method: str, url: str, headers: Dict[str, str] | None = None, body: Dict[str, Any] | None = None) -> tuple[int, Dict[str, Any] | str]:
    import urllib.request, urllib.error, json as _json
    data = None
    if body is not None:
        data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            code = getattr(resp, "status", 200)
            txt = resp.read().decode("utf-8", errors="ignore")
            try:
                return int(code), _json.loads(txt)
            except Exception:
                return int(code), txt
    except urllib.error.HTTPError as he:  # Capture 4xx/5xx bodies
        txt = he.read().decode("utf-8", errors="ignore")
        try:
            return int(he.code or 0), _json.loads(txt)
        except Exception:
            return int(he.code or 0), txt
    except Exception as e:
        return 0, str(e)


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)


# Note: `prepare` command has been removed; configuration now prepares TOML


def cmd_run(args: argparse.Namespace) -> int:
    env = demo_core.load_env()
    # Prompt for missing SYNTH_API_KEY
    if not env.synth_api_key:
        entered = input("Enter SYNTH_API_KEY (required): ").strip()
        if not entered:
            print("SYNTH_API_KEY is required.")
            return 1
        os.environ["SYNTH_API_KEY"] = entered
        demo_core.persist_api_key(entered)
        demo_core.persist_dotenv_values({"SYNTH_API_KEY": entered})
    # Re-resolve env after potential persist
    env = demo_core.load_env()
    if not env.task_app_base_url:
        print("Task app URL missing. Run: uvx synth-ai rl_demo deploy")
        return 1
    if not env.dev_backend_url:
        print("Backend URL missing. Set DEV_BACKEND_URL in a .env or rely on default prod.")
        return 1
    if not env.env_api_key:
        print("ENVIRONMENT_API_KEY missing. Run: uvx synth-ai rl_demo configure")
        return 1
    os.environ["ENVIRONMENT_API_KEY"] = env.env_api_key

    # Detect monorepo launcher and delegate if available (aligns with run_clustered.sh which works)
    launcher = "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/start_math_clustered.py"
    if os.path.isfile(launcher):
        backend_base = env.dev_backend_url[:-4] if env.dev_backend_url.endswith("/api") else env.dev_backend_url
        run_env = os.environ.copy()
        run_env["BACKEND_URL"] = backend_base
        run_env["SYNTH_API_KEY"] = env.synth_api_key
        run_env["TASK_APP_BASE_URL"] = env.task_app_base_url
        run_env["ENVIRONMENT_API_KEY"] = env.env_api_key
        # Optional: TRAINER_START_URL passthrough if already set in environment
        run_env["TRAINER_START_URL"] = run_env.get("TRAINER_START_URL", "")
        # Forward convenience knobs
        if args.batch_size is not None:
            run_env["RL_BATCH_SIZE"] = str(int(args.batch_size))
        if args.group_size is not None:
            run_env["RL_GROUP_SIZE"] = str(int(args.group_size))
        if args.model:
            run_env["RL_MODEL"] = args.model
        cmd = ["uv", "run", "python", launcher]
        print(f"Launching monorepo clustered runner: {' '.join(cmd)}")
        code = _popen_stream(cmd, env=run_env)
        if code != 0:
            print(f"Clustered runner exited with code {code}")
        return code

    # Fallback: legacy jobs API flow
    import tomllib
    # Determine config path: --config overrides; otherwise prompt from detected candidates
    cfg_path = None
    if getattr(args, "config", None):
        cfg_path = os.path.abspath(args.config)
        if not os.path.isfile(cfg_path):
            print(f"Config not found: {cfg_path}")
            return 1
    else:
        candidates: list[str] = []
        # Prepared in CWD and home
        cwd_prepared = os.path.abspath(os.path.join(os.getcwd(), "demo_config.toml"))
        home_prepared = os.path.expanduser("~/.synth-ai/demo_config.toml")
        if os.path.isfile(cwd_prepared):
            candidates.append(cwd_prepared)
        if os.path.isfile(home_prepared):
            candidates.append(home_prepared)
        # Monorepo math_online.toml if present
        mono = "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/math_online.toml"
        if os.path.isfile(mono):
            candidates.append(mono)
        # Packaged default
        packaged = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_task_apps", "math", "config.toml"))
        candidates.append(packaged)
        # Deduplicate while preserving order
        seen = set()
        uniq: list[str] = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        print("Choose a TOML config:")
        for i, p in enumerate(uniq, 1):
            print(f"  [{i}] {p}")
        sel = input(f"Enter choice [1-{len(uniq)}] (default 1): ").strip() or "1"
        try:
            idx = max(1, min(int(sel), len(uniq))) - 1
        except Exception:
            idx = 0
        cfg_path = uniq[idx]
    with open(cfg_path, "rb") as fh:
        inline_cfg = tomllib.load(fh)
    with open(cfg_path, "r") as fh2:
        toml_text = fh2.read()
    if args.batch_size is not None:
        inline_cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.group_size is not None:
        inline_cfg.setdefault("training", {})["group_size"] = int(args.group_size)
    model_name = args.model or (inline_cfg.get("model", {}) or {}).get("name", "Qwen/Qwen3-0.6B")
    api = env.dev_backend_url.rstrip("/") + ("" if env.dev_backend_url.endswith("/api") else "/api")
    data_fragment: Dict[str, Any] = {
        "model": model_name,
        "endpoint_base_url": env.task_app_base_url,
        "config": inline_cfg,
        "config_toml": toml_text,
        "config_source": "toml_inline",
        "metadata": {"source": "synth-ai rl_demo", "cwd": os.getcwd()},
    }
    if env.env_api_key:
        data_fragment["environment_api_key"] = env.env_api_key
    for k in ("training", "evaluation", "rollout", "topology", "vllm"):
        if isinstance(inline_cfg.get(k), dict):
            data_fragment[k] = inline_cfg[k]
    compute = {}
    if isinstance(inline_cfg.get("compute"), dict):
        if inline_cfg["compute"].get("gpu_type"):
            compute["gpu_type"] = str(inline_cfg["compute"]["gpu_type"]).upper()
        if inline_cfg["compute"].get("gpu_count"):
            compute["gpu_count"] = int(inline_cfg["compute"]["gpu_count"]) 
    if not compute:
        topo = inline_cfg.get("topology") or {}
        gshape = str(topo.get("gpu_type") or "")
        if ":" in gshape:
            t, c = gshape.split(":", 1)
            compute = {"gpu_type": t.upper(), "gpu_count": int(c)}
    body: Dict[str, Any] = {
        "job_type": "rl",
        "data": data_fragment,
    }
    if compute:
        body["compute"] = compute
    code, js = _http("POST", api + "/rl/jobs", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {env.synth_api_key}",
    }, body=body)
    if code not in (200, 201) or not isinstance(js, dict):
        print("Job create failed:", code)
        try:
            if isinstance(js, dict):
                print(json.dumps(js, indent=2))
            else:
                print(str(js))
        except Exception:
            print(str(js))
        print("Request body was:\n" + json.dumps(body, indent=2))
        return 2
    job_id = js.get("job_id") or js.get("id") or ""
    if not job_id:
        print("Job id missing in response:", js)
        print("Request body was:\n" + json.dumps(body, indent=2))
        return 2
    print("JOB_ID:", job_id)
    _http("POST", api + f"/rl/jobs/{job_id}/start", headers={"Authorization": f"Bearer {env.synth_api_key}"})
    since = 0
    terminal = {"succeeded", "failed", "cancelled", "error", "completed"}
    last_status = ""
    start_t = time.time()
    while True:
        sc, sj = _http("GET", api + f"/learning/jobs/{job_id}")
        status = (sj.get("status") if isinstance(sj, dict) else "") if sc == 200 else ""
        if status and status != last_status:
            last_status = status
            print("status →", status)
        if status and status.lower() in terminal:
            print("FINAL:", status)
            break
        ec, ej = _http("GET", api + f"/orchestration/jobs/{job_id}/events?since_seq={since}&limit=200")
        if ec == 200 and isinstance(ej, dict):
            events = ej.get("events") or ej.get("data") or []
            for e in events:
                seq = int(e.get("seq") or 0)
                if seq <= since:
                    continue
                since = seq
                typ = str(e.get("type") or e.get("event_type") or "").lower()
                msg = e.get("message") or e.get("msg") or ""
                if typ in ("rl.eval.started", "rl.eval.summary", "rl.train.step", "rl.metrics", "rl.performance.metrics"):
                    print(f"[{seq}] {typ}: {msg}")
        mc, mj = _http("GET", api + f"/learning/jobs/{job_id}/metrics?after_step=-1&limit=50")
        if mc == 200 and isinstance(mj, dict):
            pts = mj.get("points") or []
            for p in pts:
                name = p.get("name")
                if name == "eval.reward_mean":
                    print(f"metric eval.reward_mean step={p.get('step')} value={p.get('value')}")
                    break
        if time.time() - start_t > (args.timeout or 600):
            print("Timeout waiting for terminal state.")
            break
        time.sleep(2)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="synth-ai")
    sub = p.add_subparsers(dest="cmd")

    def _add_parser(names: list[str], *, configure: Callable[[argparse.ArgumentParser], None]) -> None:
        for name in names:
            parser = sub.add_parser(name)
            configure(parser)

    _add_parser(["rl_demo.check", "demo.check"], configure=lambda parser: parser.set_defaults(func=cmd_check))

    # (prepare command removed)

    def _deploy_opts(parser):
        parser.add_argument("--local", action="store_true", help="Run local FastAPI instead of Modal deploy")
        parser.add_argument("--app", type=str, default=None, help="Path to Modal app.py for uv run modal deploy")
        parser.add_argument("--name", type=str, default="synth-math-demo", help="Modal app name")
        parser.add_argument("--script", type=str, default=None, help="Path to deploy_task_app.sh (optional legacy)")
        parser.set_defaults(func=cmd_deploy)

    _add_parser(["rl_demo.deploy", "demo.deploy"], configure=_deploy_opts)

    _add_parser(["rl_demo.configure", "demo.configure"], configure=lambda parser: parser.set_defaults(func=cmd_configure))

    def _run_opts(parser):
        parser.add_argument("--config", type=str, default=None, help="Path to TOML config (skip prompt)")
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--group-size", type=int, default=None)
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--timeout", type=int, default=600)
        parser.add_argument("--dry-run", action="store_true", help="Print request body and exit")
        parser.set_defaults(func=cmd_run)

    _add_parser(["rl_demo.run", "demo.run"], configure=_run_opts)

    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
