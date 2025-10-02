from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Callable
import shutil
import stat
import textwrap

from synth_ai.demos.demo_task_apps import core as demo_core
from synth_ai.handshake import run_handshake, HandshakeError
from synth_ai.demos.demo_task_apps.core import DemoEnv


def _is_modal_public_url(u: str) -> bool:
    try:
        s = (u or "").strip().lower()
        if not (s.startswith("http://") or s.startswith("https://")):
            return False
        return (".modal.run" in s) and ("modal.local" not in s) and ("pypi-mirror" not in s)
    except Exception:
        return False


def cmd_setup(_args: argparse.Namespace) -> int:
    # 1) Always perform SDK handshake and overwrite .env with returned keys
    try:
        print("\n⏳ Connecting SDK to your browser session…")
        res = run_handshake()
        user = res.get("user") or {}
        org = res.get("org") or {}
        keys = res.get("keys") or {}
        synth_key = str(keys.get("synth") or "").strip()
        rl_env_key = str(keys.get("rl_env") or "").strip()
        if not synth_key or not rl_env_key:
            raise HandshakeError("handshake returned missing keys")
        # Overwrite .env with the latest values from the account/org
        demo_core.persist_dotenv_values({
            "SYNTH_API_KEY": synth_key,
            "ENVIRONMENT_API_KEY": rl_env_key,
        })
        org_name = (org.get("name") or "this organization")
        print(f"✅ Connected to {org_name}!")
    except HandshakeError as e:
        print(f"Handshake failed: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected handshake error: {e}")
        return 1

    # 2) Reload env after handshake to pick up values from .env (suppress env prints)
    import io
    import contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
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
        elif not _is_modal_public_url(current):
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
            if _is_modal_public_url(token):
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

    # Keys have been written already via handshake; avoid any interactive prompts
    synth_key = env.synth_api_key.strip()
    if not local_env.get("SYNTH_API_KEY") and synth_key:
        demo_core.persist_dotenv_values({"SYNTH_API_KEY": synth_key})
        _refresh_env()

    # Check Modal auth silently to avoid noisy output
    modal_ok, modal_msg = demo_core.modal_auth_status()

    _maybe_fix_task_url()

    ok_backend = False
    ok_task = False
    if env.dev_backend_url:
        api = env.dev_backend_url.rstrip("/") + ("" if env.dev_backend_url.endswith("/api") else "/api")
        ok_backend = demo_core.assert_http_ok(api + "/health", method="GET")
        # Intentionally suppress backend health print for concise output
    if env.task_app_base_url:
        ok_task = demo_core.assert_http_ok(env.task_app_base_url.rstrip("/") + "/health", method="GET") or \
                  demo_core.assert_http_ok(env.task_app_base_url.rstrip("/"), method="GET")
        # Intentionally suppress task app health print
    else:
        print("\nSet your task app URL by running:\nuvx synth-ai rl_demo deploy\n")

    # Omit uv version print to keep output concise

    # Keep exit code neutral; not all checks are critical for pairing
    return 0


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


def _popen_stream_capture(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> tuple[int, str]:
    """Stream subprocess output to stdout and also capture it into a buffer."""
    import subprocess
    import threading

    buf_lines: list[str] = []
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
        return 1, ""

    def _pump(stdout) -> None:
        try:
            for line in stdout:
                line = line.rstrip()
                print(line)
                buf_lines.append(line)
        except Exception:
            pass

    if proc.stdout is not None:
        t = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=1.0)
    else:
        proc.wait()
    return int(proc.returncode or 0), "\n".join(buf_lines)


def _mask_secret_args(args: list[str]) -> list[str]:
    masked: list[str] = []
    for a in args:
        if "=" in a and any(a.startswith(prefix) for prefix in ("ENVIRONMENT_API_KEY=", "OPENAI_API_KEY=", "SYNTH_API_KEY=")):
            try:
                key, value = a.split("=", 1)
                tail = value[-5:] if len(value) >= 5 else value
                masked.append(f"{key}=***{tail}")
            except Exception:
                masked.append("<masked>")
        else:
            masked.append(a)
    return masked


def _ensure_modal_secret(
    secret_name: str,
    *,
    values: dict[str, str],
    label: str = "deploy",
    replace: bool = False,
) -> bool:
    prefix = f"[{label}]"
    if not secret_name.strip():
        raise RuntimeError("Secret name is required")

    if not values:
        raise RuntimeError("No values provided to create Modal secret")

    create_args = [f"{k}={v}" for k, v in values.items()]
    create_cmd = ["uv", "run", "modal", "secret", "create", secret_name, *create_args]

    if replace:
        print(f"{prefix} Removing Modal secret '{secret_name}' (if present)…")
        delete_cmd = ["bash", "-lc", f"printf 'y\\n' | uv run modal secret delete {secret_name}"]
        print(f"{prefix} Command:", " ".join(delete_cmd))
        delete_code = _popen_stream(delete_cmd)
        if delete_code != 0:
            print(f"{prefix} Warning: delete command exited with {delete_code}; continuing to create")

    print(f"\n{prefix} Creating Modal secret '{secret_name}'…")
    print(f"{prefix} Command:", " ".join(_mask_secret_args(create_cmd)))
    code = _popen_stream(create_cmd)
    if code != 0:
        raise RuntimeError("Failed to provision Modal secret (see logs above)")

    return True


def _fmt_float(value: float) -> str:
    return f"{value:.10g}"


def _find_asgi_apps(root: Path) -> list[Path]:
    """Recursively search for Python files that declare a Modal ASGI app.

    A file is considered a Modal task app candidate if it contains one of:
      - "@asgi_app()"
      - "@modal.asgi_app()"
    """
    results: list[Path] = []
    skip_dirs = {".git", ".hg", ".svn", "node_modules", "dist", "build", "__pycache__", ".ruff_cache", ".mypy_cache", "venv", ".venv"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                if ("@asgi_app()" in txt) or ("@modal.asgi_app()" in txt):
                    results.append(path)
            except Exception:
                continue
    # Stable order: prioritize files under synth_demo/ first, then alphabetical
    def _priority(p: Path) -> tuple[int, str]:
        rel = str(p.resolve())
        in_demo = "/synth_demo/" in rel or rel.endswith("/synth_demo/task_app.py")
        return (0 if in_demo else 1, rel)
    results.sort(key=_priority)
    return results


def _prompt_value(label: str, default: str | int | float, cast: Callable[[str], Any] | None = None) -> Any:
    prompt = f"{label} [{default}]: "
    try:
        raw = input(prompt).strip()
    except Exception:
        raw = ""
    if not raw:
        return default
    if cast is None:
        return raw
    try:
        return cast(raw)
    except Exception:
        print(f"Invalid value; keeping default {default}")
        return default


def _find_vllm_tomls(root: Path) -> list[Path]:
    results: list[Path] = []
    skip_dirs = {".git", ".hg", ".svn", "node_modules", "dist", "build", "__pycache__", ".ruff_cache", ".mypy_cache", "venv", ".venv"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not name.endswith(".toml"):
                continue
            path = Path(dirpath) / name
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    if "[vllm]" in fh.read().lower():
                        results.append(path)
            except Exception:
                continue
    return results


def _create_new_config(env: DemoEnv) -> str:
    default_path = os.path.join(os.getcwd(), "demo_config.toml")
    while True:
        try:
            destination = input(f"Path to save new config [{default_path}]: ").strip() or default_path
        except Exception:
            destination = default_path
        destination = os.path.abspath(destination)
        if os.path.isdir(destination):
            print("Path points to a directory; provide a file path.")
            continue
        if os.path.exists(destination):
            try:
                overwrite = input(f"{destination} exists. Overwrite? [y/N]: ").strip().lower() or "n"
            except Exception:
                overwrite = "n"
            if not overwrite.startswith("y"):
                continue
        break

    env_name = _prompt_value("Environment name", "Crafter")
    policy_name = _prompt_value("Policy name", "crafter-react")
    model_name = _prompt_value("Model name", "Qwen/Qwen3-0.6B")
    compute_gpu_type = _prompt_value("Compute GPU type", "H100")
    compute_gpu_count = _prompt_value("Compute GPU count", 4, int)
    topology_gpu_type = _prompt_value("Topology GPU type", f"{compute_gpu_type}:{compute_gpu_count}")
    gpus_for_vllm = _prompt_value("Topology gpus_for_vllm", 2, int)
    gpus_for_training = _prompt_value("Topology gpus_for_training", 1, int)
    tensor_parallel = _prompt_value("Topology tensor_parallel", 2, int)
    gpus_for_ref = _prompt_value("Topology gpus_for_ref", 1, int)
    vllm_tp_size = _prompt_value("vLLM tensor parallel size", tensor_parallel, int)
    vllm_max_model_len = _prompt_value("vLLM max_model_len", 8192, int)
    vllm_max_num_seqs = _prompt_value("vLLM max_num_seqs", 32, int)
    vllm_gpu_mem_util = _prompt_value("vLLM gpu_memory_utilization", 0.9, float)
    vllm_max_parallel = _prompt_value("vLLM max_parallel_generations", 4, int)
    training_num_epochs = _prompt_value("Training num_epochs", 1, int)
    training_iters = _prompt_value("Training iterations_per_epoch", 2, int)
    training_batch = _prompt_value("Training batch_size", 1, int)
    training_group = _prompt_value("Training group_size", 8, int)
    training_lr = _prompt_value("Training learning_rate", 5e-6, float)
    task_url_default = env.task_app_base_url or ""
    services_task_url = _prompt_value("services.task_url", task_url_default)

    template = textwrap.dedent(
        f"""\
        # Crafter online RL training configuration (research local copy)

        [model]
        #name = \"fft:Qwen/Qwen3-4B:job_7243b8aa76fe4b59\"
        name = \"{model_name}\"
        dtype = \"bfloat16\"
        seed = 42
        trainer_mode = \"full\"

        [lora]
        r = 16
        alpha = 32
        dropout = 0.05
        target_modules = [
          \"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",
          \"gate_proj\", \"up_proj\", \"down_proj\",
        ]

        [rdma]
        enabled = false
        ifname = \"eth0\"
        ip_type = \"ipv4\"
        p2p_disable = 0
        shm_disable = 0
        fast_nccl = false

        gid_index = 3
        cross_nic = 0
        collnet_enable = 0
        net_gdr_level = 2

        nsocks_perthread = 4
        socket_nthreads = 2

        algo = \"Ring\"
        proto = \"Simple\"
        p2p_level = \"SYS\"
        debug = \"INFO\"

        [compute]
        gpu_type = \"{compute_gpu_type}\"
        gpu_count = {compute_gpu_count}

        [topology]
        type = \"single_node_split\"
        gpu_type = \"{topology_gpu_type}\"
        use_rdma = false
        gpus_for_vllm = {gpus_for_vllm}
        gpus_for_training = {gpus_for_training}
        tensor_parallel = {tensor_parallel}
        gpus_for_ref = {gpus_for_ref}

        [vllm]
        tensor_parallel_size = {vllm_tp_size}
        gpu_memory_utilization = {_fmt_float(vllm_gpu_mem_util)}
        max_model_len = {vllm_max_model_len}
        max_num_seqs = {vllm_max_num_seqs}
        enforce_eager = false
        max_parallel_generations = {vllm_max_parallel}

        # Reference scoring server (dedicated GPU)
        [reference]
        placement = \"dedicated\"
        gpu_index = 1
        port = 8002
        tp = 1
        health_max_wait_s = 180
        health_interval_ms = 300

        [training]
        num_epochs = {training_num_epochs}
        iterations_per_epoch = {training_iters}
        batch_size = {training_batch}
        group_size = {training_group}
        learning_rate = {_fmt_float(training_lr)}
        max_grad_norm = 0.5
        log_interval = 1
        update_reference_interval = 0
        weight_sync_interval = 1

        [training.weight_sync]
        enable = true
        targets = [\"policy\"]

        [rollout]
        env_name = \"{env_name}\"
        policy_name = \"{policy_name}\"
        env_config = {{}}
        max_steps_per_episode = 5
        sampling_temperature = 0.3
        sampling_top_p = 0.95
        max_tokens = 1024
        max_concurrent_rollouts = 4
        ops_per_rollout = 14
        on_done = \"reset\"
        thinking_mode = \"think\"
        thinking_budget = 512

        [policy]
        config = {{}}

        [evaluation]
        seeds = [0, 1, 2, 3, 4, 5, 6, 7]
        rollouts_per_seed = 1
        instances = 0
        max_concurrent_rollouts = 4
        thinking_mode = \"think\"
        every_n_iters = 5

        [hyperparams]
        epsilon_low = 0.1
        epsilon_high = 0.3
        delta = 5.0
        beta = 0.01
        kl_penalty = 0.01
        advantage_normalization = true
        group_normalization = true
        num_inner_steps = 1
        clip_epsilon = 0.2
        completion_only = false

        [step_rewards]
        enabled = false
        mode = \"off\"
        step_beta = 0.0
        indicator_lambda = 0.0

        [trainer]
        allow_ref_fallback = false

        [checkpoint]
        interval = 10
        directory = \"/checkpoints\"
        keep_last_n = 3
        save_optimizer = true
        save_scheduler = true
        enabled = true

        [services]
        task_url = \"{services_task_url}\"
        """
    ).strip() + "\n"

    with open(destination, "w", encoding="utf-8") as fh:
        fh.write(template)
    print(f"Wrote config to {destination}")
    return destination


def _select_or_create_config(explicit: str | None, env: DemoEnv) -> str:
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    search_root = Path(os.getcwd())
    discovered = _find_vllm_tomls(search_root)

    extras: list[Path] = []
    packaged = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_task_apps", "math", "config.toml")))
    extras.append(packaged)
    home_cfg = Path(os.path.expanduser("~/.synth-ai/demo_config.toml"))
    extras.append(home_cfg)

    all_paths: list[Path] = []
    seen: set[str] = set()
    for candidate in discovered + extras:
        if candidate.is_file():
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                all_paths.append(candidate)

    if not all_paths:
        print("No existing RL TOML configs with [vllm] found; creating a new one.")
        return _create_new_config(env)

    print("Select a TOML config (found [vllm] section):")
    for idx, path in enumerate(all_paths, 1):
        rel = os.path.relpath(str(path), os.getcwd())
        print(f"  [{idx}] {rel}")
    create_idx = len(all_paths) + 1
    print(f"  [{create_idx}] Create new config")
    try:
        sel = input(f"Enter choice [1-{create_idx}] (default 1): ").strip() or "1"
    except Exception:
        sel = "1"
    try:
        choice = int(sel)
    except Exception:
        choice = 1
    if choice == create_idx:
        return _create_new_config(env)
    choice = max(1, min(choice, len(all_paths)))
    selected = os.path.abspath(all_paths[choice - 1])
    print(f"Using config: {selected}")
    return selected


def _ensure_task_app_ready(env: DemoEnv, synth_key: str, *, label: str) -> DemoEnv:
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)

    env_key = (env.env_api_key or "").strip()
    if not env_key:
        raise RuntimeError(f"[{label}] ENVIRONMENT_API_KEY missing. Run `uvx synth-ai rl_demo deploy` first.")

    task_url = env.task_app_base_url
    if not task_url or not _is_modal_public_url(task_url):
        resolved = ""
        if env.task_app_name:
            try:
                choice = input(
                    f"Resolve URL from Modal for app '{env.task_app_name}'? [Y/n]: "
                ).strip().lower() or "y"
            except Exception:
                choice = "y"
            if choice.startswith("y"):
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
                if code == 0 and out:
                    for tok in out.split():
                        if _is_modal_public_url(tok):
                            resolved = tok.strip().rstrip("/")
                            break
        if not resolved:
            print(f"[{label}] Task app URL not configured or not a valid Modal public URL.")
            print("Examples: https://<app-name>-fastapi-app.modal.run")
            entered = input("Enter Task App base URL (must contain '.modal.run'), or press Enter to abort: ").strip()
            if not entered or not _is_modal_public_url(entered):
                raise RuntimeError(f"[{label}] Valid Task App URL is required.")
            task_url = entered.rstrip("/")
        else:
            task_url = resolved
        demo_core.persist_task_url(task_url, name=(env.task_app_name or None))

    app_name = env.task_app_name.strip()
    if not app_name:
        fallback = input("Enter Modal app name for the task app (required): ").strip()
        if not fallback:
            raise RuntimeError(f"[{label}] Task app name is required.")
        app_name = fallback
        demo_core.persist_task_url(task_url, name=app_name)

    secret_name = env.task_app_secret_name.strip() or f"{app_name}-secret"
    demo_core.persist_task_url(task_url, name=app_name)
    demo_core.persist_dotenv_values({
        "TASK_APP_BASE_URL": task_url,
        "TASK_APP_NAME": app_name,
        "TASK_APP_SECRET_NAME": secret_name,
    })

    openai_key = (os.environ.get("OPENAI_API_KEY") or local_env.get("OPENAI_API_KEY") or "").strip()
    secret_values: dict[str, str] = {"ENVIRONMENT_API_KEY": env_key}
    if openai_key:
        secret_values["OPENAI_API_KEY"] = openai_key
    if synth_key:
        secret_values["SYNTH_API_KEY"] = synth_key

    _ensure_modal_secret(secret_name, values=secret_values, label=label, replace=True)

    rollout_url = task_url.rstrip("/") + "/health/rollout"
    print(f"[{label}] Verifying rollout health:")
    try:
        ek = (env_key or "").strip()
        ek_len = len(ek)
        ek_tail = ek[-5:] if ek_len >= 5 else ek
        print(f"[{label}] Using ENVIRONMENT_API_KEY len={ek_len} last5={ek_tail}")
    except Exception:
        pass
    health_base = task_url.rstrip("/")
    health_urls = [f"{health_base}/health/rollout", f"{health_base}/health"]
    rc = 0
    body: Any = ""
    for h in health_urls:
        print(f"[{label}] GET", h)
        rc, body = _http("GET", h, headers={"X-API-Key": env_key})
        if rc == 200:
            rollout_url = h
            break
    print(f"[{label}] status: {rc}")
    try:
        import json as _json

        preview = _json.dumps(body)[:800] if isinstance(body, dict) else str(body)[:800]
    except Exception:
        preview = str(body)[:800]
    print(f"[{label}] body:", preview)
    if rc != 200:
        print(f"[{label}] Warning: rollout health check failed ({rc}). Response: {body}")
    else:
        print(f"[{label}] Task app rollout health check OK.")

    os.environ["TASK_APP_BASE_URL"] = task_url
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    updated_env = demo_core.load_env()
    updated_env.env_api_key = env_key
    updated_env.task_app_base_url = task_url
    updated_env.task_app_name = app_name
    updated_env.task_app_secret_name = secret_name
    return updated_env


def cmd_deploy(args: argparse.Namespace) -> int:
    env = demo_core.load_env()
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)
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
            # Auto-detect app path if not supplied; prompt interactively from discovered ASGI apps
            app_path = os.path.abspath(args.app) if args.app else None
            if not app_path or not os.path.isfile(app_path):
                # First pass: look for known common filenames
                candidates = [
                    os.path.abspath(os.path.join(os.getcwd(), "synth_demo", "task_app.py")),
                    os.path.abspath(os.path.join(os.getcwd(), "task_app.py")),
                    os.path.abspath(os.path.join(os.getcwd(), "app.py")),
                    os.path.abspath(os.path.join(os.getcwd(), "math_task_app.py")),
                ]
                app_path = next((p for p in candidates if os.path.isfile(p)), None)
                # If still not found, scan for any file containing @asgi_app()
                if not app_path:
                    found = _find_asgi_apps(Path(os.getcwd()))
                    if found:
                        print("Select a Modal ASGI app to deploy:")
                        for idx, pth in enumerate(found, 1):
                            rel = os.path.relpath(str(pth), os.getcwd())
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
            if not app_path and args.script:
                # Legacy script fallback if user supplied --script explicitly
                from synth_ai.demos.demo_task_apps.math.deploy_modal import deploy as modal_deploy
                url = modal_deploy(script_path=args.script, env_api_key=env.env_api_key)
                if args.name:
                    app_name = args.name
            else:
                if not app_path:
                    entered = input("Path to Modal app.py (e.g., ./task_app.py): ").strip()
                    if not entered:
                        raise FileNotFoundError("No app.py path provided and auto-detect failed")
                    app_path = os.path.abspath(entered)
                if not os.path.isfile(app_path):
                    raise FileNotFoundError(f"App file not found: {app_path}")
                # Surface the app path before asking for the name
                print(f"Using task app: {app_path}")
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

                secret_name = (env.task_app_secret_name or "").strip() or f"{name_in}-secret"
                existing_env_key = (env.env_api_key or "").strip()
                env_key: str | None = existing_env_key or None
                if existing_env_key:
                    try:
                        reuse_choice = input(
                            "Use existing ENVIRONMENT_API_KEY from state/.env? [Y/n]: "
                        ).strip().lower() or "y"
                    except Exception:
                        reuse_choice = "y"
                    if not reuse_choice.startswith("y"):
                        env_key = None

                if env_key is None:
                    from synth_ai.rl.secrets import mint_environment_api_key

                    env_key = mint_environment_api_key()
                    demo_core.persist_env_api_key(env_key)
                    demo_core.persist_dotenv_values({"ENVIRONMENT_API_KEY": env_key})
                    os.environ["ENVIRONMENT_API_KEY"] = env_key
                    env.env_api_key = env_key
                    local_env["ENVIRONMENT_API_KEY"] = env_key
                    print("[deploy] Minted new ENVIRONMENT_API_KEY")
                
                # Optionally upload the new key to the backend using sealed box helper
                backend_base = env.dev_backend_url or ""
                synth_key = (env.synth_api_key or os.environ.get("SYNTH_API_KEY") or local_env.get("SYNTH_API_KEY") or "").strip()
                if backend_base and synth_key:
                        backend_base = backend_base.rstrip("/")
                        if not backend_base.endswith("/api"):
                            backend_base = f"{backend_base}/api"
                        try:
                            choice = input(
                                f"Upload ENVIRONMENT_API_KEY to backend {backend_base}? [Y/n]: "
                            ).strip().lower() or "y"
                        except Exception:
                            choice = "y"
                        if choice.startswith("y"):
                            try:
                                print(f"[deploy] Uploading ENVIRONMENT_API_KEY to {backend_base} …")
                                from synth_ai.rl.env_keys import setup_environment_api_key

                                setup_environment_api_key(backend_base.rstrip("/"), synth_key, token=env_key)
                                print("[deploy] Backend sealed-box upload complete.")
                            except Exception as upload_err:
                                print(f"[deploy] Failed to upload ENVIRONMENT_API_KEY: {upload_err}")
                                print(
                                    "Hint: run `uvx python -c \"from synth_ai.rl.env_keys import setup_environment_api_key as s;"
                                    " s('<backend>', '<synth_api_key>')\"` once the backend is reachable."
                                )

                synth_key = (env.synth_api_key or os.environ.get("SYNTH_API_KEY") or local_env.get("SYNTH_API_KEY") or "").strip()
                if not synth_key:
                    synth_key = input("Enter SYNTH_API_KEY for Modal secret (required): ").strip()
                    if not synth_key:
                        print("SYNTH_API_KEY is required to create the Modal secret.")
                        return 1
                    demo_core.persist_api_key(synth_key)
                    demo_core.persist_dotenv_values({"SYNTH_API_KEY": synth_key})
                    env.synth_api_key = synth_key

                openai_key = (os.environ.get("OPENAI_API_KEY") or local_env.get("OPENAI_API_KEY") or "").strip()
                if not openai_key:
                    openai_key = input(
                        "Enter your OpenAI API key, found at https://platform.openai.com/api-keys\n> "
                    ).strip()
                    if not openai_key:
                        print("OPENAI_API_KEY is required to create the Modal secret.")
                        return 1
                    demo_core.persist_dotenv_values({"OPENAI_API_KEY": openai_key})
                    local_env["OPENAI_API_KEY"] = openai_key

                values = {"SYNTH_API_KEY": synth_key, "OPENAI_API_KEY": openai_key}
                if env_key:
                    values["ENVIRONMENT_API_KEY"] = env_key

                try:
                    created = _ensure_modal_secret(secret_name, values=values, label="deploy", replace=True)
                except RuntimeError as secret_err:
                    print(f"Failed to prepare Modal secret '{secret_name}': {secret_err}")
                    return 2
                if created:
                    print(f"[deploy] Modal secret '{secret_name}' provisioned.")

                deploy_cmd = ["uv", "run", "python", "-m", "modal", "deploy", "--name", name_in, app_path]
                print("\nStreaming Modal build/deploy logs (this can take several minutes on first run)…\n")
                code, deploy_logs = _popen_stream_capture(deploy_cmd)
                if code != 0:
                    raise RuntimeError(f"modal deploy failed (exit {code})")
                # Try to parse URL directly from streamed logs
                if not url:
                    try:
                        import re as _re
                        m_all = _re.findall(r"https?://[^\s]+\.modal\.run", deploy_logs or "")
                        if m_all:
                            url = m_all[-1].strip().rstrip("/")
                    except Exception:
                        pass
                url_cmd = ["uv", "run", "python", "-m", "modal", "app", "url", name_in]
                code2, out2 = _popen_capture(url_cmd)
                if code2 == 0:
                    for token in out2.split():
                        if _is_modal_public_url(token):
                            url = token.strip().rstrip("/")
                            break
                # Fallback: try reading recent Modal logs for the app to find a URL line
                if not url:
                    code3, out3 = _popen_capture(["uv", "run", "python", "-m", "modal", "app", "list"])
                    if code3 == 0 and out3:
                        for line in out3.splitlines():
                            if name_in in line:
                                for token in line.split():
                                    if _is_modal_public_url(token):
                                        url = token.strip().rstrip("/")
                                        break
                            if url:
                                break
                # Prompt user if still no valid URL
                if not url:
                    print("\nCould not auto-detect a public Modal URL for the app.")
                    entered = input("Enter the Modal public URL (must contain '.modal.run'), or press Enter to abort: ").strip()
                    if entered and _is_modal_public_url(entered):
                        url = entered.rstrip("/")
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
        print("Next: uvx synth-ai run")
        return 0
    except Exception as e:
        print(f"Deploy error: {e}")
        return 2


    print("`rl_demo configure` prepares environment and secrets; `synth-ai run` now handles launches.")
    env = demo_core.load_env()
    synth_key = (env.synth_api_key or "").strip()
    if not synth_key:
        entered = input("Enter SYNTH_API_KEY (required): ").strip()
        if not entered:
            print("SYNTH_API_KEY is required.")
            return 1
        os.environ["SYNTH_API_KEY"] = entered
        demo_core.persist_api_key(entered)
        demo_core.persist_dotenv_values({"SYNTH_API_KEY": entered})
    env = demo_core.load_env()
    synth_key = (env.synth_api_key or "").strip()
    if not env.dev_backend_url:
        print("Backend URL missing. Set DEV_BACKEND_URL or BACKEND_OVERRIDE.")
        return 1
    try:
        env = _ensure_task_app_ready(env, synth_key, label="configure")
    except RuntimeError as exc:
        print(exc)
        return 1
    os.environ["ENVIRONMENT_API_KEY"] = env.env_api_key
    try:
        review = input("Review or create an RL config now? [Y/n]: ").strip().lower() or "y"
    except Exception:
        review = "y"
    if review.startswith("y"):
        _select_or_create_config(None, env)
    print("Environment ready. Use `uvx synth-ai run` to launch an RL job.")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a Modal-ready Math Task App in the current directory.

    Copies `examples/rl/task_app.py` and `examples/rl/deploy_task_app.sh` into CWD.
    Creates a `.env` with placeholders if it does not exist.
    """
    try:
        # Ensure `modal` is installed for deployment flows
        def _has_modal() -> bool:
            try:
                import importlib.util as _iu
                return _iu.find_spec("modal") is not None
            except Exception:
                return False

        if not _has_modal():
            print("modal not found; installing…")
            # Prefer uv if available; otherwise fallback to pip
            try:
                if shutil.which("uv"):
                    code, out = _popen_capture(["uv", "pip", "install", "modal>=1.1.4"])
                else:
                    code, out = _popen_capture([sys.executable, "-m", "pip", "install", "modal>=1.1.4"])
                if code != 0:
                    print(out)
                    print("Failed to install modal; continuing may fail.")
                else:
                    print("modal installed successfully.")
            except Exception as e:
                print(f"modal install error: {e}")
            # Re-check
            if not _has_modal():
                print("Warning: modal is still not importable after install attempt.")
        else:
            print("modal found")

        here = os.getcwd()
        demo_dir = os.path.join(here, "synth_demo")
        os.makedirs(demo_dir, exist_ok=True)
        # Paths inside synth_demo/
        dst_task_py = os.path.join(demo_dir, "task_app.py")
        dst_deploy = os.path.join(demo_dir, "deploy_task_app.sh")
        env_path = os.path.join(demo_dir, ".env")
        dst_cfg = os.path.join(demo_dir, "demo_config.toml")

        # Copy packaged math modal task app into synth_demo/task_app.py
        src_modal = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_task_apps", "math", "modal_task_app.py"))
        if not os.path.isfile(src_modal):
            print("Init failed: packaged math modal task app not found.")
            print(f"Looked for: {src_modal}")
            return 1
        if os.path.exists(dst_task_py) and not getattr(args, "force", False):
            print(f"Refusing to overwrite existing file: {dst_task_py} (use --force)")
            return 1
        shutil.copy2(src_modal, dst_task_py)

        # Create deploy script in synth_demo/
        deploy_text = r"""#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
APP="$HERE/task_app.py"
if [ -f "$HERE/.env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "$HERE/.env" | xargs -I{} echo {})
fi
uv run modal deploy "$APP" | tee "$HERE/.last_deploy.log"
URL=$(grep -Eo 'https://[^ ]+\.modal\.run' "$HERE/.last_deploy.log" | tail -1 || true)
if [ -n "$URL" ]; then
  if grep -q '^TASK_APP_BASE_URL=' "$HERE/.env" 2>/dev/null; then
    sed -i.bak "s#^TASK_APP_BASE_URL=.*#TASK_APP_BASE_URL=$URL#" "$HERE/.env" || true
  else
    echo "TASK_APP_BASE_URL=$URL" >> "$HERE/.env"
  fi
  echo "Saved TASK_APP_BASE_URL to $HERE/.env"
fi
"""
        _write_text(dst_deploy, deploy_text)
        try:
            st = os.stat(dst_deploy)
            os.chmod(dst_deploy, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

        # Seed .env if not present
        if not os.path.exists(env_path):
            _write_text(env_path, "\n".join([
                "# Required for task app auth to environment service",
                "ENVIRONMENT_API_KEY=",
                "",
                "# Optional: for CLI job submission and proxying OpenAI models",
                "SYNTH_API_KEY=",
                "OPENAI_API_KEY=",
                "",
                "# Optional: set to 'prod' to use production names",
                "ENVIRONMENT=",
            ]) + "\n")

        # Seed demo_config.toml from packaged default if not present (or overwrite with --force)
        packaged_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_task_apps", "math", "config.toml"))
        try:
            if os.path.isfile(packaged_cfg):
                if not os.path.exists(dst_cfg) or getattr(args, "force", False):
                    shutil.copy2(packaged_cfg, dst_cfg)
        except Exception:
            pass

        print("Initialized Math Task App in synth_demo/:")
        print(f"  - {dst_task_py}")
        print(f"  - {dst_deploy}")
        print(f"  - {env_path} (created if missing)")
        if os.path.exists(dst_cfg):
            print(f"  - {dst_cfg} (seeded)")
        print("")
        print("Next steps:")
        print("  1) cd synth_demo && put your ENVIRONMENT_API_KEY in ./.env")
        print("  2) Deploy to Modal:")
        print("     uvx bash ./deploy_task_app.sh")
        print("  3) From project root, run: uvx synth-ai run")
        return 0
    except Exception as e:
        print(f"Init error: {e}")
        return 2


def _http(method: str, url: str, headers: Dict[str, str] | None = None, body: Dict[str, Any] | None = None) -> tuple[int, Dict[str, Any] | str]:
    import urllib.request, urllib.error, json as _json, ssl
    data = None
    if body is not None:
        data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
    try:
        # Default: disable SSL verification for local/dev convenience.
        # Set SYNTH_SSL_VERIFY=1 to enable verification.
        ctx = ssl._create_unverified_context()  # nosec: disabled by default for dev
        if os.getenv("SYNTH_SSL_VERIFY", "0") == "1":
            ctx = None
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
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
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)

    synth_key = (env.synth_api_key or "").strip()
    if not synth_key:
        entered = input("Enter SYNTH_API_KEY (required): ").strip()
        if not entered:
            print("SYNTH_API_KEY is required.")
            return 1
        os.environ["SYNTH_API_KEY"] = entered
        demo_core.persist_api_key(entered)
        demo_core.persist_dotenv_values({"SYNTH_API_KEY": entered})
    env = demo_core.load_env()
    synth_key = (env.synth_api_key or "").strip()
    if not synth_key:
        print("SYNTH_API_KEY missing after persist.")
        return 1

    if not env.dev_backend_url:
        print("Backend URL missing. Set DEV_BACKEND_URL or BACKEND_OVERRIDE.")
        return 1

    try:
        env = _ensure_task_app_ready(env, synth_key, label="run")
    except RuntimeError as exc:
        print(exc)
        return 1

    os.environ["ENVIRONMENT_API_KEY"] = env.env_api_key

    import tomllib

    try:
        cfg_path = _select_or_create_config(getattr(args, "config", None), env)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    # Detect monorepo launcher and delegate if available (aligns with run_clustered.sh which works)
    launcher = "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/start_math_clustered.py"
    if os.path.isfile(launcher):
        backend_base = env.dev_backend_url[:-4] if env.dev_backend_url.endswith("/api") else env.dev_backend_url
        run_env = os.environ.copy()
        run_env["BACKEND_URL"] = backend_base
        run_env["SYNTH_API_KEY"] = env.synth_api_key
        run_env["TASK_APP_BASE_URL"] = env.task_app_base_url
        run_env["ENVIRONMENT_API_KEY"] = env.env_api_key
        run_env["RL_CONFIG_PATH"] = cfg_path
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
            # Actionable guidance for common auth issues
            try:
                base_url = backend_base.rstrip("/") + "/api"
            except Exception:
                base_url = backend_base
            sk = (env.synth_api_key or "").strip()
            ek = (env.env_api_key or "").strip()
            print("Hint: If backend responded 401, verify SYNTH_API_KEY for:", base_url)
            if sk:
                print(f"  SYNTH_API_KEY len={len(sk)} last5={sk[-5:]}")
            if ek:
                print(f"  ENVIRONMENT_API_KEY len={len(ek)} last5={ek[-5:]}")
            print("Also ensure your Modal secret contains ENVIRONMENT_API_KEY and matches the task app.")
        return code

    # Fallback: legacy jobs API flow
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
    # Print backend and key preview before request for clearer diagnostics
    try:
        sk = (env.synth_api_key or "").strip()
        sk_len = len(sk)
        sk_tail = sk[-5:] if sk_len >= 5 else sk
        print(f"[run] Backend API: {api}")
        print(f"[run] Using SYNTH_API_KEY len={sk_len} last5={sk_tail}")
    except Exception:
        pass
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
        print(f"Backend: {api}")
        try:
            if isinstance(js, dict):
                print(json.dumps(js, indent=2))
            else:
                print(str(js))
        except Exception:
            print(str(js))
        print("Request body was:\n" + json.dumps(body, indent=2))
        # Extra hints for auth failures
        try:
            sk = (env.synth_api_key or "").strip()
            if int(code) == 401 or (isinstance(js, dict) and any(isinstance(v, str) and "Invalid API key" in v for v in js.values())):
                base_url = env.dev_backend_url
                print("Hint: HTTP 401 Unauthorized from backend. Verify SYNTH_API_KEY for:", base_url)
                if sk:
                    print(f"  SYNTH_API_KEY len={len(sk)} last5={sk[-5:]}")
                print("Also ensure your Modal secret contains a valid ENVIRONMENT_API_KEY.")
        except Exception:
            pass
        return 2
    job_id = js.get("job_id") or js.get("id") or ""
    if not job_id:
        print("Job id missing in response:", js)
        print("Request body was:\n" + json.dumps(body, indent=2))
        return 2
    print("JOB_ID:", job_id)
    # Original behavior: start job and stream status/events until terminal
    _http(
        "POST",
        api + f"/rl/jobs/{job_id}/start",
        headers={"Authorization": f"Bearer {env.synth_api_key}"},
    )
    # Inform the user immediately that the job has started and where to track it
    print("Your job is running. Visit usesynth.ai to view its progress")
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
        ec, ej = _http(
            "GET",
            api + f"/orchestration/jobs/{job_id}/events?since_seq={since}&limit=200",
        )
        if ec == 200 and isinstance(ej, dict):
            events = ej.get("events") or ej.get("data") or []
            for e in events:
                seq = int(e.get("seq") or 0)
                if seq <= since:
                    continue
                since = seq
                typ = str(e.get("type") or e.get("event_type") or "").lower()
                msg = e.get("message") or e.get("msg") or ""
                if typ in (
                    "rl.eval.started",
                    "rl.eval.summary",
                    "rl.train.step",
                    "rl.metrics",
                    "rl.performance.metrics",
                ):
                    print(f"[{seq}] {typ}: {msg}")
        mc, mj = _http(
            "GET", api + f"/learning/jobs/{job_id}/metrics?after_step=-1&limit=50"
        )
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

    _add_parser(["rl_demo.setup", "demo.setup"], configure=lambda parser: parser.set_defaults(func=cmd_setup))

    def _init_opts(parser):
        parser.add_argument("--force", action="store_true", help="Overwrite existing files in CWD")
        parser.set_defaults(func=cmd_init)

    _add_parser(["rl_demo.init", "demo.init"], configure=_init_opts)

    # (prepare command removed)

    def _deploy_opts(parser):
        parser.add_argument("--local", action="store_true", help="Run local FastAPI instead of Modal deploy")
        parser.add_argument("--app", type=str, default=None, help="Path to Modal app.py for uv run modal deploy")
        parser.add_argument("--name", type=str, default="synth-math-demo", help="Modal app name")
        parser.add_argument("--script", type=str, default=None, help="Path to deploy_task_app.sh (optional legacy)")
        parser.set_defaults(func=cmd_deploy)

    _add_parser(["rl_demo.deploy", "demo.deploy"], configure=_deploy_opts)

    _add_parser(["rl_demo.configure", "demo.configure"], configure=lambda parser: parser.set_defaults(func=cmd_run))

    def _run_opts(parser):
        parser.add_argument("--config", type=str, default=None, help="Path to TOML config (skip prompt)")
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--group-size", type=int, default=None)
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--timeout", type=int, default=600)
        parser.add_argument("--dry-run", action="store_true", help="Print request body and exit")
        parser.set_defaults(func=cmd_run)

    _add_parser(["run", "rl_demo.run", "demo.run"], configure=_run_opts)

    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
