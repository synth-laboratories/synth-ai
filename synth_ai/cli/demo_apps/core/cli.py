from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from synth_ai.cli.demo_apps.demo_registry import (
    DemoTemplate,
    get_demo_template,
    list_demo_templates,
)
from synth_ai.cli.demo_apps.demo_task_apps import core as demo_core
from synth_ai.cli.demo_apps.demo_task_apps.core import DEFAULT_TASK_APP_SECRET_NAME, DemoEnv
from synth_ai.core.process import get_subprocess_env, should_filter_log_line

try:
    from synth_ai.handshake import HandshakeError, run_handshake  # type: ignore[import-untyped]
except ImportError:
    # handshake module may not exist in all environments
    HandshakeError = Exception  # type: ignore[assignment,misc]
    def run_handshake() -> None:  # type: ignore[misc]
        pass


def _key_preview(value: str, label: str) -> str:
    """Return a short descriptor for a secret without leaking the full value."""
    try:
        text = value or ""
        length = len(text)
        prefix = text[:6] if length >= 6 else text
        suffix = text[-5:] if length >= 5 else text
        return f"{label} len={length} prefix={prefix} last5={suffix}"
    except Exception:
        return f"{label} len=0"


def _is_modal_public_url(u: str) -> bool:
    try:
        s = (u or "").strip().lower()
        if not (s.startswith("http://") or s.startswith("https://")):
            return False
        return (".modal.run" in s) and ("modal.local" not in s) and ("pypi-mirror" not in s)
    except Exception:
        return False


def setup() -> int:
    # Change to demo directory if stored
    demo_dir = demo_core.load_demo_dir()
    if demo_dir and os.path.isdir(demo_dir):
        os.chdir(demo_dir)
        print(f"Using demo directory: {demo_dir}")

    # 1) Try to fetch keys from frontend; fall back to manual input if fetch fails
    synth_key = ""
    rl_env_key = ""
    org_name = "this organization"

    try:
        print("\n⏳ Connecting SDK to your browser session…")
        res = run_handshake()
        # Type narrowing for dict access - run_handshake returns dict[str, Any] | None
        res_dict: dict[str, Any] = res if isinstance(res, dict) else {}
        org_val = res_dict.get("org")  # type: ignore[misc]
        org: dict[str, Any] = org_val if isinstance(org_val, dict) else {}
        keys_val = res_dict.get("keys")  # type: ignore[misc]
        keys: dict[str, Any] = keys_val if isinstance(keys_val, dict) else {}
        synth_key = str(keys.get("synth") or "").strip()  # type: ignore[misc]
        rl_env_key = str(keys.get("rl_env") or "").strip()  # type: ignore[misc]
        org_name = org.get("name") or "this organization"  # type: ignore[misc]
        print(f"✅ Connected to {org_name}!")
    except (HandshakeError, Exception) as e:
        print(f"⚠️  Failed to fetch keys from frontend: {e}")
        print("Falling back to manual entry...")

    # Prompt for manual input if any key is missing
    if not synth_key:
        try:
            synth_key = input(
                "Failed to fetch your Synth API key. Please enter your Synth API key here:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return 1
        if not synth_key:
            print("Synth API key is required.")
            return 1

    if not rl_env_key:
        try:
            rl_env_key = input(
                "Failed to fetch your RL Environment API key. Please enter your RL Environment API key here:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return 1
        if not rl_env_key:
            print("RL Environment API key is required.")
            return 1

    # Persist both keys to .env
    dotenv_path = demo_core.persist_dotenv_values(
        {
            "SYNTH_API_KEY": synth_key,
            "ENVIRONMENT_API_KEY": rl_env_key,
        }
    )

    # Store .env path for subsequent commands
    demo_core.persist_env_file_path(dotenv_path)

    # 2) Reload env after handshake to pick up values from .env (suppress env prints)
    import contextlib
    import io

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
        if not current or not _is_modal_public_url(current):
            needs_lookup = True
        if not needs_lookup:
            return
        code, out = _popen_capture(
            [
                "uv",
                "run",
                "python",
                "-m",
                "modal",
                "app",
                "url",
                env.task_app_name,
            ]
        )
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
            }
            demo_core.persist_dotenv_values(dotenv_values)
            os.environ["TASK_APP_BASE_URL"] = new_url
            _refresh_env()

    # Keys have been written already via handshake; avoid any interactive prompts
    synth_key = env.synth_api_key.strip()
    if not local_env.get("SYNTH_API_KEY") and synth_key:  # type: ignore[misc]
        demo_core.persist_dotenv_values({"SYNTH_API_KEY": synth_key})
        _refresh_env()

    # Check Modal auth silently to avoid noisy output
    modal_ok, modal_msg = demo_core.modal_auth_status()

    _maybe_fix_task_url()

    if env.dev_backend_url:
        api = env.dev_backend_url.rstrip("/") + (
            "" if env.dev_backend_url.endswith("/api") else "/api"
        )
        demo_core.assert_http_ok(api + "/health", method="GET")
        # Intentionally suppress backend health print for concise output
    if env.task_app_base_url:
        demo_core.assert_http_ok(
            env.task_app_base_url.rstrip("/") + "/health", method="GET"
        ) or demo_core.assert_http_ok(env.task_app_base_url.rstrip("/"), method="GET")
        # Intentionally suppress task app health print
    else:
        print("\nSet your task app URL by running:\nuvx synth-ai rl_demo deploy\n")

    # Omit uv version print to keep output concise

    # Keep exit code neutral; not all checks are critical for pairing
    print(f"\nKeys saved to: {dotenv_path}")
    return 0


def _popen_capture(
    cmd: list[str], cwd: str | None = None, env: dict | None = None
) -> tuple[int, str]:
    import subprocess

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=get_subprocess_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
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
            env=get_subprocess_env(env),
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
                if not should_filter_log_line(line):
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


def _popen_stream_capture(
    cmd: list[str], cwd: str | None = None, env: dict | None = None
) -> tuple[int, str]:
    """Stream subprocess output to stdout and also capture it into a buffer."""
    import subprocess
    import threading

    buf_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=get_subprocess_env(env),
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
                if not should_filter_log_line(line):
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


def _fmt_float(value: float) -> str:
    return f"{value:.10g}"


def _find_asgi_apps(root: Path) -> list[Path]:
    """Recursively search for Python files that declare a Modal ASGI app.

    A file is considered a Modal task app candidate if it contains one of:
      - "@asgi_app()"
      - "@modal.asgi_app()"
    """
    results: list[Path] = []
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        "venv",
        ".venv",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not str(name).endswith(".py"):
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


def _prompt_value(
    label: str, default: str | int | float, cast: Callable[[str], Any] | None = None
) -> Any:
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
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        "venv",
        ".venv",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not str(name).endswith(".toml"):
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
            destination = (
                input(f"Path to save new config [{default_path}]: ").strip() or default_path
            )
        except Exception:
            destination = default_path
        destination = os.path.abspath(destination)
        if os.path.isdir(destination):
            print("Path points to a directory; provide a file path.")
            continue
        if os.path.exists(destination):
            try:
                overwrite = (
                    input(f"{destination} exists. Overwrite? [y/N]: ").strip().lower() or "n"
                )
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
    topology_gpu_type = _prompt_value(
        "Topology GPU type", f"{compute_gpu_type}:{compute_gpu_count}"
    )
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

    template = (
        textwrap.dedent(
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
        ).strip()
        + "\n"
    )

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
    packaged = Path(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "demo_task_apps", "math", "config.toml")
        )
    )
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
        raise RuntimeError(
            f"[{label}] ENVIRONMENT_API_KEY missing. Run `uvx synth-ai rl_demo deploy` first."
        )

    task_url = env.task_app_base_url
    if not task_url or not _is_modal_public_url(task_url):
        resolved = ""
        if env.task_app_name:
            try:
                choice = (
                    input(f"Resolve URL from Modal for app '{env.task_app_name}'? [Y/n]: ")
                    .strip()
                    .lower()
                    or "y"
                )
            except Exception:
                choice = "y"
            if choice.startswith("y"):
                code, out = _popen_capture(
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "modal",
                        "app",
                        "url",
                        env.task_app_name,
                    ]
                )
                if code == 0 and out:
                    for tok in out.split():
                        if _is_modal_public_url(tok):
                            resolved = tok.strip().rstrip("/")
                            break
        if not resolved:
            print(f"[{label}] Task app URL not configured or not a valid Modal public URL.")
            print("Examples: https://<app-name>-fastapi-app.modal.run")
            entered = input(
                "Enter Task App base URL (must contain '.modal.run'), or press Enter to abort: "
            ).strip()
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

    demo_core.persist_task_url(task_url, name=app_name)
    demo_core.persist_dotenv_values(
        {
            "TASK_APP_BASE_URL": task_url,
            "TASK_APP_NAME": app_name,
            "TASK_APP_SECRET_NAME": DEFAULT_TASK_APP_SECRET_NAME,
        }
    )

    if synth_key:
        os.environ["SYNTH_API_KEY"] = synth_key

    openai_key = (os.environ.get("OPENAI_API_KEY") or local_env.get("OPENAI_API_KEY") or "").strip()  # type: ignore[misc]
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    print(f"[{label}] Verifying rollout health:")
    try:
        ek = (env_key or "").strip()
        print(f"[{label}] {_key_preview(ek, 'ENVIRONMENT_API_KEY')}")
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
        with contextlib.suppress(Exception):
            print(f"[{label}] Sent header X-API-Key → {_key_preview(env_key, 'X-API-Key')}")
    else:
        print(f"[{label}] Task app rollout health check OK.")

    os.environ["TASK_APP_BASE_URL"] = task_url
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    os.environ["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
    updated_env = demo_core.load_env()
    updated_env.env_api_key = env_key
    updated_env.task_app_base_url = task_url
    updated_env.task_app_name = app_name
    updated_env.task_app_secret_name = DEFAULT_TASK_APP_SECRET_NAME
    return updated_env


def deploy(
    local: bool = False, app: str | None = None, name: str | None = None, script: str | None = None
) -> int:
    # Change to demo directory if stored
    demo_dir = demo_core.load_demo_dir()
    if demo_dir and os.path.isdir(demo_dir):
        os.chdir(demo_dir)
        print(f"Using demo directory: {demo_dir}")

    env = demo_core.load_env()
    os.environ["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    local_env = demo_core.load_dotenv_file(cwd_env_path)
    url = ""
    app_name = env.task_app_name or ""
    try:
        if local:
            print("Starting local Task App…")
            import subprocess

            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from synth_ai.cli.demo_apps.demo_task_apps.math.app import run; run()",
                ],
                env=get_subprocess_env(),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            target = "http://127.0.0.1:8080"
            app_name = ""
            for _ in range(30):
                if demo_core.assert_http_ok(
                    target + "/health", method="GET"
                ) or demo_core.assert_http_ok(target, method="GET"):
                    url = target
                    break
                time.sleep(1)
        else:
            # Auto-detect app path if not supplied; prompt interactively from discovered ASGI apps
            app_path = os.path.abspath(app) if app else None
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
                            sel = (
                                input(f"Enter choice [1-{len(found)}] (default 1): ").strip() or "1"
                            )
                        except Exception:
                            sel = "1"
                        try:
                            choice = int(sel)
                        except Exception:
                            choice = 1
                        choice = max(1, min(choice, len(found)))
                        app_path = str(found[choice - 1].resolve())
            if not app_path and script:
                # Legacy script fallback if user supplied --script explicitly
                from synth_ai.cli.demo_apps.demo_task_apps.math.deploy_modal import (
                    deploy as modal_deploy,
                )

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
                # Surface the app path before asking for the name
                print(f"Using task app: {app_path}")
                existing_name = (name or env.task_app_name or "").strip()
                if not existing_name:
                    existing_name = f"synth-{os.path.splitext(os.path.basename(app_path))[0]}"
                suggested_name = existing_name
                name_in = input(f"Modal app name [{suggested_name}]: ").strip() or suggested_name
                app_name = name_in
                print("\nAbout to deploy with:")
                print(f"  app.py: {app_path}")
                print(f"  name:   {name_in}")
                proceed = (input("Proceed? [Y/n]: ").strip().lower() or "y").startswith("y")
                if not proceed:
                    print("Aborted by user.")
                    return 1

                existing_env_key = (env.env_api_key or "").strip()
                env_key: str | None = existing_env_key or None
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
                    from synth_ai.sdk.learning.rl.secrets import mint_environment_api_key

                    env_key = mint_environment_api_key()
                    demo_core.persist_env_api_key(env_key)
                    demo_core.persist_dotenv_values({"ENVIRONMENT_API_KEY": env_key})
                    os.environ["ENVIRONMENT_API_KEY"] = env_key
                    env.env_api_key = env_key
                    local_env["ENVIRONMENT_API_KEY"] = env_key
                    print("[deploy] Minted new ENVIRONMENT_API_KEY")
                elif env_key:
                    os.environ["ENVIRONMENT_API_KEY"] = env_key

                # Optionally upload the new key to the backend using sealed box helper
                backend_base = (env.dev_backend_url or "").rstrip("/")
                synth_key = (
                    env.synth_api_key
                    or os.environ.get("SYNTH_API_KEY")  # type: ignore[misc]
                    or local_env.get("SYNTH_API_KEY")  # type: ignore[misc]
                    or ""
                ).strip()
                if backend_base and synth_key:
                    # Pass a base WITHOUT trailing /api to setup_environment_api_key,
                    # since it appends /api/v1/... internally.
                    non_api_base = (
                        backend_base[:-4] if backend_base.endswith("/api") else backend_base
                    )
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
                            from synth_ai.sdk.learning.rl.env_keys import setup_environment_api_key

                            setup_environment_api_key(non_api_base, synth_key, token=env_key)
                            print("[deploy] Backend sealed-box upload complete.")
                        except Exception as upload_err:
                            print(f"[deploy] Failed to upload ENVIRONMENT_API_KEY: {upload_err}")
                            print(
                                'Hint: run `uvx python -c "from synth_ai.sdk.learning.rl.env_keys import setup_environment_api_key as s;'
                                " s('<backend>', '<synth_api_key>')\"` once the backend is reachable."
                            )

                synth_key = (
                    env.synth_api_key
                    or os.environ.get("SYNTH_API_KEY")  # type: ignore[misc]
                    or local_env.get("SYNTH_API_KEY")  # type: ignore[misc]
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
                    name_in,
                    app_path,
                ]
                print(
                    "\nStreaming Modal build/deploy logs (this can take several minutes on first run)…\n"
                )
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
                    code3, out3 = _popen_capture(
                        ["uv", "run", "python", "-m", "modal", "app", "list"]
                    )
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
                    entered = input(
                        "Enter the Modal public URL (must contain '.modal.run'), or press Enter to abort: "
                    ).strip()
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
        print("\nNext step:\n$ uvx synth-ai run")
        return 0
    except Exception as e:
        print(f"Deploy error: {e}")
        return 2

    print(
        "`rl_demo configure` prepares environment and secrets; `synth-ai run` now handles launches."
    )
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


def _ensure_modal_installed() -> None:
    """Install the modal package if it is not already available and check authentication."""

    # Check if modal is installed
    modal_installed = False
    try:
        import importlib.util as _iu

        if _iu.find_spec("modal") is not None:
            modal_installed = True
    except Exception:
        pass

    # Install modal if needed
    if not modal_installed:
        print("modal not found; installing…")
        try:
            if shutil.which("uv"):
                code, out = _popen_capture(["uv", "pip", "install", "modal>=1.1.4"])
            else:
                code, out = _popen_capture([sys.executable, "-m", "pip", "install", "modal>=1.1.4"])
            if code != 0:
                print(out)
                print("Failed to install modal; continuing may fail.")
                return
            else:
                print("✓ modal installed successfully")
                modal_installed = True
        except Exception as exc:
            print(f"modal install error: {exc}")
            return

    # Verify modal is importable
    if modal_installed:
        try:
            import importlib.util as _iu

            if _iu.find_spec("modal") is None:
                print("Warning: modal is still not importable after install attempt.")
                return
        except Exception:
            print("Warning: unable to verify modal installation.")
            return

    # Check modal authentication status
    auth_ok, auth_msg = demo_core.modal_auth_status()
    if auth_ok:
        print(f"✓ Modal authenticated: {auth_msg}")
    else:
        print("\n⚠️  Modal authentication required")
        print(f"   Status: {auth_msg}")
        print("\n   To authenticate Modal, run:")
        print("     modal setup")
        print("\n   Or set environment variables:")
        print("     export MODAL_TOKEN_ID=your-token-id")
        print("     export MODAL_TOKEN_SECRET=your-token-secret")
        print("\n   You can deploy later after authenticating.\n")


def init(template: str | None = None, dest: str | None = None, force: bool = False) -> int:
    """Materialise a demo task app template into the current directory."""

    templates = list(list_demo_templates())
    if not templates:
        print("No demo templates registered. Update synth_ai/demo_registry.py to add entries.")
        return 1

    selected: DemoTemplate | None = None
    if template:
        selected = get_demo_template(template)
        if selected is None:
            available = ", ".join(t.template_id for t in templates)
            print(f"Unknown template '{template}'. Available: {available}")
            return 1
    else:
        if force:
            selected = templates[0]
            print(
                f"Using default template: {selected.name} ({selected.template_id}) "
                f"(pass --template to choose another)"
            )
        else:
            print("Select a demo template:" + "\n")
            for idx, tpl in enumerate(templates, start=1):
                print(f"  [{idx}] {tpl.name} ({tpl.template_id})")
                print(f"      {tpl.description}")
            try:
                choice_raw = input(f"Enter choice [1-{len(templates)}] (default 1): ").strip() or "1"
            except Exception:
                choice_raw = "1"
            if not choice_raw.isdigit():
                print("Selection must be a number.")
                return 1
            choice_idx = int(choice_raw)
            if not 1 <= choice_idx <= len(templates):
                print("Selection out of range.")
                return 1
            selected = templates[choice_idx - 1]

    assert selected is not None

    default_subdir = selected.default_subdir or selected.template_id

    # Check if default destination is already occupied and switch to local_demos/ if needed
    if dest:
        default_dest = Path(dest).expanduser().resolve()
    else:
        primary_dest = Path.cwd() / default_subdir
        if primary_dest.exists() and any(primary_dest.iterdir()):
            # Switch to local_demos/ automatically if primary location is occupied
            default_dest = (Path.cwd() / "local_demos" / default_subdir).resolve()
        else:
            default_dest = primary_dest.resolve()

    if force:
        dest_input = ""
    else:
        try:
            dest_input = input(f"Destination directory [{default_dest}]: ").strip()
        except Exception:
            dest_input = ""
    destination = Path(dest_input).expanduser().resolve() if dest_input else default_dest

    # Track whether we should skip individual file prompts (if we already cleared the directory)
    directory_cleared = False

    if destination.exists():
        if destination.is_file():
            print(f"Destination {destination} is a file. Provide a directory path.")
            return 1
        if any(destination.iterdir()):
            if force:
                response = "y"
            else:
                try:
                    response = (
                        input(f"Destination {destination} is not empty. Overwrite? [y/N]: ")
                        .strip()
                        .lower()
                    )
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return 1
            if response not in ("y", "yes"):
                print("Cancelled. Choose another directory or delete the existing one.")
                return 1
            # User agreed to overwrite - clear the entire directory including hidden files
            print(f"Clearing {destination}...")
            try:
                # Remove all contents including hidden files (.env, .git, etc.)
                shutil.rmtree(destination)
            except Exception as e:
                print(f"Error clearing directory: {e}")
                print("Please manually remove the directory and try again.")
                return 1
            # Recreate empty directory
            destination.mkdir(parents=True, exist_ok=True)
            # Verify it's actually empty
            if any(destination.iterdir()):
                print(f"Warning: Directory {destination} still contains files after clearing.")
                print("Some files may not have been removed. Please check manually.")
                return 1
            directory_cleared = True
    else:
        destination.mkdir(parents=True, exist_ok=True)

    if selected.requires_modal:
        _ensure_modal_installed()

    try:
        for spec in selected.iter_copy_specs():
            src_path = spec.absolute_source()
            if not src_path.exists():
                print(f"Template source missing: {src_path}")
                return 1
            dest_path = (destination / spec.destination).resolve()

            # Handle directory copying
            if src_path.is_dir():
                if dest_path.exists() and not directory_cleared:
                    if force:
                        response = "y"
                    else:
                        try:
                            response = (
                                input(f"Directory {dest_path.name} exists. Overwrite? [y/N]: ")
                                .strip()
                                .lower()
                            )
                        except (EOFError, KeyboardInterrupt):
                            print("\nCancelled.")
                            return 1
                    if response not in ("y", "yes"):
                        print(f"Skipping {dest_path.name}")
                        continue
                    shutil.rmtree(dest_path)
                elif dest_path.exists() and directory_cleared:
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            else:
                # Handle file copying
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() and not directory_cleared:
                    if force:
                        response = "y"
                    else:
                        try:
                            response = (
                                input(f"File {dest_path.name} exists. Overwrite? [y/N]: ")
                                .strip()
                                .lower()
                            )
                        except (EOFError, KeyboardInterrupt):
                            print("\nCancelled.")
                            return 1
                    if response not in ("y", "yes"):
                        print(f"Skipping {dest_path.name}")
                        continue
                shutil.copy2(src_path, dest_path)
                if spec.make_executable:
                    try:
                        st = os.stat(dest_path)
                        os.chmod(dest_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    except Exception:
                        pass

        if selected.env_lines:
            env_path = destination / ".env"
            should_write = True
            if env_path.exists() and not directory_cleared:
                if force:
                    response = "y"
                else:
                    try:
                        response = input("File .env exists. Overwrite? [y/N]: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nCancelled.")
                        return 1
                should_write = response in ("y", "yes")
            if should_write:
                _write_text(str(env_path), "\n".join(selected.env_lines) + "\n")
            elif not directory_cleared:
                print("Skipping .env")

        config_src = selected.config_source_path()
        if config_src and config_src.exists():
            cfg_dst = (destination / selected.config_destination).resolve()
            should_copy = True
            if cfg_dst.exists() and not directory_cleared:
                if force:
                    response = "y"
                else:
                    try:
                        response = (
                            input(f"File {cfg_dst.name} exists. Overwrite? [y/N]: ").strip().lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        print("\nCancelled.")
                        return 1
                should_copy = response in ("y", "yes")
            if should_copy:
                cfg_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_src, cfg_dst)
            elif not directory_cleared:
                print(f"Skipping {cfg_dst.name}")

        if selected.post_copy is not None:
            try:
                selected.post_copy(destination)
            except Exception as post_exc:
                print(f"Post-processing failed: {post_exc}")
                return 1

        # Store demo directory for subsequent commands
        demo_core.persist_demo_dir(str(destination))

        # Store .env path if it was created
        env_file = destination / ".env"
        if env_file.exists():
            demo_core.persist_env_file_path(str(env_file))

        print(f"Demo template '{selected.name}' materialised at {destination}.")
        print("Files created:")
        for spec in selected.iter_copy_specs():
            print(f"  - {spec.destination}")
        if selected.env_lines:
            print("  - .env")
        if selected.config_source_path():
            print(f"  - {selected.config_destination}")
        print("\nDemo directory stored. Subsequent commands will use this directory automatically.")
        print("Review the files, edit .env, and run any provided deploy scripts when ready.")
        return 0
    except KeyboardInterrupt:
        print("Aborted")
        return 1
    except Exception as exc:
        print(f"Init failed: {exc}")
        return 1


def _http(
    method: str, url: str, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any] | str]:
    import json as _json
    import ssl
    import urllib.error
    import urllib.request

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


def run(
    config: str | None = None,
    batch_size: int | None = None,
    group_size: int | None = None,
    model: str | None = None,
    timeout: int = 600,
    dry_run: bool = False,
) -> int:
    # Change to demo directory if stored
    demo_dir = demo_core.load_demo_dir()
    if demo_dir and os.path.isdir(demo_dir):
        os.chdir(demo_dir)
        print(f"Using demo directory: {demo_dir}")

    env = demo_core.load_env()
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    demo_core.load_dotenv_file(cwd_env_path)

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
        cfg_path = _select_or_create_config(config, env)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    # Detect monorepo launcher and delegate if available (aligns with run_clustered.sh which works)
    launcher = "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/start_math_clustered.py"
    if os.path.isfile(launcher):
        backend_base = (
            env.dev_backend_url[:-4]
            if env.dev_backend_url.endswith("/api")
            else env.dev_backend_url
        )
        run_env = os.environ.copy()
        run_env["BACKEND_URL"] = backend_base
        run_env["SYNTH_API_KEY"] = env.synth_api_key
        run_env["TASK_APP_BASE_URL"] = env.task_app_base_url
        run_env["ENVIRONMENT_API_KEY"] = env.env_api_key
        run_env["RL_CONFIG_PATH"] = cfg_path
        # Optional: TRAINER_START_URL passthrough if already set in environment
        run_env["TRAINER_START_URL"] = run_env.get("TRAINER_START_URL", "")  # type: ignore[misc]
        # Forward convenience knobs
        if batch_size is not None:
            run_env["RL_BATCH_SIZE"] = str(int(batch_size))
        if group_size is not None:
            run_env["RL_GROUP_SIZE"] = str(int(group_size))
        if model:
            run_env["RL_MODEL"] = model
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
                print(f"  {_key_preview(sk, 'SYNTH_API_KEY')}")
            if ek:
                print(f"  {_key_preview(ek, 'ENVIRONMENT_API_KEY')}")
            print(
                "Ensure the ENVIRONMENT_API_KEY you deployed with matches the task app and remains exported."
            )
        return code

    # Fallback: legacy jobs API flow
    with open(cfg_path, "rb") as fh:
        inline_cfg = tomllib.load(fh)
    with open(cfg_path) as fh2:
        toml_text = fh2.read()
    if batch_size is not None:
        inline_cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    if group_size is not None:
        inline_cfg.setdefault("training", {})["group_size"] = int(group_size)
    model_name = model or (inline_cfg.get("model", {}) or {}).get("name", "Qwen/Qwen3-0.6B")  # type: ignore[misc]
    api = env.dev_backend_url.rstrip("/") + ("" if env.dev_backend_url.endswith("/api") else "/api")
    # Print backend and key preview before request for clearer diagnostics
    try:
        sk = (env.synth_api_key or "").strip()
        print(f"[run] Backend API: {api}")
        print(f"[run] {_key_preview(sk, 'SYNTH_API_KEY')}")
    except Exception:
        pass
    data_fragment: dict[str, Any] = {
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
        if isinstance(inline_cfg.get(k), dict):  # type: ignore[misc]
            data_fragment[k] = inline_cfg[k]
    compute = {}
    if isinstance(inline_cfg.get("compute"), dict):  # type: ignore[misc]
        if inline_cfg["compute"].get("gpu_type"):  # type: ignore[misc]
            compute["gpu_type"] = str(inline_cfg["compute"]["gpu_type"]).upper()
        if inline_cfg["compute"].get("gpu_count"):  # type: ignore[misc]
            compute["gpu_count"] = int(inline_cfg["compute"]["gpu_count"])
    if not compute:
        topo = inline_cfg.get("topology") or {}  # type: ignore[misc]
        gshape = str(topo.get("gpu_type") or "")  # type: ignore[misc]
        if ":" in gshape:
            t, c = gshape.split(":", 1)
            compute = {"gpu_type": t.upper(), "gpu_count": int(c)}
    body: dict[str, Any] = {
        "job_type": "rl",
        "data": data_fragment,
    }
    if compute:
        body["compute"] = compute
    code, js = _http(
        "POST",
        api + "/rl/jobs",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env.synth_api_key}",
        },
        body=body,
    )
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
        try:
            auth_preview = _key_preview(env.synth_api_key or "", "SYNTH_API_KEY (auth)")
            print(f"[run] {auth_preview}")
        except Exception:
            pass
        try:
            data_block = body.get("data") if isinstance(body, dict) else None  # type: ignore[misc]
            env_key_body = ""
            if isinstance(data_block, dict):
                env_key_body = str(data_block.get("environment_api_key") or "")  # type: ignore[misc]
            if env_key_body:
                print(f"[run] {_key_preview(env_key_body, 'environment_api_key (body)')}")
        except Exception:
            pass
        try:
            current_env_key = env.env_api_key or ""
            if current_env_key:
                print(f"[run] {_key_preview(current_env_key, 'ENVIRONMENT_API_KEY (current)')}")
        except Exception:
            pass
        if isinstance(js, dict):
            detail = js.get("detail")  # type: ignore[misc]
            if isinstance(detail, dict):
                try:
                    sent_key = detail.get("sent_key")  # type: ignore[misc]
                    if isinstance(sent_key, str):
                        print(
                            f"[run] Backend detail.sent_key {_key_preview(sent_key, 'detail.sent_key')}"
                        )
                except Exception:
                    pass
                try:
                    sent_keys = detail.get("sent_keys")  # type: ignore[misc]
                    if isinstance(sent_keys, list | tuple):
                        previews = []
                        for idx, val in enumerate(sent_keys):
                            if isinstance(val, str):
                                previews.append(_key_preview(val, f"detail.sent_keys[{idx}]"))
                        if previews:
                            joined = "; ".join(previews)
                            print(f"[run] Backend detail.sent_keys previews: {joined}")
                except Exception:
                    pass
                try:
                    key_prefix = detail.get("sent_key_prefix")  # type: ignore[misc]
                    if isinstance(key_prefix, str):
                        print(f"[run] Backend detail.sent_key_prefix={key_prefix}")
                except Exception:
                    pass
                try:
                    health_url = detail.get("health_url")
                    if isinstance(health_url, str):
                        print(f"[run] Backend detail.health_url={health_url}")
                except Exception:
                    pass
        # Extra hints for auth failures
        try:
            sk = (env.synth_api_key or "").strip()
            if int(code) == 401 or (
                isinstance(js, dict)
                and any(isinstance(v, str) and "Invalid API key" in v for v in js.values())
            ):
                base_url = env.dev_backend_url
                print(
                    "Hint: HTTP 401 Unauthorized from backend. Verify SYNTH_API_KEY for:", base_url
                )
                if sk:
                    print(f"  {_key_preview(sk, 'SYNTH_API_KEY')}")
                print(
                    "Ensure the ENVIRONMENT_API_KEY and OPENAI_API_KEY used for deployment remain valid."
                )
        except Exception:
            pass
        return 2
    job_id = js.get("job_id") or js.get("id") or ""  # type: ignore[misc]
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
        status = (sj.get("status") if isinstance(sj, dict) else "") if sc == 200 else ""  # type: ignore[misc]
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
            events = ej.get("events") or ej.get("data") or []  # type: ignore[misc]
            for e in events:
                seq = int(e.get("seq") or 0)
                if seq <= since:
                    continue
                since = seq
                typ = str(e.get("type") or e.get("event_type") or "").lower()  # type: ignore[misc]
                msg = e.get("message") or e.get("msg") or ""  # type: ignore[misc]
                if typ in (
                    "rl.eval.started",
                    "rl.eval.summary",
                    "rl.train.step",
                    "rl.metrics",
                    "rl.performance.metrics",
                ):
                    print(f"[{seq}] {typ}: {msg}")
        mc, mj = _http("GET", api + f"/learning/jobs/{job_id}/metrics?after_step=-1&limit=50")
        if mc == 200 and isinstance(mj, dict):
            pts = mj.get("points") or []  # type: ignore[misc]
            for p in pts:
                name = p.get("name")  # type: ignore[misc]
                if name == "eval.reward_mean":
                    print(f"metric eval.reward_mean step={p.get('step')} value={p.get('value')}")  # type: ignore[misc]
                    break
        if time.time() - start_t > (timeout or 600):
            print("Timeout waiting for terminal state.")
            break
        time.sleep(2)
    return 0
