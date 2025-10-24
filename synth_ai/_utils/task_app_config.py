from __future__ import annotations

import os
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from synth_ai.demos.core import DemoEnv

__all__ = [
    "fmt_float",
    "find_vllm_tomls",
    "prompt_value",
    "create_new_config",
    "select_or_create_config",
]


def fmt_float(value: float) -> str:
    return f"{value:.10g}"


def prompt_value(label: str, default: str | int | float, cast: Callable[[str], Any] | None = None) -> Any:
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


def find_vllm_tomls(root: Path) -> list[Path]:
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


def create_new_config(env: DemoEnv) -> str:
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

    env_name = prompt_value("Environment name", "Crafter")
    policy_name = prompt_value("Policy name", "crafter-react")
    model_name = prompt_value("Model name", "Qwen/Qwen3-0.6B")
    compute_gpu_type = prompt_value("Compute GPU type", "H100")
    compute_gpu_count = prompt_value("Compute GPU count", 4, int)
    topology_gpu_type = prompt_value("Topology GPU type", f"{compute_gpu_type}:{compute_gpu_count}")
    gpus_for_vllm = prompt_value("Topology gpus_for_vllm", 2, int)
    gpus_for_training = prompt_value("Topology gpus_for_training", 1, int)
    tensor_parallel = prompt_value("Topology tensor_parallel", 2, int)
    gpus_for_ref = prompt_value("Topology gpus_for_ref", 1, int)
    vllm_tp_size = prompt_value("vLLM tensor parallel size", tensor_parallel, int)
    vllm_max_model_len = prompt_value("vLLM max_model_len", 8192, int)
    vllm_max_num_seqs = prompt_value("vLLM max_num_seqs", 32, int)
    vllm_gpu_mem_util = prompt_value("vLLM gpu_memory_utilization", 0.9, float)
    vllm_max_parallel = prompt_value("vLLM max_parallel_generations", 4, int)
    training_num_epochs = prompt_value("Training num_epochs", 1, int)
    training_iters = prompt_value("Training iterations_per_epoch", 2, int)
    training_batch = prompt_value("Training batch_size", 1, int)
    training_group = prompt_value("Training group_size", 8, int)
    training_lr = prompt_value("Training learning_rate", 5e-6, float)
    task_url_default = env.task_app_base_url or ""
    services_task_url = prompt_value("services.task_url", task_url_default)

    template = (
        textwrap.dedent(
            f"""\
        # Crafter online RL training configuration (research local copy)

        [model]
        #name = "fft:Qwen/Qwen3-4B:job_7243b8aa76fe4b59"
        name = "{model_name}"
        dtype = "bfloat16"
        seed = 42
        trainer_mode = "full"

        [lora]
        r = 16
        alpha = 32
        dropout = 0.05
        target_modules = [
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj",
        ]

        [rdma]
        enabled = false
        ifname = "eth0"
        ip_type = "ipv4"
        p2p_disable = 0
        shm_disable = 0
        fast_nccl = false

        gid_index = 3
        cross_nic = 0
        collnet_enable = 0
        net_gdr_level = 2

        nsocks_perthread = 4
        socket_nthreads = 2

        algo = "Ring"
        proto = "Simple"
        p2p_level = "SYS"
        debug = "INFO"

        [compute]
        gpu_type = "{compute_gpu_type}"
        gpu_count = {compute_gpu_count}

        [topology]
        type = "single_node_split"
        gpu_type = "{topology_gpu_type}"
        use_rdma = false
        gpus_for_vllm = {gpus_for_vllm}
        gpus_for_training = {gpus_for_training}
        tensor_parallel = {tensor_parallel}
        gpus_for_ref = {gpus_for_ref}

        [vllm]
        tensor_parallel_size = {vllm_tp_size}
        gpu_memory_utilization = {fmt_float(vllm_gpu_mem_util)}
        max_model_len = {vllm_max_model_len}
        max_num_seqs = {vllm_max_num_seqs}
        enforce_eager = false
        max_parallel_generations = {vllm_max_parallel}

        # Reference scoring server (dedicated GPU)
        [reference]
        placement = "dedicated"
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
        learning_rate = {fmt_float(training_lr)}
        max_grad_norm = 0.5
        log_interval = 1
        update_reference_interval = 0
        weight_sync_interval = 1

        [training.weight_sync]
        enable = true
        targets = ["policy"]

        [rollout]
        env_name = "{env_name}"
        policy_name = "{policy_name}"
        env_config = {{}}
        max_steps_per_episode = 5
        sampling_temperature = 0.3
        sampling_top_p = 0.95
        max_tokens = 1024
        max_concurrent_rollouts = 4
        ops_per_rollout = 14
        on_done = "reset"
        thinking_mode = "think"
        thinking_budget = 512

        [policy]
        config = {{}}

        [evaluation]
        seeds = [0, 1, 2, 3, 4, 5, 6, 7]
        rollouts_per_seed = 1
        instances = 0
        max_concurrent_rollouts = 4
        thinking_mode = "think"
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
        mode = "off"
        step_beta = 0.0
        indicator_lambda = 0.0

        [trainer]
        allow_ref_fallback = false

        [checkpoint]
        interval = 10
        directory = "/checkpoints"
        keep_last_n = 3
        save_optimizer = true
        save_scheduler = true
        enabled = true

        [services]
        task_url = "{services_task_url}"
        """
        ).strip()
        + "\n"
    )

    with open(destination, "w", encoding="utf-8") as fh:
        fh.write(template)
    print(f"Wrote config to {destination}")
    return destination


def select_or_create_config(explicit: str | None, env: DemoEnv) -> str:
    if explicit:
        path = os.path.abspath(explicit)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    search_root = Path(os.getcwd())
    discovered = find_vllm_tomls(search_root)

    extras: list[Path] = []
    packaged = Path(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo", "math", "config.toml"))
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
        return create_new_config(env)

    print("Select a TOML config (found [vllm] section):")
    for idx, path in enumerate(all_paths, 1):
        resolved = path.resolve()
        print(f"[{idx}] {resolved.name}")
        print(f"    {resolved}")
    create_idx = len(all_paths) + 1
    print(f"[{create_idx}] Create new config")
    try:
        sel = input(f"Enter choice [1-{create_idx}] (default 1): ").strip() or "1"
    except Exception:
        sel = "1"
    try:
        choice = int(sel)
    except Exception:
        choice = 1
    if choice == create_idx:
        return create_new_config(env)
    choice = max(1, min(choice, len(all_paths)))
    selected = os.path.abspath(all_paths[choice - 1])
    print(f"Using config: {selected}")
    return selected

