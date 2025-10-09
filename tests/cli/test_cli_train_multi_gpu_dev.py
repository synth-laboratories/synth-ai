import asyncio
import json
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Awaitable

import pytest

from synth_ai.jobs.client import JobsClient


pytestmark = [pytest.mark.integration, pytest.mark.slow]


_JOB_ID_PATTERN = re.compile(r"Job created \(id=([^\)]+)\)")


REPO_ROOT = Path(__file__).resolve().parents[2]


def _maybe_load_env() -> None:
    if os.getenv("SYNTH_API_KEY") and os.getenv("DEV_BACKEND_URL"):
        return
    env_candidates = [
        ".env.test.dev",
        ".env.test",
        ".env",
    ]
    cwd = Path.cwd()
    for candidate in env_candidates:
        path = cwd / candidate
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            continue


def _resolve_backend_and_key() -> tuple[str, str]:
    _maybe_load_env()
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("DEV_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    api_key = os.getenv("SYNTH_API_KEY")
    if not backend or not api_key:
        pytest.skip("SYNTH_API_KEY and backend base URL required for CLI train integration test")
    return backend.rstrip("/"), api_key


def _write_dataset(tmp_dir: Path) -> Path:
    dataset_path = tmp_dir / "train_dataset.jsonl"
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with a friendly greeting."},
            ],
            "response": "Hello there!",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a careful math assistant."},
                {"role": "user", "content": "What is 12 plus 30?"},
            ],
            "response": "12 plus 30 equals 42.",
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return dataset_path.resolve()


def _write_config(tmp_dir: Path, dataset_path: Path, gpu_count: int) -> Path:
    config_path = tmp_dir / f"multi_gpu_{gpu_count}.toml"
    text = textwrap.dedent(
        f"""
        [job]
        model = \"Qwen/Qwen3-32B\"
        data = \"{dataset_path.as_posix()}\"

        [compute]
        gpu_type = \"H100\"
        gpu_count = {gpu_count}
        nodes = 1
        variant = \"H100-4x\"
        gpus_per_node = {gpu_count}

        [data.topology]
        container_count = {gpu_count}
        gpus_per_node = {gpu_count}
        total_gpus = {gpu_count}
        nodes = 1
        variant = \"H100-4x\"

        [training]
        mode = \"full_finetune\"
        use_qlora = false

        [training.validation]
        enabled = false
        evaluation_strategy = \"steps\"
        eval_steps = 100
        save_best_model_at_end = false
        metric_for_best_model = \"val.loss\"
        greater_is_better = false

        [hyperparameters]
        n_epochs = 1
        train_kind = \"fft\"
        per_device_batch = 1
        gradient_accumulation_steps = 16
        sequence_length = 4096
        learning_rate = 5e-6
        warmup_ratio = 0.03
        global_batch = 64

        [hyperparameters.parallelism]
        fsdp = true
        fsdp_sharding_strategy = \"full_shard\"
        fsdp_auto_wrap_policy = \"transformer_block\"
        fsdp_use_orig_params = true
        tensor_parallel_size = 1
        pipeline_parallel_size = 1
        bf16 = true
        fp16 = false
        use_deepspeed = false
        activation_checkpointing = true
        """
    ).strip()
    config_path.write_text(text + "\n", encoding="utf-8")
    return config_path.resolve()


def _write_env_file(tmp_dir: Path, api_key: str) -> Path:
    env_path = tmp_dir / "train.env"
    env_path.write_text(f"SYNTH_API_KEY={api_key}\n", encoding="utf-8")
    return env_path.resolve()


def _extract_job_id(stdout: str) -> str:
    match = _JOB_ID_PATTERN.search(stdout)
    if not match:
        pytest.fail("Failed to extract job id from CLI output")
    return match.group(1).strip()


def _run_async(coro: Awaitable[Any]) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:  # pragma: no cover - safety net for nested loops
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def _wait_for_job_success(
    backend_base: str,
    api_key: str,
    job_id: str,
    *,
    timeout_s: float,
    interval_s: float,
) -> dict[str, Any]:
    async with JobsClient(base_url=backend_base, api_key=api_key, timeout=timeout_s) as client:
        deadline = time.time() + timeout_s
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = await client._http.get(f"/api/learning/jobs/{job_id}")
            status = str(last.get("status", "")).lower()
            if status in {"succeeded", "completed"}:
                return last
            if status in {"failed", "errored", "error", "cancelled", "canceled"}:
                raise AssertionError(f"Job {job_id} ended unsuccessfully: {last}")
            await asyncio.sleep(max(interval_s, 1.0))
        raise AssertionError(f"Timed out waiting for job {job_id} to succeed; last={last}")


@pytest.mark.parametrize("gpu_count", (4,))
def test_cli_train_multi_gpu(tmp_path: Path, gpu_count: int) -> None:
    backend_base, api_key = _resolve_backend_and_key()

    dataset_path = _write_dataset(tmp_path)
    config_path = _write_config(tmp_path, dataset_path, gpu_count)
    env_file = _write_env_file(tmp_path, api_key)

    poll_timeout = os.getenv("SYNTH_TRAIN_TEST_POLL_TIMEOUT", "900")
    poll_interval = os.getenv("SYNTH_TRAIN_TEST_POLL_INTERVAL", "10")
    try:
        poll_timeout_s = float(poll_timeout)
    except ValueError:
        poll_timeout_s = 900.0
    try:
        poll_interval_s = float(poll_interval)
    except ValueError:
        poll_interval_s = 10.0
    timeout_buffer = int(poll_timeout_s) + 120

    cmd = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "sft",
        "--config",
        str(config_path),
        "--dataset",
        str(dataset_path),
        "--env-file",
        str(env_file),
        "--backend",
        backend_base,
        "--poll-timeout",
        poll_timeout,
        "--poll-interval",
        poll_interval,
        "--no-poll",
    ]

    env = os.environ.copy()
    env.setdefault("SYNTH_GPU_TYPE", "H100-4x")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_buffer,
    )

    if proc.returncode != 0:
        pytest.fail(
            "CLI train command failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    assert "✓ Job created" in proc.stdout
    assert "Qwen/Qwen3-32B" in proc.stdout
    assert "\"fsdp\": true" in proc.stdout
    assert f"gpu_count = {gpu_count}" in config_path.read_text(encoding="utf-8")

    job_id = _extract_job_id(proc.stdout)

    final = _run_async(
        _wait_for_job_success(
            backend_base,
            api_key,
            job_id,
            timeout_s=poll_timeout_s,
            interval_s=poll_interval_s,
        )
    )
    assert str(final.get("status", "")).lower() in {"succeeded", "completed"}
    assert final.get("model_id") == "Qwen/Qwen3-32B"
    metadata = (final.get("metadata") or {}).get("effective_config", {})
    compute = metadata.get("compute", {})
    assert compute.get("gpu_type") == "H100"
    assert compute.get("gpu_count") == gpu_count
    assert compute.get("variant") == "H100-4x"
    parallelism = (final.get("hyperparameters") or {}).get("parallelism", {})
    assert parallelism.get("fsdp") is True
