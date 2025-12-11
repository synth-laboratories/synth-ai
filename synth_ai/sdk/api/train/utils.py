from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests

from synth_ai.core.telemetry import log_error, log_info

try:
    sft_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft"))
    collect_sft_jsonl_errors = cast(
        Callable[..., list[dict[str, Any]]], sft_module.collect_sft_jsonl_errors
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load SFT JSONL helpers") from exc

from synth_ai.core.ssl import SSLConfig

REPO_ROOT = Path(__file__).resolve().parents[3]

class TrainError(RuntimeError):
    """Raised for interactive CLI failures."""


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:  # pragma: no cover - guarded by CLI
        raise TrainError(f"Config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - malformed input
        raise TrainError(f"Failed to parse TOML: {path}\n{exc}") from exc


def mask_value(value: str | None) -> str:
    if not value:
        return "<unset>"
    value = str(value)
    if len(value) <= 6:
        return "****"
    return f"{value[:4]}…{value[-2:]}"


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?(?P<key>[A-Za-z0-9_]+)\s*=\s*(?P<value>.*)$")


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _ENV_LINE.match(line)
        if not m:
            continue
        raw = m.group("value").strip()
        if raw and raw[0] == raw[-1] and raw[0] in {'"', "'"} and len(raw) >= 2:
            raw = raw[1:-1]
        data[m.group("key")] = raw
    return data


def write_env_value(path: Path, key: str, value: str) -> None:
    existing = []
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    updated = False
    new_lines: list[str] = []
    for line in existing:
        m = _ENV_LINE.match(line)
        if m and m.group("key") == key:
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@dataclass(slots=True)
class CLIResult:
    code: int
    stdout: str
    stderr: str


def run_cli(
    args: Iterable[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CLIResult:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        env=dict(os.environ, **(env or {})),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CLIResult(code=proc.returncode, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())


def http_post(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    json_body: Any | None = None,
    timeout: float = 60.0,
) -> requests.Response:
    ctx: dict[str, Any] = {"url": url, "timeout": timeout, "method": "POST"}
    log_info("http_post", ctx=ctx)
    try:
        resp = requests.post(
            url,
            headers=dict(headers or {}),
            json=json_body,
            timeout=timeout,
            verify=SSLConfig.get_verify_setting(),
        )
        ctx["status_code"] = resp.status_code
        log_info("http_post completed", ctx=ctx)
        return resp
    except Exception as exc:
        ctx["error"] = type(exc).__name__
        log_error("http_post failed", ctx=ctx)
        raise


def http_get(
    url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 30.0
) -> requests.Response:
    ctx: dict[str, Any] = {"url": url, "timeout": timeout, "method": "GET"}
    log_info("http_get", ctx=ctx)
    try:
        resp = requests.get(
            url,
            headers=dict(headers or {}),
            timeout=timeout,
            verify=SSLConfig.get_verify_setting(),
        )
        ctx["status_code"] = resp.status_code
        log_info("http_get completed", ctx=ctx)
        return resp
    except Exception as exc:
        ctx["error"] = type(exc).__name__
        log_error("http_get failed", ctx=ctx)
        raise


def post_multipart(
    url: str, *, api_key: str, file_field: str, file_path: Path, purpose: str = "fine-tune"
) -> requests.Response:
    ctx: dict[str, Any] = {
        "url": url,
        "file_path": str(file_path),
        "file_field": file_field,
        "purpose": purpose,
    }
    log_info("post_multipart", ctx=ctx)
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {file_field: (file_path.name, file_path.read_bytes(), "application/jsonl")}
    data = {"purpose": purpose}
    try:
        resp = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=300,
            verify=SSLConfig.get_verify_setting(),
        )
        ctx["status_code"] = resp.status_code
        log_info("post_multipart completed", ctx=ctx)
        return resp
    except Exception as exc:
        ctx["error"] = type(exc).__name__
        log_error("post_multipart failed", ctx=ctx)
        raise


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(secs):02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{int(hours)}h{int(mins):02d}m"


def validate_sft_jsonl(path: Path, *, max_errors: int = 20) -> None:
    if not path.exists():
        raise TrainError(f"Dataset not found: {path}")

    issues = collect_sft_jsonl_errors(path, min_messages=1, max_errors=max_errors)
    if not issues:
        return

    truncated = max_errors is not None and len(issues) >= max_errors
    suffix = "" if not truncated else f" (showing first {max_errors} issues)"
    details = "\n - ".join(cast("list[str]", issues))
    raise TrainError(f"{path}: Dataset validation failed{suffix}:\n - {details}")


def limit_jsonl_examples(src: Path, limit: int) -> Path:
    if limit <= 0:
        raise TrainError("Example limit must be positive")
    if not src.exists():
        raise TrainError(f"Dataset not found: {src}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sft_subset_"))
    dest = tmp_dir / f"{src.stem}.head{limit}{src.suffix}"

    written = 0
    with src.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            written += 1
            if written >= limit:
                break

    if written == 0:
        raise TrainError("Subset dataset is empty; check limit value")

    return dest


def ensure_api_base(base: str) -> str:
    base = base.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"
    return base


def preview_json(data: Any, limit: int = 600) -> str:
    try:
        return json.dumps(data, indent=2)[:limit]
    except Exception:
        return str(data)[:limit]


def sleep(seconds: float) -> None:
    time.sleep(seconds)


__all__ = [
    "CLIResult",
    "REPO_ROOT",
    "TrainError",
    "ensure_api_base",
    "fmt_duration",
    "http_get",
    "http_post",
    "load_toml",
    "mask_value",
    "post_multipart",
    "preview_json",
    "read_env_file",
    "run_cli",
    "sleep",
    "limit_jsonl_examples",
    "validate_sft_jsonl",
    "write_env_value",
]
