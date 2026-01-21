import importlib
import json
import os
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests

try:
    sft_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft"))
    collect_sft_jsonl_errors = cast(
        Callable[..., list[dict[str, Any]]], sft_module.collect_sft_jsonl_errors
    )
    _SFT_AVAILABLE = True
except Exception:  # pragma: no cover - SFT moved to research repo
    collect_sft_jsonl_errors = None  # type: ignore[assignment]
    _SFT_AVAILABLE = False

from synth_ai.sdk.optimization._impl.ssl import SSLConfig

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
    try:
        resp = requests.post(
            url,
            headers=dict(headers or {}),
            json=json_body,
            timeout=timeout,
            verify=SSLConfig.get_verify_setting(),
        )
        return resp
    except Exception:
        raise


def http_get(
    url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 30.0
) -> requests.Response:
    try:
        resp = requests.get(
            url,
            headers=dict(headers or {}),
            timeout=timeout,
            verify=SSLConfig.get_verify_setting(),
        )
        return resp
    except Exception:
        raise


def post_multipart(
    url: str, *, api_key: str, file_field: str, file_path: Path, purpose: str = "fine-tune"
) -> requests.Response:
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
        return resp
    except Exception:
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
    """Validate SFT JSONL file. Requires SFT module (moved to research repo)."""
    if not _SFT_AVAILABLE:
        raise TrainError(
            "SFT validation requires the research repo. Install synth_ai.sdk.learning.sft."
        )

    if not path.exists():
        raise TrainError(f"Dataset not found: {path}")

    issues = collect_sft_jsonl_errors(path, min_messages=1, max_errors=max_errors)  # type: ignore[misc]
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
    "run_cli",
    "sleep",
    "limit_jsonl_examples",
    "validate_sft_jsonl",
]
