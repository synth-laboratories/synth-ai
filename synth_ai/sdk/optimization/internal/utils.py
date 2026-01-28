import asyncio
import importlib
import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests

from synth_ai.core.rust_core.urls import ensure_api_base as _ensure_api_base

try:
    sft_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft"))
    collect_sft_jsonl_errors = cast(
        Callable[..., list[dict[str, Any]]], sft_module.collect_sft_jsonl_errors
    )
    _SFT_AVAILABLE = True
except Exception:  # pragma: no cover - SFT moved to research repo
    collect_sft_jsonl_errors = None  # type: ignore[assignment]
    _SFT_AVAILABLE = False

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for optimization utils.") from exc

from synth_ai.sdk.optimization.internal.ssl import SSLConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


class TrainError(RuntimeError):
    """Raised for interactive CLI failures."""


def run_sync(coro: Any, *, allow_nested: bool = False, label: str = "async call") -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if allow_nested:
        try:
            import nest_asyncio  # type: ignore[unresolved-import]

            nest_asyncio.apply()
            return asyncio.run(coro)
        except ImportError as exc:
            raise RuntimeError(
                f"{label} cannot be called from an async context without nest_asyncio."
            ) from exc
    raise RuntimeError(f"{label} cannot be called from an async context.")


def load_toml(path: Path) -> dict[str, Any]:
    try:
        payload = synth_ai_py.load_toml(str(path))
    except FileNotFoundError as exc:  # pragma: no cover - guarded by CLI
        raise TrainError(f"Config not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - malformed input
        raise TrainError(f"Failed to parse TOML: {path}\n{exc}") from exc
    if isinstance(payload, dict):
        return payload
    return dict(payload)


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
        logging.getLogger(__name__).exception("HTTP POST failed: %s", url)
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
        logging.getLogger(__name__).exception("HTTP GET failed: %s", url)
        raise


def http_delete(
    url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 30.0
) -> requests.Response:
    try:
        resp = requests.delete(
            url,
            headers=dict(headers or {}),
            timeout=timeout,
            verify=SSLConfig.get_verify_setting(),
        )
        return resp
    except Exception:
        logging.getLogger(__name__).exception("HTTP DELETE failed: %s", url)
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
        logging.getLogger(__name__).exception("HTTP multipart POST failed: %s", url)
        raise


def parse_json_response(
    response: requests.Response,
    *,
    context: str,
    expect_dict: bool = True,
) -> dict[str, Any]:
    if not response.ok:
        raise RuntimeError(f"{context} failed: HTTP {response.status_code} {response.text}")
    try:
        payload = response.json()
    except Exception as exc:  # pragma: no cover - defensive
        snippet = response.text[:200]
        raise RuntimeError(
            f"{context} returned invalid JSON: HTTP {response.status_code} {snippet}"
        ) from exc
    if expect_dict and not isinstance(payload, dict):
        raise RuntimeError(f"{context} returned unexpected JSON type: {type(payload).__name__}")
    return payload if isinstance(payload, dict) else {}


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
    return _ensure_api_base(base)


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
    "run_sync",
    "ensure_api_base",
    "fmt_duration",
    "http_get",
    "http_delete",
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
