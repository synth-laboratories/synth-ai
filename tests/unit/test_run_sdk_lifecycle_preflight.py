from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "managed_research"
    / "run_sdk_lifecycle.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_sdk_lifecycle_example", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_container_url_from_project_snapshot():
    mod = _load_module()
    project = {
        "config_snapshot": {
            "synth_ai": {
                "policy_optimization": {
                    "container_url": "http://127.0.0.1:8102",
                }
            }
        }
    }
    assert (
        mod._extract_container_url_from_project(project)  # noqa: SLF001
        == "http://127.0.0.1:8102"
    )


def test_resolve_eval_health_url_prefers_explicit_arg(monkeypatch):
    mod = _load_module()
    args = SimpleNamespace(eval_health_url="http://localhost:9999/custom")
    project = {
        "synth_ai": {
            "policy_optimization": {
                "container_url": "http://127.0.0.1:8102",
            }
        }
    }
    monkeypatch.setenv("SMR_SYNTH_AI_CONTAINER_URL", "http://127.0.0.1:8103")
    resolved = mod._resolve_eval_health_url(args, project)  # noqa: SLF001
    assert resolved == "http://localhost:9999/custom/health"


def test_resolve_eval_health_url_falls_back_to_project(monkeypatch):
    mod = _load_module()
    args = SimpleNamespace(eval_health_url=None)
    project = {
        "synth_ai": {
            "policy_optimization": {
                "container_url": "http://127.0.0.1:8102",
            }
        }
    }
    monkeypatch.delenv("SMR_SYNTH_AI_CONTAINER_URL", raising=False)
    monkeypatch.delenv("SMR_EVAL_URL", raising=False)
    resolved = mod._resolve_eval_health_url(args, project)  # noqa: SLF001
    assert resolved == "http://127.0.0.1:8102/health"


def test_probe_eval_health_failure_records_status(monkeypatch):
    mod = _load_module()

    class _Resp:
        status_code = 503
        text = "not ready"

    monkeypatch.setattr(mod.requests, "get", lambda *_a, **_kw: _Resp())
    result = mod._probe_eval_health(  # noqa: SLF001
        "http://127.0.0.1:8102/health",
        timeout_seconds=0.1,
        retries=2,
        retry_sleep_seconds=0.0,
    )
    assert result["ok"] is False
    assert result["status_code"] == 503
    assert "HTTP 503" in result["last_error"]


def test_format_eval_health_preflight_failure_is_actionable():
    mod = _load_module()
    msg = mod._format_eval_health_preflight_failure(  # noqa: SLF001
        {
            "url": "http://127.0.0.1:8102/health",
            "attempts": 3,
            "last_error": "connection refused",
        }
    )
    assert "Start the eval server first" in msg
    assert "--no-check-eval-health" in msg
    assert "skip_health_check=True only skips SDK-side" in msg
