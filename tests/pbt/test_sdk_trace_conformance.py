"""Unit and negative tests for SDK JSONL trace emission, Python TLA+ oracle, and everystate conform wrapper."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from .trace import (
    EverystateStatus,
    TraceConformanceError,
    TraceRecorder,
    TraceSpecKind,
    find_vendor_leaks,
    verify_trace_conformance,
    verify_trace_python_oracle,
)


def _write_fake_everystate(tmp_path: Path, exit_code: int, message: str = "") -> Path:
    script = tmp_path / "fake-everystate"
    script.write_text(f"#!/bin/sh\necho '{message}' >&2\nexit {exit_code}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_sdk_vendor_leak_detector_catches_upstream_providers() -> None:
    for sample in (
        '{"provider": "openai"}',
        '{"provider": "daytona"}',
        '{"provider": "modal"}',
        '{"provider": "openrouter"}',
        '{"route": "hosted.openai_direct"}',
    ):
        assert find_vendor_leaks(sample)
    assert find_vendor_leaks('{"profile_id":"luna-low","model":"gpt-6-luna"}') == []


def test_sdk_trace_oracle_rejects_finality_regression(tmp_path: Path) -> None:
    rec = TraceRecorder(TraceSpecKind.RUN_CONTROL)
    rec.record_init(
        {
            "run_state": "absent",
            "limits_revision": 0,
            "execution_closed": False,
            "accounting_final": False,
            "cleanup_complete": False,
            "message_count": 0,
            "idempotent_replays": 0,
            "idempotency_conflicts": 0,
        }
    )
    rec.record_step(
        "CloseExecutionCompleted",
        {
            "run_state": "completed",
            "limits_revision": 1,
            "execution_closed": True,
            "accounting_final": False,
            "cleanup_complete": False,
            "message_count": 0,
            "idempotent_replays": 0,
            "idempotency_conflicts": 0,
        },
    )
    rec.record_step(
        "ResumeRun",
        {
            "run_state": "running",
            "limits_revision": 1,
            "execution_closed": False,
            "accounting_final": False,
            "cleanup_complete": False,
            "message_count": 0,
            "idempotent_replays": 0,
            "idempotency_conflicts": 0,
        },
    )
    bad = rec.write_jsonl(tmp_path / "bad.jsonl")
    with pytest.raises(TraceConformanceError, match="regressed"):
        verify_trace_python_oracle(bad, TraceSpecKind.RUN_CONTROL)


def test_sdk_everystate_wrapper_maps_conform_and_non_conform_exit_codes(tmp_path: Path) -> None:
    rec = TraceRecorder(TraceSpecKind.HOSTED_OPTIONS)
    rec.record_init(
        {
            "menu_queries": 0,
            "preflight_checks": 0,
            "launch_attempts": 0,
            "vendor_leaks_detected": 0,
            "agreement_violations": 0,
            "last_menu_admissible": False,
            "last_preflight_admissible": False,
            "last_launch_accepted": False,
        }
    )
    rec.record_step(
        "QueryLaunchChoices",
        {
            "menu_queries": 1,
            "preflight_checks": 0,
            "launch_attempts": 0,
            "vendor_leaks_detected": 0,
            "agreement_violations": 0,
            "last_menu_admissible": True,
            "last_preflight_admissible": False,
            "last_launch_accepted": False,
        },
    )
    trace_path = rec.write_jsonl(tmp_path / "hosted.jsonl")

    ok_bin = _write_fake_everystate(tmp_path, 0)
    ok = verify_trace_conformance(
        trace_path, TraceSpecKind.HOSTED_OPTIONS, everystate_bin=str(ok_bin)
    )
    assert ok.ok and ok.everystate_status is EverystateStatus.CONFORMED

    fail_bin = _write_fake_everystate(tmp_path, 20, "step 1 diverges")
    with pytest.raises(TraceConformanceError, match="non_conforming"):
        verify_trace_conformance(
            trace_path, TraceSpecKind.HOSTED_OPTIONS, everystate_bin=str(fail_bin)
        )
