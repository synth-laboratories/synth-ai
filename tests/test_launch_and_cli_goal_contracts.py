"""Fail-before -> pass-after proofs for typed launch validation, --goal-file/stdin, and mcp_hosted auth (SYN-5942)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import httpx
import pytest

from synth_ai import AsyncSynthClient, cli
from synth_ai._generated.models import RoleAssignment
from synth_ai._generated.policy import MESSAGE_MAX_BYTES
from synth_ai.errors import LaunchValidationError, SynthError


def _forbid_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"invalid launch reached the network: {request.method} {request.url}")


@pytest.mark.parametrize(
    ("message", "kwargs"),
    [
        (None, {"budget": "1.00"}),
        ("   ", {"budget": "1.00"}),
        ("x" * (MESSAGE_MAX_BYTES + 1), {"budget": "1.00"}),
        ("solve", {"budget": "-1.00"}),
        ("solve", {"budget": "not-a-number"}),
        ("solve", {"budget": "1.00", "domicile_requirement": "mars_only"}),
        ("solve", {"budget": "1.00", "profile": "fork-flush", "research_task": "task_1"}),
        ("solve", {"budget": "1.00", "profile": "swarms-v2", "research_task": "task_1"}),
        (
            "solve",
            {
                "budget": "1.00",
                "role_assignments": {"worker": RoleAssignment(inference_profile="luna-high")},
            },
        ),
        ("solve", {"budget": "1.00", "optimizer": {"topology": "swarms_v2", "version": 0}}),
    ],
)
async def test_all_contract_invalid_launches_raise_typed_synth_error(
    message: object, kwargs: dict[str, object]
) -> None:
    """Every contract-invalid launch must raise typed LaunchValidationError (a SynthError), never raw ValidationError or bare ValueError."""
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(_forbid_network)) as http,
        AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client,
    ):
        with pytest.raises(LaunchValidationError) as caught:
            await client.swarms.launch(
                cast("str", message),
                **kwargs,  # ty: ignore[invalid-argument-type]
            )
    error = caught.value
    assert isinstance(error, SynthError)
    assert isinstance(error, ValueError)
    assert error.code == "invalid_launch_request"
    assert error.status == 422
    assert error.service == "synth-ai"
    assert error.outcome == "not_accepted"


def test_cli_goal_file_flag_and_oversize_over_1mib_avoids_e2big(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`synth run --goal-file PATH` handles >1 MiB goals without OS E2BIG and refuses with typed JSON."""
    oversize_goal = tmp_path / "goal.txt"
    oversize_goal.write_text("x" * (MESSAGE_MAX_BYTES + 4096), encoding="utf-8")
    exit_code = cli.main(["run", "--goal-file", str(oversize_goal), "--budget", "1.00", "--json"])
    assert exit_code == cli.EXIT_CLIENT_ERROR
    error = json.loads(capsys.readouterr().err)
    assert error["code"] in {"invalid_launch_request", "client_error"}
    assert "x" * 64 not in error["detail"]


def test_cli_goal_file_dash_reads_stdin_including_oversize_over_1mib(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`synth run --goal-file -` reads from stdin so >1 MiB goals never pass through argv."""
    payload = ("y" * (MESSAGE_MAX_BYTES + 4096)).encode("utf-8")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"))
    exit_code = cli.main(["run", "--goal-file", "-", "--budget", "1.00", "--json"])
    assert exit_code == cli.EXIT_CLIENT_ERROR
    error = json.loads(capsys.readouterr().err)
    assert error["code"] in {"invalid_launch_request", "client_error"}
    assert "y" * 64 not in error["detail"]


def test_cli_piped_stdin_subprocess_handles_oversize_over_1mib_without_e2big(
    tmp_path: Path,
) -> None:
    """Piped stdin to real `synth run` subprocess handles >1 MiB input cleanly without E2BIG."""
    oversize = "z" * (MESSAGE_MAX_BYTES + 4096)
    completed = subprocess.run(
        [sys.executable, "-m", "synth_ai.cli", "run", "--budget", "1.00", "--json"],
        input=oversize,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == cli.EXIT_CLIENT_ERROR
    error = json.loads(completed.stderr)
    assert error["code"] in {"invalid_launch_request", "client_error"}


def test_cli_goal_file_dash_launches_valid_goal_from_stdin(
    contract: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`synth run --goal-file -` forwards a valid stdin goal into createRun."""
    paths = cast(
        "dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, object]]]]]]]]",
        contract["paths"],
    )
    snapshot = dict(
        paths["/v1/runs"]["post"]["responses"]["202"]["content"]["application/json"]["example"]
    )
    snapshot["run_id"] = "run_BBBBBBBBBBBBBBBBBBBBBB"
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(202, json=snapshot)

    monkeypatch.setenv("SYNTH_API_KEY", "fixture-key")
    monkeypatch.setattr(
        sys, "stdin", io.TextIOWrapper(io.BytesIO(b"Solve via stdin"), encoding="utf-8")
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code = cli.main(
        [
            "--base-url",
            "https://example.test",
            "run",
            "--goal-file",
            "-",
            "--budget",
            "2.00",
            "--json",
        ],
        http_client=http,
    )
    assert code == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["run_id"] == "run_BBBBBBBBBBBBBBBBBBBBBB"
    assert seen and seen[0]["message"] == "Solve via stdin"


def test_hosted_endpoint_rejects_blank_or_header_breaking_bearer() -> None:
    from scripts.acceptance.mcp_transport import HostedEndpoint

    with pytest.raises(ValueError, match="bearer"):
        HostedEndpoint("http://127.0.0.1:8080/mcp", "")
    with pytest.raises(ValueError, match="bearer"):
        HostedEndpoint("http://127.0.0.1:8080/mcp", "bad\ntoken")
    with pytest.raises(ValueError, match="bearer"):
        HostedEndpoint("http://127.0.0.1:8080/mcp", "   ")
