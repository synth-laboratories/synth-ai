"""Unified 4-surface E2E suite: Sync SDK, Async SDK, Isolated-Wheel CLI, MCP stdio & hosted /mcp."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import aclosing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.acceptance.j1 import AdapterError
from scripts.acceptance.mcp_driver import MCPRefusal
from scripts.acceptance.mcp_transport import (
    HostedEndpoint,
    SessionToolCaller,
    StdioCommand,
    hosted_session,
    stdio_session,
)
from synth_ai import AsyncSynthClient, RetryPolicy, StreamCheckpoint, SynthClient
from synth_ai._generated.models import (
    OptimizerParams,
    OptimizerPin,
    OptimizerPinTopology,
    RunOutput,
)
from synth_ai._generated.policy import MESSAGE_MAX_BYTES
from synth_ai.models import InputFile, MessageKind, RunInputs
from tests.e2e.contract_server import (
    EVT_ID_1,
    EVT_ID_2,
    ROOT,
    RUN_ID_FAILED,
    RUN_ID_NOT_FOUND,
    RUN_ID_PAUSED,
    RUN_ID_TIMEOUT,
    VALID_API_KEY,
    VALID_MCP_BEARER,
    LocalContractServer,
)
from tests.e2e.specs import CLI_EXIT_SPECS, LAUNCH_SPECS, REFUSAL_SPECS, LaunchSpec, RefusalSpec

if TYPE_CHECKING:
    from collections.abc import Generator

FAST_RETRY = RetryPolicy(
    attempts=3, deadline_seconds=10.0, backoff_seconds=0.01, jitter_seconds=0.0
)


@pytest.fixture(scope="module")
def contract_server() -> Generator[LocalContractServer, None, None]:
    server = LocalContractServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="module")
def isolated_cli_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the exact synth-ai wheel and install it into an isolated venv for CLI E2E."""
    root_tmp = tmp_path_factory.mktemp("isolated-wheel-cli")
    dist_dir = root_tmp / "dist"
    dist_dir.mkdir()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(dist_dir.glob("synth_ai-*.whl"))
    assert len(wheels) == 1
    venv_dir = root_tmp / "venv"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_dir / "bin/python"),
            "--reinstall-package",
            "synth-ai",
            str(wheels[0]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_bin = venv_dir / "bin/synth"
    # Warm up first-import of native extensions (e.g. pydantic_core on macOS) once per isolated venv.
    subprocess.run(
        [str(cli_bin), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"},
    )
    return cli_bin


def _optimizer_for(spec: LaunchSpec) -> OptimizerPin:
    params = (
        OptimizerParams(
            max_tasks=spec.max_tasks,
            max_repairs_per_task=spec.max_repairs_per_task,
        )
        if spec.max_tasks is not None or spec.max_repairs_per_task is not None
        else None
    )
    return OptimizerPin(
        topology=OptimizerPinTopology(spec.topology),
        version=spec.version,
        params=params,
    )


def _create_upload_bundle(directory: Path) -> tuple[Path, Path]:
    single = directory / "brief.md"
    single.write_text("# Task Brief\nCompare topologies.\n", encoding="utf-8")
    bundle = directory / "bundle"
    (bundle / "nested").mkdir(parents=True)
    (bundle / "a.txt").write_text("alpha\n", encoding="utf-8")
    (bundle / "nested" / "b.json").write_text('{"ok": true}\n', encoding="utf-8")
    return single, bundle


# ---------------------------------------------------------------------------
# Surface 1: Sync SDK (SynthClient)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", LAUNCH_SPECS, ids=[s.name for s in LAUNCH_SPECS])
def test_sync_sdk_full_lifecycle_uploads_sse_reconnect_limits_messages_and_outputs(
    contract_server: LocalContractServer, tmp_path: Path, spec: LaunchSpec
) -> None:
    single, bundle = _create_upload_bundle(tmp_path)
    with SynthClient(
        VALID_API_KEY, base_url=contract_server.base_url, retry_policy=FAST_RETRY
    ) as client:
        uploaded_file = client.swarms.upload_file(str(single))
        uploaded_dir = client.swarms.upload_directory(
            str(bundle), idempotency_key=f"sync-dir-{spec.name}"
        )
        assert {item.name for item in uploaded_dir} == {"a.txt", "nested/b.json"}

        run = client.swarms.launch(
            spec.goal,
            budget=spec.budget,
            optimizer=_optimizer_for(spec),
            inputs=RunInputs(
                files=[
                    InputFile(
                        file_id=uploaded_file.file_id,
                        revision_id=uploaded_file.revision_id,
                        digest_sha256=uploaded_file.digest_sha256,
                    )
                ]
            ),
            idempotency_key=f"sync-launch-{spec.name}",
        )
        assert run.optimizer is not None
        assert run.optimizer.topology == spec.topology

        # Live limits + horizon update + message + pause/resume
        before_limits = client.swarms.limits(run.run_id)
        updated_limits = client.swarms.update_limits(
            run.run_id,
            expected_revision=before_limits.revision,
            budget="5.00",
            wall_time_seconds=1200,
            idempotency_key=f"sync-lim-{spec.name}",
        )
        assert updated_limits.revision == before_limits.revision + 1

        sent = client.swarms.send_message(
            run.run_id,
            "Prioritize precision.",
            kind=MessageKind.GUIDANCE,
            idempotency_key=f"sync-msg-{spec.name}",
        )
        assert sent.run_id == run.run_id

        paused = client.swarms.pause(run.run_id, idempotency_key=f"sync-pause-{spec.name}")
        assert paused.command == "pause"
        resumed = client.swarms.resume(run.run_id, idempotency_key=f"sync-resume-{spec.name}")
        assert resumed.command == "resume"

        # SSE follow() across mid-stream retryable event_owner_unavailable reconnect
        checkpoint = StreamCheckpoint(run_id=run.run_id)
        events = list(client.swarms.follow(run.run_id, checkpoint=checkpoint, timeout=5.0))
        assert [event.event_id for event in events] == [EVT_ID_1, EVT_ID_2]
        record = contract_server.state.runs[run.run_id]
        assert record.sse_connections >= 2
        assert record.sse_last_event_ids[:2] == [None, "1"]

        # wait(), report(), result() (`result.md`), 2-page paginated outputs, pinned download
        final = client.swarms.wait(run.run_id, timeout=5.0, poll_interval=0.05)
        assert final.finished
        report = client.swarms.report(run.run_id)
        assert report.run_id == run.run_id
        result = client.swarms.result(run.run_id)
        assert result.answer is not None
        assert spec.topology in result.answer
        assert result.result_file is not None
        assert result.result_file.filename == "result.md"

        outputs: list[RunOutput] = list(client.swarms.iter_outputs(run.run_id))
        assert [item.filename for item in outputs] == ["result.md", "notes.txt"]
        first_ref = outputs[0].reference
        assert first_ref is not None
        dest = tmp_path / "sync_result.md"
        saved = client.swarms.download_output(
            run.run_id,
            outputs[0].output_id,
            str(dest),
            revision_id=str(first_ref["revision_id"]),
            digest_sha256=str(first_ref["digest_sha256"]),
        )
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == first_ref["digest_sha256"]

        # Cancel a separate running run cleanly
        to_cancel = client.swarms.launch("Cancel me", budget="1.00")
        cancelled = client.swarms.cancel(to_cancel.run_id, idempotency_key=f"sync-can-{spec.name}")
        assert cancelled.state == "cancelled"


@pytest.mark.parametrize("refusal", REFUSAL_SPECS, ids=[r.name for r in REFUSAL_SPECS])
def test_sync_sdk_typed_refusal_contracts(
    contract_server: LocalContractServer, refusal: RefusalSpec
) -> None:
    api_key = "sk_invalid" if refusal.status == 401 else VALID_API_KEY

    def _trigger(client: SynthClient) -> None:
        if refusal.status == 404:
            client.swarms.get(RUN_ID_NOT_FOUND)
        else:
            client.swarms.launch(f"REFUSE:{refusal.status}", budget="1.00")

    with SynthClient(api_key, base_url=contract_server.base_url, retry_policy=FAST_RETRY) as client:
        with pytest.raises(refusal.error_class) as caught:
            _trigger(client)
    assert caught.value.status == refusal.status
    assert caught.value.code == refusal.code
    if refusal.retry_after is not None:
        assert caught.value.retry_after_seconds == refusal.retry_after


# ---------------------------------------------------------------------------
# Surface 2: Async SDK (AsyncSynthClient)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", LAUNCH_SPECS, ids=[s.name for s in LAUNCH_SPECS])
async def test_async_sdk_full_lifecycle_uploads_sse_reconnect_limits_messages_and_outputs(
    contract_server: LocalContractServer, tmp_path: Path, spec: LaunchSpec
) -> None:
    single, bundle = _create_upload_bundle(tmp_path)
    async with AsyncSynthClient(
        VALID_API_KEY, base_url=contract_server.base_url, retry_policy=FAST_RETRY
    ) as client:
        uploaded_file = await client.swarms.upload_file(str(single))
        uploaded_dir = await client.swarms.upload_directory(
            str(bundle), idempotency_key=f"async-dir-{spec.name}"
        )
        assert {item.name for item in uploaded_dir} == {"a.txt", "nested/b.json"}

        run = await client.swarms.launch(
            spec.goal,
            budget=spec.budget,
            optimizer=_optimizer_for(spec),
            inputs=RunInputs(
                files=[
                    InputFile(
                        file_id=uploaded_file.file_id,
                        revision_id=uploaded_file.revision_id,
                        digest_sha256=uploaded_file.digest_sha256,
                    )
                ]
            ),
            idempotency_key=f"async-launch-{spec.name}",
        )
        before_limits = await client.swarms.limits(run.run_id)
        updated = await client.swarms.update_limits(
            run.run_id,
            expected_revision=before_limits.revision,
            budget="6.00",
            wall_time_seconds=1800,
            idempotency_key=f"async-lim-{spec.name}",
        )
        assert updated.revision == before_limits.revision + 1

        msg = await client.swarms.send_message(
            run.run_id,
            "Async guidance.",
            kind=MessageKind.GUIDANCE,
            idempotency_key=f"async-msg-{spec.name}",
        )
        assert msg.run_id == run.run_id
        await client.swarms.pause(run.run_id, idempotency_key=f"async-pause-{spec.name}")
        await client.swarms.resume(run.run_id, idempotency_key=f"async-resume-{spec.name}")

        checkpoint = StreamCheckpoint(run_id=run.run_id)
        events = []
        async with aclosing(
            client.swarms.follow(run.run_id, checkpoint=checkpoint, timeout=5.0)
        ) as stream:
            async for item in stream:
                events.append(item)
        assert [event.event_id for event in events] == [EVT_ID_1, EVT_ID_2]

        final = await client.swarms.wait(run.run_id, timeout=5.0, poll_interval=0.05)
        assert final.finished
        report = await client.swarms.report(run.run_id)
        assert report.run_id == run.run_id
        result = await client.swarms.result(run.run_id)
        assert result.result_file is not None
        assert result.result_file.filename == "result.md"

        outputs = [item async for item in client.swarms.iter_outputs(run.run_id)]
        assert [item.filename for item in outputs] == ["result.md", "notes.txt"]
        first_ref = outputs[0].reference
        assert first_ref is not None
        dest = tmp_path / "async_result.md"
        saved = await client.swarms.download_output(
            run.run_id,
            outputs[0].output_id,
            str(dest),
            revision_id=str(first_ref["revision_id"]),
            digest_sha256=str(first_ref["digest_sha256"]),
        )
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == first_ref["digest_sha256"]

        to_cancel = await client.swarms.launch("Cancel async", budget="1.00")
        cancelled = await client.swarms.cancel(
            to_cancel.run_id, idempotency_key=f"async-can-{spec.name}"
        )
        assert cancelled.state == "cancelled"


@pytest.mark.parametrize("refusal", REFUSAL_SPECS, ids=[r.name for r in REFUSAL_SPECS])
async def test_async_sdk_typed_refusal_contracts(
    contract_server: LocalContractServer, refusal: RefusalSpec
) -> None:
    api_key = "sk_invalid" if refusal.status == 401 else VALID_API_KEY

    async def _trigger(client: AsyncSynthClient) -> None:
        if refusal.status == 404:
            await client.swarms.get(RUN_ID_NOT_FOUND)
        else:
            await client.swarms.launch(f"REFUSE:{refusal.status}", budget="1.00")

    async with AsyncSynthClient(
        api_key, base_url=contract_server.base_url, retry_policy=FAST_RETRY
    ) as client:
        with pytest.raises(refusal.error_class) as caught:
            await _trigger(client)
    assert caught.value.status == refusal.status
    assert caught.value.code == refusal.code


# ---------------------------------------------------------------------------
# Surface 3: Isolated-Wheel CLI (`synth`)
# ---------------------------------------------------------------------------


def _run_cli(
    cli_bin: Path,
    base_url: str,
    args: list[str],
    *,
    tmp_path: Path,
    api_key: str = VALID_API_KEY,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYNTH_API_KEY": api_key,
        "SYNTH_AUTH_FILE": str(tmp_path / "absent-auth.json"),
        "SYNTH_STATE_DIR": str(tmp_path / "state"),
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
    }
    if stdin_text is None:
        return subprocess.run(
            [str(cli_bin), "--base-url", base_url, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=30,
            check=False,
        )
    return subprocess.run(
        [str(cli_bin), "--base-url", base_url, *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=30,
        check=False,
    )


def test_isolated_wheel_cli_full_journey_and_exit_contracts(
    contract_server: LocalContractServer, isolated_cli_bin: Path, tmp_path: Path
) -> None:
    single, bundle = _create_upload_bundle(tmp_path)

    # Single-file and directory uploads
    up_file = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["upload", str(single), "--json"],
        tmp_path=tmp_path,
    )
    assert up_file.returncode == 0, up_file.stderr
    assert json.loads(up_file.stdout)[0]["name"] == "brief.md"

    up_dir = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["upload", "--directory", str(bundle), "--idempotency-key", "cli-dir-1", "--json"],
        tmp_path=tmp_path,
    )
    assert up_dir.returncode == 0, up_dir.stderr
    assert {item["name"] for item in json.loads(up_dir.stdout)} == {"a.txt", "nested/b.json"}

    # Launch via --goal-file PATH (swarms_v2) and piped stdin (fork_flush)
    goal_file = tmp_path / "goal.txt"
    goal_file.write_text(LAUNCH_SPECS[0].goal, encoding="utf-8")
    launched = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        [
            "run",
            "--goal-file",
            str(goal_file),
            "--budget",
            "2.50",
            "--optimizer",
            "swarms_v2@1",
            "--max-tasks",
            "4",
            "--max-repairs-per-task",
            "2",
            "--json",
        ],
        tmp_path=tmp_path,
    )
    assert launched.returncode == 0, launched.stderr
    run_id = json.loads(launched.stdout)["run_id"]

    fork_launched = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["run", "--budget", "3.00", "--optimizer", "fork_flush@1", "--json"],
        tmp_path=tmp_path,
        stdin_text=LAUNCH_SPECS[1].goal,
    )
    assert fork_launched.returncode == 0, fork_launched.stderr
    assert json.loads(fork_launched.stdout)["optimizer"]["topology"] == "fork_flush"

    # >1 MiB goal over --goal-file - refuses cleanly without E2BIG
    oversize_res = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["run", "--goal-file", "-", "--budget", "1.00", "--json"],
        tmp_path=tmp_path,
        stdin_text="x" * (MESSAGE_MAX_BYTES + 2048),
    )
    assert oversize_res.returncode == 1
    assert json.loads(oversize_res.stderr)["code"] in {"client_error", "invalid_launch_request"}

    # Live limits, message, pause, resume, SSE logs -f reconnect, report, result, outputs, download
    lim_get = _run_cli(
        isolated_cli_bin, contract_server.base_url, ["limits", run_id, "--json"], tmp_path=tmp_path
    )
    assert lim_get.returncode == 0, lim_get.stderr
    rev = json.loads(lim_get.stdout)["revision"]
    lim_set = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        [
            "limits",
            "set",
            run_id,
            "--expected-revision",
            str(rev),
            "--budget",
            "4.00",
            "--wall",
            "900",
            "--json",
        ],
        tmp_path=tmp_path,
    )
    assert lim_set.returncode == 0, lim_set.stderr
    assert json.loads(lim_set.stdout)["revision"] == rev + 1

    msg_res = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["message", run_id, "CLI guidance", "--json"],
        tmp_path=tmp_path,
    )
    assert msg_res.returncode == 0, msg_res.stderr

    assert (
        _run_cli(
            isolated_cli_bin,
            contract_server.base_url,
            ["pause", run_id, "--json"],
            tmp_path=tmp_path,
        ).returncode
        == 0
    )
    assert (
        _run_cli(
            isolated_cli_bin,
            contract_server.base_url,
            ["resume", run_id, "--json"],
            tmp_path=tmp_path,
        ).returncode
        == 0
    )

    logs_res = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["logs", run_id, "-f", "--json"],
        tmp_path=tmp_path,
    )
    assert logs_res.returncode == 0, logs_res.stderr
    event_ids = [json.loads(line)["event_id"] for line in logs_res.stdout.strip().splitlines()]
    assert event_ids == [EVT_ID_1, EVT_ID_2]

    assert (
        _run_cli(
            isolated_cli_bin,
            contract_server.base_url,
            ["report", run_id, "--json"],
            tmp_path=tmp_path,
        ).returncode
        == 0
    )
    res_cmd = _run_cli(
        isolated_cli_bin, contract_server.base_url, ["result", run_id, "--json"], tmp_path=tmp_path
    )
    assert res_cmd.returncode == 0, res_cmd.stderr
    assert json.loads(res_cmd.stdout)["result_file"]["filename"] == "result.md"

    dl_dir = tmp_path / "downloads"
    dl_cmd = _run_cli(
        isolated_cli_bin,
        contract_server.base_url,
        ["download", run_id, "-o", str(dl_dir), "--json"],
        tmp_path=tmp_path,
    )
    assert dl_cmd.returncode == 0, dl_cmd.stderr
    assert {Path(entry["path"]).name for entry in json.loads(dl_cmd.stdout)} == {
        "result.md",
        "notes.txt",
    }

    # Verify all 6 CLI exit-code contracts (0, 1, 2, 3, 4, 5)
    exit_invocations: dict[str, list[str]] = {
        "success_0": ["status", run_id, "--json"],
        "usage_error_1": ["run", "--unknown-flag", "--json"],
        "wait_timeout_2": ["wait", RUN_ID_TIMEOUT, "--timeout", "0.15", "--json"],
        "run_failed_3": ["wait", RUN_ID_FAILED, "--timeout", "2", "--json"],
        "server_refusal_4": ["run", "REFUSE:422", "--budget", "1.00", "--json"],
        "run_paused_5": ["wait", RUN_ID_PAUSED, "--timeout", "2", "--json"],
    }
    for spec in CLI_EXIT_SPECS:
        result = _run_cli(
            isolated_cli_bin,
            contract_server.base_url,
            exit_invocations[spec.name],
            tmp_path=tmp_path,
        )
        assert result.returncode == spec.expected_exit, (
            f"{spec.name}: expected exit {spec.expected_exit}, got {result.returncode}; stderr={result.stderr}"
        )
        if spec.expected_code is not None:
            assert json.loads(result.stderr)["code"] == spec.expected_code


# ---------------------------------------------------------------------------
# Surface 4: MCP stdio (`synth-ai-mcp`) + Hosted `/mcp` (Streamable HTTP)
# ---------------------------------------------------------------------------


def _exercise_mcp_caller(caller: SessionToolCaller, prefix: str) -> None:
    tools = caller.tools()
    assert {
        "swarms_upload_file",
        "swarms_run_launch",
        "swarms_run_get",
        "swarms_run_limits_update",
        "swarms_run_message_send",
        "swarms_run_pause",
        "swarms_run_resume",
        "swarms_run_cancel",
        "swarms_run_report",
        "swarms_run_result",
        "swarms_run_outputs",
        "swarms_run_output_content",
    } <= tools

    upload = caller.call(
        "swarms_upload_file",
        {
            "body": {"name": "task.md", "media_type": "text/markdown", "content": "# MCP Task\n"},
            "idempotency_key": f"{prefix}-up",
        },
    )
    assert not upload.is_error, upload.text
    file_doc = json.loads(upload.text)

    for spec in LAUNCH_SPECS:
        launch = caller.call(
            "swarms_run_launch",
            {
                "idempotency_key": f"{prefix}-{spec.name}",
                "body": {
                    "message": spec.goal,
                    "limits": {"maximum_charge": {"amount": spec.budget, "currency": "USD"}},
                    "optimizer": _optimizer_for(spec).model_dump(mode="json", exclude_none=True),
                    "inputs": {
                        "files": [
                            {
                                "file_id": file_doc["file_id"],
                                "revision_id": file_doc["revision_id"],
                                "digest_sha256": file_doc["digest_sha256"],
                            }
                        ]
                    },
                },
            },
        )
        assert not launch.is_error, launch.text
        run_id = json.loads(launch.text)["run_id"]
        assert json.loads(launch.text)["optimizer"]["topology"] == spec.topology

        lim = caller.call(
            "swarms_run_limits_update",
            {
                "run_id": run_id,
                "idempotency_key": f"{prefix}-lim-{spec.name}",
                "body": {
                    "expected_revision": 1,
                    "limits": {"wall_time_seconds": 900},
                },
            },
        )
        assert not lim.is_error, lim.text
        assert json.loads(lim.text)["revision"] == 2

        msg = caller.call(
            "swarms_run_message_send",
            {
                "run_id": run_id,
                "idempotency_key": f"{prefix}-msg-{spec.name}",
                "body": {"kind": "guidance", "text": "MCP guidance"},
            },
        )
        assert not msg.is_error, msg.text
        assert not caller.call(
            "swarms_run_pause", {"run_id": run_id, "idempotency_key": f"{prefix}-p-{spec.name}"}
        ).is_error
        assert not caller.call(
            "swarms_run_resume", {"run_id": run_id, "idempotency_key": f"{prefix}-r-{spec.name}"}
        ).is_error

        report = caller.call("swarms_run_report", {"run_id": run_id})
        assert not report.is_error, report.text
        res = caller.call("swarms_run_result", {"run_id": run_id})
        assert not res.is_error, res.text
        assert json.loads(res.text)["result_file"]["filename"] == "result.md"

        page1 = caller.call("swarms_run_outputs", {"run_id": run_id})
        assert not page1.is_error, page1.text
        p1_doc = json.loads(page1.text)
        assert p1_doc["has_more"] is True
        first_out = p1_doc["items"][0]
        content = caller.call(
            "swarms_run_output_content",
            {
                "run_id": run_id,
                "output_id": first_out["output_id"],
                "revision_id": first_out["reference"]["revision_id"],
                "digest_sha256": first_out["reference"]["digest_sha256"],
            },
        )
        assert not content.is_error, content.text
        assert spec.topology in json.loads(content.text)["content"]

        cancelled = caller.call(
            "swarms_run_cancel",
            {"run_id": run_id, "idempotency_key": f"{prefix}-c-{spec.name}"},
        )
        assert not cancelled.is_error, cancelled.text

    # Typed isError refusal contracts through MCP
    refused_404 = caller.call("swarms_run_get", {"run_id": RUN_ID_NOT_FOUND})
    assert refused_404.is_error
    assert json.loads(refused_404.text)["code"] == "run_not_found"

    refused_422 = caller.call(
        "swarms_run_launch",
        {
            "idempotency_key": f"{prefix}-ref422",
            "body": {
                "message": "REFUSE:422",
                "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
            },
        },
    )
    assert refused_422.is_error
    assert json.loads(refused_422.text)["code"] == "launch_budget_below_actor_floor"

    invalid_args = caller.call(
        "swarms_run_launch",
        {"idempotency_key": f"{prefix}-invalid", "body": {"message": ""}},
    )
    assert invalid_args.is_error
    assert json.loads(invalid_args.text)["code"] == "tool_arguments_invalid"


def test_mcp_stdio_full_surface_e2e(contract_server: LocalContractServer, tmp_path: Path) -> None:
    mcp_bin = Path(sys.executable).with_name("synth-ai-mcp")
    command = StdioCommand(
        executable=mcp_bin,
        arguments=("--env", "local"),
        environment={
            "PATH": os.environ.get("PATH", ""),
            "SYNTH_API_KEY": VALID_API_KEY,
            "SYNTH_LOCAL_API_BASE_URL": contract_server.base_url,
            "SYNTH_AUTH_FILE": str(tmp_path / "absent-auth.json"),
        },
        working_directory=ROOT,
    )
    with SessionToolCaller(stdio_session(command)) as caller:
        _exercise_mcp_caller(caller, "stdio")


def test_mcp_hosted_streamable_http_auth_and_full_surface_e2e(
    contract_server: LocalContractServer,
) -> None:
    # 1. Rejected bearer token fails closed with typed TRANSPORT_FAILED (HTTP 401 on /mcp).
    bad_endpoint = HostedEndpoint(contract_server.mcp_url, "wrong-bearer-token")
    with pytest.raises(AdapterError) as caught:
        with SessionToolCaller(hosted_session(bad_endpoint)):
            pass
    assert caught.value.code == MCPRefusal.TRANSPORT_FAILED
    assert any(
        token == "wrong-bearer-token" and status == 401
        for token, status in contract_server.state.mcp_auth_checks
    )

    # 2. Authorized opaque bearer completes initialize, tools/list, and full tools/call lifecycle.
    good_endpoint = HostedEndpoint(contract_server.mcp_url, VALID_MCP_BEARER)
    with SessionToolCaller(hosted_session(good_endpoint)) as caller:
        _exercise_mcp_caller(caller, "hosted")
    assert any(
        token == VALID_MCP_BEARER and status == 200
        for token, status in contract_server.state.mcp_auth_checks
    )
    assert good_endpoint.bearer == VALID_MCP_BEARER
