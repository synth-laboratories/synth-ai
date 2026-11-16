"""Public-model fixtures through the real SDK and an actual CLI subprocess.

No provider or slot is used. OpenAPI examples are the fixture authority; their
different run IDs deliberately exercise cross-run rejection.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.acceptance.j1 import AdapterError, CLIDriver, SDKDriver, observe
from scripts.acceptance.launch_journal import LaunchJournal, LaunchJournalError
from synth_ai import SynthClient

pytestmark = pytest.mark.acceptance
RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"


@contextmanager
def fixture_api(contract, *, coherent=False, launch_resolution=False):
    examples = {}
    for path, item in contract["paths"].items():
        if "get" in item and path in {
            "/v1/runs/{run_id}",
            "/v1/runs/{run_id}/report",
            "/v1/runs/{run_id}/result",
            "/v1/runs/{run_id}/outputs",
        }:
            examples[path.replace("{run_id}", RUN)] = item["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
    if coherent:
        # Join examples for a parser/adapter test only; this is not a recorded
        # customer run or a cached acceptance receipt.
        examples = json.loads(json.dumps(examples))
        report = examples[f"/v1/runs/{RUN}/report"]
        report["run_id"] = RUN
        examples[f"/v1/runs/{RUN}/result"]["charge"]["amount"] = report["settlement"]["total"]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = json.dumps(examples[self.path.split("?")[0]]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("Idempotency-Key")))
            response = json.loads(json.dumps(examples[f"/v1/runs/{RUN}"]))
            if launch_resolution:
                response["optimizer"] = {
                    "topology": "swarms_v2",
                    "version": 2,
                    "params": {},
                    "reason": "environment_default",
                    "definition_digest": "sha256:" + "a" * 64,
                }
            raw = json.dumps(response).encode()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_sdk_reads_real_public_models(contract):
    with (
        fixture_api(contract) as (origin, requests),
        SynthClient("fixture", base_url=origin) as client,
    ):
        driver = SDKDriver(client)
        assert driver.snapshot(RUN).run_id == RUN
        assert driver.report(RUN).settlement.total.amount == "0.147061330000"
        assert driver.result(RUN).run.run_id == RUN
        assert driver.outputs(RUN) == ()
    assert len(requests) == 4


def test_real_cli_subprocess_reads_same_public_models(contract):
    import sys

    root = Path(__file__).parents[2]
    with fixture_api(contract) as (origin, requests):
        driver = CLIDriver(
            python=Path(sys.executable),
            environment={
                **os.environ,
                "SYNTH_LOCAL_API_BASE_URL": origin,
                "SYNTH_API_KEY": "fixture",
            },
            working_directory=root,
        )
        assert driver.snapshot(RUN).run_id == RUN
        assert driver.report(RUN).settlement.total.amount == "0.147061330000"
        assert driver.result(RUN).run.run_id == RUN
        assert driver.outputs(RUN) == ()
    assert len(requests) == 4


def test_mixed_run_fixture_is_not_normalized_into_success(contract, tmp_path):
    with fixture_api(contract) as (origin, _), SynthClient("fixture", base_url=origin) as client:
        with pytest.raises(AdapterError, match="observation_run_mismatch"):
            observe(SDKDriver(client), run_id=RUN, output_directory=tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_cli_uses_explicit_local_profile_and_ignores_ambient_origin(contract):
    import sys

    root = Path(__file__).parents[2]
    with fixture_api(contract) as (origin, requests):
        driver = CLIDriver(
            python=Path(sys.executable),
            environment={
                **os.environ,
                "SYNTH_LOCAL_API_BASE_URL": origin,
                "SYNTH_API_BASE_URL": "https://must-not-contact.invalid",
                "SYNTH_API_KEY": "fixture",
            },
            working_directory=root,
        )
        assert driver.snapshot(RUN).run_id == RUN
    assert len(requests) == 1


def test_supported_observation_retains_missing_journey_projections(contract, tmp_path):
    from scripts.acceptance.j1 import Missing

    with (
        fixture_api(contract, coherent=True) as (origin, _),
        SynthClient("fixture", base_url=origin) as client,
    ):
        result = observe(SDKDriver(client), run_id=RUN, output_directory=tmp_path / "outputs")
    assert Missing.ATTEMPT_TREE in result.missing
    assert Missing.RESULT_MARKDOWN in result.missing
    assert result.result_markdown is None
    assert result.outputs == ()


def test_sdk_launch_auto_preserves_wire_selection_and_mutation_identity(contract, tmp_path):
    with (
        fixture_api(contract, launch_resolution=True) as (origin, requests),
        SynthClient("fixture", base_url=origin) as client,
    ):
        snapshot = SDKDriver(client).launch_auto(
            message="task",
            budget_usd="1.00",
            files=(),
            idempotency_key="j1-sdk-once",
            journal=LaunchJournal(tmp_path / "launch"),
        )
    assert snapshot.optimizer.topology == "swarms_v2"
    assert len(requests) == 1
    assert requests[0][0] == "/v1/runs"
    assert requests[0][1]["optimizer"] == "auto"
    assert requests[0][2] == "j1-sdk-once"


def test_cli_launch_auto_uses_real_subprocess_and_same_wire_selection(contract, tmp_path):
    import sys

    with fixture_api(contract, launch_resolution=True) as (origin, requests):
        driver = CLIDriver(
            python=Path(sys.executable),
            environment={
                **os.environ,
                "SYNTH_LOCAL_API_BASE_URL": origin,
                "SYNTH_API_KEY": "fixture",
            },
            working_directory=Path(__file__).parents[2],
        )
        snapshot = driver.launch_auto(
            message="task",
            budget_usd="1.00",
            files=(),
            idempotency_key="j1-cli-once",
            journal=LaunchJournal(tmp_path / "launch"),
        )
    assert snapshot.optimizer.topology == "swarms_v2"
    assert len(requests) == 1
    assert requests[0][1]["optimizer"] == "auto"
    assert requests[0][2] == "j1-cli-once"


def test_missing_optimizer_receipt_preserves_accepted_run_without_relaunch(contract, tmp_path):
    with (
        fixture_api(contract) as (origin, requests),
        SynthClient("fixture", base_url=origin) as client,
    ):
        with pytest.raises(AdapterError, match="optimizer_resolution_unobserved") as failure:
            SDKDriver(client).launch_auto(
                message="task",
                budget_usd="1.00",
                files=(),
                idempotency_key="j1-missing-receipt",
                journal=LaunchJournal(tmp_path / "launch"),
            )
        assert failure.value.run_id == RUN
    assert len(requests) == 1


def test_launch_intent_prevents_relaunch_after_accepted_response(contract, tmp_path):
    journal = LaunchJournal(tmp_path / "launch")
    with (
        fixture_api(contract, launch_resolution=True) as (origin, requests),
        SynthClient("fixture", base_url=origin) as client,
    ):
        driver = SDKDriver(client)
        accepted = driver.launch_auto(
            message="task",
            budget_usd="1.00",
            files=(),
            idempotency_key="single-launch",
            journal=journal,
        )
        assert journal.read_accepted() == accepted
        with pytest.raises(LaunchJournalError, match="launch_journal_record_exists"):
            driver.launch_auto(
                message="task",
                budget_usd="1.00",
                files=(),
                idempotency_key="single-launch",
                journal=journal,
            )
    assert len(requests) == 1


def test_missing_optimizer_still_retains_accepted_run_receipt(contract, tmp_path):
    journal = LaunchJournal(tmp_path / "launch")
    with (
        fixture_api(contract) as (origin, requests),
        SynthClient("fixture", base_url=origin) as client,
    ):
        with pytest.raises(AdapterError, match="optimizer_resolution_unobserved"):
            SDKDriver(client).launch_auto(
                message="task",
                budget_usd="1.00",
                files=(),
                idempotency_key="retain-accepted",
                journal=journal,
            )
    assert journal.read_accepted().run_id == RUN
    assert len(requests) == 1


@pytest.mark.parametrize("selected", ["sdk", "cli"])
def test_read_only_entrypoint_retains_real_models_without_launch(contract, tmp_path, selected):
    import subprocess
    import sys

    target = tmp_path / selected
    with fixture_api(contract, coherent=True) as (origin, requests):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.acceptance",
                "--driver",
                selected,
                "--run-id",
                RUN,
                "--receipt-dir",
                str(target),
            ],
            cwd=Path(__file__).parents[2],
            env={**os.environ, "SYNTH_LOCAL_API_BASE_URL": origin, "SYNTH_API_KEY": "fixture"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    assert result.returncode == 2, result.stderr
    receipt = json.loads((target / "observation.json").read_text())
    assert receipt["result"] == "Inconclusive"
    assert "attempt_tree_contract_unavailable" in receipt["missing"]
    assert (target / "snapshot.json").stat().st_mode & 0o777 == 0o600
    assert not (target / "launch-accepted.json").exists()
    assert not (target / "launch-intent.json").exists()
    assert len(requests) == 4
    assert all(isinstance(request, str) for request in requests)


def test_cli_child_entrypoint_composes_public_http_client(contract, tmp_path):
    import sys

    # The child constructs a real HTTP client, invokes the actual public CLI
    # and explicitly closes the borrowed client after CLI completion.
    entrypoint = tmp_path / "cli_with_client.py"
    closed = tmp_path / "closed"
    entrypoint.write_text(
        "import asyncio, os, sys\n"
        "from pathlib import Path\n"
        "import httpx\n"
        "from synth_ai.cli import main\n"
        "http = httpx.AsyncClient(follow_redirects=False)\n"
        "try:\n"
        "    status = main(sys.argv[1:], http_client=http)\n"
        "finally:\n"
        "    asyncio.run(http.aclose())\n"
        "Path(os.environ['CLOSE_RECEIPT']).write_text(str(http.is_closed))\n"
        "raise SystemExit(status)\n"
    )
    with fixture_api(contract, launch_resolution=True) as (origin, requests):
        driver = CLIDriver(
            python=Path(sys.executable),
            entrypoint=entrypoint,
            environment={
                **os.environ,
                "SYNTH_LOCAL_API_BASE_URL": origin,
                "SYNTH_API_KEY": "fixture",
                "CLOSE_RECEIPT": str(closed),
            },
            working_directory=Path(__file__).parents[2],
        )
        snapshot = driver.launch_auto(
            message="task",
            budget_usd="1.00",
            files=(),
            idempotency_key="j1-injected-child-once",
            journal=LaunchJournal(tmp_path / "launch"),
        )
    assert snapshot.run_id == RUN
    assert closed.read_text() == "True"
    assert len(requests) == 1
    assert requests[0][1]["optimizer"] == "auto"
    assert requests[0][2] == "j1-injected-child-once"
