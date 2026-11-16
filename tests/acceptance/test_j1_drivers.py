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
from synth_ai import SynthClient

pytestmark = pytest.mark.acceptance
RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"


@contextmanager
def fixture_api(contract, *, coherent=False):
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
