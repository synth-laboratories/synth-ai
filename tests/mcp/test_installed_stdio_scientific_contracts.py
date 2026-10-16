"""SYN-3992/3994/3995/4007: fresh installed SDK through actual MCP stdio.

Only an owned loopback HTTP fixture is used. JSON-RPC, public handlers and real
HTTP codecs run without monkeypatches, providers or customer credentials.
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from synth_ai.sdk.research.contracts.forge.contracts import ExactReference, contract_digest
from synth_ai.sdk.research.contracts.forge.operations import PublicWrite
from synth_ai.sdk.research.contracts.forge.records import Measurement, Result

PROJECT = "00000000-0000-4000-8000-00000000000b"
ORG = "00000000-0000-4000-8000-00000000000a"


def test_installed_mcp_retains_null_receipt_citations_and_closed_arguments__SYN4007(tmp_path):
    record = {
        "authority": "forge",
        "kind": "result",
        "record_id": "result",
        "revision": "1",
        "digest_sha256": "a" * 64,
    }
    citation = {
        "schema_version": "forge.citation-verification.v1",
        "record": record,
        "citations": [
            {"reference": record, "status": "retained", "code": "", "authority_code": ""}
        ],
        "retained": True,
    }
    result = Result(
        trial=ExactReference(
            authority="forge", kind="trial", record_id="trial", revision="1", digest_sha256="b" * 64
        ),
        evaluator=ExactReference(
            authority="orchestra",
            kind="scorer",
            record_id="scorer",
            revision="1",
            digest_sha256="c" * 64,
        ),
        outcome="null",
        measurements=(Measurement(name="accuracy", value=None, unit="fraction"),),
        missing_evidence=("numeric measurement unavailable",),
    )
    record["digest_sha256"] = contract_digest(result)
    values = {
        "operation_id": "stdio-result",
        "record_id": "result",
        "expected_revision": 0,
        "payload": result.model_dump(mode="json"),
    }
    write = PublicWrite(**values, request_digest_sha256=contract_digest(values)).model_dump(
        mode="json"
    )
    received = []
    failures = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            received.append(("GET", self.path, None))
            if self.path.endswith("/records/result/citations?revision=1"):
                self.answer(200, citation)
            else:
                failures.append("unexpected read route")
                self.answer(404, {"detail": {"error_code": "fixture_route_unknown"}})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            received.append(("POST", self.path, payload))
            if not self.path.endswith("/research/execution-operations"):
                failures.append("unexpected write route")
                self.answer(404, {"detail": {"error_code": "fixture_route_unknown"}})
                return
            receipt = {
                "schema_version": "forge.receipt.v1",
                "scope": {"organization_id": ORG, "project_id": PROJECT},
                "operation_id": payload["write"]["operation_id"],
                "payload_digest_sha256": "d" * 64,
                "reference": record,
                "cursor": 1,
                "recorded_at": "2026-10-06T00:00:00Z",
            }
            if payload["write"]["operation_id"] == "stdio-corrupt":
                receipt["reference"] = {**record, "digest_sha256": "f" * 64}
            self.answer(201, receipt)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    corrupt_values = {**values, "operation_id": "stdio-corrupt"}
    corrupt_write = PublicWrite(
        **corrupt_values, request_digest_sha256=contract_digest(corrupt_values)
    ).model_dump(mode="json")
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "research_get_record_citations",
                "arguments": {"project_id": PROJECT, "record_id": "result", "revision": 1},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "research_record_execution_result",
                "arguments": {
                    "project_id": PROJECT,
                    "organization_id": ORG,
                    "admission_id": "adm_" + "0" * 26,
                    "write": write,
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "research_get_record_citations",
                "arguments": {"project_id": PROJECT, "record_id": "result", "unknown": True},
            },
        },
    ]
    requests.append(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "research_record_execution_result",
                "arguments": {
                    "project_id": PROJECT,
                    "organization_id": ORG,
                    "admission_id": "adm_" + "0" * 26,
                    "write": corrupt_write,
                },
            },
        }
    )
    code = (
        "import synth_ai; assert 'site-packages' in synth_ai.__file__; "
        "from synth_ai.mcp.research.server import main; main()"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "SYNTH_API_KEY": "offline-dummy",
        "SYNTH_BACKEND_URL": f"http://127.0.0.1:{server.server_port}",
        "SYNTH_REQUIRE_EXPLICIT_BACKEND": "1",
        "SYNTH_RESEARCH_MCP_ADVANCED_TOOLS": "1",
        "SYNTH_INDEX_MCP_ENABLED": "true",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=environment,
            input="".join(json.dumps(request) + "\n" for request in requests),
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert completed.returncode == 0, completed.stderr
    responses = {
        response["id"]: response for response in map(json.loads, completed.stdout.splitlines())
    }
    advertised = {tool["name"] for tool in responses[2]["result"]["tools"]}
    assert {"research_get_record_citations", "research_record_execution_result"} <= advertised
    assert "result" in responses[3], responses[3]
    assert responses[3]["result"]["structuredContent"] == citation
    assert "result" in responses[4], responses[4]
    assert responses[4]["result"]["structuredContent"]["operation_id"] == "stdio-result"
    assert responses[5]["error"]["code"] == -32602
    assert responses[5]["error"]["data"]["error"] == "tool_arguments_invalid"
    assert responses[5]["error"]["data"]["mutation_applied"] is False
    assert responses[6]["error"]["code"] == -32010
    assert responses[6]["error"]["data"]["error"] == "outcome_uncertain"
    uncertain = responses[6]["error"]["data"]["detail"]
    assert uncertain["operation_id"] == "stdio-corrupt"
    assert uncertain["retryable"] is False and uncertain["mutation_applied"] is None
    assert uncertain["cause_code"] == "scientific_receipt_invalid"
    assert len(received) == 3 and not failures
    payload = received[1][2]["write"]["payload"]
    assert payload["outcome"] == "null" and payload["measurements"][0]["value"] is None
    assert payload["evaluator"]["record_id"] == "scorer"
    assert "producer" not in received[1][2]["write"]
