import json

import httpx

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.cli import parser


def reply(contract):
    return contract["paths"]["/v1/inference/sessions"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["example"]


async def test_session_money_project_and_mutation_identity(contract):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(201, json=reply(contract))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture",
            base_url="https://example.test",
            project_id="proj_AAAAAAAAAAAAAAAAAAAAAA",
            http_client=http,
        ) as client:
            session = await client.inference.session(
                budget="2.00", ttl=600, idempotency_key="trial-once"
            )
    assert session.session_token.startswith("gw_")
    assert requests[0].url.path == "/v1/inference/sessions"
    assert requests[0].headers["Idempotency-Key"] == "trial-once"
    assert json.loads(requests[0].content) == {
        "project_id": "proj_AAAAAAAAAAAAAAAAAAAAAA",
        "preset": "hosted.openrouter_zdr_us",
        "budget": {"amount": "2.00", "currency": "USD"},
        "ttl": 600,
    }


async def test_sync_and_cli_expose_session(contract):
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(201, json=reply(contract)))
    )
    with SynthClient(
        "fixture",
        base_url="https://example.test",
        project_id="proj_AAAAAAAAAAAAAAAAAAAAAA",
        http_client=http,
    ) as client:
        assert client.inference.session(idempotency_key="once").model == "gpt-6-luna"
    args = parser().parse_args(
        [
            "inference",
            "session",
            "--project-id",
            "proj_AAAAAAAAAAAAAAAAAAAAAA",
            "--preset",
            "hosted.openrouter_zdr_us",
            "--budget",
            "2",
            "--shell",
        ]
    )
    assert (args.operation, args.budget, args.shell) == ("createInferenceSession", "2", True)


def test_cli_shell_and_json_request(contract):
    import os
    import shlex
    import subprocess
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    requests = []
    response = reply(contract)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                (
                    self.path,
                    self.headers["Idempotency-Key"],
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            payload = json.dumps(response).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    body = {
        "project_id": "proj_AAAAAAAAAAAAAAAAAAAAAA",
        "budget": {"amount": "0.10", "currency": "USD"},
        "ttl": 60,
        "preset": "hosted.openai_direct",
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "synth_ai.cli",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "inference",
                "session",
                "--json",
                json.dumps(body),
                "--idempotency-key",
                "session-once",
                "--shell",
            ],
            env={**os.environ, "SYNTH_API_KEY": "fixture"},
            text=True,
            capture_output=True,
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert result.returncode == 0, result.stderr
    assert requests == [("/v1/inference/sessions", "session-once", body)]
    exports = dict(shlex.split(line)[1].split("=", 1) for line in result.stdout.splitlines())
    assert exports == {
        "SYNTH_INFERENCE_BASE_URL": response["base_url"],
        "SYNTH_INFERENCE_SESSION_TOKEN": response["session_token"],
    }
