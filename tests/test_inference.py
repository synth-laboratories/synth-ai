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
