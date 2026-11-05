import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient


@pytest.mark.parametrize("synchronous", [False, True])
async def test_seven_public_operations_through_typed_clients(contract, synchronous):
    calls = []
    operations = {}
    for path, item in contract["paths"].items():
        for method, op in item.items():
            if "operationId" in op:
                operations[op["operationId"]] = (method.upper(), path, op)
    expected = [
        "pauseRun",
        "resumeRun",
        "getRunProgram",
        "getRunConfiguration",
        "listRunTasks",
        "createUsageExport",
        "getUsageExport",
    ]
    run_id = "run_AAAAAAAAAAAAAAAAAAAAAA"
    export_id = "exp_AAAAAAAAAAAAAAAAAAAAAA"

    def handle(request):
        oid = expected[len(calls)]
        method, path, op = operations[oid]
        assert request.method == method
        assert request.url.path == path.replace("{run_id}", run_id).replace(
            "{export_id}", export_id
        )
        if oid in {"pauseRun", "resumeRun"}:
            assert request.content == b""
            assert request.headers["Idempotency-Key"] == oid
        if oid == "listRunTasks":
            assert request.url.params["limit"] == "1"
        if oid == "createUsageExport":
            assert request.headers["Idempotency-Key"] == oid
            body = json.loads(request.content)
            assert body["format"] == "csv"
            assert body["since"].startswith("2026-11-01T00:00:00")
        calls.append(oid)
        code = "202" if method == "POST" else "200"
        payload = op["responses"][code]["content"]["application/json"]["example"]
        return httpx.Response(int(code), json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        options = {"base_url": "https://example.test", "http_client": http}
        if synchronous:
            with SynthClient("fixture", **options) as client:
                paused = client.swarms.pause(run_id, idempotency_key="pauseRun")
                client.swarms.resume(run_id, idempotency_key="resumeRun")
                client.swarms.program(run_id)
                client.swarms.configuration(run_id)
                client.swarms.tasks(run_id, limit=1)
                client.account.export_usage(
                    since="2026-11-01T00:00:00Z",
                    until="2026-11-02T00:00:00Z",
                    idempotency_key="createUsageExport",
                )
                client.account.usage_export(export_id)
        else:
            async with AsyncSynthClient("fixture", **options) as client:
                paused = await client.swarms.pause(run_id, idempotency_key="pauseRun")
                await client.swarms.resume(run_id, idempotency_key="resumeRun")
                await client.swarms.program(run_id)
                await client.swarms.configuration(run_id)
                await client.swarms.tasks(run_id, limit=1)
                await client.account.export_usage(
                    since="2026-11-01T00:00:00Z",
                    until="2026-11-02T00:00:00Z",
                    idempotency_key="createUsageExport",
                )
                await client.account.usage_export(export_id)
    assert calls == expected
    assert paused.state == "accepted"
    assert paused.execution_closed is False
    assert paused.accounting_final is False
