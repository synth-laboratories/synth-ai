import httpx

from synth_ai import AsyncSynthClient


async def test_limits_preserves_unknown_and_large_counters(contract):
    body = dict(contract["paths"]["/v1/runs/{run_id}/limits"]["get"]["responses"]["200"]["content"]["application/json"]["example"])
    body["primary"] = [{"dimension": "tokens", "used": 9007199254740993, "remaining": None, "unknown_reserved": 17, "state": "unavailable"}]
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client:
            result = await client.swarms.limits(body["run_id"])
    assert seen[0].method == "GET"
    assert seen[0].url.path.endswith("/limits")
    assert result.primary[0].used == 9007199254740993
    assert result.primary[0].remaining is None
    assert result.primary[0].unknown_reserved == 17


async def test_update_limits_preserves_revision_and_idempotency(contract):
    body = contract["paths"]["/v1/runs/{run_id}/limits"]["get"]["responses"]["200"]["content"]["application/json"]["example"]
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client:
            await client.swarms.update_limits(body["run_id"], expected_revision=3, budget="2.25", maximum_tokens=9007199254740991, wall_time_seconds=86400, pause_retention_seconds=86400, resume=True, idempotency_key="fixed-revision-operation")
    import json
    payload = json.loads(seen[0].content)
    assert payload == {"expected_revision": 3, "limits": {"maximum_charge": {"amount": "2.25", "currency": "USD"}, "maximum_tokens": 9007199254740991, "wall_time_seconds": 86400, "pause_retention_seconds": 86400}, "resume": True}
    assert seen[0].method == "POST"
    assert seen[0].headers["Idempotency-Key"] == "fixed-revision-operation"
