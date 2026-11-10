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
