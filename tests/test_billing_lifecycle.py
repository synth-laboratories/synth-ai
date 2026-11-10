import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._generated.models import Money

pytestmark = pytest.mark.asyncio

SUBSCRIPTION = {
    "plan_id": "standard.monthly",
    "status": "active",
    "current_period_end": "2026-12-01T00:00:00Z",
    "cancel_at_period_end": False,
    "cancel_at": None,
}


async def _client(handle):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = AsyncSynthClient("fixture", base_url="https://example.test", http_client=http)
    return http, client


async def test_subscription_null_and_mutations_on_the_wire():
    seen = []

    def handle(request):
        seen.append(request)
        if request.url.path == "/v1/billing/subscription":
            return httpx.Response(200, content=b"null", headers={"content-type": "application/json"})
        return httpx.Response(200, json=SUBSCRIPTION)

    http, client = await _client(handle)
    async with http:
        assert await client.billing.subscription() is None
        cancelled = await client.billing.cancel_subscription(idempotency_key="k1")
        await client.billing.change_subscription_plan("max.monthly", idempotency_key="k2")
    assert cancelled.status == "active"
    assert seen[1].method == "POST" and seen[1].headers["Idempotency-Key"] == "k1"
    assert json.loads(seen[2].content) == {"plan_id": "max.monthly"}
    assert seen[2].url.path == "/v1/billing/subscription/change"


async def test_put_bodies_keep_explicit_null_and_omit_unset():
    seen = []
    alerts = {"enabled": True, "monthly_amount": None, "thresholds": [], "last_notified": None}

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=alerts)

    http, client = await _client(handle)
    async with http:
        await client.billing.set_spend_alerts(monthly_amount=None)
        await client.billing.set_spend_alerts(
            monthly_amount=Money(amount="5", currency="USD"), enabled=False
        )
    assert seen[0].method == "PUT"
    assert json.loads(seen[0].content) == {"monthly_amount": None}
    assert json.loads(seen[1].content)["enabled"] is False
