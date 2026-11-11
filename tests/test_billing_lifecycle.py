import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._generated.models import Money
from synth_ai.errors import ForbiddenError

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
            return httpx.Response(
                200, content=b"null", headers={"content-type": "application/json"}
            )
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


AUTO_RELOAD = {
    "enabled": False,
    "threshold": None,
    "amount": None,
    "disabled_reason": None,
    "payment_method_on_file": True,
    "disclosure": {
        "version": "auto-reload-2026-11-10",
        "template": "Charge {amount} to {card} when credit falls below {threshold}.",
        "card": {"brand": "visa", "last4": "4242"},
        "limits": {
            "max_amount": {"amount": "1000.00", "currency": "USD"},
            "daily_count": 3,
            "monthly_ceiling": {"amount": "3000.00", "currency": "USD"},
        },
    },
}


async def test_auto_reload_disclosure_and_off_needs_no_version():
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=AUTO_RELOAD)

    http, client = await _client(handle)
    async with http:
        setting = await client.billing.auto_reload()
        await client.billing.set_auto_reload(enabled=False, idempotency_key="k3")
    assert setting.disclosure.version == "auto-reload-2026-11-10"
    assert setting.disclosure.limits.daily_count == 3
    assert seen[1].method == "PUT" and json.loads(seen[1].content) == {"enabled": False}


async def test_enabling_auto_reload_with_an_api_key_is_refused_not_faked():
    seen = []
    problem = {
        "type": "https://docs.usesynth.ai/errors/browser_session_required",
        "title": "browser session required",
        "status": 403,
        "detail": "turning auto-reload on needs a signed-in browser session",
        "instance": "/v1/billing/auto-reload",
        "code": "browser_session_required",
        "service": "synth-backend",
        "stage": "billing",
        "outcome": "not_accepted",
    }

    def handle(request):
        seen.append(request)
        return httpx.Response(
            403, json=problem, headers={"content-type": "application/problem+json"}
        )

    http, client = await _client(handle)
    async with http:
        with pytest.raises(ForbiddenError) as caught:
            await client.billing.set_auto_reload(
                enabled=True,
                threshold=Money(amount="10.00", currency="USD"),
                amount=Money(amount="25.00", currency="USD"),
                disclosure_version="auto-reload-2026-11-10",
                idempotency_key="k4",
            )
    assert caught.value.code == "browser_session_required"
    assert json.loads(seen[0].content)["disclosure_version"] == "auto-reload-2026-11-10"
