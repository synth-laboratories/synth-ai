import json

import httpx
import pytest
from pydantic import ValidationError

from synth_ai import AsyncSynthClient
from synth_ai.models import LaunchRequest, RunProfileCatalog


@pytest.mark.parametrize("field", ["program", "profile_overrides", "method"])
def test_old_selection_fields_are_removed_from_public_request(field):
    with pytest.raises(ValidationError):
        LaunchRequest.model_validate(
            {"limits": {"maximum_charge": {"amount": "1", "currency": "USD"}}, field: {}}
        )


@pytest.mark.parametrize("options", [{"timeout": 0}, {"poll_interval": 0}, {"until": "unknown"}])
async def test_invalid_wait_options_never_launch(options):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: pytest.fail("launch sent"))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises(ValueError):
                await api.swarms.run("task", budget="1", **options)


async def test_named_profile_and_research_task_reach_launch(contract):
    calls = []
    response = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(202, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            await api.swarms.launch("task", budget="1")
            await api.swarms.launch("task", budget="1", profile="swarms-v2")
            await api.swarms.launch(
                "task", budget="1", profile="fork-flush", research_task="published-task"
            )
    assert "profile" not in calls[0]
    assert calls[1]["profile"] == "swarms-v2"
    assert calls[2]["research_task"] == "published-task"
    assert all(
        "program" not in call and "method" not in call and "profile_overrides" not in call
        for call in calls
    )


async def test_profile_discovery_preserves_unavailability_reasons():
    catalog = {
        "default_profile": "swarms-v2",
        "profiles": [
            {
                "id": "swarms-v2",
                "revision": "1",
                "name": "Swarms v2",
                "description": "General",
                "requires_research_task": False,
                "default": True,
                "available": True,
            },
            {
                "id": "fork-flush",
                "revision": "1",
                "name": "Fork & Flush",
                "description": "Research",
                "requires_research_task": True,
                "default": False,
                "available": False,
                "unavailable_reason": "research_method_revoked",
            },
        ],
    }

    def handle(request):
        assert request.url.path == "/v1/profiles"
        assert request.url.params["research_task"] == "task"
        return httpx.Response(200, json=catalog)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            got = await api.swarms.profiles(research_task="task")
    assert isinstance(got, RunProfileCatalog)
    assert got.profiles[1].unavailable_reason == "research_method_revoked"


async def test_future_profile_requirements_come_from_discovery(contract):
    snapshot = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    launches = []

    def handle(request):
        if request.url.path == "/v1/profiles":
            return httpx.Response(
                200,
                json={
                    "default_profile": "future-general",
                    "profiles": [
                        {
                            "id": "future-research",
                            "revision": "1",
                            "name": "Research",
                            "description": "Future",
                            "requires_research_task": True,
                            "default": False,
                            "available": True,
                        }
                    ],
                },
            )
        launches.append(json.loads(request.content))
        return httpx.Response(202, json=snapshot)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "key", base_url="https://example.test", http_client=http
        ) as api:
            # Unknown future IDs are sent to the authoritative server without discovery.
            await api.swarms.launch("task", budget="1", profile="future-general")
            await api.swarms.profiles()
            with pytest.raises(ValueError, match="requires research_task"):
                await api.swarms.launch("task", budget="1", profile="future-research")
            await api.swarms.launch(
                "task", budget="1", profile="future-research", research_task="task"
            )
    assert len(launches) == 2
    assert launches[-1]["research_task"] == "task"
