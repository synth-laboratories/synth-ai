import copy
import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._cli_view import report
from synth_ai.models import RunReport


@pytest.mark.parametrize("requirement", ["us_only", "us_or_europe", "unrestricted", None])
async def test_launch_freezes_requested_domicile_without_an_implicit_override(
    contract, requirement
):
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
            await api.swarms.launch("task", budget="1", domicile_requirement=requirement)
    if requirement is None:
        assert "domicile_requirement" not in calls[0]
    else:
        assert calls[0]["domicile_requirement"] == requirement


async def test_invalid_domicile_is_refused_before_upload_or_launch():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("request sent"))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises(ValueError):
                await api.swarms.launch("task", budget="1", domicile_requirement="future")


def test_report_preserves_profile_charge_startup_and_emulated_basis(contract):
    raw = copy.deepcopy(
        contract["paths"]["/v1/runs/{run_id}/report"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    )
    money = {"amount": "0", "currency": "USD"}
    moment = {"status": "observed", "at": "2026-11-11T00:00:00Z"}
    raw["sandboxes"] = [
        {
            "sandbox_id": "sbx_AAAAAAAAAAAAAAAAAAAAAA",
            "role": "worker",
            "state": "destroyed",
            "image": {"status": "resolved", "digest": "sha256:" + "a" * 64},
            "cpu_millis": 4000,
            "memory_bytes": 8 << 30,
            "created": moment,
            "ready": moment,
            "destroyed": moment,
            "observation": "current",
            "prepared": moment,
            "time_to_ready_ms": 1450,
            "lifetime_ms": 600000,
            "usage": {
                "calls": 0,
                "failed_calls": 0,
                "charge": money,
                "subsidized": money,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "model_time_ms": 0,
            },
            "usage_attribution": "attributed",
            "compute_metering": "metered",
            "compute_charge": {"amount": "0.0363975", "currency": "USD"},
            "profile": {
                "actor_id": "worker",
                "profile_id": "4vcpu-8gib-local",
                "profile_digest": "sha256:" + "b" * 64,
                "size_id": "4vcpu-8gib",
                "domicile": "local",
            },
        }
    ]
    raw["sandbox_performance"] = [
        {
            "size_id": "4vcpu-8gib",
            "sample_count": 1,
            "status": "unknown",
            "ready_p50_ms": None,
            "ready_p90_ms": None,
        }
    ]
    parsed = RunReport.model_validate_json(json.dumps(raw))
    assert parsed.sandbox_performance[0].ready_p90_ms is None
    text = report(parsed)
    for expected in ("worker", "4vcpu-8gib", "local", "1450", "0.0363975"):
        assert expected in text
