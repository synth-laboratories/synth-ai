import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import smoke
from synth_ai import AsyncSynthClient


@pytest.mark.parametrize("state", ["failed", "cancelled", "partial", "running"])
def test_output_does_not_substitute_for_successful_execution(state):
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        smoke.require_successful_run(SimpleNamespace(state=state, result_available=True))


def test_completed_run_is_required():
    smoke.require_successful_run(SimpleNamespace(state="completed"))


async def test_inference_replay_keeps_one_reserved_budget_and_redacts_token(contract):
    response = contract["paths"]["/v1/inference/sessions"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["example"]
    requests = []
    receipt = {"operations": {}}
    saved = []

    def handle(request):
        requests.append(request)
        # The mutation identity is persisted before a request can be sent.
        assert (
            saved and json.loads(saved[-1])["inference_key"] == request.headers["Idempotency-Key"]
        )
        return httpx.Response(201, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:

            def save():
                saved.append(json.dumps(receipt))

            await smoke.qualify_inference(client, "proj_AAAAAAAAAAAAAAAAAAAAAA", receipt, save)
            await smoke.qualify_inference(client, "proj_AAAAAAAAAAAAAAAAAAAAAA", receipt, save)
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert json.loads(requests[0].content)["budget"] == {"amount": "0.10", "currency": "USD"}
    assert json.loads(requests[0].content)["ttl"] == 60
    assert receipt["operations"]["createInferenceSession"] == {"status": "passed"}
    assert all(response["session_token"] not in item for item in saved)


@pytest.mark.parametrize("case", ["existing", "different_api", "too_many_runs", "legacy_schema"])
async def test_smoke_budget_preflight(tmp_path, monkeypatch, case):
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "synth.public-sdk.smoke.v1"
                if case == "legacy_schema"
                else "synth.public-sdk.smoke.v2",
                "base_url": "https://other.example.test"
                if case == "different_api"
                else "https://api.example.test",
                "runs": [{}] * (3 if case == "too_many_runs" else 0),
            }
        )
    )
    with pytest.raises(ValueError):
        await smoke.run(receipt, resume=case != "existing")


@pytest.mark.parametrize(
    "change,accepted",
    [
        ({}, True),
        ({"idempotency_key": "different"}, False),
        ({"hold_released": False}, False),
        ({"gateway_revoked": False}, False),
        ({"charge_final": False}, False),
        ({"observed_at": "2026-11-04T00:00:00Z"}, False),
    ],
)
def test_inference_cleanup_requires_final_matching_evidence(tmp_path, change, accepted):
    receipt = {
        "inference_key": "session-once",
        "inference": {"expires_at": "2026-11-04T00:01:00Z", "cleanup_complete": False},
    }
    proof = {
        "schema": "synth.inference.cleanup-proof.v1",
        "idempotency_key": "session-once",
        "observed_at": "2026-11-04T00:07:00Z",
        "gateway_revoked": True,
        "core_closed": True,
        "hold_released": True,
        "charge_final": True,
        **change,
    }
    path = tmp_path / "closure.json"
    path.write_text(json.dumps(proof))
    if accepted:
        smoke.apply_inference_cleanup_proof(receipt, path)
        assert receipt["inference"]["cleanup_complete"] is True
    else:
        with pytest.raises(ValueError):
            smoke.apply_inference_cleanup_proof(receipt, path)
        assert receipt["inference"]["cleanup_complete"] is False
