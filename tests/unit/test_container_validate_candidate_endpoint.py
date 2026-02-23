from __future__ import annotations

from fastapi.testclient import TestClient

from synth_ai.sdk.container._impl.contracts import (
    ValidateCandidateRequest,
    ValidateCandidateResponse,
)
from synth_ai.sdk.container._impl.server import ContainerConfig, create_container


def _base_config() -> ContainerConfig:
    async def _rollout(_rollout_request, _request):
        return {"trace_correlation_id": "corr-1", "reward_info": {"outcome_reward": 1.0}}

    return ContainerConfig(
        app_id="validate-candidate-unit",
        name="Validate Candidate Unit",
        description="Unit tests for optional /validate-candidate endpoint.",
        provide_taskset_description=lambda: {"name": "demo"},
        provide_task_instances=lambda _seeds: [],
        rollout=_rollout,
        require_api_key=False,
        ensure_container_auth=False,
        expose_debug_env=False,
    )


def _request_payload() -> dict[str, object]:
    return {
        "candidate_id": "cand-1",
        "artifact_kind": "json",
        "artifact_payload": {"x": 1},
    }


def test_validate_candidate_endpoint_not_registered_without_executor() -> None:
    app = create_container(_base_config())
    client = TestClient(app)
    response = client.post("/validate-candidate", json=_request_payload())
    assert response.status_code == 404


def test_validate_candidate_endpoint_accepts_mapping_response() -> None:
    config = _base_config()
    observed_request: ValidateCandidateRequest | None = None

    async def _validate_candidate(validation_request: ValidateCandidateRequest, _request):
        nonlocal observed_request
        observed_request = validation_request
        return {"status": "valid", "warnings": [{"code": "minor_warning"}]}

    config.validate_candidate = _validate_candidate
    app = create_container(config)
    client = TestClient(app)
    response = client.post("/validate-candidate", json=_request_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "valid"
    assert body["warnings"] == [{"code": "minor_warning", "message": None, "path": None, "constraint": None}]
    assert observed_request is not None
    assert observed_request.candidate_id == "cand-1"
    assert observed_request.artifact_kind == "json"


def test_validate_candidate_endpoint_accepts_typed_response() -> None:
    config = _base_config()

    async def _validate_candidate(_validation_request, _request):
        return ValidateCandidateResponse(status="valid")

    config.validate_candidate = _validate_candidate
    app = create_container(config)
    client = TestClient(app)
    response = client.post("/validate-candidate", json=_request_payload())
    assert response.status_code == 200
    assert response.json()["status"] == "valid"


def test_validate_candidate_endpoint_rejects_invalid_mapping_response() -> None:
    config = _base_config()

    async def _validate_candidate(_validation_request, _request):
        return {"errors": []}

    config.validate_candidate = _validate_candidate
    app = create_container(config)
    client = TestClient(app)
    response = client.post("/validate-candidate", json=_request_payload())
    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["error"]["code"] == "invalid_validate_candidate_response"
