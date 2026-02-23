from __future__ import annotations

from synth_ai.sdk.container import (
    CandidateValidationIssue,
    ValidateCandidateRequest,
    ValidateCandidateResponse,
)


def test_container_public_exports_include_validate_candidate_contracts() -> None:
    request = ValidateCandidateRequest(
        candidate_id="cand-1",
        artifact_kind="json",
        artifact_payload={"x": 1},
    )
    response = ValidateCandidateResponse(
        status="valid",
        warnings=[CandidateValidationIssue(code="warn")],
    )
    assert request.candidate_id == "cand-1"
    assert response.status == "valid"
    assert response.warnings[0].code == "warn"
