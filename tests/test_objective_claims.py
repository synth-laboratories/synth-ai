"""ObjectiveClaim decodes typed progress/achievement claims with exact Ref evidence offline."""

import json

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms import ObjectiveClaim, ObjectiveResult
from synth_ai.sdk.swarms.types import ObjectiveClaim as TypesObjectiveClaim
from synth_ai.types.refs import Ref

ARTIFACT = Ref.artifact("artifact-1", "3", "a" * 64)
COMMIT = Ref.commit("repo-1", "c" * 40, "b" * 64)


def decode(model, body):
    """Responses arrive as JSON; strict models decode them on the wire path."""
    return model.model_validate_json(json.dumps(body))


def claim(**changes):
    body = {
        "id": "claim-1",
        "kind": "progress",
        "summary": "Half of the criteria hold.",
        "percent_complete": 50,
        "evidence": [ARTIFACT.model_dump(mode="json"), COMMIT.model_dump(mode="json")],
        "by": "worker-1",
    }
    body.update(changes)
    return body


def result(**changes):
    body = {
        "id": "deo-1",
        "kind": "deo",
        "title": "Ship it",
        "statement": "Ship the feature.",
        "required": True,
        "criteria": ["tests pass"],
        "definition_revision": 1,
        "state": "claimed",
    }
    body.update(changes)
    return body


def test_public_exports_are_one_type():
    assert ObjectiveClaim is TypesObjectiveClaim


@pytest.mark.parametrize(("kind", "percent"), [("progress", 50), ("achievement", 100)])
def test_claims_decode_with_exact_ref_evidence(kind, percent):
    decoded = decode(ObjectiveResult, result(claims=[claim(kind=kind, percent_complete=percent)]))
    (typed,) = decoded.claims
    assert isinstance(typed, ObjectiveClaim)
    assert typed.kind == kind and typed.percent_complete == percent
    assert typed.evidence == (ARTIFACT, COMMIT)
    assert typed.evidence[0].revision == "3" and typed.evidence[1].kind == "commit"


def test_claim_round_trips_without_loss():
    typed = decode(ObjectiveClaim, claim())
    assert decode(ObjectiveClaim, typed.model_dump(mode="json")) == typed
    assert typed.model_dump(mode="json")["evidence"][0] == ARTIFACT.model_dump(mode="json")


@pytest.mark.parametrize(
    "evidence",
    [
        [{"authority": "artifacts", "kind": "artifact", "record_id": "a", "revision": "1"}],
        [{**ARTIFACT.model_dump(mode="json"), "digest_sha256": "not-a-digest"}],
        ["artifact-1@latest"],
        [ARTIFACT.model_dump(mode="json")] * 65,
    ],
    ids=["missing-digest", "bad-digest", "alias-string", "too-many"],
)
def test_malformed_evidence_is_refused(evidence):
    with pytest.raises(ValidationError):
        decode(ObjectiveClaim, claim(evidence=evidence))


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "verdict"},
        {"percent_complete": 101},
        {"percent_complete": -1},
        {"percent_complete": "50"},
        {"percent_complete": 50.0},
    ],
    ids=["unknown-kind", "over-100", "negative", "string-percent", "float-percent"],
)
def test_malformed_claim_fields_are_refused(changes):
    with pytest.raises(ValidationError):
        decode(ObjectiveClaim, claim(**changes))


@pytest.mark.parametrize("missing", ["id", "kind", "summary", "percent_complete", "evidence", "by"])
def test_required_claim_fields(missing):
    body = claim()
    del body[missing]
    with pytest.raises(ValidationError):
        decode(ObjectiveClaim, body)


def test_omitted_optionals_default_and_stay_omitted():
    decoded = decode(ObjectiveResult, result())
    assert decoded.claims == () and decoded.verdict is None and decoded.answer is None
    wire = decoded.model_dump(mode="json")
    assert wire["claims"] == [] and "verdict" not in wire and "answer" not in wire
    empty = decode(ObjectiveClaim, claim(evidence=[]))
    assert empty.evidence == ()


def test_responses_tolerate_future_fields():
    body = result(
        claims=[claim(future_field={"x": 1}, kind="achievement", percent_complete=100)],
        future_result_field=True,
    )
    body["claims"][0]["evidence"][0]["alias"] = "latest"
    decoded = decode(ObjectiveResult, body)
    assert decoded.claims[0].kind == "achievement"
    assert decoded.claims[0].evidence == (ARTIFACT, COMMIT)
    assert "future_field" not in decoded.claims[0].model_dump(mode="json")
