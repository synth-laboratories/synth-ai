"""Public provenance constraints survive SDK generation and typed parsing."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from scripts.gen_contract import JSONObject, constraint_expression
from synth_ai._generated.models import RunAttemptScore, RunRecordReference, RunResearchNote


def reference(kind, **changes):
    return {
        "source": "research",
        "kind": kind,
        "record_id": "record-1",
        "revision": "revision-1",
        "digest_sha256": "a" * 64,
        **changes,
    }


def score(**changes):
    return {
        "record": reference("result"),
        "trial": reference("trial"),
        "outcome": "inconclusive",
        "measurements": [],
        "evaluator": None,
        "verification": "unverified",
        "scorer_receipt": None,
        "evidence": [],
        "usage": [],
        "missing_evidence": ["No scorer receipt"],
        "limitations": [],
        **changes,
    }


def note(**changes):
    return {
        "record": reference("log_entry"),
        "log": reference("research_log"),
        "category": "observation",
        "text": "A bounded observation.",
        "occurred_at": "2026-11-17T00:00:00Z",
        "correction_of": None,
        "sources": [],
        **changes,
    }


@pytest.mark.parametrize("missing", list(reference("trial")))
def test_record_identity_fields_are_required(missing):
    payload = reference("trial")
    del payload[missing]
    with pytest.raises(ValidationError):
        RunRecordReference.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"digest_sha256": "bad"},
        {"revision": ""},
        {"record_id": "intern-exec:" + "b" * 64},
    ],
)
def test_record_identity_constraints_reject_invalid_provenance(changes):
    with pytest.raises(ValidationError):
        RunRecordReference.model_validate(reference("trial", **changes))


def test_execution_reference_retains_all_identity_fields():
    payload = reference("mloky_execution", source="execution", record_id="intern-exec:" + "b" * 64)
    assert json.loads(RunRecordReference.model_validate(payload).model_dump_json()) == payload


@pytest.mark.parametrize(
    "changes",
    [
        {"record": reference("trial")},
        {"trial": reference("result")},
        {"record": reference("result", source="billing")},
        {"verification": "producer_bound"},
        {"outcome": "measured"},
    ],
)
def test_score_refuses_false_provenance_or_missing_evidence(changes):
    with pytest.raises(ValidationError):
        RunAttemptScore.model_validate(score(**changes))


def measured_score():
    return score(
        outcome="measured",
        verification="producer_bound",
        scorer_receipt=reference("scorer_receipt", source="evaluation"),
        evaluator=reference("evaluator", source="evaluation"),
        evidence=[reference("evidence", source="evaluation")],
        measurements=[{"name": "score", "value": 0.5, "unit": "score", "uncertainty": None}],
    )


def test_measured_and_unverified_scores_round_trip_without_field_loss():
    for payload in [score(), measured_score()]:
        assert json.loads(RunAttemptScore.model_validate(payload).model_dump_json()) == payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluator", None),
        ("evidence", []),
        ("measurements", []),
        ("measurements", [{"name": "score", "value": None, "unit": "score", "uncertainty": None}]),
        ("scorer_receipt", None),
    ],
)
def test_measured_score_requires_each_scientific_proof(field, value):
    payload = measured_score()
    payload[field] = value
    with pytest.raises(ValidationError):
        RunAttemptScore.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"category": "correction"},
        {"category": "correction", "correction_of": reference("trial")},
        {"category": "correction", "correction_of": reference("log_entry", source="billing")},
        {"correction_of": reference("log_entry")},
    ],
)
def test_corrections_require_exact_research_log_entry(changes):
    with pytest.raises(ValidationError):
        RunResearchNote.model_validate(note(**changes))


def test_correction_and_observation_round_trip():
    for payload in [note(), note(category="correction", correction_of=reference("log_entry"))]:
        assert json.loads(RunResearchNote.model_validate(payload).model_dump_json()) == payload


def test_unknown_constraint_vocabulary_fails_generation():
    with pytest.raises(ValueError, match="unsupported schema constraint"):
        constraint_expression(JSONObject.parse({"not": {"const": "unsafe"}}), "self.value")
