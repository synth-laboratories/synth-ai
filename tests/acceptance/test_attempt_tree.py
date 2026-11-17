"""Typed public tree provenance and ancestry, without a live provider."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.acceptance.attempt_tree import Attempt, TreeError, project_attempts
from synth_ai.models import RunAttemptTreeAvailable, RunAttemptTreeUnavailable

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"


def reference(kind, identity):
    return {
        "source": "research",
        "kind": kind,
        "record_id": identity,
        "revision": "1",
        "digest_sha256": "a" * 64,
    }


def tree_payload():
    protocol = reference("protocol", "protocol-1")
    experiment = reference("experiment", "experiment-1")
    root = {
        "trial": reference("trial", "trial-root"),
        "parent_trial": None,
        "protocol": protocol,
        "experiment": experiment,
        "arm": None,
        "state": "succeeded",
        "scores": [],
        "reviews": [],
        "notes": [],
        "input_gaps": [],
        "deviations": [],
        "charge": None,
    }
    child = deepcopy(root)
    child.update(
        trial=reference("trial", "trial-child"), parent_trial=root["trial"], state="failed"
    )
    return {
        "state": "available",
        "run_id": RUN,
        "protocol": protocol,
        "experiment": experiment,
        "research_log": reference("research_log", "log-1"),
        "report": reference("report", "report-1"),
        "attempts": [root, child],
        "notes": [],
    }


def project(payload):
    return project_attempts(RunAttemptTreeAvailable.model_validate(payload), run_id=RUN)


def test_public_trial_identities_and_parent_links_are_order_independent():
    payload = tree_payload()
    expected = (
        Attempt("trial-child", "trial-root", "failed"),
        Attempt("trial-root", None, "succeeded"),
    )
    assert project(payload) == expected
    payload["attempts"].reverse()
    assert project(payload) == expected


def test_unavailable_differs_from_authoritative_empty_tree():
    unavailable = RunAttemptTreeUnavailable.model_validate(
        {
            "state": "unavailable",
            "run_id": RUN,
            "code": "run_records_projection_unavailable",
            "stage": "run_result",
            "cause": "No reader",
        }
    )
    assert project_attempts(unavailable, run_id=RUN) is None
    payload = tree_payload()
    payload["attempts"] = []
    assert project(payload) == ()


def test_run_scope_checked_even_for_unavailable_projection():
    unavailable = RunAttemptTreeUnavailable.model_validate(
        {
            "state": "unavailable",
            "run_id": RUN,
            "code": "run_records_projection_unavailable",
            "stage": "run_result",
            "cause": "No reader",
        }
    )
    with pytest.raises(TreeError, match="attempt_tree_run_mismatch"):
        project_attempts(unavailable, run_id="run_BBBBBBBBBBBBBBBBBBBBBB")


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("duplicate", "duplicate_trial"),
        ("orphan", "parent_reference_mismatch"),
        ("revision", "parent_reference_mismatch"),
        ("digest", "parent_reference_mismatch"),
        ("cycle", "cycle"),
        ("self", "cycle"),
        ("protocol", "organizing_reference_mismatch"),
        ("experiment", "organizing_reference_mismatch"),
    ],
)
def test_invalid_ancestry_is_not_parity_evidence(mutation, code):
    payload = tree_payload()
    root, child = payload["attempts"]
    if mutation == "duplicate":
        payload["attempts"].append(deepcopy(root))
    elif mutation == "orphan":
        child["parent_trial"] = reference("trial", "missing")
    elif mutation in {"revision", "digest"}:
        child["parent_trial"] = deepcopy(root["trial"])
        child["parent_trial"]["revision" if mutation == "revision" else "digest_sha256"] = (
            "2" if mutation == "revision" else "b" * 64
        )
    elif mutation == "cycle":
        root["parent_trial"] = child["trial"]
    elif mutation == "self":
        root["parent_trial"] = root["trial"]
    else:
        child[mutation] = reference(mutation, "wrong")
    with pytest.raises(TreeError, match="attempt_tree_" + code):
        project(payload)


def test_deep_tree_does_not_depend_on_python_recursion_limit():
    payload = tree_payload()
    template = payload["attempts"][0]
    payload["attempts"] = []
    for index in range(2000):
        node = deepcopy(template)
        node["trial"] = reference("trial", f"trial-{index}")
        node["parent_trial"] = None if index == 0 else reference("trial", f"trial-{index - 1}")
        payload["attempts"].append(node)
    payload["attempts"].reverse()
    assert len(project(payload)) == 2000


def test_score_cannot_be_attributed_to_another_trial():
    payload = tree_payload()
    root, child = payload["attempts"]
    root["scores"] = [
        {
            "record": reference("result", "result-1"),
            "trial": child["trial"],
            "outcome": "inconclusive",
            "measurements": [],
            "evaluator": None,
            "verification": "unverified",
            "scorer_receipt": None,
            "evidence": [],
            "usage": [],
            "missing_evidence": [],
            "limitations": [],
        }
    ]
    with pytest.raises(TreeError, match="attempt_tree_score_reference_mismatch"):
        project(payload)


@pytest.mark.parametrize("at_root", [True, False])
def test_note_must_reference_this_research_log(at_root):
    payload = tree_payload()
    note = {
        "record": reference("log_entry", "entry-1"),
        "log": reference("research_log", "wrong"),
        "category": "observation",
        "text": "A bounded note",
        "occurred_at": "2026-11-17T00:00:00Z",
        "correction_of": None,
        "sources": [],
    }
    owner = payload if at_root else payload["attempts"][0]
    owner["notes"] = [note]
    with pytest.raises(TreeError, match="attempt_tree_note_reference_mismatch"):
        project(payload)
