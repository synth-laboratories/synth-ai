"""Typed RunLaunchRequest carries OEQ/DEO/required_work_products/require_report (SYN-2892)."""

from __future__ import annotations

import pytest
from synth_ai.managed_research.models.run_launch import RunLaunchRequest


def test_required_work_products_and_require_report_flow_through_client_kwargs() -> None:
    req = RunLaunchRequest(
        runbook_preset="lite",
        required_work_products=[{"kind": "report", "title": "Final Report"}],
        require_report=False,
    )
    kwargs = req.to_client_kwargs()
    assert kwargs["required_work_products"] == ({"kind": "report", "title": "Final Report"},)
    assert kwargs["require_report"] is False


def test_open_ended_question_flows_through_client_kwargs() -> None:
    req = RunLaunchRequest(runbook_preset="lite", open_ended_question={"question_text": "What?"})
    assert req.to_client_kwargs()["open_ended_question"] == {"question_text": "What?"}


def test_directed_effort_outcome_flows_through_client_kwargs() -> None:
    req = RunLaunchRequest(
        runbook_preset="lite",
        directed_effort_outcome={"outcome_text": "Ship it"},
    )
    assert req.to_client_kwargs()["directed_effort_outcome"] == {"outcome_text": "Ship it"}


def test_oeq_and_deo_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either open_ended_question or directed_effort_outcome"):
        RunLaunchRequest(
            runbook_preset="lite",
            open_ended_question={"question_text": "What?"},
            directed_effort_outcome={"outcome_text": "Ship it"},
        )


def test_round_trip_preserves_objective_obligation_fields() -> None:
    req = RunLaunchRequest(
        runbook_preset="lite",
        directed_effort_outcome={"outcome_text": "Ship it"},
        required_work_products=[{"kind": "report"}],
        require_report=True,
    )
    restored = RunLaunchRequest.from_client_kwargs(req.to_client_kwargs())
    assert restored.directed_effort_outcome == {"outcome_text": "Ship it"}
    assert restored.required_work_products == ({"kind": "report"},)
    assert restored.require_report is True


def test_open_ended_question_reaches_primary_parent_in_wire_payload() -> None:
    req = RunLaunchRequest(runbook_preset="lite", open_ended_question={"question_text": "What?"})
    payload = req.to_wire()
    assert payload["primary_parent"] == {
        "kind": "open_ended_question",
        "open_ended_question": {"question_text": "What?"},
    }
