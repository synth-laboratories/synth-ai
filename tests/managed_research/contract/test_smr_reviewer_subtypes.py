from synth_ai.managed_research.mcp.server import ManagedResearchMcpServer
from synth_ai.managed_research.models.smr_actor_models import (
    SMR_ACTOR_SUBTYPE_VALUES,
    SMR_REVIEWER_SUBTYPE_VALUES,
    SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES,
    SmrActorModelAssignment,
    SmrActorType,
    SmrReviewerSubtype,
    coerce_smr_actor_model_assignment,
    coerce_smr_actor_subtype,
)


def test_reviewer_subtypes_are_public_actor_subtypes() -> None:
    expected = {
        "main",
        "task_completion",
        "run_completion",
        "safety",
        "objective",
    }

    assert expected.issubset(set(SMR_ACTOR_SUBTYPE_VALUES))
    assert expected == set(SMR_REVIEWER_SUBTYPE_VALUES)


def test_reviewer_subtypes_validate_for_actor_model_overrides() -> None:
    for subtype in (
        SmrReviewerSubtype.TASK_COMPLETION,
        SmrReviewerSubtype.RUN_COMPLETION,
        SmrReviewerSubtype.SAFETY,
        SmrReviewerSubtype.OBJECTIVE,
    ):
        assignment = coerce_smr_actor_model_assignment(
            {
                "actor_type": SmrActorType.REVIEWER.value,
                "actor_subtype": subtype.value,
                "agent_model": SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES[0],
            }
        )

        assert isinstance(assignment, SmrActorModelAssignment)
        assert assignment.actor_type is SmrActorType.REVIEWER
        assert assignment.actor_subtype.value == subtype.value


def test_reviewer_subtypes_are_not_valid_worker_subtypes() -> None:
    try:
        coerce_smr_actor_subtype(
            SmrReviewerSubtype.RUN_COMPLETION.value,
            actor_type=SmrActorType.WORKER,
        )
    except ValueError as exc:
        assert "not valid for actor_type 'worker'" in str(exc)
    else:
        raise AssertionError("reviewer subtype unexpectedly accepted for worker")


def test_mcp_actor_model_override_schema_lists_reviewer_subtypes() -> None:
    server = ManagedResearchMcpServer()
    trigger_schema = server.get_tool_definition("smr_trigger_run").input_schema
    actor_model_overrides = trigger_schema["properties"]["actor_model_overrides"]
    actor_subtype_schema = actor_model_overrides["items"]["properties"]["actor_subtype"]

    assert {
        "task_completion",
        "run_completion",
        "safety",
        "objective",
    }.issubset(set(actor_subtype_schema["enum"]))
