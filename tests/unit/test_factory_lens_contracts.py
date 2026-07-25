"""Unit tier: SDK contracts for the champion-free Result authority.

No network. These pin the two things a typed client exists to guarantee:

* a request that the backend would reject is rejected locally, with the same
  failure name, so the customer sees it in their editor rather than in a 400;
* a response decodes into a shape whose *totality* the caller can rely on —
  ``outcome`` always answers, and ``result_id`` is never populated for an
  outcome that has no winner.
"""

from __future__ import annotations

import pytest

from synth_ai.core.research.contracts.factory_lenses import (
    FactoryBestOutcome,
    FactoryBestResult,
    FactoryBestResults,
    FactoryEvaluationLens,
    FactoryEvaluationStatus,
    FactoryLensDirection,
    FactoryLensMissingPolicy,
    FactoryLensSpec,
    FactoryLensTieBreak,
    FactoryPreferenceAction,
    FactoryPreferenceRequest,
    FactoryResultEvaluationRequest,
    FactoryResultKind,
)
from synth_ai.core.research.factories import (
    AsyncFactoriesAPI,
    AsyncFactoryLensesAPI,
    FactoriesAPI,
    FactoryLensesAPI,
)
from synth_ai.core.research.operations import RESEARCH_OPERATIONS


# ---------------------------------------------------------------------------
# Lens spec
# ---------------------------------------------------------------------------


def test_minimal_lens_spec_defaults_are_the_conservative_ones() -> None:
    """Unscored Results are ineligible by default, not ranked as worst.

    Ranking an unevaluated Result would let it become "best" the moment every
    scored Result was superseded.
    """

    spec = FactoryLensSpec(lens_key="score", direction=FactoryLensDirection.MAXIMIZE)
    assert spec.missing_policy is FactoryLensMissingPolicy.INELIGIBLE
    assert spec.tie_break is FactoryLensTieBreak.EARLIEST_RESULT
    assert spec.eligible_result_kinds == ()


def test_empty_lens_key_is_rejected_locally() -> None:
    with pytest.raises(ValueError):
        FactoryLensSpec(lens_key="   ", direction=FactoryLensDirection.MAXIMIZE)


def test_lens_spec_wire_body_is_fully_explicit() -> None:
    """Every policy the backend enforces travels on the wire.

    A field the client omits and the backend defaults is a field the customer
    cannot see on read.
    """

    spec = FactoryLensSpec(
        lens_key="craftax_score",
        direction=FactoryLensDirection.MINIMIZE,
        objective="cost per achievement",
        missing_policy=FactoryLensMissingPolicy.WORST,
        tie_break=FactoryLensTieBreak.LOWEST_RESULT_ID,
        eligible_result_kinds=(FactoryResultKind.CODE_CHANGE, FactoryResultKind.PROMPT),
    )
    wire = spec.to_wire()
    assert wire["lens_key"] == "craftax_score"
    assert wire["direction"] == "minimize"
    assert wire["missing_policy"] == "worst"
    assert wire["tie_break"] == "lowest_result_id"
    assert wire["eligible_result_kinds"] == ["code_change", "prompt"]
    assert wire["objective"] == "cost per achievement"


def test_lens_decodes_from_backend_shape() -> None:
    lens = FactoryEvaluationLens.from_wire(
        {
            "lens_id": "lens-1",
            "factory_id": "fac-1",
            "lens_key": "craftax_score",
            "lens_version": 4,
            "direction": "maximize",
            "missing_policy": "ineligible",
            "tie_break": "earliest_result",
            "status": "active",
            "created_at": "2026-07-24T12:00:00Z",
            "objective": "achievements per dollar",
            "eligible_result_kinds": ["code_change"],
            "definition": {"seeds": 64},
        }
    )
    assert lens.lens_version == 4
    assert lens.direction is FactoryLensDirection.MAXIMIZE
    assert lens.eligible_result_kinds == ("code_change",)
    assert lens.definition == {"seeds": 64}


# ---------------------------------------------------------------------------
# Best-so-far totality
# ---------------------------------------------------------------------------


def _best_wire(**overrides) -> dict:
    payload = {
        "lens_key": "craftax_score",
        "lens_version": 3,
        "direction": "maximize",
        "outcome": "best",
        "reason": "best of 6 ranked Result(s)",
        "considered": 10,
        "eligible": 8,
        "scored": 6,
        "result_id": "res-1",
        "score": 0.81,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "outcome", ["best", "no_eligible_results", "no_scored_results"]
)
def test_every_backend_outcome_decodes(outcome) -> None:
    """An unknown outcome must raise, not silently become a falsy default."""

    best = FactoryBestResult.from_wire(_best_wire(outcome=outcome))
    assert best.outcome is FactoryBestOutcome(outcome)


def test_unknown_outcome_fails_loudly() -> None:
    with pytest.raises(ValueError):
        FactoryBestResult.from_wire(_best_wire(outcome="probably_fine"))


def test_has_best_is_false_without_a_winner() -> None:
    """The guard a caller should branch on before touching ``result_id``."""

    empty = FactoryBestResult.from_wire(
        _best_wire(outcome="no_scored_results", result_id=None, score=None)
    )
    assert empty.has_best is False
    assert empty.result_id is None
    # Counts still travel: 8 eligible with 0 scored is actionable information.
    assert empty.eligible == 8
    assert empty.scored == 6


def test_unlensed_factory_decodes_as_optimizes_false() -> None:
    """The ordinary non-hillclimbing Factory. Not an error, not empty results."""

    results = FactoryBestResults.from_wire(
        {"factory_id": "fac-1", "optimizes": False, "lenses": []}
    )
    assert results.optimizes is False
    assert results.lenses == ()


def test_multi_objective_factory_decodes_every_lens() -> None:
    results = FactoryBestResults.from_wire(
        {
            "factory_id": "fac-1",
            "optimizes": True,
            "lenses": [
                _best_wire(lens_key="speed"),
                _best_wire(lens_key="cost", direction="minimize"),
            ],
        }
    )
    assert [lens.lens_key for lens in results.lenses] == ["speed", "cost"]
    assert results.lenses[1].direction is FactoryLensDirection.MINIMIZE


# ---------------------------------------------------------------------------
# Evaluation request
# ---------------------------------------------------------------------------


def test_evaluated_verdict_without_a_score_is_rejected_locally() -> None:
    """Same failure name the backend raises, surfaced before the request."""

    with pytest.raises(ValueError, match="evaluation_score_required"):
        FactoryResultEvaluationRequest(
            lens_key="score",
            attempt_key="attempt-1",
            status=FactoryEvaluationStatus.EVALUATED,
        )


@pytest.mark.parametrize(
    "status",
    [
        FactoryEvaluationStatus.PENDING,
        FactoryEvaluationStatus.FAILED,
        FactoryEvaluationStatus.TIMEOUT,
    ],
)
def test_non_evaluated_verdicts_need_no_score(status) -> None:
    """A grader that timed out has nothing to report but the timeout."""

    request = FactoryResultEvaluationRequest(
        lens_key="score", attempt_key="attempt-1", status=status
    )
    assert "score" not in request.to_wire()


def test_evaluation_request_carries_the_attempt_key() -> None:
    """Idempotency is the caller's to control, so it must be explicit."""

    wire = FactoryResultEvaluationRequest(
        lens_key="score",
        attempt_key="factorybench-run-77",
        status=FactoryEvaluationStatus.EVALUATED,
        score=0.81,
        baseline_score=0.62,
        evaluator="factorybench",
        record={"seeds": 64},
    ).to_wire()
    assert wire["attempt_key"] == "factorybench-run-77"
    assert wire["score"] == pytest.approx(0.81)
    assert wire["baseline_score"] == pytest.approx(0.62)
    assert wire["record"] == {"seeds": 64}


def test_blank_attempt_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        FactoryResultEvaluationRequest(
            lens_key="score", attempt_key="", status=FactoryEvaluationStatus.PENDING
        )


# ---------------------------------------------------------------------------
# Preference
# ---------------------------------------------------------------------------


def test_prefer_without_a_result_is_rejected_locally() -> None:
    with pytest.raises(ValueError, match="preference_result_required"):
        FactoryPreferenceRequest(idempotency_key="k1", reason="looks right")


def test_retract_needs_no_result() -> None:
    """Retracting is withdrawing a stance, which may name nothing."""

    wire = FactoryPreferenceRequest(
        idempotency_key="k1",
        reason="superseded by review",
        action=FactoryPreferenceAction.RETRACT,
    ).to_wire()
    assert wire["action"] == "retract"
    assert "result_id" not in wire


def test_preference_requires_a_reason() -> None:
    """An unexplained human override is unauditable a month later."""

    with pytest.raises(ValueError):
        FactoryPreferenceRequest(idempotency_key="k1", reason="  ", result_id="res-1")


# ---------------------------------------------------------------------------
# API wiring and sync/async parity
# ---------------------------------------------------------------------------


def test_lenses_namespace_is_reachable_from_both_clients() -> None:
    """SS12: shipped-but-unwired code is the same as absent code."""

    assert "lenses" in FactoriesAPI.__init__.__code__.co_names
    assert "lenses" in AsyncFactoriesAPI.__init__.__code__.co_names


def test_sync_and_async_lens_surfaces_are_identical() -> None:
    """Async parity is a launch contract, not a nice-to-have."""

    def _public(cls) -> set[str]:
        return {name for name in dir(cls) if not name.startswith("_")}

    assert _public(FactoryLensesAPI) == _public(AsyncFactoryLensesAPI)


@pytest.mark.parametrize(
    "operation_id,method,path",
    [
        ("define_factory_evaluation_lens", "POST", "/smr/factories/{factory_id}/lenses"),
        ("list_factory_evaluation_lenses", "GET", "/smr/factories/{factory_id}/lenses"),
        (
            "retrieve_factory_best_results",
            "GET",
            "/smr/factories/{factory_id}/results/best-so-far",
        ),
        (
            "record_factory_result_preference",
            "POST",
            "/smr/factories/{factory_id}/results/prefer",
        ),
        (
            "record_factory_result_evaluation",
            "POST",
            "/smr/factories/{factory_id}/results/{result_id}/evaluations",
        ),
    ],
)
def test_operations_match_the_backend_routes(operation_id, method, path) -> None:
    operation = RESEARCH_OPERATIONS[operation_id]
    assert operation.method.value == method
    assert operation.path_template == path


def test_evaluation_and_preference_are_declared_idempotent() -> None:
    """Both are keyed writes; a transport retry must be safe."""

    assert RESEARCH_OPERATIONS["record_factory_result_evaluation"].idempotent is True
    assert RESEARCH_OPERATIONS["record_factory_result_preference"].idempotent is True


def test_defining_a_lens_is_not_idempotent() -> None:
    """Re-posting appends a new version by design; retrying is not a no-op."""

    assert RESEARCH_OPERATIONS["define_factory_evaluation_lens"].idempotent is False
