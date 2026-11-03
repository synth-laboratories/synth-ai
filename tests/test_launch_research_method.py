"""Fork-and-Flush method selection on launch (SYN-4852), offline."""

import json

import pytest
from pydantic import ValidationError
from synth_ai.errors import ContractError
from synth_ai.sdk.swarms import (
    ForkFlushMethod,
    ForkFlushOverrides,
    ForkFlushRound,
    Limits,
    ResolvedResearchMethod,
)
from synth_ai.sdk.swarms.client import launch_request, validate_preflight
from synth_ai.sdk.swarms.types.runs import LaunchPreflight
from synth_ai.types.money import Money
from tests.http_fixtures import METADATA

LIMITS = Limits(
    maximum_charge=Money(coefficient=1, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)
ROUNDS = (ForkFlushRound(after_budget_bps=2000, exploration_budget_bps=1000),)


def _resolved(branches: int) -> dict:
    return {
        "metadata": METADATA,
        "release": "research.fork_flush@1",
        "program_id": "research.fork_flush",
        "program_version": 1,
        "definition_digest": "a" * 64,
        "configuration_digest": "b" * 64,
        "forge_profile_digest": "c" * 64,
        "context_policy_digest": "d" * 64,
        "parameters": {
            "branches": branches,
            "rounds": [r.model_dump(mode="json") for r in ROUNDS],
        },
    }


def _preflight(method: dict | None) -> LaunchPreflight:
    configuration = {
        "metadata": METADATA,
        "run_id": None,
        "project_id": "p",
        "configuration_id": "cfg",
        "digest_sha256": "d" * 64,
        "profile_id": "prof",
        "runtime_image_digest": "e" * 64,
        "limits": LIMITS.model_dump(mode="json"),
        "inputs": {"files": []},
        "providers": {"default": "openai"},
        "models": {"default": "gpt-6-luna"},
    }
    if method is not None:
        configuration["method"] = method
    return LaunchPreflight.model_validate_json(
        json.dumps(
            {
                "metadata": METADATA,
                "ready": True,
                "blockers": [],
                "configuration": configuration,
                "expires_at": "2026-10-20T00:00:00Z",
            }
        )
    )


def test_absent_method_is_omitted_from_the_wire():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert "method" not in json.loads(request.model_dump_json(exclude_none=False))


def test_selected_method_and_overrides_reach_the_wire():
    method = ForkFlushMethod(overrides=ForkFlushOverrides(branches=8, rounds=ROUNDS))
    request = launch_request(None, "p", LIMITS, None, None, None, message="m", method=method)
    wire = json.loads(request.model_dump_json(exclude_none=False))
    assert wire["method"]["release"] == "research.fork_flush@1"
    assert wire["method"]["overrides"]["branches"] == 8
    assert wire["method"]["overrides"]["rounds"] == [
        {"after_budget_bps": 2000, "exploration_budget_bps": 1000}
    ]


@pytest.mark.parametrize("branches", [1, 129])
def test_branch_count_outside_the_release_range_is_refused(branches):
    with pytest.raises(ValidationError):
        ForkFlushOverrides(branches=branches)


def test_rounds_must_leave_positive_exploitation_intervals():
    with pytest.raises(ValidationError):
        ForkFlushOverrides(
            rounds=(
                ForkFlushRound(after_budget_bps=2000, exploration_budget_bps=1000),
                ForkFlushRound(after_budget_bps=3000, exploration_budget_bps=1000),
            )
        )
    with pytest.raises(ValidationError):
        ForkFlushOverrides(
            rounds=(ForkFlushRound(after_budget_bps=9000, exploration_budget_bps=1000),)
        )


def test_unknown_release_is_refused():
    with pytest.raises(ValidationError):
        ForkFlushMethod.model_validate({"release": "research.fork_flush@2"})


def test_resolved_method_satisfies_only_the_exact_overrides():
    resolved = ResolvedResearchMethod.model_validate_json(json.dumps(_resolved(8)))
    assert resolved.satisfies(ForkFlushMethod())
    assert resolved.satisfies(ForkFlushMethod(overrides=ForkFlushOverrides(branches=8)))
    assert resolved.satisfies(ForkFlushMethod(overrides=ForkFlushOverrides(rounds=ROUNDS)))
    assert not resolved.satisfies(ForkFlushMethod(overrides=ForkFlushOverrides(branches=4)))


def test_preflight_must_echo_the_selected_method_without_narrowing():
    method = ForkFlushMethod(overrides=ForkFlushOverrides(branches=8))
    request = launch_request(None, "p", LIMITS, None, None, None, message="m", method=method)
    validate_preflight(_preflight(_resolved(8)), request)
    with pytest.raises(ContractError):
        validate_preflight(_preflight(None), request)
    with pytest.raises(ContractError):
        validate_preflight(_preflight(_resolved(4)), request)


def test_preflight_without_method_tolerates_a_server_default_method():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert validate_preflight(_preflight(_resolved(8)), request).configuration.method is not None
