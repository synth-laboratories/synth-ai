"""Launch on a repository at an exact base commit (SYN-4885), offline."""

import json

import pytest
from pydantic import ValidationError
from synth_ai.errors import ContractError
from synth_ai.sdk.swarms import Limits, RepoSelector
from synth_ai.sdk.swarms.client import launch_request, validate_preflight
from synth_ai.sdk.swarms.types.runs import LaunchPreflight
from synth_ai.types.money import Money
from synth_ai.types.refs import Ref
from tests.http_fixtures import METADATA

LIMITS = Limits(
    maximum_charge=Money(coefficient=1, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)
SHA = "a" * 40
BASE = Ref.commit("repo_1", SHA, "b" * 64)
SELECTOR = RepoSelector(repo_id="repo_1", base=BASE)


def test_absent_repo_is_omitted_from_the_wire():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert "repo" not in json.loads(request.model_dump_json(exclude_none=False))


def test_repo_selector_carries_the_exact_commit_ref():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m", repo=SELECTOR)
    wire = json.loads(request.model_dump_json(exclude_none=False))
    assert wire["repo"] == {
        "repo_id": "repo_1",
        "base": {
            "authority": "git",
            "kind": "commit",
            "record_id": "repo_1",
            "revision": SHA,
            "digest_sha256": "b" * 64,
        },
    }


@pytest.mark.parametrize(
    "base",
    [
        Ref.artifact("art_1", "1", "c" * 64),
        Ref.commit("other_repo", SHA, "b" * 64),
        Ref(
            authority="git",
            kind="commit",
            record_id="repo_1",
            revision="main",
            digest_sha256="b" * 64,
        ),
    ],
    ids=["artifact", "other-repo", "branch-name"],
)
def test_repo_selector_refuses_anything_but_an_exact_commit_of_the_repo(base):
    with pytest.raises(ValidationError):
        RepoSelector(repo_id="repo_1", base=base)


def test_repo_selector_refuses_unknown_fields():
    with pytest.raises(ValidationError):
        RepoSelector.model_validate(
            {"repo_id": "repo_1", "base": BASE.model_dump(mode="json"), "branch": "main"}
        )


def _preflight(repo):
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
    if repo is not None:
        configuration["repo"] = repo
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


def test_preflight_must_echo_the_selected_base_commit():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m", repo=SELECTOR)
    validate_preflight(_preflight(SELECTOR.model_dump(mode="json")), request)
    with pytest.raises(ContractError):
        validate_preflight(_preflight(None), request)
    moved = RepoSelector(repo_id="repo_1", base=Ref.commit("repo_1", "f" * 40, "b" * 64))
    with pytest.raises(ContractError):
        validate_preflight(_preflight(moved.model_dump(mode="json")), request)


def test_preflight_without_repo_tolerates_absent_repo_echo():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert validate_preflight(_preflight(None), request).configuration.repo is None
