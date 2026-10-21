"""Run download/bundle, labels, objectives, resources and launch serialization offline."""

import json

import pytest
from pydantic import ValidationError
from synth_ai.errors import ContractError, ResultNotReadyError
from synth_ai.sdk.swarms import (
    Limits,
    ResourceGrant,
    ResourceGrantSet,
    ResourceRequest,
    ResourceUsage,
    RunSnapshot,
)
from synth_ai.sdk.swarms.bundle import TERMINAL
from synth_ai.sdk.swarms.client import launch_request
from synth_ai.types.money import Money
from synth_ai.types.refs import Ref
from tests.artifact_fixtures import (
    REF,
    Fake,
    artifact_body,
    drive,
    objective,
    snapshot_body,
)

pytestmark = pytest.mark.unit
MODES = pytest.mark.parametrize("async_mode", [False, True])
ART = Ref.model_validate(artifact_body()["ref"])
TIP = "refs/heads/worker/run-run-1"


def ready(server, *, git=False, **snapshot):
    server.snapshot = snapshot_body(
        objectives=[objective(), objective("Oeq-2", "Second.\n")], **snapshot
    )
    server.links = [ART, *([REF] if git else [])]
    server.refs_pages = [[{"name": TIP, "commit_sha": REF.revision, "ref": REF.model_dump()}]]
    return server


@MODES
def test_download_writes_artifacts_objectives_and_manifest_atomically(tmp_path, async_mode):
    server = ready(Fake())
    destination = tmp_path / "bundle"

    async def steps(client, w):
        run = await w(client.swarms.get("run-1"))
        return await w(run.download(destination))

    receipt = drive(async_mode, server, steps)
    assert (destination / "artifacts/art-1/1/report.md").read_bytes() == b"hello artifact\n"
    assert (destination / "answers/oeq-1.md").read_text() == "The answer.\n"
    manifest = json.loads((destination / "MANIFEST.json").read_text())
    assert manifest["schema_version"] == "synth.run-bundle.v1"
    assert manifest["artifacts"] == [ART.model_dump(mode="json")]
    assert "final_commit" not in manifest and receipt.final_commit is None
    objectives = json.loads((destination / "objectives.json").read_text())
    assert [o["id"] for o in objectives["objectives"]] == ["oeq-1", "Oeq-2"]
    assert receipt.artifacts == (ART,) and receipt.destination == destination
    assert not list(tmp_path.glob(".synth-run-*"))


@MODES
def test_download_with_code_selects_the_exact_branch_across_pages(tmp_path, async_mode):
    server = ready(Fake(), git=True)
    other = {
        "name": TIP + "x",
        "commit_sha": "c" * 40,
        "ref": Ref.commit("repo-test", "c" * 40, "e" * 64).model_dump(),
    }
    exact = {"name": TIP, "commit_sha": REF.revision, "ref": REF.model_dump()}
    # A colliding longer run id sorts first; the exact branch arrives on page 2.
    server.refs_pages = [[other], [exact]]
    destination = tmp_path / "bundle"

    async def steps(client, w):
        run = await w(client.swarms.get("run-1"))
        return await w(run.download(destination))

    receipt = drive(async_mode, server, steps)
    assert receipt.final_commit == REF
    assert (destination / "code/nested/file").read_bytes() == b"verified\n"
    refs_calls = [r for r in server.requests if r.url.path.endswith("/refs")]
    assert refs_calls and all(r.url.params["prefix"] == TIP for r in refs_calls)


@MODES
def test_download_refuses_when_only_a_colliding_branch_exists(tmp_path, async_mode):
    server = ready(Fake(), git=True)
    server.refs_pages = [[{"name": TIP + "x", "commit_sha": REF.revision, "ref": REF.model_dump()}]]

    async def steps(client, w):
        run = await w(client.swarms.get("run-1"))
        return await w(run.download(tmp_path / "bundle"))

    with pytest.raises(ResultNotReadyError) as caught:
        drive(async_mode, server, steps)
    assert caught.value.code == "run_final_commit_unavailable"
    assert not (tmp_path / "bundle").exists()


@MODES
@pytest.mark.parametrize(
    "changes",
    [
        {"state": "running"},
        {"state": "succeeded"},  # a task outcome, not a RunState
        {"execution_closed": False},
        {"accounting_final": False},
        {"cleanup_complete": False},
    ],
)
def test_download_waits_on_real_finality_not_a_terminal_looking_state(
    tmp_path, changes, async_mode
):
    server = ready(Fake(), **changes)

    async def steps(client, w):
        run = await w(client.swarms.get("run-1"))
        return await w(run.download(tmp_path / "bundle"))

    with pytest.raises(ResultNotReadyError) as caught:
        drive(async_mode, server, steps)
    assert caught.value.code == "run_not_terminal" and caught.value.retryable
    assert not (tmp_path / "bundle").exists()
    assert "succeeded" not in TERMINAL


@MODES
def test_download_refuses_case_colliding_objective_ids(tmp_path, async_mode):
    server = ready(Fake())
    server.snapshot["objectives"] = [objective("Same"), objective("same")]

    async def steps(client, w):
        run = await w(client.swarms.get("run-1"))
        return await w(run.download(tmp_path / "bundle"))

    with pytest.raises(ContractError) as caught:
        drive(async_mode, server, steps)
    assert caught.value.code == "objective_id_collision"
    assert not (tmp_path / "bundle").exists() and not list(tmp_path.glob(".synth-run-*"))


@MODES
def test_run_labels_decode_tolerantly_and_filter_and_update(async_mode):
    server = Fake()
    server.snapshot = snapshot_body(labels={"score": "0.9", "Legacy Key": "x"})
    server.run_list_items = [server.snapshot]

    async def steps(client, w):
        page = await w(client.swarms.list(limit=10, label={"score": "0.9"}))
        run = await w(client.swarms.get("run-1"))
        current = await w(run.labels.get())
        updated = await w(
            run.labels.update(set={"tier": "gold"}, remove=("score",), expected_revision=1)
        )
        scalar = await w(run.labels.set(attempt=3, flag=True, ratio=0.5))
        with pytest.raises(ValueError):
            await w(run.labels.set(bad=float("nan")))
        return page, current, updated, scalar

    page, current, updated, scalar = drive(async_mode, server, steps)
    listing = [r for r in server.requests if r.url.path.endswith("/swarms/runs")][0]
    assert listing.url.params.get_list("label") == ["score=0.9"]
    assert page.items[0].labels["Legacy Key"] == "x" and current["score"] == "0.9"
    assert server.label_calls[0] == {
        "set": {"tier": "gold"},
        "remove": ["score"],
        "expected_revision": 1,
    }
    assert updated.labels == {"Legacy Key": "x", "tier": "gold"}
    assert server.label_calls[1]["set"] == {"attempt": "3", "flag": "true", "ratio": "0.5"}
    assert server.label_calls[1]["expected_revision"] == 2 and scalar.labels_revision == 3


@MODES
def test_run_objectives_decode_answers_and_ignore_future_fields(async_mode):
    server = ready(Fake())

    async def steps(client, w):
        run = await w(client.swarms.get("run-1"))
        return await w(run.objectives())

    results = drive(async_mode, server, steps)
    assert [r.id for r in results] == ["oeq-1", "Oeq-2"]
    assert results[0].answer.text == "The answer.\n" and results[0].state == "verified"


def test_resource_request_stays_strict_and_omits_absent_fields():
    request = ResourceRequest(
        plugin="openrouter.inference",
        roles=("worker",),
        limits={"spend_micros": 5000},
        models=("openai/gpt-6-luna",),
    )
    assert request.model_dump(mode="json") == {
        "plugin": "openrouter.inference",
        "version": 1,
        "roles": ["worker"],
        "limits": {"spend_micros": 5000},
        "models": ["openai/gpt-6-luna"],
    }
    with pytest.raises(ValidationError):
        ResourceRequest(plugin="index.search", roles=("worker",), limits={"calls": 1}, future=1)
    with pytest.raises(ValidationError):
        ResourceRequest(plugin="openrouter.inference", roles=("worker",), limits={"calls": 1})
    with pytest.raises(ValidationError):
        ResourceRequest(
            plugin="index.search",
            roles=("worker",),
            limits={"calls": 1},
            role_limits={"worker": {"calls": 2}},
            collections=("c",),
        )


def test_resource_grants_and_usage_decode_tolerantly():
    grant = {
        "plugin": "future.plugin",
        "version": 2,
        "roles": ["worker", "future-role"],
        "limits": {"spend_micros": 1, "new_dimension": 2},
        "funding_lane": "platform",
        "registry_digest": "a" * 64,
        "grant_id": "grant-" + "x" * 200,
        "new_field": {"ignored": True},
    }
    decoded = ResourceGrantSet.model_validate_json(
        json.dumps({"schema_version": "orchestra.resource-grants.v2", "grants": [grant]})
    )
    assert (
        decoded.grants[0].plugin == "future.plugin"
        and decoded.grants[0].limits["new_dimension"] == 2
    )
    assert ResourceGrant.model_validate_json(json.dumps(grant)).funding_lane == "platform"
    usage = ResourceUsage.model_validate_json(
        json.dumps(
            {
                "plugin": "p",
                "dimension": "tokens",
                "usage_class": "agent",
                "role": "worker",
                "state": "measured",
                "amount": 7,
                "new": 1,
            }
        )
    )
    assert usage.amount == 7
    with pytest.raises(ValidationError):
        ResourceUsage.model_validate_json(
            json.dumps(
                {
                    "plugin": "p",
                    "dimension": "tokens",
                    "usage_class": "agent",
                    "role": "w",
                    "state": "unknown",
                    "amount": 0,
                }
            )
        )


LIMITS = Limits(
    maximum_charge=Money(coefficient=1, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)


def test_launch_request_omits_absent_fields_and_sends_message_for_the_alias():
    with pytest.warns(DeprecationWarning):
        legacy = launch_request("do it", "p", LIMITS, None, None, None)
    modern = launch_request(None, "p", LIMITS, None, None, None, message="do it")
    wire = json.loads(legacy.model_dump_json(exclude_none=False))
    assert wire == json.loads(modern.model_dump_json(exclude_none=False))
    assert set(wire) == {"message", "project_id", "limits"}
    assert legacy.objective == "do it" and "objective" not in wire
    with pytest.raises(ValueError):
        launch_request("a", "p", LIMITS, None, None, None, message="b")
    with pytest.raises(ValueError):
        launch_request(None, "p", LIMITS, None, None, None)


def test_launch_request_wire_carries_only_selected_optional_groups():
    resource = ResourceRequest(
        plugin="index.search", roles=("worker",), limits={"calls": 3}, collections=("c",)
    )
    request = launch_request(
        None, "p", LIMITS, [ART], None, None, labels={"k": "v"}, message="m", resources=(resource,)
    )
    wire = json.loads(request.model_dump_json(exclude_none=False))
    assert set(wire) == {"message", "project_id", "limits", "inputs", "labels", "resources"}
    assert wire["inputs"] == {"files": [], "artifacts": [ART.model_dump(mode="json")]}


def test_snapshot_labels_accept_what_requests_refuse():
    body = snapshot_body(labels={"UPPER case": "v" * 1000})
    assert RunSnapshot.model_validate_json(json.dumps(body)).labels["UPPER case"] == "v" * 1000
    with pytest.raises(ValidationError):
        launch_request(None, "p", LIMITS, None, None, None, labels={"UPPER case": "v"}, message="m")
