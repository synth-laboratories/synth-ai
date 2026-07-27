from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from synth_ai.core.research.contracts.research_intern import (
    DataBindingCreateRequest,
    DatasetRevisionCreateRequest,
    MAGI_CANONICAL_USER_BY_MODE,
    MagiCanonicalUser,
    MagiDecisionKind,
    MagiDecisionRequest,
    MagiMode,
    ProjectComputerProvisionRequest,
    ProjectComputerReplaceRequest,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
)
from synth_ai.core.research.operations import RESEARCH_OPERATIONS
from synth_ai.core.research.research_intern import (
    ProjectComputerAPI,
    ProjectDataBindingsAPI,
    ResearchInternAPI,
)


class RecordingTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[Any] = []

    def execute(self, request: Any) -> object:
        self.requests.append(request)
        return self.responses.pop(0)


def _intern_wire() -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    return {
        "research_intern_id": "intern-a",
        "org_id": "org-a",
        "display_name": "Research Intern",
        "status": "active",
        "policies": {
            "organization": {},
            "team": {},
            "user": {},
            "organization_policy_ref": None,
            "team_policy_ref": None,
            "user_policy_ref": None,
        },
        "attribution_user_id": "user-a",
        "attribution_team_id": None,
        "state_generation": 0,
        "durable_state": {},
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }


def _decision_wire() -> dict[str, object]:
    return {
        "receipt_id": "receipt-a",
        "research_intern_id": "intern-a",
        "org_id": "org-a",
        "factory_id": "factory-a",
        "project_id": "project-a",
        "experiment_id": "experiment-a",
        "mode": "seraph",
        "canonical_user": "Balthasar",
        "decision_kind": "verdict",
        "idempotency_key": "decision-a",
        "state_generation": 1,
        "evidence_refs": ["trace-v5:publication-a"],
        "state_patch": {"last_verdict": "retain"},
        "rationale": "Heldout evidence improved.",
        "verdict": "retain",
        "uncertainty": 0.1,
        "decided_by_user_id": "user-a",
        "created_at": datetime.now(UTC).isoformat(),
    }


def test_operation_registry_matches_backend_research_intern_routes() -> None:
    expected = {
        "provision_research_intern": ("POST", "/smr/research-intern"),
        "get_research_intern": ("GET", "/smr/research-intern"),
        "patch_research_intern": ("PATCH", "/smr/research-intern"),
        "attach_research_intern_factory": (
            "POST",
            "/smr/research-intern/factories/{factory_id}",
        ),
        "list_research_intern_factories": (
            "GET",
            "/smr/research-intern/factories",
        ),
        "record_magi_decision": (
            "POST",
            "/smr/research-intern/decisions",
        ),
        "list_magi_decisions": (
            "GET",
            "/smr/research-intern/decisions",
        ),
        "provision_project_computer": (
            "POST",
            "/smr/projects/{project_id}/computer",
        ),
        "get_project_computer": (
            "GET",
            "/smr/projects/{project_id}/computer",
        ),
        "replace_project_computer": (
            "POST",
            "/smr/projects/{project_id}/computer/replace",
        ),
        "retire_project_computer": (
            "DELETE",
            "/smr/projects/{project_id}/computer",
        ),
        "create_data_binding": (
            "POST",
            "/smr/projects/{project_id}/data-bindings",
        ),
        "list_data_bindings": (
            "GET",
            "/smr/projects/{project_id}/data-bindings",
        ),
        "create_dataset_revision": (
            "POST",
            (
                "/smr/projects/{project_id}/data-bindings/"
                "{data_binding_id}/revisions"
            ),
        ),
        "list_dataset_revisions": (
            "GET",
            (
                "/smr/projects/{project_id}/data-bindings/"
                "{data_binding_id}/revisions"
            ),
        ),
    }
    for operation_id, (method, path) in expected.items():
        operation = RESEARCH_OPERATIONS[operation_id]
        assert operation.method.value == method
        assert operation.path_template == path


def test_magi_names_are_canonical_and_verdicts_require_seraph() -> None:
    assert MAGI_CANONICAL_USER_BY_MODE == {
        MagiMode.SYNC: MagiCanonicalUser.CASPER,
        MagiMode.ASYNC: MagiCanonicalUser.MELCHIOR,
        MagiMode.SERAPH: MagiCanonicalUser.BALTHASAR,
    }
    with pytest.raises(ValidationError, match="Seraph mode"):
        MagiDecisionRequest(
            mode=MagiMode.SYNC,
            decision_kind=MagiDecisionKind.VERDICT,
            idempotency_key="bad-verdict",
            rationale="This is not a Seraph decision.",
            verdict="retain",
        )


def test_patch_preserves_explicit_null_for_team_attribution_clear() -> None:
    assert ResearchInternPatchRequest(attribution_team_id=None).to_wire() == {
        "attribution_team_id": None
    }


def test_intern_provision_attach_and_decision_keep_one_identity() -> None:
    membership = {
        "research_intern_id": "intern-a",
        "org_id": "org-a",
        "factory_id": "factory-a",
        "role": "owner",
        "attached_by_user_id": "user-a",
        "attached_at": datetime.now(UTC).isoformat(),
    }
    transport = RecordingTransport([_intern_wire(), membership, _decision_wire()])
    api = ResearchInternAPI(transport)  # type: ignore[arg-type]

    intern = api.provision(ResearchInternProvisionRequest())
    attached = api.factories.attach("factory-a")
    decision = api.decisions.record(
        MagiDecisionRequest(
            mode=MagiMode.SERAPH,
            decision_kind=MagiDecisionKind.VERDICT,
            idempotency_key="decision-a",
            factory_id="factory-a",
            project_id="project-a",
            experiment_id="experiment-a",
            evidence_refs=["trace-v5:publication-a"],
            state_patch={"last_verdict": "retain"},
            rationale="Heldout evidence improved.",
            verdict="retain",
            uncertainty=0.1,
        )
    )

    assert intern.research_intern_id == attached.research_intern_id
    assert attached.research_intern_id == decision.research_intern_id
    assert decision.canonical_user is MagiCanonicalUser.BALTHASAR
    assert [request.path for request in transport.requests] == [
        "/smr/research-intern",
        "/smr/research-intern/factories/factory-a",
        "/smr/research-intern/decisions",
    ]


def test_project_computer_and_dataset_revision_are_exactly_scoped() -> None:
    now = datetime.now(UTC).isoformat()
    digest = "sha256:" + "1" * 64
    computer = {
        "project_computer_id": "computer-a",
        "org_id": "org-a",
        "research_intern_id": "intern-a",
        "factory_id": "factory-a",
        "project_id": "project-a",
        "provider_kind": "exe.dev",
        "adapter_kind": "exe.dev.v1",
        "provider_resource_ref": "exe-a",
        "source_repository_id": "repo-a",
        "source_revision": "a" * 40,
        "snapshot_digest": digest,
        "lifecycle": "ready",
        "generation": 1,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }
    binding = {
        "data_binding_id": "binding-a",
        "org_id": "org-a",
        "research_intern_id": "intern-a",
        "factory_id": "factory-a",
        "project_id": "project-a",
        "name": "training",
        "binding_kind": "dataset",
        "authority_ref": "s3://datasets/training",
        "access_policy": {},
        "metadata": {},
        "created_at": now,
    }
    revision = {
        "dataset_revision_id": "revision-a",
        "data_binding_id": "binding-a",
        "org_id": "org-a",
        "factory_id": "factory-a",
        "project_id": "project-a",
        "revision_digest": digest,
        "manifest_uri": "s3://datasets/training/manifest.json",
        "schema_version": "dataset.v1",
        "parent_revision_id": None,
        "metadata": {},
        "created_at": now,
    }
    transport = RecordingTransport([computer, binding, revision])

    computer_api = ProjectComputerAPI(transport)  # type: ignore[arg-type]
    data_api = ProjectDataBindingsAPI(transport)  # type: ignore[arg-type]
    observed_computer = computer_api.provision(
        "project-a",
        ProjectComputerProvisionRequest(
            factory_id="factory-a",
            provider_kind="exe.dev",
            adapter_kind="exe.dev.v1",
            source_repository_id="repo-a",
            source_revision="a" * 40,
            snapshot_digest=digest,
        ),
    )
    observed_binding = data_api.create(
        "project-a",
        DataBindingCreateRequest(
            factory_id="factory-a",
            name="training",
            binding_kind="dataset",
            authority_ref="s3://datasets/training",
        ),
    )
    observed_revision = data_api.create_revision(
        "project-a",
        "binding-a",
        DatasetRevisionCreateRequest(
            revision_digest=digest,
            manifest_uri="s3://datasets/training/manifest.json",
            schema_version="dataset.v1",
        ),
    )

    assert observed_computer.source_revision == "a" * 40
    assert observed_binding.factory_id == "factory-a"
    assert observed_revision.data_binding_id == "binding-a"
    assert [request.path for request in transport.requests] == [
        "/smr/projects/project-a/computer",
        "/smr/projects/project-a/data-bindings",
        "/smr/projects/project-a/data-bindings/binding-a/revisions",
    ]


def test_project_computer_replacement_request_cannot_self_assert_restoration() -> None:
    request = ProjectComputerReplaceRequest(
        factory_id="factory-a",
        provider_kind="exe.dev",
        adapter_kind="exe.dev.v1",
        source_repository_id="repo-a",
        source_revision="b" * 40,
        snapshot_digest="sha256:" + "2" * 64,
    )
    assert "replacement_receipt" not in request.to_wire()
    assert "provider_resource_ref" not in request.to_wire()


def test_project_resource_responses_fail_closed_on_factory_drift() -> None:
    now = datetime.now(UTC).isoformat()
    transport = RecordingTransport(
        [
            {
                "project_computer_id": "computer-a",
                "org_id": "org-a",
                "research_intern_id": "intern-a",
                "factory_id": "factory-b",
                "project_id": "project-a",
                "provider_kind": "exe.dev",
                "adapter_kind": "exe.dev.v1",
                "provider_resource_ref": "exe-a",
                "source_repository_id": "repo-a",
                "source_revision": "a" * 40,
                "snapshot_digest": "sha256:" + "1" * 64,
                "lifecycle": "ready",
                "generation": 1,
                "metadata": {},
                "created_at": now,
                "updated_at": now,
            }
        ]
    )
    with pytest.raises(ValueError, match="requested boundary"):
        ProjectComputerAPI(transport).provision(  # type: ignore[arg-type]
            "project-a",
            ProjectComputerProvisionRequest(
                factory_id="factory-a",
                provider_kind="exe.dev",
                adapter_kind="exe.dev.v1",
                source_repository_id="repo-a",
                source_revision="a" * 40,
            ),
        )
