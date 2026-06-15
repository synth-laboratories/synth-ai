"""Bound project handle for the flatter noun-first SDK surface."""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from synth_ai.managed_research.models.canonical_usage import (
    SmrProjectUsage,
    SmrResourceLimitExtension,
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrResourceLimitSelector,
)
from synth_ai.managed_research.models.project import ManagedResearchProject
from synth_ai.managed_research.models.project_workspace import ProjectWorkspaceProjection
from synth_ai.managed_research.models.run_control import (
    ManagedResearchActorControlAck,
    ManagedResearchActorControlAction,
)
from synth_ai.managed_research.models.run_state import ManagedResearchRun
from synth_ai.managed_research.models.types import SmrLaunchPreflight, SmrProjectSetup


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


@dataclass
class _BoundProjectReposAPI:
    _client: Any
    project_id: str

    def list(self) -> List[dict[str, Any]]:
        return self._client.list_project_repo_bindings(self.project_id)

    def attach(
        self,
        *,
        github_repo: str,
        pr_write_enabled: bool = False,
    ) -> dict[str, Any]:
        return self._client.attach_project_repo(
            self.project_id,
            repo=github_repo,
            pr_write_enabled=pr_write_enabled,
        )

    def detach(self, *, github_repo: str) -> dict[str, Any]:
        return self._client.detach_project_repo(self.project_id, repo=github_repo)


@dataclass
class _BoundProjectExternalRepositoriesAPI:
    _client: Any
    project_id: str

    def list(self) -> List[dict[str, Any]]:
        return self._client.list_project_external_repositories(self.project_id)

    def create(
        self,
        *,
        name: str,
        url: str,
        default_branch: str | None = None,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.create_project_external_repository(
            self.project_id,
            name=name,
            url=url,
            default_branch=default_branch,
            role=role,
            metadata=metadata,
        )

    def patch(
        self,
        repository_id: str,
        *,
        url: str | None = None,
        default_branch: str | None = None,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.patch_project_external_repository(
            self.project_id,
            repository_id,
            url=url,
            default_branch=default_branch,
            role=role,
            metadata=metadata,
        )


@dataclass
class _BoundProjectFilesAPI:
    _client: Any
    project_id: str

    def list(
        self,
        *,
        visibility: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_project_files(
            self.project_id,
            visibility=visibility,
            limit=limit,
        )

    def upload(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        visibility: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = Path(path)
        raw_bytes = file_path.read_bytes()
        payload = {
            "path": name or file_path.name,
            "content": base64.b64encode(raw_bytes).decode("ascii"),
            "content_type": _guess_content_type(file_path),
            "encoding": "base64",
            "visibility": visibility,
            "metadata": dict(metadata or {}),
        }
        return self._client.create_project_files(self.project_id, [payload])

    def upload_source_bundle(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        visibility: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = Path(path)
        return self._client.create_project_source_bundle(
            self.project_id,
            file_path,
            path=name or file_path.name,
            visibility=visibility,
            metadata=metadata,
        )

    def content(self, file_id: str) -> dict[str, Any]:
        return self._client.get_project_file_content(self.project_id, file_id)


@dataclass
class _BoundProjectDatasetsAPI:
    _client: Any
    project_id: str

    def list(self) -> List[dict[str, Any]]:
        return self._client.list_project_datasets(self.project_id)

    def upload(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        format: str | None = None,
        row_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        file_path = Path(path)
        raw_bytes = file_path.read_bytes()
        return self._client.upload_project_dataset(
            self.project_id,
            {
                "name": name or file_path.name,
                "content": base64.b64encode(raw_bytes).decode("ascii"),
                "encoding": "base64",
                "content_type": _guess_content_type(file_path),
                "format": format,
                "row_count": row_count,
                "metadata": dict(metadata or {}),
            },
        )

    def download(
        self,
        dataset_id: str,
        *,
        to: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = self._client.download_project_dataset(self.project_id, dataset_id)
        destination = Path(to) if to is not None else None
        if destination is not None:
            if payload.get("encoding") == "base64":
                destination.write_bytes(base64.b64decode(str(payload["content"])))
            else:
                destination.write_text(str(payload["content"]), encoding="utf-8")
        return payload


@dataclass
class _BoundProjectContextAPI:
    _client: Any
    project_id: str

    def get_notes(self) -> dict[str, Any]:
        return self._client.get_project_notes(self.project_id)

    def set_notes(self, notes: str) -> dict[str, Any]:
        return self._client.set_project_notes(self.project_id, notes)

    def append_notes(self, notes: str) -> dict[str, Any]:
        return self._client.append_project_notes(self.project_id, notes)

    def get_project_knowledge(self) -> dict[str, Any]:
        return self._client.get_project_knowledge(self.project_id)

    def set_project_knowledge(self, content: str) -> dict[str, Any]:
        return self._client.set_project_knowledge(self.project_id, content)

    def get_org_knowledge(self) -> dict[str, Any]:
        return self._client.get_org_knowledge()

    def set_org_knowledge(self, content: str) -> dict[str, Any]:
        return self._client.set_org_knowledge(content)


@dataclass
class _BoundProjectCredentialsAPI:
    _client: Any
    project_id: str

    def list(
        self,
        *,
        kind: str | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_project_credential_refs(
            self.project_id,
            kind=kind,
        )

    def create(
        self,
        *,
        kind: str,
        label: str,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.create_project_credential_ref(
            self.project_id,
            kind=kind,
            label=label,
            provider=provider,
            funding_source=funding_source,
            credential_name=credential_name,
            metadata=metadata,
        )

    def patch(
        self,
        credential_ref_id: str,
        *,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.patch_project_credential_ref(
            self.project_id,
            credential_ref_id,
            provider=provider,
            funding_source=funding_source,
            credential_name=credential_name,
            metadata=metadata,
        )


@dataclass
class _BoundProjectOutputsAPI:
    _client: Any
    project_id: str

    def list(self) -> List[dict[str, Any]]:
        return self._client.list_project_outputs(self.project_id)


@dataclass
class _BoundProjectPrsAPI:
    _client: Any
    project_id: str

    def list(self) -> List[dict[str, Any]]:
        return self._client.list_project_prs(self.project_id)

    def get(self, pr_id: str) -> dict[str, Any]:
        return self._client.get_project_pr(self.project_id, pr_id)


@dataclass
class _BoundProjectSetupAPI:
    _client: Any
    project_id: str

    def get(self) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup(self.project_id))

    def get_authority(self) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup_authority(self.project_id))

    def prepare(self) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.prepare_project_setup(self.project_id))

    def prepare_authority(self) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(
            self._client.prepare_project_setup_authority(self.project_id)
        )

    def start_onboarding(self) -> dict[str, Any]:
        return self._client.start_project_onboarding(self.project_id)

    def complete_onboarding_step(
        self,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.complete_project_onboarding_step(
            self.project_id,
            step=step,
            status=status,
            detail=detail,
        )

    def dry_run_onboarding(self) -> dict[str, Any]:
        return self._client.run_project_onboarding_dry_run(self.project_id)

    def get_onboarding_status(self) -> dict[str, Any]:
        return self._client.get_project_onboarding_status(self.project_id)


@dataclass
class _BoundProjectModelsAPI:
    _client: Any
    project_id: str

    def list(self) -> List[dict[str, Any]]:
        return self._client.list_project_models(self.project_id)

    def get(self, model_id: str) -> dict[str, Any]:
        return self._client.get_project_model(self.project_id, model_id)

    def download(
        self,
        model_id: str,
        *,
        to: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = self._client.download_project_model(self.project_id, model_id)
        destination = Path(to) if to is not None else None
        if destination is not None:
            if payload.get("encoding") == "base64":
                destination.write_bytes(base64.b64decode(str(payload["content"])))
            else:
                destination.write_text(str(payload["content"]), encoding="utf-8")
        return payload

    def export(self, model_id: str) -> dict[str, Any]:
        return self._client.export_project_model(self.project_id, model_id)


@dataclass
class _BoundProjectRunsAPI:
    _client: Any
    project_id: str

    def start(self, objective: str, **kwargs: Any):
        return self._client.runs.start(
            objective,
            project_id=self.project_id,
            **kwargs,
        )

    def launch(self, objective: str, **kwargs: Any):
        return self.start(objective, **kwargs)

    def trigger(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.trigger_run(self.project_id, **kwargs)

    def preflight(self, **kwargs: Any) -> SmrLaunchPreflight:
        return SmrLaunchPreflight.from_wire(
            self._client.get_launch_preflight(self.project_id, **kwargs)
        )

    def launch_preflight(self, **kwargs: Any) -> SmrLaunchPreflight:
        return self.preflight(**kwargs)

    def list(self, *, active_only: bool = False, **kwargs: Any) -> List[dict[str, Any]]:
        return self._client.list_runs(
            self.project_id,
            active_only=active_only,
            **kwargs,
        )

    def list_active(self) -> List[dict[str, Any]]:
        return self._client.list_active_runs(self.project_id)

    def get(self, run_id: str) -> ManagedResearchRun:
        return self._client.runs.get(run_id, project_id=self.project_id)

    def control_actor(
        self,
        run_id: str,
        actor_id: str,
        *,
        action: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self._client.runs.control_actor(
            self.project_id,
            run_id,
            actor_id,
            action=action,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def pause_actor(
        self,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            run_id,
            actor_id,
            action=ManagedResearchActorControlAction.PAUSE.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def resume_actor(
        self,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            run_id,
            actor_id,
            action=ManagedResearchActorControlAction.RESUME.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def interrupt_actor(
        self,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            run_id,
            actor_id,
            action=ManagedResearchActorControlAction.INTERRUPT.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )


@dataclass
class _BoundProjectObjectivesAPI:
    _client: Any
    project_id: str

    def list(
        self,
        *,
        kind: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_objectives(
            self.project_id,
            kind=kind,
            run_id=run_id,
            limit=limit,
        )

    def create(self, *, kind: str, **payload: Any) -> dict[str, Any]:
        return self._client.create_objective(
            self.project_id,
            {"kind": kind, **payload},
        )

    def get(self, objective_id: str, *, kind: str | None = None) -> dict[str, Any]:
        return self._client.get_objective(self.project_id, objective_id, kind=kind)

    def status(
        self,
        objective_id: str,
        *,
        kind: str | None = None,
        task_limit: int | None = None,
        claim_limit: int | None = None,
        event_limit: int | None = 50,
        milestone_limit: int | None = None,
    ) -> dict[str, Any]:
        return self._client.get_objective_status(
            self.project_id,
            objective_id,
            kind=kind,
            task_limit=task_limit,
            claim_limit=claim_limit,
            event_limit=event_limit,
            milestone_limit=milestone_limit,
        )

    def patch(
        self,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return self._client.patch_objective(
            self.project_id,
            objective_id,
            payload,
            kind=kind,
        )

    def pause(self, objective_id: str, *, kind: str | None = None) -> dict[str, Any]:
        return self._client.pause_objective(self.project_id, objective_id, kind=kind)

    def resume(self, objective_id: str, *, kind: str | None = None) -> dict[str, Any]:
        return self._client.resume_objective(self.project_id, objective_id, kind=kind)

    def withdraw(self, objective_id: str, *, kind: str | None = None) -> dict[str, Any]:
        return self._client.withdraw_objective(self.project_id, objective_id, kind=kind)

    def progress(self, objective_id: str, *, kind: str | None = None) -> dict[str, Any]:
        return self._client.get_objective_progress(
            self.project_id,
            objective_id,
            kind=kind,
        )

    def tasks(
        self,
        objective_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_objective_tasks(
            self.project_id,
            objective_id,
            kind=kind,
            limit=limit,
        )

    def claims(
        self,
        objective_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_objective_claims(
            self.project_id,
            objective_id,
            kind=kind,
            limit=limit,
        )

    def create_claim(
        self,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return self._client.create_objective_claim(
            self.project_id,
            objective_id,
            payload,
            kind=kind,
        )

    def request_review(
        self,
        objective_id: str,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request_objective_review(
            self.project_id,
            objective_id,
            payload,
            kind=kind,
        )


@dataclass
class _BoundProjectMilestonesAPI:
    _client: Any
    project_id: str

    def list(
        self,
        *,
        run_id: str | None = None,
        parent_kind: str | None = None,
        parent_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_project_milestones(
            self.project_id,
            run_id=run_id,
            parent_kind=parent_kind,
            parent_id=parent_id,
            limit=limit,
        )

    def create(self, payload: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
        return self._client.create_project_milestone(self.project_id, payload)

    def get(self, milestone_id: str) -> dict[str, Any]:
        return self._client.get_project_milestone(self.project_id, milestone_id)

    def patch(
        self,
        milestone_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return self._client.patch_project_milestone(
            self.project_id,
            milestone_id,
            payload,
        )

    def transition(
        self,
        milestone_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return self._client.transition_project_milestone(
            self.project_id,
            milestone_id,
            payload,
        )


@dataclass
class _BoundProjectChangeSetsAPI:
    _client: Any
    project_id: str

    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_project_changesets(
            self.project_id,
            status=status,
            limit=limit,
        )

    def create(self, payload: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
        return self._client.create_project_changeset(self.project_id, payload)

    def get(self, changeset_id: str) -> dict[str, Any]:
        return self._client.get_project_changeset(self.project_id, changeset_id)

    def decide(
        self,
        changeset_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
    ) -> dict[str, Any]:
        return self._client.decide_project_changeset(
            self.project_id,
            changeset_id,
            payload,
        )


@dataclass
class ManagedResearchProjectClient:
    _client: Any
    project_id: str
    _repos_api: _BoundProjectReposAPI | None = field(init=False, default=None, repr=False)
    _external_repositories_api: _BoundProjectExternalRepositoriesAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _files_api: _BoundProjectFilesAPI | None = field(init=False, default=None, repr=False)
    _datasets_api: _BoundProjectDatasetsAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _context_api: _BoundProjectContextAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _credentials_api: _BoundProjectCredentialsAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _outputs_api: _BoundProjectOutputsAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _prs_api: _BoundProjectPrsAPI | None = field(init=False, default=None, repr=False)
    _setup_api: _BoundProjectSetupAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _models_api: _BoundProjectModelsAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _runs_api: _BoundProjectRunsAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _objectives_api: _BoundProjectObjectivesAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _milestones_api: _BoundProjectMilestonesAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _changesets_api: _BoundProjectChangeSetsAPI | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def repositories(self) -> _BoundProjectReposAPI:
        if self._repos_api is None:
            self._repos_api = _BoundProjectReposAPI(self._client, self.project_id)
        return self._repos_api

    @property
    def repos(self) -> _BoundProjectReposAPI:
        return self.repositories

    @property
    def external_repositories(self) -> _BoundProjectExternalRepositoriesAPI:
        if self._external_repositories_api is None:
            self._external_repositories_api = _BoundProjectExternalRepositoriesAPI(
                self._client,
                self.project_id,
            )
        return self._external_repositories_api

    @property
    def files(self) -> _BoundProjectFilesAPI:
        if self._files_api is None:
            self._files_api = _BoundProjectFilesAPI(self._client, self.project_id)
        return self._files_api

    @property
    def datasets(self) -> _BoundProjectDatasetsAPI:
        if self._datasets_api is None:
            self._datasets_api = _BoundProjectDatasetsAPI(self._client, self.project_id)
        return self._datasets_api

    @property
    def context(self) -> _BoundProjectContextAPI:
        if self._context_api is None:
            self._context_api = _BoundProjectContextAPI(self._client, self.project_id)
        return self._context_api

    @property
    def credentials(self) -> _BoundProjectCredentialsAPI:
        if self._credentials_api is None:
            self._credentials_api = _BoundProjectCredentialsAPI(
                self._client,
                self.project_id,
            )
        return self._credentials_api

    @property
    def outputs(self) -> _BoundProjectOutputsAPI:
        if self._outputs_api is None:
            self._outputs_api = _BoundProjectOutputsAPI(self._client, self.project_id)
        return self._outputs_api

    @property
    def prs(self) -> _BoundProjectPrsAPI:
        if self._prs_api is None:
            self._prs_api = _BoundProjectPrsAPI(self._client, self.project_id)
        return self._prs_api

    @property
    def setup(self) -> _BoundProjectSetupAPI:
        if self._setup_api is None:
            self._setup_api = _BoundProjectSetupAPI(self._client, self.project_id)
        return self._setup_api

    @property
    def models(self) -> _BoundProjectModelsAPI:
        if self._models_api is None:
            self._models_api = _BoundProjectModelsAPI(self._client, self.project_id)
        return self._models_api

    @property
    def runs(self) -> _BoundProjectRunsAPI:
        if self._runs_api is None:
            self._runs_api = _BoundProjectRunsAPI(self._client, self.project_id)
        return self._runs_api

    @property
    def objectives(self) -> _BoundProjectObjectivesAPI:
        if self._objectives_api is None:
            self._objectives_api = _BoundProjectObjectivesAPI(
                self._client,
                self.project_id,
            )
        return self._objectives_api

    @property
    def milestones(self) -> _BoundProjectMilestonesAPI:
        if self._milestones_api is None:
            self._milestones_api = _BoundProjectMilestonesAPI(
                self._client,
                self.project_id,
            )
        return self._milestones_api

    @property
    def changesets(self) -> _BoundProjectChangeSetsAPI:
        if self._changesets_api is None:
            self._changesets_api = _BoundProjectChangeSetsAPI(
                self._client,
                self.project_id,
            )
        return self._changesets_api

    def get(self) -> ManagedResearchProject:
        return ManagedResearchProject.from_wire(self._client.get_project(self.project_id))

    def readiness(self) -> dict[str, Any]:
        return self._client.get_project_readiness(self.project_id)

    def workspace(self) -> ProjectWorkspaceProjection:
        return ProjectWorkspaceProjection.from_wire(
            self._client.get_project_workspace(self.project_id)
        )

    def usage(self) -> SmrProjectUsage:
        return self._client.get_project_usage(self.project_id)

    def resource_limits(self) -> SmrResourceLimits:
        return self._client.get_project_resource_limits(self.project_id)

    def progress_toward_resource_limits(self) -> SmrResourceLimitProgress:
        return self._client.get_project_progress_toward_resource_limits(self.project_id)

    def extend_resource_limit(
        self,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str = "spend_usd",
        unit: str = "usd",
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return self._client.extend_project_resource_limit(
            self.project_id,
            limit_value=limit_value,
            additional_value=additional_value,
            reason=reason,
            selector=selector,
            resource_limit_id=resource_limit_id,
            metric=metric,
            unit=unit,
            resolve_blockers=resolve_blockers,
            resume=resume,
            idempotency_key=idempotency_key,
        )

    def get_schedule(self) -> dict[str, Any]:
        return self.get().schedule

    def update_schedule(self, schedule: dict[str, Any]) -> ManagedResearchProject:
        return ManagedResearchProject.from_wire(
            self._client.update_project_schedule(self.project_id, schedule)
        )


__all__ = ["ManagedResearchProjectClient"]
