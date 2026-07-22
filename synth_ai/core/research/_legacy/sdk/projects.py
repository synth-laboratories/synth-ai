"""Project-scoped SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from synth_ai.core.research._legacy.models.canonical_usage import (
    BillingEntitlementSnapshot,
    SmrProjectUsage,
)
from synth_ai.core.research._legacy.models.project import (
    CreateRunnableResult,
    ManagedResearchProject,
)
from synth_ai.core.research._legacy.models.project_workspace import ProjectWorkspaceProjection
from synth_ai.core.research._legacy.models.types import (
    ProjectCodeSource,
    ProjectDataPoolUploadResult,
    ProjectLaunchProfile,
    ProviderKeyStatus,
    SmrLaunchPreflight,
    SmrProjectSetup,
    SmrRunnableProjectRequest,
)
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace
from synth_ai.core.research._legacy.sdk.project import ManagedResearchProjectClient


class ProjectsAPI(_ClientNamespace):
    def create_runnable(
        self,
        request: SmrRunnableProjectRequest | dict[str, Any],
    ) -> CreateRunnableResult:
        return CreateRunnableResult.from_wire(self._client.create_runnable_project(request))

    def list(
        self, *, include_archived: bool = False, limit: int = 100
    ) -> List[ManagedResearchProject]:
        return [
            ManagedResearchProject.from_wire(item)
            for item in self._client.list_projects(
                include_archived=include_archived,
                limit=limit,
            )
        ]

    def get(self, project_id: str) -> ManagedResearchProject:
        return ManagedResearchProject.from_wire(self._client.get_project(project_id))

    def get_schedule(self, project_id: str) -> dict[str, Any]:
        return self.get(project_id).schedule

    def update_schedule(
        self,
        project_id: str,
        schedule: dict[str, Any],
    ) -> ManagedResearchProject:
        return ManagedResearchProject.from_wire(
            self._client.update_project_schedule(project_id, schedule)
        )

    def default(self) -> ManagedResearchProjectClient:
        payload = self._client.get_default_project()
        project = ManagedResearchProject.from_wire(payload)
        return ManagedResearchProjectClient(self._client, project.project_id)

    def patch(self, project_id: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._client.patch_project(project_id, payload, **kwargs)

    def rename(self, project_id: str, name: str) -> dict[str, Any]:
        return self._client.rename_project(project_id, name)

    def pause(self, project_id: str) -> dict[str, Any]:
        return self._client.pause_project(project_id)

    def resume(self, project_id: str) -> dict[str, Any]:
        return self._client.resume_project(project_id)

    def archive(self, project_id: str) -> dict[str, Any]:
        return self._client.archive_project(project_id)

    def unarchive(self, project_id: str) -> dict[str, Any]:
        return self._client.unarchive_project(project_id)

    def get_notes(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_notes(project_id)

    def set_notes(self, project_id: str, notes: str) -> dict[str, Any]:
        return self._client.set_project_notes(project_id, notes)

    def append_notes(self, project_id: str, notes: str) -> dict[str, Any]:
        return self._client.append_project_notes(project_id, notes)

    def get_knowledge(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_knowledge(project_id)

    def set_knowledge(self, project_id: str, content: str) -> dict[str, Any]:
        return self._client.set_project_knowledge(project_id, content)

    def get_status(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_status(project_id)

    def get_status_snapshot(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_status_snapshot(project_id)

    def get_workspace(self, project_id: str) -> ProjectWorkspaceProjection:
        return ProjectWorkspaceProjection.from_wire(self._client.get_project_workspace(project_id))

    def get_code_source(self, project_id: str) -> ProjectCodeSource:
        return ProjectCodeSource.from_wire(self._client.get_project_code_source(project_id))

    def upload_code_bundle(
        self,
        project_id: str,
        bundle_path: str,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        commit_message: str | None = None,
        default_branch: str = "main",
        metadata: Mapping[str, Any] | None = None,
    ) -> ProjectCodeSource:
        return ProjectCodeSource.from_wire(
            self._client.upload_project_code_bundle(
                project_id,
                bundle_path,
                filename=filename,
                content_type=content_type,
                commit_message=commit_message,
                default_branch=default_branch,
                metadata=metadata,
            )
        )

    def connect_git_source(
        self,
        project_id: str,
        *,
        provider: str | None = None,
        repo_url: str | None = None,
        branch: str | None = None,
        auth_ref: str | None = None,
        sync_policy: Mapping[str, Any] | None = None,
    ) -> ProjectCodeSource:
        return ProjectCodeSource.from_wire(
            self._client.connect_project_git_source(
                project_id,
                provider=provider,
                repo_url=repo_url,
                branch=branch,
                auth_ref=auth_ref,
                sync_policy=sync_policy,
            )
        )

    def get_launch_profile(self, project_id: str) -> ProjectLaunchProfile:
        return ProjectLaunchProfile.from_wire(self._client.get_project_launch_profile(project_id))

    def patch_launch_profile(
        self,
        project_id: str,
        launch_profile: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProjectLaunchProfile:
        return ProjectLaunchProfile.from_wire(
            self._client.patch_project_launch_profile(
                project_id,
                launch_profile,
                metadata=metadata,
            )
        )

    def upload_data_pool_files(
        self,
        project_id: str,
        files: List[Mapping[str, Any]],
        *,
        pool_id: str = "default",
        pool_name: str = "Default data pool",
        role: str = "dataset",
        access_policy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProjectDataPoolUploadResult:
        return ProjectDataPoolUploadResult.from_wire(
            self._client.upload_project_data_pool_files(
                project_id,
                files,
                pool_id=pool_id,
                pool_name=pool_name,
                role=role,
                access_policy=access_policy,
                metadata=metadata,
            )
        )

    def upload_data_pool_file(
        self,
        project_id: str,
        path: str,
        *,
        name: str | None = None,
        pool_id: str = "default",
        pool_name: str = "Default data pool",
        role: str = "dataset",
        metadata: Mapping[str, Any] | None = None,
        access_policy: Mapping[str, Any] | None = None,
        pool_metadata: Mapping[str, Any] | None = None,
    ) -> ProjectDataPoolUploadResult:
        return ProjectDataPoolUploadResult.from_wire(
            self._client.upload_project_data_pool_file(
                project_id,
                path,
                name=name,
                pool_id=pool_id,
                pool_name=pool_name,
                role=role,
                metadata=metadata,
                access_policy=access_policy,
                pool_metadata=pool_metadata,
            )
        )

    def list_changesets(
        self,
        project_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_project_changesets(
            project_id,
            status=status,
            limit=limit,
        )

    def create_changeset(
        self,
        project_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._client.create_project_changeset(project_id, payload)

    def get_changeset(
        self,
        project_id: str,
        changeset_id: str,
    ) -> dict[str, Any]:
        return self._client.get_project_changeset(project_id, changeset_id)

    def decide_changeset(
        self,
        project_id: str,
        changeset_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._client.decide_project_changeset(
            project_id,
            changeset_id,
            payload,
        )

    def get_entitlement(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_entitlement(project_id)

    def get_usage(self, project_id: str) -> SmrProjectUsage:
        return self._client.get_project_usage(project_id)

    def get_billing_entitlements(self) -> BillingEntitlementSnapshot:
        return self._client.get_billing_entitlements()

    def get_capabilities(self) -> dict[str, Any]:
        return self._client.get_capabilities()

    def get_agent_models(self) -> dict[str, Any]:
        return self._client.get_agent_models()

    def get_limits(self) -> dict[str, Any]:
        return self._client.get_limits()

    def get_capacity_lane_preview(self, project_id: str) -> dict[str, Any]:
        return self._client.get_capacity_lane_preview(project_id)

    def get_workspace_download_url(self, project_id: str) -> dict[str, Any]:
        return self._client.get_workspace_download_url(project_id)

    def get_git(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_git(project_id)

    def get_setup(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup(project_id))

    def get_setup_authority(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup_authority(project_id))

    def prepare_setup(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.prepare_project_setup(project_id))

    def prepare_setup_authority(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.prepare_project_setup_authority(project_id))

    def start_onboarding(self, project_id: str) -> dict[str, Any]:
        return self._client.start_project_onboarding(project_id)

    def complete_onboarding_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.complete_project_onboarding_step(
            project_id,
            step=step,
            status=status,
            detail=detail,
        )

    def dry_run_onboarding(self, project_id: str) -> dict[str, Any]:
        return self._client.run_project_onboarding_dry_run(project_id)

    def get_onboarding_status(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_onboarding_status(project_id)

    def get_launch_preflight(self, project_id: str, **kwargs: Any) -> SmrLaunchPreflight:
        return SmrLaunchPreflight.from_wire(self._client.get_launch_preflight(project_id, **kwargs))

    def get_run_start_blockers(self, project_id: str, **kwargs: Any) -> SmrLaunchPreflight:
        """Backward-compatible alias for launch preflight readiness checks."""

        return self.get_launch_preflight(project_id, **kwargs)

    def list_objectives(
        self,
        project_id: str,
        *,
        kind: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_objectives(
            project_id,
            kind=kind,
            run_id=run_id,
            limit=limit,
        )

    def set_provider_key(self, project_id: str, **kwargs: Any) -> ProviderKeyStatus:
        return ProviderKeyStatus.from_wire(self._client.set_provider_key(project_id, **kwargs))

    def get_provider_key_status(self, project_id: str, **kwargs: Any) -> ProviderKeyStatus:
        return ProviderKeyStatus.from_wire(
            self._client.get_provider_key_status(project_id, **kwargs)
        )

    def download_workspace_archive(
        self,
        project_id: str,
        output_path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if timeout_seconds is not None:
            return self._client.download_workspace_archive(
                project_id,
                output_path,
                timeout_seconds=timeout_seconds,
            )
        return self._client.download_workspace_archive(project_id, output_path)

    def download_code_archive(
        self,
        project_id: str,
        output_path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if timeout_seconds is not None:
            return self._client.download_project_code_archive(
                project_id,
                output_path,
                timeout_seconds=timeout_seconds,
            )
        return self._client.download_project_code_archive(project_id, output_path)

    def download_code(
        self,
        project_id: str,
        output_path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self.download_code_archive(
            project_id,
            output_path,
            timeout_seconds=timeout_seconds,
        )


__all__ = ["ProjectsAPI"]
