"""Research Intern, Magi decision, and project resource operations."""

from __future__ import annotations

from typing import cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import FactoryId, ProjectId
from synth_ai.core.research.contracts.research_intern import (
    DataBindingCreateRequest,
    DataBindingResponse,
    DatasetRevisionCreateRequest,
    DatasetRevisionResponse,
    MagiDecisionReceiptResponse,
    MagiDecisionRequest,
    ProjectComputerProvisionRequest,
    ProjectComputerReplaceRequest,
    ProjectComputerResponse,
    ResearchInternFactoryMembershipResponse,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
    ResearchInternResponse,
)
from synth_ai.core.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(research_operation(operation_id), path, body=body)


def _memberships(value: object) -> tuple[ResearchInternFactoryMembershipResponse, ...]:
    return tuple(
        ResearchInternFactoryMembershipResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_research_intern_factories",
        )
    )


def _decisions(value: object) -> tuple[MagiDecisionReceiptResponse, ...]:
    return tuple(
        MagiDecisionReceiptResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_magi_decisions",
        )
    )


def _data_bindings(value: object) -> tuple[DataBindingResponse, ...]:
    return tuple(
        DataBindingResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_data_bindings",
        )
    )


def _dataset_revisions(value: object) -> tuple[DatasetRevisionResponse, ...]:
    return tuple(
        DatasetRevisionResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_dataset_revisions",
        )
    )


class ResearchInternFactoriesAPI:
    """Factory memberships owned by the organization Research Intern."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def attach(self, factory_id: FactoryId) -> ResearchInternFactoryMembershipResponse:
        """Attach one Factory to the organization Research Intern."""
        value = self._transport.execute(
            _request(
                "attach_research_intern_factory",
                f"/smr/research-intern/factories/{factory_id}",
            )
        )
        membership = ResearchInternFactoryMembershipResponse.from_wire(value)
        if membership.factory_id != str(factory_id):
            raise ValueError("Research Intern Factory membership identity drifted")
        return membership

    def list(self) -> tuple[ResearchInternFactoryMembershipResponse, ...]:
        """List Factory memberships for the organization Research Intern."""
        return _memberships(
            self._transport.execute(
                _request(
                    "list_research_intern_factories",
                    "/smr/research-intern/factories",
                )
            )
        )


class ResearchInternDecisionsAPI:
    """Durable Casper, Melchior, and Balthasar decision receipts."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def record(self, request: MagiDecisionRequest) -> MagiDecisionReceiptResponse:
        """Record one idempotent, evidence-linked Magi decision."""
        value = self._transport.execute(
            _request(
                "record_magi_decision",
                "/smr/research-intern/decisions",
                body=cast(JsonObject, request.to_wire()),
            )
        )
        receipt = MagiDecisionReceiptResponse.from_wire(value)
        if receipt.idempotency_key != request.idempotency_key:
            raise ValueError("Magi decision idempotency identity drifted")
        if receipt.factory_id != request.factory_id:
            raise ValueError("Magi decision response crossed its Factory boundary")
        return receipt

    def list(self) -> tuple[MagiDecisionReceiptResponse, ...]:
        """List durable Magi decisions in backend order."""
        return _decisions(
            self._transport.execute(
                _request(
                    "list_magi_decisions",
                    "/smr/research-intern/decisions",
                )
            )
        )


class ResearchInternAPI:
    """One durable organization Intern and its many Factory memberships."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.factories = ResearchInternFactoriesAPI(transport)
        self.decisions = ResearchInternDecisionsAPI(transport)

    def provision(
        self,
        request: ResearchInternProvisionRequest | None = None,
    ) -> ResearchInternResponse:
        """Provision or retrieve the one Intern for the authenticated organization."""
        body = (request or ResearchInternProvisionRequest()).to_wire()
        return ResearchInternResponse.from_wire(
            self._transport.execute(
                _request(
                    "provision_research_intern",
                    "/smr/research-intern",
                    body=cast(JsonObject, body),
                )
            )
        )

    def retrieve(self) -> ResearchInternResponse:
        """Retrieve the authenticated organization's Research Intern."""
        return ResearchInternResponse.from_wire(
            self._transport.execute(
                _request("get_research_intern", "/smr/research-intern")
            )
        )

    def update(self, request: ResearchInternPatchRequest) -> ResearchInternResponse:
        """Update mutable Intern policy, attribution, state, or display fields."""
        return ResearchInternResponse.from_wire(
            self._transport.execute(
                _request(
                    "patch_research_intern",
                    "/smr/research-intern",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )


class ProjectComputerAPI:
    """Provider-neutral Project Computer lifecycle operations."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def provision(
        self,
        project_id: ProjectId,
        request: ProjectComputerProvisionRequest,
    ) -> ProjectComputerResponse:
        """Provision a Project Computer from an exact Git and snapshot identity."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "provision_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != request.factory_id
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def retrieve(self, project_id: ProjectId) -> ProjectComputerResponse:
        """Retrieve the current Project Computer for a Project."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_project_computer",
                    f"/smr/projects/{project_id}/computer",
                )
            )
        )
        if computer.project_id != str(project_id):
            raise ValueError("Project Computer response crossed its project boundary")
        return computer

    def replace(
        self,
        project_id: ProjectId,
        request: ProjectComputerReplaceRequest,
    ) -> ProjectComputerResponse:
        """Replace a Project Computer using an adapter-authored restore receipt."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "replace_project_computer",
                    f"/smr/projects/{project_id}/computer/replace",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != request.factory_id
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def retire(self, project_id: ProjectId) -> ProjectComputerResponse:
        """Retire the current Project Computer while retaining its receipt."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "retire_project_computer",
                    f"/smr/projects/{project_id}/computer",
                )
            )
        )
        if computer.project_id != str(project_id):
            raise ValueError("Project Computer response crossed its project boundary")
        return computer


class ProjectDataBindingsAPI:
    """Project-scoped data bindings and immutable dataset revisions."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(
        self,
        project_id: ProjectId,
        request: DataBindingCreateRequest,
    ) -> DataBindingResponse:
        """Create one Factory-scoped Project data binding."""
        binding = DataBindingResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_data_binding",
                    f"/smr/projects/{project_id}/data-bindings",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            binding.project_id != str(project_id)
            or binding.factory_id != request.factory_id
        ):
            raise ValueError("Data Binding response crossed its requested boundary")
        return binding

    def list(self, project_id: ProjectId) -> tuple[DataBindingResponse, ...]:
        """List data bindings without crossing the requested Project."""
        bindings = _data_bindings(
            self._transport.execute(
                _request(
                    "list_data_bindings",
                    f"/smr/projects/{project_id}/data-bindings",
                )
            )
        )
        if any(binding.project_id != str(project_id) for binding in bindings):
            raise ValueError("Data Binding list crossed its project boundary")
        return bindings

    def create_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionCreateRequest,
    ) -> DatasetRevisionResponse:
        """Append an immutable revision to a Project data binding."""
        revision = DatasetRevisionResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_dataset_revision",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or revision.data_binding_id != data_binding_id
        ):
            raise ValueError("Dataset Revision response identity drifted")
        return revision

    def list_revisions(
        self,
        project_id: ProjectId,
        data_binding_id: str,
    ) -> tuple[DatasetRevisionResponse, ...]:
        """List immutable revisions for one Project data binding."""
        revisions = _dataset_revisions(
            self._transport.execute(
                _request(
                    "list_dataset_revisions",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions"
                    ),
                )
            )
        )
        if any(
            revision.project_id != str(project_id)
            or revision.data_binding_id != data_binding_id
            for revision in revisions
        ):
            raise ValueError("Dataset Revision list identity drifted")
        return revisions


class AsyncResearchInternFactoriesAPI:
    """Native asynchronous Factory membership operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def attach(
        self,
        factory_id: FactoryId,
    ) -> ResearchInternFactoryMembershipResponse:
        """Attach one Factory to the organization Research Intern."""
        membership = ResearchInternFactoryMembershipResponse.from_wire(
            await self._transport.execute(
                _request(
                    "attach_research_intern_factory",
                    f"/smr/research-intern/factories/{factory_id}",
                )
            )
        )
        if membership.factory_id != str(factory_id):
            raise ValueError("Research Intern Factory membership identity drifted")
        return membership

    async def list(self) -> tuple[ResearchInternFactoryMembershipResponse, ...]:
        """List Factory memberships for the organization Research Intern."""
        return _memberships(
            await self._transport.execute(
                _request(
                    "list_research_intern_factories",
                    "/smr/research-intern/factories",
                )
            )
        )


class AsyncResearchInternDecisionsAPI:
    """Native asynchronous Magi decision receipt operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def record(
        self,
        request: MagiDecisionRequest,
    ) -> MagiDecisionReceiptResponse:
        """Record one idempotent, evidence-linked Magi decision."""
        receipt = MagiDecisionReceiptResponse.from_wire(
            await self._transport.execute(
                _request(
                    "record_magi_decision",
                    "/smr/research-intern/decisions",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.idempotency_key != request.idempotency_key:
            raise ValueError("Magi decision idempotency identity drifted")
        if receipt.factory_id != request.factory_id:
            raise ValueError("Magi decision response crossed its Factory boundary")
        return receipt

    async def list(self) -> tuple[MagiDecisionReceiptResponse, ...]:
        """List durable Magi decisions in backend order."""
        return _decisions(
            await self._transport.execute(
                _request(
                    "list_magi_decisions",
                    "/smr/research-intern/decisions",
                )
            )
        )


class AsyncResearchInternAPI:
    """Native asynchronous organization Research Intern operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.factories = AsyncResearchInternFactoriesAPI(transport)
        self.decisions = AsyncResearchInternDecisionsAPI(transport)

    async def provision(
        self,
        request: ResearchInternProvisionRequest | None = None,
    ) -> ResearchInternResponse:
        """Provision or retrieve the one Intern for the authenticated organization."""
        return ResearchInternResponse.from_wire(
            await self._transport.execute(
                _request(
                    "provision_research_intern",
                    "/smr/research-intern",
                    body=cast(
                        JsonObject,
                        (request or ResearchInternProvisionRequest()).to_wire(),
                    ),
                )
            )
        )

    async def retrieve(self) -> ResearchInternResponse:
        """Retrieve the authenticated organization's Research Intern."""
        return ResearchInternResponse.from_wire(
            await self._transport.execute(
                _request("get_research_intern", "/smr/research-intern")
            )
        )

    async def update(
        self,
        request: ResearchInternPatchRequest,
    ) -> ResearchInternResponse:
        """Update mutable Intern policy, attribution, state, or display fields."""
        return ResearchInternResponse.from_wire(
            await self._transport.execute(
                _request(
                    "patch_research_intern",
                    "/smr/research-intern",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )


class AsyncProjectComputerAPI:
    """Native asynchronous Project Computer lifecycle operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def provision(
        self,
        project_id: ProjectId,
        request: ProjectComputerProvisionRequest,
    ) -> ProjectComputerResponse:
        """Provision a Project Computer from an exact Git and snapshot identity."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "provision_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != request.factory_id
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def retrieve(self, project_id: ProjectId) -> ProjectComputerResponse:
        """Retrieve the current Project Computer for a Project."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_project_computer",
                    f"/smr/projects/{project_id}/computer",
                )
            )
        )
        if computer.project_id != str(project_id):
            raise ValueError("Project Computer response crossed its project boundary")
        return computer

    async def replace(
        self,
        project_id: ProjectId,
        request: ProjectComputerReplaceRequest,
    ) -> ProjectComputerResponse:
        """Replace a Project Computer using an adapter-authored restore receipt."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "replace_project_computer",
                    f"/smr/projects/{project_id}/computer/replace",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != request.factory_id
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def retire(self, project_id: ProjectId) -> ProjectComputerResponse:
        """Retire the current Project Computer while retaining its receipt."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "retire_project_computer",
                    f"/smr/projects/{project_id}/computer",
                )
            )
        )
        if computer.project_id != str(project_id):
            raise ValueError("Project Computer response crossed its project boundary")
        return computer


class AsyncProjectDataBindingsAPI:
    """Native asynchronous Project data-binding operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self,
        project_id: ProjectId,
        request: DataBindingCreateRequest,
    ) -> DataBindingResponse:
        """Create one Factory-scoped Project data binding."""
        binding = DataBindingResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_data_binding",
                    f"/smr/projects/{project_id}/data-bindings",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            binding.project_id != str(project_id)
            or binding.factory_id != request.factory_id
        ):
            raise ValueError("Data Binding response crossed its requested boundary")
        return binding

    async def list(
        self,
        project_id: ProjectId,
    ) -> tuple[DataBindingResponse, ...]:
        """List data bindings without crossing the requested Project."""
        bindings = _data_bindings(
            await self._transport.execute(
                _request(
                    "list_data_bindings",
                    f"/smr/projects/{project_id}/data-bindings",
                )
            )
        )
        if any(binding.project_id != str(project_id) for binding in bindings):
            raise ValueError("Data Binding list crossed its project boundary")
        return bindings

    async def create_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionCreateRequest,
    ) -> DatasetRevisionResponse:
        """Append an immutable revision to a Project data binding."""
        revision = DatasetRevisionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_dataset_revision",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or revision.data_binding_id != data_binding_id
        ):
            raise ValueError("Dataset Revision response identity drifted")
        return revision

    async def list_revisions(
        self,
        project_id: ProjectId,
        data_binding_id: str,
    ) -> tuple[DatasetRevisionResponse, ...]:
        """List immutable revisions for one Project data binding."""
        revisions = _dataset_revisions(
            await self._transport.execute(
                _request(
                    "list_dataset_revisions",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions"
                    ),
                )
            )
        )
        if any(
            revision.project_id != str(project_id)
            or revision.data_binding_id != data_binding_id
            for revision in revisions
        ):
            raise ValueError("Dataset Revision list identity drifted")
        return revisions


__all__ = [
    "AsyncProjectComputerAPI",
    "AsyncProjectDataBindingsAPI",
    "AsyncResearchInternAPI",
    "ProjectComputerAPI",
    "ProjectDataBindingsAPI",
    "ResearchInternAPI",
]
