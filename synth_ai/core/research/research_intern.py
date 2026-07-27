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
    DatasetRevisionLifecycleRequest,
    DatasetRevisionResponse,
    MagiDecisionReceiptResponse,
    MagiDecisionRequest,
    ProjectComputerCleanupReceiptResponse,
    ProjectComputerCleanupRequest,
    ProjectComputerProvisionRequest,
    ProjectComputerReplaceRequest,
    ProjectComputerResponse,
    ResearchInternFactoryMembershipResponse,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
    ResearchInternResponse,
)
from synth_ai.core.research.contracts.dataset_revisions import (
    DatasetRevisionFinalizeRequest,
    DatasetRevisionFinalizeResponse,
    DatasetRevisionPreparationResponse,
    DatasetRevisionPrepareRequest,
)
from synth_ai.core.research.contracts.project_runtime import (
    ProjectComputerExecuteRequest,
    ProjectComputerInspectRequest,
    ProjectComputerLeaseAcquireRequest,
    ProjectComputerLeaseReleaseRequest,
    ProjectComputerLeaseRenewRequest,
    ProjectComputerLeaseResponse,
    ProjectComputerOperationReconcileRequest,
    ProjectRuntimeOperation,
    ProjectRuntimeOperationReceipt,
)
from synth_ai.core.research.operations import (
    dataset_revision_publication_operation,
    research_operation,
)


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _dataset_publication_request(
    operation_id: str,
    path: str,
    *,
    body: JsonObject,
) -> HttpRequest:
    return HttpRequest(
        dataset_revision_publication_operation(operation_id),
        path,
        body=body,
    )


def _validate_operation_receipt(
    receipt: ProjectRuntimeOperationReceipt,
    request: (
        ProjectComputerInspectRequest
        | ProjectComputerExecuteRequest
        | ProjectComputerOperationReconcileRequest
    ),
    *,
    operation: ProjectRuntimeOperation | None,
) -> ProjectRuntimeOperationReceipt:
    if (
        receipt.operation_id != request.operation_id
        or receipt.idempotency_key != request.idempotency_key
        or receipt.resource_generation != request.expected_generation
    ):
        raise ValueError("Project Computer operation receipt identity drifted")
    if operation is not None and receipt.operation is not operation:
        raise ValueError("Project Computer operation receipt kind drifted")
    return receipt


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

    def retrieve(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retrieve the current Project Computer for a Project and Factory."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != str(factory_id)
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
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

    def retire(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retire the Project and Factory Computer while retaining its receipt."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "retire_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != str(factory_id)
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def inspect(
        self,
        project_id: ProjectId,
        request: ProjectComputerInspectRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Inspect one generation without crossing its Factory boundary."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            self._transport.execute(
                _request(
                    "inspect_project_computer",
                    f"/smr/projects/{project_id}/computer/inspect",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.INSPECT,
        )

    def execute(
        self,
        project_id: ProjectId,
        request: ProjectComputerExecuteRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Execute a bounded argv under an exact lease and fencing digest."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            self._transport.execute(
                _request(
                    "execute_project_computer",
                    f"/smr/projects/{project_id}/computer/execute",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.EXECUTE,
        )

    def reconcile(
        self,
        project_id: ProjectId,
        request: ProjectComputerOperationReconcileRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Observe an accepted operation with an unknown provider outcome."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            self._transport.execute(
                _request(
                    "reconcile_project_computer_operation",
                    f"/smr/projects/{project_id}/computer/operations/reconcile",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(receipt, request, operation=None)

    def acquire_lease(
        self,
        project_id: ProjectId,
        request: ProjectComputerLeaseAcquireRequest,
    ) -> ProjectComputerLeaseResponse:
        """Acquire one generation-fenced execution lease."""
        lease = ProjectComputerLeaseResponse.from_wire(
            self._transport.execute(
                _request(
                    "acquire_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease acquisition identity drifted")
        return lease

    def renew_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseRenewRequest,
    ) -> ProjectComputerLeaseResponse:
        """Renew one exact lease while preserving its fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            self._transport.execute(
                _request(
                    "renew_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/renew",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease renewal identity drifted")
        return lease

    def release_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseReleaseRequest,
    ) -> ProjectComputerLeaseResponse:
        """Release one exact lease and retain the final fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            self._transport.execute(
                _request(
                    "release_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/release",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "released"
        ):
            raise ValueError("Project Computer lease release identity drifted")
        return lease

    def cleanup(
        self,
        factory_id: FactoryId,
        request: ProjectComputerCleanupRequest,
    ) -> ProjectComputerCleanupReceiptResponse:
        """Retire every Project Computer in one Factory with owner-authored proof."""
        receipt = ProjectComputerCleanupReceiptResponse.from_wire(
            self._transport.execute(
                _request(
                    "cleanup_factory_project_computers",
                    (
                        "/smr/research-intern/factories/"
                        f"{factory_id}/project-computers/cleanup"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.factory_id != str(factory_id)
            or receipt.idempotency_key != request.idempotency_key
        ):
            raise ValueError("Project Computer cleanup receipt identity drifted")
        return receipt


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
            or str(revision.data_binding_id) != data_binding_id
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
            or str(revision.data_binding_id) != data_binding_id
            for revision in revisions
        ):
            raise ValueError("Dataset Revision list identity drifted")
        return revisions

    def transition_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        dataset_revision_id: str,
        request: DatasetRevisionLifecycleRequest,
    ) -> DatasetRevisionResponse:
        """Apply a fail-closed lifecycle transition to one exact revision."""
        revision = DatasetRevisionResponse.from_wire(
            self._transport.execute(
                _request(
                    "transition_dataset_revision_lifecycle",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revisions/{dataset_revision_id}/lifecycle"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
            or str(revision.dataset_revision_id) != dataset_revision_id
            or revision.state is not request.target_state
        ):
            raise ValueError("DatasetRevision lifecycle response identity drifted")
        return revision

    def prepare_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionPrepareRequest,
    ) -> DatasetRevisionPreparationResponse:
        """Prepare immutable upload targets for one exact DatasetRevision draft."""
        prepared = DatasetRevisionPreparationResponse.from_wire(
            self._transport.execute(
                _dataset_publication_request(
                    "prepareDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions:prepare"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        draft = request.draft
        if (
            prepared.idempotency_key != request.idempotency_key
            or prepared.project_id != str(project_id)
            or str(prepared.data_binding_id) != data_binding_id
            or prepared.dataset_revision_id != draft.dataset_revision_id
            or prepared.factory_id != draft.factory_id
            or prepared.org_id != draft.org_id
        ):
            raise ValueError("DatasetRevision preparation identity drifted")
        return prepared

    def finalize_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        preparation_id: str,
        request: DatasetRevisionFinalizeRequest,
    ) -> DatasetRevisionFinalizeResponse:
        """Finalize one prepared DatasetRevision and verify all owner receipts."""
        finalized = DatasetRevisionFinalizeResponse.from_wire(
            self._transport.execute(
                _dataset_publication_request(
                    "finalizeDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revision-preparations/{preparation_id}:finalize"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        sealed = finalized.sealed_revision
        if (
            str(finalized.preparation_id) != preparation_id
            or sealed.project_id != str(project_id)
            or str(sealed.binding_id) != data_binding_id
        ):
            raise ValueError("DatasetRevision finalization identity drifted")
        return finalized


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

    async def retrieve(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retrieve the current Project Computer for a Project and Factory."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != str(factory_id)
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
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

    async def retire(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retire the Project and Factory Computer while retaining its receipt."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "retire_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if (
            computer.project_id != str(project_id)
            or computer.factory_id != str(factory_id)
        ):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def inspect(
        self,
        project_id: ProjectId,
        request: ProjectComputerInspectRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Inspect one generation without crossing its Factory boundary."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "inspect_project_computer",
                    f"/smr/projects/{project_id}/computer/inspect",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.INSPECT,
        )

    async def execute(
        self,
        project_id: ProjectId,
        request: ProjectComputerExecuteRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Execute a bounded argv under an exact lease and fencing digest."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "execute_project_computer",
                    f"/smr/projects/{project_id}/computer/execute",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.EXECUTE,
        )

    async def reconcile(
        self,
        project_id: ProjectId,
        request: ProjectComputerOperationReconcileRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Observe an accepted operation with an unknown provider outcome."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "reconcile_project_computer_operation",
                    f"/smr/projects/{project_id}/computer/operations/reconcile",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(receipt, request, operation=None)

    async def acquire_lease(
        self,
        project_id: ProjectId,
        request: ProjectComputerLeaseAcquireRequest,
    ) -> ProjectComputerLeaseResponse:
        """Acquire one generation-fenced execution lease."""
        lease = ProjectComputerLeaseResponse.from_wire(
            await self._transport.execute(
                _request(
                    "acquire_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease acquisition identity drifted")
        return lease

    async def renew_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseRenewRequest,
    ) -> ProjectComputerLeaseResponse:
        """Renew one exact lease while preserving its fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            await self._transport.execute(
                _request(
                    "renew_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/renew",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease renewal identity drifted")
        return lease

    async def release_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseReleaseRequest,
    ) -> ProjectComputerLeaseResponse:
        """Release one exact lease and retain the final fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            await self._transport.execute(
                _request(
                    "release_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/release",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "released"
        ):
            raise ValueError("Project Computer lease release identity drifted")
        return lease

    async def cleanup(
        self,
        factory_id: FactoryId,
        request: ProjectComputerCleanupRequest,
    ) -> ProjectComputerCleanupReceiptResponse:
        """Retire every Project Computer in one Factory with owner-authored proof."""
        receipt = ProjectComputerCleanupReceiptResponse.from_wire(
            await self._transport.execute(
                _request(
                    "cleanup_factory_project_computers",
                    (
                        "/smr/research-intern/factories/"
                        f"{factory_id}/project-computers/cleanup"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.factory_id != str(factory_id)
            or receipt.idempotency_key != request.idempotency_key
        ):
            raise ValueError("Project Computer cleanup receipt identity drifted")
        return receipt


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
            or str(revision.data_binding_id) != data_binding_id
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
            or str(revision.data_binding_id) != data_binding_id
            for revision in revisions
        ):
            raise ValueError("Dataset Revision list identity drifted")
        return revisions

    async def transition_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        dataset_revision_id: str,
        request: DatasetRevisionLifecycleRequest,
    ) -> DatasetRevisionResponse:
        """Apply a fail-closed lifecycle transition to one exact revision."""
        revision = DatasetRevisionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "transition_dataset_revision_lifecycle",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revisions/{dataset_revision_id}/lifecycle"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
            or str(revision.dataset_revision_id) != dataset_revision_id
            or revision.state is not request.target_state
        ):
            raise ValueError("DatasetRevision lifecycle response identity drifted")
        return revision

    async def prepare_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionPrepareRequest,
    ) -> DatasetRevisionPreparationResponse:
        """Prepare immutable upload targets for one exact DatasetRevision draft."""
        prepared = DatasetRevisionPreparationResponse.from_wire(
            await self._transport.execute(
                _dataset_publication_request(
                    "prepareDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions:prepare"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        draft = request.draft
        if (
            prepared.idempotency_key != request.idempotency_key
            or prepared.project_id != str(project_id)
            or str(prepared.data_binding_id) != data_binding_id
            or prepared.dataset_revision_id != draft.dataset_revision_id
            or prepared.factory_id != draft.factory_id
            or prepared.org_id != draft.org_id
        ):
            raise ValueError("DatasetRevision preparation identity drifted")
        return prepared

    async def finalize_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        preparation_id: str,
        request: DatasetRevisionFinalizeRequest,
    ) -> DatasetRevisionFinalizeResponse:
        """Finalize one prepared DatasetRevision and verify all owner receipts."""
        finalized = DatasetRevisionFinalizeResponse.from_wire(
            await self._transport.execute(
                _dataset_publication_request(
                    "finalizeDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revision-preparations/{preparation_id}:finalize"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        sealed = finalized.sealed_revision
        if (
            str(finalized.preparation_id) != preparation_id
            or sealed.project_id != str(project_id)
            or str(sealed.binding_id) != data_binding_id
        ):
            raise ValueError("DatasetRevision finalization identity drifted")
        return finalized


__all__ = [
    "AsyncProjectComputerAPI",
    "AsyncProjectDataBindingsAPI",
    "AsyncResearchInternAPI",
    "ProjectComputerAPI",
    "ProjectDataBindingsAPI",
    "ResearchInternAPI",
]
