"""Independent Forge workspace lifecycle through the negotiated public API."""

from uuid import NAMESPACE_URL, uuid4, uuid5

from synth_ai.contracts import forge as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import verified
from synth_ai.options import MutationOptions, ReadOptions

from .contracts.projects import (
    ProjectArchive,
    ProjectCreate,
    ProjectMembership,
    ProjectOperation,
    ProjectUpdate,
)
from .index_drafts import AsyncIndexDrafts, IndexDrafts
from .records import AsyncReceipts, AsyncRecords, Receipts, Records, project_identity
from .reports import AsyncReports, Reports
from .types import ProjectListResponse, ProjectMutationResponse, ProjectResponse


def project_operation(project_id, change, expected_revision, mutation):
    selected = mutation or MutationOptions(idempotency_key=uuid4().hex)
    key = selected.identity.key if selected.identity is not None else selected.idempotency_key
    if key is None:
        key = uuid4().hex
        selected = selected.model_copy(update={"idempotency_key": key})
    request = ProjectOperation(
        project_id=project_identity(project_id),
        operation_id=key,
        expected_revision=expected_revision,
        change=change,
    )
    return request, selected


def create_operation(project_id, title, objective, mutation):
    selected = mutation or MutationOptions(idempotency_key=uuid4().hex)
    key = selected.identity.key if selected.identity is not None else selected.idempotency_key
    if key is None:
        key = uuid4().hex
        selected = selected.model_copy(update={"idempotency_key": key})
    return (
        project_id or str(uuid5(NAMESPACE_URL, "synth-forge-project:" + key)),
        ProjectCreate(title=title, objective=objective),
        selected,
    )


def checked_project(result, project_id):
    return verified(
        result, result.project.project_id == project_id, "Forge changed the selected project"
    )


def list_parameters(after, limit):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return {"limit": limit, **({"after": project_identity(after)} if after is not None else {})}


def checked_page(result, after, limit):
    projects = result.page.projects
    identities = [item.project_id for item in projects]
    return verified(
        result,
        len(identities) <= limit
        and identities == sorted(set(identities))
        and (after is None or all(identity > after for identity in identities))
        and len({item.organization_id for item in projects}) <= 1
        and (
            result.page.next_cursor is None
            or (identities and result.page.next_cursor == identities[-1])
        ),
        "Forge returned an invalid project page",
    )


def checked_mutation(result, request):
    return verified(
        result,
        result.receipt.project_id == request.project_id
        and result.receipt.operation_id == request.operation_id,
        "Forge changed the selected project operation",
    )


class Projects:
    def __init__(self, transport: HttpTransport):
        self._transport = transport

    def list(
        self, *, after=None, limit=50, request: ReadOptions | None = None
    ) -> ProjectListResponse:
        query = list_parameters(after, limit)
        return self._transport.request(
            operations.PROJECT_LIST,
            ProjectListResponse,
            query=query,
            options=request,
            validate=lambda result: checked_page(result, after, limit),
        )

    def get(self, project_id: str, *, request: ReadOptions | None = None) -> ProjectResponse:
        project_id = project_identity(project_id)
        return self._transport.request(
            operations.PROJECT_GET,
            ProjectResponse,
            path=operations.PROJECT_GET.path_for(project_id=project_id),
            options=request,
            validate=lambda result: checked_project(result, project_id),
        )

    def apply(
        self, project_id, change, *, expected_revision, mutation: MutationOptions | None = None
    ) -> ProjectMutationResponse:
        request, selected = project_operation(project_id, change, expected_revision, mutation)
        return self._transport.request(
            operations.PROJECT_APPLY,
            ProjectMutationResponse,
            path=operations.PROJECT_APPLY.path_for(project_id=request.project_id),
            body=request,
            options=selected,
            validate=lambda result: checked_mutation(result, request),
        )

    def create(self, *, title, objective, project_id=None, mutation=None):
        project_id, change, selected = create_operation(project_id, title, objective, mutation)
        return self.apply(project_id, change, expected_revision=0, mutation=selected)

    def update(self, project_id, *, title, objective, expected_revision, mutation=None):
        return self.apply(
            project_id,
            ProjectUpdate(title=title, objective=objective),
            expected_revision=expected_revision,
            mutation=mutation,
        )

    def archive(self, project_id, *, archived=True, expected_revision, mutation=None):
        return self.apply(
            project_id,
            ProjectArchive(archived=archived),
            expected_revision=expected_revision,
            mutation=mutation,
        )

    def members(self, project_id, *, members, expected_revision, mutation=None):
        return self.apply(
            project_id,
            ProjectMembership(members=members),
            expected_revision=expected_revision,
            mutation=mutation,
        )


class AsyncProjects:
    def __init__(self, transport: AsyncHttpTransport):
        self._transport = transport

    async def list(
        self, *, after=None, limit=50, request: ReadOptions | None = None
    ) -> ProjectListResponse:
        query = list_parameters(after, limit)
        return await self._transport.request(
            operations.PROJECT_LIST,
            ProjectListResponse,
            query=query,
            options=request,
            validate=lambda result: checked_page(result, after, limit),
        )

    async def get(self, project_id: str, *, request: ReadOptions | None = None) -> ProjectResponse:
        project_id = project_identity(project_id)
        return await self._transport.request(
            operations.PROJECT_GET,
            ProjectResponse,
            path=operations.PROJECT_GET.path_for(project_id=project_id),
            options=request,
            validate=lambda result: checked_project(result, project_id),
        )

    async def apply(
        self, project_id, change, *, expected_revision, mutation: MutationOptions | None = None
    ) -> ProjectMutationResponse:
        request, selected = project_operation(project_id, change, expected_revision, mutation)
        return await self._transport.request(
            operations.PROJECT_APPLY,
            ProjectMutationResponse,
            path=operations.PROJECT_APPLY.path_for(project_id=request.project_id),
            body=request,
            options=selected,
            validate=lambda result: checked_mutation(result, request),
        )

    async def create(self, *, title, objective, project_id=None, mutation=None):
        project_id, change, selected = create_operation(project_id, title, objective, mutation)
        return await self.apply(project_id, change, expected_revision=0, mutation=selected)

    async def update(self, project_id, *, title, objective, expected_revision, mutation=None):
        return await self.apply(
            project_id,
            ProjectUpdate(title=title, objective=objective),
            expected_revision=expected_revision,
            mutation=mutation,
        )

    async def archive(self, project_id, *, archived=True, expected_revision, mutation=None):
        return await self.apply(
            project_id,
            ProjectArchive(archived=archived),
            expected_revision=expected_revision,
            mutation=mutation,
        )

    async def members(self, project_id, *, members, expected_revision, mutation=None):
        return await self.apply(
            project_id,
            ProjectMembership(members=members),
            expected_revision=expected_revision,
            mutation=mutation,
        )


class ForgeClient:
    def __init__(self, transport: HttpTransport):
        self.projects = Projects(transport)
        self.records = Records(transport)
        self.receipts = Receipts(transport)
        self.index_drafts = IndexDrafts(transport)
        self.reports = Reports(transport)


class AsyncForgeClient:
    def __init__(self, transport: AsyncHttpTransport):
        self.projects = AsyncProjects(transport)
        self.records = AsyncRecords(transport)
        self.receipts = AsyncReceipts(transport)
        self.index_drafts = AsyncIndexDrafts(transport)
        self.reports = AsyncReports(transport)
