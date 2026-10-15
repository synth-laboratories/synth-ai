"""First Swarms implementation slice: one model, explicit finite launch and durable identity.

See backend/notes/specifications/sdk/swarms_overrides.md. Provider credentials,
harness and profile selection remain backend-owned; no client fallback routing.
"""

from __future__ import annotations

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.core.recovery import async_resolve_creation as recover_creation_async
from synth_ai.core.recovery import resolve_creation as recover_creation
from synth_ai.errors import ContractError
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import CreationResolution

from .files import AsyncFiles, Files
from .run import AsyncSwarmRun, SwarmRun
from .types.launch import Inputs, LaunchRequest, Limits, Models, Project, Providers
from .types.runs import LaunchPreflight, RunSnapshot


def launch_request(
    objective: str,
    project_id: str,
    limits: Limits,
    inputs: Inputs | None,
    providers: Providers | None,
    models: Models | None,
) -> LaunchRequest:
    """Preserve omitted overrides; backend project/profile defaults remain authoritative."""
    return LaunchRequest(
        objective=objective,
        project_id=project_id,
        limits=limits,
        inputs=inputs,
        providers=providers,
        models=models,
    )


def validate_preflight(result: LaunchPreflight, request: LaunchRequest) -> LaunchPreflight:
    """No model/provider substitution, additional files or widened limits."""
    configuration = result.configuration
    if configuration is None:
        return result
    resolved = configuration.limits
    requested = request.limits
    amount = resolved.maximum_charge
    ceiling = requested.maximum_charge
    within_budget = amount.coefficient * 10**ceiling.scale <= ceiling.coefficient * 10**amount.scale
    same_inputs = request.inputs is None or configuration.inputs == request.inputs
    if (
        configuration.project_id != request.project_id
        or (request.models is not None and configuration.models.default != request.models.default)
        or (
            request.providers is not None
            and configuration.providers.default != request.providers.default
        )
        or (not same_inputs)
        or (not within_budget)
        or (resolved.wall_time_seconds > requested.wall_time_seconds)
        or (resolved.maximum_tasks > requested.maximum_tasks)
        or (resolved.maximum_parallel_tasks > requested.maximum_parallel_tasks)
    ):
        raise ContractError("Preflight changed selected inputs/model/provider or widened limits")
    return result


class Projects:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, project_id: str, *, request: ReadOptions | None = None) -> Project:
        """Inspect an existing project; no implicit project creation."""
        project_id = identifier(project_id)
        return self._transport.request(
            operations.PROJECT,
            Project,
            path=f"/swarms/projects/{project_id}",
            options=request,
            validate=lambda result: verified(
                result, result.project_id == project_id, "Project identity mismatch"
            ),
        )


class SwarmsClient:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.projects = Projects(transport)
        self.files = Files(transport)

    def preflight(
        self,
        objective: str,
        *,
        project_id: str,
        limits: Limits,
        inputs: Inputs | None = None,
        providers: Providers | None = None,
        models: Models | None = None,
        request: ReadOptions | None = None,
    ) -> LaunchPreflight:
        """Preview qualification without reserving budget or starting a run."""
        body = launch_request(objective, project_id, limits, inputs, providers, models)
        return self._transport.request(
            operations.PREFLIGHT,
            LaunchPreflight,
            body=body,
            options=request,
            validate=lambda result: validate_preflight(result, body),
        )

    def create(
        self,
        objective: str,
        *,
        project_id: str,
        limits: Limits,
        inputs: Inputs | None = None,
        providers: Providers | None = None,
        models: Models | None = None,
        mutation: MutationOptions | None = None,
    ) -> SwarmRun:
        """Admit exactly the submitted single-model intent and retain uncertain identity."""
        body = launch_request(objective, project_id, limits, inputs, providers, models)
        result = self._transport.request(
            operations.CREATE,
            RunSnapshot,
            body=body,
            options=mutation,
            validate=lambda result: verified(
                result,
                result.project_id == project_id
                and (models is None or result.model == models.default)
                and (providers is None or result.provider == providers.default),
                "Admitted project or requested override mismatch",
            ),
        )
        return SwarmRun(self._transport, result.run_id)

    def get(self, run_id: str, *, request: ReadOptions | None = None) -> SwarmRun:
        """Reattach to an authorized durable run without restarting it."""
        run_id = identifier(run_id)
        self._transport.request(
            operations.GET,
            RunSnapshot,
            path=f"/swarms/runs/{run_id}",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == run_id, "Run identity mismatch"
            ),
        )
        return SwarmRun(self._transport, run_id)

    def list(
        self,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        project_id: str | None = None,
        request: ReadOptions | None = None,
    ) -> Page[RunSnapshot]:
        """Read one current authorized page, optionally scoped to a project."""
        query = page_parameters(limit, cursor)
        if project_id is not None:
            query["project_id"] = identifier(project_id)
        return self._transport.request(
            operations.LIST,
            Page[RunSnapshot],
            query=query,
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and (
                    project_id is None
                    or all(item.project_id == project_id for item in result.items)
                ),
                "Run page scope or bound mismatch",
            ),
        )

    def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[SwarmRun]:
        return recover_creation(
            self._transport,
            identity,
            operations.CREATE,
            "swarm",
            lambda resource_id: SwarmRun(self._transport, resource_id),
            request,
        )


class AsyncProjects:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, project_id: str, *, request: ReadOptions | None = None) -> Project:
        """Inspect an existing project; no implicit project creation."""
        project_id = identifier(project_id)
        return await self._transport.request(
            operations.PROJECT,
            Project,
            path=f"/swarms/projects/{project_id}",
            options=request,
            validate=lambda result: verified(
                result, result.project_id == project_id, "Project identity mismatch"
            ),
        )


class AsyncSwarmsClient:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.projects = AsyncProjects(transport)
        self.files = AsyncFiles(transport)

    async def preflight(
        self,
        objective: str,
        *,
        project_id: str,
        limits: Limits,
        inputs: Inputs | None = None,
        providers: Providers | None = None,
        models: Models | None = None,
        request: ReadOptions | None = None,
    ) -> LaunchPreflight:
        """Preview qualification without reserving budget or starting a run."""
        body = launch_request(objective, project_id, limits, inputs, providers, models)
        return await self._transport.request(
            operations.PREFLIGHT,
            LaunchPreflight,
            body=body,
            options=request,
            validate=lambda result: validate_preflight(result, body),
        )

    async def create(
        self,
        objective: str,
        *,
        project_id: str,
        limits: Limits,
        inputs: Inputs | None = None,
        providers: Providers | None = None,
        models: Models | None = None,
        mutation: MutationOptions | None = None,
    ) -> AsyncSwarmRun:
        """Admit exactly the submitted single-model intent and retain uncertain identity."""
        body = launch_request(objective, project_id, limits, inputs, providers, models)
        result = await self._transport.request(
            operations.CREATE,
            RunSnapshot,
            body=body,
            options=mutation,
            validate=lambda result: verified(
                result,
                result.project_id == project_id
                and (models is None or result.model == models.default)
                and (providers is None or result.provider == providers.default),
                "Admitted project or requested override mismatch",
            ),
        )
        return AsyncSwarmRun(self._transport, result.run_id)

    async def get(self, run_id: str, *, request: ReadOptions | None = None) -> AsyncSwarmRun:
        """Reattach to an authorized durable run without restarting it."""
        run_id = identifier(run_id)
        await self._transport.request(
            operations.GET,
            RunSnapshot,
            path=f"/swarms/runs/{run_id}",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == run_id, "Run identity mismatch"
            ),
        )
        return AsyncSwarmRun(self._transport, run_id)

    async def list(
        self,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        project_id: str | None = None,
        request: ReadOptions | None = None,
    ) -> Page[RunSnapshot]:
        """Read one current authorized page, optionally scoped to a project."""
        query = page_parameters(limit, cursor)
        if project_id is not None:
            query["project_id"] = identifier(project_id)
        return await self._transport.request(
            operations.LIST,
            Page[RunSnapshot],
            query=query,
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and (
                    project_id is None
                    or all(item.project_id == project_id for item in result.items)
                ),
                "Run page scope or bound mismatch",
            ),
        )

    async def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[AsyncSwarmRun]:
        async def attach(resource_id: str) -> AsyncSwarmRun:
            return AsyncSwarmRun(self._transport, resource_id)

        return await recover_creation_async(
            self._transport, identity, operations.CREATE, "swarm", attach, request
        )
