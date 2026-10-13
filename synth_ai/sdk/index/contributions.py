"""Owner workflows and public metadata. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts import operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.errors import AuthenticationError, ContractError
from synth_ai.options import MutationOptions, ReadOptions, TransferOptions
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import CreationResolution, MutationResolution

from .types.contributions import (
    AttestationReceipt,
    Contribution,
    ContributionDraft,
    ContributionFilter,
    ContributionRevision,
    CorrectionReceipt,
    ResearchCorrection,
    RightsAttestation,
    SubmissionReceipt,
    SubmissionRequest,
)
from .types.uploads import LocalContributionPackage, UploadReceipt
from .uploads import AsyncUploads, Uploads


class Contributions:
    def __init__(self, transport: HttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._keyed = keyed
        self.uploads = Uploads(transport)

    def list(
        self,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        filter: ContributionFilter | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Contribution]:
        """Read one bounded authorized page; no implicit all-pages traversal.

        See: backend/notes/specifications/sdk/index_contract.md."""
        selected = filter or ContributionFilter()
        if not self._keyed and selected.status not in {None, "published"}:
            raise AuthenticationError("Anonymous browse is limited to published metadata")
        query = page_parameters(limit, cursor)
        query["query"] = selected.query
        if selected.status is not None:
            query["status"] = selected.status
        result = self._transport.request(
            operations.CONTRIBUTIONS,
            Page[Contribution],
            query=query,
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    len(received.items) > limit
                    or (
                        not self._keyed
                        and any(item.status != "published" for item in received.items)
                    )
                ),
                "Contribution list violated authority or bounds",
            ),
        )
        return result

    def get(self, contribution_id: str, *, request: ReadOptions | None = None) -> Contribution:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = self._transport.request(
            operations.CONTRIBUTION,
            Contribution,
            path=f"/index/contributions/{contribution_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.contribution_id != contribution_id
                    or (not self._keyed and received.status != "published")
                ),
                "Contribution read violated identity or public visibility",
            ),
        )
        return result

    def create(self, *, mutation: MutationOptions | None = None) -> Contribution:
        """Create one durable owner resource; uncertain responses preserve mutation identity.

        See: backend/notes/specifications/sdk/index_contract.md."""
        return self._transport.request(
            operations.CREATE_CONTRIBUTION, Contribution, options=mutation
        )

    def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[Contribution]:
        """Resolve a retained create identity without executing another mutation.

        See: backend/notes/specifications/sdk/index_contract.md."""
        if identity.operation != operations.CREATE_CONTRIBUTION.id:
            raise ValueError("Contribution resolver requires a contribution-create identity")
        resolution = self._transport.request(
            operations.RESOLVE,
            MutationResolution,
            query={
                "key": identity.key,
                "operation": identity.operation,
                "resource_path": identity.resource_path,
                "payload_sha256": identity.payload_sha256,
            },
            options=request,
            validate=lambda received: verified(
                received, received.identity == identity, "Resolver identity mismatch"
            ),
        )
        resource: Contribution | None = None
        if resolution.resource is not None:
            if resolution.resource.kind != "contribution":
                raise ContractError("Contribution resolver returned another resource type")
            resource = self.get(resolution.resource.resource_id, request=request)
        return CreationResolution(resolution.state, resource, resolution.metadata)

    def update(
        self,
        contribution_id: str,
        *,
        draft: ContributionDraft,
        mutation: MutationOptions | None = None,
    ) -> ContributionRevision:
        """Create an owner revision using the expected revision to prevent lost updates.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = self._transport.request(
            operations.UPDATE_CONTRIBUTION,
            ContributionRevision,
            path=f"/index/contributions/{contribution_id}",
            body=draft,
            options=mutation,
            validate=lambda received: verified(
                received,
                received.contribution_id == contribution_id,
                "Contribution receipt identity mismatch",
            ),
        )
        return result

    def attest(
        self,
        contribution_id: str,
        *,
        attestation: RightsAttestation,
        mutation: MutationOptions | None = None,
    ) -> AttestationReceipt:
        """Attest exact revision rights; this does not publish or grant review authority.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = self._transport.request(
            operations.ATTEST,
            AttestationReceipt,
            path=f"/index/contributions/{contribution_id}/attest",
            body=attestation,
            options=mutation,
            validate=lambda received: verified(
                received,
                received.contribution_id == contribution_id,
                "Contribution receipt identity mismatch",
            ),
        )
        return result

    def correct(
        self,
        contribution_id: str,
        *,
        correction: ResearchCorrection,
        mutation: MutationOptions | None = None,
    ) -> CorrectionReceipt:
        """Record an exact source/replacement revision correction with owner authority.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = self._transport.request(
            operations.CORRECT,
            CorrectionReceipt,
            path=f"/index/contributions/{contribution_id}/corrections",
            body=correction,
            options=mutation,
            validate=lambda received: verified(
                received,
                received.contribution_id == contribution_id,
                "Contribution receipt identity mismatch",
            ),
        )
        return result

    def submit(
        self, contribution_id: str, *, revision: str, mutation: MutationOptions | None = None
    ) -> SubmissionReceipt:
        """Submit the exact owner revision; publication remains backend-owned.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        revision = identifier(revision)
        result = self._transport.request(
            operations.SUBMIT,
            SubmissionReceipt,
            path=f"/index/contributions/{contribution_id}/submit",
            body=SubmissionRequest(revision_id=revision),
            options=mutation,
            validate=lambda received: verified(
                received,
                not (
                    received.contribution_id != contribution_id or received.revision_id != revision
                ),
                "Submission receipt identity mismatch",
            ),
        )
        return result

    def upload(
        self,
        contribution_id: str,
        *,
        package: LocalContributionPackage,
        transfer: TransferOptions,
        mutation: MutationOptions | None = None,
    ) -> UploadReceipt:
        """Prepare, transfer and finalize a sealed package; never attest or submit implicitly. Restartable flows use explicit prepare/status/finalize identities.

        See: backend/notes/specifications/sdk/index_contract.md."""
        if mutation is not None and mutation.identity is not None:
            raise ValueError("Use explicit prepare/finalize identities for restartable uploads")
        session = self.uploads.prepare(
            contribution_id, manifest=package.source.manifest, mutation=mutation
        )
        session.transfer(package.source, transfer=transfer)
        return session.finalize(mutation=mutation)


class AsyncContributions:
    def __init__(self, transport: AsyncHttpTransport, keyed: bool) -> None:
        self._transport = transport
        self._keyed = keyed
        self.uploads = AsyncUploads(transport)

    async def list(
        self,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        filter: ContributionFilter | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Contribution]:
        """Read one bounded authorized page; no implicit all-pages traversal.

        See: backend/notes/specifications/sdk/index_contract.md."""
        selected = filter or ContributionFilter()
        if not self._keyed and selected.status not in {None, "published"}:
            raise AuthenticationError("Anonymous browse is limited to published metadata")
        query = page_parameters(limit, cursor)
        query["query"] = selected.query
        if selected.status is not None:
            query["status"] = selected.status
        result = await self._transport.request(
            operations.CONTRIBUTIONS,
            Page[Contribution],
            query=query,
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    len(received.items) > limit
                    or (
                        not self._keyed
                        and any(item.status != "published" for item in received.items)
                    )
                ),
                "Contribution list violated authority or bounds",
            ),
        )
        return result

    async def get(
        self, contribution_id: str, *, request: ReadOptions | None = None
    ) -> Contribution:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = await self._transport.request(
            operations.CONTRIBUTION,
            Contribution,
            path=f"/index/contributions/{contribution_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.contribution_id != contribution_id
                    or (not self._keyed and received.status != "published")
                ),
                "Contribution read violated identity or public visibility",
            ),
        )
        return result

    async def create(self, *, mutation: MutationOptions | None = None) -> Contribution:
        """Create one durable owner resource; uncertain responses preserve mutation identity.

        See: backend/notes/specifications/sdk/index_contract.md."""
        return await self._transport.request(
            operations.CREATE_CONTRIBUTION, Contribution, options=mutation
        )

    async def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[Contribution]:
        """Resolve a retained create identity without executing another mutation.

        See: backend/notes/specifications/sdk/index_contract.md."""
        if identity.operation != operations.CREATE_CONTRIBUTION.id:
            raise ValueError("Contribution resolver requires a contribution-create identity")
        resolution = await self._transport.request(
            operations.RESOLVE,
            MutationResolution,
            query={
                "key": identity.key,
                "operation": identity.operation,
                "resource_path": identity.resource_path,
                "payload_sha256": identity.payload_sha256,
            },
            options=request,
            validate=lambda received: verified(
                received, received.identity == identity, "Resolver identity mismatch"
            ),
        )
        resource: Contribution | None = None
        if resolution.resource is not None:
            if resolution.resource.kind != "contribution":
                raise ContractError("Contribution resolver returned another resource type")
            resource = await self.get(resolution.resource.resource_id, request=request)
        return CreationResolution(resolution.state, resource, resolution.metadata)

    async def update(
        self,
        contribution_id: str,
        *,
        draft: ContributionDraft,
        mutation: MutationOptions | None = None,
    ) -> ContributionRevision:
        """Create an owner revision using the expected revision to prevent lost updates.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = await self._transport.request(
            operations.UPDATE_CONTRIBUTION,
            ContributionRevision,
            path=f"/index/contributions/{contribution_id}",
            body=draft,
            options=mutation,
            validate=lambda received: verified(
                received,
                received.contribution_id == contribution_id,
                "Contribution receipt identity mismatch",
            ),
        )
        return result

    async def attest(
        self,
        contribution_id: str,
        *,
        attestation: RightsAttestation,
        mutation: MutationOptions | None = None,
    ) -> AttestationReceipt:
        """Attest exact revision rights; this does not publish or grant review authority.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = await self._transport.request(
            operations.ATTEST,
            AttestationReceipt,
            path=f"/index/contributions/{contribution_id}/attest",
            body=attestation,
            options=mutation,
            validate=lambda received: verified(
                received,
                received.contribution_id == contribution_id,
                "Contribution receipt identity mismatch",
            ),
        )
        return result

    async def correct(
        self,
        contribution_id: str,
        *,
        correction: ResearchCorrection,
        mutation: MutationOptions | None = None,
    ) -> CorrectionReceipt:
        """Record an exact source/replacement revision correction with owner authority.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = await self._transport.request(
            operations.CORRECT,
            CorrectionReceipt,
            path=f"/index/contributions/{contribution_id}/corrections",
            body=correction,
            options=mutation,
            validate=lambda received: verified(
                received,
                received.contribution_id == contribution_id,
                "Contribution receipt identity mismatch",
            ),
        )
        return result

    async def submit(
        self, contribution_id: str, *, revision: str, mutation: MutationOptions | None = None
    ) -> SubmissionReceipt:
        """Submit the exact owner revision; publication remains backend-owned.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        revision = identifier(revision)
        result = await self._transport.request(
            operations.SUBMIT,
            SubmissionReceipt,
            path=f"/index/contributions/{contribution_id}/submit",
            body=SubmissionRequest(revision_id=revision),
            options=mutation,
            validate=lambda received: verified(
                received,
                not (
                    received.contribution_id != contribution_id or received.revision_id != revision
                ),
                "Submission receipt identity mismatch",
            ),
        )
        return result

    async def upload(
        self,
        contribution_id: str,
        *,
        package: LocalContributionPackage,
        transfer: TransferOptions,
        mutation: MutationOptions | None = None,
    ) -> UploadReceipt:
        """Prepare, transfer and finalize a sealed package; never attest or submit implicitly. Restartable flows use explicit prepare/status/finalize identities.

        See: backend/notes/specifications/sdk/index_contract.md."""
        if mutation is not None and mutation.identity is not None:
            raise ValueError("Use explicit prepare/finalize identities for restartable uploads")
        session = await self.uploads.prepare(
            contribution_id, manifest=package.source.manifest, mutation=mutation
        )
        await session.transfer(package.source, transfer=transfer)
        return await session.finalize(mutation=mutation)
