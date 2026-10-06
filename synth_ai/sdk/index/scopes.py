"""OAuth scope classes for Contribution lifecycle and private QA operations.

Scopes gate the CLASS of operation only. Who may act on which case (contributor
ownership, reviewer assignment, coordinator role, organization isolation) is
decided by the backend on every request. This table is advertisement and error
guidance, never a grant: the backend table
(``packages/auth/mcp_oauth.py::INDEX_OPERATIONS``) is the authority and the
SDK test suite asserts equality with it.

Each entry is any-of, ordered least privilege first. A token holding any listed
scope may reach the route; it is still refused when the role or assignment
check fails.
"""

from __future__ import annotations

from collections.abc import Mapping

INTAKE = "index:intake"
REVIEW = "index:review"
COORDINATE = "index:coordinate"
PUBLISH = "index:publish"
QA_READ = "index:qa:read"


# operation_id -> any-of scopes. Operation ids are the keys of client.OPERATIONS.
OPERATION_SCOPES: Mapping[str, tuple[str, ...]] = {
    "index.research.forge_notices.list": ("index:read",),
    # Contribution lifecycle owned by the contributor.
    "index.contributions.create": ("index:intake",),
    "index.contributions.revisions.create": ("index:intake",),
    "index.contributions.upload.prepare": ("index:intake",),
    "index.contributions.upload.finalize": ("index:intake",),
    "index.contributions.submit": ("index:intake",),
    "index.contributions.research.correction_registration": ("index:intake",),
    "index.research.rights_attestation.create": ("index:intake",),
    # Publication is a separate authority; QA acceptance never grants it.
    "index.contributions.publication.create": ("index:publish",),
    "index.contributions.withdrawal.create": ("index:publish",),
    # QA: a contributor with only index:intake can create and read its own case,
    # respond, appeal and escalate; reading requires a separate read/review/coordinate scope.
    "index.qa.cases.create": ("index:intake",),
    "index.qa.cases.get": ("index:read", "index:review", "index:coordinate"),
    "index.qa.events.list": ("index:read", "index:review", "index:coordinate"),
    "index.qa.events.create": ("index:intake", "index:coordinate", "index:review"),
    "index.qa.appeals.create": ("index:intake",),
    "index.qa.escalations.create": ("index:intake", "index:review"),
    "index.qa.notes.create": ("index:review", "index:coordinate"),
    "index.qa.adjudications.create": ("index:coordinate",),
    "index.qa.assignments.create": ("index:coordinate",),
    "index.qa.assignments.revoke": ("index:coordinate",),
    "index.qa.assignments.list": ("index:read", "index:review", "index:coordinate"),
    "index.qa.assignments.accept": ("index:review",),
    "index.qa.reviews.record": ("index:review",),
    "index.qa.reviews.list": ("index:read", "index:review", "index:coordinate"),
    "index.qa.checks.record": ("index:review",),
    "index.qa.checks.list": ("index:read", "index:review", "index:coordinate"),
    "index.qa.checks.preflight": ("index:review",),
    "index.qa.checks.secret_scan": ("index:review",),
    # Bytes: qa:read here, plus an accepted assignment enforced by the backend.
    "index.qa.package.retrieve": ("index:qa:read",),
    "index.qa.assets.retrieve": ("index:qa:read",),
}


def required_scopes(operation_id: str) -> tuple[str, ...]:
    """Read the scope class required by a supported SDK operation.

    Args:
        operation_id: Exact operation identifier in OPERATION_SCOPES.

    Returns:
        Any-of scope names for discovery guidance; backend role and assignment checks remain required.

    Raises:
        KeyError: The operation is not covered by the SDK scope table.
    """
    return OPERATION_SCOPES[operation_id]
