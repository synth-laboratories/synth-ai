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

_CONTRIBUTOR = (INTAKE, REVIEW, COORDINATE)
_REVIEWER = (REVIEW, COORDINATE)

# operation_id -> any-of scopes. Operation ids are the keys of client.OPERATIONS.
OPERATION_SCOPES: Mapping[str, tuple[str, ...]] = {
    # Contribution lifecycle owned by the contributor.
    "index.contributions.create": (INTAKE,),
    "index.contributions.revisions.create": (INTAKE,),
    "index.contributions.upload.prepare": (INTAKE,),
    "index.contributions.upload.finalize": (INTAKE,),
    "index.contributions.submit": (INTAKE,),
    # Publication is a separate authority; QA acceptance never grants it.
    "index.contributions.publication.create": (PUBLISH,),
    "index.contributions.withdrawal.create": (PUBLISH,),
    # QA: a contributor with only index:intake can create and read its own case,
    # read shared events, respond, appeal and escalate.
    "index.qa.cases.create": (INTAKE, COORDINATE),
    "index.qa.cases.get": _CONTRIBUTOR,
    "index.qa.events.list": _CONTRIBUTOR,
    "index.qa.events.create": _CONTRIBUTOR,
    "index.qa.appeals.create": _CONTRIBUTOR,
    "index.qa.escalations.create": _CONTRIBUTOR,
    "index.qa.notes.create": _REVIEWER,
    "index.qa.adjudications.create": (COORDINATE,),
    "index.qa.assignments.create": (COORDINATE,),
    "index.qa.assignments.revoke": (COORDINATE,),
    "index.qa.assignments.list": _REVIEWER,
    "index.qa.assignments.accept": _REVIEWER,
    "index.qa.reviews.record": _REVIEWER,
    "index.qa.reviews.list": _REVIEWER,
    "index.qa.checks.record": _REVIEWER,
    "index.qa.checks.list": _REVIEWER,
    "index.qa.checks.preflight": _REVIEWER,
    "index.qa.checks.secret_scan": _REVIEWER,
    # Bytes: qa:read here, plus an accepted assignment enforced by the backend.
    "index.qa.package.retrieve": (QA_READ,),
    "index.qa.assets.retrieve": (QA_READ,),
}


def required_scopes(operation_id: str) -> tuple[str, ...]:
    """Any-of scopes for an operation id; KeyError for one this table does not cover."""
    return OPERATION_SCOPES[operation_id]
