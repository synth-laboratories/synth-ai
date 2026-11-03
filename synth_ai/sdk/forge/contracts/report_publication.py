# Generated from Forge packages/forge/report_publication.py (Forge dev 6d47a173).
# fmt: off
"""Internal-port requests for publishing report versions as the acting user.

SYN-5155/5157. The backend (public API/MCP owner) signs these with the project
authorization proof, exactly like workspace science writes. The acting user's
own backend API key travels only in the ``x-forge-acting-credential`` header,
never in the signed body, receipts or logs; Forge uses it for evidence custody
and share calls so Artifact Platform grants stay the user's own.
"""

from __future__ import annotations

from typing import Literal

from .contracts import Contract, Identifier, Producer
from .project_authorization import ProjectAuthorization
from .projects import ProjectId, ProjectPrincipal
from .report_manifests import ReportVersion

ACTING_CREDENTIAL_HEADER = "x-forge-acting-credential"


class ReportVersionPublish(Contract):
    schema_version: Literal["forge.report-publication.v1"] = (
        "forge.report-publication.v1"
    )
    principal: ProjectPrincipal
    project_id: ProjectId
    producer: Producer
    version: ReportVersion
    authorization: ProjectAuthorization


class ReportShareRevocationRetry(Contract):
    schema_version: Literal["forge.report-publication.v1"] = (
        "forge.report-publication.v1"
    )
    principal: ProjectPrincipal
    project_id: ProjectId
    lineage_id: Identifier
    authorization: ProjectAuthorization
