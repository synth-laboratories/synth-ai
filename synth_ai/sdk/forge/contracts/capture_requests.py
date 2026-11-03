# Generated from Forge producer source; see scripts/export_capture_contracts.py.
"""Producer-owned original-intent capture lifecycle requests (internal v1)."""
from .capture import CaptureAdmissionRequest, CaptureProfile, ObservedRerun, ReproductionPackage
from .contracts import Contract, Digest, Scope
from .project_authorization import ProjectAuthorization
from .projects import ProjectPrincipal


class CaptureScopedRequest(Contract):
    scope: Scope
    principal: ProjectPrincipal
    authorization: ProjectAuthorization


class CaptureAdmitRequest(CaptureScopedRequest):
    request: CaptureAdmissionRequest
    profile: CaptureProfile
    require_owner_protection: bool = True


class CaptureGetRequest(CaptureScopedRequest):
    request_digest_sha256: Digest


class CaptureReleaseRequest(CaptureGetRequest):
    reason: str


class CaptureRerunRequest(CaptureScopedRequest):
    package: ReproductionPackage
    rerun: ObservedRerun
    admission_request_digest_sha256: Digest


class CaptureRerunRead(CaptureScopedRequest):
    package_digest_sha256: Digest

