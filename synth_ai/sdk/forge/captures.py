"""Pre-admit capture custody and read or release its retained owner state."""

from pydantic import TypeAdapter

from synth_ai.contracts import forge as operations
from synth_ai.core.parameters import verified

from .capture_types import (
    CaptureAdmitIntent, CaptureAdmittedResponse, CaptureReleaseIntent,
    CaptureRetainedListResponse, CaptureRetainedResponse,
)
from .contracts.capture import CaptureAdmissionRequest, CaptureProfile
from .contracts.contracts import Digest, ExactReference, contract_digest
from .records import project_identity
from .reports import _selected


def intent_reference(request):
    digest = contract_digest(request)
    return ExactReference(authority="artifact-platform", kind="capture_admission_request",
                          record_id=digest, revision="1", digest_sha256=digest)


def admit_intent(request, profile, require_owner_protection, mutation):
    key, selected = _selected(mutation)
    request = CaptureAdmissionRequest.model_validate(request)
    profile = CaptureProfile.model_validate(profile)
    return CaptureAdmitIntent(operation_id=key, request=request, profile=profile,
                              require_owner_protection=require_owner_protection), selected


def release_intent(digest, reason, mutation):
    key, selected = _selected(mutation)
    return CaptureReleaseIntent(operation_id=key, request_digest_sha256=digest,
                                reason=reason), selected


def _admit_call(project_id, request, profile, require_owner_protection, mutation):
    project_id = project_identity(project_id)
    intent, selected = admit_intent(request, profile, require_owner_protection, mutation)
    if intent.request.scope.project_id != project_id:
        raise ValueError("capture request belongs to a different project")
    return dict(operation=operations.CAPTURE_ADMIT, model=CaptureAdmittedResponse,
        path=operations.CAPTURE_ADMIT.path_for(project_id=project_id), body=intent,
        options=selected, validate=lambda result: verified(result,
            result.admission.request_digest_sha256 == contract_digest(intent.request)
            and result.intent_reference == intent_reference(intent.request),
            "Forge returned a different capture admission"))


def _checked_retained(result, project_id, digest):
    value = result.capture
    return verified(result,
        value.request.scope.project_id == project_id
        and contract_digest(value.request) == digest
        and value.admission.request_digest_sha256 == digest
        and result.intent_reference == intent_reference(value.request),
        "Forge returned a different retained capture")


def _get_call(project_id, digest, request):
    project_id = project_identity(project_id)
    digest = TypeAdapter(Digest).validate_python(digest)
    return dict(operation=operations.CAPTURE_GET, model=CaptureRetainedResponse,
        path=operations.CAPTURE_GET.path_for(project_id=project_id, digest=digest),
        options=request, validate=lambda result: _checked_retained(result, project_id, digest))


def _list_call(project_id, request):
    project_id = project_identity(project_id)

    def validate(result):
        digests = [contract_digest(item.request) for item in result.captures]
        return verified(result, len(digests) == len(set(digests)) and all(
            item.request.scope.project_id == project_id
            and item.admission.request_digest_sha256 == contract_digest(item.request)
            for item in result.captures), "Forge returned an invalid retained capture list")

    return dict(operation=operations.CAPTURE_RETAINED, model=CaptureRetainedListResponse,
        path=operations.CAPTURE_RETAINED.path_for(project_id=project_id),
        options=request, validate=validate)


def _release_call(project_id, digest, reason, mutation):
    project_id = project_identity(project_id)
    digest = TypeAdapter(Digest).validate_python(digest)
    intent, selected = release_intent(digest, reason, mutation)
    return dict(operation=operations.CAPTURE_RELEASE, model=CaptureRetainedResponse,
        path=operations.CAPTURE_RELEASE.path_for(project_id=project_id, digest=digest),
        body=intent, options=selected,
        validate=lambda result: _checked_retained(result, project_id, digest))


def _send(transport, call):
    return transport.request(call.pop("operation"), call.pop("model"), **call)


class Captures:
    def __init__(self, transport):
        self._transport = transport

    def admit(self, project_id, request, profile, *, require_owner_protection=True,
              mutation=None) -> CaptureAdmittedResponse:
        return _send(self._transport, _admit_call(project_id, request, profile,
                                                require_owner_protection, mutation))

    def get(self, project_id, digest, *, request=None) -> CaptureRetainedResponse:
        return _send(self._transport, _get_call(project_id, digest, request))

    def retained(self, project_id, *, request=None) -> CaptureRetainedListResponse:
        return _send(self._transport, _list_call(project_id, request))

    def release(self, project_id, digest, reason, *, mutation=None) -> CaptureRetainedResponse:
        return _send(self._transport, _release_call(project_id, digest, reason, mutation))


class AsyncCaptures:
    def __init__(self, transport):
        self._transport = transport

    async def admit(self, project_id, request, profile, *, require_owner_protection=True,
                    mutation=None) -> CaptureAdmittedResponse:
        return await _send(self._transport, _admit_call(project_id, request, profile,
                                                      require_owner_protection, mutation))

    async def get(self, project_id, digest, *, request=None) -> CaptureRetainedResponse:
        return await _send(self._transport, _get_call(project_id, digest, request))

    async def retained(self, project_id, *, request=None) -> CaptureRetainedListResponse:
        return await _send(self._transport, _list_call(project_id, request))

    async def release(self, project_id, digest, reason, *, mutation=None) -> CaptureRetainedResponse:
        return await _send(self._transport, _release_call(project_id, digest, reason, mutation))
