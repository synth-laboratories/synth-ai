"""Browse, open, compare and export Forge report versions as the caller (SYN-5158).

Every response is re-verified locally before it is returned:

- ``open``: the base64 bytes hash to the requested digest and re-encode to the
  returned ``ReportVersion`` (canonical JSON), in the requested project.
- ``compare``: both sides are the requested digests in the requested project.
- ``export``: the manifest digest, the explicit selection, and every file's
  bytes and parsed version agree with the provenance manifest.

Safe rendering: report text is plain data (``rendering == "plain_text"``).
Render it escaped; never interpret it as HTML or script. Reading or sharing a
version is never an Index contribution.
"""

import base64
import binascii
import hashlib

from synth_ai.contracts.forge import (
    REPORT_READER_BROWSE,
    REPORT_READER_COMPARE,
    REPORT_READER_EXPORT,
    REPORT_READER_OPEN,
)
from synth_ai.core.parameters import verified
from synth_ai.errors import ContractError

from .contracts.contracts import ForgeError, canonical_bytes, contract_digest
from .contracts.report_manifests import read_version
from .records import project_identity
from .types import (
    REPORT_EXPORT_MAX_VERSIONS,
    ReportExportBundle,
    ReportExportResponse,
    ReportVersionBrowseResponse,
    ReportVersionComparisonResponse,
    ReportVersionDocument,
    ReportVersionDocumentResponse,
    report_export_path,
)

EXPORT_MANIFEST_PATH = "manifest.json"


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("version digest must be 64 lowercase hex characters")
    return value


def _bytes(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ContractError("Report content is not valid base64") from error


def _exact(content: bytes, digest: str, version) -> bool:
    try:
        return read_version(content, digest) == version
    except ForgeError:
        return False


def verify_document(document: ReportVersionDocument, project_id: str, digest: str) -> bytes:
    """Exact version bytes of an opened document, or ``ContractError``."""
    content = _bytes(document.content_base64)
    if not (
        document.scope.project_id == project_id
        and document.version_sha256 == digest
        and document.version.manifest.scope == document.scope
        and document.size_bytes == len(content)
        and hashlib.sha256(content).hexdigest() == digest
        and _exact(content, digest, document.version)
    ):
        raise ContractError("Forge returned report bytes that differ from the requested version")
    return content


def export_files(bundle: ReportExportBundle) -> dict[str, bytes]:
    """Verified portable files: canonical ``manifest.json`` plus each version."""
    manifest = bundle.manifest
    if contract_digest(manifest) != bundle.manifest_sha256:
        raise ContractError("Report export manifest differs from its digest")
    entries = {entry.path: entry for entry in manifest.entries}
    if (
        len(entries) != len(manifest.entries)
        or [entry.version_sha256 for entry in manifest.entries] != list(manifest.selection)
        or [item.path for item in bundle.files] != [entry.path for entry in manifest.entries]
    ):
        raise ContractError("Report export files differ from the provenance manifest")
    files = {EXPORT_MANIFEST_PATH: canonical_bytes(manifest)}
    for item in bundle.files:
        entry = entries[item.path]
        content = _bytes(item.content_base64)
        try:
            version = read_version(content, entry.version_sha256)
        except ForgeError as error:
            raise ContractError("Report export bytes differ from the manifest") from error
        if not (
            item.version_sha256 == entry.version_sha256
            and item.path == report_export_path(entry.version_sha256)
            and entry.size_bytes == len(content)
            and version.lineage_id == entry.lineage_id
            and version.version == entry.version
            and version.disposition == entry.disposition
            and version.manifest.scope == manifest.scope
        ):
            raise ContractError("Report export bytes differ from the manifest")
        files[item.path] = content
    return files


def check_browse(result, project_id, after, limit):
    digests = [entry.version_sha256 for entry in result.versions]
    return verified(
        result,
        len(digests) <= limit
        and digests == sorted(set(digests))
        and (after is None or all(digest > after for digest in digests))
        and all(entry.scope.project_id == project_id for entry in result.versions),
        "Forge returned an out-of-contract report version page",
    )


def check_open(result, project_id, digest):
    verify_document(result.document, project_id, digest)
    return result


def check_compare(result, project_id, digest, other_digest):
    comparison = result.comparison
    return verified(
        result,
        comparison.scope.project_id == project_id
        and comparison.base.version_sha256 == digest
        and comparison.head.version_sha256 == other_digest,
        "Forge returned a comparison of different report versions",
    )


def check_export(result, project_id, selection):
    manifest = result.export.manifest
    verified(
        result,
        manifest.scope.project_id == project_id and manifest.selection == selection,
        "Forge exported a different report selection",
    )
    export_files(result.export)
    return result


def _browse_call(project_id, after, limit, request):
    project_id = project_identity(project_id)
    if after is not None:
        after = _digest(after)
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")
    query = {"limit": limit} if after is None else {"after": after, "limit": limit}
    return {
        "operation": REPORT_READER_BROWSE,
        "model": ReportVersionBrowseResponse,
        "path": REPORT_READER_BROWSE.path_for(project_id=project_id),
        "query": query,
        "options": request,
        "validate": lambda result: check_browse(result, project_id, after, limit),
    }


def _open_call(project_id, digest, request):
    project_id, digest = project_identity(project_id), _digest(digest)
    return {
        "operation": REPORT_READER_OPEN,
        "model": ReportVersionDocumentResponse,
        "path": REPORT_READER_OPEN.path_for(project_id=project_id, digest=digest),
        "options": request,
        "validate": lambda result: check_open(result, project_id, digest),
    }


def _compare_call(project_id, digest, other_digest, request):
    project_id = project_identity(project_id)
    digest, other_digest = _digest(digest), _digest(other_digest)
    return {
        "operation": REPORT_READER_COMPARE,
        "model": ReportVersionComparisonResponse,
        "path": REPORT_READER_COMPARE.path_for(
            project_id=project_id, digest=digest, other_digest=other_digest
        ),
        "options": request,
        "validate": lambda result: check_compare(result, project_id, digest, other_digest),
    }


def _export_call(project_id, digests, request):
    project_id = project_identity(project_id)
    if isinstance(digests, str):
        raise ValueError("digests must be a sequence of version digests")
    selection = tuple(_digest(item) for item in digests)
    if not 1 <= len(selection) <= REPORT_EXPORT_MAX_VERSIONS or len(set(selection)) != len(
        selection
    ):
        raise ValueError(
            f"export selects 1 to {REPORT_EXPORT_MAX_VERSIONS} distinct version digests"
        )
    return {
        "operation": REPORT_READER_EXPORT,
        "model": ReportExportResponse,
        "path": REPORT_READER_EXPORT.path_for(project_id=project_id),
        "query": {"digest": selection},
        "options": request,
        "validate": lambda result: check_export(result, project_id, selection),
    }


def _send(transport, call):
    call = dict(call)
    return transport.request(call.pop("operation"), call.pop("model"), **call)


class ReportReader:
    def __init__(self, transport):
        self._transport = transport

    def browse(
        self, project_id, *, after=None, limit=100, request=None
    ) -> ReportVersionBrowseResponse:
        """Report versions you own or that are shared with you, by digest."""
        return _send(self._transport, _browse_call(project_id, after, limit, request))

    def open(self, project_id, digest, *, request=None) -> ReportVersionDocumentResponse:
        """One exact version, digest-verified here as well as by the server."""
        return _send(self._transport, _open_call(project_id, digest, request))

    def compare(
        self, project_id, digest, other_digest, *, request=None
    ) -> ReportVersionComparisonResponse:
        """Structured diff from ``digest`` (base) to ``other_digest`` (head)."""
        return _send(self._transport, _compare_call(project_id, digest, other_digest, request))

    def export(self, project_id, digests, *, request=None) -> ReportExportResponse:
        """Explicit selection with a provenance manifest; see ``export_files``."""
        return _send(self._transport, _export_call(project_id, digests, request))


class AsyncReportReader:
    def __init__(self, transport):
        self._transport = transport

    async def browse(
        self, project_id, *, after=None, limit=100, request=None
    ) -> ReportVersionBrowseResponse:
        return await _send(self._transport, _browse_call(project_id, after, limit, request))

    async def open(self, project_id, digest, *, request=None) -> ReportVersionDocumentResponse:
        return await _send(self._transport, _open_call(project_id, digest, request))

    async def compare(
        self, project_id, digest, other_digest, *, request=None
    ) -> ReportVersionComparisonResponse:
        return await _send(
            self._transport, _compare_call(project_id, digest, other_digest, request)
        )

    async def export(self, project_id, digests, *, request=None) -> ReportExportResponse:
        return await _send(self._transport, _export_call(project_id, digests, request))
