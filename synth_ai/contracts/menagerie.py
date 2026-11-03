"""Menagerie lifecycle, lineage and serving-identity routes (SYN-5183).

Backend ``app/api/v1/menagerie.py``. These authenticated routes are not
negotiated core.v1 operations: they carry their own ``synth.menagerie.v1``
schema, neither negotiate nor claim Index/Swarms release manifests, and refuse
with ``{"detail": {"error_code", "service": "backend.menagerie", "stage",
"cause"}}``. Every write is idempotent on the server (a repeat replays), so an
uncertain outcome is resolved by repeating the same call.
"""

from __future__ import annotations

from synth_ai.errors import UnsupportedOperationError
from synth_ai.options import ClientOptions

from .operations import Operation

CONTRACT = "menagerie.v1"


def _operation(name: str, method, path: str, mutation: bool) -> Operation:
    return Operation("menagerie." + name, method, path, True, mutation, CONTRACT)


EXPORT = _operation("export", "GET", "/menagerie/export", False)
RESOLVE_MODEL = _operation("models.get", "GET", "/menagerie/models/{identifier}", False)
ARCHIVE_MODEL = _operation("models.archive", "DELETE", "/menagerie/models/{identifier}", True)
RESTORE_MODEL = _operation("models.restore", "POST", "/menagerie/models/{identifier}/restore", True)
MODEL_LINEAGE = _operation("models.lineage", "GET", "/menagerie/models/{identifier}/lineage", False)
RECORD_LINEAGE = _operation("lineage.record", "POST", "/menagerie/lineage", True)
RECORD_DEPLOYMENT = _operation(
    "deployments.record", "PUT", "/menagerie/deployments/{deployment_id}", True
)
RETIRE_DEPLOYMENT = _operation(
    "deployments.retire", "DELETE", "/menagerie/deployments/{deployment_id}", True
)
ARCHIVE_RECORD = _operation(
    "records.archive", "DELETE", "/menagerie/catalog/records/{record_id}", True
)
RESTORE_RECORD = _operation(
    "records.restore", "POST", "/menagerie/catalog/records/{record_id}/restore", True
)

MENAGERIE_OPERATIONS = (
    EXPORT,
    RESOLVE_MODEL,
    ARCHIVE_MODEL,
    RESTORE_MODEL,
    MODEL_LINEAGE,
    RECORD_LINEAGE,
    RECORD_DEPLOYMENT,
    RETIRE_DEPLOYMENT,
    ARCHIVE_RECORD,
    RESTORE_RECORD,
)


def require_menagerie_operation(operation: Operation, options: ClientOptions) -> None:
    if operation not in MENAGERIE_OPERATIONS:
        raise UnsupportedOperationError("Unknown Menagerie operation")
    if options.release_manifest_sha256 is not None or options.candidate_manifest_sha256 is not None:
        raise UnsupportedOperationError(
            "Menagerie routes do not negotiate Index/Swarms release manifests; "
            "use a separate unpinned client",
            code="menagerie_release_not_qualified",
        )
