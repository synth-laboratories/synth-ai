"""Private, resumable intake of a locally verified research-bundle conversion.

# See: specifications/tanha/synth-index/research-export-bundle.md
The backend owns research-import authorization, source policy, and final QA.
No raw session transcript is read, rights are not attested, and no publication
operation is performed here.
"""

from __future__ import annotations

import json
import os
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from .client import IndexAPI
from .contracts import ContributionAudience, ContributionOrigin, ContributionReference
from .contributions import (
    ContributionDraft,
    ContributionUploadSpec,
    ResearchDraftSpec,
    ResearchSource,
)
from .lifecycle import MeView, RevisionView
from .package import ContributionPackage
from .submission import ContributionSubmitSpec, RevisionStatus
from .transfer import TransferTargetsExpiredError, upload_directory_sync, verify_package_directory

_RECEIPT_SCHEMA = "synth.index.research-bundle-conversion.v1"
_SOURCE_SCHEMA = "synth.index.research-source.v1"
# v2 binds saved state to its backend and verified account; v1 state named
# neither and cannot be resumed safely.
_STATE_SCHEMA = "synth.index.research-intake-state.v2"
_DENIED = frozenset(
    {
        "reference",
        "references",
        "evaluator",
        "evaluators",
        "oracle",
        "oracles",
        "private-eval",
        "private_eval",
        "gold",
        "gold_answers",
        "gold-answers",
        "gold_solution",
        "gold-solution",
    }
)
_REB_TASK = re.compile(r"^(?:\d{3}(?:[-_].*)?|reb[-_]?\d{3}(?:[-_].*)?)$", re.I)


def _preflight_source_path(path: str) -> None:
    if not path or path.startswith(("/", "~")) or "\\" in path or "\x00" in path:
        raise ValueError(f"Unsafe research source path: {path}")
    parts = path.split("/")
    lowered = [part.lower() for part in parts]
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe research source path: {path}")
    if any(part in _DENIED or part.startswith("gold_") for part in lowered):
        raise ValueError(f"Sealed research source path is forbidden: {path}")
    if "artifacts" in lowered and any(part.startswith("oracle") for part in lowered):
        raise ValueError(f"Sealed research source path is forbidden: {path}")
    if (
        "tasks" in lowered
        and any(_REB_TASK.fullmatch(part) for part in parts)
        and any(part in {"solution", "answers", "answer_key", "answer-key"} for part in lowered)
    ):
        raise ValueError(f"REB answer source path is forbidden: {path}")


def preview_conversion(directory: Path) -> tuple[ContributionPackage, ResearchDraftSpec, dict]:
    """Verify exact converted files and return a reviewable private intake summary."""
    root = directory.resolve(strict=True)
    receipt_path = root / "receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Research conversion receipt is missing or linked")
    receipt = json.loads(receipt_path.read_bytes())
    if (
        receipt.get("schema_version") != _RECEIPT_SCHEMA
        or receipt.get("visibility") != "private"
        or receipt.get("publication") != "not_granted"
        or receipt.get("provider_calls") != 0
    ):
        raise ValueError("Research conversion is not an offline private draft")
    package_root = root / "package"
    if package_root.is_symlink():
        raise ValueError("Research package root may not be a link")
    package = verify_package_directory(package_root)
    descriptor = (package_root / "contribution.json").read_bytes()
    if receipt.get("descriptor_digest") != f"sha256:{sha256(descriptor).hexdigest()}":
        raise ValueError("Research conversion descriptor receipt differs from package")
    if receipt.get("reference") != {
        "contribution_id": package.contribution_id,
        "revision_id": package.revision_id,
    }:
        raise ValueError("Research conversion reference differs from package")
    if (
        package.provenance.origin != ContributionOrigin.SYNTH
        or package.requested_audience != ContributionAudience.PRIVATE
        or package.rights_attested
        or package.parent_revision_id is not None
    ):
        raise ValueError(
            "Research conversion must preserve private SYNTH provenance without rights attestation"
        )
    source_asset = next(
        (
            asset
            for asset in package.assets
            if asset.object.logical_path == "evidence/research-source.json"
        ),
        None,
    )
    if source_asset is None:
        raise ValueError("Research conversion lacks its source evidence asset")
    by_id = {asset.asset_id: asset.object.logical_path for asset in package.assets}
    session_assets = [path for path in by_id.values() if path.startswith("session/")]
    if session_assets != ["session/decision-log.md"]:
        raise ValueError(
            "Research conversion may contain only a redacted decision log under session/"
        )
    decision_log = (package_root / "session/decision-log.md").read_bytes()
    if not decision_log.startswith(b"# Redacted research decision log\n"):
        raise ValueError("Research decision log lacks its redacted marker")
    source_record = json.loads((package_root / "evidence/research-source.json").read_bytes())
    if source_record.get("schema_version") != _SOURCE_SCHEMA:
        raise ValueError("Research source evidence has an unknown schema")
    bundle_digest = source_record.get("bundle_digest")
    if bundle_digest != receipt.get("bundle_digest"):
        raise ValueError("Research source evidence differs from conversion receipt")
    source = ResearchSource.model_validate(source_record.get("source"))
    if (
        by_id.get(source_record.get("run_ledger_asset_id")) != "runs/ledger.json"
        or by_id.get(source_record.get("decision_log_asset_id")) != "session/decision-log.md"
    ):
        raise ValueError("Research source evidence does not bind its ledger and redacted log")
    mapping = json.dumps(
        ["research", source.organization_id, source.project_id, source.arc_id],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if receipt.get("stable_source_mapping") != mapping:
        raise ValueError("Research conversion source identity differs from receipt")
    for path in source.source_paths:
        _preflight_source_path(path)
    spec = ResearchDraftSpec(bundle_digest=bundle_digest, source=source)
    summary = {
        "title": package.title,
        "bundle_digest": spec.bundle_digest,
        "source_repository_url": source.source_repository_url,
        "source_revision": source.source_revision,
        "arc_id": source.arc_id,
        "asset_count": len(package.assets),
        "evidence_count": len(package.evidence),
        "claim_count": len(package.claims),
        "rights_attested": False,
        "requested_audience": "private",
        "qualification": "pending",
        "source_path_audit": "preflight passed; backend and reviewer checks required",
    }
    return package, spec, summary


class IntakeStateError(RuntimeError):
    """Saved intake state cannot be trusted; recovery is a deliberate decision."""


class IntakeLockedError(RuntimeError):
    """Another intake process holds this state file."""


class TerminalRevisionError(RuntimeError):
    """The server has decided this revision; resubmitting it would be wrong."""


# Statuses the server can reach after a submission that mean the submitted work
# is still the same work. Reaching one of these is success for a resumed intake,
# not a reason to submit again.
_ADVANCED = frozenset(
    {RevisionStatus.SUBMITTED, RevisionStatus.QUALIFIED, RevisionStatus.PUBLISHED}
)
# Statuses that end this revision. A caller must decide what to do next; the
# intake never silently starts over.
_TERMINAL = frozenset(
    {
        RevisionStatus.REJECTED,
        RevisionStatus.WITHDRAWN,
        RevisionStatus.CHANGES_REQUESTED,
    }
)


def _backend_identity(api: IndexAPI) -> str:
    """The canonical backend this state belongs to, without any credential."""
    url = urlsplit(getattr(api._transport, "base_url", "").rstrip("/"))
    if not url.scheme or not url.hostname:
        raise IntakeStateError("Index client has no usable backend base URL")
    port = f":{url.port}" if url.port else ""
    return f"{url.scheme}://{url.hostname.lower()}{port}{url.path.rstrip('/')}"


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive sidecar lock for the whole intake.

    Atomic replacement keeps a state file readable, but it does not stop two
    intakes from allocating, uploading and submitting against the same saved
    identity at once. The lock names its holder so a stale one can be cleared
    deliberately rather than guessed at.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    holder = json.dumps({"pid": os.getpid(), "host": socket.gethostname()}, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        try:
            current = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            current = "unknown"
        raise IntakeLockedError(
            f"Another research intake holds {lock_path} ({current}). Wait for it to "
            "finish, or remove that file if the process is gone."
        ) from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(holder + "\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _new_state(bundle_digest: str, backend: str, account: MeView) -> dict:
    return {
        "schema_version": _STATE_SCHEMA,
        "bundle_digest": bundle_digest,
        "backend": backend,
        "org_id": account.org_id,
        "principal_id": account.principal_id,
        "draft_key": uuid4().hex,
        "publication_id": str(uuid4()),
    }


def _state(path: Path, bundle_digest: str, backend: str, account: MeView) -> dict:
    """Load state bound to this bundle, backend and verified account, or start one.

    Saved state names the exact identities it was created under. Resuming against
    a different backend, organization or account is refused rather than replayed:
    an allocated draft and its idempotency keys mean nothing there, and reusing
    them could attach this work to the wrong owner.
    """
    if not path.exists():
        state = _new_state(bundle_digest, backend, account)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
        return state
    if path.is_symlink():
        raise IntakeStateError("Intake state may not be a link")
    try:
        state = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise IntakeStateError(
            f"Intake state at {path} is unreadable or corrupt. Inspect it, then "
            "delete it to start a fresh allocation."
        ) from error
    if not isinstance(state, dict) or state.get("schema_version") != _STATE_SCHEMA:
        raise IntakeStateError(
            f"Intake state at {path} has schema {state.get('schema_version') if isinstance(state, dict) else 'unknown'!r}, "
            f"not {_STATE_SCHEMA!r}. Delete it to start a fresh allocation."
        )
    if state.get("bundle_digest") != bundle_digest:
        raise IntakeStateError("Intake state belongs to a different research bundle")
    if state.get("backend") != backend:
        raise IntakeStateError(
            f"Intake state was allocated against {state.get('backend')!r}, not {backend!r}"
        )
    if (state.get("org_id"), state.get("principal_id")) != (
        account.org_id,
        account.principal_id,
    ):
        raise IntakeStateError("Intake state belongs to a different account or organization")
    try:
        UUID(state["publication_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntakeStateError("Intake state has no usable publication ID") from error
    if not isinstance(state.get("draft_key"), str) or not re.fullmatch(
        r"[a-zA-Z0-9_.-]{1,128}", state["draft_key"]
    ):
        raise IntakeStateError("Intake state has an invalid draft key")
    if state.get("reference") is not None:
        ContributionReference.model_validate(state["reference"])
    if state.get("manifest_digest") is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(state["manifest_digest"])
    ):
        raise IntakeStateError("Intake state has an invalid manifest digest")
    return state


def _save_state(path: Path, state: dict) -> None:
    """Replace the state file atomically, under the caller's exclusive lock.

    Replacement keeps a reader from ever seeing a half-written file; it is the
    lock in ``_state_lock`` that keeps a second intake from writing at all.
    """
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(state, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _rebind(package: ContributionPackage, reference: ContributionReference) -> ContributionPackage:
    payload = package.model_dump(mode="json")
    payload["contribution_id"] = reference.contribution_id
    payload["revision_id"] = reference.revision_id
    return ContributionPackage.model_validate(payload)


def _reconcile(api: IndexAPI, state: dict) -> RevisionView | None:
    """Ask the server what actually happened, when a mutation's answer was lost.

    Any allocation, finalize or submit call can fail after the server committed
    it. The saved identities are stable across retries, so the authoritative
    answer is always one read away; guessing from a local flag is not allowed.
    """
    if state.get("reference") is None:
        return None
    reference = ContributionReference.model_validate(state["reference"])
    return api.contributions.revisions.retrieve(reference)


def _server_status(view: RevisionView, state: dict) -> RevisionStatus:
    """Read one revision's status, refusing a manifest that is not ours."""
    if (
        view.manifest_digest is not None
        and state.get("manifest_digest") is not None
        and view.manifest_digest != state["manifest_digest"]
    ):
        raise IntakeStateError(
            "Server revision carries a different manifest than this intake prepared"
        )
    return view.status


def _result(summary: dict, state: dict, status: str) -> dict:
    return {
        **summary,
        "reference": state.get("reference"),
        "manifest_digest": state.get("manifest_digest"),
        "status": status,
    }


def _allocate(api: IndexAPI, spec: ResearchDraftSpec, state: dict, state_path: Path):
    """Allocate the draft, or recover the one a lost response already created."""
    draft = api.contributions.create_research(spec, idempotency_key=state["draft_key"])
    allocated = draft.reference.model_dump(mode="json")
    if state.get("reference") not in (None, allocated):
        raise IntakeStateError("Research draft replay returned a different server identity")
    if state.get("reference") != allocated:
        state["reference"] = allocated
        _save_state(state_path, state)
    return draft


def _upload_and_finalize(
    api: IndexAPI,
    draft: ContributionDraft,
    package: ContributionPackage,
    directory: Path,
    state: dict,
    state_path: Path,
) -> None:
    """Transfer and finalize, preparing again if storage refused a stale target.

    Preparing again is safe and does not discard completed work: the server
    omits objects it already holds under this publication ID, so the second
    attempt transfers only what is still missing.
    """
    rebound = _rebind(package, draft.reference)
    upload_spec = ContributionUploadSpec(
        publication_id=UUID(state["publication_id"]), package=rebound
    )
    prepared = api.contributions.prepare_upload(draft, upload_spec)
    try:
        upload_directory_sync(prepared, directory / "package")
    except TransferTargetsExpiredError:
        prepared = api.contributions.prepare_upload(draft, upload_spec)
        upload_directory_sync(prepared, directory / "package")
    api.contributions.finalize(draft, prepared)
    state["status"] = "finalized"
    state["manifest_digest"] = prepared.transfer.manifest_digest
    _save_state(state_path, state)


def submit_conversion(
    api: IndexAPI, directory: Path, state_path: Path, *, finalize_only: bool = False
) -> dict:
    """Resume private allocation/upload; submit only when review gates allow.

    Re-running reuses immutable IDs and obtains fresh storage targets. Every
    resumption reconciles against the server before acting, so a response lost
    after the server committed it is recovered instead of repeated. A revision
    the server has already advanced past submission is reported as it stands; a
    revision the server has decided against raises ``TerminalRevisionError`` rather
    than being quietly submitted again. One process at a time holds the state
    file.
    """
    package, spec, summary = preview_conversion(directory)
    backend = _backend_identity(api)
    account = api.account.retrieve()
    with _state_lock(state_path):
        state = _state(state_path, spec.bundle_digest, backend, account)
        view = _reconcile(api, state)
        if view is not None:
            status = _server_status(view, state)
            if status in _TERMINAL:
                raise TerminalRevisionError(
                    f"Server revision is {status.value}; decide explicitly before "
                    "preparing another revision"
                )
            if status in _ADVANCED:
                # Submitted, qualified or published: the same work moved on.
                state["status"] = "submitted"
                state["manifest_digest"] = view.manifest_digest or state.get("manifest_digest")
                _save_state(state_path, state)
                return _result(summary, state, status.value)
            if status is RevisionStatus.DRAFT and view.manifest_digest is not None:
                # Finalize landed even though its answer did not reach us.
                state["status"] = "finalized"
                state["manifest_digest"] = view.manifest_digest
                _save_state(state_path, state)
        draft = _allocate(api, spec, state, state_path)
        if state.get("status") != "finalized":
            _upload_and_finalize(api, draft, package, directory, state, state_path)
        if finalize_only:
            return _result(summary, state, "finalized_private_draft")
        submitted = api.contributions.submit(
            draft.reference,
            ContributionSubmitSpec(publication_id=UUID(state["publication_id"])),
        )
        if (
            submitted.status not in _ADVANCED
            or submitted.manifest_digest != state["manifest_digest"]
        ):
            raise IntakeStateError("Submitted revision differs from the prepared manifest")
        state["status"] = "submitted"
        state["manifest_digest"] = submitted.manifest_digest
        _save_state(state_path, state)
        return _result(summary, state, submitted.status.value)
