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

from synth_ai.core.errors import SynthError
from synth_ai.core.utils.urls import normalize_backend_base

from .client import IndexAPI
from .contracts import ContributionAudience, ContributionOrigin, ContributionReference
from .contributions import (
    ContributionDraft,
    ContributionUploadSpec,
    ResearchDraftSpec,
    ResearchLookupView,
    ResearchSource,
)
from .errors import IndexErrorCode, index_error_code
from .lifecycle import MeView, RevisionView
from .package import ContributionPackage
from .submission import ContributionSubmitSpec, RevisionStatus
from .transfer import TransferTargetsExpired, upload_directory_sync, verify_package_directory

_RECEIPT_SCHEMA = "synth.index.research-bundle-conversion.v1"
_SOURCE_SCHEMA = "synth.index.research-source.v1"
# v2 binds saved state to its backend and verified account. v1 state named
# neither; ``recover_v1_state`` converts it by asking the server, never by
# allocating again.
_STATE_SCHEMA = "synth.index.research-intake-state.v2"
_STATE_SCHEMA_V1 = "synth.index.research-intake-state.v1"
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


class IntakeLocked(RuntimeError):
    """Another intake process holds this state file."""


class TerminalRevision(RuntimeError):
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
_LOCAL_STATUSES = frozenset({None, "finalized", "submitted"})
# A signed storage target can expire between prepare and transfer. Each refresh
# transfers only what storage still lacks, so a few are enough for a slow link.
_MAX_TARGET_REFRESHES = 3


def _canonical_backend(base_url: str) -> str:
    url = urlsplit(base_url.rstrip("/"))
    if not url.scheme or not url.hostname:
        raise IntakeStateError("Index client has no usable backend base URL")
    port = f":{url.port}" if url.port else ""
    return f"{url.scheme}://{url.hostname.lower()}{port}{url.path.rstrip('/')}"


def _backend_identity(api: IndexAPI) -> str:
    """The canonical backend this state belongs to, without any credential."""
    return _canonical_backend(getattr(api._transport, "base_url", ""))


def _try_lock(descriptor: int) -> bool:
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive OS lock on a sidecar file for the whole intake.

    Atomic replacement keeps a state file readable, but it does not stop two
    intakes from allocating, uploading and submitting against the same saved
    identity at once. The lock is an operating-system lock, so a crashed
    process releases it with no manual cleanup; the sidecar file itself stays
    in place (removing it would let two processes lock different files) and
    names the current holder for diagnostics.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
    except OSError as error:
        raise IntakeStateError(f"Cannot open intake lock {lock_path}: {error}") from error
    try:
        if not _try_lock(descriptor):
            try:
                current = lock_path.read_text(encoding="utf-8").strip() or "unknown"
            except OSError:
                current = "unknown"
            raise IntakeLocked(
                f"Another research intake is running against {path} (holder {current}). "
                "Wait for it to finish; the lock is released when that process exits."
            )
        holder = json.dumps({"pid": os.getpid(), "host": socket.gethostname()}, sort_keys=True)
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, (holder + "\n").encode(), 0)
        try:
            yield
        finally:
            os.ftruncate(descriptor, 0)
    finally:
        os.close(descriptor)


def _write_state(path: Path, state: dict) -> None:
    """Replace the state file atomically and durably, under the caller's lock.

    A reader never sees a half-written file, and a crash after ``os.replace``
    cannot roll the directory entry back to the previous state.
    """
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


_save_state = _write_state


def _read_state(path: Path) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise IntakeStateError(f"Intake state {path} must be a regular file, not a link")
    try:
        state = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise IntakeStateError(
            f"Intake state at {path} is unreadable or corrupt. Do not start over "
            "blindly: if an earlier run allocated a draft, a fresh state file "
            "would allocate a second one. Restore the file from a backup, or move "
            "it aside and start fresh only if no run ever reached the server."
        ) from error
    if not isinstance(state, dict):
        raise IntakeStateError(f"Intake state at {path} is not a JSON object")
    return state


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


def _check_identities(state: dict, path: Path) -> None:
    """Shape checks shared by v1 and v2 state."""
    try:
        UUID(state["publication_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise IntakeStateError(f"Intake state {path} has no usable publication ID") from error
    if not isinstance(state.get("draft_key"), str) or not re.fullmatch(
        r"[a-zA-Z0-9_.-]{1,128}", state["draft_key"]
    ):
        raise IntakeStateError(f"Intake state {path} has an invalid draft key")
    if state.get("reference") is not None:
        try:
            ContributionReference.model_validate(state["reference"])
        except ValueError as error:
            raise IntakeStateError(f"Intake state {path} has an invalid reference") from error
    if state.get("manifest_digest") is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(state["manifest_digest"])
    ):
        raise IntakeStateError(f"Intake state {path} has an invalid manifest digest")
    if state.get("status") not in _LOCAL_STATUSES:
        raise IntakeStateError(f"Intake state {path} has an unknown status")


def _state(path: Path, bundle_digest: str, backend: str, account: MeView) -> dict:
    """Load state bound to this bundle, backend and verified account, or start one.

    Saved state names the exact identities it was created under. Resuming against
    a different backend, organization or account is refused rather than replayed:
    an allocated draft and its idempotency keys mean nothing there, and reusing
    them could attach this work to the wrong owner.
    """
    state = _read_state(path)
    if state is None:
        state = _new_state(bundle_digest, backend, account)
        _write_state(path, state)
        return state
    schema = state.get("schema_version")
    if schema == _STATE_SCHEMA_V1:
        raise IntakeStateError(
            f"Intake state at {path} was written by an older synth-ai "
            f"({_STATE_SCHEMA_V1!r}) and names no backend or account. Do not delete "
            "it: its draft key is how the server finds work already allocated. Run "
            "`synth-ai index research recover-state` with the backend it was used "
            "against to convert it without creating a duplicate submission."
        )
    if schema != _STATE_SCHEMA:
        raise IntakeStateError(
            f"Intake state at {path} has unknown schema {schema!r}, not "
            f"{_STATE_SCHEMA!r}; this synth-ai cannot resume it."
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
    _check_identities(state, path)
    if state.get("collection_id") is not None:
        try:
            UUID(str(state["collection_id"]))
        except ValueError as error:
            raise IntakeStateError(f"Intake state {path} has an invalid collection") from error
    if not isinstance(state.get("allocation_attempted", False), bool):
        raise IntakeStateError(f"Intake state {path} has an invalid allocation flag")
    if state.get("status") is not None and state.get("reference") is None:
        raise IntakeStateError(f"Intake state {path} records progress without a draft")
    return state


def _rebind(package: ContributionPackage, reference: ContributionReference) -> ContributionPackage:
    payload = package.model_dump(mode="json")
    payload["contribution_id"] = reference.contribution_id
    payload["revision_id"] = reference.revision_id
    return ContributionPackage.model_validate(payload)


def _lookup(api: IndexAPI, spec: ResearchDraftSpec, draft_key: str) -> ResearchLookupView | None:
    """The server's receipt for this allocation key, or None if nothing was allocated."""
    try:
        return api.contributions.lookup_research(spec, idempotency_key=draft_key)
    except SynthError as error:
        code = index_error_code(error)
        if code is IndexErrorCode.RESEARCH_RECEIPT_ABSENT:
            return None
        if code is IndexErrorCode.RESEARCH_RECEIPT_PENDING:
            raise IntakeStateError(
                "The server is still committing an allocation under this state's key; "
                "retry in a moment"
            ) from error
        if code in (
            IndexErrorCode.RESEARCH_RECEIPT_CONFLICT,
            IndexErrorCode.RESEARCH_REGISTRATION_CONFLICT,
        ):
            raise IntakeStateError(
                "This state's draft key was used on this backend for different research "
                "input; it cannot be resumed for this conversion"
            ) from error
        if code in (IndexErrorCode.RESEARCH_LOOKUP_FORBIDDEN, IndexErrorCode.FORBIDDEN):
            raise IntakeStateError(
                "This account cannot read the allocation this state names; the state "
                "belongs to another account or organization"
            ) from error
        raise


def _bind(state: dict, draft: ContributionDraft, path: Path) -> None:
    allocated = draft.reference.model_dump(mode="json")
    if state.get("reference") not in (None, allocated):
        raise IntakeStateError("Server allocation differs from the saved draft identity")
    state["reference"] = allocated
    state["collection_id"] = str(draft.collection_id)
    _write_state(path, state)


def _ask_once(original: Exception, read):
    """Check whether a failed mutation committed; if the check fails, keep the first error.

    Intake decisions (a terminal revision, a conflicting receipt) still surface,
    because they say more than the transport failure that prompted the check.
    """
    try:
        return read()
    except (IntakeStateError, TerminalRevision):
        raise
    except Exception:
        raise original from None


def _allocate(api: IndexAPI, spec: ResearchDraftSpec, state: dict, path: Path) -> None:
    """Allocate the draft once, or find the one a lost response already created.

    The key is recorded as used before the request is sent. A later run that
    finds that mark asks the server what the key produced (a read) instead of
    sending the allocation again.
    """
    if state.get("allocation_attempted"):
        found = _lookup(api, spec, state["draft_key"])
        if found is not None:
            _bind(state, found.draft, path)
            return
    else:
        state["allocation_attempted"] = True
        _write_state(path, state)
    try:
        draft = api.contributions.create_research(spec, idempotency_key=state["draft_key"])
    except Exception as error:
        # The allocation may have committed before the failure. Ask once.
        found = _ask_once(error, lambda: _lookup(api, spec, state["draft_key"]))
        if found is None:
            raise
        draft = found.draft
    _bind(state, draft, path)


def _draft(api: IndexAPI, spec: ResearchDraftSpec, state: dict, path: Path) -> ContributionDraft:
    reference = ContributionReference.model_validate(state["reference"])
    if state.get("collection_id") is None:
        found = _lookup(api, spec, state["draft_key"])
        if found is None:
            raise IntakeStateError(
                "The server has no allocation for this state's key; it was not "
                "created on this backend by this account"
            )
        _bind(state, found.draft, path)
    return ContributionDraft(reference=reference, collection_id=UUID(state["collection_id"]))


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


def _view(api: IndexAPI, state: dict) -> RevisionView:
    reference = ContributionReference.model_validate(state["reference"])
    return api.contributions.revisions.retrieve(reference)


def _upload_and_finalize(
    api: IndexAPI,
    draft: ContributionDraft,
    package: ContributionPackage,
    directory: Path,
    state: dict,
    state_path: Path,
) -> None:
    """Transfer and finalize, preparing again whenever storage refused a stale target.

    Preparing again is safe and does not discard completed work: the server
    omits objects it already holds under this publication ID, so each refresh
    transfers only what is still missing.
    """
    rebound = _rebind(package, draft.reference)
    upload_spec = ContributionUploadSpec(
        publication_id=UUID(state["publication_id"]), package=rebound
    )
    for attempt in range(_MAX_TARGET_REFRESHES + 1):
        prepared = api.contributions.prepare_upload(draft, upload_spec)
        try:
            upload_directory_sync(prepared, directory / "package")
            break
        except TransferTargetsExpired:
            if attempt == _MAX_TARGET_REFRESHES:
                raise
    try:
        api.contributions.finalize(draft, prepared)
    except Exception as error:
        # Finalize may have committed before the failure. Ask once.
        view = _ask_once(error, lambda: _view(api, state))
        if view.manifest_digest != prepared.transfer.manifest_digest:
            raise
    state["status"] = "finalized"
    state["manifest_digest"] = prepared.transfer.manifest_digest
    _write_state(state_path, state)


def _reconciled(api: IndexAPI, state: dict, path: Path, summary: dict) -> dict | None:
    """Apply what the server says about the saved revision; a result ends the run."""
    view = _view(api, state)
    status = _server_status(view, state)
    if status in _TERMINAL:
        raise TerminalRevision(
            f"Server revision is {status.value}; decide explicitly before "
            "preparing another revision"
        )
    if status in _ADVANCED:
        # Submitted, qualified or published: the same work moved on.
        state["status"] = "submitted"
        state["manifest_digest"] = view.manifest_digest or state.get("manifest_digest")
        _write_state(path, state)
        return _result(summary, state, status.value)
    if view.manifest_digest is not None and state.get("status") != "finalized":
        # Finalize landed even though its answer did not reach us.
        state["status"] = "finalized"
        state["manifest_digest"] = view.manifest_digest
        _write_state(path, state)
    return None


def submit_conversion(
    api: IndexAPI, directory: Path, state_path: Path, *, finalize_only: bool = False
) -> dict:
    """Resume private allocation/upload; submit only when review gates allow.

    Re-running reuses immutable IDs and obtains fresh storage targets. Every
    resumption reconciles against the server before acting, so a response lost
    after the server committed it is recovered instead of repeated: allocation
    is found through the non-mutating research lookup, finalize and submit
    through the revision read. A revision the server has already advanced past
    submission is reported as it stands; a revision the server has decided
    against raises ``TerminalRevision`` rather than being quietly submitted
    again. One process at a time holds the state file.
    """
    package, spec, summary = preview_conversion(directory)
    backend = _backend_identity(api)
    account = api.account.retrieve()
    with _state_lock(state_path):
        state = _state(state_path, spec.bundle_digest, backend, account)
        if state.get("reference") is None:
            _allocate(api, spec, state, state_path)
        finished = _reconciled(api, state, state_path, summary)
        if finished is not None:
            return finished
        draft = _draft(api, spec, state, state_path)
        if state.get("status") != "finalized":
            _upload_and_finalize(api, draft, package, directory, state, state_path)
        if finalize_only:
            return _result(summary, state, "finalized_private_draft")
        try:
            submitted = api.contributions.submit(
                draft.reference,
                ContributionSubmitSpec(publication_id=UUID(state["publication_id"])),
            )
        except Exception as error:
            # Submit may have committed before the failure. Ask once.
            finished = _ask_once(error, lambda: _reconciled(api, state, state_path, summary))
            if finished is None:
                raise
            return finished
        if (
            submitted.status not in _ADVANCED
            or submitted.manifest_digest != state["manifest_digest"]
        ):
            raise IntakeStateError("Submitted revision differs from the prepared manifest")
        state["status"] = "submitted"
        state["manifest_digest"] = submitted.manifest_digest
        _write_state(state_path, state)
        return _result(summary, state, submitted.status.value)


def _backup(path: Path, content: bytes) -> Path:
    """Keep the original bytes next to the state; never overwrite a backup."""
    for index in range(1000):
        suffix = ".v1-backup" if index == 0 else f".v1-backup.{index}"
        backup = path.with_name(f"{path.name}{suffix}")
        try:
            descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if backup.read_bytes() == content:
                return backup
            continue
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return backup
    raise IntakeStateError(f"Too many state backups beside {path}")


def recover_v1_state(api: IndexAPI, directory: Path, state_path: Path, *, backend_url: str) -> dict:
    """Convert v1 intake state to v2 by asking the server what its key produced.

    v1 state recorded a draft key and publication ID but not the backend or
    account it was used with. Recovery never allocates, uploads, finalizes or
    submits. It reads the server's receipt for the saved key (the research
    lookup) under the current account, and then:

    - if the server allocated a draft under the key, v2 state is bound to that
      draft and its server status, so the next ``submit_conversion`` resumes it;
    - if nothing was allocated, v2 state keeps the same key and publication ID,
      so a later allocation is still the first one under that key;
    - if v1 state says work reached the server but this backend and account
      have no record of it, recovery refuses: the state belongs elsewhere.

    ``backend_url`` must name the client's backend explicitly; v1 state cannot
    say where it was used, so the caller has to. The original file is kept
    beside the new state as ``<name>.v1-backup``.
    """
    _, spec, _ = preview_conversion(directory)
    backend = _backend_identity(api)
    if _canonical_backend(normalize_backend_base(backend_url)) != backend:
        raise IntakeStateError(
            f"Recovery backend {backend_url!r} is not the client's backend {backend!r}"
        )
    account = api.account.retrieve()
    with _state_lock(state_path):
        legacy = _read_state(state_path)
        if legacy is None:
            raise IntakeStateError(f"No intake state at {state_path}")
        if legacy.get("schema_version") == _STATE_SCHEMA:
            state = _state(state_path, spec.bundle_digest, backend, account)
            return {
                "recovered": False,
                "reason": "state is already v2",
                "reference": state.get("reference"),
                "state_file": str(state_path),
            }
        if legacy.get("schema_version") != _STATE_SCHEMA_V1:
            raise IntakeStateError(
                f"Intake state at {state_path} is not v1 state; nothing to recover"
            )
        if legacy.get("bundle_digest") != spec.bundle_digest:
            raise IntakeStateError("v1 intake state belongs to a different research bundle")
        _check_identities(legacy, state_path)
        original = state_path.read_bytes()
        found = _lookup(api, spec, legacy["draft_key"])
        state = _new_state(spec.bundle_digest, backend, account)
        state["draft_key"] = legacy["draft_key"]
        state["publication_id"] = legacy["publication_id"]
        # A v1 run may have sent the allocation; always look before creating.
        state["allocation_attempted"] = True
        if found is None:
            if legacy.get("reference") is not None or legacy.get("status") is not None:
                raise IntakeStateError(
                    "v1 state records a server draft, but this backend has no allocation "
                    "under its key for this account. It was used against another backend "
                    "or account; recover it there. Nothing was changed."
                )
            server_status = "not_allocated"
        else:
            allocated = found.draft.reference.model_dump(mode="json")
            if legacy.get("reference") not in (None, allocated):
                raise IntakeStateError(
                    "v1 state names a different draft than the server allocated under its key"
                )
            state["reference"] = allocated
            state["collection_id"] = str(found.draft.collection_id)
            view = _view(api, state)
            if (
                view.manifest_digest is not None
                and legacy.get("manifest_digest") is not None
                and view.manifest_digest != legacy["manifest_digest"]
            ):
                raise IntakeStateError(
                    "Server revision carries a different manifest than v1 state recorded"
                )
            if view.status in _ADVANCED:
                state["status"] = "submitted"
            elif view.manifest_digest is not None:
                state["status"] = "finalized"
            if view.manifest_digest is not None:
                state["manifest_digest"] = view.manifest_digest
            server_status = view.status.value
        backup = _backup(state_path, original)
        _write_state(state_path, state)
        return {
            "recovered": True,
            "server_status": server_status,
            "reference": state.get("reference"),
            "state_file": str(state_path),
            "backup_file": str(backup),
            "next_step": (
                "Run `synth-ai index research submit` with the same conversion, "
                "state file and account to resume."
            ),
        }
