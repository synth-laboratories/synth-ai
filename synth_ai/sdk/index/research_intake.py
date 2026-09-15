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
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from .client import IndexAPI
from .contracts import ContributionAudience, ContributionOrigin, ContributionReference
from .contributions import ContributionUploadSpec, ResearchDraftSpec, ResearchSource
from .package import ContributionPackage
from .submission import ContributionSubmitSpec, RevisionStatus
from .transfer import upload_directory_sync, verify_package_directory

_RECEIPT_SCHEMA = "synth.index.research-bundle-conversion.v1"
_SOURCE_SCHEMA = "synth.index.research-source.v1"
_STATE_SCHEMA = "synth.index.research-intake-state.v1"
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


def _state(path: Path, bundle_digest: str) -> dict:
    if path.exists():
        if path.is_symlink():
            raise ValueError("Intake state may not be a link")
        state = json.loads(path.read_bytes())
        if (
            state.get("schema_version") != _STATE_SCHEMA
            or state.get("bundle_digest") != bundle_digest
        ):
            raise ValueError("Intake state belongs to a different research bundle")
        UUID(state["publication_id"])
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", state["draft_key"]):
            raise ValueError("Intake state has an invalid draft key")
        return state
    state = {
        "schema_version": _STATE_SCHEMA,
        "bundle_digest": bundle_digest,
        "draft_key": uuid4().hex,
        "publication_id": str(uuid4()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(state, handle, sort_keys=True)
        handle.write("\n")
    return state


def _save_state(path: Path, state: dict) -> None:
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


def submit_conversion(
    api: IndexAPI, directory: Path, state_path: Path, *, finalize_only: bool = False
) -> dict:
    """Resume private allocation/upload; submit only when review gates allow.

    Re-running reuses immutable IDs and obtains fresh storage targets. The
    backend may omit objects already stored under the publication ID.
    """
    package, spec, summary = preview_conversion(directory)
    state = _state(state_path, spec.bundle_digest)
    if state.get("status") == "submitted":
        reference = ContributionReference.model_validate(state["reference"])
        view = api.contributions.revisions.retrieve(reference)
        if view.status != RevisionStatus.SUBMITTED or view.manifest_digest != state.get(
            "manifest_digest"
        ):
            raise ValueError("Saved submission state differs from the server revision")
        return {
            **summary,
            "reference": reference.model_dump(mode="json"),
            "manifest_digest": view.manifest_digest,
            "status": "submitted",
        }
    draft = api.contributions.create_research(spec, idempotency_key=state["draft_key"])
    if state.get("reference") and state["reference"] != draft.reference.model_dump(mode="json"):
        raise ValueError("Research draft replay returned a different server identity")
    state["reference"] = draft.reference.model_dump(mode="json")
    _save_state(state_path, state)
    if state.get("status") != "finalized":
        rebound = _rebind(package, draft.reference)
        prepared = api.contributions.prepare_upload(
            draft,
            ContributionUploadSpec(publication_id=UUID(state["publication_id"]), package=rebound),
        )
        upload_directory_sync(prepared, directory / "package")
        api.contributions.finalize(draft, prepared)
        state["status"] = "finalized"
        state["manifest_digest"] = prepared.transfer.manifest_digest
        _save_state(state_path, state)
    if finalize_only:
        return {
            **summary,
            "reference": state["reference"],
            "manifest_digest": state["manifest_digest"],
            "status": "finalized_private_draft",
        }
    submitted = api.contributions.submit(
        draft.reference,
        ContributionSubmitSpec(publication_id=UUID(state["publication_id"])),
    )
    if (
        submitted.status != RevisionStatus.SUBMITTED
        or submitted.manifest_digest != state["manifest_digest"]
    ):
        raise ValueError("Submitted revision differs from the prepared manifest")
    state["status"] = "submitted"
    state["manifest_digest"] = submitted.manifest_digest
    _save_state(state_path, state)
    return {
        **summary,
        "reference": state["reference"],
        "manifest_digest": submitted.manifest_digest,
        "status": "submitted",
    }
