"""Prepare or run the outstanding SYN-5737 cells from W16's 1.35.1 receipt on contract 1.40.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx
import yaml

import synth_ai
from synth_ai import SynthClient
from synth_ai._generated.models import PlanRequestState, PrincipalCredentialKind, UploadState
from synth_ai.errors import SynthError

from .journey_main import read_ids, surface_factory, write_steps
from .journeys import JourneyContext, Secret, Verdict, cross_org
from .live_limits import retain
from .surfaces import SurfaceName

STAGING_ORIGIN = "https://core-api-staging-staging-acef.up.railway.app"
DRIVERS = (SurfaceName.SDK, SurfaceName.CLI, SurfaceName.MCP_STDIO)
RELEASE = "1.40.0"
WHEEL_SHA256 = "eda387ddb890af9c1bf09cd6ecfaad939dc2f94f6936b70e19c94b9f0a9bea93"
CONTRACT_SHA256 = "121bc078ec8cde848542e75badfc94b77aa30c4caba44f122ff4430c4fb645e4"
RETAINED_SHA256 = "29d0e58df8b5635ed698a11a5b5ac5c2f1c3c9a1b724231fa441b86724dbd331"
GUARDED_MUTATIONS = {
    "updateApiKey": "key",
    "updateRunLabels": "labels",
    "withdrawPlanRequest": "plan_request",
    "finalizeUpload": "upload",
    "sendRunMessage": "messages",
    "revokeOAuthGrant": "grants",
}


class FixtureBlocked(ValueError):
    """A named precondition needs an authorized fixture operator."""


@dataclass(frozen=True, slots=True)
class Cell:
    driver: SurfaceName
    operation: str
    retained_result: str

    @property
    def disposition(self) -> str:
        if self.retained_result == f"not_in_release:{RELEASE}":
            return self.retained_result
        return "pending:owned_fixture"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def outstanding(path: Path) -> tuple[Cell, ...]:
    data = path.read_bytes()
    if sha256(data) != RETAINED_SHA256:
        raise FixtureBlocked("retained_W16_receipt_sha256_mismatch")
    retained = json.loads(data)["drivers"]
    return tuple(
        Cell(driver, operation, result)
        for driver in DRIVERS
        for operation, result in retained[driver.value]["cells"].items()
        if result != "pass"
    )


def principal(client: SynthClient) -> tuple[str, str]:
    account = client.account.get()
    user = account.principal.user_id
    if user is None or user == "" or account.org.id != account.principal.org_id:
        raise FixtureBlocked("user_bound_organization_required")
    return account.org.id, user


@dataclass(frozen=True, slots=True)
class OwnerSnapshot:
    key: str
    labels: str
    plan_request: str
    upload: str
    key_version: int
    labels_revision: int
    messages: str = "absent"
    grants: str = "absent"

    @classmethod
    def read(
        cls,
        owner: SynthClient,
        session: SynthClient,
        ids: dict[str, str],
        *,
        require_ready: bool,
        origin: str = STAGING_ORIGIN,
        session_token: str = "",
    ) -> OwnerSnapshot:
        run = owner.swarms.get(ids["run_id"])
        if run.run_id != ids["run_id"] or not run.execution_closed:
            raise FixtureBlocked("owner_disposable_terminal_run_required")
        key = next(
            (
                item
                for item in session.account.api_keys.list(limit=100).items
                if item.key_id == ids["key_id"]
            ),
            None,
        )
        if require_ready and (
            key is None or not key.is_active or key.name != ids["fixture_key_name"]
        ):
            raise FixtureBlocked("owner_disposable_key_required")
        plan = next(
            (
                item
                for item in owner.account.plan_requests().items
                if item.request_id == ids["request_id"]
            ),
            None,
        )
        if require_ready and (plan is None or plan.state != PlanRequestState.PENDING):
            raise FixtureBlocked("owner_pending_plan_request_required")
        upload = owner.swarms.upload(ids["upload_id"])
        if upload.upload_id != ids["upload_id"] or (
            require_ready and upload.state != UploadState.PENDING
        ):
            raise FixtureBlocked("owner_pending_disposable_upload_required")
        messages = [
            [message.message_id, message.sequence, message.state.model_dump(mode="json")]
            for message in owner.swarms.iter_messages(ids["run_id"])
        ]
        if require_ready and ids["message_id"] not in {item[0] for item in messages}:
            raise FixtureBlocked("owner_fixture_message_required")
        grants = owner_grants(origin, session_token)
        grant = next((item for item in grants if item.get("grant_id") == ids["grant_id"]), None)
        if require_ready and (grant is None or grant.get("status") != "active"):
            raise FixtureBlocked("owner_active_oauth_grant_required")
        return cls(
            key="absent"
            if key is None
            else sha256(key.model_dump_json(exclude={"last_used_at"}).encode()),
            labels=sha256(json.dumps([run.labels_revision, run.labels], sort_keys=True).encode()),
            plan_request="absent" if plan is None else sha256(plan.model_dump_json().encode()),
            upload=sha256(upload.model_dump_json(exclude={"put"}).encode()),
            key_version=0 if key is None else key.authorization_version,
            labels_revision=run.labels_revision,
            messages=sha256(json.dumps(messages, sort_keys=True).encode()),
            grants="absent"
            if grant is None
            else sha256(json.dumps(grant, sort_keys=True).encode()),
        )


def owner_grants(origin: str, session_token: str) -> list[dict[str, Any]]:
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        response = client.get(
            origin + "/mcp/oauth/grants",
            params={"include_inactive": "true"},
            headers={"X-Clerk-Session-Token": session_token},
        )
    if response.status_code != 200:
        raise FixtureBlocked("owner_oauth_grants_unreadable")
    grants = response.json().get("grants")
    if not isinstance(grants, list):
        raise FixtureBlocked("owner_oauth_grants_unreadable")
    return [item for item in grants if isinstance(item, dict)]


def cell_rows(cells: tuple[Cell, ...]) -> list[dict[str, object]]:
    return [
        {
            "driver": cell.driver.value,
            "operation": cell.operation,
            "retained_result": cell.retained_result,
            "result": cell.disposition,
        }
        for cell in cells
    ]


def verify_output(owner: SynthClient, ids: dict[str, str]) -> None:
    output = owner.swarms.output(ids["run_id"], ids["output_id"])
    reference = output.reference
    if output.run_id != ids["run_id"] or output.output_id != ids["output_id"] or reference is None:
        raise FixtureBlocked("owner_output_reference_required")
    revision = reference.get("revision_id")
    digest = reference.get("digest_sha256")
    if revision is None or revision == "" or digest is None or digest == "":
        raise FixtureBlocked("owner_output_revision_and_digest_required")
    if ids.get("output_revision_id", revision) != revision:
        raise FixtureBlocked("owner_output_revision_mismatch")
    if ids.get("output_digest_sha256", digest) != digest:
        raise FixtureBlocked("owner_output_digest_mismatch")
    content = owner.swarms.output_content(
        ids["run_id"], ids["output_id"], revision_id=revision, digest_sha256=digest
    )
    if sha256(content) != digest:
        raise FixtureBlocked("owner_output_content_digest_mismatch")
    ids["output_revision_id"] = revision
    ids["output_digest_sha256"] = digest


def run_driver(
    driver: SurfaceName,
    directory: Path,
    cells: list[dict[str, object]],
    ids: dict[str, str],
    secrets: dict[str, str],
    baseline: OwnerSnapshot,
    document: dict[str, Any],
    *,
    origin: str = STAGING_ORIGIN,
    profile: str = "staging",
) -> Verdict:
    directory.mkdir(mode=0o700)
    make = surface_factory(
        driver, origin=origin, work_directory=directory, mcp_url=None, profile=profile
    )
    operations = frozenset(str(row["operation"]) for row in cells)
    with make(secrets["SYNTH_API_KEY"]) as surface:
        receipt = cross_org(
            JourneyContext(
                surface=surface,
                driver=driver,
                origin=origin,
                make=make,
                receipt_dir=directory,
                foreign_ids=ids,
                secrets=secrets,
                isolation_operations=operations,
                isolation_contract=document,
                isolation_bodies={
                    "updateApiKey": {
                        "expected_version": baseline.key_version,
                        "scopes": ["account:read"],
                    },
                    "updateRunLabels": {
                        "expected_revision": baseline.labels_revision,
                        "labels": {"isolation-probe": "disposable"},
                    },
                },
            )
        )
    write_steps(directory, driver, [receipt], "driver-receipt.json")
    evidence = {str(item["operation"]): item for item in receipt.evidence}
    for row in cells:
        observed = evidence.get(str(row["operation"]))
        row["result"] = (
            observed["result"]
            if observed is not None
            else ("unproven:" + (receipt.cause or "cell_receipt_missing"))
        )
        if observed is not None:
            row["evidence"] = observed
    return receipt.verdict


def run(
    rows: list[dict[str, object]],
    directory: Path,
    ids: dict[str, str],
    document: dict[str, Any],
    *,
    origin: str = STAGING_ORIGIN,
    profile: str = "staging",
) -> None:
    required = {
        "owner_org_id",
        "owner_user_id",
        "foreign_org_id",
        "foreign_user_id",
        "run_id",
        "output_id",
        "key_id",
        "upload_id",
        "request_id",
        "fixture_key_name",
        "message_id",
        "grant_id",
    }
    if any(ids.get(name, "") == "" for name in required):
        raise FixtureBlocked("designated_fixture_ids_required")
    names = ("SYNTH_API_KEY", Secret.SESSION, Secret.FOREIGN, Secret.FOREIGN_SESSION)
    if any(os.environ.get(name, "") == "" for name in names):
        raise FixtureBlocked("owner_and_foreign_key_and_session_required")
    secrets = {str(name): os.environ[name] for name in names}
    with (
        SynthClient(secrets["SYNTH_API_KEY"], base_url=origin) as owner,
        SynthClient(secrets[Secret.SESSION], base_url=origin) as owner_session,
        SynthClient(secrets[Secret.FOREIGN], base_url=origin) as foreign,
        SynthClient(secrets[Secret.FOREIGN_SESSION], base_url=origin) as foreign_session,
    ):
        owner_id = (ids["owner_org_id"], ids["owner_user_id"])
        foreign_id = (ids["foreign_org_id"], ids["foreign_user_id"])
        if owner_id[0] == foreign_id[0] or owner_id[1] == foreign_id[1]:
            raise FixtureBlocked("the_two_fixture_principals_must_be_distinct")
        if principal(owner) != owner_id or principal(owner_session) != owner_id:
            raise FixtureBlocked("owner_fixture_principal_mismatch")
        if principal(foreign) != foreign_id or principal(foreign_session) != foreign_id:
            raise FixtureBlocked("foreign_fixture_principal_mismatch")
        if any(
            client.account.get().principal.credential_kind != PrincipalCredentialKind.SESSION
            for client in (owner_session, foreign_session)
        ):
            raise FixtureBlocked("clerk_sessions_required")
        if "account:write" not in foreign.account.get().principal.scopes:
            raise FixtureBlocked("foreign_account_write_scope_required_for_plan_withdrawal")
        verify_output(owner, ids)
        for row in rows:
            if row["operation"] == "getRunOutputContent":
                row["owner_output_pin_verified"] = True
        readback = {"origin": origin, "session_token": secrets[Secret.SESSION]}
        before = OwnerSnapshot.read(owner, owner_session, ids, require_ready=True, **readback)
        for driver in DRIVERS:
            cells = [
                row
                for row in rows
                if row["driver"] == driver and row["result"] == "pending:owned_fixture"
            ]
            try:
                verdict = run_driver(
                    driver,
                    directory / driver.value,
                    cells,
                    ids,
                    secrets,
                    before,
                    document,
                    origin=origin,
                    profile=profile,
                )
            finally:
                try:
                    after = OwnerSnapshot.read(
                        owner, owner_session, ids, require_ready=False, **readback
                    )
                except (FixtureBlocked, SynthError):
                    for row in cells:
                        if row["operation"] in GUARDED_MUTATIONS:
                            row["owner_state"] = "unproven"
                            if row["result"] == "pass":
                                row["result"] = "unproven:owner_readback_failed"
                    raise
                for row in cells:
                    field = GUARDED_MUTATIONS.get(str(row["operation"]))
                    if field is not None:
                        prior = getattr(before, field)
                        current = getattr(after, field)
                        unchanged = prior == current
                        row["owner_state"] = "unchanged" if unchanged else "changed"
                        row["owner_snapshot"] = {"before": prior, "after": current}
                        if not unchanged:
                            row["result"] = "fail:owner_state_changed"
                if before != after:
                    for row in cells:
                        if row["operation"] in GUARDED_MUTATIONS and row["result"] == "pass":
                            row["result"] = "unproven:owner_fixture_changed"
                    raise FixtureBlocked("owner_fixture_changed_stop_remaining_drivers")
            if verdict == Verdict.FAIL:
                raise FixtureBlocked("isolation_failed_stop_remaining_drivers")


def verify_wheel(wheel: Path) -> None:
    if sha256(wheel.read_bytes()) != WHEEL_SHA256 or not synth_ai.__file__:
        raise FixtureBlocked("frozen_wheel_sha256_mismatch")
    installed_package = Path(synth_ai.__file__).parent
    with ZipFile(wheel) as archive:
        sources = [
            name
            for name in archive.namelist()
            if name.startswith("synth_ai/") and name.endswith(".py")
        ]
        if not sources:
            raise FixtureBlocked("frozen_wheel_sources_absent")
        for name in sources:
            source = installed_package / name.removeprefix("synth_ai/")
            if not source.is_file() or source.read_bytes() != archive.read(name):
                raise FixtureBlocked("installed_SDK_differs_from_frozen_wheel")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--fixture-ids", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    try:
        rows = cell_rows(outstanding(options.retained))
        contract_bytes = options.contract.read_bytes()
        if sha256(contract_bytes) != CONTRACT_SHA256:
            raise FixtureBlocked("pinned_contract_sha256_mismatch")
        document = yaml.safe_load(contract_bytes)
        if document["info"]["version"] != RELEASE:
            raise FixtureBlocked("frozen_contract_version_mismatch")
    except FixtureBlocked as error:
        parser.error(str(error))
    options.receipt_dir.mkdir(mode=0o700, parents=True)
    error_code = None
    try:
        if options.execute:
            if os.environ.get("SYNTH_STAGING_API_BASE_URL", "").rstrip("/") != STAGING_ORIGIN:
                raise FixtureBlocked("staging_origin_mismatch")
            if (
                options.cohort_manifest is None
                or options.fixture_ids is None
                or options.wheel is None
            ):
                raise FixtureBlocked("current_staging_manifest_fixture_ids_and_wheel_required")
            verify_wheel(options.wheel)
            manifest = options.cohort_manifest.read_bytes()
            if (
                sha256(manifest) != options.manifest_sha256
                or json.loads(manifest)["target"] != "staging"
            ):
                raise FixtureBlocked("current_staging_manifest_mismatch")
            os.environ["PYTHONSAFEPATH"] = "1"
            run(rows, options.receipt_dir, dict(read_ids(options.fixture_ids)), document)
    except FixtureBlocked as error:
        error_code = str(error)
    except (KeyError, OSError, SynthError, ValueError) as error:
        error_code = type(error).__name__
    finally:
        if options.execute:
            for row in rows:
                if row["result"] == "pending:owned_fixture":
                    row["result"] = "unproven:dispatch_not_completed"
        retain(
            options.receipt_dir / "remaining-cells.json",
            json.dumps(
                {
                    "schema": "synth.isolation.remaining.v1",
                    "release": RELEASE,
                    "release_freeze_manifest_sha256": options.manifest_sha256,
                    "retained_matrix_sha256": RETAINED_SHA256,
                    "contract_sha256": CONTRACT_SHA256,
                    "wheel_sha256": WHEEL_SHA256 if options.execute else None,
                    "execution_requested": options.execute,
                    "probe_error": error_code,
                    "cells": rows,
                },
                indent=2,
            ),
        )
    if any(
        row["result"] in {"leak", "differs"} or str(row["result"]).startswith("fail:")
        for row in rows
    ):
        return 1
    if options.execute and (
        error_code is not None
        or any(row["result"] not in {"pass", f"not_in_release:{RELEASE}"} for row in rows)
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
