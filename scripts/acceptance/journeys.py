"""J2 money, J3 access and J-refusals steps through SDK, CLI and MCP (SYN-5695).

The journey definitions (step order, drivers, expectations, preconditions and
product gaps) live in synth-dev `local_dev/release_tests/journeys.py`. This
module runs one step on one surface and returns a typed `StepReceipt`. A step
whose precondition is absent is Inconclusive; it never passes by default.

The steps never launch an uncached run: a cached-slot launch must go through the
cached harness's replay transport. Refusal probes send a launch that Core must
refuse; an unexpected acceptance is cancelled at once and recorded as Fail.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import base64
import re
import secrets
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import yaml

from ._journeys_access import (
    cli_login as cli_login,
)
from ._journeys_access import (
    cli_oauth as cli_oauth,
)
from ._journeys_access import (
    hosted_token as hosted_token,
)
from ._journeys_access import (
    mcp_consent as mcp_consent,
)
from ._journeys_access import (
    member_admin as member_admin,
)
from ._journeys_access import (
    mint_key as mint_key,
)
from ._journeys_access import (
    oauth_failure as oauth_failure,
)
from ._journeys_access import (
    owned_grant as owned_grant,
)
from ._journeys_access import (
    private_auth_file as private_auth_file,
)
from ._journeys_access import (
    revoke_key as revoke_key,
)
from ._journeys_access import (
    rotate_key as rotate_key,
)
from ._journeys_common import (
    ADMIN_ROLES as ADMIN_ROLES,
)
from ._journeys_common import (
    API_KEY_INVALID as API_KEY_INVALID,
)
from ._journeys_common import (
    ID_PARAMETERS,
    MATRIX_BODIES,
    RELEASES_WITHOUT_MESSAGE_ROUTES,
    REQUIRED_MESSAGE_OPERATIONS,
    ROOT,
    JourneyContext,
    JourneyPrincipal,
    Secret,
    Step,
    StepReceipt,
    Verdict,
    covered,
    key,
    missing,
    obj,
    receipt,
)
from ._journeys_common import (
    MEMBER_ROLES as MEMBER_ROLES,
)
from ._journeys_common import (
    OVERLOAD_CODES as OVERLOAD_CODES,
)
from ._journeys_common import (
    POLL_SECONDS as POLL_SECONDS,
)
from ._journeys_common import (
    PROBE_CHARGE as PROBE_CHARGE,
)
from ._journeys_common import (
    RESUMED as RESUMED,
)
from ._journeys_common import (
    STATE_SECONDS as STATE_SECONDS,
)
from ._journeys_common import (
    TERMINAL as TERMINAL,
)
from ._journeys_common import (
    SurfaceFactory as SurfaceFactory,
)
from ._journeys_common import (
    api_key_refused as api_key_refused,
)
from ._journeys_common import (
    credential_works as credential_works,
)
from ._journeys_common import (
    expect_refusal as expect_refusal,
)
from ._journeys_common import (
    items as items,
)
from ._journeys_common import (
    launch_body as launch_body,
)
from ._journeys_common import (
    probe_launch as probe_launch,
)
from ._journeys_common import (
    wait_state as wait_state,
)
from ._journeys_money import (
    CHECKOUT_COMPLETION_WAIT_SECONDS as CHECKOUT_COMPLETION_WAIT_SECONDS,
)
from ._journeys_money import (
    CHECKOUT_POLL_SECONDS as CHECKOUT_POLL_SECONDS,
)
from ._journeys_money import (
    HISTORY_MAX_PAGES as HISTORY_MAX_PAGES,
)
from ._journeys_money import (
    HISTORY_PAGE_LIMIT as HISTORY_PAGE_LIMIT,
)
from ._journeys_money import (
    PURCHASE_USD as PURCHASE_USD,
)
from ._journeys_money import (
    TERMINAL_CHECKOUT as TERMINAL_CHECKOUT,
)
from ._journeys_money import (
    buy_credits as buy_credits,
)
from ._journeys_money import (
    checkout_state as checkout_state,
)
from ._journeys_money import (
    entry_row as entry_row,
)
from ._journeys_money import (
    history_interval as history_interval,
)
from ._journeys_money import (
    instant as instant,
)
from ._journeys_money import (
    limit_paused as limit_paused,
)
from ._journeys_money import (
    pause_on_limit as pause_on_limit,
)
from ._journeys_money import (
    raise_four as raise_four,
)
from ._journeys_money import (
    raise_limit as raise_limit,
)
from ._journeys_money import (
    reconcile_purchase as reconcile_purchase,
)
from ._journeys_money import (
    resume as resume,
)
from ._journeys_money import (
    sku_refusal as sku_refusal,
)
from ._journeys_money import (
    spend_cap as spend_cap,
)
from ._journeys_money import (
    statement as statement,
)
from ._journeys_money import (
    usd as usd,
)
from ._journeys_money import (
    utc_stamp as utc_stamp,
)
from ._journeys_refusals import (
    ADMISSION_PROBE_CHARGE as ADMISSION_PROBE_CHARGE,
)
from ._journeys_refusals import (
    ADMISSION_QUEUE_FULL as ADMISSION_QUEUE_FULL,
)
from ._journeys_refusals import (
    CLIENT_REFUSALS as CLIENT_REFUSALS,
)
from ._journeys_refusals import (
    LAUNCH_WINDOW_FILL as LAUNCH_WINDOW_FILL,
)
from ._journeys_refusals import (
    VERSION_REFUSALS as VERSION_REFUSALS,
)
from ._journeys_refusals import (
    admission_overload_refusal as admission_overload_refusal,
)
from ._journeys_refusals import (
    expired_key as expired_key,
)
from ._journeys_refusals import (
    index as index,
)
from ._journeys_refusals import (
    overload_recovery as overload_recovery,
)
from ._journeys_refusals import (
    overload_refusal as overload_refusal,
)
from ._journeys_refusals import (
    oversize as oversize,
)
from ._journeys_refusals import (
    queued_launch as queued_launch,
)
from ._journeys_refusals import (
    rate_limit_refusal as rate_limit_refusal,
)
from ._journeys_refusals import (
    topology as topology,
)
from ._journeys_refusals import (
    topology_version as topology_version,
)
from .isolation import classify
from .session_files import RefreshingSurface, SessionRefused
from .surfaces import CLI_ARGV, SDK_CALLS, Call, Outcome, Surface, SurfaceName, catalog

if TYPE_CHECKING:
    from pydantic import JsonValue

__all__ = [
    "JourneyContext",
    "Step",
    "StepReceipt",
    "Verdict",
    "capabilities",
    "matrix_operations",
    "run_step",
]


def contract() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "contracts/synth-v1.openapi.yaml").read_text())


def path_parameters(document: dict[str, Any], operation: dict[str, Any]) -> list[str]:
    components = document["components"]
    assert isinstance(components, dict)
    names: list[str] = []
    for parameter in operation.get("parameters") or []:
        if isinstance(parameter, dict) and "$ref" in parameter:
            parameter = components["parameters"][str(parameter["$ref"]).rsplit("/", 1)[1]]
        if isinstance(parameter, dict) and parameter.get("in") == "path":
            names.append(str(parameter["name"]))
    return names


def session_only_operations(document: Optional[dict[str, Any]] = None) -> frozenset[str]:
    """Matrix operations whose contract security admits only a Clerk session."""
    document = contract() if document is None else document
    paths = document["paths"]
    assert isinstance(paths, dict)
    return frozenset(
        str(operation["operationId"])
        for methods in paths.values()
        for method, operation in methods.items()
        if method in {"get", "post", "put", "patch", "delete"}
        and operation.get("security") == [{"clerkSession": []}]
    )


def contract_version() -> str:
    return str(contract()["info"]["version"])


def matrix_operations(
    document: Optional[dict[str, Any]] = None,
) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    """Every contract operation addressed by a resource ID: (op, method, path, params)."""
    document = contract() if document is None else document
    rows: list[tuple[str, str, str, tuple[str, ...]]] = []
    paths = document["paths"]
    assert isinstance(paths, dict)
    for route, methods in sorted(paths.items()):
        for method, operation in methods.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            names = tuple(path_parameters(document, operation))
            if any(name in ID_PARAMETERS or name.endswith("_id") for name in names):
                rows.append((str(operation["operationId"]), method, route, names))
    return tuple(rows)


ID_SHAPE = re.compile(r"^(?P<prefix>[a-z]+_)(?P<tail>[A-Za-z0-9_-]+)$")


def absent_like(foreign: str) -> str:
    """A well-formed ID of the same shape that no org owns."""
    match = ID_SHAPE.match(foreign)
    if match is not None:
        tail = match.group("tail")
        if len(tail) == 22:  # Core public IDs: prefix + unpadded base64url of a UUID.
            encoded = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")
            return str(match.group("prefix")) + encoded
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        return str(match.group("prefix")) + "".join(secrets.choice(alphabet) for _ in tail)
    try:
        uuid.UUID(foreign)
        return str(uuid.uuid4())
    except ValueError:
        return "absent" + secrets.token_hex(8)


# Mutations whose cell also needs org A's own call as the positive control: org B's 404
# only proves isolation when the same request from the owner reaches the resource.
OWNER_CONTROLLED_MUTATIONS = frozenset(
    {"updateApiKey", "updateRunLabels", "finalizeUpload", "withdrawPlanRequest"}
)
# Optional --foreign-ids entries carrying the owner fixture's optimistic-concurrency value.
MATRIX_BODY_PINS: Mapping[str, tuple[str, str]] = {
    "updateApiKey": ("expected_version", "key_version"),
    "updateRunLabels": ("expected_revision", "labels_revision"),
}


# Coordinator ruling 2026-11-19T06:43:58Z (SYN-5737): a foreign API key refused before
# the grant is resolved is re-probed with the Clerk session this operation also admits.
# Explicit on purpose: no other row may turn green through a second credential.
SESSION_REPROBED_OPERATIONS = frozenset({"revokeOAuthGrant"})


def operation_scopes(document: dict[str, Any]) -> dict[str, str]:
    paths = document["paths"]
    assert isinstance(paths, dict)
    return {
        str(operation["operationId"]): str(operation.get("x-synth-scope", ""))
        for methods in paths.values()
        for method, operation in methods.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }


def matrix_body(context: JourneyContext, operation: str) -> Optional[Mapping[str, JsonValue]]:
    body = context.isolation_bodies.get(operation, MATRIX_BODIES.get(operation))
    pin = MATRIX_BODY_PINS.get(operation)
    if body is None or pin is None or context.isolation_operations is not None:
        return body
    field_name, id_name = pin
    pinned = context.foreign_ids.get(id_name, "")
    return {**body, field_name: int(pinned)} if pinned.isdigit() else body


def matrix_call(
    operation: str,
    body: Optional[Mapping[str, JsonValue]],
    query: Mapping[str, JsonValue],
    mutation: bool,
    ids: Mapping[str, str],
) -> Call:
    return Call(
        operation,
        path=ids,
        body=body,
        query=query,
        idempotency_key=key("matrix") if mutation else None,
    )


def principal_scopes(account: Outcome) -> frozenset[str]:
    scopes = obj(obj(account.value).get("principal")).get("scopes")
    return frozenset(str(scope) for scope in scopes) if isinstance(scopes, list) else frozenset()


def owner_control(observed: Outcome) -> Optional[str]:
    """How org A's own request reached its resource, or None when it did not."""
    if observed.accepted:
        return "accepted"
    if observed.typed and observed.status == 409:
        # A consumed one-shot fixture or a moved version: Core resolved the owner's resource.
        return f"resolved_conflict:{observed.code}"
    return None


def cross_org(context: JourneyContext) -> StepReceipt:
    """Org B (actor) addresses org A's IDs and absent IDs; answers must be identical 404s."""
    step = Step.CROSS_ORG_404
    actor = context.secrets.get(Secret.FOREIGN)
    if actor is None:
        return missing(context, step, "foreign_org_api_key")
    if not context.foreign_ids:
        return missing(context, step, "foreign_org_ids")
    rows: list[Mapping[str, JsonValue]] = []
    failed = uncovered = 0
    document = context.isolation_contract or contract()
    operations = matrix_operations(document)
    if not operations:
        return missing(context, step, "id_bearing_operation_matrix_empty")
    published = {operation for operation, _, _, _ in operations}
    version = (
        contract_version()
        if context.isolation_contract is None
        else str(document["info"]["version"])
    )
    selected = context.isolation_operations
    if selected is not None:
        unknown = selected - published - REQUIRED_MESSAGE_OPERATIONS
        if unknown:
            raise ValueError("selected_isolation_operation_unknown")
    for operation in sorted(REQUIRED_MESSAGE_OPERATIONS - published):
        if selected is not None and operation not in selected:
            continue
        if version in RELEASES_WITHOUT_MESSAGE_ROUTES:
            rows.append({"operation": operation, "result": "not_in_release:" + version})
            continue
        uncovered += 1
        rows.append({"operation": operation, "result": "uncovered:contract_operation_missing"})
    # Session tokens are read at each use (a 0600 file a refresher rewrites, or a
    # fixed value); every newly read token reopens its surface and re-runs the
    # principal check below before any other request uses it.
    session = context.secrets.get(Secret.FOREIGN_SESSION)
    session_only = session_only_operations(document)
    scopes = operation_scopes(document)
    owner_session = context.secrets.get(Secret.SESSION)
    with context.make(actor) as keyed, ExitStack() as stack:
        owner_account = context.surface.call(Call("getAccount"))
        actor_account = keyed.call(Call("getAccount"))
        owner = JourneyPrincipal.from_account(owner_account)
        foreign_actor = JourneyPrincipal.from_account(actor_account)
        if owner is None or foreign_actor is None:
            return missing(context, step, "user_bound_organization_principals")
        if owner.org_id == foreign_actor.org_id or owner.user_id == foreign_actor.user_id:
            return receipt(
                context,
                step,
                Verdict.FAIL,
                "foreign_principal_matches_owner",
                owner_account,
                actor_account,
            )
        if selected is not None:
            expected = context.foreign_ids
            if (owner.org_id, owner.user_id) != (
                expected.get("owner_org_id"),
                expected.get("owner_user_id"),
            ) or (foreign_actor.org_id, foreign_actor.user_id) != (
                expected.get("foreign_org_id"),
                expected.get("foreign_user_id"),
            ):
                return receipt(
                    context, step, Verdict.FAIL, "designated_fixture_principals_mismatch"
                )

        def verify_foreign(account: Outcome) -> Optional[str]:
            actor_ = JourneyPrincipal.from_account(account)
            if actor_ is None:
                return "precondition_missing:user_bound_foreign_session"
            if (actor_.org_id, actor_.user_id) != (foreign_actor.org_id, foreign_actor.user_id):
                return "foreign_session_principal_mismatch"
            return None

        def verify_owner(account: Outcome) -> Optional[str]:
            owner_ = JourneyPrincipal.from_account(account)
            if owner_ is None or (owner_.org_id, owner_.user_id) != (owner.org_id, owner.user_id):
                return "owner_session_principal_mismatch"
            return None

        def account_of(surface: Surface) -> Outcome:
            return surface.call(Call("getAccount"))

        sessioned = (
            stack.enter_context(
                RefreshingSurface(
                    context.make,
                    lambda: context.secrets[Secret.FOREIGN_SESSION],
                    account_of,
                    verify_foreign,
                )
            )
            if session is not None
            else None
        )
        owner_sessioned = (
            stack.enter_context(
                RefreshingSurface(
                    context.make, lambda: context.secrets[Secret.SESSION], account_of, verify_owner
                )
            )
            if owner_session is not None and selected is None
            else None
        )
        try:
            for refreshing in (sessioned, owner_sessioned):
                if refreshing is not None:
                    refreshing.surface()
            rows_result = matrix_rows(
                context,
                operations,
                selected,
                session_only,
                scopes,
                keyed,
                sessioned,
                owner_sessioned,
                owner,
                foreign_actor,
                actor_account,
                rows,
            )
        except SessionRefused as refused:
            verdict = (
                Verdict.INCONCLUSIVE
                if refused.cause.startswith("precondition_missing:")
                else Verdict.FAIL
            )
            evidence = [
                *(dict(row) for row in rows),
                (owner_account if "owner" in refused.cause else actor_account).evidence(),
                refused.account.evidence(),
            ]
            return StepReceipt(step, context.driver, verdict, refused.cause, tuple(evidence))
        failed, uncovered = rows_result[0], uncovered + rows_result[1]
    verdict = Verdict.FAIL if failed else (Verdict.INCONCLUSIVE if uncovered else Verdict.PASS)
    cause = (
        ""
        if verdict == Verdict.PASS
        else (f"failed={failed}" if failed else f"uncovered={uncovered}")
    )
    return StepReceipt(step, context.driver, verdict, cause, tuple(rows))


def matrix_rows(
    context: JourneyContext,
    operations: tuple[tuple[str, str, str, tuple[str, ...]], ...],
    selected: Optional[frozenset[str]],
    session_only: frozenset[str],
    scopes: Mapping[str, str],
    keyed: Surface,
    sessioned: Optional[RefreshingSurface],
    owner_sessioned: Optional[RefreshingSurface],
    owner: JourneyPrincipal,
    foreign_actor: JourneyPrincipal,
    actor_account: Outcome,
    rows: list[Mapping[str, JsonValue]],
) -> tuple[int, int]:
    """The per-operation probes; appends rows and returns (failed, uncovered)."""
    failed = uncovered = 0
    for operation, method, route, names in operations:
        if selected is not None and operation not in selected:
            continue
        by_session = operation in session_only and sessioned is not None
        surface = sessioned if by_session else keyed
        if not covered(surface, operation):
            uncovered += 1
            rows.append({"operation": operation, "result": "uncovered:surface_unmapped"})
            continue
        if selected is not None and operation in session_only and sessioned is None:
            uncovered += 1
            rows.append({"operation": operation, "result": "uncovered:foreign_session_absent"})
            continue
        needed = [name for name in names if name != "kind"]
        if any(name not in context.foreign_ids for name in needed):
            uncovered += 1
            rows.append({"operation": operation, "result": "uncovered:foreign_id_absent"})
            continue
        query: dict[str, JsonValue] = {"timeout_seconds": 1} if operation == "waitRun" else {}
        if operation == "getRunOutputContent":
            revision = context.foreign_ids.get("output_revision_id", "")
            digest = context.foreign_ids.get("output_digest_sha256", "")
            if not revision or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                uncovered += 1
                rows.append(
                    {
                        "operation": operation,
                        "result": "uncovered:output_pin_missing_or_invalid",
                    }
                )
                continue
            # The owned output pin is identical for both requests so only the IDs vary.
            query = {"revision_id": revision, "digest_sha256": digest}
        foreign = {name: context.foreign_ids[name] for name in needed}
        absent = {name: absent_like(value) for name, value in foreign.items()}
        if "kind" in names:
            foreign["kind"] = absent["kind"] = context.foreign_ids.get("kind", "task")
        body = matrix_body(context, operation)
        mutation = method != "get"
        identifiers = tuple(foreign.values()) + tuple(absent.values())
        request = partial(matrix_call, operation, body, query, mutation)

        observed_foreign = surface.call(request(foreign))
        observed_absent = (
            None
            if selected is not None and observed_foreign.accepted
            else surface.call(request(absent))
        )
        result = (
            "leak"
            if observed_absent is None
            else classify(observed_foreign, observed_absent, identifiers)
        )
        extra: dict[str, JsonValue] = {}
        if (
            selected is None
            and result == "unproven:uniform_403"
            and not by_session
            and sessioned is not None
            and operation in SESSION_REPROBED_OPERATIONS
        ):
            # The API key was refused before the resource was resolved, so it proves
            # nothing about isolation. The cell is proven with the session credential
            # the operation also admits; the key's refusal is kept as an observation
            # and is never itself a pass of the contract's apiKey scheme.
            scope = scopes.get(operation, "")
            held = scope in principal_scopes(actor_account)
            extra["api_key_observation"] = {
                "status": observed_foreign.status,
                "code": observed_foreign.code,
                "required_scope": scope,
                "key_holds_required_scope": held,
                "finding": (
                    "contract_deviation:scoped_user_bound_api_key_refused"
                    if held
                    else "key_lacks_required_scope"
                ),
                "api_key_scheme": "not_proven",
            }
            by_session, surface = True, sessioned
            observed_foreign = surface.call(request(foreign))
            observed_absent = surface.call(request(absent))
            result = classify(observed_foreign, observed_absent, identifiers)
        if selected is None and result == "pass" and operation in OWNER_CONTROLLED_MUTATIONS:
            # Positive control, after org B's probes: org A's own identical request.
            controller = owner_sessioned if by_session else context.surface
            if controller is None:
                result = "unproven:owner_session_absent"
            else:
                observed_owner = controller.call(request(foreign))
                control = owner_control(observed_owner)
                extra["owner_control"] = {
                    "result": control or "refused",
                    "credential": "clerk_session" if by_session else "api_key",
                    **observed_owner.evidence(),
                }
                if control is None:
                    result = "unproven:owner_control_" + str(
                        observed_owner.code or observed_owner.status or "untyped"
                    )
        failed += result in {"leak", "differs"}
        uncovered += result.startswith("unproven:")
        rows.append(
            {
                "operation": operation,
                "route": route,
                "result": result,
                "actor_credential": "clerk_session" if by_session else "api_key",
                "owner_org_id": owner.org_id,
                "owner_user_id": owner.user_id,
                "foreign_org_id": foreign_actor.org_id,
                "foreign_user_id": foreign_actor.user_id,
                "foreign": observed_foreign.evidence(),
                "absent": None if observed_absent is None else observed_absent.evidence(),
                **extra,
            }
        )
        if selected is not None and result in {"leak", "differs"}:
            break
    return failed, uncovered


def message_step(context: JourneyContext, step: Step) -> StepReceipt:
    # Lazy import: message steps reuse this module's receipts and context.
    from .message_journey import MESSAGE_OPERATIONS, MESSAGE_STEPS

    uncovered = [
        operation
        for operation in MESSAGE_OPERATIONS[step]
        if not covered(context.surface, operation)
    ]
    if uncovered:
        return missing(context, step, "message_surface_operations:" + ",".join(uncovered))
    return MESSAGE_STEPS[step](context)


StepFunction = Callable[[JourneyContext], StepReceipt]

STEPS: Mapping[Step, tuple[StepFunction, frozenset[SurfaceName]]] = {
    Step.MESSAGE_QUESTION: (
        lambda context: message_step(context, Step.MESSAGE_QUESTION),
        frozenset(SurfaceName),
    ),
    Step.MESSAGE_TERMINAL: (
        lambda context: message_step(context, Step.MESSAGE_TERMINAL),
        frozenset(SurfaceName),
    ),
    Step.MESSAGE_OVERSIZE: (
        lambda context: message_step(context, Step.MESSAGE_OVERSIZE),
        frozenset(SurfaceName),
    ),
    Step.MESSAGE_RATE: (
        lambda context: message_step(context, Step.MESSAGE_RATE),
        frozenset(SurfaceName),
    ),
    Step.MESSAGE_ISOLATION: (
        lambda context: message_step(context, Step.MESSAGE_ISOLATION),
        frozenset(SurfaceName),
    ),
    Step.BUY_CREDITS_TEST: (buy_credits, frozenset(SurfaceName)),
    Step.RAISE_LIMIT: (raise_limit, frozenset(SurfaceName)),
    Step.SKU_REFUSAL: (sku_refusal, frozenset(SurfaceName)),
    Step.PAUSE_LIMIT: (pause_on_limit, frozenset(SurfaceName)),
    Step.RESUME: (resume, frozenset(SurfaceName)),
    Step.STATEMENT: (statement, frozenset(SurfaceName)),
    Step.MINT_KEY: (mint_key, frozenset(SurfaceName)),
    Step.ROTATE_KEY: (rotate_key, frozenset(SurfaceName)),
    Step.REVOKE_KEY: (revoke_key, frozenset(SurfaceName)),
    Step.CLI_OAUTH: (cli_oauth, frozenset({SurfaceName.SDK, SurfaceName.CLI})),
    Step.MCP_CONSENT: (mcp_consent, frozenset({SurfaceName.MCP_HOSTED})),
    Step.MEMBER_ADMIN: (member_admin, frozenset(SurfaceName)),
    Step.OVERSIZE: (oversize, frozenset(SurfaceName)),
    Step.TOPOLOGY: (topology, frozenset(SurfaceName)),
    Step.TOPOLOGY_VERSION: (topology_version, frozenset(SurfaceName)),
    Step.INDEX: (index, frozenset(SurfaceName)),
    Step.EXPIRED_KEY: (expired_key, frozenset(SurfaceName)),
    Step.OVERLOAD_REFUSAL: (overload_refusal, frozenset(SurfaceName)),
    Step.OVERLOAD_RECOVERY: (overload_recovery, frozenset(SurfaceName)),
    Step.RATE_LIMIT_REFUSAL: (rate_limit_refusal, frozenset(SurfaceName)),
    Step.QUEUED_LAUNCH: (
        queued_launch,
        frozenset({SurfaceName.SDK, SurfaceName.CLI, SurfaceName.MCP_STDIO}),
    ),
    Step.CROSS_ORG_404: (cross_org, frozenset(SurfaceName)),
}


def capabilities() -> dict[str, object]:
    """What this checkout can execute; the synth-dev validator reads it offline."""
    return {
        "schema": "synth.journeys.capabilities.v1",
        "steps": {
            step.value: sorted(driver.value for driver in drivers)
            for step, (_, drivers) in STEPS.items()
        },
        "operations": {
            SurfaceName.SDK.value: sorted(SDK_CALLS),
            SurfaceName.CLI.value: sorted(CLI_ARGV),
            SurfaceName.MCP_STDIO.value: sorted(catalog()),
            SurfaceName.MCP_HOSTED.value: sorted(catalog()),
        },
        "matrix": [row[0] for row in matrix_operations()],
    }


def run_step(context: JourneyContext, step: Step) -> StepReceipt:
    function, drivers = STEPS[step]
    if context.driver not in drivers:
        return receipt(context, step, Verdict.INCONCLUSIVE, "driver_not_supported_for_step")
    return function(context)


if __name__ == "__main__":  # pragma: no cover - use `python -m scripts.acceptance.journey_main`.
    sys.exit("use python -m scripts.acceptance.journey_main")
