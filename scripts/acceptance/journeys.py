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
import hashlib
import json
import os
import re
import secrets
import select
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import JsonValue

from synth_ai import auth
from synth_ai._auth_models import ClientRegistration, OAuthToken
from synth_ai.errors import SynthError

from .isolation import classify
from .j1 import AdapterError
from .oauth_journey import HostedOAuth, OAuthJourneyError, approve, response_object
from .session_files import RefreshingSurface, SessionRefused
from .surfaces import (
    CLI_ARGV,
    CLI_ARGV_TOO_LONG,
    SDK_CALLS,
    Call,
    CLISurface,
    MCPSurface,
    Outcome,
    SDKSurface,
    Surface,
    SurfaceName,
    catalog,
    comparable,
    obj,
)

__all__ = [
    "JourneyContext",
    "Step",
    "StepReceipt",
    "Verdict",
    "capabilities",
    "matrix_operations",
    "run_step",
]

ROOT = Path(__file__).parents[2]
PROBE_CHARGE = "0.50"
POLL_SECONDS = 2.0
STATE_SECONDS = 900.0
MEMBER_ROLES = frozenset({"member"})
ADMIN_ROLES = frozenset({"owner", "admin"})
OVERLOAD_CODES = frozenset({"overloaded", "rate_limit_exceeded", "run_queue_full"})
# RunSnapshot.state terminal values (contract 1.32 examples).
TERMINAL = frozenset({"completed", "partial", "failed", "cancelled"})
# A resumed run left `paused` and did not end unsuccessfully.
RESUMED = frozenset({"running", "completed", "partial"})
# SYN-5744 contract publication is required; missing routes remain unproven.
REQUIRED_MESSAGE_OPERATIONS = frozenset({"sendRunMessage", "listRunMessages", "getRunMessage"})
# Shipped contract versions that do not publish the W16 message routes (coordinator ruling,
# run 5 T2, 2026-11-18): their absence is typed `not_in_release:<version>`, never a pass and
# never uncovered. Any other version without them stays `uncovered:contract_operation_missing`.
RELEASES_WITHOUT_MESSAGE_ROUTES = frozenset({"1.35.1"})

ID_PARAMETERS = frozenset(
    {
        "run_id",
        "output_id",
        "file_id",
        "upload_id",
        "export_id",
        "id",
        "request_id",
        "revision_id",
        "key_id",
        "grant_id",
    }
)


class Verdict(StrEnum):
    PASS = "Pass"
    FAIL = "Fail"
    INCONCLUSIVE = "Inconclusive"


class Step(StrEnum):
    """Same values as synth-dev `journeys.Step`; the orchestrator checks both."""

    BUY_CREDITS_TEST = "buy_credits_test_mode"
    RAISE_LIMIT = "raise_limit"
    SKU_REFUSAL = "off_whitelist_sku_refusal"
    PAUSE_LIMIT = "pause_on_limit"
    RESUME = "resume"
    STATEMENT = "final_statement"
    MINT_KEY = "mint_key"
    ROTATE_KEY = "rotate_key"
    REVOKE_KEY = "revoke_key"
    CLI_OAUTH = "cli_oauth_pkce"
    MCP_CONSENT = "mcp_oauth_consent"
    MEMBER_ADMIN = "member_vs_admin"
    OVERSIZE = "oversize_refusal"
    TOPOLOGY = "unknown_topology_refusal"
    TOPOLOGY_VERSION = "unknown_topology_version_refusal"
    INDEX = "index_refusal"
    EXPIRED_KEY = "expired_key_refusal"
    OVERLOAD_REFUSAL = "retryable_overload_refusal"
    OVERLOAD_RECOVERY = "overload_recovery"
    RATE_LIMIT_REFUSAL = "retryable_rate_limit_refusal"
    QUEUED_LAUNCH = "queued_launch"
    CROSS_ORG_404 = "cross_org_not_found_matrix"
    MESSAGE_QUESTION = "run_message_question"
    MESSAGE_TERMINAL = "run_message_terminal_refusal"
    MESSAGE_OVERSIZE = "run_message_oversize_refusal"
    MESSAGE_RATE = "run_message_rate_refusal"
    MESSAGE_ISOLATION = "run_message_isolation"


@dataclass(frozen=True, slots=True)
class StepReceipt:
    step: Step
    driver: SurfaceName
    verdict: Verdict
    cause: str
    evidence: tuple[Mapping[str, JsonValue], ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "step": self.step.value,
            "driver": self.driver.value,
            "verdict": self.verdict.value,
            "cause": self.cause,
            "evidence": [dict(item) for item in self.evidence],
        }


SurfaceFactory = Callable[[str], AbstractContextManager[Surface]]


@dataclass(slots=True)
class JourneyContext:
    """One driver's inputs. Secrets come from the environment and never print."""

    surface: Surface
    driver: SurfaceName
    origin: str
    make: SurfaceFactory
    receipt_dir: Path
    run_id: Optional[str] = None
    limit_run_id: Optional[str] = None
    queued_journal: Optional[Path] = None
    foreign_ids: Mapping[str, str] = field(default_factory=dict)
    isolation_operations: Optional[frozenset[str]] = None
    isolation_contract: Optional[dict[str, Any]] = None
    isolation_bodies: Mapping[str, Mapping[str, JsonValue]] = field(default_factory=dict)
    secrets: Mapping[str, str] = field(default_factory=dict)
    mcp_url: Optional[str] = None
    state: dict[str, str] = field(default_factory=dict)
    message_fixture: Optional[Path] = None
    message_recording_sha256: Optional[str] = None
    message_fixture_sha256: Optional[str] = None
    # Completes a hosted Stripe TEST checkout (URL in, None or a typed cause out).
    # Hosted staging wires stripe_checkout.complete_test_checkout; a slot leaves it
    # unset and buy_credits stays Inconclusive.
    complete_checkout: Optional[Callable[[str], Optional[str]]] = None


class Secret(StrEnum):
    """Environment names a harness supplies; each one is a named precondition."""

    SESSION = "SYNTH_JOURNEY_SESSION_TOKEN"
    MEMBER = "SYNTH_JOURNEY_MEMBER_API_KEY"
    EXPIRED = "SYNTH_JOURNEY_EXPIRED_API_KEY"
    FOREIGN = "SYNTH_JOURNEY_FOREIGN_API_KEY"
    # Org B's Clerk session, for contract operations that accept only `clerkSession`.
    FOREIGN_SESSION = "SYNTH_JOURNEY_FOREIGN_SESSION_TOKEN"
    # Hosted overload precondition: "admission_queue_full" names an org whose
    # admission queue the harness filled (running cap 0, org depth full).
    OVERLOAD_MODE = "SYNTH_JOURNEY_OVERLOAD_MODE"
    # That org's customer key; absent, the primary key's org is the saturated one.
    OVERLOAD_API_KEY = "SYNTH_JOURNEY_OVERLOAD_API_KEY"


def receipt(
    context: JourneyContext,
    step: Step,
    verdict: Verdict,
    cause: str,
    *outcomes: Outcome,
    notes: Optional[Mapping[str, JsonValue]] = None,
) -> StepReceipt:
    evidence: list[Mapping[str, JsonValue]] = [outcome.evidence() for outcome in outcomes]
    if notes is not None:
        evidence.append(dict(notes))
    return StepReceipt(step, context.driver, verdict, cause, tuple(evidence))


def missing(context: JourneyContext, step: Step, precondition: str) -> StepReceipt:
    return receipt(context, step, Verdict.INCONCLUSIVE, "precondition_missing:" + precondition)


def key(prefix: str) -> str:
    return f"syn5695-{prefix}-{uuid.uuid4()}"


def expect_refusal(
    context: JourneyContext,
    step: Step,
    outcome: Outcome,
    *,
    codes: frozenset[str],
    status: Optional[int] = None,
) -> StepReceipt:
    if outcome.accepted:
        return receipt(context, step, Verdict.FAIL, "refusal_expected_but_accepted", outcome)
    if not outcome.typed:
        return receipt(context, step, Verdict.FAIL, "refusal_untyped", outcome)
    if outcome.code not in codes or (status is not None and outcome.status != status):
        return receipt(context, step, Verdict.FAIL, "refusal_code_unexpected", outcome)
    return receipt(context, step, Verdict.PASS, "", outcome)


def probe_launch(
    context: JourneyContext, step: Step, body: dict[str, JsonValue], codes: frozenset[str]
) -> StepReceipt:
    outcome = context.surface.call(Call("createRun", body=body, idempotency_key=key(step)))
    if outcome.accepted:
        # An accepted probe is a defect; stop the run before it spends more.
        run = outcome.value.get("run_id") if isinstance(outcome.value, dict) else None
        cancelled = (
            context.surface.call(
                Call("cancelRun", path={"run_id": str(run)}, idempotency_key=key("cancel"))
            )
            if isinstance(run, str)
            else Outcome("cancelRun", False, cause="accepted run id absent")
        )
        return receipt(
            context, step, Verdict.FAIL, "refusal_expected_but_accepted", outcome, cancelled
        )
    return expect_refusal(context, step, outcome, codes=codes)


def launch_body(message: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "message": message,
        "limits": {"maximum_charge": {"amount": PROBE_CHARGE, "currency": "USD"}},
        **extra,
    }


# --- J2 money --------------------------------------------------------------


CHECKOUT_COMPLETION_WAIT_SECONDS = 60.0
CHECKOUT_POLL_SECONDS = 2.0
TERMINAL_CHECKOUT = frozenset({"completed", "failed", "expired"})


def checkout_state(outcome: Outcome) -> Optional[str]:
    state = outcome.value.get("state") if isinstance(outcome.value, dict) else None
    return state if isinstance(state, str) else None


def buy_credits(context: JourneyContext) -> StepReceipt:
    step = Step.BUY_CREDITS_TEST
    before = context.surface.call(Call("getBalance"))
    created = context.surface.call(
        Call("createCheckout", body={"pack_id": "credits.10"}, idempotency_key=key("checkout"))
    )
    if not created.accepted or not isinstance(created.value, dict):
        return receipt(context, step, Verdict.FAIL, "checkout_not_created", before, created)
    checkout = Call("getCheckout", path={"id": str(created.value["checkout_id"])})
    url = created.value.get("url")
    if context.complete_checkout is None or not isinstance(url, str):
        status = context.surface.call(checkout)
        state = checkout_state(status)
        cause = "checkout_not_completed:stripe_test_completion_unavailable"
    else:
        failure = context.complete_checkout(url)
        # The webhook lands after Stripe's redirect; poll a bounded window for it.
        deadline = time.monotonic() + CHECKOUT_COMPLETION_WAIT_SECONDS
        while True:
            status = context.surface.call(checkout)
            state = checkout_state(status)
            if failure is not None or state in TERMINAL_CHECKOUT or time.monotonic() >= deadline:
                break
            time.sleep(CHECKOUT_POLL_SECONDS)
        cause = "checkout_not_completed:" + (failure or "webhook_not_observed")
    if state != "completed":
        # Never assume credit: completion is Core's webhook-confirmed state.
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            cause,
            before,
            created,
            status,
            notes={"checkout_state": state},
        )
    after = context.surface.call(Call("getBalance"))
    gained = ledger(after) - ledger(before)
    verdict = Verdict.PASS if gained == Decimal("10") else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "balance_not_credited",
        before,
        created,
        status,
        after,
        notes={"ledger_gain_usd": str(gained)},
    )


def items(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def ledger(outcome: Outcome) -> Decimal:
    value = outcome.value
    if not outcome.accepted or not isinstance(value, dict):
        return Decimal("NaN")
    money = value.get("ledger")
    return Decimal(str(money["amount"])) if isinstance(money, dict) else Decimal("NaN")


def spend_cap(view: Outcome) -> Optional[int]:
    if not isinstance(view.value, dict):
        return None
    for row in items(view.value.get("primary")):
        if isinstance(row, dict) and row.get("dimension") == "spend_micros":
            cap = row.get("effective_cap")
            return cap if isinstance(cap, int) else None
    return None


def raise_limit(context: JourneyContext) -> StepReceipt:
    step = Step.RAISE_LIMIT
    if context.run_id is None:
        return missing(context, step, "running_run")
    if isinstance(context.surface, SDKSurface):
        return raise_four(context)
    path = {"run_id": context.run_id}
    before = context.surface.call(Call("getRunLimits", path=path))
    if before.accepted and obj(before.value).get("status") != "running":
        # The batch's run ended before this driver's turn: not this driver's fault.
        return receipt(
            context, step, Verdict.INCONCLUSIVE, "precondition_not_met:run_not_running", before
        )
    cap = spend_cap(before)
    if cap is None or not isinstance(before.value, dict):
        return receipt(context, step, Verdict.FAIL, "budget_view_unavailable", before)
    target = Decimal(cap * 2) / Decimal(1_000_000)
    body: dict[str, JsonValue] = {
        "expected_revision": before.value["revision"],
        "limits": {"maximum_charge": {"amount": str(target), "currency": "USD"}},
        "reason": "SYN-5695 J2 raise",
    }
    updated = context.surface.call(
        Call("updateRunLimits", path=path, body=body, idempotency_key=key("raise"))
    )
    after = context.surface.call(Call("getRunLimits", path=path))
    observed = spend_cap(after)
    verdict = Verdict.PASS if updated.accepted and observed == cap * 2 else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "limit_raise_not_observed",
        before,
        updated,
        after,
        notes={"spend_cap_before": cap, "spend_cap_after": observed},
    )


def raise_four(context: JourneyContext) -> StepReceipt:
    """SDK: the merged SYN-5663 helper, four raises each seen in the next view."""
    from .live_limits import (
        Dimension,
        LimitTargets,
        LiveLimitsError,
        caps,
        contract_maximum,
        raise_live_limits,
    )

    assert isinstance(context.surface, SDKSurface) and context.run_id is not None
    client = context.surface.client
    try:
        current = caps(client.swarms.limits(context.run_id), context.run_id)
    except LiveLimitsError as error:
        verdict = Verdict.INCONCLUSIVE if error.code == "run_not_running" else Verdict.FAIL
        cause = (
            "precondition_not_met:run_not_running"
            if verdict == Verdict.INCONCLUSIVE
            else error.code
        )
        return receipt(context, Step.RAISE_LIMIT, verdict, cause)

    def seconds_target(dimension: Dimension, field: str) -> Optional[int]:
        # Double, but never past the contract maximum; None (held, asserted
        # unchanged) only when the cap already sits at that maximum.
        maximum = contract_maximum(field)
        current_seconds = current[dimension] // 1000
        return None if current_seconds >= maximum else min(current_seconds * 2, maximum)

    targets = LimitTargets(
        budget_usd=str(Decimal(current[Dimension.SPEND] * 2) / Decimal(1_000_000)),
        maximum_tokens=current[Dimension.TOKENS] * 2,
        wall_time_seconds=seconds_target(Dimension.EXECUTION, "wall_time_seconds"),
        pause_retention_seconds=seconds_target(Dimension.RETENTION, "pause_retention_seconds"),
    )
    directory = context.receipt_dir / "live-limits"
    try:
        views = raise_live_limits(
            client,
            run_id=context.run_id,
            targets=targets,
            mutation_prefix=key("live-limits"),
            receipt_directory=directory,
        )
    except LiveLimitsError as error:
        return receipt(context, Step.RAISE_LIMIT, Verdict.FAIL, error.code)
    return receipt(
        context,
        Step.RAISE_LIMIT,
        Verdict.PASS,
        "",
        notes={
            "helper": "scripts/acceptance/live_limits.py",
            "receipt_directory": str(directory),
            "revisions": [view.revision for view in views],
            "held_at_contract_maximum": [dimension.value for dimension in targets.held()],
        },
    )


def sku_refusal(context: JourneyContext) -> StepReceipt:
    # Reachable whitelist refusal on a cached slot: fork_flush is a public
    # topology the release environment does not admit (run_profile_deferred).
    body = launch_body(
        "SYN-5695 whitelist probe", optimizer={"topology": "fork_flush", "version": 1}
    )
    return probe_launch(
        context, Step.SKU_REFUSAL, body, frozenset({"optimizer_topology_not_allowed"})
    )


def wait_state(
    context: JourneyContext, run_id: str, accept: Callable[[dict[str, JsonValue]], bool]
) -> tuple[Outcome, list[str]]:
    seen: list[str] = []
    deadline = time.monotonic() + STATE_SECONDS
    while True:
        outcome = context.surface.call(Call("getRun", path={"run_id": run_id}))
        value = outcome.value if isinstance(outcome.value, dict) else {}
        state = str(value.get("state"))
        if not seen or seen[-1] != state:
            seen.append(state)
        if not outcome.accepted or accept(value) or time.monotonic() > deadline:
            return outcome, seen
        time.sleep(POLL_SECONDS)


def limit_paused(value: Mapping[str, JsonValue]) -> bool:
    pause = value.get("pause")
    return (
        value.get("state") == "paused" and isinstance(pause, dict) and pause.get("cause") == "limit"
    )


def pause_on_limit(context: JourneyContext) -> StepReceipt:
    step = Step.PAUSE_LIMIT
    if context.limit_run_id is None:
        return missing(context, step, "limit_run")
    outcome, seen = wait_state(
        context, context.limit_run_id, lambda v: limit_paused(v) or v.get("state") in TERMINAL
    )
    value = outcome.value if isinstance(outcome.value, dict) else {}
    limits = context.surface.call(Call("getRunLimits", path={"run_id": context.limit_run_id}))
    verdict = Verdict.PASS if limit_paused(value) and limits.accepted else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "run_not_paused_by_limit",
        outcome,
        limits,
        notes={"states": list(seen), "pause": value.get("pause")},
    )


def resume(context: JourneyContext) -> StepReceipt:
    step = Step.RESUME
    if context.limit_run_id is None:
        return missing(context, step, "limit_run")
    path = {"run_id": context.limit_run_id}
    view = context.surface.call(Call("getRunLimits", path=path))
    cap = spend_cap(view)
    if cap is None or not isinstance(view.value, dict):
        return receipt(context, step, Verdict.FAIL, "budget_view_unavailable", view)
    body: dict[str, JsonValue] = {
        "expected_revision": view.value["revision"],
        "limits": {
            "maximum_charge": {"amount": str(Decimal(cap * 2) / 1_000_000), "currency": "USD"}
        },
        "resume": True,
        "reason": "SYN-5695 J2 resume after limit",
    }
    updated = context.surface.call(
        Call("updateRunLimits", path=path, body=body, idempotency_key=key("resume"))
    )
    outcome, seen = wait_state(context, context.limit_run_id, lambda v: v.get("state") != "paused")
    state = outcome.value.get("state") if isinstance(outcome.value, dict) else None
    verdict = Verdict.PASS if updated.accepted and state in RESUMED else Verdict.FAIL
    return receipt(
        context,
        step,
        verdict,
        "" if verdict == Verdict.PASS else "run_not_resumed",
        view,
        updated,
        outcome,
        notes={"states": list(seen)},
    )


def statement(context: JourneyContext) -> StepReceipt:
    page = context.surface.call(Call("listStatements"))
    balance = context.surface.call(Call("getBalance"))
    rows = items(obj(page.value).get("items"))
    month = datetime.now(UTC).strftime("%Y-%m")
    current = [item for item in rows if isinstance(item, dict) and item.get("month") == month]
    verdict = Verdict.PASS if current and balance.accepted else Verdict.FAIL
    return receipt(
        context,
        Step.STATEMENT,
        verdict,
        "" if verdict == Verdict.PASS else "current_statement_absent",
        page,
        balance,
        notes={"month": month, "statements": len(rows)},
    )


# --- J3 access ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JourneyPrincipal:
    org_id: str
    user_id: str
    role: Optional[str]

    @classmethod
    def from_account(cls, outcome: Outcome) -> Optional[JourneyPrincipal]:
        account = obj(outcome.value)
        principal = obj(account.get("principal"))
        organization = principal.get("org_id")
        user = principal.get("user_id")
        if not outcome.accepted or not isinstance(organization, str) or not organization:
            return None
        if (
            obj(account.get("org")).get("id") != organization
            or not isinstance(user, str)
            or not user
        ):
            return None
        role = principal.get("org_role")
        return cls(organization, user, role if isinstance(role, str) else None)


def credential_works(context: JourneyContext, secret: str) -> Outcome:
    with context.make(secret) as surface:
        return surface.call(Call("getAccount"))


API_KEY_INVALID = "api_key_invalid"


def api_key_refused(outcome: Outcome) -> bool:
    return not outcome.accepted and outcome.status == 401 and outcome.code == API_KEY_INVALID


def mint_key(context: JourneyContext) -> StepReceipt:
    step = Step.MINT_KEY
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    body: dict[str, JsonValue] = {
        "purpose": "automation",
        "name": "syn5695-" + secrets.token_hex(4),
        "scopes": ["account:read"],
        "expires_at": expires,
        "request_id": str(uuid.uuid4()),
    }
    with context.make(session) as surface:
        minted = surface.call(Call("createApiKey", body=body, idempotency_key=key("mint")))
    if not minted.accepted or not isinstance(minted.value, dict):
        return receipt(context, step, Verdict.FAIL, "key_not_minted", minted)
    context.state["key_id"] = str(minted.value["key_id"])
    context.state["secret"] = str(minted.value["key"])
    used = credential_works(context, context.state["secret"])
    verdict = Verdict.PASS if used.accepted else Verdict.FAIL
    return receipt(
        context, step, verdict, "" if used.accepted else "minted_key_refused", minted, used
    )


def rotate_key(context: JourneyContext) -> StepReceipt:
    step = Step.ROTATE_KEY
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if "key_id" not in context.state:
        return missing(context, step, "minted_key")
    with context.make(session) as surface:
        rotated = surface.call(
            Call(
                "rotateApiKey",
                path={"key_id": context.state["key_id"]},
                idempotency_key=key("rotate"),
            )
        )
    if not rotated.accepted or not isinstance(rotated.value, dict):
        return receipt(context, step, Verdict.FAIL, "key_not_rotated", rotated)
    old = credential_works(context, context.state["secret"])
    context.state["key_id"] = str(rotated.value["key_id"])
    context.state["secret"] = str(rotated.value["key"])
    new = credential_works(context, context.state["secret"])
    ok = new.accepted and api_key_refused(old)
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "rotation_not_enforced",
        rotated,
        old,
        new,
    )


def revoke_key(context: JourneyContext) -> StepReceipt:
    step = Step.REVOKE_KEY
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if "key_id" not in context.state:
        return missing(context, step, "minted_key")
    with context.make(session) as surface:
        revoked = surface.call(Call("revokeApiKey", path={"key_id": context.state["key_id"]}))
    after = credential_works(context, context.state["secret"])
    ok = revoked.accepted and api_key_refused(after)
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "revocation_not_enforced",
        revoked,
        after,
    )


@contextmanager
def private_auth_file(context: JourneyContext) -> Generator[Path]:
    path = context.receipt_dir / "oauth-credentials.json"
    previous = os.environ.get("SYNTH_AUTH_FILE")
    os.environ["SYNTH_AUTH_FILE"] = str(path)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)  # Live OAuth tokens never stay in receipts.
        if previous is None:
            os.environ.pop("SYNTH_AUTH_FILE", None)
        else:
            os.environ["SYNTH_AUTH_FILE"] = previous


def oauth_failure(
    context: JourneyContext, step: Step, error: Exception, notes: dict[str, JsonValue]
) -> StepReceipt:
    if isinstance(error, OAuthJourneyError):
        stage, code = error.stage, error.code
    elif isinstance(error, httpx.HTTPStatusError):
        operation = error.request.url.path.rsplit("/", 1)[-1]
        stage = (
            operation
            if operation in {"register", "token", "revoke", "oauth-authorization-server"}
            else "oauth"
        )
        notes[f"{stage}_status"] = error.response.status_code
        code = "http_refused"
    else:
        stage, code = "oauth", type(error).__name__
    notes["oauth_failure_stage"] = stage
    notes["oauth_failure_code"] = code
    return receipt(context, step, Verdict.FAIL, f"oauth_failed:{stage}:{code}", notes=notes)


def cli_oauth(context: JourneyContext) -> StepReceipt:
    step = Step.CLI_OAUTH
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    notes: dict[str, JsonValue] = {}
    oauth = auth.oauth_base(context.origin.rstrip("/"))

    def consent(url: str) -> str:
        return approve(oauth, session, url, oauth + "/mcp", notes)

    try:
        with private_auth_file(context) as path:
            if context.driver == SurfaceName.SDK:
                auth.login(context.origin, open_browser=False, redirect_source=consent)
            else:
                cli_login(context, path, consent, notes)
            record = auth.read_tokens().get(context.origin.rstrip("/"))
            if record is None:
                raise OAuthJourneyError("token_store", "oauth_credential_missing")
            try:
                used = credential_works(context, record.access_token)
            finally:
                auth.logout(context.origin)
                notes["logout_acknowledged"] = True
                after = credential_works(context, record.access_token)
                notes["revoked_token_verified"] = (
                    not after.accepted
                    and after.status == 401
                    and after.code == "oauth_token_invalid"
                )
    except (
        OAuthJourneyError,
        AdapterError,
        SynthError,
        httpx.HTTPError,
        ValueError,
        OSError,
    ) as error:
        return oauth_failure(context, step, error, notes)
    kind = obj(obj(used.value).get("principal")).get("credential_kind")
    notes["credential_kind"] = kind
    cause = ""
    if not used.accepted or kind != "oauth":
        cause = "oauth_token_refused"
    elif notes["revoked_token_verified"] is not True:
        cause = "oauth_token_revocation_unproven"
    return receipt(
        context,
        step,
        Verdict.FAIL if cause else Verdict.PASS,
        cause,
        used,
        after,
        notes=notes,
    )


def cli_login(
    context: JourneyContext,
    path: Path,
    consent: Callable[[str], str],
    notes: dict[str, JsonValue],
) -> None:
    surface = context.surface
    assert isinstance(surface, CLISurface)
    environment = {
        key_: value for key_, value in surface.environment.items() if key_ != "SYNTH_API_KEY"
    }
    environment["SYNTH_AUTH_FILE"] = str(path)
    with subprocess.Popen(
        [
            str(surface.python),
            "-m",
            "synth_ai.cli",
            "--base-url",
            context.origin,
            "login",
            "--no-browser",
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stderr is not None
        try:
            readable, _, _ = select.select([child.stderr], [], [], 65)
            if not readable:
                raise OAuthJourneyError("cli_login", "cli_authorize_timeout")
            authorization_url = child.stderr.readline().strip()
            if not authorization_url:
                raise OAuthJourneyError("cli_login", "cli_authorize_url_missing")
            callback = consent(authorization_url)
            child.communicate(callback + "\n", timeout=120)
            notes["cli_login_exit"] = child.returncode
            if child.returncode != 0:
                raise OAuthJourneyError("cli_login", "cli_token_exchange_failed")
        except subprocess.TimeoutExpired as error:
            raise OAuthJourneyError("cli_login", "cli_exchange_timeout") from error
        finally:
            if child.poll() is None:
                child.kill()


def hosted_token(
    client: httpx.Client, oauth: HostedOAuth, session: str, notes: dict[str, JsonValue]
) -> tuple[str, str]:
    redirect_uri = "http://127.0.0.1:9/callback"
    registration = response_object(
        client.post(
            oauth.endpoint("registration_endpoint"),
            json={
                "client_name": "SYN-5695 journey",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        ),
        "register",
        201,
        notes,
    )
    client_id = ClientRegistration.model_validate(registration).client_id
    state = secrets.token_urlsafe(16)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    authorization = str(
        httpx.URL(
            oauth.endpoint("authorization_endpoint"),
            params={
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "resource": oauth.resource,
                # Core refuses any authorize without the required scope
                # (publicpolicy.RequiredScopes: user:org:read, alias of account:read).
                "scope": "user:org:read",
            },
        )
    )
    callback = approve(oauth.origin, session, authorization, oauth.resource, notes)
    result = auth.callback_result(callback, state=state, issuer=oauth.origin)
    if (
        urlparse(callback)._replace(query="") != urlparse(redirect_uri)
        or result is None
        or "code" not in result
        or "error" in result
    ):
        raise OAuthJourneyError("callback", "oauth_callback_mismatch")
    token = response_object(
        client.post(
            oauth.endpoint("token_endpoint"),
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": result["code"],
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": oauth.resource,
            },
        ),
        "token",
        200,
        notes,
    )
    return client_id, OAuthToken.model_validate(token).access_token


def owned_grant(
    client: httpx.Client,
    grants_url: str,
    clerk: dict[str, str],
    client_id: str,
    notes: dict[str, JsonValue],
) -> str:
    grants = response_object(client.get(grants_url, headers=clerk), "grants", 200, notes)
    matches = [
        obj(grant)
        for grant in items(grants.get("grants"))
        if obj(grant).get("client_id") == client_id
    ]
    if len(matches) != 1:
        raise OAuthJourneyError("grants", "oauth_grant_not_unique")
    grant = matches[0]
    identity = grant.get("grant_id")
    if (
        grant.get("status") != "active"
        or grant.get("owned_by_viewer") is not True
        or not isinstance(identity, str)
        or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity) is None
    ):
        raise OAuthJourneyError("grants", "oauth_grant_identity_invalid")
    notes["grant_bound"] = True
    return identity


def mcp_consent(context: JourneyContext) -> StepReceipt:
    """Exercise the hosted MCP OAuth grant through discovery, tool use and revoke."""
    step = Step.MCP_CONSENT
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if context.mcp_url is None:
        return missing(context, step, "hosted_mcp_url")
    notes: dict[str, JsonValue] = {}
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            oauth = HostedOAuth.discover(client, context.mcp_url, notes)
            revocation_endpoint = oauth.endpoint("revocation_endpoint")
            client_id, access = hosted_token(client, oauth, session, notes)
            grants_url = oauth.origin + "/mcp/oauth/grants"
            clerk = {"X-Clerk-Session-Token": session}
            grant_revoked = False
            try:
                identity = owned_grant(client, grants_url, clerk, client_id, notes)
                try:
                    used = credential_works(context, access)
                finally:
                    revoked = response_object(
                        client.delete(grants_url + "/" + identity, headers=clerk),
                        "revoke",
                        200,
                        notes,
                    )
                    grant_revoked = (
                        revoked.get("grant_id") == identity
                        and revoked.get("status") == "revoked"
                        and revoked.get("outcome") == "revoked"
                    )
                    if not grant_revoked:
                        raise OAuthJourneyError("revoke", "oauth_grant_revocation_unproven")
            finally:
                if not grant_revoked:
                    oauth.cleanup_token(client, revocation_endpoint, client_id, access, notes)
            after = client.get(oauth.resource, headers={"Authorization": "Bearer " + access})
            refused = oauth.revoked(after, notes)
    except (
        OAuthJourneyError,
        AdapterError,
        SynthError,
        httpx.HTTPError,
        ValueError,
        OSError,
    ) as error:
        return oauth_failure(context, step, error, notes)
    kind = obj(obj(used.value).get("principal")).get("credential_kind")
    notes["credential_kind"] = kind
    ok = used.accepted and kind == "oauth" and refused
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "mcp_oauth_grant_lifecycle_failed",
        used,
        notes=notes,
    )


def member_admin(context: JourneyContext) -> StepReceipt:
    step = Step.MEMBER_ADMIN
    member = context.secrets.get(Secret.MEMBER)
    if member is None:
        return missing(context, step, "member_api_key")
    admin = context.surface.call(Call("getAccount"))
    with context.make(member) as surface:
        identity = surface.call(Call("getAccount"))
        administrator = JourneyPrincipal.from_account(admin)
        participant = JourneyPrincipal.from_account(identity)
        if administrator is None or participant is None:
            return missing(context, step, "user_bound_role_principals")
        roles = {"admin": administrator.role, "member": participant.role}
        if administrator.role not in ADMIN_ROLES or participant.role not in MEMBER_ROLES:
            return receipt(
                context, step, Verdict.FAIL, "roles_unexpected", admin, identity, notes=roles
            )
        if (
            administrator.org_id != participant.org_id
            or administrator.user_id == participant.user_id
        ):
            return receipt(
                context,
                step,
                Verdict.FAIL,
                "role_principals_mismatch",
                admin,
                identity,
                notes=roles,
            )
        roles.update(
            admin_user_id=administrator.user_id,
            member_user_id=participant.user_id,
            org_id=participant.org_id,
        )
        refused = surface.call(Call("cancelSubscription", idempotency_key=key("member-cancel")))
    result = expect_refusal(
        context, step, refused, codes=frozenset({"billing_admin_required"}), status=403
    )
    return StepReceipt(
        step,
        context.driver,
        result.verdict,
        result.cause,
        (admin.evidence(), identity.evidence(), *result.evidence, roles),
    )


# --- J-refusals --------------------------------------------------------------


CLIENT_REFUSALS = frozenset(
    {"launch_request_too_large", "invalid_launch_request", "client_error", "tool_arguments_invalid"}
)


def oversize(context: JourneyContext) -> StepReceipt:
    body = launch_body("x" * (1_048_576 + 1))
    if isinstance(context.surface, CLISurface):
        outcome = context.surface.call(Call("createRun", body=body, idempotency_key=key("cli")))
        if outcome.cause == CLI_ARGV_TOO_LONG:
            return receipt(
                context,
                Step.OVERSIZE,
                Verdict.INCONCLUSIVE,
                "capability_missing:cli_goal_only_argv",
                outcome,
            )
        return expect_refusal(context, Step.OVERSIZE, outcome, codes=CLIENT_REFUSALS)
    return probe_launch(
        context,
        Step.OVERSIZE,
        body,
        frozenset(
            {
                "launch_request_too_large",
                "invalid_launch_request",
                "client_error",
                "tool_arguments_invalid",
            }
        ),
    )


def topology(context: JourneyContext) -> StepReceipt:
    body = launch_body(
        "SYN-5695 topology probe", optimizer={"topology": "no_such_topology", "version": 1}
    )
    return probe_launch(
        context,
        Step.TOPOLOGY,
        body,
        frozenset(
            {
                "optimizer_topology_unavailable",
                "invalid_launch_request",
                "client_error",
                "tool_arguments_invalid",
            }
        ),
    )


def topology_version(context: JourneyContext) -> StepReceipt:
    body = launch_body(
        "SYN-5695 version probe", optimizer={"topology": "swarms_v2", "version": 999}
    )
    return probe_launch(
        context,
        Step.TOPOLOGY_VERSION,
        body,
        frozenset({"optimizer_version_unavailable", "optimizer_topology_unavailable"}),
    )


def index(context: JourneyContext) -> StepReceipt:
    """Index is not a Swarms surface; each client must refuse it readably."""
    step = Step.INDEX
    if context.driver == SurfaceName.SDK:
        # The SDK has no Index method; the route itself must refuse with a Problem.
        key_ = context.secrets.get("SYNTH_API_KEY", "")
        with httpx.Client(timeout=30) as client:
            response = client.get(
                context.origin + "/v1/index/search",
                params={"q": "syn5695"},
                headers={"Authorization": "Bearer " + key_},
            )
        try:
            problem = response.json()
        except ValueError:
            problem = {}
        code = problem.get("code") if isinstance(problem, dict) else None
        outcome = Outcome(
            "index_search",
            response.is_success,
            status=response.status_code,
            code=code if isinstance(code, str) else None,
            problem=comparable(problem) if isinstance(problem, dict) else None,
        )
        return expect_refusal(
            context, step, outcome, codes=frozenset({"index_not_available"}), status=404
        )
    if isinstance(context.surface, CLISurface):
        outcome = context.surface.run("index_search", ["index", "search", "syn5695"])
        return expect_refusal(context, step, outcome, codes=frozenset({"index_not_available"}))
    assert isinstance(context.surface, MCPSurface)
    listed = "index_search" in context.surface.available
    outcome = context.surface.caller.call("index_search", {"query": "syn5695"})
    ok = outcome.is_error and not listed
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "index_tool_answered",
        notes={"listed": listed, "is_error": outcome.is_error},
    )


def expired_key(context: JourneyContext) -> StepReceipt:
    expired = context.secrets.get(Secret.EXPIRED)
    if expired is None:
        return missing(context, Step.EXPIRED_KEY, "expired_api_key")
    outcome = credential_works(context, expired)
    return expect_refusal(
        context, Step.EXPIRED_KEY, outcome, codes=frozenset({API_KEY_INVALID}), status=401
    )


ADMISSION_QUEUE_FULL = "admission_queue_full"
ADMISSION_PROBE_CHARGE = "2.50"


def admission_overload_refusal(context: JourneyContext) -> StepReceipt:
    """Hosted overload: the org's admission queue is full (running cap 0 and
    every waiting place taken), so a valid launch must be refused 429
    run_queue_full with retry advice and no accepted effect. The org launch
    window's 429 rate_limit_exceeded is a different refusal and does not pass."""
    step = Step.OVERLOAD_REFUSAL
    # A valid budget (the J1 producer default) so Core reaches admission;
    # PROBE_CHARGE is refused run_budget_below_call_bound before the queue.
    body: dict[str, JsonValue] = {
        "message": "SYN-5696 hosted overload probe (refusal expected)",
        "limits": {"maximum_charge": {"amount": ADMISSION_PROBE_CHARGE, "currency": "USD"}},
    }
    saturated = context.secrets.get(Secret.OVERLOAD_API_KEY)
    with ExitStack() as stack:
        surface = (
            context.surface if saturated is None else stack.enter_context(context.make(saturated))
        )
        outcome = surface.call(Call("createRun", body=body, idempotency_key=key("overload")))
        if outcome.accepted:
            run = outcome.value.get("run_id") if isinstance(outcome.value, dict) else None
            cancelled = (
                surface.call(
                    Call("cancelRun", path={"run_id": str(run)}, idempotency_key=key("cancel"))
                )
                if isinstance(run, str)
                else Outcome("cancelRun", False, cause="accepted run id absent")
            )
            return receipt(
                context, step, Verdict.FAIL, "refusal_expected_but_accepted", outcome, cancelled
            )
    result = expect_refusal(
        context, step, outcome, codes=frozenset({"run_queue_full"}), status=None
    )
    if result.verdict != Verdict.PASS:
        return result
    if outcome.status not in (None, 429):
        return receipt(context, step, Verdict.FAIL, "refusal_status_unexpected", outcome)
    if outcome.retryable is not True and outcome.retry_after is None:
        return receipt(context, step, Verdict.FAIL, "retry_advice_absent", outcome)
    return result


def overload_refusal(context: JourneyContext) -> StepReceipt:
    if context.secrets.get(Secret.OVERLOAD_MODE) == ADMISSION_QUEUE_FULL:
        return admission_overload_refusal(context)
    # The orchestrator holds the inbound bulkhead for longer than the client's
    # retry budget, so the client must surface a typed retryable refusal.
    outcome = context.surface.call(Call("getAccount"))
    result = expect_refusal(context, Step.OVERLOAD_REFUSAL, outcome, codes=OVERLOAD_CODES)
    if result.verdict == Verdict.PASS and (
        outcome.retry_after is None and outcome.retryable is not True
    ):
        return receipt(context, Step.OVERLOAD_REFUSAL, Verdict.FAIL, "retry_after_absent", outcome)
    return result


# Core's per-organization launch window (core public/operation_limits.go,
# organizationLaunchesPerMinute = 10). Every authenticated POST /v1/runs counts
# before the handler runs, so empty bodies (422 before any effect) fill it.
LAUNCH_WINDOW_FILL = 12
VERSION_REFUSALS = frozenset({"optimizer_version_unavailable", "optimizer_topology_unavailable"})


def rate_limit_refusal(context: JourneyContext) -> StepReceipt:
    """SYN-5723 retryable 429: fill the org launch window, then launch through the client.

    The probe launch reaches Core but is refused before any effect (unknown
    optimizer version), so a client that honours Retry-After and retries ends
    on that refusal no sooner than Retry-After; a client whose budget is shorter
    surfaces the typed retryable 429. Either is a Pass; an early retry is not.
    """
    step = Step.RATE_LIMIT_REFUSAL
    api_key = context.secrets.get("SYNTH_API_KEY", "")
    statuses: list[int] = []
    retry_after: Optional[int] = None
    with httpx.Client(timeout=30) as client:
        for _ in range(LAUNCH_WINDOW_FILL):
            response = client.post(
                context.origin + "/v1/runs",
                json={},
                headers={"Authorization": "Bearer " + api_key, "Idempotency-Key": key("fill")},
            )
            statuses.append(response.status_code)
            if response.status_code == 429:
                value = response.headers.get("Retry-After", "")
                retry_after = int(value) if value.isdigit() else None
                break
    notes: dict[str, JsonValue] = {"fill_statuses": list(statuses), "retry_after": retry_after}
    if statuses[-1:] != [429] or retry_after is None:
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:launch_window_not_saturated",
            notes=notes,
        )
    body = launch_body("SYN-5723 rate probe", optimizer={"topology": "swarms_v2", "version": 999})
    started = time.monotonic()
    outcome = context.surface.call(Call("createRun", body=body, idempotency_key=key("rate")))
    notes["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if outcome.accepted:
        # An accepted probe is a defect; stop it before it spends more.
        run = outcome.value.get("run_id") if isinstance(outcome.value, dict) else None
        cancelled = (
            context.surface.call(
                Call("cancelRun", path={"run_id": str(run)}, idempotency_key=key("cancel"))
            )
            if isinstance(run, str)
            else Outcome("cancelRun", False, cause="accepted run id absent")
        )
        return receipt(
            context,
            step,
            Verdict.FAIL,
            "refusal_expected_but_accepted",
            outcome,
            cancelled,
            notes=notes,
        )
    if outcome.code == "rate_limit_exceeded":
        ok = outcome.status == 429 and outcome.retryable is True and outcome.retry_after is not None
        return receipt(
            context,
            step,
            Verdict.PASS if ok else Verdict.FAIL,
            "" if ok else "rate_limit_not_retryable",
            outcome,
            notes=notes,
        )
    if outcome.code in VERSION_REFUSALS:
        # Retry-After was read before this call; the window cannot reopen sooner than one
        # second before it, so an earlier answer means the client did not wait.
        waited = time.monotonic() - started >= retry_after - 1
        return receipt(
            context,
            step,
            Verdict.PASS if waited else Verdict.FAIL,
            "" if waited else "retried_before_retry_after",
            outcome,
            notes=notes,
        )
    return receipt(context, step, Verdict.FAIL, "refusal_code_unexpected", outcome, notes=notes)


def overload_recovery(context: JourneyContext) -> StepReceipt:
    outcome = context.surface.call(Call("getAccount"))
    verdict = Verdict.PASS if outcome.accepted else Verdict.FAIL
    return receipt(
        context,
        Step.OVERLOAD_RECOVERY,
        verdict,
        "" if outcome.accepted else "not_recovered",
        outcome,
    )


def queued_launch(context: JourneyContext) -> StepReceipt:
    """Follow a run the cached harness launched (public-client-<kind>/launch-row.json)."""
    step = Step.QUEUED_LAUNCH
    if context.queued_journal is None:
        return missing(context, step, "cached_harness_queued_launch")
    row_path = context.queued_journal / "launch-row.json"
    if not row_path.is_file():
        return missing(context, step, "launch_row")
    row = obj(json.loads(row_path.read_text()))
    snapshot = obj(row.get("body"))
    run = row.get("run_id")
    queue = obj(snapshot.get("queue"))
    client = "mcp" if context.driver == SurfaceName.MCP_STDIO else context.driver.value
    if row.get("status") != 202 or row.get("client") != client:
        return receipt(context, step, Verdict.FAIL, "launch_row_client_or_status_mismatch")
    if context.run_id is None:
        return missing(context, step, "selected_queued_run")
    if (
        not isinstance(run, str)
        or not run
        or run != context.run_id
        or snapshot.get("run_id") != run
    ):
        return receipt(context, step, Verdict.FAIL, "launch_row_run_mismatch")
    if snapshot.get("state") != "queued" or not queue:
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:capacity_not_saturated",
            notes={"acknowledged_state": snapshot.get("state")},
        )
    outcome, seen = wait_state(
        context, run, lambda view: view.get("run_id") != run or view.get("state") in TERMINAL
    )
    if not outcome.accepted or obj(outcome.value).get("run_id") != run:
        return receipt(context, step, Verdict.FAIL, "queue_observation_run_mismatch", outcome)
    position = queue.get("position")
    notes: dict[str, JsonValue] = {"run_id": run, "queue": queue, "states": list(seen)}
    if type(position) is not int or position < 1:
        return receipt(context, step, Verdict.FAIL, "queue_position_invalid", outcome, notes=notes)
    estimate = queue.get("estimated_start_at")
    if not isinstance(estimate, str) or not estimate:
        return receipt(
            context, step, Verdict.INCONCLUSIVE, "queue_estimate_unobserved", outcome, notes=notes
        )
    if seen[-1:] and seen[-1] in TERMINAL and "running" in seen:
        return receipt(context, step, Verdict.PASS, "", outcome, notes=notes)
    if seen and seen[0] in TERMINAL:
        # This driver's turn came after the run ended; its progression is unseen.
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:run_already_terminal",
            outcome,
            notes=notes,
        )
    return receipt(
        context, step, Verdict.FAIL, "queued_progression_unobserved", outcome, notes=notes
    )


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
    try:
        uuid.UUID(foreign)
        return str(uuid.uuid4())
    except ValueError:
        pass
    match = ID_SHAPE.match(foreign)
    if match is None:
        return "absent" + secrets.token_hex(8)
    tail = match.group("tail")
    if len(tail) == 22:  # Core public IDs: prefix + unpadded base64url of a UUID.
        encoded = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")
        return str(match.group("prefix")) + encoded
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return str(match.group("prefix")) + "".join(secrets.choice(alphabet) for _ in tail)


MATRIX_BODIES: Mapping[str, Mapping[str, JsonValue]] = {
    "updateRunLimits": {"expected_revision": 0, "limits": {"wall_time_seconds": 60}},
    "sendRunMessage": {"kind": "guidance", "text": "Tenant-isolation fixture probe."},
    "updateRunLabels": {"expected_revision": 1, "labels": {}},
    "updateApiKey": {"expected_version": 1, "scopes": ["account:read"]},
}


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


def covered(surface: Surface | RefreshingSurface, operation: str) -> bool:
    if isinstance(surface, RefreshingSurface):
        surface = surface.surface()
    if isinstance(surface, MCPSurface):
        return operation in surface.mapped()
    if isinstance(surface, CLISurface):
        return operation in CLI_ARGV
    return operation in SDK_CALLS


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
