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
import json
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import yaml
from pydantic import JsonValue

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
    QUEUED_LAUNCH = "queued_launch"
    CROSS_ORG_404 = "cross_org_not_found_matrix"


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
    secrets: Mapping[str, str] = field(default_factory=dict)
    mcp_url: Optional[str] = None
    state: dict[str, str] = field(default_factory=dict)


class Secret(StrEnum):
    """Environment names a harness supplies; each one is a named precondition."""

    SESSION = "SYNTH_JOURNEY_SESSION_TOKEN"
    MEMBER = "SYNTH_JOURNEY_MEMBER_API_KEY"
    EXPIRED = "SYNTH_JOURNEY_EXPIRED_API_KEY"
    FOREIGN = "SYNTH_JOURNEY_FOREIGN_API_KEY"


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


def buy_credits(context: JourneyContext) -> StepReceipt:
    step = Step.BUY_CREDITS_TEST
    before = context.surface.call(Call("getBalance"))
    created = context.surface.call(
        Call("createCheckout", body={"pack_id": "credits.10"}, idempotency_key=key("checkout"))
    )
    if not created.accepted or not isinstance(created.value, dict):
        return receipt(context, step, Verdict.FAIL, "checkout_not_created", before, created)
    status = context.surface.call(
        Call("getCheckout", path={"id": str(created.value["checkout_id"])})
    )
    state = status.value.get("state") if isinstance(status.value, dict) else None
    if state != "completed":
        # Completion needs Stripe test mode to deliver checkout.session.completed
        # to this Core. The slot has no Stripe stand-in; never assume credit.
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "checkout_not_completed:stripe_test_completion_unavailable",
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
    from .live_limits import Dimension, LimitTargets, LiveLimitsError, caps, raise_live_limits

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
    targets = LimitTargets(
        budget_usd=str(Decimal(current[Dimension.SPEND] * 2) / Decimal(1_000_000)),
        maximum_tokens=current[Dimension.TOKENS] * 2,
        wall_time_seconds=current[Dimension.EXECUTION] * 2 // 1000,
        pause_retention_seconds=current[Dimension.RETENTION] * 2 // 1000,
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


def account_role(outcome: Outcome) -> Optional[str]:
    if not isinstance(outcome.value, dict):
        return None
    principal = outcome.value.get("principal")
    role = principal.get("org_role") if isinstance(principal, dict) else None
    return role if isinstance(role, str) else None


def credential_works(context: JourneyContext, secret: str) -> Outcome:
    with context.make(secret) as surface:
        return surface.call(Call("getAccount"))


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
    ok = new.accepted and not old.accepted and old.code == "authentication_required"
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
    ok = revoked.accepted and not after.accepted and after.code == "authentication_required"
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "revocation_not_enforced",
        revoked,
        after,
    )


def approve(origin: str, session: str, authorization_url: str) -> str:
    """Console consent through its public protocol: authorize -> request -> consent."""
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        redirect = client.get(authorization_url)
        if redirect.status_code != 303:
            raise RuntimeError(f"authorize answered {redirect.status_code}, not 303")
        request = parse_qs(urlparse(redirect.headers["location"]).query).get("request")
        if request is None or not request:
            raise RuntimeError("consent redirect lacks request")
        shown = client.get(origin + "/mcp/oauth/request", params={"request": request[0]})
        shown.raise_for_status()
        decided = client.post(
            origin + "/mcp/oauth/consent",
            json={"request": request[0], "approve": True},
            headers={"Authorization": "Bearer " + session},
        )
        decided.raise_for_status()
        return str(decided.json()["redirect_uri"])


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


def cli_oauth(context: JourneyContext) -> StepReceipt:
    step = Step.CLI_OAUTH
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    with private_auth_file(context) as path:
        if context.driver == SurfaceName.SDK:
            from synth_ai import auth

            try:
                auth.login(
                    context.origin,
                    open_browser=False,
                    redirect_source=lambda url: approve(context.origin, session, url),
                )
            except Exception as error:
                return receipt(
                    context, step, Verdict.FAIL, "pkce_login_failed:" + type(error).__name__
                )
        else:
            assert isinstance(context.surface, CLISurface)
            failure = cli_login(context, session, path)
            if failure:
                return receipt(context, step, Verdict.FAIL, failure)
        token = json.loads(path.read_text())[context.origin.rstrip("/")]["access_token"]
    used = credential_works(context, str(token))
    kind = obj(obj(used.value).get("principal")).get("credential_kind")
    ok = used.accepted and kind == "oauth"
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "oauth_token_refused",
        used,
        notes={"credential_kind": kind},
    )


def cli_login(context: JourneyContext, session: str, path: Path) -> str:
    surface = context.surface
    assert isinstance(surface, CLISurface)
    environment = {
        key_: value for key_, value in surface.environment.items() if key_ != "SYNTH_API_KEY"
    }
    environment["SYNTH_AUTH_FILE"] = str(path)
    child = subprocess.Popen(
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdin is not None and child.stderr is not None
    url = child.stderr.readline().strip()
    try:
        redirect = approve(context.origin, session, url)
    except Exception as error:
        child.kill()
        return "consent_failed:" + type(error).__name__
    child.stdin.write(redirect + "\n")
    child.stdin.flush()
    code = child.wait(timeout=120)
    return "" if code == 0 else f"cli_login_exit_{code}"


def mcp_consent(context: JourneyContext) -> StepReceipt:
    """Hosted MCP OAuth: register, authorize, consent, exchange, call, list, revoke."""
    step = Step.MCP_CONSENT
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if context.mcp_url is None:
        return missing(context, step, "hosted_mcp_url")
    import base64
    import hashlib

    origin = context.origin
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    redirect_uri = "http://127.0.0.1:9/callback"
    notes: dict[str, JsonValue] = {}
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        registered = client.post(
            origin + "/mcp/oauth/register",
            json={
                "client_name": "SYN-5695 journey",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        if registered.status_code >= 300:
            return receipt(context, step, Verdict.FAIL, f"register_{registered.status_code}")
        client_id = registered.json()["client_id"]
        state = secrets.token_urlsafe(16)
        authorization = str(
            httpx.URL(
                origin + "/mcp/oauth/authorize",
                params={
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "response_type": "code",
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": state,
                    "resource": origin + "/mcp",
                    "scope": "account:read",
                },
            )
        )
        try:
            granted = approve(origin, session, authorization)
        except Exception as error:
            return receipt(context, step, Verdict.FAIL, "consent_failed:" + type(error).__name__)
        code = parse_qs(urlparse(granted).query).get("code")
        if code is None or not code:
            return receipt(context, step, Verdict.FAIL, "consent_redirect_lacks_code")
        token = client.post(
            origin + "/mcp/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code[0],
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": origin + "/mcp",
            },
        )
        if token.status_code >= 300:
            return receipt(context, step, Verdict.FAIL, f"token_{token.status_code}")
        access = str(token.json()["access_token"])
        grants = client.get(
            origin + "/mcp/oauth/grants", headers={"Authorization": "Bearer " + session}
        )
        notes["grants_status"] = grants.status_code
        listed = [
            grant
            for grant in items(
                obj(grants.json() if grants.status_code == 200 else {}).get("grants")
            )
            if obj(grant).get("client_id") == client_id
        ]
        notes["grant_listed"] = len(listed) == 1
    with context.make(access) as surface:
        used = surface.call(Call("getAccount"))
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for grant in listed:
            revoked = client.delete(
                origin + "/mcp/oauth/grants/" + str(obj(grant).get("grant_id")),
                headers={"Authorization": "Bearer " + session},
            )
            notes["grant_revoke_status"] = revoked.status_code
    with context.make(access) as surface:
        after = surface.call(Call("getAccount"))
    ok = used.accepted and len(listed) == 1 and not after.accepted
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "mcp_oauth_grant_lifecycle_failed",
        used,
        after,
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
        refused = surface.call(Call("cancelSubscription", idempotency_key=key("member-cancel")))
    roles = {"admin": account_role(admin), "member": account_role(identity)}
    if roles["admin"] not in ADMIN_ROLES or roles["member"] not in MEMBER_ROLES:
        return receipt(
            context, step, Verdict.FAIL, "roles_unexpected", admin, identity, notes=roles
        )
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
            context, step, outcome, codes=frozenset({"route_not_found"}), status=404
        )
    if isinstance(context.surface, CLISurface):
        outcome = context.surface.run("index_search", ["index", "search", "syn5695"])
        return expect_refusal(context, step, outcome, codes=frozenset({"usage_error"}))
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
        context, Step.EXPIRED_KEY, outcome, codes=frozenset({"authentication_required"}), status=401
    )


def overload_refusal(context: JourneyContext) -> StepReceipt:
    # The orchestrator holds the inbound bulkhead for longer than the client's
    # retry budget, so the client must surface a typed retryable refusal.
    outcome = context.surface.call(Call("getAccount"))
    result = expect_refusal(context, Step.OVERLOAD_REFUSAL, outcome, codes=OVERLOAD_CODES)
    if result.verdict == Verdict.PASS and (
        outcome.retry_after is None and outcome.retryable is not True
    ):
        return receipt(context, Step.OVERLOAD_REFUSAL, Verdict.FAIL, "retry_after_absent", outcome)
    return result


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
    if not isinstance(run, str) or snapshot.get("run_id") != run:
        return receipt(context, step, Verdict.FAIL, "launch_row_run_mismatch")
    if snapshot.get("state") != "queued" or not queue:
        return receipt(
            context,
            step,
            Verdict.INCONCLUSIVE,
            "precondition_not_met:capacity_not_saturated",
            notes={"acknowledged_state": snapshot.get("state")},
        )
    outcome, seen = wait_state(context, run, lambda v: v.get("state") in TERMINAL)
    position = queue.get("position")
    notes: dict[str, JsonValue] = {"run_id": run, "queue": queue, "states": list(seen)}
    if not isinstance(position, int) or position < 1:
        return receipt(context, step, Verdict.FAIL, "queue_position_invalid", outcome, notes=notes)
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


def matrix_operations() -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    """Every contract operation addressed by a resource ID: (op, method, path, params)."""
    document = contract()
    rows: list[tuple[str, str, str, tuple[str, ...]]] = []
    paths = document["paths"]
    assert isinstance(paths, dict)
    for route, methods in sorted(paths.items()):
        for method, operation in methods.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            names = tuple(path_parameters(document, operation))
            if any(name in ID_PARAMETERS for name in names):
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
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return str(match.group("prefix")) + "".join(secrets.choice(alphabet) for _ in tail)


MATRIX_BODIES: Mapping[str, Mapping[str, JsonValue]] = {
    "updateRunLimits": {"expected_revision": 0, "limits": {"wall_time_seconds": 60}},
    "updateRunLabels": {"expected_revision": 0, "labels": {}},
    "updateApiKey": {"expected_version": 1, "scopes": ["account:read"]},
}


def masked(problem: Optional[Mapping[str, JsonValue]], identifiers: tuple[str, ...]) -> str:
    text = json.dumps(problem or {}, sort_keys=True)
    for identifier in identifiers:
        text = text.replace(identifier, "<id>")
    return text


def classify(foreign: Outcome, absent: Outcome, identifiers: tuple[str, ...]) -> str:
    """pass | leak | differs (Fail) or unproven:* (Inconclusive)."""
    if foreign.accepted or absent.accepted:
        return "leak" if foreign.accepted else "differs"
    signature = [
        (item.status, item.code, item.cause, masked(item.problem, identifiers))
        for item in (foreign, absent)
    ]
    if signature[0] != signature[1]:
        return "differs"  # A foreign ID answers unlike an absent one: existence leaks.
    if foreign.status is None:
        return "unproven:never_reached_core"
    if foreign.status != 404:
        return f"unproven:uniform_{foreign.status}"
    return "pass"


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
    with context.make(actor) as surface:
        for operation, method, route, names in matrix_operations():
            if not covered(surface, operation):
                uncovered += 1
                rows.append({"operation": operation, "result": "uncovered:surface_unmapped"})
                continue
            needed = [name for name in names if name != "kind"]
            if any(name not in context.foreign_ids for name in needed):
                uncovered += 1
                rows.append({"operation": operation, "result": "uncovered:foreign_id_absent"})
                continue
            foreign = {name: context.foreign_ids[name] for name in needed}
            absent = {name: absent_like(value) for name, value in foreign.items()}
            if "kind" in names:
                foreign["kind"] = absent["kind"] = context.foreign_ids.get("kind", "task")
            body = MATRIX_BODIES.get(operation)
            mutation = method != "get"
            calls = [
                Call(
                    operation,
                    path=ids,
                    body=body,
                    idempotency_key=key("matrix") if mutation else None,
                )
                for ids in (foreign, absent)
            ]
            seen = [surface.call(call) for call in calls]
            result = classify(seen[0], seen[1], tuple(foreign.values()) + tuple(absent.values()))
            failed += result in {"leak", "differs"}
            uncovered += result.startswith("unproven:")
            rows.append(
                {
                    "operation": operation,
                    "route": route,
                    "result": result,
                    "foreign": seen[0].evidence(),
                    "absent": seen[1].evidence(),
                }
            )
    verdict = Verdict.FAIL if failed else (Verdict.INCONCLUSIVE if uncovered else Verdict.PASS)
    cause = (
        ""
        if verdict == Verdict.PASS
        else (f"failed={failed}" if failed else f"uncovered={uncovered}")
    )
    return StepReceipt(step, context.driver, verdict, cause, tuple(rows))


def covered(surface: Surface, operation: str) -> bool:
    if isinstance(surface, MCPSurface):
        return operation in surface.mapped()
    if isinstance(surface, CLISurface):
        return operation in CLI_ARGV
    return operation in SDK_CALLS


StepFunction = Callable[[JourneyContext], StepReceipt]

STEPS: Mapping[Step, tuple[StepFunction, frozenset[SurfaceName]]] = {
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
    Step.QUEUED_LAUNCH: (queued_launch, frozenset({SurfaceName.SDK, SurfaceName.CLI})),
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
