"""Shared types, verdicts, and request helpers for SYN-5695 journey drivers."""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .session_files import RefreshingSurface
from .surfaces import (
    CLI_ARGV,
    SDK_CALLS,
    Call,
    CLISurface,
    MCPSurface,
    Outcome,
    Surface,
    SurfaceName,
    obj,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

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
    PASS = "Pass"  # noqa: S105
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


def items(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


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


def covered(surface: Surface | RefreshingSurface, operation: str) -> bool:
    if isinstance(surface, RefreshingSurface):
        surface = surface.surface()
    if isinstance(surface, MCPSurface):
        return operation in surface.mapped()
    if isinstance(surface, CLISurface):
        return operation in CLI_ARGV
    return operation in SDK_CALLS
