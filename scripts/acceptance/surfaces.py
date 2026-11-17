"""One public operation through SDK, CLI subprocess or MCP; the outcome stays typed.

See SYN-5695. A surface runs an OpenAPI operationId with explicit path, query,
body and idempotency key, and returns an `Outcome`. Refusals keep the public
Problem (minus per-request `instance`) so journeys can compare them exactly.
Untyped refusals, such as a raw pydantic error, are recorded as untyped.
Credentials stay in the client or child environment, never argv or receipts.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import errno
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Optional, Protocol, cast

from pydantic import BaseModel, JsonValue, ValidationError

from synth_ai import SynthClient
from synth_ai._generated.models import CheckoutRequestPackId, MintApiKeyRequestPurpose, Scope
from synth_ai.errors import SwarmsAPIError, SynthError
from synth_ai.models import Money, OptimizerRequest, RunLimitUpdate

from .mcp_driver import HostedFailure, ToolCaller

__all__ = [
    "CLI_ARGV",
    "CLI_ARGV_TOO_LONG",
    "SDK_CALLS",
    "CLISurface",
    "Call",
    "MCPSurface",
    "Outcome",
    "SDKSurface",
    "Surface",
    "SurfaceName",
]

ROOT = Path(__file__).parents[2]
CLI_SECONDS = 120
CLI_ARGV_TOO_LONG = "cli_argv_too_long"
# Problem members that identify one request rather than the refusal itself.
PER_REQUEST = frozenset({"instance", "request_id", "trace_id"})


class SurfaceName(StrEnum):
    SDK = "sdk"
    CLI = "cli"
    MCP_STDIO = "mcp_stdio"
    MCP_HOSTED = "mcp_hosted"


@dataclass(frozen=True, slots=True)
class Call:
    operation: str
    path: Mapping[str, str] = field(default_factory=dict)
    query: Mapping[str, JsonValue] = field(default_factory=dict)
    body: Optional[Mapping[str, JsonValue]] = None
    idempotency_key: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Outcome:
    """`accepted` or a refusal; a refusal is typed only when it carries a code."""

    operation: str
    accepted: bool
    value: JsonValue = None
    status: Optional[int] = None
    code: Optional[str] = None
    retry_after: Optional[int] = None
    retryable: Optional[bool] = None
    problem: Optional[Mapping[str, JsonValue]] = None
    cause: str = ""

    @property
    def typed(self) -> bool:
        return not self.accepted and self.code is not None

    def evidence(self) -> dict[str, JsonValue]:
        # Response bodies stay out of receipts except a refusal's own Problem.
        return {
            "operation": self.operation,
            "accepted": self.accepted,
            "status": self.status,
            "code": self.code,
            "retry_after": self.retry_after,
            "retryable": self.retryable,
            "typed": self.typed,
            "cause": self.cause,
        }


def comparable(problem: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in problem.items() if key not in PER_REQUEST}


def text(body: Mapping[str, JsonValue], name: str) -> Optional[str]:
    value = body.get(name)
    return value if isinstance(value, str) else None


def integer(body: Mapping[str, JsonValue], name: str) -> Optional[int]:
    value = body.get(name)
    return value if isinstance(value, int) else None


def obj(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def accepted(operation: str, value: object) -> Outcome:
    if isinstance(value, BaseModel):
        return Outcome(operation, True, value.model_dump(mode="json"))
    return Outcome(operation, True, None if value is None else str(value))


def refused_problem(
    operation: str, problem: Mapping[str, JsonValue], *, retry_after: Optional[int]
) -> Outcome:
    status = problem.get("status")
    code = problem.get("code")
    retryable = problem.get("retryable")
    return Outcome(
        operation,
        False,
        status=status if isinstance(status, int) else None,
        code=code if isinstance(code, str) else None,
        retry_after=retry_after,
        retryable=retryable if isinstance(retryable, bool) else None,
        problem=comparable(problem),
    )


class Surface(Protocol):
    name: SurfaceName

    def call(self, call: Call) -> Outcome: ...


def run_id(call: Call) -> str:
    return call.path["run_id"]


def launch(client: SynthClient, call: Call) -> object:
    # The public keyword launch, as a customer writes it. Contract-invalid input
    # reaches the SDK's own validation exactly as a customer's would.
    body = dict(call.body or {})
    limits = obj(body.get("limits"))
    charge = obj(limits.get("maximum_charge"))
    # Unvalidated on purpose: the SDK's own launch validation must judge it.
    optimizer = cast(Optional[OptimizerRequest], body.get("optimizer"))
    return client.swarms.launch(
        str(body.get("message")),
        budget=str(charge.get("amount")),
        wall_time_seconds=integer(limits, "wall_time_seconds"),
        optimizer=optimizer,
        profile=text(body, "profile"),
        idempotency_key=call.idempotency_key,
    )


def update_limits(client: SynthClient, call: Call) -> object:
    request = RunLimitUpdate.model_validate(call.body)
    return client.swarms.update_limits_request(
        run_id(call), request, idempotency_key=call.idempotency_key
    )


def checkout(client: SynthClient, call: Call) -> object:
    body = dict(call.body or {})
    credit = body.get("credit")
    return client.billing.checkout(
        plan_id=text(body, "plan_id"),
        pack_id=cast(Optional[CheckoutRequestPackId], text(body, "pack_id")),
        credit=None if credit is None else Money.model_validate(credit),
        idempotency_key=call.idempotency_key,
    )


def mint(client: SynthClient, call: Call) -> object:
    body = dict(call.body or {})
    return client.account.api_keys.mint(
        purpose=cast(Optional[MintApiKeyRequestPurpose], text(body, "purpose")),
        name=text(body, "name"),
        scopes=cast(Optional[list[Scope]], body.get("scopes")),
        expires_at=text(body, "expires_at"),
        request_id=text(body, "request_id"),
        idempotency_key=call.idempotency_key,
    )


SdkCall = Callable[[SynthClient, Call], object]


def event_page(client: SynthClient, call: Call) -> object:
    cursor = call.query.get("cursor")
    limit = call.query.get("limit", 20)
    if cursor is not None and not isinstance(cursor, str):
        raise ValueError("event_cursor_invalid")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("event_limit_invalid")
    return client.swarms.events(run_id(call), cursor=cursor, limit=limit)


# Public SDK methods only. An operation absent here is uncovered for the SDK.
SDK_CALLS: Mapping[str, SdkCall] = {
    "getAccount": lambda c, a: c.account.get(),
    "getBalance": lambda c, a: c.billing.balance(),
    "listStatements": lambda c, a: c.billing.statements(),
    "createCheckout": checkout,
    "getCheckout": lambda c, a: c.billing.checkout_status(a.path["id"]),
    "cancelSubscription": lambda c, a: c.billing.cancel_subscription(
        idempotency_key=a.idempotency_key
    ),
    "createRun": launch,
    "getRun": lambda c, a: c.swarms.get(run_id(a)),
    "getRunLimits": lambda c, a: c.swarms.limits(run_id(a)),
    "updateRunLimits": update_limits,
    "pauseRun": lambda c, a: c.swarms.pause(run_id(a), idempotency_key=a.idempotency_key),
    "resumeRun": lambda c, a: c.swarms.resume(run_id(a), idempotency_key=a.idempotency_key),
    "cancelRun": lambda c, a: c.swarms.cancel(run_id(a), idempotency_key=a.idempotency_key),
    "getRunReport": lambda c, a: c.swarms.report(run_id(a)),
    "getRunResult": lambda c, a: c.swarms.result(run_id(a)),
    "listRunOutputs": lambda c, a: c.swarms.outputs(run_id(a)),
    "getRunOutput": lambda c, a: c.swarms.output(run_id(a), a.path["output_id"]),
    "getRunAllocation": lambda c, a: c.swarms.allocation(run_id(a)),
    "getRunConfiguration": lambda c, a: c.swarms.configuration(run_id(a)),
    "getRunProgram": lambda c, a: c.swarms.program(run_id(a)),
    "getRunSandboxes": lambda c, a: c.swarms.sandboxes(run_id(a)),
    "listRunTasks": lambda c, a: c.swarms.tasks(run_id(a)),
    "listRunEvents": event_page,
    "getFile": lambda c, a: c.swarms.file(a.path["file_id"]),
    "getUsageExport": lambda c, a: c.account.usage_export(a.path["export_id"]),
    "withdrawPlanRequest": lambda c, a: c.account.withdraw_plan_request(a.path["request_id"]),
    "createApiKey": mint,
    "rotateApiKey": lambda c, a: c.account.api_keys.rotate(
        a.path["key_id"], idempotency_key=a.idempotency_key
    ),
    "revokeApiKey": lambda c, a: c.account.api_keys.revoke(a.path["key_id"]),
}


class SDKSurface:
    name = SurfaceName.SDK

    def __init__(self, client: SynthClient) -> None:
        self.client = client

    def call(self, call: Call) -> Outcome:
        method = SDK_CALLS.get(call.operation)
        if method is None:
            return Outcome(call.operation, False, cause="surface_operation_unmapped")
        try:
            return accepted(call.operation, method(self.client, call))
        except SwarmsAPIError as error:
            return refused_problem(
                call.operation,
                error.problem.model_dump(mode="json"),
                retry_after=error.retry_after_seconds,
            )
        except SynthError as error:
            # Local SDK refusals carry a documented code but no HTTP status.
            return Outcome(call.operation, False, code=error.code, cause="local_sdk_error")
        except ValidationError as error:
            return Outcome(
                call.operation,
                False,
                cause=f"untyped pydantic.ValidationError ({error.error_count()} errors)",
            )


def body_file(directory: Path, body: Mapping[str, JsonValue]) -> str:
    descriptor, name = tempfile.mkstemp(suffix=".json", dir=directory)
    with os.fdopen(descriptor, "w") as output:
        json.dump(body, output)
    return name


ArgvBuilder = Callable[[Call, Path], Sequence[str]]


def launch_argv(call: Call, directory: Path) -> Sequence[str]:
    body = dict(call.body or {})
    charge = obj(obj(body.get("limits")).get("maximum_charge"))
    message = str(body.get("message"))
    # A private file exercises the CLI boundary without the OS argv-size ceiling.
    descriptor, name = tempfile.mkstemp(suffix=".txt", dir=directory)
    with os.fdopen(descriptor, "w") as output:
        output.write(message)
    argv = ["run", "--task-file", name, "--budget", str(charge.get("amount"))]
    if call.idempotency_key is not None:
        argv += ["--idempotency-key", call.idempotency_key]
    optimizer = body.get("optimizer")
    if isinstance(optimizer, dict):
        argv += ["--optimizer", f"{optimizer['topology']}@{optimizer['version']}"]
    elif optimizer is not None:
        argv += ["--optimizer", str(optimizer)]
    if body.get("profile") is not None:
        argv += ["--profile", str(body["profile"])]
    return argv


def limits_argv(call: Call, directory: Path) -> Sequence[str]:
    body = dict(call.body or {})
    changes = obj(body.get("limits"))
    argv: list[str] = [
        "limits",
        "set",
        run_id(call),
        "--expected-revision",
        str(body.get("expected_revision")),
    ]
    charge = changes.get("maximum_charge")
    if isinstance(charge, dict):
        argv += ["--budget", str(Decimal(str(charge["amount"])))]
    for name, flag in (
        ("wall_time_seconds", "--wall"),
        ("maximum_tokens", "--tokens"),
        ("pause_retention_seconds", "--pause-retention"),
    ):
        if changes.get(name) is not None:
            argv += [flag, str(changes[name])]
    if body.get("resume") is True:
        argv.append("--resume")
    reason = text(body, "reason")
    if reason is not None:
        argv += ["--reason", reason]
    return argv


def run_verb(verb: str) -> ArgvBuilder:
    return lambda a, d: [verb, run_id(a)]


def keyed_run_verb(verb: str) -> ArgvBuilder:
    return lambda a, d: [verb, run_id(a), "--idempotency-key", str(a.idempotency_key)]


def inspect_view(view: str) -> ArgvBuilder:
    return lambda a, d: ["inspect", run_id(a), "--view", view]


def event_page_argv(call: Call, _directory: Path) -> Sequence[str]:
    argv = ["logs", run_id(call), "--limit", str(call.query.get("limit", 20))]
    cursor = call.query.get("cursor")
    if isinstance(cursor, str):
        argv += ["--cursor", cursor]
    return argv


# Real `synth` commands. An operation absent here is uncovered for the CLI.
CLI_ARGV: Mapping[str, ArgvBuilder] = {
    "getAccount": lambda a, d: ["organization", "get"],
    "getBalance": lambda a, d: ["billing", "balance"],
    "listStatements": lambda a, d: ["billing", "statements"],
    "createCheckout": lambda a, d: [
        "billing",
        "checkout",
        "--body",
        body_file(d, a.body or {}),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "getCheckout": lambda a, d: ["billing", "checkout-status", a.path["id"]],
    "cancelSubscription": lambda a, d: [
        "billing",
        "cancel-subscription",
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "createRun": launch_argv,
    "getRun": run_verb("status"),
    "listRunEvents": event_page_argv,
    "getRunLimits": run_verb("limits"),
    "updateRunLimits": lambda a, d: [
        *limits_argv(a, d),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "pauseRun": keyed_run_verb("pause"),
    "resumeRun": keyed_run_verb("resume"),
    "cancelRun": keyed_run_verb("cancel"),
    "getRunReport": run_verb("report"),
    "getRunResult": run_verb("result"),
    "getRunAllocation": run_verb("allocation"),
    "listRunTasks": run_verb("tasks"),
    "getRunProgram": inspect_view("program"),
    "getRunConfiguration": inspect_view("configuration"),
    "getRunSandboxes": inspect_view("sandboxes"),
    "getFile": lambda a, d: ["files", "get", a.path["file_id"]],
    "getUsageExport": lambda a, d: ["organization", "usage-export-status", a.path["export_id"]],
    "createApiKey": lambda a, d: [
        "keys",
        "mint",
        "--body",
        body_file(d, a.body or {}),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "rotateApiKey": lambda a, d: [
        "keys",
        "rotate",
        a.path["key_id"],
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "revokeApiKey": lambda a, d: ["keys", "revoke", a.path["key_id"]],
}


def cli_problem(text: str) -> Optional[dict[str, JsonValue]]:
    for line in reversed(text.strip().splitlines()):
        try:
            document = json.loads(line)
        except ValueError:
            continue
        if isinstance(document, dict) and isinstance(document.get("code"), str):
            return cast(dict[str, JsonValue], document)
    return None


class CLISurface:
    """`python -m synth_ai.cli --env local ... --json` in a real child process."""

    name = SurfaceName.CLI

    def __init__(
        self, *, python: Path, environment: Mapping[str, str], work_directory: Path
    ) -> None:
        self.python = python
        self.environment = dict(environment)
        self.work_directory = work_directory

    def argv(self, arguments: Sequence[str]) -> list[str]:
        return [str(self.python), "-m", "synth_ai.cli", "--env", "local", *arguments, "--json"]

    def run(self, operation: str, arguments: Sequence[str]) -> Outcome:
        try:
            result = subprocess.run(
                self.argv(arguments),
                cwd=ROOT,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=CLI_SECONDS,
                check=False,
            )
        except OSError as error:
            if error.errno == errno.E2BIG:
                # Other oversized arguments can still exceed the OS argv limit.
                return Outcome(operation, False, cause=CLI_ARGV_TOO_LONG)
            raise
        if result.returncode == 0:
            stdout = result.stdout.strip()
            return Outcome(operation, True, json.loads(stdout) if stdout else None)
        problem = cli_problem(result.stderr)
        if problem is None:
            return Outcome(operation, False, cause=f"CLI exit {result.returncode} without code")
        retry = problem.pop("retry_after_seconds", None)
        if "status" in problem:
            return refused_problem(
                operation, problem, retry_after=retry if isinstance(retry, int) else None
            )
        # Exit 1 local refusals print {"code","detail"} with no HTTP status.
        code = problem["code"]
        return Outcome(
            operation,
            False,
            code=code if isinstance(code, str) else None,
            cause=f"CLI exit {result.returncode}",
        )

    def call(self, call: Call) -> Outcome:
        builder = CLI_ARGV.get(call.operation)
        if builder is None:
            return Outcome(call.operation, False, cause="surface_operation_unmapped")
        return self.run(call.operation, builder(call, self.work_directory))


def catalog() -> dict[str, dict[str, JsonValue]]:
    tools = json.loads((ROOT / "contracts/mcp-tools.json").read_text())["tools"]
    return {tool["operationId"]: tool for tool in tools if tool.get("operationId")}


class MCPSurface:
    """`tools/call` for the catalog tool whose operationId matches the call."""

    def __init__(self, caller: ToolCaller, *, name: SurfaceName) -> None:
        self.caller = caller
        self.name = name
        self.available = caller.tools()
        self.tools = catalog()

    def arguments(self, call: Call, schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        properties = schema.get("properties")
        allowed = set(properties) if isinstance(properties, dict) else set()
        candidate: dict[str, JsonValue] = {**call.path, **call.query}
        if call.body is not None:
            candidate["body"] = dict(call.body)
        if call.idempotency_key is not None:
            candidate["idempotency_key"] = call.idempotency_key
        return {key: value for key, value in candidate.items() if key in allowed}

    def call_tool(self, operation: str, tool: str, arguments: dict[str, JsonValue]) -> Outcome:
        if tool not in self.available:
            return Outcome(operation, False, cause="mcp_tool_unavailable")
        outcome = self.caller.call(tool, arguments)
        try:
            document = json.loads(outcome.text) if outcome.text else None
        except ValueError:
            document = None
        if not outcome.is_error:
            return Outcome(operation, True, document)
        if isinstance(document, dict) and isinstance(document.get("code"), str):
            retry = document.pop("retry_after_seconds", None)
            return refused_problem(
                operation, document, retry_after=retry if isinstance(retry, int) else None
            )
        try:
            hosted = HostedFailure.model_validate(document)
        except ValidationError:
            return Outcome(operation, False, cause="mcp unstructured tool error")
        if hosted.data is not None and hosted.data.response is not None:
            return refused_problem(
                operation, hosted.data.response.model_dump(mode="json"), retry_after=None
            )
        return Outcome(operation, False, cause=f"mcp tool error {hosted.error}")

    def call(self, call: Call) -> Outcome:
        tool = self.tools.get(call.operation)
        if tool is None:
            return Outcome(call.operation, False, cause="surface_operation_unmapped")
        schema = tool["inputSchema"]
        assert isinstance(schema, dict)
        return self.call_tool(call.operation, str(tool["name"]), self.arguments(call, schema))

    def mapped(self) -> frozenset[str]:
        return frozenset(self.tools)
