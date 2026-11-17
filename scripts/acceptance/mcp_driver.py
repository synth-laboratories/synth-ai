"""J1 through the MCP tool catalog: the same journey as the SDK and CLI drivers.

See SYN-5693. Every read is an MCP `tools/call` against the generated catalog
(`contracts/mcp-tools.json`); this driver never calls SDK client methods. The
transport (stdio child or hosted streamable HTTP) is injected as a `ToolCaller`.

Output bytes are decoded from the canonical MCP envelope and checked against
the immutable output reference before they are retained as evidence.
"""

from __future__ import annotations

import binascii
import hashlib

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import json
from base64 import b64decode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from synth_ai.models import (
    File,
    InputFile,
    LaunchBudget,
    LaunchRequest,
    Limits,
    MCPOutputContent,
    OutputPage,
    Problem,
    RunInputs,
    RunOutput,
    RunReport,
    RunResult,
    RunSnapshot,
    RunWait,
    RunWaitOutcome,
    UsageRecordPage,
)

from .j1 import AdapterError, content_pin, refusal
from .launch_journal import InputIdentity, LaunchIntent, LaunchJournal
from .limits import (
    MCP_UPLOAD_BYTES_MAX,
    MCP_WAIT_CALL_SECONDS,
    MCP_WAIT_CALLS_MAX,
    OUTPUT_PAGES_MAX,
    PAGE_ITEMS,
)

__all__ = [
    "MCPDriver",
    "MCPRefusal",
    "Tool",
    "ToolCaller",
    "ToolOutcome",
    "classify_failure",
]

M = TypeVar("M", bound=BaseModel)
JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class Tool(StrEnum):
    """Catalog names this driver calls; see contracts/mcp-tools.json."""

    UPLOAD = "swarms_upload_file"
    LAUNCH = "swarms_run_launch"
    GET = "swarms_run_get"
    WAIT = "swarms_run_wait"
    REPORT = "swarms_run_report"
    RESULT = "swarms_run_result"
    OUTPUTS = "swarms_run_outputs"
    CONTENT = "swarms_run_output_content"
    USAGE = "account_usage_for_run"


class MCPRefusal(StrEnum):
    # The server's tool list does not contain a tool this journey needs.
    TOOL_UNAVAILABLE = "mcp_tool_unavailable"
    # The tool returned isError with a public Problem (stdio or hosted nested).
    TOOL_REFUSED = "mcp_tool_refused"
    # The tool returned isError without a Problem (e.g. hosted argument refusal).
    TOOL_FAILED = "mcp_tool_failed"
    # A success result whose content is not the declared public model.
    RESULT_INVALID = "mcp_result_invalid"
    # The transport itself failed (session, HTTP status, JSON-RPC error).
    TRANSPORT_FAILED = "mcp_transport_failed"
    OUTPUT_CURSOR_INVALID = "output_cursor_invalid"
    OUTPUT_PAGE_LIMIT = "output_page_limit"
    WAIT_UNSETTLED = "mcp_wait_unsettled"
    UPLOAD_MEDIA_TYPE = "mcp_upload_media_type_unsupported"
    UPLOAD_TOO_LARGE = "mcp_upload_too_large"
    OPTIMIZER_UNOBSERVED = "optimizer_resolution_unobserved"


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """One `tools/call` result: the error flag and its single text content."""

    is_error: bool
    text: str


class ToolCaller(Protocol):
    def tools(self) -> frozenset[str]: ...
    def call(self, name: str, arguments: Mapping[str, JsonValue]) -> ToolOutcome: ...


class HostedFailureData(BaseModel):
    # Core mcpoauth toolFailure: {"error", "data": {"tool", "status", "response"}}.
    model_config = ConfigDict(extra="allow", frozen=True)
    tool: str
    status: Optional[int] = None
    response: Optional[Problem] = None


class HostedFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    error: str
    data: Optional[HostedFailureData] = None


def classify_failure(tool: Tool, outcome: ToolOutcome, *, run_id: str) -> AdapterError:
    """Map an isError result to a typed refusal; bodies never enter the cause."""
    try:
        problem = Problem.model_validate_json(outcome.text)
    except ValidationError:
        problem = None
    if problem is not None:
        return refusal(
            MCPRefusal.TOOL_REFUSED,
            stage=tool,
            cause=f"problem={problem.code} status={problem.status}",
            run_id=run_id,
        )
    try:
        hosted = HostedFailure.model_validate_json(outcome.text)
    except ValidationError:
        return refusal(
            MCPRefusal.TOOL_FAILED,
            stage=tool,
            cause="unstructured tool error",
            run_id=run_id,
        )
    if hosted.data is not None and hosted.data.response is not None:
        return refusal(
            MCPRefusal.TOOL_REFUSED,
            stage=tool,
            cause=f"problem={hosted.data.response.code} status={hosted.data.response.status}",
            run_id=run_id,
        )
    status = (
        "none" if hosted.data is None or hosted.data.status is None else str(hosted.data.status)
    )
    # Hosted failure messages are fixed server constants, not response bodies.
    return refusal(
        MCPRefusal.TOOL_FAILED,
        stage=tool,
        cause=f"error={hosted.error} status={status}",
        run_id=run_id,
    )


UPLOAD_MEDIA_TYPES: Mapping[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
}


class MCPDriver:
    """Implements `j1.RecordDriver` plus wait/upload/launch/usage via MCP tools."""

    def __init__(self, caller: ToolCaller, *, surface: str) -> None:
        self.caller = caller
        # `surface` names the journeys.Driver value (mcp_stdio or mcp_hosted).
        self.surface = surface
        self._available = caller.tools()

    def _call(
        self, tool: Tool, arguments: Mapping[str, JsonValue], model: type[M], *, run_id: str
    ) -> M:
        if tool not in self._available:
            raise refusal(
                MCPRefusal.TOOL_UNAVAILABLE,
                stage=tool,
                cause="server tool list lacks " + tool,
                run_id=run_id,
            )
        outcome = self.caller.call(tool, arguments)
        if outcome.is_error:
            raise classify_failure(tool, outcome, run_id=run_id)
        try:
            return model.model_validate_json(outcome.text)
        except ValidationError as error:
            raise refusal(
                MCPRefusal.RESULT_INVALID,
                stage=tool,
                cause=f"content is not {model.__name__} ({error.error_count()} errors)",
                run_id=run_id,
            ) from error

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self._call(Tool.GET, {"run_id": run_id}, RunSnapshot, run_id=run_id)

    def report(self, run_id: str) -> RunReport:
        return self._call(Tool.REPORT, {"run_id": run_id}, RunReport, run_id=run_id)

    def result(self, run_id: str) -> RunResult:
        return self._call(Tool.RESULT, {"run_id": run_id}, RunResult, run_id=run_id)

    def outputs(self, run_id: str) -> tuple[RunOutput, ...]:
        cursor: Optional[str] = None
        items: list[RunOutput] = []
        seen: set[str] = set()
        for _ in range(OUTPUT_PAGES_MAX):
            arguments: dict[str, JsonValue] = {"run_id": run_id, "limit": PAGE_ITEMS}
            if cursor is not None:
                arguments["cursor"] = cursor
            page = self._call(Tool.OUTPUTS, arguments, OutputPage, run_id=run_id)
            items.extend(page.items)
            if not page.has_more:
                return tuple(items)
            if page.next_cursor is None or page.next_cursor in seen:
                raise refusal(
                    MCPRefusal.OUTPUT_CURSOR_INVALID,
                    stage=Tool.OUTPUTS,
                    cause="missing or repeated next cursor",
                    run_id=run_id,
                )
            cursor = page.next_cursor
            seen.add(cursor)
        raise refusal(
            MCPRefusal.OUTPUT_PAGE_LIMIT,
            stage=Tool.OUTPUTS,
            cause="bounded output page count exceeded",
            run_id=run_id,
        )

    def download(self, run_id: str, output: RunOutput, target: Path) -> Path:
        pin = content_pin(output)
        envelope = self._call(
            Tool.CONTENT,
            {
                "run_id": run_id,
                "output_id": output.output_id,
                "revision_id": pin.revision_id,
                "digest_sha256": pin.digest_sha256,
            },
            MCPOutputContent,
            run_id=run_id,
        )
        try:
            body = b64decode(envelope.data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise refusal(
                MCPRefusal.RESULT_INVALID,
                stage=Tool.CONTENT,
                cause="output is not valid base64",
                run_id=run_id,
            ) from error
        if (
            len(body) > 1 << 20
            or envelope.revision_id != pin.revision_id
            or envelope.digest_sha256 != pin.digest_sha256
            or hashlib.sha256(body).hexdigest() != pin.digest_sha256
        ):
            raise refusal(
                MCPRefusal.RESULT_INVALID,
                stage=Tool.CONTENT,
                cause="output differs from immutable reference",
                run_id=run_id,
            )
        with target.open("xb") as handle:
            handle.write(body)
        return target

    def usage(self, run_id: str) -> UsageRecordPage:
        """First usage page for the run; reconciliation reads `items` only."""
        return self._call(
            Tool.USAGE, {"run_id": run_id, "limit": PAGE_ITEMS}, UsageRecordPage, run_id=run_id
        )

    def wait(self, run_id: str) -> RunSnapshot:
        # Each tool call is one bounded server wait (catalog maximum 50 s).
        for _ in range(MCP_WAIT_CALLS_MAX):
            update = self._call(
                Tool.WAIT,
                {"run_id": run_id, "timeout_seconds": MCP_WAIT_CALL_SECONDS},
                RunWait,
                run_id=run_id,
            )
            if update.run.run_id != run_id:
                raise refusal(
                    "observation_run_mismatch",
                    stage=Tool.WAIT,
                    cause="wait returned another run identity",
                    run_id=run_id,
                )
            if update.outcome != RunWaitOutcome.TIMEOUT:
                return update.run
        raise refusal(
            MCPRefusal.WAIT_UNSETTLED,
            stage=Tool.WAIT,
            cause="run did not settle within the bounded wait calls",
            run_id=run_id,
        )

    def upload(self, path: Path, *, idempotency_key: str) -> File:
        media_type = UPLOAD_MEDIA_TYPES.get(path.suffix)
        if media_type is None:
            raise refusal(
                MCPRefusal.UPLOAD_MEDIA_TYPE,
                stage=Tool.UPLOAD,
                cause="MCP upload accepts .txt, .md or .json text only",
                run_id="not_launched",
            )
        raw = path.read_bytes()
        if len(raw) > MCP_UPLOAD_BYTES_MAX:
            raise refusal(
                MCPRefusal.UPLOAD_TOO_LARGE,
                stage=Tool.UPLOAD,
                cause="MCP upload content exceeds the catalog maximum",
                run_id="not_launched",
            )
        return self._call(
            Tool.UPLOAD,
            {
                "body": {
                    "name": path.name,
                    "media_type": media_type,
                    "content": raw.decode("utf-8"),
                },
                "idempotency_key": idempotency_key,
            },
            File,
            run_id="not_launched",
        )

    def launch_auto(
        self,
        *,
        message: str,
        budget_usd: str,
        files: Sequence[File],
        idempotency_key: str,
        journal: LaunchJournal,
    ) -> RunSnapshot:
        # Pure local request construction; the launch itself is the MCP tool.
        amount = Decimal(budget_usd)
        if not amount.is_finite() or amount <= 0:
            raise ValueError("budget must be a positive decimal")
        request = LaunchRequest(
            message=message,
            limits=Limits(maximum_charge=LaunchBudget(amount=format(amount, "f"), currency="USD")),
            inputs=RunInputs(
                files=[
                    InputFile(
                        file_id=file.file_id,
                        revision_id=file.revision_id,
                        digest_sha256=file.digest_sha256,
                    )
                    for file in files
                ]
            ),
            optimizer="auto",
        )
        body = JSON_OBJECT.validate_json(request.model_dump_json(exclude_unset=True))
        journal.begin(
            LaunchIntent(
                self.surface,
                idempotency_key,
                journal.request_digest(json.dumps(body, sort_keys=True)),
                tuple(InputIdentity(file.name, file.digest_sha256) for file in files),
            )
        )
        snapshot = self._call(
            Tool.LAUNCH,
            {"body": body, "idempotency_key": idempotency_key},
            RunSnapshot,
            run_id="launch_outcome_pending",
        )
        journal.accepted(snapshot)
        if snapshot.optimizer is None:
            raise refusal(
                MCPRefusal.OPTIMIZER_UNOBSERVED,
                stage="launch",
                cause="accepted run has no authoritative optimizer receipt; retain its identity and do not relaunch",
                run_id=snapshot.run_id,
            )
        return snapshot
