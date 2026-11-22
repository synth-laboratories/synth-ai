"""SDK and real CLI subprocess adapters for the supported J1 customer operations.

See SYN-5692. Observation/download can run on a retained existing run without
launching it again. Missing attempt-tree or result.md projections remain named
blockers; they do not prevent exercising supported public SDK/CLI methods.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Optional, Protocol

from pydantic import TypeAdapter

from synth_ai import SynthClient
from synth_ai.cli import EXIT_RUN_UNSUCCESSFUL as CLI_EXIT_RUN_UNSUCCESSFUL
from synth_ai.cli import UNSUCCESSFUL_STATES as CLI_UNSUCCESSFUL_STATES
from synth_ai.models import (
    File,
    InputFile,
    OwnerResourceRequest,
    RunInputs,
    RunOutput,
    RunReport,
    RunResult,
    RunSnapshot,
)

from .attempt_tree import TreeError, project_attempts
from .launch_journal import InputIdentity, LaunchIntent, LaunchJournal
from .limits import COMMAND_SECONDS, OUTPUT_BYTES_MAX, OUTPUT_PAGES_MAX, PAGE_ITEMS, WAIT_SECONDS

__all__ = [
    "AdapterError",
    "CLIDriver",
    "J1Observation",
    "Missing",
    "RecordDriver",
    "RecordedLaunch",
    "SDKDriver",
    "content_pin",
    "observe",
    "observe_records",
    "refusal",
]
_LOGGER = logging.getLogger(__name__)


class Missing(StrEnum):
    ATTEMPT_TREE = "attempt_tree_contract_unavailable"
    RESULT_MARKDOWN = "result_markdown_unavailable"
    FINALITY = "run_not_fully_closed"
    CHARGE = "charge_unobserved"
    # The surface lists pinned outputs but has no public content tool (MCP).
    OUTPUT_CONTENT = "output_content_surface_unavailable"


@dataclass(slots=True)
class AdapterError(Exception):
    code: str
    stage: str
    cause: str
    run_id: str


def refusal(code: str, *, stage: str, cause: str, run_id: str) -> AdapterError:
    _LOGGER.error("code=%s stage=%s run_id=%s cause=%s", code, stage, run_id, cause)
    return AdapterError(code, stage, cause, run_id)


PUBLIC_LAUNCH_FIELDS = frozenset({"message", "labels", "limits", "resources"})
# The recording's operator field; LocalReplayTransport owns it, never a public surface.
OPERATOR_LAUNCH_FIELDS = frozenset({"local_cache"})
RECORDED_LIMIT_FIELDS = frozenset({"maximum_charge", "wall_time_seconds"})
RESOURCES = TypeAdapter(list[OwnerResourceRequest])


@dataclass(frozen=True, slots=True)
class RecordedLaunch:
    """The exact public launch body of a cached recording (SYN-5692).

    Cached J1 must send these bytes' public meaning unchanged through both the SDK
    and the CLI: a changed workload must miss the recording, never be adjusted.
    """

    message: str
    budget_usd: str
    wall_time_seconds: int
    labels: Mapping[str, str]
    resources: tuple[OwnerResourceRequest, ...]

    @classmethod
    def from_body(cls, body: Mapping[str, object]) -> RecordedLaunch:
        unknown = set(body) - PUBLIC_LAUNCH_FIELDS - OPERATOR_LAUNCH_FIELDS
        if unknown or not PUBLIC_LAUNCH_FIELDS <= set(body):
            raise refusal(
                "recorded_launch_shape_unsupported",
                stage="recording",
                cause="fields " + ",".join(sorted(set(body) ^ PUBLIC_LAUNCH_FIELDS)),
                run_id="not_launched",
            )
        limits = body["limits"]
        message = body["message"]
        labels = body["labels"]
        if (
            not isinstance(limits, dict)
            or set(limits) != RECORDED_LIMIT_FIELDS
            or not isinstance(message, str)
            or not isinstance(labels, dict)
        ):
            raise refusal(
                "recorded_launch_shape_unsupported",
                stage="recording",
                cause="message, labels or limits outside the supported public subset",
                run_id="not_launched",
            )
        charge = limits["maximum_charge"]
        if not isinstance(charge, dict) or charge.get("currency") != "USD":
            raise refusal(
                "recorded_launch_shape_unsupported",
                stage="recording",
                cause="maximum_charge is not a USD amount",
                run_id="not_launched",
            )
        return cls(
            message=message,
            budget_usd=str(charge["amount"]),
            wall_time_seconds=int(limits["wall_time_seconds"]),
            labels={str(key): str(value) for key, value in labels.items()},
            resources=tuple(RESOURCES.validate_python(body["resources"])),
        )

    def resources_json(self) -> str:
        return RESOURCES.dump_json(list(self.resources), exclude_none=True).decode()


@dataclass(frozen=True, slots=True)
class OutputEvidence:
    output: RunOutput
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class J1Observation:
    snapshot: RunSnapshot
    report: RunReport
    result: RunResult
    # Every listed output record, whether or not this surface fetched bytes.
    listed: tuple[RunOutput, ...]
    outputs: tuple[OutputEvidence, ...]
    result_markdown: Optional[str]
    missing: tuple[Missing, ...]


class RecordDriver(Protocol):
    def snapshot(self, run_id: str) -> RunSnapshot: ...
    def report(self, run_id: str) -> RunReport: ...
    def result(self, run_id: str) -> RunResult: ...
    def outputs(self, run_id: str) -> tuple[RunOutput, ...]: ...


class ReadDriver(RecordDriver, Protocol):
    def download(self, run_id: str, output: RunOutput, target: Path) -> Path: ...


class SDKDriver:
    def __init__(self, client: SynthClient) -> None:
        self.client = client

    def upload(self, path: Path, *, idempotency_key: str) -> File:
        return self.client.swarms.upload_file(str(path), idempotency_key=idempotency_key)

    def launch_auto(
        self,
        *,
        message: str,
        budget_usd: str,
        files: Sequence[File],
        idempotency_key: str,
        journal: LaunchJournal,
        labels: dict[str, str] | None = None,
    ) -> RunSnapshot:
        inputs = RunInputs(
            files=[
                InputFile(
                    file_id=file.file_id,
                    revision_id=file.revision_id,
                    digest_sha256=file.digest_sha256,
                )
                for file in files
            ]
        )
        request = self.client.swarms.prepare_launch(
            message, budget=budget_usd, inputs=inputs, optimizer="auto", labels=labels
        )
        journal.begin(
            LaunchIntent(
                "sdk",
                idempotency_key,
                journal.request_digest(request.model_dump_json()),
                tuple(InputIdentity(file.name, file.digest_sha256) for file in files),
            )
        )
        snapshot = self.client.swarms.launch_request(request, idempotency_key=idempotency_key)
        journal.accepted(snapshot)
        if snapshot.optimizer is None:
            raise refusal(
                "optimizer_resolution_unobserved",
                stage="launch",
                cause="accepted run has no authoritative optimizer receipt; retain its identity and do not relaunch",
                run_id=snapshot.run_id,
            )
        return snapshot

    def launch_recorded(
        self, *, recording: RecordedLaunch, idempotency_key: str, journal: LaunchJournal
    ) -> RunSnapshot:
        # The recording omitted optimizer; J1 selects literal auto so Core resolves it
        # and the accepted run must carry the authoritative optimizer receipt.
        request = self.client.swarms.prepare_launch(
            recording.message,
            budget=recording.budget_usd,
            optimizer="auto",
            wall_time_seconds=recording.wall_time_seconds,
            labels=dict(recording.labels),
            resources=list(recording.resources),
        )
        journal.begin(
            LaunchIntent(
                "sdk", idempotency_key, journal.request_digest(request.model_dump_json()), ()
            )
        )
        snapshot = self.client.swarms.launch_request(request, idempotency_key=idempotency_key)
        journal.accepted(snapshot)
        if snapshot.optimizer is None:
            raise refusal(
                "optimizer_resolution_unobserved",
                stage="launch",
                cause="accepted run has no authoritative optimizer receipt; retain its identity and do not relaunch",
                run_id=snapshot.run_id,
            )
        return snapshot

    def wait(self, run_id: str) -> RunSnapshot:
        return self.client.swarms.wait(run_id, timeout=WAIT_SECONDS, raise_on_failure=False)

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self.client.swarms.get(run_id)

    def report(self, run_id: str) -> RunReport:
        return self.client.swarms.report(run_id)

    def result(self, run_id: str) -> RunResult:
        return self.client.swarms.result(run_id)

    def outputs(self, run_id: str) -> tuple[RunOutput, ...]:
        cursor: Optional[str] = None
        items: list[RunOutput] = []
        seen: set[str] = set()
        for _ in range(OUTPUT_PAGES_MAX):
            page = self.client.swarms.outputs(run_id, cursor=cursor, limit=PAGE_ITEMS)
            items.extend(page.items)
            if not page.has_more:
                return tuple(items)
            if page.next_cursor is None or page.next_cursor in seen:
                raise refusal(
                    "output_cursor_invalid",
                    stage="outputs",
                    cause="missing or repeated next cursor",
                    run_id=run_id,
                )
            cursor = page.next_cursor
            seen.add(cursor)
        raise refusal(
            "output_page_limit",
            stage="outputs",
            cause="bounded output page count exceeded",
            run_id=run_id,
        )

    def download(self, run_id: str, output: RunOutput, target: Path) -> Path:
        pin = content_pin(output)
        return self.client.swarms.download_output(
            run_id,
            output.output_id,
            str(target),
            revision_id=pin.revision_id,
            digest_sha256=pin.digest_sha256,
            max_bytes=OUTPUT_BYTES_MAX,
        )


@dataclass(frozen=True, slots=True)
class ContentPin:
    revision_id: str
    digest_sha256: str


def content_pin(output: RunOutput) -> ContentPin:
    if output.reference is None:
        raise refusal(
            "output_content_unavailable",
            stage="download",
            cause="authoritative output reference is absent",
            run_id=output.run_id,
        )
    # reference is the canonical SDK wire Mapping[str,str]; required keys have
    # one meaning, and an absent key is a refusal rather than a legacy shape.
    if "revision_id" not in output.reference or "digest_sha256" not in output.reference:
        raise refusal(
            "output_pin_unavailable",
            stage="download",
            cause="revision or content digest is absent",
            run_id=output.run_id,
        )
    return ContentPin(output.reference["revision_id"], output.reference["digest_sha256"])


class CLIDriver:
    def __init__(
        self,
        *,
        python: Path,
        environment: Mapping[str, str],
        working_directory: Path,
        entrypoint: Optional[Path] = None,
    ) -> None:
        self.python = python
        self.environment = environment
        self.working_directory = working_directory
        # An owned harness entrypoint may compose a public HTTP client and call
        # synth_ai.cli.main(..., http_client=...). It is still a real child CLI.
        self.entrypoint = entrypoint

    def _command(
        self,
        arguments: Sequence[str],
        *,
        run_id: str,
        seconds: int = COMMAND_SECONDS,
        terminal_exit: Optional[int] = None,
        exit_sink: Optional[list[int]] = None,
    ) -> str:
        # The caller supplies the manager-selected origin via the named local
        # profile. Credentials remain in the child environment, never argv/logs.
        entrypoint = ["-m", "synth_ai.cli"] if self.entrypoint is None else [str(self.entrypoint)]
        result = subprocess.run(
            [str(self.python), *entrypoint, "--env", "local", *arguments],
            cwd=self.working_directory,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=seconds,
            check=False,
        )
        if result.returncode and result.returncode != terminal_exit:
            # Retain the classified exit without printing server bodies or keys.
            raise refusal(
                "cli_command_failed",
                stage=arguments[0],
                cause="CLI exit " + str(result.returncode),
                run_id=run_id,
            )
        if exit_sink is not None:
            exit_sink.append(result.returncode)
        return result.stdout

    def upload(self, paths: Sequence[Path]) -> tuple[File, ...]:
        return TypeAdapter(tuple[File, ...]).validate_json(
            self._command(
                ["upload", "--json", *(str(path) for path in paths)], run_id="not_launched"
            )
        )

    def launch_auto(
        self,
        *,
        message: str,
        budget_usd: str,
        files: Sequence[Path],
        idempotency_key: str,
        journal: LaunchJournal,
        labels: dict[str, str] | None = None,
    ) -> RunSnapshot:
        arguments = [
            "run",
            message,
            "--budget",
            budget_usd,
            "--optimizer",
            "auto",
            "--idempotency-key",
            idempotency_key,
            "--json",
        ]
        for path in files:
            arguments.extend(["--file", str(path)])
        if labels is not None:
            for name, value in sorted(labels.items()):
                arguments.extend(["--label", f"{name}={value}"])
        journal.begin(
            LaunchIntent(
                "cli",
                idempotency_key,
                journal.request_digest(json.dumps(arguments)),
                tuple(
                    InputIdentity(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                    for path in files
                ),
            )
        )
        snapshot = RunSnapshot.model_validate_json(
            self._command(arguments, run_id="launch_outcome_pending")
        )
        journal.accepted(snapshot)
        if snapshot.optimizer is None:
            raise refusal(
                "optimizer_resolution_unobserved",
                stage="launch",
                cause="accepted run has no authoritative optimizer receipt; retain its identity and do not relaunch",
                run_id=snapshot.run_id,
            )
        return snapshot

    def launch_recorded(
        self,
        *,
        recording: RecordedLaunch,
        idempotency_key: str,
        journal: LaunchJournal,
        work_directory: Path,
    ) -> RunSnapshot:
        resources = work_directory / "resources.json"
        resources.write_text(recording.resources_json())
        arguments = [
            "run",
            recording.message,
            "--budget",
            recording.budget_usd,
            "--wall",
            str(recording.wall_time_seconds),
            "--resources",
            str(resources),
            "--optimizer",
            "auto",
            "--idempotency-key",
            idempotency_key,
            "--json",
        ]
        for key, value in sorted(recording.labels.items()):
            arguments.extend(["--label", key + "=" + value])
        journal.begin(
            LaunchIntent("cli", idempotency_key, journal.request_digest(json.dumps(arguments)), ())
        )
        snapshot = RunSnapshot.model_validate_json(
            self._command(arguments, run_id="launch_outcome_pending")
        )
        journal.accepted(snapshot)
        if snapshot.optimizer is None:
            raise refusal(
                "optimizer_resolution_unobserved",
                stage="launch",
                cause="accepted run has no authoritative optimizer receipt; retain its identity and do not relaunch",
                run_id=snapshot.run_id,
            )
        return snapshot

    def wait(self, run_id: str) -> RunSnapshot:
        # `synth wait` exits 3 (EXIT_RUN_UNSUCCESSFUL) after printing the final snapshot of a
        # failed or cancelled run: a terminal answer, not a command failure (P7 staging p7-cli-1).
        # The exit and the printed state must agree; any other non-zero exit stays a refusal.
        result: list[int] = []
        snapshot = RunSnapshot.model_validate_json(
            self._command(
                ["wait", run_id, "--timeout", str(WAIT_SECONDS), "--json"],
                run_id=run_id,
                seconds=WAIT_SECONDS + COMMAND_SECONDS,
                terminal_exit=CLI_EXIT_RUN_UNSUCCESSFUL,
                exit_sink=result,
            )
        )
        unsuccessful = str(snapshot.state) in CLI_UNSUCCESSFUL_STATES
        if (result[0] == CLI_EXIT_RUN_UNSUCCESSFUL) != unsuccessful or snapshot.run_id != run_id:
            raise refusal(
                "cli_wait_exit_state_mismatch",
                stage="wait",
                cause="CLI exit " + str(result[0]) + " with state " + str(snapshot.state),
                run_id=run_id,
            )
        return snapshot

    def snapshot(self, run_id: str) -> RunSnapshot:
        return RunSnapshot.model_validate_json(
            self._command(["status", run_id, "--json"], run_id=run_id)
        )

    def report(self, run_id: str) -> RunReport:
        return RunReport.model_validate_json(
            self._command(["report", run_id, "--json"], run_id=run_id)
        )

    def result(self, run_id: str) -> RunResult:
        return RunResult.model_validate_json(
            self._command(["result", run_id, "--json"], run_id=run_id)
        )

    def outputs(self, run_id: str) -> tuple[RunOutput, ...]:
        # Current CLI emits one OutputPage, preserving the cursor contract.
        from synth_ai.models import OutputPage

        cursor: Optional[str] = None
        items: list[RunOutput] = []
        seen: set[str] = set()
        for _ in range(OUTPUT_PAGES_MAX):
            arguments = ["outputs", run_id, "--json", "--limit", str(PAGE_ITEMS)]
            if cursor is not None:
                arguments.extend(["--cursor", cursor])
            page = OutputPage.model_validate_json(self._command(arguments, run_id=run_id))
            items.extend(page.items)
            if not page.has_more:
                return tuple(items)
            if page.next_cursor is None or page.next_cursor in seen:
                raise refusal(
                    "output_cursor_invalid",
                    stage="outputs",
                    cause="missing or repeated next cursor",
                    run_id=run_id,
                )
            cursor = page.next_cursor
            seen.add(cursor)
        raise refusal(
            "output_page_limit",
            stage="outputs",
            cause="bounded output page count exceeded",
            run_id=run_id,
        )

    def download(self, run_id: str, output: RunOutput, target: Path) -> Path:
        self._command(
            ["download", run_id, output.output_id, "--out", str(target.parent), "--json"],
            run_id=run_id,
        )
        return target


@dataclass(frozen=True, slots=True)
class Records:
    snapshot: RunSnapshot
    report: RunReport
    result: RunResult
    outputs: tuple[RunOutput, ...]
    missing: tuple[Missing, ...]


def observe_records(driver: RecordDriver, *, run_id: str) -> Records:
    """Read and cross-check the run records every surface exposes."""
    snapshot, report, result = driver.snapshot(run_id), driver.report(run_id), driver.result(run_id)
    if snapshot.run_id != run_id or report.run_id != run_id or result.run.run_id != run_id:
        raise refusal(
            "observation_run_mismatch",
            stage="observation",
            cause="surface returned another run identity",
            run_id=run_id,
        )
    try:
        attempts = project_attempts(result.attempt_tree, run_id=run_id)
    except TreeError as error:
        raise refusal(
            error.code, stage="attempt_tree", cause=error.record_id, run_id=run_id
        ) from error
    missing = [] if attempts is not None else [Missing.ATTEMPT_TREE]
    if not (snapshot.execution_closed and snapshot.accounting_final and snapshot.cleanup_complete):
        missing.append(Missing.FINALITY)
    if result.charge is None or report.settlement.total is None:
        missing.append(Missing.CHARGE)
    if result.charge is not None and report.settlement.total is not None:
        if result.charge.amount.currency != "USD" or report.settlement.total.currency != "USD":
            raise refusal(
                "charge_currency_invalid",
                stage="cost",
                cause="J1 cost comparison requires USD",
                run_id=run_id,
            )
        if Decimal(result.charge.amount.amount) != Decimal(report.settlement.total.amount):
            raise refusal(
                "charge_projection_mismatch",
                stage="cost",
                cause="result and report disagree on customer charge",
                run_id=run_id,
            )
    outputs = driver.outputs(run_id)
    names: set[str] = set()
    for output in outputs:
        if output.run_id != run_id:
            raise refusal(
                "output_run_mismatch",
                stage="outputs",
                cause="output belongs to another run",
                run_id=run_id,
            )
        if (
            Path(output.filename).name != output.filename
            or output.filename in {"", ".", ".."}
            or output.filename in names
        ):
            raise refusal(
                "output_filename_invalid",
                stage="outputs",
                cause="unsafe or duplicate destination name",
                run_id=run_id,
            )
        names.add(output.filename)
        content_pin(output)
    return Records(snapshot, report, result, outputs, tuple(missing))


def observe_listed(driver: RecordDriver, *, run_id: str) -> J1Observation:
    """Observe a surface that lists pinned outputs but cannot fetch their bytes."""
    records = observe_records(driver, run_id=run_id)
    missing = [*records.missing, Missing.RESULT_MARKDOWN]
    if records.outputs:
        missing.append(Missing.OUTPUT_CONTENT)
    return J1Observation(
        records.snapshot,
        records.report,
        records.result,
        records.outputs,
        (),
        None,
        tuple(missing),
    )


def observe(driver: ReadDriver, *, run_id: str, output_directory: Path) -> J1Observation:
    records = observe_records(driver, run_id=run_id)
    missing = list(records.missing)
    output_directory.mkdir(parents=True, exist_ok=True)
    evidence: list[OutputEvidence] = []
    markdown: Optional[str] = None
    for output in records.outputs:
        pin = content_pin(output)
        path = driver.download(run_id, output, output_directory / output.filename)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pin.digest_sha256:
            raise refusal(
                "output_digest_mismatch",
                stage="download",
                cause="downloaded bytes differ from pinned output",
                run_id=run_id,
            )
        evidence.append(OutputEvidence(output, path, digest))
        result_file = records.result.result_file
        if (
            result_file is not None
            and output.output_id == result_file.output_id
            and digest == result_file.reference.digest_sha256
        ):
            markdown = path.read_text(encoding="utf-8")
    if markdown is None:
        missing.append(Missing.RESULT_MARKDOWN)
    return J1Observation(
        records.snapshot,
        records.report,
        records.result,
        records.outputs,
        tuple(evidence),
        markdown,
        tuple(missing),
    )
