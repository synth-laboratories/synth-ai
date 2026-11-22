"""Human-readable CLI rendering and the per-run event checkpoint store."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ._auth_models import AuthStatus
from ._generated.models import (
    Account,
    Balance,
    EventPage,
    File,
    LaunchChoices,
    LimitList,
    LimitMeter,
    Money,
    OAuthGrantRevoked,
    OutputPage,
    PlanRequest,
    PlanRequestList,
    RunAllocation,
    RunAllocationChoice,
    RunCharge,
    RunControlReceipt,
    RunEvent,
    RunLimitsProgress,
    RunMessage,
    RunMessagePage,
    RunOutput,
    RunPage,
    RunProfileCatalog,
    RunReport,
    RunResult,
    RunSnapshot,
    RunTaskPage,
    UsageRecordPage,
    UsageSummary,
)
from ._stream import StreamCheckpoint, StreamCheckpointData


class SavedOutput(BaseModel):
    """One downloaded run output and where it was written."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    output: RunOutput
    path: str


def money(value: Money | None) -> str:
    return "-" if value is None else value.amount + " " + value.currency


def facts(rows: Sequence[tuple[str, object]]) -> str:
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        label.ljust(width) + "  " + ("-" if value is None else str(value)) for label, value in rows
    )


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [list(header), *[["-" if c is None else str(c) for c in row] for row in rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in cells
    )


def snapshot(value: RunSnapshot) -> str:
    rows: list[tuple[str, object]] = [
        ("run", value.run_id),
        ("state", value.state),
        ("profile", value.profile.id if value.profile is not None else None),
        ("model", value.model),
        ("result", "available" if value.result_available else "pending"),
        ("created", value.created_at),
        ("closed", value.closed_at),
    ]
    # Contract 1.32.0: only a run waiting for capacity carries its queue standing.
    if value.queue is not None:
        rows += [
            ("queue", f"position {value.queue.position} ({value.queue.ahead} ahead)"),
            ("starts", value.queue.estimated_start_at),
            ("expires", value.queue.expires_at),
        ]
    return facts(rows)


def receipt(value: RunControlReceipt) -> str:
    return facts([("run", value.resource_id), ("command", value.command), ("state", value.state)])


def event(value: RunEvent) -> str:
    return "  ".join([str(value.sequence).rjust(6), value.observed_at or "-", value.kind])


def message(value: RunMessage) -> str:
    """Keep retention unavailability distinct from a blank message."""
    text = (
        value.body.text
        if value.body.kind == "available"
        else "message no longer available (retention)"
    )
    return "  ".join([value.message_id, value.author, value.state.kind, text])


def messages(value: RunMessagePage) -> str:
    if value.availability.kind == "unavailable":
        return value.availability.reason
    return "\n".join(message(item) for item in value.items)


def events(value: EventPage) -> str:
    return "\n".join(event(item) for item in value.items)


def runs(value: RunPage) -> str:
    return table(
        ["RUN", "STATE", "PROFILE", "CREATED"],
        [
            [r.run_id, r.state, r.profile.id if r.profile is not None else None, r.created_at]
            for r in value.items
        ],
    )


def tasks(value: RunTaskPage) -> str:
    return table(["TASK", "STATE"], [[t.task_id, t.state] for t in value.items])


def outputs(value: OutputPage) -> str:
    return table(
        ["OUTPUT", "STATE", "FILENAME"], [[o.output_id, o.state, o.filename] for o in value.items]
    )


def profiles(value: RunProfileCatalog) -> str:
    return table(
        ["PROFILE", "NAME", "AVAILABLE", "RESEARCH TASK", "DEFAULT"],
        [
            [
                p.id,
                p.name,
                "yes" if p.available else "no",
                "required" if p.requires_research_task else "-",
                "yes" if p.id == value.default_profile else "",
            ]
            for p in value.profiles
        ],
    )


def files(value: Sequence[File]) -> str:
    return table(["FILE", "NAME", "BYTES"], [[f.file_id, f.name, f.size_bytes] for f in value])


def saved(value: Sequence[SavedOutput]) -> str:
    return table(["OUTPUT", "PATH"], [[s.output.output_id, s.path] for s in value])


def launch_choices(value: LaunchChoices) -> str:
    text = facts(
        [
            ("menu revision", value.menu_revision),
            ("run profile", value.run_profile),
            ("domicile", value.domicile_requirement.value),
            ("affordability", value.affordability.status.value),
        ]
    )
    text += "\n\n" + table(
        ["ACTOR", "ROLE", "DEFAULT INFERENCE", "DEFAULT SANDBOX", "PLANNER MAY DECIDE"],
        [
            [
                r.actor_id,
                r.role.value,
                r.default_inference_profile,
                r.default_sandbox_profile or "planner decides",
                "yes" if r.planner_decides_allowed else "no",
            ]
            for r in value.roles
        ],
    )
    text += "\n\n" + table(
        ["INFERENCE", "MODEL", "EFFORT", "INPUT /1M", "OUTPUT /1M", "PER CALL CAP"],
        [
            [
                p.profile_id,
                p.model,
                p.effort,
                money(p.input_price_per_million_tokens),
                money(p.output_price_per_million_tokens),
                money(p.per_call_spend_cap),
            ]
            for p in value.inference_profiles
        ],
    )
    text += "\n\n" + table(
        ["SANDBOX", "SIZE", "DOMICILE", "HOURLY", "OPTIONS"],
        [
            [
                s.profile_id,
                s.size_id,
                s.domicile.value,
                money(s.hourly_price),
                ("choose one of " if s.option_required else "") + str(len(s.options)),
            ]
            for s in value.sandbox_profiles
        ],
    )
    if value.exclusions:
        text += "\n\n" + table(
            ["EXCLUDED", "KIND", "OPTION", "CAUSE"],
            [[e.profile_id, e.kind.value, e.option_id, e.cause.value] for e in value.exclusions],
        )
    return text


def _choice(value: RunAllocationChoice) -> str:
    chosen = "planner decides" if value.profile_id is None else value.profile_id
    return chosen + " (" + value.source.value + ")"


def allocation(value: RunAllocation) -> str:
    text = facts(
        [
            ("run", value.run_id),
            ("menu revision", value.menu_revision),
            ("final", "yes" if value.final else "no"),
        ]
    )
    if value.actors:
        text += "\n\n" + table(
            ["ACTOR", "INFERENCE", "SANDBOX", "PLACED", "REPLACED", "CALLS", "COMPUTE"],
            [
                [
                    a.actor_id,
                    _choice(a.launch.inference),
                    _choice(a.launch.sandbox),
                    a.placements[-1].profile_id if a.placements else None,
                    a.replacements,
                    a.usage.calls,
                    money(a.compute_charge)
                    if a.compute_charge is not None
                    else a.compute_metering.value,
                ]
                for a in value.actors
            ],
        )
    return text


def report(value: RunReport) -> str:
    rows: list[tuple[str, object]] = [
        ("run", value.run_id),
        ("settlement", value.settlement.status.value),
        ("charge", money(value.settlement.total)),
        ("reward", value.reward.status.value),
        ("teardown", value.teardown.value),
    ]
    text = facts(rows)
    if value.by_model:
        text += "\n\n" + table(
            ["PROVIDER", "MODEL", "CALLS", "CHARGE"],
            [[m.provider, m.model, m.calls, money(m.charge)] for m in value.by_model],
        )
    if value.sandboxes:
        text += "\n\n" + table(
            ["ACTOR", "SIZE", "DOMICILE", "READY MS", "COMPUTE"],
            [
                [
                    item.profile.actor_id if item.profile is not None else item.role.value,
                    item.profile.size_id
                    if item.profile is not None
                    else f"{item.cpu_millis / 1000:g} CPU / {item.memory_bytes / (1 << 30):g} GiB",
                    item.profile.domicile.value if item.profile is not None else "unknown",
                    item.time_to_ready_ms if item.time_to_ready_ms is not None else "unknown",
                    money(item.compute_charge)
                    if item.compute_charge is not None
                    else item.compute_metering.value,
                ]
                for item in value.sandboxes
            ],
        )
    return text


def charge_line(value: RunCharge | None) -> str | None:
    """One closing line: the run's charge and whether it is final."""
    if value is None:
        return None
    return "charge  " + money(value.amount) + (" (final)" if value.final else " (not final)")


def result(value: RunResult) -> str:
    """The answer text alone, so stdout can be captured; the CLI prints the charge on stderr."""
    if value.answer is not None:
        return value.answer
    summary = facts(
        [
            ("run", value.run.run_id),
            ("state", value.run.state),
            ("answer", "none (no available text output or objective answer)"),
        ]
    )
    return summary


def status_line(value: RunResult) -> str:
    """The run's authoritative terminal outcome (completed, partial, failed or cancelled)."""
    return "status  " + value.status


def accounting_line(value: RunResult) -> str:
    """The final ledger-backed charge, or that Core has no accounting reader for it."""
    if value.accounting is None:
        return "accounting  unavailable"
    return "accounting  " + money(value.accounting.amount) + " (final)"


def result_file_line(value: RunResult) -> str | None:
    """The run's result.md output, read with `download RUN OUTPUT`; None when unavailable."""
    if value.result_file is None:
        return None
    return "result.md  " + value.result_file.output_id


def result_json(value: RunResult) -> str:
    """`RunResult` as Core returned it: run_id, status, accounting, the answer and the charge."""
    return value.model_dump_json(exclude_unset=True)


def usage_summary(value: UsageSummary) -> str:
    return facts(
        [
            ("since", value.since),
            ("until", value.until),
            ("charge", money(value.total.charge)),
            ("input tokens", value.total.input_tokens),
            ("output tokens", value.total.output_tokens),
            ("records", value.total.records),
        ]
    )


def usage_records(value: UsageRecordPage) -> str:
    return table(
        ["RECORD", "KIND", "MODEL", "INPUT", "OUTPUT", "CHARGE"],
        [
            [r.record_id, r.kind.value, r.preset, r.input_tokens, r.output_tokens, money(r.charge)]
            for r in value.items
        ],
    )


def balance(value: Balance) -> str:
    return facts(
        [
            ("available", money(value.available)),
            ("ledger", money(value.ledger)),
            ("held", money(value.held)),
            ("headroom", money(value.headroom)),
        ]
    )


def limits(value: LimitList) -> str:
    return facts([("plan", value.plan)]) + (
        "\n\n"
        + table(
            ["RESOURCE", "METRIC", "WINDOW", "CURRENT", "CAP"],
            [[i.resource, i.metric, i.window, i.current, i.cap] for i in value.items],
        )
        if value.items
        else ""
    )


def plan_request(value: PlanRequest) -> str:
    return facts(
        [
            ("request", value.request_id),
            ("state", value.state),
            ("created", value.created_at),
            ("decided", value.decided_at),
            ("reason", value.decision_reason),
        ]
    )


def oauth_grant_revoked(value: OAuthGrantRevoked) -> str:
    return facts([("grant", value.grant_id), ("status", value.status), ("outcome", value.outcome)])


def plan_requests(value: PlanRequestList) -> str:
    return table(
        ["REQUEST", "STATE", "CREATED", "DECIDED", "REASON"],
        [
            [i.request_id, i.state, i.created_at, i.decided_at, i.decision_reason]
            for i in value.items
        ],
    )


def account(value: Account) -> str:
    return facts(
        [
            ("org", value.org.name),
            ("org id", value.org.id),
            ("plan", value.org.plan),
            ("user", value.principal.user_id),
            ("credential", value.principal.credential_kind.value),
        ]
    )


def auth_status(value: AuthStatus) -> str:
    return facts(
        [
            ("api", value.base_url),
            ("signed in", "yes" if value.authenticated else "no"),
            ("source", value.source.value),
        ]
    )


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def state_dir() -> Path:
    configured = os.getenv("SYNTH_STATE_DIR")
    return (
        Path(configured).expanduser()
        if configured is not None and configured != ""
        else Path.home() / ".config" / "synth"
    )


def checkpoint_path(run_id: str) -> Path:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run id cannot name a checkpoint file")
    return state_dir() / "streams" / (run_id + ".json")


def load_checkpoint(run_id: str) -> StreamCheckpoint:
    path = checkpoint_path(run_id)
    if not path.exists():
        return StreamCheckpoint(run_id=run_id)
    if path.stat().st_mode & 0o077:
        raise ValueError("checkpoint file must have permissions 0600")
    checkpoint = StreamCheckpoint.from_data(
        StreamCheckpointData.model_validate_json(path.read_bytes())
    )
    checkpoint.bind(run_id)
    return checkpoint


def save_checkpoint(checkpoint: StreamCheckpoint) -> None:
    if checkpoint.run_id is None:
        raise ValueError("checkpoint is not bound to a run")
    path = checkpoint_path(checkpoint.run_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(checkpoint.to_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run_limits(value: RunLimitsProgress) -> str:
    def amount(number: int | None) -> str:
        return "unknown" if number is None else str(number)

    lines = [f"{value.run_id}  revision {value.revision}  {value.status}"]

    def rows(label: str, meters: list[LimitMeter]) -> None:
        lines.append(label)
        lines.append(
            "Limit | Cap | Used | Reserved | Unknown reserved | Unknown receipts | Remaining"
        )
        for meter in meters:
            lines.append(
                " | ".join(
                    [
                        meter.dimension,
                        amount(
                            meter.effective_cap if meter.effective_cap is not None else meter.cap
                        ),
                        amount(meter.used),
                        amount(meter.reserved),
                        amount(meter.unknown_reserved),
                        amount(meter.unknown_receipts),
                        amount(meter.remaining),
                    ]
                )
            )

    rows("Primary", value.primary)
    rows("Secondary", value.secondary)
    for actor in value.actors:
        rows(f"{actor.assignment}  {actor.state or 'unknown'}", actor.limits)
    return "\n".join(lines)
