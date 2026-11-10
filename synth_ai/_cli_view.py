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
    LimitList,
    Money,
    OutputPage,
    RunControlReceipt,
    RunEvent,
    RunOutput,
    RunPage,
    RunProfileCatalog,
    RunReport,
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
    return facts(
        [
            ("run", value.run_id),
            ("state", value.state),
            ("profile", value.profile.id if value.profile is not None else None),
            ("model", value.model),
            ("result", "available" if value.result_available else "pending"),
            ("created", value.created_at),
            ("closed", value.closed_at),
        ]
    )


def receipt(value: RunControlReceipt) -> str:
    return facts([("run", value.resource_id), ("command", value.command), ("state", value.state)])


def event(value: RunEvent) -> str:
    return "  ".join([str(value.sequence).rjust(6), value.observed_at or "-", value.kind])


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
    return text


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
