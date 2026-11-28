"""Real `synth` CLI argv builders for acceptance surface parity."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from .surfaces import Call

ArgvBuilder = Callable[["Call", "Path"], Sequence[str]]


def obj(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def text(body: Mapping[str, JsonValue], name: str) -> str | None:
    value = body.get(name)
    return value if isinstance(value, str) else None


def run_id(call: Call) -> str:
    return call.path["run_id"]


def continuation_parent(call: Call) -> str | None:
    if call.body is None or "continue_from" not in call.body:
        return None
    parent = call.body["continue_from"]
    if not isinstance(parent, str):
        raise ValueError("continue_from_invalid")
    return parent


def body_file(directory: Path, body: Mapping[str, JsonValue]) -> str:
    descriptor, name = tempfile.mkstemp(suffix=".json", dir=directory)
    with os.fdopen(descriptor, "w") as output:
        json.dump(body, output)
    return name


def text_file(directory: Path, content: str) -> str:
    descriptor, name = tempfile.mkstemp(suffix=".txt", dir=directory)
    with os.fdopen(descriptor, "w") as output:
        output.write(content)
    return name


def launch_argv(call: Call, directory: Path) -> Sequence[str]:
    body = dict(call.body or {})
    charge = obj(obj(body.get("limits")).get("maximum_charge"))
    message = str(body.get("message"))
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
    parent = continuation_parent(call)
    if parent is not None:
        argv += ["--continue-from", parent]
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


def message_argv(call: Call, _directory: Path) -> Sequence[str]:
    body = call.body
    kind = text(body, "kind") if body is not None else None
    message_text = text(body, "text") if body is not None else None
    if kind is None or message_text is None or call.idempotency_key is None:
        raise ValueError("message_request_invalid")
    return [
        "message",
        run_id(call),
        message_text,
        "--kind",
        kind,
        "--idempotency-key",
        call.idempotency_key,
    ]


def history_page_argv(call: Call, _directory: Path) -> Sequence[str]:
    argv: list[str] = ["billing", "history", "--limit", str(call.query.get("limit", 20))]
    for name in ("since", "until", "cursor"):
        value = call.query.get(name)
        if isinstance(value, str):
            argv += ["--" + name, value]
    return argv


def message_page_argv(call: Call, _directory: Path) -> Sequence[str]:
    argv = ["messages", run_id(call), "--limit", str(call.query.get("limit", 20))]
    cursor = call.query.get("cursor")
    if isinstance(cursor, str):
        argv += ["--cursor", cursor]
    return argv


# Real `synth` commands. An operation absent here is uncovered for the CLI.
CLI_ARGV: Mapping[str, ArgvBuilder] = {
    "getAccount": lambda a, d: ["organization", "get"],
    "getBalance": lambda a, d: ["billing", "balance"],
    "listStatements": lambda a, d: ["billing", "statements"],
    "listBillingHistory": history_page_argv,
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
    "sendRunMessage": message_argv,
    "listRunMessages": message_page_argv,
    "getRunMessage": lambda a, d: ["messages", run_id(a), "--message", a.path["message_id"]],
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
    "listRunApprovals": lambda a, d: ["runs", "approvals", run_id(a)],
    "decideRunApproval": lambda a, d: [
        "runs",
        "decide-approval",
        run_id(a),
        a.path["approval_id"],
        "--body",
        body_file(d, a.body or {"decision": "reject"}),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "listRunTasks": run_verb("tasks"),
    "getRunProgram": inspect_view("program"),
    "listRunOutputs": run_verb("outputs"),
    "getRunOutput": lambda a, d: [
        *inspect_view("output")(a, d),
        "--output-id",
        a.path["output_id"],
    ],
    "getRunOutputContent": lambda a, d: [
        "content",
        run_id(a),
        a.path["output_id"],
        "--revision",
        str(a.query["revision_id"]),
        "--sha256",
        str(a.query["digest_sha256"]),
        "--out",
        str(d / "isolation-content"),
    ],
    "waitRun": lambda a, d: [*inspect_view("wait-update")(a, d), "--timeout-seconds", "1"],
    "updateRunLabels": lambda a, d: ["labels", run_id(a), "--body", body_file(d, a.body or {})],
    "getUpload": lambda a, d: ["uploads", "get", a.path["upload_id"]],
    "finalizeUpload": lambda a, d: [
        "uploads",
        "finalize",
        a.path["upload_id"],
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "withdrawPlanRequest": lambda a, d: ["plan", "withdraw", a.path["request_id"]],
    "revokeOAuthGrant": lambda a, d: ["grants", "revoke", a.path["grant_id"]],
    "updateApiKey": lambda a, d: [
        "keys",
        "update",
        a.path["key_id"],
        "--body",
        body_file(d, a.body or {}),
    ],
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
    "getRunReview": inspect_view("review"),
    "getArtifact": lambda a, d: ["artifacts", "get", a.path["artifact_id"]],
    "updateArtifact": lambda a, d: [
        "artifacts",
        "update",
        a.path["artifact_id"],
        "--summary",
        str((a.body or {}).get("summary", "probe")),
        "--file",
        text_file(d, str((a.body or {}).get("content", "<p>probe</p>"))),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "getArtifactContent": lambda a, d: ["artifacts", "content", a.path["artifact_id"]],
    "shareArtifact": lambda a, d: [
        "artifacts",
        "share",
        a.path["artifact_id"],
        "--scope",
        str((a.body or {}).get("scope", "org")),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "revokeArtifactShare": lambda a, d: ["artifacts", "revoke-share", a.path["artifact_id"]],
    "listNotebookArtifacts": lambda a, d: [
        "artifacts",
        "list",
        "--notebook-id",
        a.path["notebook_id"],
    ],
    "createExperiment": lambda a, d: [
        "experiments",
        "create",
        "--notebook-id",
        a.path["notebook_id"],
        "--run-id",
        str((a.body or {}).get("run_id", "")),
        "--title",
        str((a.body or {}).get("title", "probe")),
        "--specification",
        str((a.body or {}).get("specification", "probe")),
        "--arm",
        "baseline",
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "listNotebookExperiments": lambda a, d: [
        "experiments",
        "list",
        "--notebook-id",
        a.path["notebook_id"],
    ],
    "getExperiment": lambda a, d: ["experiments", "get", a.path["experiment_id"]],
    "reviseExperiment": lambda a, d: [
        "experiments",
        "revise",
        a.path["experiment_id"],
        "--title",
        str((a.body or {}).get("title", "revised")),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "initResearchLog": lambda a, d: [
        "research-log",
        "init",
        "--notebook-id",
        a.path["notebook_id"],
        "--title",
        str((a.body or {}).get("title", "probe")),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "appendResearchLogEntry": lambda a, d: [
        "research-log",
        "append",
        "--notebook-id",
        a.path["notebook_id"],
        "--category",
        str((a.body or {}).get("category", "observation")),
        "--text",
        str((a.body or {}).get("text", "probe")),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "searchResearchLog": lambda a, d: [
        "research-log",
        "search",
        "--notebook-id",
        a.path["notebook_id"],
    ],
    "correctResearchLogEntry": lambda a, d: [
        "research-log",
        "correct",
        a.path["entry_id"],
        "--notebook-id",
        a.path["notebook_id"],
        "--text",
        str((a.body or {}).get("text", "correction")),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "exportResearchLog": lambda a, d: [
        "research-log",
        "export",
        "--notebook-id",
        a.path["notebook_id"],
    ],
    "listNotebooks": lambda a, d: ["notebooks", "list"],
    "createArtifactoryRecord": lambda a, d: [
        "artifactory",
        "create",
        "--notebook-id",
        a.path["notebook_id"],
        "--title",
        str((a.body or {}).get("title", "probe")),
        "--category",
        "hypothesis",
        "--origin",
        "agent",
        "--file",
        body_file(d, {"statement": "Tenant-isolation probe."}),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
    "listArtifactoryRecords": lambda a, d: [
        "artifactory",
        "list",
        "--notebook-id",
        a.path["notebook_id"],
    ],
    "resolveEffectivePreference": lambda a, d: [
        "artifactory",
        "preference",
        "--notebook-id",
        a.path["notebook_id"],
    ],
    "getArtifactoryRecord": lambda a, d: ["artifactory", "get", a.path["record_id"]],
    "reviseArtifactoryRecord": lambda a, d: [
        "artifactory",
        "revise",
        a.path["record_id"],
        "--summary",
        str((a.body or {}).get("summary", "Tenant-isolation probe.")),
        "--idempotency-key",
        str(a.idempotency_key),
    ],
}
