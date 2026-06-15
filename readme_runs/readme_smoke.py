#!/usr/bin/env python3
"""Run ReportBench README smoke through the public ``synth-ai`` Research SDK.

Lane contract and scoring helpers live in ``evals/reportbench/readme_smoke_harness``.
This script owns every SDK call: limits, project create, setup, launch, poll,
workspace download, and terminal scoring.

Per-run artifacts land under ``readme_runs/runs/<timestamp>_<target>/`` (see
``readme_runs/runs/README.md``). A ``latest`` symlink points at the most recent run.

Typical T1 (slot1, Codex gpt-5.4-mini worker + gpt-5.3-codex-spark post-run judge):

```bash
cd ~/Documents/GitHub/synth-ai
uv sync --group dev
export SYNTH_API_KEY=...  # or use --use-default-slot1
uv run python readme_runs/readme_smoke.py --use-default-slot1
# or: bash scripts/run_readme_smoke_slot1.sh
```

# See: Jstack daily note research_api_alpha_release_plan.md (T1 acceptance).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from synth_ai import SynthClient
from synth_ai.research import (
    ResearchApiError,
    ResearchClient,
    ResearchControlClient,
    ResearchHostKind,
    ResearchWorkMode,
)
from synth_ai.research.errors import (
    ResearchConcurrentRunLimitExceededError,
    ResearchInsufficientCreditsError,
    ResearchLimitExceededError,
    ResearchProjectMonthlyBudgetExhaustedError,
    ResearchStructuredDenialError,
)

LogFn = Callable[[str], None]

_POLL_PREFIX_RE = re.compile(r"^\[poll\](?P<still> still)? ")
_POLL_KV_RE = re.compile(r"(\w+)=('[^']*'|\S+)")
_POLL_NOISE_KEYS = frozenset({"last_progress_at", "next_poll_s", "stale_for_s"})
_BOUNDARY_ERROR_FIELD_SUFFIX = "_error"


def _record_boundary_degradation(
    container: dict[str, Any],
    *,
    field_name: str,
    operation: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Persist a deliberate non-fatal boundary failure on the smoke summary.

    # See: specifications/tanha/references/synthstyle.md (Errors: precise messages)
    """
    from reportbench.readme_smoke_harness import boundary_degradation_error

    payload = boundary_degradation_error(operation=operation, exc=exc)
    container[field_name] = payload
    return payload


def _collect_boundary_degradation_lines(summary: dict[str, Any]) -> list[str]:
    from reportbench.readme_smoke_harness import format_boundary_error_message

    lines: list[str] = []
    for key, value in sorted(summary.items()):
        if not str(key).endswith(_BOUNDARY_ERROR_FIELD_SUFFIX):
            continue
        message = format_boundary_error_message(value)
        if message:
            lines.append(message)
    return lines


def _reformat_log_message(msg: str) -> str:
    m = _POLL_PREFIX_RE.match(msg)
    if not m:
        return msg
    kvs = _POLL_KV_RE.findall(msg[m.end() :])
    kv_dict = {k: v.strip("'") for k, v in kvs}
    state = kv_dict.get("state", "unknown")
    try:
        elapsed = int(kv_dict.get("elapsed_s", "0"))
    except ValueError:
        elapsed = 0
    minutes, secs = divmod(elapsed, 60)
    elapsed_str = f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"
    prefix = "still " if m.group("still") else ""
    skip = {"state", "elapsed_s"} | _POLL_NOISE_KEYS
    extra = " ".join(f"{k}={v}" for k, v in kvs if k not in skip)
    suffix = f"  {extra}" if extra else ""
    return f"  [poll {prefix}+{elapsed_str}] {state}{suffix}"


DEFAULT_CODEX_VERIFIER_MODEL = "gpt-5.3-codex-spark"
DEFAULT_CODEX_VERIFIER_REASONING_EFFORT = "low"
DEFAULT_CODEX_VERIFIER_PASS_THRESHOLD = 0.99


@dataclass(frozen=True)
class ReadmeSmokeLaunch:
    """Resolved launch target for one README smoke run."""

    target: str
    backend: str
    api_key: str
    host_kind: ResearchHostKind
    worker_pool_id: str
    slot_id: str | None
    slot_mode: str | None
    slot_contract: Any | None
    local_eval_contract: Any | None


def _resolve_evals_root() -> Path:
    env_root = os.environ.get("EVALS_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    workspace = (
        Path(os.environ.get("SYNTH_WORKSPACE_ROOT") or Path(__file__).resolve().parent.parent)
        .expanduser()
        .resolve()
    )
    return (workspace / "evals").resolve()


def _ensure_evals_importable(evals_root: Path) -> None:
    if not evals_root.is_dir():
        raise FileNotFoundError(
            "evals checkout not found. Set EVALS_ROOT or SYNTH_WORKSPACE_ROOT "
            f"(looked for {evals_root})."
        )
    workspace_root = (
        Path(os.environ.get("SYNTH_WORKSPACE_ROOT") or evals_root.parent).expanduser().resolve()
    )
    # ``reportbench`` resolves from the evals repo root; ``evals.*`` needs the parent
    # on sys.path because the checkout directory is named ``evals``.
    for path in (evals_root, workspace_root):
        root_text = str(path)
        if path.is_dir() and root_text not in sys.path:
            sys.path.insert(0, root_text)


def _utcstamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def synth_ai_repo_root() -> Path:
    """Synth-ai checkout root (parent of ``readme_runs/``)."""
    return Path(__file__).resolve().parent.parent


def readme_runs_root() -> Path:
    return synth_ai_repo_root() / "readme_runs"


def readme_runs_dir() -> Path:
    """Directory that holds one folder per smoke run."""
    return readme_runs_root() / "runs"


def _safe_target_label(target: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target or "").strip())
    return cleaned or "local"


def default_run_output_root(*, target: str) -> Path:
    """``readme_runs/runs/<UTC>_<target>/`` for a new smoke run."""
    run_dir = readme_runs_dir() / f"{_utcstamp()}_{_safe_target_label(target)}"
    return run_dir.expanduser().resolve()


def resolve_output_root(path: str | None, *, target: str) -> Path:
    if path and str(path).strip():
        return Path(path).expanduser().resolve()
    return default_run_output_root(target=target)


def _parse_iso_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _wall_elapsed_seconds(summary: dict[str, Any]) -> float | None:
    started = _parse_iso_timestamp(summary.get("started_at"))
    finished = _parse_iso_timestamp(summary.get("finished_at"))
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _cost_usd_from_run_usage(summary: dict[str, Any]) -> float | None:
    run_usage = summary.get("run_usage")
    if not isinstance(run_usage, dict):
        return None
    cost = run_usage.get("cost")
    if isinstance(cost, dict):
        for key in ("total_usd", "charged_usd", "internal_cost_usd"):
            raw = cost.get(key)
            if isinstance(raw, (int, float)):
                return float(raw)
    for key in ("total_cost_usd", "total_charged_usd", "total_internal_cost_usd"):
        raw = run_usage.get(key)
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def _verifier_judge_cost_usd(summary: dict[str, Any]) -> float | None:
    verifier = summary.get("reportbench_verifier")
    if not isinstance(verifier, dict):
        return None
    judge = verifier.get("judge")
    if isinstance(judge, dict):
        raw = judge.get("cost_usd")
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def _format_duration_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    if minutes:
        return f"{minutes}m {secs}s ({seconds:.1f}s)"
    return f"{secs}s ({seconds:.1f}s)"


def _format_usd(amount: float | None) -> str:
    if amount is None:
        return "unknown"
    return f"${amount:.4f}"


def _format_count_map(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "-"
    return ",".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _observability_digest_from_snapshot(snapshot: Any) -> dict[str, Any]:
    lifecycle = getattr(snapshot, "lifecycle", None)
    terminal_outcome = str(getattr(lifecycle, "terminal_outcome", "") or "").strip() or None
    if not terminal_outcome:
        terminal_outcome = str(getattr(snapshot, "terminal_outcome", "") or "").strip() or None
    if terminal_outcome is not None and hasattr(terminal_outcome, "value"):
        terminal_outcome = str(terminal_outcome.value)

    actors = getattr(snapshot, "actors", None)
    tasks = getattr(snapshot, "tasks", None)
    runtime = getattr(snapshot, "runtime", None)
    anomalies = getattr(snapshot, "anomalies", None) or []

    liveness_phase = getattr(snapshot, "liveness_phase", None)
    if liveness_phase is not None and hasattr(liveness_phase, "value"):
        liveness_phase = liveness_phase.value

    public_state = getattr(snapshot, "public_state", None)
    if public_state is not None and hasattr(public_state, "value"):
        public_state = public_state.value

    task_keys: list[str] = []
    for item in getattr(tasks, "items", None) or []:
        task_key = getattr(item, "task_key", None)
        if task_key:
            task_keys.append(str(task_key))

    return {
        "public_state": str(public_state or "").strip() or None,
        "liveness_phase": str(liveness_phase or "").strip() or None,
        "terminal_outcome": terminal_outcome,
        "completion_classifier": str(getattr(snapshot, "completion_classifier", "") or "").strip()
        or None,
        "status_reason": str(getattr(snapshot, "status_reason", "") or "").strip() or None,
        "actors_by_role": dict(getattr(actors, "counts_by_role", {}) or {}),
        "actors_by_state": dict(getattr(actors, "counts_by_state", {}) or {}),
        "tasks_by_state": dict(getattr(tasks, "counts_by_state", {}) or {}),
        "task_keys": task_keys,
        "actor_count": int(getattr(actors, "total_count", 0) or 0),
        "task_count": int(getattr(tasks, "total_count", 0) or 0),
        "anomaly_count": len(anomalies) if isinstance(anomalies, list) else 0,
        "last_progress_at": str(getattr(runtime, "last_progress_at", "") or "").strip() or None,
    }


def _format_observability_digest_line(digest: dict[str, Any]) -> str:
    outcome = digest.get("terminal_outcome") or digest.get("public_state") or "unknown"
    roles = _format_count_map(digest.get("actors_by_role"))
    task_states = _format_count_map(digest.get("tasks_by_state"))
    anomalies = digest.get("anomaly_count", 0)
    liveness = digest.get("liveness_phase")
    liveness_bit = f" liveness={liveness}" if liveness else ""
    task_keys = digest.get("task_keys")
    task_key_bit = ""
    if isinstance(task_keys, list) and task_keys:
        task_key_bit = f" task_keys={','.join(str(key) for key in task_keys)}"
    return (
        f"[o11y] outcome={outcome}{liveness_bit} "
        f"roles={roles} task_states={task_states}{task_key_bit} anomalies={anomalies}"
    )


def _fetch_terminal_observability_snapshot(
    client: ResearchControlClient,
    project_id: str,
    run_id: str,
) -> Any:
    return client.get_run_observability_snapshot(
        project_id,
        run_id,
        detail_level="control",
        event_limit=20,
        actor_limit=20,
        task_limit=20,
        question_limit=5,
        timeline_limit=5,
        message_limit=5,
    )


def _fetch_terminal_observability_digest(
    client: ResearchControlClient,
    project_id: str,
    run_id: str,
) -> dict[str, Any]:
    return _observability_digest_from_snapshot(
        _fetch_terminal_observability_snapshot(client, project_id, run_id)
    )


def _persist_terminal_run_evidence(
    *,
    client: ResearchControlClient,
    summary: dict[str, Any],
    project_id: str,
    run_id: str,
) -> None:
    """Capture snapshot, usage, and task events for failed runs (verifier may be skipped)."""
    from reportbench.readme_smoke_harness import jsonish

    try:
        snapshot = _fetch_terminal_observability_snapshot(client, project_id, run_id)
        summary["final_observability_snapshot"] = jsonish(snapshot)
        summary["run_observability_digest"] = _observability_digest_from_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        _record_boundary_degradation(
            summary,
            field_name="final_observability_snapshot_error",
            operation="get_run_observability_snapshot",
            exc=exc,
        )

    try:
        summary["actor_usage"] = jsonish(client.get_run_actor_usage(run_id))
    except Exception as exc:  # noqa: BLE001
        _record_boundary_degradation(
            summary,
            field_name="actor_usage_error",
            operation="get_run_actor_usage",
            exc=exc,
        )

    try:
        summary["task_events"] = jsonish(client.list_run_task_events(project_id, run_id, limit=100))
    except Exception as exc:  # noqa: BLE001
        _record_boundary_degradation(
            summary,
            field_name="task_events_error",
            operation="list_run_task_events",
            exc=exc,
        )


def _row_status(row: dict[str, Any]) -> str:
    for key in ("status", "state", "public_task_state", "evaluation_state"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _summarize_rows(
    rows: Any,
    *,
    trim_fields: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {"total": 0, "by_status": {}, "items": []}
    by_status: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = _row_status(row)
        by_status[status] = by_status.get(status, 0) + 1
        item = {
            field: row.get(field)
            for field in trim_fields
            if row.get(field) is not None and str(row.get(field)).strip()
        }
        if item:
            items.append(item)
    return {"total": len(rows), "by_status": by_status, "items": items[:8]}


def _summarize_transcript(transcript: Any) -> dict[str, Any]:
    if not isinstance(transcript, dict):
        return {"event_count": 0, "by_kind": {}}
    events = transcript.get("events")
    if not isinstance(events, list):
        count_raw = transcript.get("event_count")
        try:
            event_count = int(count_raw or 0)
        except (TypeError, ValueError):
            event_count = 0
        return {"event_count": event_count, "by_kind": {}}
    by_kind: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"event_count": len(events), "by_kind": by_kind}


def _normalize_archive_path(path: str) -> str:
    return path.lstrip("./").strip("/")


def _normalize_workspace_compare_path(path: str) -> str:
    normalized = _normalize_archive_path(path)
    if normalized.startswith("project/"):
        return normalized[len("project/") :]
    return normalized


def _paths_from_workspace_archive(archive_path: Path) -> list[str]:
    if not archive_path.is_file():
        return []
    paths: list[str] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                paths.append(_normalize_archive_path(member.name))
    return sorted(paths)


def _starting_paths_from_output_root(output_root: Path) -> set[str]:
    paths: set[str] = set()
    starting_root = output_root / "starting-data"
    if starting_root.is_dir():
        for file_path in starting_root.rglob("*"):
            if file_path.is_file():
                paths.add(file_path.relative_to(output_root).as_posix())
    return paths


def _is_workspace_noise_path(path: str) -> bool:
    normalized = _normalize_workspace_compare_path(path)
    if normalized in {".smr-bootstrap", "smr-bootstrap"}:
        return True
    if normalized == ".git" or normalized.endswith("/.git"):
        return True
    return "/.git/" in f"/{normalized}/"


def _workspace_file_delta(
    *,
    archive_paths: list[str],
    starting_paths: set[str],
) -> dict[str, Any]:
    new_paths = sorted(
        path
        for path in archive_paths
        if _normalize_workspace_compare_path(path) not in starting_paths
        and not _normalize_workspace_compare_path(path).startswith("starting-data/")
    )
    git_internal = sorted(path for path in new_paths if _is_workspace_noise_path(path))
    deliverable = sorted(
        _normalize_workspace_compare_path(path)
        for path in new_paths
        if not _is_workspace_noise_path(path)
    )
    return {
        "archive_file_count": len(archive_paths),
        "starting_file_count": len(starting_paths),
        "added_paths": deliverable,
        "added_count": len(deliverable),
        "deliverable_paths": deliverable,
        "deliverable_count": len(deliverable),
        "git_internal_paths": git_internal,
        "git_internal_count": len(git_internal),
    }


def _fetch_project_git_status(
    client: ResearchControlClient,
    project_id: str,
) -> dict[str, Any] | None:
    try:
        payload = client._request_json(  # noqa: SLF001
            "GET",
            f"/smr/projects/{project_id}/git/status",
            params={"max_tree_entries": 200, "max_commits": 20},
        )
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _summarize_git_status(git_status: dict[str, Any] | None) -> dict[str, Any] | None:
    if not git_status:
        return None
    tree_paths = git_status.get("tree_paths")
    if not isinstance(tree_paths, list):
        tree_paths = []
    normalized_tree = [str(path).strip().lstrip("/") for path in tree_paths if str(path).strip()]
    commits = git_status.get("recent_commits")
    recent_commits: list[dict[str, Any]] = []
    if isinstance(commits, list):
        for commit in commits[:5]:
            if not isinstance(commit, dict):
                continue
            recent_commits.append(
                {
                    "sha": str(commit.get("sha") or commit.get("commit_sha") or "")[:12] or None,
                    "subject": commit.get("subject") or commit.get("message"),
                }
            )
    return {
        "head_commit_sha": git_status.get("head_commit_sha"),
        "branch": git_status.get("branch"),
        "tree_path_count": len(normalized_tree),
        "tree_paths_sample": normalized_tree[:12],
        "tree_truncated": git_status.get("tree_truncated"),
        "recent_commit_count": len(commits) if isinstance(commits, list) else 0,
        "recent_commits": recent_commits,
    }


def _format_path_preview(paths: list[str] | None, *, limit: int = 5) -> str:
    if not paths:
        return "-"
    preview = paths[:limit]
    text = ", ".join(preview)
    extra = len(paths) - limit
    if extra > 0:
        text += f", +{extra} more"
    return text


def _project_archived_in_summary(summary: dict[str, Any]) -> bool:
    archive_project = summary.get("archive_project")
    if not isinstance(archive_project, dict):
        return False
    if archive_project.get("archived") is True:
        return True
    archived_at = str(archive_project.get("archived_at") or "").strip()
    return bool(archived_at)


def _archive_project_after_summary(
    *,
    launch: ReadmeSmokeLaunch,
    project_id: str,
    summary: dict[str, Any],
    emit: LogFn,
) -> None:
    """Archive ephemeral eval projects after the final summary.

    Uses a fresh control client because the main session closes its HTTP
    transport on ``with client`` exit and cannot be reopened.
    """
    from reportbench.readme_smoke_harness import jsonish

    cleanup_client = build_research_client(
        api_key=launch.api_key,
        base_url=launch.backend,
    ).control(timeout_seconds=120.0)
    try:
        with cleanup_client:
            emit("research.archive_project ...")
            summary["archive_project"] = jsonish(cleanup_client.archive_project(project_id))
    except Exception as exc:  # noqa: BLE001
        _record_boundary_degradation(
            summary,
            field_name="archive_project_error",
            operation="archive_project",
            exc=exc,
        )


def _collect_run_progress_metadata(
    *,
    client: ResearchControlClient,
    summary: dict[str, Any],
    output_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    """Collect post-run o11y from archive + live API surfaces.

    Project workspace reads require an unarchived project
    (``build_project_workspace_projection`` uses ``include_archived=False``).
    Milestone list routes are not mounted on the alpha ``/smr`` HTTP API yet.
    """
    project_id = str(summary.get("project_id") or "").strip()
    run_id = str(summary.get("run_id") or "").strip()
    progress: dict[str, Any] = {"schema_version": "readme_smoke_run_progress.v1"}

    digest = summary.get("run_observability_digest")
    if isinstance(digest, dict):
        progress["outcome"] = digest

    messages_summary: dict[str, Any] = {"runtime_message_count": 0, "by_status": {}}
    if project_id and run_id:
        try:
            runtime_messages = client.list_project_run_runtime_messages(
                project_id,
                run_id,
                limit=500,
            )
            if isinstance(runtime_messages, list):
                messages_summary["runtime_message_count"] = len(runtime_messages)
                by_status: dict[str, int] = {}
                for row in runtime_messages:
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status") or "unknown")
                    by_status[status] = by_status.get(status, 0) + 1
                messages_summary["by_status"] = by_status
        except Exception as exc:  # noqa: BLE001
            messages_summary["error"] = _record_boundary_degradation(
                progress,
                field_name="runtime_messages_error",
                operation="list_project_run_runtime_messages",
                exc=exc,
            )
    run_evidence = summary.get("run_evidence")
    if isinstance(run_evidence, dict):
        cached_count = run_evidence.get("runtime_message_count")
        if (
            isinstance(cached_count, int)
            and cached_count > messages_summary["runtime_message_count"]
        ):
            messages_summary["runtime_message_count"] = cached_count
    progress["messages"] = messages_summary
    progress["transcript"] = _summarize_transcript(summary.get("runtime_transcript"))

    if project_id and not _project_archived_in_summary(summary):
        try:
            workspace = client.get_project_workspace(project_id)
            if isinstance(workspace, dict):
                objectives_raw = workspace.get("objectives")
                progress["objectives"] = _summarize_rows(
                    objectives_raw if isinstance(objectives_raw, list) else [],
                    trim_fields=(
                        "objective_id",
                        "title",
                        "status",
                        "percent_complete",
                        "progress_count",
                        "run_id",
                    ),
                )
                changesets_raw = workspace.get("changesets")
                progress["changesets"] = _summarize_rows(
                    changesets_raw if isinstance(changesets_raw, list) else [],
                    trim_fields=("changeset_id", "title", "status", "item_count", "run_id"),
                )
                workspace_summary = workspace.get("summary")
                if isinstance(workspace_summary, dict):
                    progress["workspace_summary"] = {
                        "phase": workspace_summary.get("phase"),
                        "readiness": workspace_summary.get("readiness"),
                        "objective_count": workspace_summary.get("objective_count"),
                        "event_count": workspace_summary.get("event_count"),
                    }
        except Exception as exc:  # noqa: BLE001
            _record_boundary_degradation(
                progress,
                field_name="workspace_error",
                operation="get_project_workspace",
                exc=exc,
            )
    elif project_id and _project_archived_in_summary(summary):
        progress["workspace_skipped"] = "project_archived_before_workspace_fetch"

    starting_paths = _starting_paths_from_output_root(output_root)
    archive_paths = _paths_from_workspace_archive(archive_path)
    progress["workspace_files"] = _workspace_file_delta(
        archive_paths=archive_paths,
        starting_paths=starting_paths,
    )
    git_summary = _summarize_git_status(
        _fetch_project_git_status(client, project_id) if project_id else None
    )
    if git_summary:
        progress["git_server"] = git_summary

    failure_summary = _build_failure_summary(summary)
    if failure_summary:
        progress["failure"] = failure_summary
        failure_line = _format_failure_summary_line(summary)
        if failure_line:
            progress["failure_line"] = failure_line

    boundary_degradations = _collect_boundary_degradation_lines(summary)
    boundary_degradations.extend(_collect_boundary_degradation_lines(progress))
    if boundary_degradations:
        progress["boundary_degradations"] = boundary_degradations

    actor_table = _build_actor_table(
        client=client,
        summary=summary,
        project_id=project_id,
        run_id=run_id,
        force_refresh_usage=True,
    )
    progress["actor_table"] = actor_table
    summary["actor_table"] = actor_table
    return progress


def _finalize_run_progress_o11y(
    *,
    client: ResearchControlClient,
    summary: dict[str, Any],
    output_root: Path,
    archive_path: Path,
    emit: LogFn,
) -> None:
    """Collect and emit post-run o11y while the control client session is still open."""
    project_id = str(summary.get("project_id") or "").strip()
    run_id = str(summary.get("run_id") or "").strip()
    if not project_id or not run_id:
        return
    try:
        summary["run_progress_metadata"] = _collect_run_progress_metadata(
            client=client,
            summary=summary,
            output_root=output_root,
            archive_path=archive_path,
        )
        emit(_format_run_progress_emit_line(summary["run_progress_metadata"]))
        actor_table = summary.get("actor_table")
        if isinstance(actor_table, list) and actor_table:
            billed_total = sum(
                float(row.get("cost_usd") or 0.0)
                for row in actor_table
                if isinstance(row.get("cost_usd"), (int, float))
            )
            emit(f"[o11y] actors count={len(actor_table)} billed_total=${billed_total:.4f}")
    except Exception as exc:  # noqa: BLE001
        from reportbench.readme_smoke_harness import format_boundary_error_message

        payload = _record_boundary_degradation(
            summary,
            field_name="run_progress_metadata_error",
            operation="collect_run_progress_metadata",
            exc=exc,
        )
        emit(f"[o11y] progress metadata unavailable: {format_boundary_error_message(payload)}")


_ACTOR_ROLE_SORT_ORDER = {"orchestrator": 0, "worker": 1, "reviewer": 2, "unknown": 9}
_ACTOR_TABLE_COLUMNS: tuple[tuple[str, int], ...] = (
    ("role", 12),
    ("actor", 10),
    ("state", 8),
    ("time", 8),
    ("cost", 8),
    ("tokens", 8),
    ("model", 16),
)


def _is_placeholder_actor_row(row: dict[str, Any]) -> bool:
    actor_id = str(row.get("actor_id") or "").strip().lower()
    if actor_id in {"orchestrator:main", "worker:main", "reviewer:main"}:
        return True
    if not actor_id.endswith(":main"):
        return False
    state = str(row.get("state") or "").strip().lower()
    has_usage = isinstance(row.get("cost_usd"), (int, float)) or isinstance(
        row.get("token_total"), int
    )
    return state in {"created", "pending", "requested"} and not has_usage


def _usage_actor_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    for container_key in ("actor_usage",):
        container = summary.get(container_key)
        if isinstance(container, dict):
            actors = container.get("actors")
            if isinstance(actors, list):
                return [row for row in actors if isinstance(row, dict)]
    run_evidence = summary.get("run_evidence")
    if isinstance(run_evidence, dict):
        container = run_evidence.get("actor_usage")
        if isinstance(container, dict):
            actors = container.get("actors")
            if isinstance(actors, list):
                return [row for row in actors if isinstance(row, dict)]
    return []


def _observability_actor_items(summary: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = summary.get("final_observability_snapshot")
    if not isinstance(snapshot, dict):
        return []
    actors = snapshot.get("actors")
    if not isinstance(actors, dict):
        return []
    items = actors.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _actor_ids_match(left: str, right: str) -> bool:
    left_text = str(left or "").strip().lower()
    right_text = str(right or "").strip().lower()
    if not left_text or not right_text:
        return False
    return (
        left_text == right_text
        or left_text.startswith(right_text)
        or right_text.startswith(left_text)
    )


def _usage_actor_index(usage_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    usage_by_id: dict[str, dict[str, Any]] = {}
    for usage_row in usage_rows:
        actor_id = str(usage_row.get("actor_id") or "").strip()
        if actor_id:
            usage_by_id[actor_id] = usage_row
        worker_id = str(usage_row.get("worker_id") or "").strip()
        if not worker_id:
            continue
        usage_by_id[worker_id] = usage_row
        parts = worker_id.split(":")
        if len(parts) >= 4:
            usage_by_id[parts[3]] = usage_row
    return usage_by_id


def _match_usage_row(
    actor_id: str,
    usage_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    direct = usage_by_id.get(actor_id)
    if isinstance(direct, dict):
        return direct
    for usage_id, usage_row in usage_by_id.items():
        if _actor_ids_match(actor_id, usage_id):
            return usage_row
    return None


def _infer_participant_role(
    *,
    snapshot_row: dict[str, Any] | None,
    usage_row: dict[str, Any] | None,
) -> str:
    for row in (snapshot_row, usage_row):
        if not isinstance(row, dict):
            continue
        for key in ("participant_role", "actor_type"):
            value = str(row.get(key) or "").strip().lower()
            if value and value != "unknown":
                return value
    if isinstance(usage_row, dict):
        worker_id = str(usage_row.get("worker_id") or "")
        parts = worker_id.split(":")
        if len(parts) >= 2:
            candidate = parts[1].strip().lower()
            if candidate in _ACTOR_ROLE_SORT_ORDER:
                return candidate
    return "unknown"


def _short_actor_id(actor_id: str) -> str:
    text = str(actor_id or "").strip()
    if ":" in text:
        suffix = text.split(":", 1)[-1]
        return suffix[:8] if len(suffix) > 8 else suffix
    return text[:8] if len(text) > 8 else text


def _actor_duration_seconds(snapshot_row: dict[str, Any] | None) -> float | None:
    if not isinstance(snapshot_row, dict):
        return None
    started = _parse_iso_timestamp(snapshot_row.get("started_at"))
    completed = _parse_iso_timestamp(snapshot_row.get("completed_at"))
    if started is None or completed is None:
        return None
    return max(0.0, (completed - started).total_seconds())


def _actor_cost_usd(usage_row: dict[str, Any] | None) -> float | None:
    if not isinstance(usage_row, dict):
        return None
    cents = usage_row.get("billed_amount_cents")
    if isinstance(cents, (int, float)):
        return float(cents) / 100.0
    return None


def _actor_model_label(
    *,
    snapshot_row: dict[str, Any] | None,
    usage_row: dict[str, Any] | None,
) -> str:
    if isinstance(snapshot_row, dict):
        model = str(snapshot_row.get("model") or "").strip()
        if model:
            return model
        harness = str(snapshot_row.get("agent_harness") or "").strip()
        if harness:
            return harness
        profile_id = str(snapshot_row.get("profile_id") or "").strip()
        if profile_id:
            return profile_id
    if isinstance(usage_row, dict):
        by_model = usage_row.get("by_model")
        if isinstance(by_model, dict):
            candidates = [
                (str(key), float(value))
                for key, value in by_model.items()
                if str(key).strip().lower() != "unknown" and isinstance(value, (int, float))
            ]
            if candidates:
                return max(candidates, key=lambda item: item[1])[0]
    return "-"


def _actor_token_total(usage_row: dict[str, Any] | None) -> int | None:
    if not isinstance(usage_row, dict):
        return None
    token_usage = usage_row.get("token_usage")
    if not isinstance(token_usage, dict):
        return None
    total = token_usage.get("total_tokens")
    if isinstance(total, int):
        return total
    return None


def _format_token_count(count: int | None) -> str:
    if count is None:
        return "-"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def _format_duration_compact(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_actor_table_cell(text: str, width: int) -> str:
    normalized = str(text or "")
    if len(normalized) > width:
        if width <= 1:
            return normalized[:width]
        return normalized[: width - 1] + "…"
    return normalized.ljust(width)


def _ensure_actor_table_sources(
    *,
    client: ResearchControlClient,
    summary: dict[str, Any],
    project_id: str,
    run_id: str,
    force_refresh_usage: bool = False,
) -> None:
    from reportbench.readme_smoke_harness import jsonish

    if run_id and (force_refresh_usage or not _usage_actor_rows(summary)):
        try:
            summary["actor_usage"] = jsonish(client.get_run_actor_usage(run_id))
        except Exception as exc:  # noqa: BLE001
            _record_boundary_degradation(
                summary,
                field_name="actor_usage_error",
                operation="get_run_actor_usage",
                exc=exc,
            )

    if project_id and run_id and not _observability_actor_items(summary):
        try:
            snapshot = client.get_run_observability_snapshot(
                project_id,
                run_id,
                detail_level="control",
                event_limit=5,
                actor_limit=20,
                task_limit=20,
                question_limit=1,
                timeline_limit=1,
                message_limit=1,
            )
            summary["final_observability_snapshot"] = jsonish(snapshot)
        except Exception as exc:  # noqa: BLE001
            _record_boundary_degradation(
                summary,
                field_name="final_observability_snapshot_error",
                operation="get_run_observability_snapshot",
                exc=exc,
            )


def _assemble_actor_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot_items = _observability_actor_items(summary)
    usage_rows = _usage_actor_rows(summary)
    usage_by_id = _usage_actor_index(usage_rows)

    rows: list[dict[str, Any]] = []
    matched_usage_ids: set[str] = set()

    for snapshot_row in snapshot_items:
        actor_id = str(snapshot_row.get("actor_id") or "").strip()
        if not actor_id:
            continue
        usage_row = _match_usage_row(actor_id, usage_by_id)
        if isinstance(usage_row, dict):
            usage_id = str(usage_row.get("actor_id") or "").strip()
            if usage_id:
                matched_usage_ids.add(usage_id)
        duration_s = _actor_duration_seconds(snapshot_row)
        rows.append(
            {
                "actor_id": actor_id,
                "actor_short": _short_actor_id(actor_id),
                "role": _infer_participant_role(snapshot_row=snapshot_row, usage_row=usage_row),
                "state": str(snapshot_row.get("state") or snapshot_row.get("phase") or "-"),
                "duration_s": duration_s,
                "duration": _format_duration_compact(duration_s),
                "cost_usd": _actor_cost_usd(usage_row),
                "token_total": _actor_token_total(usage_row),
                "model": _actor_model_label(snapshot_row=snapshot_row, usage_row=usage_row),
                "event_count": usage_row.get("event_count")
                if isinstance(usage_row, dict)
                else None,
                "task_key": snapshot_row.get("task_key"),
            }
        )

    for usage_row in usage_rows:
        actor_id = str(usage_row.get("actor_id") or "").strip()
        if not actor_id or actor_id in matched_usage_ids:
            continue
        if any(
            _actor_ids_match(actor_id, str(item.get("actor_id") or "")) for item in snapshot_items
        ):
            continue
        rows.append(
            {
                "actor_id": actor_id,
                "actor_short": _short_actor_id(actor_id),
                "role": _infer_participant_role(snapshot_row=None, usage_row=usage_row),
                "state": "-",
                "duration_s": None,
                "duration": "-",
                "cost_usd": _actor_cost_usd(usage_row),
                "token_total": _actor_token_total(usage_row),
                "model": _actor_model_label(snapshot_row=None, usage_row=usage_row),
                "event_count": usage_row.get("event_count"),
                "task_key": usage_row.get("task_key"),
            }
        )

    rows.sort(
        key=lambda row: (
            _ACTOR_ROLE_SORT_ORDER.get(str(row.get("role") or "unknown"), 9),
            str(row.get("actor_id") or ""),
        )
    )
    return [row for row in rows if not _is_placeholder_actor_row(row)]


def _build_actor_table(
    *,
    client: ResearchControlClient,
    summary: dict[str, Any],
    project_id: str,
    run_id: str,
    force_refresh_usage: bool = False,
) -> list[dict[str, Any]]:
    _ensure_actor_table_sources(
        client=client,
        summary=summary,
        project_id=project_id,
        run_id=run_id,
        force_refresh_usage=force_refresh_usage,
    )
    return _assemble_actor_table_rows(summary)


def _format_actor_table_lines(rows: list[dict[str, Any]] | None) -> list[str]:
    if not rows:
        return ["  actors:          (none)"]

    header_cells = [_format_actor_table_cell(title, width) for title, width in _ACTOR_TABLE_COLUMNS]
    separator_cells = ["-" * width for _, width in _ACTOR_TABLE_COLUMNS]
    lines = [
        "  actors:",
        "  " + " ".join(header_cells),
        "  " + " ".join(separator_cells),
    ]
    for row in rows:
        cost = row.get("cost_usd")
        token_total = row.get("token_total")
        if isinstance(cost, (int, float)) and cost <= 0 and (not token_total or token_total == 0):
            cost_text = "-"
        elif isinstance(cost, (int, float)):
            cost_text = _format_usd(cost)
        else:
            cost_text = "-"
        lines.append(
            "  "
            + " ".join(
                [
                    _format_actor_table_cell(
                        str(row.get("role") or "-"), _ACTOR_TABLE_COLUMNS[0][1]
                    ),
                    _format_actor_table_cell(
                        str(row.get("actor_short") or "-"),
                        _ACTOR_TABLE_COLUMNS[1][1],
                    ),
                    _format_actor_table_cell(
                        str(row.get("state") or "-"), _ACTOR_TABLE_COLUMNS[2][1]
                    ),
                    _format_actor_table_cell(
                        str(row.get("duration") or "-"), _ACTOR_TABLE_COLUMNS[3][1]
                    ),
                    _format_actor_table_cell(cost_text, _ACTOR_TABLE_COLUMNS[4][1]),
                    _format_actor_table_cell(
                        _format_token_count(
                            row.get("token_total")
                            if isinstance(row.get("token_total"), int)
                            else None
                        ),
                        _ACTOR_TABLE_COLUMNS[5][1],
                    ),
                    _format_actor_table_cell(
                        str(row.get("model") or "-"), _ACTOR_TABLE_COLUMNS[6][1]
                    ),
                ]
            )
        )
    return lines


def _transcript_kind_preview(transcript: dict[str, Any]) -> str:
    by_kind = transcript.get("by_kind")
    if not isinstance(by_kind, dict) or not by_kind:
        return "-"
    preferred = (
        "turn.input",
        "turn.started",
        "turn.completed",
        "message.sent",
        "reasoning.summary",
        "token.usage",
    )
    preview: dict[str, int] = {}
    for kind in preferred:
        if kind in by_kind:
            preview[kind] = int(by_kind[kind])
    if not preview:
        for kind, count in sorted(by_kind.items())[:4]:
            preview[str(kind)] = int(count)
    return _format_count_map(preview)


def _run_progress_summary_lines(progress: dict[str, Any] | None) -> list[str]:
    if not isinstance(progress, dict):
        return ["  run outcome:     unknown"]
    lines: list[str] = []
    outcome = progress.get("outcome") if isinstance(progress.get("outcome"), dict) else {}
    outcome_label = outcome.get("terminal_outcome") or outcome.get("public_state") or "unknown"
    task_keys = outcome.get("task_keys")
    task_key_bit = ""
    if isinstance(task_keys, list) and task_keys:
        task_key_bit = f" keys={','.join(str(key) for key in task_keys)}"
    lines.append(
        "  run outcome:     "
        f"{outcome_label} "
        f"(roles {_format_count_map(outcome.get('actors_by_role'))}, "
        f"tasks {_format_count_map(outcome.get('tasks_by_state'))}{task_key_bit})"
    )
    failure_line = progress.get("failure_line")
    if isinstance(failure_line, str) and failure_line.strip():
        lines.append(f"  failure:         {failure_line}")
    boundary_degradations = progress.get("boundary_degradations")
    if isinstance(boundary_degradations, list):
        for index, degradation in enumerate(boundary_degradations):
            text = str(degradation or "").strip()
            if not text:
                continue
            label = "degraded:" if index == 0 else "                 "
            lines.append(f"  {label:<17}{text}")

    objectives = progress.get("objectives") if isinstance(progress.get("objectives"), dict) else {}
    milestones = progress.get("milestones") if isinstance(progress.get("milestones"), dict) else {}
    primary_parent = (
        progress.get("primary_parent_milestones")
        if isinstance(progress.get("primary_parent_milestones"), dict)
        else {}
    )
    obj_total = int(objectives.get("total") or 0)
    milestone_total = int(milestones.get("total") or 0)
    parent_total = int(primary_parent.get("total") or 0)
    if obj_total or milestone_total or parent_total:
        lines.append(
            "  progress:        "
            f"objectives={obj_total} ({_format_count_map(objectives.get('by_status'))}) "
            f"milestones={milestone_total} ({_format_count_map(milestones.get('by_status'))}) "
            f"primary_parent={parent_total}"
        )

    messages = progress.get("messages") if isinstance(progress.get("messages"), dict) else {}
    transcript = progress.get("transcript") if isinstance(progress.get("transcript"), dict) else {}
    lines.append(
        "  messages:        "
        f"runtime={messages.get('runtime_message_count', 0)} "
        f"transcript_events={transcript.get('event_count', 0)} "
        f"({_transcript_kind_preview(transcript)})"
    )

    workspace_files = (
        progress.get("workspace_files") if isinstance(progress.get("workspace_files"), dict) else {}
    )
    deliverable_paths = workspace_files.get("deliverable_paths")
    if not isinstance(deliverable_paths, list):
        deliverable_paths = workspace_files.get("added_paths")
    if isinstance(deliverable_paths, list):
        git_internal_count = int(workspace_files.get("git_internal_count") or 0)
        lines.append(
            "  workspace:       "
            f"archive_files={workspace_files.get('archive_file_count', 0)} "
            f"deliverables={workspace_files.get('deliverable_count', len(deliverable_paths))} "
            f"git_internal={git_internal_count} "
            f"[{_format_path_preview([str(path) for path in deliverable_paths])}]"
        )

    git_server = progress.get("git_server") if isinstance(progress.get("git_server"), dict) else {}
    if git_server:
        head = str(git_server.get("head_commit_sha") or "")[:12]
        lines.append(
            "  git-server:      "
            f"branch={git_server.get('branch') or '-'} "
            f"head={head or '-'} "
            f"commits={git_server.get('recent_commit_count', 0)} "
            f"tree_paths={git_server.get('tree_path_count', 0)} "
            f"[{_format_path_preview(git_server.get('tree_paths_sample'))}]"
        )

    changesets = progress.get("changesets") if isinstance(progress.get("changesets"), dict) else {}
    if int(changesets.get("total") or 0) > 0:
        lines.append(
            "  changesets:      "
            f"total={changesets.get('total', 0)} "
            f"({_format_count_map(changesets.get('by_status'))})"
        )
    return lines


def _task_failure_hint(summary: dict[str, Any]) -> str | None:
    from reportbench.readme_smoke_harness import extract_interpretable_failure_evidence

    evidence = extract_interpretable_failure_evidence(summary)
    primary_cause = str(evidence.get("primary_cause") or "").strip()
    if primary_cause:
        return primary_cause
    return None


def _build_failure_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    final_state = str(summary.get("final_state") or "").strip().lower()
    if final_state not in {"failed", "stopped", "canceled", "cancelled", "blocked"}:
        return None
    digest = summary.get("run_observability_digest")
    digest = digest if isinstance(digest, dict) else {}
    terminal_failure = summary.get("terminal_failure")
    terminal_failure = terminal_failure if isinstance(terminal_failure, dict) else {}
    from reportbench.readme_smoke_harness import extract_interpretable_failure_evidence

    task_hint = _task_failure_hint(summary)
    failure_evidence = extract_interpretable_failure_evidence(summary)
    return {
        "final_state": final_state,
        "status_reason": digest.get("status_reason")
        or summary.get("final_run", {}).get("status_reason")
        if isinstance(summary.get("final_run"), dict)
        else None,
        "terminal_failure": terminal_failure,
        "task_hint": task_hint,
        "failure_evidence": failure_evidence,
    }


def _format_failure_summary_line(summary: dict[str, Any]) -> str | None:
    failure = _build_failure_summary(summary)
    if not failure:
        return None
    from reportbench.readme_smoke_harness import format_interpretable_failure_line

    failure_evidence = failure.get("failure_evidence")
    if isinstance(failure_evidence, dict):
        formatted = format_interpretable_failure_line(failure_evidence)
        if formatted:
            return formatted
    parts: list[str] = []
    terminal_failure = failure.get("terminal_failure")
    if isinstance(terminal_failure, dict):
        for key in ("detail", "reason", "classification", "family", "stage"):
            value = str(terminal_failure.get(key) or "").strip()
            if value and value not in parts:
                parts.append(value)
    task_hint = str(failure.get("task_hint") or "").strip()
    if task_hint and task_hint not in parts:
        parts.append(task_hint)
    status_reason = str(failure.get("status_reason") or "").strip()
    if status_reason and status_reason not in parts:
        parts.append(status_reason)
    if not parts:
        parts.append(f"final_state={failure.get('final_state')}")
    return " | ".join(parts)


def _format_run_progress_emit_line(progress: dict[str, Any]) -> str:
    messages = progress.get("messages") if isinstance(progress.get("messages"), dict) else {}
    transcript = progress.get("transcript") if isinstance(progress.get("transcript"), dict) else {}
    workspace_files = (
        progress.get("workspace_files") if isinstance(progress.get("workspace_files"), dict) else {}
    )
    objectives = progress.get("objectives") if isinstance(progress.get("objectives"), dict) else {}
    milestones = progress.get("milestones") if isinstance(progress.get("milestones"), dict) else {}
    deliverable_paths = workspace_files.get("deliverable_paths")
    if not isinstance(deliverable_paths, list):
        deliverable_paths = workspace_files.get("added_paths")
    deliverable_preview = _format_path_preview(
        [str(path) for path in deliverable_paths] if isinstance(deliverable_paths, list) else None
    )
    git_internal_count = workspace_files.get("git_internal_count", 0)
    return (
        "[o11y] progress "
        f"runtime_messages={messages.get('runtime_message_count', 0)} "
        f"transcript_events={transcript.get('event_count', 0)} "
        f"objectives={objectives.get('total', 0)} "
        f"milestones={milestones.get('total', 0)} "
        f"workspace_deliverables={workspace_files.get('deliverable_count', workspace_files.get('added_count', 0))} "
        f"git_internal={git_internal_count} "
        f"paths=[{deliverable_preview}]"
    )


def _write_run_metrics_json(
    *,
    output_root: Path,
    summary: dict[str, Any],
    exit_code: int,
) -> None:
    digest = summary.get("run_observability_digest")
    metrics = {
        "schema_version": "readme_smoke_run_metrics.v1",
        "exit_code": exit_code,
        "project_id": summary.get("project_id"),
        "run_id": summary.get("run_id"),
        "final_state": summary.get("final_state"),
        "smr_run_elapsed_s": summary.get("smr_run_elapsed_s"),
        "verifier_elapsed_s": summary.get("verifier_elapsed_s"),
        "wall_elapsed_s": summary.get("wall_elapsed_s"),
        "smr_run_cost_usd": summary.get("smr_run_cost_usd"),
        "verifier_cost_usd": summary.get("verifier_cost_usd"),
        "total_cost_usd": summary.get("total_cost_usd"),
        "verifier_score": (summary.get("reportbench_verifier") or {}).get("score")
        if isinstance(summary.get("reportbench_verifier"), dict)
        else None,
        "observability": digest if isinstance(digest, dict) else None,
        "progress": summary.get("run_progress_metadata")
        if isinstance(summary.get("run_progress_metadata"), dict)
        else None,
        "actors": summary.get("actor_table")
        if isinstance(summary.get("actor_table"), list)
        else None,
    }
    (output_root / "run_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _probe_backend_health(
    base_url: str,
    *,
    timeout_seconds: float = 5.0,
) -> tuple[bool, str]:
    """Return (live, detail) where detail is the health URL or last error."""
    import httpx  # noqa: PLC0415
    from synth_ai.core.utils.urls import normalize_backend_base  # noqa: PLC0415

    base = normalize_backend_base(base_url).rstrip("/")
    last_error = "no health endpoints responded"
    for path in ("/health", "/healthz", "/api/v1/health"):
        url = f"{base}{path}"
        try:
            response = httpx.get(url, timeout=timeout_seconds)
        except httpx.RequestError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if response.status_code == 200:
            return True, url
        last_error = f"GET {url} returned HTTP {response.status_code}"
    return False, last_error


def _require_backend_live(launch: ReadmeSmokeLaunch) -> None:
    """Fail fast with slot start instructions when the API is not listening."""
    live, detail = _probe_backend_health(launch.backend)
    if live:
        print(f"[readme-smoke] backend live ({detail})", flush=True)
        return

    lines = [
        f"[readme-smoke] backend not reachable at {launch.backend}",
        f"  probe: {detail}",
        "",
    ]
    if launch.slot_id:
        lines.extend(
            [
                f"Slot {launch.slot_id!r} does not look up. Start it first:",
                "  cd ~/Documents/GitHub/synth-dev",
                f"  ./scripts/local.sh up {launch.slot_id} --workspace backend",
                f"  ./scripts/local.sh status {launch.slot_id}",
                "",
                "Then rerun:",
                "  bash scripts/run_readme_smoke_slot1.sh",
            ]
        )
    else:
        lines.extend(
            [
                "Point SYNTH_BACKEND_URL at a running Synth API, or pass --slot slot1.",
                "For local slot1:",
                "  cd ~/Documents/GitHub/synth-dev && ./scripts/local.sh up slot1 --workspace backend",
            ]
        )
    raise SystemExit("\n".join(lines))


def _format_retrieved_readme_lines(
    *,
    summary: dict[str, Any],
    output_root: Path,
) -> list[str]:
    validation = summary.get("validation")
    body: str | None = None
    if isinstance(validation, dict):
        readme_body = validation.get("readme_body")
        if isinstance(readme_body, str) and readme_body.strip():
            body = readme_body
    if body is None:
        archive_path = output_root / "workspace.tar.gz"
        if archive_path.is_file():
            from reportbench.readme_smoke_harness import read_workspace_readme_body

            body = read_workspace_readme_body(archive_path)
    if not body or not body.strip():
        return ["  retrieved readme: (not found)"]
    lines = ["  retrieved readme:"]
    for line in body.splitlines():
        lines.append(f"    {line}")
    return lines


def _print_final_run_summary(
    *,
    summary: dict[str, Any],
    output_root: Path,
    exit_code: int,
    driver_elapsed_s: float,
) -> None:
    verifier = (
        summary.get("reportbench_verifier")
        if isinstance(summary.get("reportbench_verifier"), dict)
        else {}
    )
    smr_cost_usd = _cost_usd_from_run_usage(summary)
    verifier_cost_usd = _verifier_judge_cost_usd(summary)
    total_cost_usd = None
    if smr_cost_usd is not None or verifier_cost_usd is not None:
        total_cost_usd = float(smr_cost_usd or 0.0) + float(verifier_cost_usd or 0.0)
    wall_elapsed_s = _wall_elapsed_seconds(summary) or driver_elapsed_s

    smr_run_elapsed_s = summary.get("smr_run_elapsed_s")
    if not isinstance(smr_run_elapsed_s, (int, float)):
        smr_run_elapsed_s = None
    verifier_elapsed_s = summary.get("verifier_elapsed_s")
    if not isinstance(verifier_elapsed_s, (int, float)):
        verifier_elapsed_s = summary.get("reportbench_verifier_elapsed_s")
    if not isinstance(verifier_elapsed_s, (int, float)):
        verifier_elapsed_s = None

    summary["driver_elapsed_s"] = round(driver_elapsed_s, 3)
    summary["wall_elapsed_s"] = round(wall_elapsed_s, 3) if wall_elapsed_s is not None else None
    # Legacy runs only recorded codex judge time; approximate SMR as wall minus verifier.
    if smr_run_elapsed_s is None and wall_elapsed_s is not None and verifier_elapsed_s is not None:
        smr_run_elapsed_s = round(max(0.0, wall_elapsed_s - verifier_elapsed_s), 3)
        summary["smr_run_elapsed_s"] = smr_run_elapsed_s
    if verifier_elapsed_s is not None and summary.get("verifier_elapsed_s") is None:
        summary["verifier_elapsed_s"] = verifier_elapsed_s
    summary["smr_run_cost_usd"] = smr_cost_usd
    summary["verifier_cost_usd"] = verifier_cost_usd
    summary["total_cost_usd"] = total_cost_usd

    score_text = "n/a"
    if verifier:
        score_raw = verifier.get("score")
        verdict = str(verifier.get("verdict") or "").strip() or "unknown"
        score_text = f"{score_raw} ({verdict})" if score_raw is not None else verdict

    progress_lines = _run_progress_summary_lines(
        summary.get("run_progress_metadata")
        if isinstance(summary.get("run_progress_metadata"), dict)
        else None
    )
    actor_table = summary.get("actor_table")
    if not isinstance(actor_table, list):
        actor_table = _assemble_actor_table_rows(summary)
    actor_lines = _format_actor_table_lines(actor_table)
    if not any("failure:" in line for line in progress_lines):
        failure_line = _format_failure_summary_line(summary)
        if failure_line:
            progress_lines = [
                progress_lines[0] if progress_lines else "  run outcome:     unknown",
                f"  failure:         {failure_line}",
                *progress_lines[1:],
            ]

    lines = [
        "",
        "======== README smoke result ========",
        f"  exit:            {exit_code}",
        f"  project_id:      {summary.get('project_id') or 'unknown'}",
        f"  run_id:          {summary.get('run_id') or 'unknown'}",
        *actor_lines,
        *progress_lines,
        f"  verifier score:  {score_text}",
        f"  smr run cost:    {_format_usd(smr_cost_usd)}",
        f"  verifier cost:   {_format_usd(verifier_cost_usd)}",
        f"  total cost:      {_format_usd(total_cost_usd)}",
        f"  smr run time:    {_format_duration_seconds(smr_run_elapsed_s)}",
        f"  verifier time:   {_format_duration_seconds(verifier_elapsed_s)}",
        f"  wall time:       {_format_duration_seconds(wall_elapsed_s)}",
        f"  output:          {output_root}",
        f"  metrics:         {output_root / 'run_metrics.json'}",
        f"  latest:          {readme_runs_dir() / 'latest'}",
        *_format_retrieved_readme_lines(summary=summary, output_root=output_root),
        "=====================================",
        "",
    ]
    for line in lines:
        print(line, flush=True)


def _record_run_placement(
    *,
    output_root: Path,
    summary: dict[str, Any],
    exit_code: int,
) -> None:
    """Symlink ``runs/latest`` and append one line to ``runs/index.jsonl``."""
    runs_dir = readme_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)
    latest = runs_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(output_root.name, target_is_directory=True)

    index_path = runs_dir / "index.jsonl"
    record = {
        "output_root": str(output_root),
        "run_dir": output_root.name,
        "project_id": summary.get("project_id"),
        "run_id": summary.get("run_id"),
        "final_state": summary.get("final_state"),
        "exit_code": exit_code,
        "target": summary.get("target"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "verifier_score": (summary.get("reportbench_verifier") or {}).get("score")
        if isinstance(summary.get("reportbench_verifier"), dict)
        else None,
        "total_cost_usd": summary.get("total_cost_usd"),
        "smr_run_elapsed_s": summary.get("smr_run_elapsed_s"),
        "verifier_elapsed_s": summary.get("verifier_elapsed_s"),
        "wall_elapsed_s": summary.get("wall_elapsed_s"),
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def build_research_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 120.0,
) -> ResearchClient:
    """Build ``SynthClient().research`` (explicit credentials for slot contracts)."""
    _ = SynthClient
    if api_key is None or base_url is None:
        from synth_ai.core.utils.env import get_api_key  # noqa: PLC0415
        from synth_ai.core.utils.urls import (  # noqa: PLC0415
            BACKEND_URL_BASE,
            normalize_backend_base,
        )

        api_key = api_key or get_api_key(required=True)
        base_url = normalize_backend_base(
            base_url or os.environ.get("SYNTH_BACKEND_URL") or BACKEND_URL_BASE
        )
    return ResearchClient(
        api_key=str(api_key).strip(),
        base_url=str(base_url).strip(),
        timeout_seconds=timeout_seconds,
    )


def resolve_readme_smoke_launch(
    *,
    slot: str | None,
    slot_mode: str | None,
    backend: str | None,
    api_key: str | None,
    worker_pool: str | None,
    use_default_slot1: bool,
) -> ReadmeSmokeLaunch:
    if use_default_slot1:
        slot = "slot1"
        slot_mode = slot_mode or "local-dockerized"

    if slot:
        from evals.launch_target_contract import (  # noqa: PLC0415
            env_from_launch_target_contract,
            load_local_launch_target_contract,
        )
        from reportbench.readme_smoke_harness import substrate_to_host_kind
        from standard.shared.core.evals_core.local_contract import (  # noqa: PLC0415
            expected_contract_path as expected_local_eval_contract_path,
        )
        from standard.shared.core.evals_core.local_contract import (
            load_local_eval_contract,
        )

        contract, contract_path = load_local_launch_target_contract(
            slot,
            target=slot_mode,
        )
        contract_env = env_from_launch_target_contract(
            contract,
            contract_path=contract_path,
        )
        local_eval = load_local_eval_contract(expected_local_eval_contract_path(slot))
        resolved_backend = (
            str(getattr(local_eval, "backend_url", "") or "").strip()
            or contract.network.backend_url
        )
        resolved_api_key = str(contract_env.get("SYNTH_API_KEY") or "").strip()
        if not resolved_backend or not resolved_api_key:
            raise RuntimeError(f"slot {slot!r} contract missing backend_url or SYNTH_API_KEY")
        worker_pool_id = str(worker_pool or contract.slot_id or contract.worker_pool_id).strip()
        if not worker_pool_id:
            raise RuntimeError("--worker-pool or slot contract worker_pool_id required")
        host_kind_value = substrate_to_host_kind(str(contract.substrate))
        return ReadmeSmokeLaunch(
            target=str(contract.target_name or slot),
            backend=resolved_backend,
            api_key=resolved_api_key,
            host_kind=ResearchHostKind(host_kind_value),
            worker_pool_id=worker_pool_id,
            slot_id=slot,
            slot_mode=slot_mode,
            slot_contract=contract,
            local_eval_contract=local_eval,
        )

    from synth_ai.core.utils.env import get_api_key  # noqa: PLC0415
    from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base  # noqa: PLC0415

    resolved_backend = normalize_backend_base(
        backend or os.environ.get("SYNTH_BACKEND_URL") or BACKEND_URL_BASE
    )
    resolved_api_key = str(api_key or get_api_key(required=True) or "").strip()
    if not resolved_api_key:
        raise RuntimeError("SYNTH_API_KEY is required when --slot is not set")
    worker_pool_id = str(worker_pool or "").strip()
    if not worker_pool_id:
        raise RuntimeError("--worker-pool is required without --slot")
    host_kind_raw = str(os.environ.get("SMR_HOST_KIND") or "docker").strip()
    return ReadmeSmokeLaunch(
        target="local",
        backend=resolved_backend,
        api_key=resolved_api_key,
        host_kind=ResearchHostKind(host_kind_raw),
        worker_pool_id=worker_pool_id,
        slot_id=None,
        slot_mode=None,
        slot_contract=None,
        local_eval_contract=None,
    )


def _apply_slot_trigger_overrides(
    trigger_kwargs: dict[str, Any],
    launch: ReadmeSmokeLaunch,
    *,
    source_repo_cfg: dict[str, Any] | None,
) -> None:
    if launch.slot_contract is None or launch.local_eval_contract is None:
        return
    from standard.shared.core.evals_core.local_contract import (  # noqa: PLC0415
        local_execution_profile_payload,
    )

    host_kind_value = launch.host_kind.value
    trigger_kwargs["local_execution"] = {
        "slot_id": launch.slot_contract.slot_id,
        "runtime_id": launch.slot_contract.runtime_id,
        "dispatch_pool": launch.slot_contract.slot_id,
        "host_kind": host_kind_value,
        "requires_hosted_capacity": launch.slot_contract.requires_hosted_capacity,
    }
    trigger_kwargs["execution_profile"] = local_execution_profile_payload(
        launch.local_eval_contract,
        host_kind=host_kind_value,
        source_repo=source_repo_cfg if isinstance(source_repo_cfg, dict) else None,
        product="readme_smoke",
    )


def _apply_agent_profile_overrides(
    runnable_project_request: dict[str, Any],
    trigger_kwargs: dict[str, Any],
    *,
    agent_harness: str | None,
    agent_model: str | None,
    agent_profile_id: str | None,
) -> str | None:
    from reportbench.readme_smoke_harness import (
        apply_agent_profiles_to_runnable_project,
        resolve_profile_by_id,
        resolve_profile_by_model,
    )

    if not (agent_profile_id or agent_harness or agent_model):
        return None
    for key in (
        "agent_profile",
        "agent_harness",
        "agent_kind",
        "agent_model",
        "agent_model_params",
        "actor_model_overrides",
    ):
        trigger_kwargs.pop(key, None)
    requested_kind = agent_harness or "codex"
    requested_model = agent_model
    resolved_id: str | None = None
    if agent_profile_id:
        profile = resolve_profile_by_id(
            profile_id=agent_profile_id,
            requested_agent_kind=requested_kind,
            requested_model=requested_model or "",
        )
        resolved_id = str(profile["profile_id"])
    elif requested_model:
        profile = resolve_profile_by_model(agent_kind=requested_kind, model=requested_model)
        resolved_id = str(profile["profile_id"])
    if resolved_id:
        apply_agent_profiles_to_runnable_project(
            runnable_project_request,
            profile_id=resolved_id,
        )
    return resolved_id


def _readme_smoke_task_root(evals_root: Path) -> Path:
    return evals_root / "runbench" / "reportbench" / "tasks" / "readme_smoke"


def _terminal_success(final_state: str) -> bool:
    return str(final_state or "").lower() in {"completed", "succeeded", "done"}


def _safe_extract_archive(archive_path: Path, output_root: Path) -> None:
    root = output_root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if root not in (target, *target.parents):
                raise RuntimeError(f"unsafe workspace archive member: {member.name}")
        archive.extractall(root)


def _ensure_readme_at_output_root(output_root: Path) -> Path | None:
    direct = output_root / "README.md"
    if direct.is_file():
        return direct
    candidates = sorted(output_root.rglob("README.md"))
    for path in candidates:
        if path.is_file():
            direct.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            return direct
    return None


def _collect_run_transcript_pages(
    client: ResearchControlClient,
    run_id: str,
    *,
    max_events: int,
) -> dict[str, Any]:
    from reportbench.readme_smoke_harness import jsonish

    page_limit = min(200, max(1, max_events))
    cursor: str | None = None
    events: list[Any] = []
    merged: dict[str, Any] = {}
    while len(events) < max_events:
        page = jsonish(
            client.get_run_transcript(
                run_id,
                cursor=cursor,
                limit=min(page_limit, max_events - len(events)),
            )
        )
        if not isinstance(page, dict):
            break
        if not merged:
            merged = {key: value for key, value in page.items() if key != "events"}
        page_events = page.get("events")
        if isinstance(page_events, list):
            events.extend(page_events)
        cursor = str(page.get("next_cursor") or "").strip() or None
        if not cursor or not page_events:
            break
    merged["events"] = events
    merged["event_count"] = len(events)
    return merged


def _task_summary_from_events(task_events: dict[str, Any] | None) -> dict[str, Any]:
    events = (
        task_events.get("events")
        if isinstance(task_events, dict) and isinstance(task_events.get("events"), list)
        else []
    )
    by_task: dict[str, str] = {}
    by_owner: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        task_key = str(event.get("task_key") or event.get("task_id") or "").strip()
        if not task_key:
            continue
        state = str(event.get("event_state") or "").strip()
        if state:
            by_task[task_key] = state
        owner = str(event.get("participant_role") or "").strip()
        if owner:
            by_owner[owner] = by_owner.get(owner, 0) + 1
    counts_by_state: dict[str, int] = {}
    for state in by_task.values():
        counts_by_state[state] = counts_by_state.get(state, 0) + 1
    return {
        "total": len(by_task),
        "counts_by_state": counts_by_state,
        "counts_by_owner": by_owner,
        "source": "task_events",
    }


def _actors_from_observability_snapshot(snapshot: Any) -> dict[str, Any]:
    from reportbench.readme_smoke_harness import jsonish

    payload = jsonish(snapshot)
    if not isinstance(payload, dict):
        return {}
    actors = payload.get("actors") if isinstance(payload.get("actors"), dict) else {}
    total_raw = actors.get("total_count", actors.get("total"))
    try:
        total = int(total_raw or 0)
    except (TypeError, ValueError):
        total = 0
    counts_by_state = actors.get("counts_by_state")
    counts_by_role = actors.get("counts_by_role")
    return {
        "total": total,
        "counts_by_state": dict(counts_by_state) if isinstance(counts_by_state, dict) else {},
        "counts_by_role": dict(counts_by_role) if isinstance(counts_by_role, dict) else {},
        "source": "observability_snapshot",
    }


def _tasks_from_observability_snapshot(snapshot: Any) -> dict[str, Any]:
    from reportbench.readme_smoke_harness import jsonish

    payload = jsonish(snapshot)
    if not isinstance(payload, dict):
        return {}
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), dict) else {}
    total_raw = tasks.get("total_count", tasks.get("total"))
    try:
        total = int(total_raw or 0)
    except (TypeError, ValueError):
        total = 0
    counts_by_state = tasks.get("counts_by_state")
    counts_by_owner = tasks.get("counts_by_owner")
    return {
        "total": total,
        "counts_by_state": dict(counts_by_state) if isinstance(counts_by_state, dict) else {},
        "counts_by_owner": dict(counts_by_owner) if isinstance(counts_by_owner, dict) else {},
        "source": "observability_snapshot",
    }


def _write_reportbench_runtime_trace(
    *,
    output_root: Path,
    summary: dict[str, Any],
    transcript: dict[str, Any] | None,
    task_events: dict[str, Any] | None,
) -> dict[str, Any]:
    run_evidence = (
        summary.get("run_evidence") if isinstance(summary.get("run_evidence"), dict) else {}
    )
    final_observability = (
        run_evidence.get("final_observability_summary")
        if isinstance(run_evidence.get("final_observability_summary"), dict)
        else {}
    )
    observed_tasks = final_observability.get("tasks")
    if not isinstance(observed_tasks, dict) or not observed_tasks.get("total"):
        observed_tasks = _task_summary_from_events(task_events)
    trace = {
        "schema_version": "reportbench_runtime_trace.v1",
        "task_id": summary.get("task_id"),
        "project_id": summary.get("project_id"),
        "run_id": summary.get("run_id"),
        "final_state": summary.get("final_state"),
        "objectives": run_evidence.get("objectives"),
        "milestones": run_evidence.get("milestones"),
        "objective_events": run_evidence.get("objective_events"),
        "tasks": observed_tasks,
        "actors": final_observability.get("actors"),
        "runtime_messages": run_evidence.get("runtime_messages"),
        "runtime_message_count": run_evidence.get("runtime_message_count"),
        "actor_usage": run_evidence.get("actor_usage"),
        "task_events": task_events or {},
        "transcript": transcript or {},
    }
    artifacts_dir = output_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = artifacts_dir / "runtime_trace.json"
    runtime_path.write_text(
        json.dumps(trace, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    full_trace = {
        **trace,
        "schema_version": "reportbench_full_trace.v1",
        "trace_policy": {
            "includes": [
                "runtime transcript events returned by backend",
                "task events",
                "observability snapshot summary",
                "runtime messages",
            ],
            "excludes": ["raw hidden chain-of-thought not exposed by backend transcript API"],
        },
    }
    full_trace_path = artifacts_dir / "full_trace.json"
    full_trace_path.write_text(
        json.dumps(full_trace, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return trace


def _artifact_manifest(output_root: Path, relative_paths: list[str]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = output_root / relative
        if not path.is_file():
            continue
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "content_type": "application/json" if path.suffix == ".json" else "text/markdown",
            }
        )
    return {
        "schema_version": "artifact_manifest.v1",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }


def _write_rubric_companion_artifacts(
    *,
    output_root: Path,
    summary: dict[str, Any],
    validation: dict[str, Any],
    verifier_review: dict[str, Any] | None,
) -> None:
    from reportbench.readme_smoke_harness import TASK_ID

    marker_ok = bool(validation.get("readme_marker_present"))
    reward_value = 1.0 if marker_ok else 0.0
    verifier_score = None
    if isinstance(verifier_review, dict):
        verifier_score = verifier_review.get("score")
    reportbench_output = {
        "schema_version": "reportbench_output.v1",
        "task_id": TASK_ID,
        "reward": {
            "value": reward_value,
            "primary_metric": "readme_smoke_precheck",
        },
        "verifier": verifier_review,
        "summary": {
            "readme_present": bool(validation.get("readme_present")),
            "proof_marker_present": marker_ok,
        },
    }
    artifacts_dir = output_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "reportbench_output.json").write_text(
        json.dumps(reportbench_output, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if isinstance(verifier_review, dict):
        (artifacts_dir / "verifier_review.json").write_text(
            json.dumps(verifier_review, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    relative_paths = [
        "README.md",
        "artifacts/reportbench_output.json",
        "artifacts/runtime_trace.json",
        "artifacts/full_trace.json",
        "evals_summary.json",
        "artifact_manifest.json",
    ]
    if isinstance(verifier_review, dict):
        relative_paths.insert(2, "artifacts/verifier_review.json")
    primary_score = (
        float(verifier_score) if isinstance(verifier_score, (int, float)) else reward_value
    )
    evals_summary = {
        "schema_version": "evals_summary.v1",
        "task_id": TASK_ID,
        "project_id": summary.get("project_id"),
        "run_id": summary.get("run_id"),
        "final_state": "passed"
        if marker_ok
        and primary_score
        >= summary.get("codex_verifier_pass_threshold", DEFAULT_CODEX_VERIFIER_PASS_THRESHOLD)
        else "failed",
        "primary_score": primary_score,
    }
    (output_root / "evals_summary.json").write_text(
        json.dumps(evals_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = _artifact_manifest(output_root, relative_paths)
    (output_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _materialize_codex_verifier_evidence(
    *,
    client: ResearchControlClient,
    output_root: Path,
    archive_path: Path,
    summary: dict[str, Any],
    validation: dict[str, Any],
    project_id: str,
    run_id: str,
    trace_limit: int,
    log: LogFn,
) -> None:
    from reportbench.readme_smoke_harness import jsonish

    if archive_path.exists() and archive_path.stat().st_size > 0:
        log("codex verifier: extract workspace archive ...")
        _safe_extract_archive(archive_path, output_root)
        _ensure_readme_at_output_root(output_root)

    log("codex verifier: fetch observability snapshot ...")
    try:
        observability_snapshot = client.get_run_observability_snapshot(
            project_id,
            run_id,
            detail_level="full",
            event_limit=80,
            actor_limit=50,
            task_limit=80,
            question_limit=40,
            timeline_limit=20,
            message_limit=40,
        )
        summary["final_observability_snapshot"] = jsonish(observability_snapshot)
        tasks_summary = _tasks_from_observability_snapshot(observability_snapshot)
        actors_summary = _actors_from_observability_snapshot(observability_snapshot)
        run_evidence = summary.setdefault("run_evidence", {})
        if isinstance(run_evidence, dict):
            run_evidence["final_observability_summary"] = {
                "tasks": tasks_summary,
                "actors": actors_summary,
                "runtime_messages": {},
            }
    except Exception as exc:  # noqa: BLE001
        from reportbench.readme_smoke_harness import format_boundary_error_message

        payload = _record_boundary_degradation(
            summary,
            field_name="final_observability_snapshot_error",
            operation="get_run_observability_snapshot",
            exc=exc,
        )
        log(f"codex verifier: observability unavailable: {format_boundary_error_message(payload)}")

    log("codex verifier: fetch transcript ...")
    try:
        transcript = _collect_run_transcript_pages(
            client,
            run_id,
            max_events=max(1, trace_limit),
        )
        summary["runtime_transcript"] = transcript
    except Exception as exc:  # noqa: BLE001
        from reportbench.readme_smoke_harness import format_boundary_error_message

        payload = _record_boundary_degradation(
            summary,
            field_name="runtime_transcript_error",
            operation="collect_run_transcript_pages",
            exc=exc,
        )
        transcript = {}
        log(f"codex verifier: transcript unavailable: {format_boundary_error_message(payload)}")

    log("codex verifier: fetch task events ...")
    task_events: dict[str, Any] = {}
    try:
        task_events = jsonish(
            client.list_run_task_events(project_id, run_id, limit=max(1, trace_limit))
        )
        summary["task_events"] = task_events
    except Exception as exc:  # noqa: BLE001
        from reportbench.readme_smoke_harness import format_boundary_error_message

        payload = _record_boundary_degradation(
            summary,
            field_name="task_events_error",
            operation="list_run_task_events",
            exc=exc,
        )
        log(f"codex verifier: task events unavailable: {format_boundary_error_message(payload)}")

    try:
        runtime_messages = client.list_project_run_runtime_messages(
            project_id,
            run_id,
            limit=500,
        )
        run_evidence = summary.setdefault("run_evidence", {})
        if isinstance(run_evidence, dict):
            run_evidence["runtime_messages"] = jsonish(runtime_messages)
            if isinstance(runtime_messages, list):
                run_evidence["runtime_message_count"] = len(runtime_messages)
    except Exception as exc:  # noqa: BLE001
        _record_boundary_degradation(
            summary,
            field_name="runtime_messages_error",
            operation="list_project_run_runtime_messages",
            exc=exc,
        )

    try:
        summary["actor_usage"] = jsonish(client.get_run_actor_usage(run_id))
        run_evidence = summary.setdefault("run_evidence", {})
        if isinstance(run_evidence, dict):
            run_evidence["actor_usage"] = summary["actor_usage"]
    except Exception as exc:  # noqa: BLE001
        _record_boundary_degradation(
            summary,
            field_name="actor_usage_error",
            operation="get_run_actor_usage",
            exc=exc,
        )

    _write_reportbench_runtime_trace(
        output_root=output_root,
        summary=summary,
        transcript=summary.get("runtime_transcript")
        if isinstance(summary.get("runtime_transcript"), dict)
        else {},
        task_events=task_events if isinstance(task_events, dict) else {},
    )
    _write_rubric_companion_artifacts(
        output_root=output_root,
        summary=summary,
        validation=validation,
        verifier_review=None,
    )


def _run_codex_reportbench_verifier(
    *,
    output_root: Path,
    task_root: Path,
    model: str,
    reasoning_effort: str,
    pass_threshold: float,
) -> dict[str, Any]:
    from runbench.reportbench.verifier import run_verifier

    return run_verifier(
        output_root=output_root,
        task_root=task_root,
        model=model,
        reasoning_effort=reasoning_effort,
        pass_threshold=pass_threshold,
    )


def run_readme_smoke(
    *,
    launch: ReadmeSmokeLaunch,
    output_root: Path,
    research: ResearchClient | None = None,
    agent_harness: str | None = None,
    agent_model: str | None = None,
    agent_profile_id: str | None = None,
    run_timebox_s: int = 1500,
    poll_timebox_s: int = 3600,
    launch_retry_timebox_s: int = 3600,
    setup_retry_timebox_s: int = 180,
    run_codex_verifier: bool = True,
    codex_verifier_model: str = DEFAULT_CODEX_VERIFIER_MODEL,
    codex_verifier_reasoning_effort: str = DEFAULT_CODEX_VERIFIER_REASONING_EFFORT,
    codex_verifier_pass_threshold: float = DEFAULT_CODEX_VERIFIER_PASS_THRESHOLD,
    codex_verifier_trace_limit: int = 500,
    log: LogFn | None = None,
) -> int:
    """Execute README smoke using the Research SDK control plane."""
    from reportbench.ai_cache_request import ai_cache_request_from_env
    from reportbench.project_config import build_staged_reportbench_launch_bundle
    from reportbench.readme_smoke_harness import (
        LANE_ROOT,
        TASK_ID,
        api_error_payload,
        extract_terminal_failure,
        field_value,
        is_retryable_launch_backpressure,
        is_retryable_pre_run_transport_error,
        is_retryable_setup_backpressure,
        jsonish,
        launch_blocker_code,
        poll_run_until_terminal,
        readme_smoke_exit_code,
        setup_retry_delay_seconds,
        trigger_kwargs_from_bundle,
        validate_workspace_archive,
        write_eval_summary_outputs,
    )

    _log = log or (lambda message: print(message, flush=True))
    driver_started = time.monotonic()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    archive_path = output_root / "workspace.tar.gz"
    log_path = output_root / "run.log"
    log_lines: list[str] = []

    def _emit(msg: str) -> None:
        line = f"[{datetime.now(UTC).isoformat()}] {msg}"
        log_lines.append(line)
        _log(_reformat_log_message(msg))

    def _section(name: str) -> None:
        bar = "─" * max(0, 42 - len(name))
        _log(f"\n── {name} {bar}")

    client = (
        research
        or build_research_client(
            api_key=launch.api_key,
            base_url=launch.backend,
        )
    ).control(timeout_seconds=120.0)

    summary: dict[str, Any] = {
        "sdk": "synth-ai",
        "task_id": TASK_ID,
        "output_root": str(output_root),
        "target": launch.target,
        "backend": launch.backend,
        "host_kind": launch.host_kind.value,
        "started_at": datetime.now(UTC).isoformat(),
        "lane_root": str(LANE_ROOT),
        "poll_timebox_seconds": poll_timebox_s,
        "launch_retry_timebox_seconds": launch_retry_timebox_s,
        "setup_retry_timebox_seconds": setup_retry_timebox_s,
        "run_timebox_seconds": run_timebox_s,
        "codex_verifier_enabled": run_codex_verifier,
        "codex_verifier_model": codex_verifier_model,
        "codex_verifier_reasoning_effort": codex_verifier_reasoning_effort,
        "codex_verifier_pass_threshold": codex_verifier_pass_threshold,
    }

    bundle = build_staged_reportbench_launch_bundle(
        task_id=TASK_ID,
        nick=launch.worker_pool_id,
        worker_pool_id=launch.worker_pool_id,
    )
    worker_pool_id = str(bundle.get("worker_pool_id") or launch.worker_pool_id).strip()
    runnable_project_request = dict(bundle.get("runnable_project_request") or {})
    files = bundle.get("workspace_inputs", {}).get("files")
    if not isinstance(files, list):
        files = []
    work_mode_raw = str((bundle.get("trigger_payload") or {}).get("work_mode") or "directed_effort")
    work_mode = ResearchWorkMode(work_mode_raw)
    trigger_kwargs = trigger_kwargs_from_bundle(
        host_kind=launch.host_kind,
        work_mode=work_mode,
        bundle=bundle,
        run_timebox_seconds=run_timebox_s,
    )
    ai_cache_request = ai_cache_request_from_env()
    if ai_cache_request is not None:
        trigger_kwargs["ai_cache"] = ai_cache_request

    resolved_profile_id = _apply_agent_profile_overrides(
        runnable_project_request,
        trigger_kwargs,
        agent_harness=agent_harness,
        agent_model=agent_model,
        agent_profile_id=agent_profile_id,
    )
    source_repo_cfg = (
        bundle.get("workspace_inputs", {}).get("source_repo")
        if isinstance(bundle.get("workspace_inputs"), dict)
        else None
    )
    _apply_slot_trigger_overrides(
        trigger_kwargs,
        launch,
        source_repo_cfg=source_repo_cfg if isinstance(source_repo_cfg, dict) else None,
    )

    project_name = str(runnable_project_request.get("name") or "").strip() or (
        f"ReportBench README Smoke slot1-{_utcstamp()}"
    )
    summary["worker_pool_id"] = worker_pool_id
    summary["project_name"] = project_name
    summary["workspace_input_count"] = len(files)
    summary["work_mode"] = work_mode.value
    if resolved_profile_id:
        summary["resolved_agent_profile_id"] = resolved_profile_id
        summary["requested_agent_profile_id"] = agent_profile_id
        summary["requested_agent_harness"] = agent_harness or "codex"
        summary["requested_agent_model"] = agent_model

    retention_policy = runnable_project_request.get("retention_policy")
    should_auto_archive = (
        isinstance(retention_policy, dict)
        and str(retention_policy.get("class") or "").strip().lower() == "local_ephemeral_eval"
        and str(retention_policy.get("auto_archive") or "true").strip().lower()
        not in {"false", "0", "no"}
    )

    _section("Config")
    _emit(f"target={launch.target} backend={launch.backend} host_kind={launch.host_kind.value}")
    _emit(
        f"worker_pool={worker_pool_id} run_timebox_s={run_timebox_s} "
        f"poll_timebox_s={poll_timebox_s}"
    )
    if resolved_profile_id:
        _emit(
            "agent_profile_override "
            f"profile_id={resolved_profile_id} harness={agent_harness or 'codex'} "
            f"model={agent_model or '<profile default>'}"
        )
    _emit(f"workspace_inputs={len(files)} files; project_name={project_name!r}")

    try:
        with client:
            _section("Setup")
            _emit("research.get_limits ...")
            limits = client.get_limits()
            summary["limits"] = jsonish(limits)

            _emit("research.projects.create_runnable_project ...")
            project = client.create_runnable_project(runnable_project_request)
            project_id = str(field_value(project, "project_id", default="") or "").strip()
            if not project_id:
                raise ResearchApiError(f"create_runnable_project returned no project_id: {project}")
            summary["project_id"] = project_id
            _emit(f"project_id={project_id}")

            source_repo_url = (
                str(source_repo_cfg.get("url") or "").strip()
                if isinstance(source_repo_cfg, dict)
                else ""
            )
            if source_repo_url:
                _emit(f"research.attach_source_repo repo={source_repo_url} ...")
                client.attach_source_repo(
                    project_id,
                    source_repo_url,
                    default_branch=(
                        str(source_repo_cfg.get("default_branch") or "").strip() or None
                    )
                    if isinstance(source_repo_cfg, dict)
                    else None,
                )

            setup_deadline = time.monotonic() + setup_retry_timebox_s
            upload_attempt = 0
            while True:
                upload_attempt += 1
                try:
                    _emit(f"research.upload_workspace_files attempt={upload_attempt} ...")
                    upload_result = client.upload_workspace_files(project_id, files)
                    break
                except ResearchApiError as exc:
                    if not is_retryable_setup_backpressure(exc):
                        raise
                    if time.monotonic() >= setup_deadline:
                        raise ResearchApiError(
                            f"upload_workspace_files blocked after {setup_retry_timebox_s}s: {exc}"
                        ) from exc
                    delay_s = setup_retry_delay_seconds(upload_attempt)
                    _emit(f"setup backpressure upload delay_s={delay_s:.1f}")
                    time.sleep(delay_s)
            summary["upload_result"] = jsonish(upload_result)
            _emit(f"uploaded {len(files)} files")

            prepare_attempt = 0
            while True:
                prepare_attempt += 1
                try:
                    _emit(f"research.prepare_project_setup attempt={prepare_attempt} ...")
                    setup = client.prepare_project_setup(project_id)
                    break
                except ResearchApiError as exc:
                    if not is_retryable_setup_backpressure(exc):
                        raise
                    if time.monotonic() >= setup_deadline:
                        raise ResearchApiError(
                            f"prepare_project_setup blocked after {setup_retry_timebox_s}s: {exc}"
                        ) from exc
                    delay_s = setup_retry_delay_seconds(prepare_attempt)
                    _emit(f"setup backpressure prepare delay_s={delay_s:.1f}")
                    time.sleep(delay_s)
            setup_state = str(field_value(setup, "state", default="") or "").strip().lower()
            summary["setup"] = jsonish(setup)
            _emit(f"setup state={setup_state or '<unknown>'}")
            if setup_state != "ready":
                raise ResearchApiError(f"setup not ready (state={setup_state!r}): {setup}")

            _section("Launch")
            launch_deadline = time.monotonic() + launch_retry_timebox_s
            launch_attempt = 0
            run_id = ""
            while True:
                launch_attempt += 1
                try:
                    _emit(
                        "research.get_launch_preflight "
                        f"attempt={launch_attempt} work_mode={work_mode.value} ..."
                    )
                    preflight = client.get_launch_preflight(project_id, **trigger_kwargs)
                    summary["launch_preflight"] = jsonish(preflight)
                    if not bool(field_value(preflight, "clear_to_trigger", default=False)):
                        if is_retryable_launch_backpressure(preflight):
                            if time.monotonic() >= launch_deadline:
                                raise ResearchApiError(
                                    f"launch preflight blocked after {launch_retry_timebox_s}s"
                                )
                            code = launch_blocker_code(preflight) or "launch_backpressure"
                            _emit(f"launch preflight backpressure code={code}")
                            time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                            continue
                        raise ResearchApiError(f"launch preflight blocked: {preflight}")

                    _emit(
                        "research.runs.trigger_run "
                        f"attempt={launch_attempt} work_mode={work_mode.value} ..."
                    )
                    run = client.trigger_run(project_id, **trigger_kwargs)
                    run_id = str(field_value(run, "run_id", "id", default="") or "").strip()
                    if not run_id:
                        raise ResearchApiError(f"trigger_run returned no run_id: {run}")
                    break
                except ResearchApiError as exc:
                    if is_retryable_launch_backpressure(exc) and time.monotonic() < launch_deadline:
                        _emit(f"launch backpressure trigger_run: {exc}")
                        time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                        continue
                    raise
                except Exception as exc:  # noqa: BLE001
                    if (
                        is_retryable_pre_run_transport_error(exc)
                        and time.monotonic() < launch_deadline
                    ):
                        _emit(f"launch transport retry: {type(exc).__name__}: {exc}")
                        time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                        continue
                    raise

            summary["run_id"] = run_id
            summary["trigger_response"] = jsonish(run)
            _emit(f"run_id={run_id}")

            _section("Running")
            smr_run_started = time.monotonic()
            _emit("research.runs.poll until terminal ...")
            final_run = poll_run_until_terminal(
                client,
                project_id,
                run_id,
                timebox_s=poll_timebox_s,
                log=_emit,
            )
            final_state = str(field_value(final_run, "public_state", default="") or "").lower()
            summary["final_state"] = final_state
            summary["final_run"] = jsonish(final_run)
            summary["smr_run_elapsed_s"] = round(time.monotonic() - smr_run_started, 3)
            _emit(f"final_state={final_state} smr_elapsed_s={summary['smr_run_elapsed_s']}")
            try:
                _persist_terminal_run_evidence(
                    client=client,
                    summary=summary,
                    project_id=project_id,
                    run_id=run_id,
                )
                terminal_digest = summary.get("run_observability_digest")
                if isinstance(terminal_digest, dict):
                    _emit(_format_observability_digest_line(terminal_digest))
                failure_line = _format_failure_summary_line(summary)
                if failure_line:
                    _emit(f"[o11y] failure {failure_line}")
            except Exception as exc:  # noqa: BLE001
                from reportbench.readme_smoke_harness import format_boundary_error_message

                payload = _record_boundary_degradation(
                    summary,
                    field_name="run_observability_digest_error",
                    operation="persist_terminal_run_evidence",
                    exc=exc,
                )
                _emit(
                    f"[o11y] terminal digest unavailable: {format_boundary_error_message(payload)}"
                )

            if final_state in {"failed", "stopped", "canceled", "cancelled", "blocked"}:
                terminal_failure = extract_terminal_failure(final_run)
                if terminal_failure:
                    summary["terminal_failure"] = terminal_failure

            try:
                summary["run_usage"] = jsonish(client.get_run_usage(run_id))
            except ResearchApiError as exc:
                _record_boundary_degradation(
                    summary,
                    field_name="run_usage_error",
                    operation="get_run_usage",
                    exc=exc,
                )

            archive_meta: dict[str, Any] = {}
            try:
                _emit("research.download_workspace_archive ...")
                archive_meta = client.download_workspace_archive(project_id, archive_path)
                _emit(f"archive bytes={archive_meta.get('bytes_written')}")
            except ResearchApiError as exc:
                from reportbench.readme_smoke_harness import format_boundary_error_message

                payload = _record_boundary_degradation(
                    summary,
                    field_name="workspace_download_error",
                    operation="download_workspace_archive",
                    exc=exc,
                )
                _emit(f"workspace download failed: {format_boundary_error_message(payload)}")
            summary["archive_meta"] = jsonish(archive_meta)

            validation: dict[str, Any] = {}
            if archive_path.exists() and archive_path.stat().st_size > 0:
                validation = validate_workspace_archive(archive_path)
                summary["validation"] = validation

            if run_codex_verifier and archive_path.exists() and archive_path.stat().st_size > 0:
                _section("Verification")
                evals_root = _resolve_evals_root()
                task_root = _readme_smoke_task_root(evals_root)
                if not _terminal_success(final_state):
                    summary["reportbench_verifier_skipped"] = {
                        "reason": "non_terminal_final_state",
                        "final_state": final_state,
                    }
                    _emit(
                        "codex verifier skipped: "
                        f"final_state={final_state!r} is not terminal-success"
                    )
                elif not task_root.is_dir():
                    summary["reportbench_verifier_error"] = f"missing task root: {task_root}"
                    _emit(f"codex verifier failed: missing task root {task_root}")
                else:
                    verifier_phase_started = time.monotonic()
                    try:
                        _materialize_codex_verifier_evidence(
                            client=client,
                            output_root=output_root,
                            archive_path=archive_path,
                            summary=summary,
                            validation=validation,
                            project_id=project_id,
                            run_id=run_id,
                            trace_limit=codex_verifier_trace_limit,
                            log=_emit,
                        )
                        _emit(
                            "codex verifier: run_reportbench_verifier "
                            f"model={codex_verifier_model} "
                            f"reasoning_effort={codex_verifier_reasoning_effort} ..."
                        )
                        verifier_started = time.monotonic()
                        verifier_review = _run_codex_reportbench_verifier(
                            output_root=output_root,
                            task_root=task_root,
                            model=codex_verifier_model,
                            reasoning_effort=codex_verifier_reasoning_effort,
                            pass_threshold=codex_verifier_pass_threshold,
                        )
                        summary["reportbench_verifier"] = verifier_review
                        summary["reportbench_verifier_elapsed_s"] = round(
                            time.monotonic() - verifier_started,
                            3,
                        )
                        summary["verifier_elapsed_s"] = round(
                            time.monotonic() - verifier_phase_started,
                            3,
                        )
                        _write_rubric_companion_artifacts(
                            output_root=output_root,
                            summary=summary,
                            validation=validation,
                            verifier_review=verifier_review,
                        )
                        _emit(
                            "codex verifier done "
                            f"score={verifier_review.get('score')} "
                            f"verdict={verifier_review.get('verdict')} "
                            f"verifier_elapsed_s={summary['verifier_elapsed_s']}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        from reportbench.readme_smoke_harness import format_boundary_error_message

                        summary["verifier_elapsed_s"] = round(
                            time.monotonic() - verifier_phase_started,
                            3,
                        )
                        payload = _record_boundary_degradation(
                            summary,
                            field_name="reportbench_verifier_error",
                            operation="run_codex_reportbench_verifier",
                            exc=exc,
                        )
                        _emit(
                            "codex verifier failed: "
                            f"{format_boundary_error_message(payload)} "
                            f"verifier_elapsed_s={summary['verifier_elapsed_s']}"
                        )

            _finalize_run_progress_o11y(
                client=client,
                summary=summary,
                output_root=output_root,
                archive_path=archive_path,
                emit=_emit,
            )

    except ResearchApiError as exc:
        _emit(f"FATAL ResearchApiError: {exc}")
        summary["fatal_error"] = api_error_payload(exc)
    except Exception as exc:  # noqa: BLE001
        _emit(f"FATAL {type(exc).__name__}: {exc}")
        summary["fatal_error"] = api_error_payload(exc)

    summary["finished_at"] = datetime.now(UTC).isoformat()
    driver_elapsed_s = time.monotonic() - driver_started

    exit_code = readme_smoke_exit_code(summary)
    validation = summary.get("validation") if isinstance(summary.get("validation"), dict) else {}
    verifier = (
        summary.get("reportbench_verifier")
        if isinstance(summary.get("reportbench_verifier"), dict)
        else {}
    )
    if exit_code == 0:
        if verifier:
            _emit(f"score: passed (README marker + codex verifier score={verifier.get('score')})")
        else:
            _emit("score: passed (worker README marker present in workspace archive)")
    elif summary.get("fatal_error"):
        _emit("score: fatal error")
    elif summary.get("reportbench_verifier_error"):
        from reportbench.readme_smoke_harness import format_boundary_error_message

        _emit(
            "score: failed "
            f"(codex verifier error: "
            f"{format_boundary_error_message(summary['reportbench_verifier_error'])})"
        )
    elif verifier and exit_code != 0:
        _emit(
            "score: failed "
            f"(codex verifier score={verifier.get('score')} "
            f"verdict={verifier.get('verdict')})"
        )
    elif validation.get("readme_marker_present") is False:
        _emit("score: failed (README marker missing in workspace archive)")
    else:
        _emit(f"score: failed (final_state={summary.get('final_state')})")

    _print_final_run_summary(
        summary=summary,
        output_root=output_root,
        exit_code=exit_code,
        driver_elapsed_s=driver_elapsed_s,
    )
    _write_run_metrics_json(
        output_root=output_root,
        summary=summary,
        exit_code=exit_code,
    )

    project_id_for_archive = str(summary.get("project_id") or "").strip()
    if should_auto_archive and project_id_for_archive:
        _section("Cleanup")
        _archive_project_after_summary(
            launch=launch,
            project_id=project_id_for_archive,
            summary=summary,
            emit=_emit,
        )

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    write_eval_summary_outputs(
        output_root=output_root,
        summary=summary,
        archive_path=archive_path,
        log_path=log_path,
    )
    _record_run_placement(output_root=output_root, summary=summary, exit_code=exit_code)
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research README smoke via synth-ai Research SDK.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Run output directory (default: readme_runs/runs/<UTC>_<target>/). "
            "Override with OUTPUT_ROOT env."
        ),
    )
    parser.add_argument("--slot", default=None)
    parser.add_argument("--slot-mode", default=None)
    parser.add_argument("--worker-pool", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--poll-timebox", type=int, default=3600)
    parser.add_argument("--run-timebox", type=int, default=1500)
    parser.add_argument("--agent-harness", default=None)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--agent-profile-id", default=None)
    parser.add_argument("--use-default-slot1", action="store_true")
    parser.add_argument(
        "--skip-codex-verifier",
        action="store_true",
        help="Skip post-run ReportBench Codex judge (default: run with gpt-5.3-codex-spark).",
    )
    parser.add_argument(
        "--codex-verifier-model",
        default=DEFAULT_CODEX_VERIFIER_MODEL,
        help=f"Codex CLI judge model (default: {DEFAULT_CODEX_VERIFIER_MODEL}).",
    )
    parser.add_argument(
        "--codex-verifier-reasoning-effort",
        default=DEFAULT_CODEX_VERIFIER_REASONING_EFFORT,
    )
    parser.add_argument(
        "--codex-verifier-pass-threshold",
        type=float,
        default=DEFAULT_CODEX_VERIFIER_PASS_THRESHOLD,
    )
    parser.add_argument(
        "--codex-verifier-trace-limit",
        type=int,
        default=500,
        help="Max transcript/task-event rows collected for verifier evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    evals_root = _resolve_evals_root()
    _ensure_evals_importable(evals_root)

    if args.use_default_slot1:
        args.slot = args.slot or "slot1"
        args.slot_mode = args.slot_mode or "local-dockerized"
        args.agent_harness = args.agent_harness or "codex"
        args.agent_profile_id = args.agent_profile_id or "codex_gpt_5_4_mini_medium"

    launch = resolve_readme_smoke_launch(
        slot=args.slot,
        slot_mode=args.slot_mode,
        backend=args.backend,
        api_key=args.api_key,
        worker_pool=args.worker_pool,
        use_default_slot1=args.use_default_slot1,
    )
    output_root = resolve_output_root(
        args.output_root or os.environ.get("OUTPUT_ROOT"),
        target=launch.target,
    )
    os.environ.setdefault("SYNTH_BACKEND_URL", launch.backend)
    os.environ.setdefault("RESEARCH_SDK", "synth-ai")

    print(f"[readme-smoke] output_root={output_root}", flush=True)
    _require_backend_live(launch)

    research = build_research_client(api_key=launch.api_key, base_url=launch.backend)
    try:
        limits = research.get_limits()
    except ResearchApiError as exc:
        raise SystemExit(
            f"[readme-smoke] backend is up but GET /smr/limits failed: {exc}\n"
            "Check SYNTH_API_KEY for this slot (slot contract or synth-ai/.env)."
        ) from exc
    print(
        f"[synth-ai research] backend={launch.backend} limits_keys={sorted(limits.keys())[:8]}",
        flush=True,
    )

    return run_readme_smoke(
        launch=launch,
        output_root=output_root,
        research=research,
        agent_harness=args.agent_harness,
        agent_model=args.agent_model,
        agent_profile_id=args.agent_profile_id,
        run_timebox_s=args.run_timebox,
        poll_timebox_s=args.poll_timebox,
        run_codex_verifier=not args.skip_codex_verifier,
        codex_verifier_model=args.codex_verifier_model,
        codex_verifier_reasoning_effort=args.codex_verifier_reasoning_effort,
        codex_verifier_pass_threshold=args.codex_verifier_pass_threshold,
        codex_verifier_trace_limit=args.codex_verifier_trace_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CODEX_VERIFIER_MODEL",
    "default_run_output_root",
    "readme_runs_dir",
    "readme_runs_root",
    "ReadmeSmokeLaunch",
    "ResearchApiError",
    "ResearchClient",
    "ResearchControlClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchHostKind",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchProjectMonthlyBudgetExhaustedError",
    "ResearchStructuredDenialError",
    "ResearchWorkMode",
    "build_research_client",
    "main",
    "resolve_output_root",
    "resolve_readme_smoke_launch",
    "run_readme_smoke",
    "synth_ai_repo_root",
]
