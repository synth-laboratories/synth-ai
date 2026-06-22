#!/usr/bin/env python3
"""Inspect SMR actor agent progress: transcript stream, thinking summaries, tool calls.

Polls the public SMR actor trace API (durable + live transcript events), run task
events, actor usage, and observability snapshot so you can verify an in-flight run
is making progress without tailing smr-runtime logs.

Examples:

```bash
cd ~/Documents/GitHub/synth-ai
uv run python readme_runs/inspect_actor_progress.py \\
  --project-id <project_id> --run-id <run_id> --slot slot1 --watch

# Focus on orchestrator only:
uv run python readme_runs/inspect_actor_progress.py \\
  --project-id <project_id> --run-id <run_id> --slot slot1 \\
  --actor orchestrator --watch --interval 5
```
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PROGRESS_KINDS = frozenset(
    {
        "turn.input",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "turn.interrupted",
        "tool.call.started",
        "tool.call.completed",
        "tool.call.failed",
        "mcp_tool_call",
        "command_execution",
        "reasoning.summary",
        "token.usage",
        "message.completed",
        "goal.updated",
        "goal.cleared",
    }
)

TERMINAL_RUN_STATES = frozenset({"completed", "failed", "canceled", "cancelled", "stopped", "done"})


@dataclass
class ActorTraceState:
    actor_key: str
    actor_id: str = ""
    participant_role: str = ""
    participant_session_id: str = ""
    actor_state: str = ""
    transcript_cursor: str | None = None
    live_cursor: str | None = None
    seen_event_ids: set[str] = field(default_factory=set)
    tool_calls: int = 0
    reasoning_events: int = 0
    transcript_events: int = 0
    token_usage_total: int = 0
    token_usage_events: int = 0


def _repo_paths() -> None:
    from readme_runs.smr_slot_client import ensure_evals_importable  # noqa: PLC0415

    ensure_evals_importable()


def _first_text(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _event_id(event: Mapping[str, Any]) -> str:
    for key in ("event_id", "transcript_event_id", "id"):
        value = str(event.get(key) or "").strip()
        if value:
            return value
    return ""


def _event_kind(event: Mapping[str, Any]) -> str:
    return str(event.get("kind") or event.get("event_kind") or "").strip()


def _is_tool_event_kind(kind: str) -> bool:
    return kind.startswith("tool.call.") or kind in {
        "mcp_tool_call",
        "command_execution",
    }


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _format_event_line(
    *,
    role: str,
    actor_id: str,
    event: Mapping[str, Any],
) -> str:
    kind = _event_kind(event)
    occurred_at = str(event.get("occurred_at") or event.get("ts") or "").strip()
    ts = occurred_at[11:19] if len(occurred_at) >= 19 else occurred_at or "?"
    turn_id = str(event.get("turn_id") or "").strip()
    turn_suffix = f" turn={turn_id[-8:]}" if turn_id else ""
    payload = _event_payload(event)

    if _is_tool_event_kind(kind):
        tool_name = _first_text(payload, ("tool_name", "name", "tool"))
        status = kind.rsplit(".", 1)[-1] if kind.startswith("tool.call.") else kind
        detail = _first_text(payload, ("summary", "result_summary", "error", "message"))
        if kind == "command_execution":
            command = _first_text(payload, ("command", "cmd"))
            if command:
                detail = command
        body = f"tool={tool_name or '?'} status={status}"
        if detail:
            body = f"{body} {detail[:160]}"
        return f"[{role} {actor_id[:8]}] {ts} {kind}{turn_suffix} {body}"

    if kind == "reasoning.summary":
        summary = _first_text(payload, ("summary", "text", "content"))
        hidden = bool(payload.get("hidden_reasoning_removed"))
        suffix = " (hidden reasoning removed)" if hidden else ""
        text = summary[:200] + ("..." if len(summary) > 200 else "")
        return f"[{role} {actor_id[:8]}] {ts} reasoning.summary{turn_suffix}{suffix} {text!r}"

    if kind == "token.usage":
        total = payload.get("total_tokens") or payload.get("tokens")
        model = _first_text(payload, ("model",))
        model_suffix = f" model={model}" if model else ""
        return f"[{role} {actor_id[:8]}] {ts} token.usage{turn_suffix}{model_suffix} total={total}"

    if kind in {"turn.started", "turn.completed", "turn.failed", "turn.interrupted"}:
        detail = _first_text(payload, ("summary", "message", "error"))
        detail_suffix = f" {detail[:120]}" if detail else ""
        return f"[{role} {actor_id[:8]}] {ts} {kind}{turn_suffix}{detail_suffix}"

    excerpt = _first_text(payload, ("summary", "text", "message", "instructions"))
    if excerpt:
        excerpt = excerpt[:160] + ("..." if len(excerpt) > 160 else "")
        return f"[{role} {actor_id[:8]}] {ts} {kind}{turn_suffix} {excerpt!r}"
    return f"[{role} {actor_id[:8]}] {ts} {kind}{turn_suffix}"


def _kind_allowed(kind: str, allowed_kinds: frozenset[str] | None) -> bool:
    if not kind:
        return False
    if allowed_kinds is None:
        return True
    if kind in allowed_kinds:
        return True
    return any(
        kind.startswith(prefix.rstrip("*")) for prefix in allowed_kinds if prefix.endswith("*")
    )


def _iter_trace_events(trace_payload: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for bucket in ("events", "live_events"):
        rows = trace_payload.get(bucket)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    yield dict(row)


def _match_actor_filter(actor: Mapping[str, Any], actor_filter: str | None) -> bool:
    if not actor_filter:
        return True
    needle = actor_filter.strip().lower()
    if not needle:
        return True
    haystacks = [
        str(actor.get("actor_id") or ""),
        str(actor.get("actor_key") or actor.get("key") or ""),
        str(actor.get("participant_role") or ""),
        str(actor.get("actor_type") or ""),
    ]
    return any(needle in value.lower() for value in haystacks if value)


def _load_actors(client: Any, project_id: str, run_id: str) -> list[dict[str, Any]]:
    snapshot = client.get_run_observability_snapshot_control(
        project_id,
        run_id,
        actor_limit=50,
        task_limit=50,
        event_limit=20,
    )
    actors = [dict(item.__dict__) for item in snapshot.actors.items]
    if actors:
        return actors
    return client.get_project_run_actors(project_id, run_id)


def _participant_session_id(
    actor: Mapping[str, Any],
    usage_by_actor: Mapping[str, Any],
) -> str:
    labels = actor.get("labels")
    if isinstance(labels, Mapping):
        value = str(labels.get("participant_session_id") or "").strip()
        if value:
            return value
    actor_id = str(actor.get("actor_id") or "").strip()
    usage_row = usage_by_actor.get(actor_id)
    worker_id = str(getattr(usage_row, "worker_id", "") or "").strip()
    return worker_id


def _fetch_transcript_events(
    client: Any,
    *,
    run_id: str,
    participant_session_id: str | None,
    state: ActorTraceState,
    trace_limit: int,
    transcript_view: str,
) -> dict[str, Any]:
    """Fetch durable + live transcript events from the deployed runtime transcript API."""
    combined: dict[str, Any] = {"events": [], "live_events": []}

    durable = client.get_run_transcript(
        run_id,
        cursor=state.transcript_cursor,
        limit=trace_limit,
        participant_session_id=participant_session_id,
        view=transcript_view,
    )
    combined["events"] = list(durable.get("events") or [])
    state.transcript_cursor = (
        str(durable.get("next_cursor") or state.transcript_cursor or "").strip() or None
    )
    resume_cursor = str(durable.get("live_resume_cursor") or "").strip()
    if resume_cursor and state.live_cursor is None:
        state.live_cursor = resume_cursor

    if state.live_cursor:
        live = client.get_run_transcript(
            run_id,
            cursor=state.live_cursor,
            limit=trace_limit,
            participant_session_id=participant_session_id,
            view=transcript_view,
        )
        # Live tail reuses transcript paging; keep any newly appended events.
        live_events = list(live.get("events") or [])
        combined["live_events"] = live_events
        next_live = str(live.get("next_cursor") or live.get("live_resume_cursor") or "").strip()
        if next_live:
            state.live_cursor = next_live
    return combined


def _print_task_events(client: Any, project_id: str, run_id: str, *, limit: int) -> None:
    try:
        payload = client.list_run_task_events(project_id, run_id, limit=limit)
    except Exception as exc:
        print(f"[inspect] task-events unavailable: {exc}")
        return
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        print("[inspect] task-events: none")
        return
    print(f"[inspect] task-events ({len(events)} recent):")
    for event in events[-limit:]:
        if not isinstance(event, Mapping):
            continue
        summary = str(event.get("summary") or event.get("display_kind") or "").strip()
        state = str(event.get("event_state") or event.get("event_kind") or "").strip()
        task_key = str(event.get("task_key") or event.get("task_id") or "").strip()
        occurred_at = str(event.get("occurred_at") or "")[:19]
        print(f"  {occurred_at} task={task_key or '?'} state={state} {summary[:140]}")


def _print_actor_header(actor: Mapping[str, Any], trace_key: str) -> str:
    role = str(actor.get("participant_role") or actor.get("actor_type") or "?").strip()
    actor_id = str(actor.get("actor_id") or "").strip()
    state = str(actor.get("state") or "?").strip()
    session = str(actor.get("session_status") or "").strip()
    session_suffix = f" session={session}" if session else ""
    return f"{role} id={actor_id[:8]} key={trace_key} state={state}{session_suffix}"


def inspect_once(
    client: Any,
    *,
    project_id: str,
    run_id: str,
    actor_filter: str | None,
    allowed_kinds: frozenset[str] | None,
    actor_states: dict[str, ActorTraceState],
    task_event_limit: int,
    trace_limit: int,
    show_task_events: bool,
    emit_json: bool,
    transcript_view: str,
) -> dict[str, Any]:
    snapshot = client.get_run_observability_snapshot_control(
        project_id,
        run_id,
        actor_limit=50,
        task_limit=50,
        event_limit=20,
    )
    run_state = str(getattr(snapshot.run, "public_state", "") or "").strip().lower()
    task_count = int(getattr(snapshot.tasks, "total_count", 0) or 0)
    actor_count = int(getattr(snapshot.actors, "total_count", 0) or 0)
    roles = dict(getattr(snapshot.actors, "counts_by_role", {}) or {})

    usage = client.get_project_run_actor_usage(project_id, run_id)
    usage_by_actor = {str(row.actor_id): row for row in getattr(usage, "actors", ()) or ()}

    from readme_runs.smr_slot_client import actor_trace_key  # noqa: PLC0415

    actors = _load_actors(client, project_id, run_id)

    emitted: list[dict[str, Any]] = []
    new_lines: list[str] = []

    for actor in actors:
        if not _match_actor_filter(actor, actor_filter):
            continue
        actor_id = str(actor.get("actor_id") or "").strip()
        trace_key = actor_trace_key(dict(actor))
        session_id = _participant_session_id(actor, usage_by_actor)
        state = actor_states.setdefault(
            actor_id or trace_key or session_id or "unknown",
            ActorTraceState(
                actor_key=trace_key or actor_id,
                actor_id=actor_id,
                participant_role=str(actor.get("participant_role") or ""),
                participant_session_id=session_id,
                actor_state=str(actor.get("state") or ""),
            ),
        )
        state.actor_id = actor_id or state.actor_id
        state.participant_role = str(actor.get("participant_role") or state.participant_role)
        state.actor_state = str(actor.get("state") or state.actor_state)
        if session_id:
            state.participant_session_id = session_id

        if not session_id:
            continue

        trace_payload = _fetch_transcript_events(
            client,
            run_id=run_id,
            participant_session_id=session_id,
            state=state,
            trace_limit=trace_limit,
            transcript_view=transcript_view,
        )

        for event in _iter_trace_events(trace_payload):
            event_id = _event_id(event)
            kind = _event_kind(event)
            if event_id and event_id in state.seen_event_ids:
                continue
            if not _kind_allowed(kind, allowed_kinds):
                if event_id:
                    state.seen_event_ids.add(event_id)
                continue
            if event_id:
                state.seen_event_ids.add(event_id)
            state.transcript_events += 1
            if _is_tool_event_kind(kind):
                state.tool_calls += 1
            if kind == "reasoning.summary":
                state.reasoning_events += 1
            role = str(event.get("participant_role") or state.participant_role or "?")
            line = _format_event_line(role=role, actor_id=state.actor_id or trace_key, event=event)
            new_lines.append(line)
            emitted.append({"actor_key": trace_key, "line": line, "event": event})

    usage_events = sum(int(getattr(row, "event_count", 0) or 0) for row in usage_by_actor.values())
    transcript_total = sum(s.transcript_events for s in actor_states.values())
    tool_total = sum(s.tool_calls for s in actor_states.values())
    reasoning_total = sum(s.reasoning_events for s in actor_states.values())

    summary = {
        "run_state": run_state,
        "tasks": task_count,
        "actors": actor_count,
        "roles": roles,
        "usage_events": usage_events,
        "transcript_events": transcript_total,
        "tool_calls": tool_total,
        "reasoning_events": reasoning_total,
        "tracked_actors": len(actor_states),
        "new_events": len(emitted),
    }

    print(
        "[inspect] "
        f"run={run_state or '?'} tasks={task_count} actors={actor_count} "
        f"roles={roles} usage_events={usage_events} "
        f"transcript={transcript_total} tool_calls={tool_total} reasoning={reasoning_total}"
    )

    if show_task_events:
        _print_task_events(client, project_id, run_id, limit=task_event_limit)

    for actor in actors:
        if not _match_actor_filter(actor, actor_filter):
            continue
        actor_id = str(actor.get("actor_id") or "").strip()
        trace_key = actor_trace_key(dict(actor))
        session_id = _participant_session_id(actor, usage_by_actor)
        usage_row = usage_by_actor.get(actor_id)
        usage_detail = ""
        if usage_row is not None:
            usage_detail = f" usage_events={usage_row.event_count}"
        session_suffix = (
            f" session={session_id[:36]}..."
            if len(session_id) > 36
            else (f" session={session_id}" if session_id else " session=none")
        )
        print(
            f"[inspect] actor {_print_actor_header(actor, trace_key or actor_id)}"
            f"{session_suffix}{usage_detail}"
        )
        if not session_id:
            print(f"[inspect]   no participant_session_id yet for {actor_id[:8]} — agent not bound")

    for line in new_lines:
        print(line)

    if emit_json:
        print(json.dumps({"summary": summary, "events": emitted}, indent=2, default=str))

    summary["terminal"] = run_state in TERMINAL_RUN_STATES
    return summary


def parse_kind_filter(raw: str | None) -> frozenset[str] | None:
    if raw is None:
        return DEFAULT_PROGRESS_KINDS
    text = raw.strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return frozenset(part.strip() for part in text.split(",") if part.strip())


def main() -> int:
    _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--slot", default="slot1")
    parser.add_argument("--slot-mode", default="local-dockerized")
    parser.add_argument("--backend", default=None, help="Override backend base URL")
    parser.add_argument("--api-key", default=None, help="Override SYNTH_API_KEY")
    parser.add_argument("--actor", default=None, help="Filter by actor id, key, or role substring")
    parser.add_argument("--watch", action="store_true", help="Poll until terminal or timeout")
    parser.add_argument("--interval", type=float, default=5.0, help="Watch poll interval seconds")
    parser.add_argument("--timeout", type=int, default=3600, help="Watch timeout seconds")
    parser.add_argument("--trace-limit", type=int, default=200)
    parser.add_argument("--task-event-limit", type=int, default=8)
    parser.add_argument(
        "--kinds",
        default=None,
        help="Comma-separated transcript kinds (default: progress set; use 'all' for everything)",
    )
    parser.add_argument("--task-events", action="store_true", help="Print recent run task events")
    parser.add_argument(
        "--view",
        default="debug",
        choices=("operator", "debug", "public"),
        help="Transcript visibility view (debug shows tool/thinking summaries)",
    )
    parser.add_argument("--json", action="store_true", help="Also emit JSON event payload")
    args = parser.parse_args()

    from readme_runs.smr_slot_client import build_managed_research_client_for_slot  # noqa: PLC0415

    client = build_managed_research_client_for_slot(
        args.slot,
        slot_mode=args.slot_mode,
        api_key=args.api_key,
        backend_base=args.backend,
    )
    allowed_kinds = parse_kind_filter(args.kinds)
    actor_states: dict[str, ActorTraceState] = {}
    started = time.monotonic()
    deadline = started + max(1, int(args.timeout))

    try:
        while True:
            summary = inspect_once(
                client,
                project_id=args.project_id,
                run_id=args.run_id,
                actor_filter=args.actor,
                allowed_kinds=allowed_kinds,
                actor_states=actor_states,
                task_event_limit=args.task_event_limit,
                trace_limit=args.trace_limit,
                show_task_events=args.task_events,
                emit_json=args.json and not args.watch,
                transcript_view=args.view,
            )
            if not args.watch:
                return 0
            if summary.get("terminal"):
                print(f"[inspect] run reached terminal state={summary.get('run_state')!r}")
                return 0
            if time.monotonic() >= deadline:
                print("[inspect] watch timeout reached")
                return 0
            time.sleep(max(0.5, float(args.interval)))
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
