#!/usr/bin/env python3
"""Instrumented readme-smoke watcher: asserts on orchestrator bootstrap milestones."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

BAD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"stale_orphaned_orchestrator_start.*row_state=missing", "bootstrap_missing_queue_sigkill"),
    (r"actor_terminal_boundary_preserved", "terminal_boundary_block"),
    (r"RuntimeParticipantStartRejected.*actor_terminal", "participant_start_actor_terminal"),
    (r"no_durable_effect", "orchestrator_no_plan_tasks"),
    (r"external_sigkill", "orchestrator_sigkill"),
)

GOOD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"start_queue_projection_pending", "bootstrap_grace_active"),
    (r"runtime_intent_plan_tasks|plan_tasks", "plan_tasks_seen"),
    (
        r"smr\.runtime\.orchestrator_cycle\.(session_start|turn_complete)",
        "orchestrator_turn_progress",
    ),
)


@dataclass
class DiagState:
    run_id: str = ""
    project_id: str = ""
    bad_hits: dict[str, list[str]] = field(default_factory=dict)
    good_hits: dict[str, int] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    max_tasks: int = 0
    max_actors: int = 0
    roles_seen: set[str] = field(default_factory=set)


def _run(cmd: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def tail_smrt_logs(container: str, since: str) -> list[str]:
    proc = _run(
        ["docker", "logs", "--since", since, container],
        timeout=180,
    )
    blob = (proc.stdout or "") + (proc.stderr or "")
    return [line for line in blob.splitlines() if line.strip()]


def scan_logs(state: DiagState, lines: list[str]) -> None:
    for line in lines:
        state.log_lines.append(line)
        for pattern, key in BAD_PATTERNS:
            if re.search(pattern, line):
                state.bad_hits.setdefault(key, []).append(line[-500:])
        for pattern, key in GOOD_PATTERNS:
            if re.search(pattern, line):
                state.good_hits[key] = state.good_hits.get(key, 0) + 1


def poll_run(client: Any, project_id: str, run_id: str, state: DiagState) -> dict[str, Any]:
    summary = client.get_run_observability_snapshot_control(project_id, run_id)
    tasks = int(getattr(summary.tasks, "total_count", 0) or 0)
    actors = int(getattr(summary.actors, "total_count", 0) or 0)
    state.max_tasks = max(state.max_tasks, tasks)
    state.max_actors = max(state.max_actors, actors)
    counts = getattr(summary.actors, "counts_by_role", None) or {}
    if isinstance(counts, dict):
        state.roles_seen.update(str(k) for k in counts)
    run = getattr(summary, "run", None)
    return {
        "public_state": getattr(run, "public_state", None) or getattr(summary, "state", ""),
        "tasks": tasks,
        "actors": actors,
        "counts_by_role": dict(counts) if isinstance(counts, dict) else {},
        "liveness": getattr(run, "liveness_phase", ""),
    }


def assert_milestones(state: DiagState, elapsed_s: float) -> list[str]:
    failures: list[str] = []
    if elapsed_s <= 45 and state.bad_hits.get("bootstrap_missing_queue_sigkill"):
        failures.append(
            "FAIL<=45s: stale_orphaned_orchestrator_start with row_state=missing "
            "(bootstrap queue projection race)"
        )
    if state.bad_hits.get("terminal_boundary_block"):
        failures.append("FAIL: actor_terminal_boundary_preserved — orchestrator cannot recover")
    if state.bad_hits.get("participant_start_actor_terminal"):
        failures.append("FAIL: participant start rejected with actor_terminal")
    if elapsed_s >= 150 and state.max_tasks == 0 and state.bad_hits.get("orchestrator_sigkill"):
        failures.append(
            "FAIL>=150s: orchestrator SIGKILL before plan_tasks "
            "(likely stale queue row cancel during first turn)"
        )
    if elapsed_s >= 300 and state.max_tasks == 0 and not state.good_hits.get("plan_tasks_seen"):
        failures.append("FAIL>=300s: no plan_tasks / durable planning effect observed")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--container", default="synth-slot1-smr-runtime-1")
    parser.add_argument("--since", default="5m")
    parser.add_argument("--watch-seconds", type=int, default=420)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--slot", default="slot1")
    args = parser.parse_args()

    from readme_runs.smr_slot_client import (  # noqa: WPS433
        build_managed_research_client_for_slot,
        ensure_evals_importable,
    )

    ensure_evals_importable()

    client = build_managed_research_client_for_slot(args.slot, slot_mode="local-dockerized")
    state = DiagState(run_id=args.run_id, project_id=args.project_id)
    started = time.monotonic()
    deadline = started + args.watch_seconds
    last_fail_print = 0.0

    print(
        f"[diag] watching run_id={args.run_id} project_id={args.project_id} "
        f"container={args.container} for {args.watch_seconds}s"
    )

    while time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        lines = tail_smrt_logs(args.container, args.since)
        scan_logs(state, lines[-400:])
        snap = poll_run(client, args.project_id, args.run_id, state)
        print(
            f"[diag] elapsed={elapsed:.0f}s state={snap['public_state']!r} "
            f"tasks={snap['tasks']} actors={snap['actors']} "
            f"roles={snap['counts_by_role']} good={state.good_hits} "
            f"bad_keys={list(state.bad_hits.keys())}"
        )
        failures = assert_milestones(state, elapsed)
        if failures:
            now = time.monotonic()
            if now - last_fail_print > 30 or snap["tasks"] > 0:
                for msg in failures:
                    print(f"[diag] {msg}")
                last_fail_print = now
        if snap["tasks"] > 0 and "worker" in state.roles_seen:
            print("[diag] PASS: tasks planned and worker role appeared")
            print(json.dumps({"state": state, "snap": snap}, indent=2, default=str))
            return 0
        pub = str(snap["public_state"] or "").lower()
        if pub in {"completed", "failed", "canceled", "cancelled"}:
            failures = assert_milestones(state, elapsed)
            print(f"[diag] terminal state={pub}")
            for msg in failures:
                print(f"[diag] {msg}")
            print(json.dumps({"state": state, "snap": snap}, indent=2, default=str))
            return 1 if failures or snap["tasks"] == 0 else 0
        time.sleep(args.poll_seconds)

    failures = assert_milestones(state, time.monotonic() - started)
    print("[diag] watch timeout")
    for msg in failures:
        print(f"[diag] {msg}")
    print(json.dumps({"state": state}, indent=2, default=str))
    return 1 if failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
