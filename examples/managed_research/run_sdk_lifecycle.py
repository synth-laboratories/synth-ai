#!/usr/bin/env python3
"""SDK-first Managed Research lifecycle example.

This script mirrors a "hello world" end-to-end flow using the Python SDK:
1. List projects.
2. Pick a project (or use --project-id).
3. Fetch status and existing runs.
4. Trigger a run.
5. Optionally stop the run if it is still active.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from synth_ai.sdk.managed_research import ACTIVE_RUN_STATES, SmrControlClient, first_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an SMR lifecycle flow via synth-ai SDK")
    parser.add_argument(
        "--backend-base",
        default=os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000"),
        help="Synth backend base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SYNTH_API_KEY"),
        help="Synth API key (defaults to SYNTH_API_KEY env var).",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Project id to use. If omitted, first visible project is used.",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived projects in listing (selection still uses first match).",
    )
    parser.add_argument(
        "--trigger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger a run as part of the flow.",
    )
    parser.add_argument(
        "--timebox-seconds",
        type=int,
        default=120,
        help="Optional run timebox when triggering a run.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=12,
        help="How long to poll for the newly triggered run to appear.",
    )
    parser.add_argument(
        "--stop-active-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the triggered run is active, stop it before exit.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Emit compact JSON instead of pretty JSON.",
    )
    return parser.parse_args()


def emit(payload: dict[str, Any], *, compact: bool) -> None:
    if compact:
        print(json.dumps(payload, separators=(",", ":"), default=str))
    else:
        print(json.dumps(payload, indent=2, default=str))


def find_run_by_id(runs: list[dict[str, Any]], run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    for run in runs:
        candidate_id = run.get("run_id") or run.get("id")
        if isinstance(candidate_id, str) and candidate_id == run_id:
            return run
    return None


def main() -> None:
    args = parse_args()

    summary: dict[str, Any] = {
        "backend_base": args.backend_base,
        "project_id": args.project_id,
        "trigger_enabled": args.trigger,
        "stopped_triggered_run": False,
    }

    with SmrControlClient(api_key=args.api_key, backend_base=args.backend_base) as client:
        projects = client.list_projects(include_archived=args.include_archived)
        summary["projects_visible"] = len(projects)
        selected_project_id = args.project_id or first_id(projects, "project_id")
        if not selected_project_id:
            raise SystemExit("No projects available for this account/backend.")

        summary["project_id"] = selected_project_id
        status = client.get_project_status(selected_project_id)
        summary["project_status"] = status.get("state") or status.get("status")

        runs_before = client.list_runs(selected_project_id)
        summary["runs_before"] = len(runs_before)

        if args.trigger:
            trigger_result = client.trigger_run(selected_project_id, timebox_seconds=args.timebox_seconds)
            run_id = trigger_result.get("run_id") or trigger_result.get("id")
            summary["trigger_run_id"] = run_id

            run = None
            deadline = time.time() + max(0, int(args.poll_seconds))
            while time.time() < deadline:
                runs_now = client.list_runs(selected_project_id)
                run = find_run_by_id(runs_now, run_id)
                if run is not None:
                    break
                time.sleep(1.0)

            if run is None and isinstance(run_id, str) and run_id:
                try:
                    run = client.get_run(run_id, project_id=selected_project_id)
                except Exception:
                    run = None

            if run is not None:
                run_state = str(run.get("state") or "").lower()
                summary["trigger_run_state"] = run_state
                summary["trigger_run_id"] = run.get("run_id") or run.get("id") or run_id
                if args.stop_active_run and run_state in ACTIVE_RUN_STATES:
                    client.stop_run(str(summary["trigger_run_id"]))
                    summary["stopped_triggered_run"] = True

        runs_after = client.list_runs(selected_project_id)
        summary["runs_after"] = len(runs_after)

    emit(summary, compact=args.compact_json)


if __name__ == "__main__":
    main()
