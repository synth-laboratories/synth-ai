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

import requests

from synth_ai.sdk.managed_research import ACTIVE_RUN_STATES, SmrControlClient, first_id


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
    parser.add_argument(
        "--check-eval-health",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify eval server reachability before triggering. "
            "Prevents long backend-side health-check failures when container_url points to a local app."
        ),
    )
    parser.add_argument(
        "--eval-health-url",
        default=os.environ.get("SMR_EVAL_HEALTH_URL"),
        help=(
            "Explicit eval health URL to probe before trigger (for example "
            "http://127.0.0.1:8102/health). If omitted, the script tries project config, "
            "then SMR_SYNTH_AI_CONTAINER_URL/SMR_EVAL_URL."
        ),
    )
    parser.add_argument(
        "--eval-health-timeout-seconds",
        type=float,
        default=_env_float("SMR_EVAL_HEALTH_TIMEOUT_SECONDS", 2.5),
        help="Per-attempt timeout for eval health probing.",
    )
    parser.add_argument(
        "--eval-health-retries",
        type=int,
        default=_env_int("SMR_EVAL_HEALTH_RETRIES", 3),
        help="Number of eval health probe attempts before failing.",
    )
    parser.add_argument(
        "--eval-health-retry-sleep-seconds",
        type=float,
        default=_env_float("SMR_EVAL_HEALTH_RETRY_SLEEP_SECONDS", 1.0),
        help="Sleep duration between eval health probe attempts.",
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


def _normalize_health_url(raw: str | None) -> str | None:
    value = str(raw or "").strip()
    if not value:
        return None
    value = value.rstrip("/")
    if value.endswith("/health"):
        return value
    return f"{value}/health"


def _extract_container_url_from_project(project: dict[str, Any]) -> str | None:
    if not isinstance(project, dict):
        return None

    scopes: list[dict[str, Any]] = [project]
    for key in ("config_snapshot", "config"):
        nested = project.get(key)
        if isinstance(nested, dict):
            scopes.append(nested)

    for scope in scopes:
        synth_ai = scope.get("synth_ai")
        if isinstance(synth_ai, dict):
            for section in ("policy_optimization", "prompt_learning"):
                cfg = synth_ai.get(section)
                if isinstance(cfg, dict):
                    candidate = str(cfg.get("container_url") or "").strip()
                    if candidate:
                        return candidate
        prompt_learning = scope.get("prompt_learning")
        if isinstance(prompt_learning, dict):
            candidate = str(prompt_learning.get("container_url") or "").strip()
            if candidate:
                return candidate
    return None


def _resolve_eval_health_url(args: argparse.Namespace, project: dict[str, Any]) -> str | None:
    explicit = _normalize_health_url(getattr(args, "eval_health_url", None))
    if explicit:
        return explicit

    project_container_url = _extract_container_url_from_project(project)
    if project_container_url:
        return _normalize_health_url(project_container_url)

    env_candidate = (
        (os.environ.get("SMR_SYNTH_AI_CONTAINER_URL") or "").strip()
        or (os.environ.get("SMR_EVAL_URL") or "").strip()
    )
    if env_candidate:
        return _normalize_health_url(env_candidate)
    return None


def _probe_eval_health(
    url: str,
    *,
    timeout_seconds: float,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    attempt_count = max(1, int(retries))
    timeout = max(0.1, float(timeout_seconds))
    retry_sleep = max(0.0, float(retry_sleep_seconds))
    last_error: str | None = None
    last_status_code: int | None = None

    for attempt in range(1, attempt_count + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            last_status_code = int(resp.status_code)
            if 200 <= resp.status_code < 300:
                return {
                    "ok": True,
                    "attempts": attempt,
                    "status_code": int(resp.status_code),
                    "url": url,
                }
            body_preview = (resp.text or "").strip().replace("\n", " ")
            if len(body_preview) > 240:
                body_preview = body_preview[:240] + "..."
            last_error = (
                f"HTTP {resp.status_code}"
                + (f" body={body_preview!r}" if body_preview else "")
            )
        except Exception as exc:
            last_error = str(exc)

        if attempt < attempt_count and retry_sleep > 0:
            time.sleep(retry_sleep)

    return {
        "ok": False,
        "attempts": attempt_count,
        "status_code": last_status_code,
        "last_error": last_error or "<unknown>",
        "url": url,
    }


def _format_eval_health_preflight_failure(result: dict[str, Any]) -> str:
    url = str(result.get("url") or "<unknown>")
    attempts = int(result.get("attempts") or 0)
    detail = str(result.get("last_error") or f"HTTP {result.get('status_code')}")
    return (
        "Eval server preflight failed before triggering run: "
        f"{url} was unreachable after {attempts} attempt(s) ({detail}). "
        "Start the eval server first (for example the banking77 task app on :8102), "
        "or pass --no-check-eval-health to bypass. "
        "Note: skip_health_check=True only skips SDK-side preflight checks; backend workers "
        "still run container_url health checks."
    )


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
        project = client.get_project(selected_project_id)
        summary["project_name"] = project.get("name")
        status = client.get_project_status(selected_project_id)
        summary["project_status"] = status.get("state") or status.get("status")

        runs_before = client.list_runs(selected_project_id)
        summary["runs_before"] = len(runs_before)

        if args.trigger:
            if args.check_eval_health:
                eval_health_url = _resolve_eval_health_url(args, project)
                summary["eval_health_url"] = eval_health_url
                if eval_health_url:
                    preflight = _probe_eval_health(
                        eval_health_url,
                        timeout_seconds=args.eval_health_timeout_seconds,
                        retries=args.eval_health_retries,
                        retry_sleep_seconds=args.eval_health_retry_sleep_seconds,
                    )
                    summary["eval_health_preflight"] = preflight
                    if not preflight.get("ok"):
                        raise SystemExit(_format_eval_health_preflight_failure(preflight))
                else:
                    summary["eval_health_preflight"] = {
                        "ok": None,
                        "status": "skipped_no_url",
                        "reason": (
                            "No eval health URL resolved from --eval-health-url, project config, "
                            "SMR_SYNTH_AI_CONTAINER_URL, or SMR_EVAL_URL."
                        ),
                    }

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
