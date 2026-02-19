#!/usr/bin/env python3
"""Managed Research Banking77 eval flow via SDK + starting-data upload.

This script uploads the bundled Banking77 starting-data files, writes
`execution.input_spec` from `input_spec.json`, and optionally triggers a run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from synth_ai.sdk.managed_research import ACTIVE_RUN_STATES, SmrControlClient


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Banking77 managed-research eval via SDK")
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
    parser.add_argument("--project-id", required=True, help="Managed research project id.")
    parser.add_argument(
        "--dataset-ref",
        default="starting-data/banking77",
        help="Dataset ref persisted to harness state for starting-data upload.",
    )
    parser.add_argument(
        "--starting-data-dir",
        default=str(here / "banking77_starting_data"),
        help="Directory of files to upload as starting data.",
    )
    parser.add_argument(
        "--trigger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trigger a run after upload + input-spec patch.",
    )
    parser.add_argument(
        "--patch-input-spec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch project.execution.input_spec from starting_data/input_spec.json.",
    )
    parser.add_argument(
        "--timebox-seconds",
        type=int,
        default=2 * 60 * 60,
        help="Optional run timebox when triggering.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="How long to poll for the triggered run state.",
    )
    parser.add_argument(
        "--stop-active-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop the triggered run before exit if still active.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Emit compact JSON summary.",
    )
    return parser.parse_args()


def emit(payload: dict[str, Any], *, compact: bool) -> None:
    if compact:
        print(json.dumps(payload, separators=(",", ":"), default=str))
    else:
        print(json.dumps(payload, indent=2, default=str))


def _read_input_spec(starting_data_dir: Path) -> dict[str, Any]:
    spec_path = starting_data_dir / "input_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing input spec file: {spec_path}")
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"input_spec.json must contain a JSON object: {spec_path}")
    return data


def _find_run(runs: list[dict[str, Any]], run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    for run in runs:
        candidate = run.get("run_id") or run.get("id")
        if isinstance(candidate, str) and candidate == run_id:
            return run
    return None


def main() -> None:
    args = parse_args()
    starting_data_dir = Path(args.starting_data_dir).expanduser().resolve()

    summary: dict[str, Any] = {
        "backend_base": args.backend_base,
        "project_id": args.project_id,
        "dataset_ref": args.dataset_ref,
        "starting_data_dir": str(starting_data_dir),
        "patch_input_spec": args.patch_input_spec,
        "trigger_enabled": args.trigger,
        "stopped_triggered_run": False,
    }

    input_spec = _read_input_spec(starting_data_dir)

    with SmrControlClient(api_key=args.api_key, backend_base=args.backend_base) as client:
        upload_result = client.upload_starting_data_directory(
            args.project_id,
            starting_data_dir,
            dataset_ref=args.dataset_ref,
        )
        uploads = upload_result.get("uploads")
        upload_count = len(uploads) if isinstance(uploads, list) else None
        summary["starting_data_s3_path"] = upload_result.get("s3_path")
        summary["starting_data_upload_count"] = upload_count

        if args.patch_input_spec:
            project = client.get_project(args.project_id)
            execution = project.get("execution") if isinstance(project, dict) else None
            execution_payload = dict(execution) if isinstance(execution, dict) else {}
            existing_input_spec = execution_payload.get("input_spec")
            if existing_input_spec == input_spec:
                summary["input_spec_patched"] = False
            else:
                execution_payload["input_spec"] = input_spec
                client.patch_project(args.project_id, {"execution": execution_payload})
                summary["input_spec_patched"] = True
            summary["input_spec_kind"] = input_spec.get("kind")
        else:
            summary["input_spec_patched"] = False

        if args.trigger:
            trigger_result = client.trigger_run(args.project_id, timebox_seconds=args.timebox_seconds)
            run_id = trigger_result.get("run_id") or trigger_result.get("id")
            summary["trigger_run_id"] = run_id

            run = None
            deadline = time.time() + max(0, int(args.poll_seconds))
            while time.time() < deadline:
                runs_now = client.list_runs(args.project_id)
                run = _find_run(runs_now, run_id)
                if run is not None:
                    break
                time.sleep(1.0)

            if run is None and isinstance(run_id, str) and run_id:
                try:
                    run = client.get_run(run_id, project_id=args.project_id)
                except Exception:
                    run = None

            if run is not None:
                run_state = str(run.get("state") or "").lower()
                summary["trigger_run_state"] = run_state
                summary["trigger_run_id"] = run.get("run_id") or run.get("id") or run_id
                if args.stop_active_run and run_state in ACTIVE_RUN_STATES:
                    client.stop_run(str(summary["trigger_run_id"]))
                    summary["stopped_triggered_run"] = True

    emit(summary, compact=args.compact_json)


if __name__ == "__main__":
    main()
