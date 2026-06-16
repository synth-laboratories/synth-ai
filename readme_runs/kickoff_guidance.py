"""Guidance-only kickoff contract patching for ReportBench lane drivers."""

from __future__ import annotations

import copy
import json
from typing import Any, Sequence


def assert_guidance_only_kickoff(kickoff: dict[str, Any]) -> None:
    """Fail fast when structural kickoff tasks or contract WP specs remain."""
    tasks = kickoff.get("tasks")
    if isinstance(tasks, list) and tasks:
        raise ValueError(
            f"guidance-only kickoff requires empty tasks; found {len(tasks)} seeded task(s)"
        )
    required_work_products = kickoff.get("required_work_products")
    if isinstance(required_work_products, list) and required_work_products:
        raise ValueError(
            "guidance-only kickoff requires empty contract required_work_products; "
            f"found {len(required_work_products)} spec(s)"
        )


def _guidance_only_kickoff_payload(
    *,
    project_notes: str,
    task_briefs: Sequence[str],
    plan_task_payloads: Sequence[dict[str, Any]] | None = None,
    base_kickoff: dict[str, Any],
) -> dict[str, Any]:
    kickoff = copy.deepcopy(base_kickoff)
    kickoff["project_notes_framing"] = str(project_notes or "").strip()
    kickoff["task_briefs"] = [str(brief).strip() for brief in task_briefs if str(brief).strip()]
    if plan_task_payloads is not None:
        kickoff["plan_task_payloads"] = [
            copy.deepcopy(dict(payload))
            for payload in plan_task_payloads
            if isinstance(payload, dict)
        ]
    kickoff["tasks"] = []
    kickoff["required_work_products"] = []
    assert_guidance_only_kickoff(kickoff)
    return kickoff


def _patch_workspace_file_content(
    files: Any,
    *,
    kickoff: dict[str, Any],
    task_instructions_md: str | None,
) -> None:
    if not isinstance(files, list):
        return
    kickoff_json = json.dumps(kickoff, indent=2, ensure_ascii=True)
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if path.endswith("kickoff_contract.json"):
            item["content"] = kickoff_json
        if task_instructions_md is not None and path.endswith("TASK_INSTRUCTIONS.md"):
            item["content"] = task_instructions_md


def apply_guidance_only_kickoff(
    bundle: dict[str, Any],
    *,
    project_notes: str,
    task_briefs: Sequence[str],
    plan_task_payloads: Sequence[dict[str, Any]] | None = None,
    task_instructions_md: str | None = None,
) -> dict[str, Any]:
    """Patch all kickoff copies to guidance-only shape (empty tasks, rich briefs)."""
    trigger_payload = bundle.setdefault("trigger_payload", {})
    base_kickoff = trigger_payload.get("kickoff_contract")
    if not isinstance(base_kickoff, dict):
        raise RuntimeError("launch bundle missing trigger_payload.kickoff_contract")

    kickoff = _guidance_only_kickoff_payload(
        project_notes=project_notes,
        task_briefs=task_briefs,
        plan_task_payloads=plan_task_payloads,
        base_kickoff=base_kickoff,
    )

    trigger_payload["kickoff_contract"] = copy.deepcopy(kickoff)

    runnable = bundle.get("runnable_project_request")
    if isinstance(runnable, dict):
        runnable["notes"] = kickoff["project_notes_framing"]
        research = runnable.setdefault("research", {})
        if isinstance(research, dict):
            research["staged_kickoff_contract"] = copy.deepcopy(kickoff)

    effective = bundle.setdefault("effective_config", {})
    smr = effective.setdefault("smr", {})
    if isinstance(smr, dict):
        smr["staged_kickoff_contract"] = copy.deepcopy(kickoff)

    workspace_inputs = bundle.get("workspace_inputs")
    if isinstance(workspace_inputs, dict):
        _patch_workspace_file_content(
            workspace_inputs.get("files"),
            kickoff=kickoff,
            task_instructions_md=task_instructions_md,
        )
        _patch_workspace_file_content(
            workspace_inputs.get("all_files"),
            kickoff=kickoff,
            task_instructions_md=task_instructions_md,
        )

    return kickoff


def kickoff_guidance_summary(kickoff: dict[str, Any]) -> dict[str, Any]:
    """Metadata for driver summary.json."""
    task_briefs = kickoff.get("task_briefs")
    brief_count = len(task_briefs) if isinstance(task_briefs, list) else 0
    plan_task_payloads = kickoff.get("plan_task_payloads")
    plan_task_payload_count = len(plan_task_payloads) if isinstance(plan_task_payloads, list) else 0
    tasks = kickoff.get("tasks")
    task_count = len(tasks) if isinstance(tasks, list) else 0
    return {
        "kickoff_guidance_mode": "guidance_only",
        "kickoff_tasks_count": task_count,
        "task_brief_count": brief_count,
        "plan_task_payload_count": plan_task_payload_count,
    }
