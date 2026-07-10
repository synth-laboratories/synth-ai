"""One-shot Research Factory stand-up workflow over the public SDK."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from synth_ai.managed_research.models.types import SmrRunnableProjectRequest
from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return _jsonable(to_wire())
    return value


def _json_object(raw: str, *, field: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("@"):
        value = Path(value[1:]).expanduser().read_text()
    elif not value.startswith("{"):
        candidate = Path(value).expanduser()
        if candidate.exists():
            value = candidate.read_text()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object or @path to one")
    return dict(parsed)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _required_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _required_string(parent: Mapping[str, Any], key: str) -> str:
    value = str(parent.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _project_id_from_response(payload: Mapping[str, Any]) -> str:
    project_id = _optional_string(payload.get("project_id") or payload.get("id"))
    if project_id:
        return project_id
    project = payload.get("project")
    if isinstance(project, Mapping):
        project_id = _optional_string(project.get("project_id") or project.get("id"))
        if project_id:
            return project_id
    raise ValueError("create_project response did not include project_id")


def _factory_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    factory = _required_mapping(plan, "factory")
    return {
        "name": _required_string(factory, "name"),
        "kind": str(factory.get("kind") or "customer"),
        "description": _optional_string(factory.get("description")),
        "budget_policy": _mapping(
            factory.get("budget_policy"),
            field="factory.budget_policy",
        ),
        "cap_policy": _mapping(factory.get("cap_policy"), field="factory.cap_policy"),
        "publication_policy": _mapping(
            factory.get("publication_policy"),
            field="factory.publication_policy",
        ),
        "authorization_policy": _mapping(
            factory.get("authorization_policy"),
            field="factory.authorization_policy",
        ),
        "metadata": _mapping(factory.get("metadata"), field="factory.metadata"),
    }


def _project_link_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    project = _mapping(plan.get("project"), field="project")
    return {
        "role": str(project.get("role") or "canonical"),
        "status": str(project.get("status") or "active"),
        "display_name": _optional_string(project.get("display_name")),
        "description": _optional_string(project.get("description")),
        "workspace_policy": _mapping(
            project.get("workspace_policy"),
            field="project.workspace_policy",
        ),
        "resource_bindings": _mapping(
            project.get("resource_bindings"),
            field="project.resource_bindings",
        ),
        "feed_health": _mapping(project.get("feed_health"), field="project.feed_health"),
        "default_launch_profile": _mapping(
            project.get("default_launch_profile"),
            field="project.default_launch_profile",
        ),
        "metadata": _mapping(project.get("metadata"), field="project.metadata"),
    }


def _effort_plans(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    efforts = plan.get("efforts")
    if not isinstance(efforts, list) or not efforts:
        raise ValueError("efforts must be a non-empty JSON array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(efforts):
        if not isinstance(item, Mapping):
            raise ValueError(f"efforts[{index}] must be a JSON object")
        result.append({str(item_key): item_value for item_key, item_value in item.items()})
    return result


def _effort_kwargs(effort: Mapping[str, Any], *, default_project_id: str) -> dict[str, Any]:
    return {
        "name": _required_string(effort, "name"),
        "project_id": _optional_string(effort.get("project_id")) or default_project_id,
        "hypothesis_or_topic": _optional_string(
            effort.get("hypothesis_or_topic") or effort.get("topic")
        ),
        "effort_type": str(effort.get("effort_type") or effort.get("type") or "research"),
        "status": str(effort.get("status") or "active"),
        "recurrence_policy": _mapping(
            effort.get("recurrence_policy"),
            field="effort.recurrence_policy",
        ),
        "next_wake_at": _optional_string(effort.get("next_wake_at")),
        "latest_run_id": _optional_string(effort.get("latest_run_id")),
        "latest_report_id": _optional_string(effort.get("latest_report_id")),
        "latest_work_product_id": _optional_string(effort.get("latest_work_product_id")),
        "decision_needed": bool(effort.get("decision_needed") or False),
        "decision_note": _optional_string(effort.get("decision_note")),
        "budget_policy": _mapping(
            effort.get("budget_policy"),
            field="effort.budget_policy",
        ),
        "publication_policy": _mapping(
            effort.get("publication_policy"),
            field="effort.publication_policy",
        ),
        "authorization_policy": _mapping(
            effort.get("authorization_policy"),
            field="effort.authorization_policy",
        ),
        "actor_notes": _mapping(effort.get("actor_notes"), field="effort.actor_notes"),
        "metadata": _mapping(effort.get("metadata"), field="effort.metadata"),
    }


def _wake_due_kwargs(plan: Mapping[str, Any], *, launch: bool) -> dict[str, Any]:
    wake_due = _mapping(plan.get("wake_due"), field="wake_due")
    dry_run = wake_due.get("dry_run")
    return {
        "launch_request": _mapping(
            wake_due.get("launch_request"),
            field="wake_due.launch_request",
        )
        or None,
        "limit": int(wake_due.get("limit") or 10),
        "allow_overlap": bool(wake_due.get("allow_overlap") or False),
        "continue_on_error": bool(wake_due.get("continue_on_error", True)),
        "dry_run": bool(dry_run) if dry_run is not None else not launch,
    }


def execute_factory_standup(
    plan: Mapping[str, Any],
    *,
    client: ManagedResearchClient,
    dry_run: bool = False,
    wake_due: bool = False,
    wake_due_launch: bool = False,
) -> dict[str, Any]:
    factory_payload = _factory_payload(plan)
    project = _mapping(plan.get("project"), field="project")
    create_project = _mapping(plan.get("create_project"), field="create_project")
    project_id = _optional_string(project.get("project_id"))
    effort_plans = _effort_plans(plan)
    link_payload = _project_link_payload(plan)

    if project_id is None and not create_project:
        raise ValueError("project.project_id or create_project is required")
    if create_project:
        SmrRunnableProjectRequest.from_wire(create_project)

    if dry_run:
        return {
            "dry_run": True,
            "factory": factory_payload,
            "project_id": project_id,
            "create_project": create_project or None,
            "project_link": link_payload,
            "efforts": effort_plans,
            "wake_due": _wake_due_kwargs(plan, launch=wake_due_launch)
            if wake_due or plan.get("wake_due")
            else None,
        }

    created_project: dict[str, Any] | None = None
    if project_id is None:
        created_project = client.create_runnable_project(create_project)
        project_id = _project_id_from_response(created_project)
    if project_id is None:
        raise RuntimeError("project_id resolution failed")

    factory = client.factories.create(factory_payload)
    link = client.factories.link_project(
        factory.factory_id,
        project_id,
        **link_payload,
    )
    efforts = [
        client.factories.create_effort(
            factory.factory_id,
            **_effort_kwargs(effort, default_project_id=project_id),
        )
        for effort in effort_plans
    ]
    wake_result = None
    if wake_due or plan.get("wake_due"):
        wake_result = client.factories.wake_due(
            factory.factory_id,
            **_wake_due_kwargs(plan, launch=wake_due_launch),
        )

    return {
        "dry_run": False,
        "factory_id": factory.factory_id,
        "project_id": project_id,
        "effort_ids": [effort.effort_id for effort in efforts],
        "factory": _jsonable(factory),
        "created_project": _jsonable(created_project),
        "project_link": _jsonable(link),
        "efforts": _jsonable(efforts),
        "wake_due": _jsonable(wake_result),
        "status": _jsonable(client.factories.status(factory.factory_id)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synth-ai-research-factory-standup",
        description="Create, link, and seed a Research Factory from one JSON plan.",
    )
    parser.add_argument("--api-key", default=None, help="defaults to $SYNTH_API_KEY")
    parser.add_argument("--backend", default=None, help="defaults to $SYNTH_BACKEND_URL")
    parser.add_argument("--plan", required=True, help="JSON object or @path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wake-due",
        action="store_true",
        help="call wake-due after effort creation; defaults to wake dry-run",
    )
    parser.add_argument(
        "--wake-due-launch",
        action="store_true",
        help="allow wake-due to launch real runs instead of dry-run previews",
    )
    parser.add_argument("--output", default=None, help="optional proof JSON path")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = _json_object(args.plan, field="plan")
        client = ManagedResearchClient(api_key=args.api_key, backend_base=args.backend)
        proof = execute_factory_standup(
            plan,
            client=client,
            dry_run=args.dry_run,
            wake_due=args.wake_due,
            wake_due_launch=args.wake_due_launch,
        )
        rendered = json.dumps(proof, indent=args.indent, sort_keys=True)
        if args.output:
            Path(args.output).expanduser().write_text(f"{rendered}\n")
        print(rendered)
        return 0
    except Exception as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
