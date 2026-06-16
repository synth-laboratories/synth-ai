"""Shell CLI over the Managed Research Factory SDK."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _emit(obj: Any) -> None:
    def _default(value: Any) -> Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.asdict(value)
        for attr in ("to_wire", "model_dump", "_asdict"):
            fn = getattr(value, attr, None)
            if callable(fn):
                return fn()
        if hasattr(value, "__dict__"):
            return {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return str(value)

    print(json.dumps(obj, indent=2, default=_default, sort_keys=True))


def _json_object(raw: str | None, *, field: str) -> dict[str, Any]:
    if raw is None:
        return {}
    value = raw.strip()
    if not value:
        return {}
    if value.startswith("@"):
        value = Path(value[1:]).read_text()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object or @path to one")
    return parsed


def _client(args: argparse.Namespace) -> ManagedResearchClient:
    return ManagedResearchClient(api_key=args.api_key, backend_base=args.backend)


def _cmd_factory(args: argparse.Namespace) -> int:
    api = _client(args).factories
    if args.factory_cmd == "list":
        _emit(api.list(include_archived=args.include_archived))
    elif args.factory_cmd == "get":
        _emit(api.get(args.factory_id))
    elif args.factory_cmd == "status":
        _emit(api.status(args.factory_id))
    elif args.factory_cmd == "workspace":
        _emit(api.workspace(args.factory_id, include_archived=args.include_archived))
    elif args.factory_cmd == "create":
        _emit(
            api.create(
                {
                    "name": args.name,
                    "kind": args.kind,
                    "description": args.description,
                    "budget_policy": _json_object(args.budget_policy, field="budget_policy"),
                    "cap_policy": _json_object(args.cap_policy, field="cap_policy"),
                    "publication_policy": _json_object(
                        args.publication_policy, field="publication_policy"
                    ),
                    "authorization_policy": _json_object(
                        args.authorization_policy, field="authorization_policy"
                    ),
                    "metadata": _json_object(args.metadata, field="metadata"),
                }
            )
        )
    elif args.factory_cmd == "archive":
        _emit(api.archive(args.factory_id))
    elif args.factory_cmd == "wake-due":
        _emit(
            api.wake_due(
                args.factory_id,
                dry_run=args.dry_run,
                limit=args.limit,
                allow_overlap=args.allow_overlap,
                continue_on_error=not args.stop_on_error,
                launch_request=_json_object(args.launch_request, field="launch_request"),
            )
        )
    else:
        raise SystemExit(2)
    return 0


def _cmd_project(args: argparse.Namespace) -> int:
    api = _client(args).factories
    if args.project_cmd == "list":
        _emit(api.list_projects(args.factory_id, include_archived=args.include_archived))
    elif args.project_cmd == "get":
        _emit(api.get_project(args.factory_id, args.project_id))
    elif args.project_cmd == "link":
        _emit(
            api.link_project(
                args.factory_id,
                args.project_id,
                role=args.role,
                status=args.status,
                display_name=args.display_name,
                description=args.description,
                workspace_policy=_json_object(args.workspace_policy, field="workspace_policy"),
                resource_bindings=_json_object(args.resource_bindings, field="resource_bindings"),
                feed_health=_json_object(args.feed_health, field="feed_health"),
                default_launch_profile=_json_object(
                    args.default_launch_profile, field="default_launch_profile"
                ),
                metadata=_json_object(args.metadata, field="metadata"),
            )
        )
    elif args.project_cmd == "patch":
        payload: dict[str, Any] = {}
        for key in ("role", "status", "display_name", "description"):
            value = getattr(args, key)
            if value is not None:
                payload[key] = value
        for key in (
            "workspace_policy",
            "resource_bindings",
            "feed_health",
            "default_launch_profile",
            "metadata",
        ):
            value = getattr(args, key)
            if value is not None:
                payload[key] = _json_object(value, field=key)
        _emit(api.patch_project(args.factory_id, args.project_id, payload))
    elif args.project_cmd == "archive":
        _emit(api.archive_project(args.factory_id, args.project_id))
    else:
        raise SystemExit(2)
    return 0


def _cmd_effort(args: argparse.Namespace) -> int:
    client = _client(args)
    api = client.efforts
    factories = client.factories
    if args.effort_cmd == "get":
        _emit(api.get(args.effort_id))
    elif args.effort_cmd == "create":
        _emit(
            factories.create_effort(
                args.factory_id,
                name=args.name,
                project_id=args.project,
                hypothesis_or_topic=args.topic,
                effort_type=args.type,
                recurrence_policy=_every_minutes(args.every_minutes),
                budget_policy=_json_object(args.budget_policy, field="budget_policy"),
                publication_policy=_json_object(args.publication_policy, field="publication_policy"),
                authorization_policy=_json_object(args.authorization_policy, field="authorization_policy"),
                actor_notes=_json_object(args.actor_notes, field="actor_notes"),
                metadata=_json_object(args.metadata, field="metadata"),
            )
        )
    elif args.effort_cmd == "schedule":
        _emit(
            api.schedule(
                args.effort_id,
                next_wake_at=args.next_wake_at,
                recurrence_policy=_every_minutes(args.every_minutes),
                launch_request=_json_object(args.launch_request, field="launch_request"),
            )
        )
    elif args.effort_cmd == "pause":
        _emit(api.pause(args.effort_id))
    elif args.effort_cmd == "resume":
        _emit(api.resume(args.effort_id))
    elif args.effort_cmd == "ready-for-review":
        _emit(api.mark_ready_for_review(args.effort_id, note=args.note))
    elif args.effort_cmd == "resolve-decision":
        _emit(api.resolve_decision(args.effort_id, note=args.note))
    else:
        raise SystemExit(2)
    return 0


def _every_minutes(value: int | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"every_minutes": value}


def _cmd_idea(args: argparse.Namespace) -> int:
    api = _client(args).factories
    if args.idea_cmd == "list":
        _emit(
            api.list_ideas(
                args.factory_id,
                status=args.status,
                source=args.source,
                include_archived=args.include_archived,
                limit=args.limit,
            )
        )
    elif args.idea_cmd == "get":
        _emit(api.get_idea(args.factory_id, args.idea_id))
    elif args.idea_cmd == "create":
        _emit(
            api.create_idea(
                args.factory_id,
                title=args.title,
                body=args.body,
                source=args.source,
                project_id=args.project,
                effort_id=args.effort,
                run_id=args.run,
                priority=args.priority,
                tags=tuple(args.tag or ()),
                promotion_target=_json_object(args.promotion_target, field="promotion_target"),
                metadata=_json_object(args.metadata, field="metadata"),
            )
        )
    elif args.idea_cmd == "promote":
        _emit(
            api.promote_idea(
                args.factory_id,
                args.idea_id,
                promotion_target=_json_object(args.promotion_target, field="promotion_target"),
            )
        )
    elif args.idea_cmd == "archive":
        _emit(api.archive_idea(args.factory_id, args.idea_id))
    else:
        raise SystemExit(2)
    return 0


def _actor_refs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "project_id": args.project,
        "effort_id": args.effort,
        "run_id": args.run,
        "report_id": args.report,
        "work_product_id": args.work_product,
        "payload": _json_object(args.payload, field="payload"),
        "metadata": _json_object(args.metadata, field="metadata"),
    }


def _cmd_actor_output(args: argparse.Namespace) -> int:
    api = _client(args).factories
    if args.actor_cmd == "list":
        _emit(
            api.list_actor_outputs(
                args.factory_id,
                actor_role=args.role,
                kind=args.kind,
                status=args.status,
                include_archived=args.include_archived,
                limit=args.limit,
            )
        )
    elif args.actor_cmd == "get":
        _emit(api.get_actor_output(args.factory_id, args.actor_output_id))
    elif args.actor_cmd == "create":
        _emit(
            api.create_actor_output(
                args.factory_id,
                actor_role=args.role,
                kind=args.kind,
                title=args.title,
                summary=args.summary,
                status=args.status,
                **_actor_refs(args),
            )
        )
    elif args.actor_cmd == "seraph-brief":
        _emit(
            api.record_seraph_brief(
                args.factory_id,
                title=args.title,
                summary=args.summary,
                **_actor_refs(args),
            )
        )
    elif args.actor_cmd == "gardener-digest":
        _emit(
            api.record_gardener_digest(
                args.factory_id,
                title=args.title,
                summary=args.summary,
                **_actor_refs(args),
            )
        )
    elif args.actor_cmd == "architect-feed-health":
        _emit(
            api.record_architect_feed_health(
                args.factory_id,
                title=args.title,
                summary=args.summary,
                **_actor_refs(args),
            )
        )
    elif args.actor_cmd == "patch":
        payload: dict[str, Any] = {}
        for key in ("actor_role", "kind", "title", "summary", "status"):
            value = getattr(args, key, None)
            if value is not None:
                payload[key] = value
        for key in ("project", "effort", "run", "report", "work_product"):
            value = getattr(args, key)
            if value is not None:
                payload[f"{key}_id"] = value
        for key in ("payload", "metadata"):
            value = getattr(args, key)
            if value is not None:
                payload[key] = _json_object(value, field=key)
        _emit(api.patch_actor_output(args.factory_id, args.actor_output_id, payload))
    else:
        raise SystemExit(2)
    return 0


def _add_auth_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", default=None, help="defaults to $SYNTH_API_KEY")
    parser.add_argument("--backend", default=None, help="defaults to $SYNTH_BACKEND_URL")


def _add_factory_create_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", default="internal")
    parser.add_argument("--description", default=None)
    parser.add_argument("--budget-policy", default=None)
    parser.add_argument("--cap-policy", default=None)
    parser.add_argument("--publication-policy", default=None)
    parser.add_argument("--authorization-policy", default=None)
    parser.add_argument("--metadata", default=None)


def _add_project_payload_flags(parser: argparse.ArgumentParser, *, link: bool) -> None:
    parser.add_argument("factory_id")
    parser.add_argument("project_id")
    parser.add_argument("--role", default="canonical" if link else None)
    parser.add_argument("--status", default="active" if link else None)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--workspace-policy", default=None)
    parser.add_argument("--resource-bindings", default=None)
    parser.add_argument("--feed-health", default=None)
    parser.add_argument("--default-launch-profile", default=None)
    parser.add_argument("--metadata", default=None)


def _add_effort_payload_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None, help="project_id; defaults to canonical")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--type", default="research")
    parser.add_argument("--every-minutes", type=int, default=None)
    parser.add_argument("--budget-policy", default=None)
    parser.add_argument("--publication-policy", default=None)
    parser.add_argument("--authorization-policy", default=None)
    parser.add_argument("--actor-notes", default=None)
    parser.add_argument("--metadata", default=None)


def _add_actor_ref_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--work-product", default=None)
    parser.add_argument("--payload", default=None, help="JSON object or @path")
    parser.add_argument("--metadata", default=None, help="JSON object or @path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synth-ai-managed-research",
        description="Drive the Managed Research Factory control plane from a shell.",
    )
    _add_auth_flags(parser)
    sub = parser.add_subparsers(dest="group", required=True)

    factory = sub.add_parser("factory", help="factory verbs")
    f = factory.add_subparsers(dest="factory_cmd", required=True)
    f_list = f.add_parser("list")
    f_list.add_argument("--include-archived", action="store_true")
    f.add_parser("get").add_argument("factory_id")
    f.add_parser("status").add_argument("factory_id")
    f_workspace = f.add_parser("workspace")
    f_workspace.add_argument("factory_id")
    f_workspace.add_argument("--include-archived", action="store_true")
    _add_factory_create_flags(f.add_parser("create"))
    f.add_parser("archive").add_argument("factory_id")
    f_wake = f.add_parser("wake-due")
    f_wake.add_argument("factory_id")
    f_wake.add_argument("--dry-run", action="store_true")
    f_wake.add_argument("--limit", type=int, default=10)
    f_wake.add_argument("--allow-overlap", action="store_true")
    f_wake.add_argument("--stop-on-error", action="store_true")
    f_wake.add_argument("--launch-request", default=None)

    project = sub.add_parser("project", help="factory project-link verbs")
    p = project.add_subparsers(dest="project_cmd", required=True)
    p_list = p.add_parser("list")
    p_list.add_argument("factory_id")
    p_list.add_argument("--include-archived", action="store_true")
    p_get = p.add_parser("get")
    p_get.add_argument("factory_id")
    p_get.add_argument("project_id")
    _add_project_payload_flags(p.add_parser("link"), link=True)
    _add_project_payload_flags(p.add_parser("patch"), link=False)
    p_archive = p.add_parser("archive")
    p_archive.add_argument("factory_id")
    p_archive.add_argument("project_id")

    effort = sub.add_parser("effort", help="effort verbs")
    e = effort.add_subparsers(dest="effort_cmd", required=True)
    e.add_parser("get").add_argument("effort_id")
    e_create = e.add_parser("create")
    e_create.add_argument("factory_id")
    e_create.add_argument("--name", required=True)
    _add_effort_payload_flags(e_create)
    e_schedule = e.add_parser("schedule")
    e_schedule.add_argument("effort_id")
    e_schedule.add_argument("--next-wake-at", required=True)
    e_schedule.add_argument("--every-minutes", type=int, default=None)
    e_schedule.add_argument("--launch-request", default=None)
    e.add_parser("pause").add_argument("effort_id")
    e.add_parser("resume").add_argument("effort_id")
    e_review = e.add_parser("ready-for-review")
    e_review.add_argument("effort_id")
    e_review.add_argument("--note", default=None)
    e_decision = e.add_parser("resolve-decision")
    e_decision.add_argument("effort_id")
    e_decision.add_argument("--note", default=None)

    idea = sub.add_parser("idea", help="garden idea verbs")
    i = idea.add_subparsers(dest="idea_cmd", required=True)
    i_list = i.add_parser("list")
    i_list.add_argument("factory_id")
    i_list.add_argument("--status", default=None)
    i_list.add_argument("--source", default=None)
    i_list.add_argument("--include-archived", action="store_true")
    i_list.add_argument("--limit", type=int, default=50)
    i_get = i.add_parser("get")
    i_get.add_argument("factory_id")
    i_get.add_argument("idea_id")
    i_create = i.add_parser("create")
    i_create.add_argument("factory_id")
    i_create.add_argument("--title", required=True)
    i_create.add_argument("--body", default=None)
    i_create.add_argument("--source", default="human")
    i_create.add_argument("--project", default=None)
    i_create.add_argument("--effort", default=None)
    i_create.add_argument("--run", default=None)
    i_create.add_argument("--priority", default=None)
    i_create.add_argument("--tag", action="append")
    i_create.add_argument("--promotion-target", default=None)
    i_create.add_argument("--metadata", default=None)
    i_promote = i.add_parser("promote")
    i_promote.add_argument("factory_id")
    i_promote.add_argument("idea_id")
    i_promote.add_argument("--promotion-target", default=None)
    i_archive = i.add_parser("archive")
    i_archive.add_argument("factory_id")
    i_archive.add_argument("idea_id")

    actor = sub.add_parser("actor-output", help="typed Factory actor outputs")
    a = actor.add_subparsers(dest="actor_cmd", required=True)
    a_list = a.add_parser("list")
    a_list.add_argument("factory_id")
    a_list.add_argument("--role", default=None)
    a_list.add_argument("--kind", default=None)
    a_list.add_argument("--status", default=None)
    a_list.add_argument("--include-archived", action="store_true")
    a_list.add_argument("--limit", type=int, default=50)
    a_get = a.add_parser("get")
    a_get.add_argument("factory_id")
    a_get.add_argument("actor_output_id")
    a_create = a.add_parser("create")
    a_create.add_argument("factory_id")
    a_create.add_argument("--role", required=True)
    a_create.add_argument("--kind", required=True)
    a_create.add_argument("--title", required=True)
    a_create.add_argument("--summary", default=None)
    a_create.add_argument("--status", default="draft")
    _add_actor_ref_flags(a_create)
    for name in ("seraph-brief", "gardener-digest", "architect-feed-health"):
        cmd = a.add_parser(name)
        cmd.add_argument("factory_id")
        cmd.add_argument("--title", required=True)
        cmd.add_argument("--summary", default=None)
        _add_actor_ref_flags(cmd)
    a_patch = a.add_parser("patch")
    a_patch.add_argument("factory_id")
    a_patch.add_argument("actor_output_id")
    a_patch.add_argument("--actor-role", default=None)
    a_patch.add_argument("--kind", default=None)
    a_patch.add_argument("--title", default=None)
    a_patch.add_argument("--summary", default=None)
    a_patch.add_argument("--status", default=None)
    _add_actor_ref_flags(a_patch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.group == "factory":
            return _cmd_factory(args)
        if args.group == "project":
            return _cmd_project(args)
        if args.group == "effort":
            return _cmd_effort(args)
        if args.group == "idea":
            return _cmd_idea(args)
        if args.group == "actor-output":
            return _cmd_actor_output(args)
    except Exception as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
