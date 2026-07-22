"""Dev-environment CLI commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import click

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base


def _resolve_backend_url(backend_url: str | None) -> str:
    return normalize_backend_base(
        backend_url or os.environ.get("SYNTH_BACKEND_URL") or BACKEND_URL_BASE
    )


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or get_api_key(required=False) or "").strip()
    if not resolved:
        raise click.ClickException("api_key is required (pass --api-key or set SYNTH_API_KEY)")
    return resolved


def _json_object(value: str | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} must be a JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise click.ClickException(f"{label} must be a JSON object.")
    return data


def _json_object_array(value: str | None, *, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    text = str(value or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} must be a JSON array: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise click.ClickException(f"{label} must be a JSON array of objects.")
    return [dict(item) for item in data]


def _json_file(path: str, *, label: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise click.ClickException(f"unable to read {label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise click.ClickException(f"{label} must be a JSON object.")
    return data


def _optional_json_object(value: str | None, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _json_object(value, label=label)


def _mapping_at(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, dict) else {}


def _list_at(payload: dict[str, Any], *path: str) -> list[object]:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return []
        current = current.get(key)
    return list(current) if isinstance(current, list) else []


def _text_at(payload: dict[str, Any], *path: str) -> str | None:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    text = str(current or "").strip()
    return text or None


def _positive_int(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, float):
        return value > 0
    try:
        return int(str(value or "0")) > 0
    except (TypeError, ValueError):
        return False


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item or "").strip())]


def _git_proofs_from_evidence(
    payload: dict[str, Any],
    *,
    expected_run_ids: set[str],
) -> list[dict[str, Any]]:
    proofs: list[dict[str, Any]] = []
    for item in _list_at(payload, "receipts", "items"):
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "") != "dev_environment_run_receipt":
            continue
        run_id = str(item.get("run_id") or "").strip()
        if expected_run_ids and run_id not in expected_run_ids:
            continue
        git_raw = item.get("git")
        git = dict(git_raw) if isinstance(git_raw, Mapping) else {}
        run_git_raw = git.get("run_git_context")
        run_git = dict(run_git_raw) if isinstance(run_git_raw, Mapping) else {}
        project_git_raw = git.get("project_git")
        project_git = dict(project_git_raw) if isinstance(project_git_raw, Mapping) else {}
        branch = str(run_git.get("branch") or project_git.get("default_branch") or "").strip()
        commit_sha = str(
            run_git.get("head_commit_sha") or project_git.get("commit_sha") or ""
        ).strip()
        source_refs = (
            dict(item.get("source_refs")) if isinstance(item.get("source_refs"), dict) else {}
        )
        if branch and commit_sha:
            proofs.append(
                {
                    "run_id": run_id,
                    "branch": branch,
                    "commit_sha": commit_sha,
                    "source": run_git.get("source") or "unknown",
                    "last_push_confirmed": run_git.get("last_push_confirmed"),
                    "project_git_status": source_refs.get("project_git_status"),
                    "run_git": source_refs.get("run_git"),
                }
            )
    return proofs


def _verify_cloud_s0_evidence_payload(
    payload: dict[str, Any],
    *,
    expected_dev_environment_id: str | None = None,
    expected_project_id: str | None = None,
    expected_run_ids: tuple[str, ...] = (),
    expected_host_kind: str | None = "daytona",
) -> dict[str, Any]:
    summary = _mapping_at(payload, "summary")
    run_binding_summary = _mapping_at(payload, "summary", "run_binding_summary")
    cloud_s0_proof = _mapping_at(payload, "summary", "cloud_s0_proof")
    cloud_s0_checks = _mapping_at(payload, "summary", "cloud_s0_proof", "checks")
    environment = _mapping_at(payload, "environment")
    bound_run_ids = _string_list(run_binding_summary.get("bound_run_ids"))
    expected_run_id_set = {
        run_id for item in expected_run_ids if (run_id := str(item or "").strip())
    }
    actual_dev_environment_id = (
        _text_at(payload, "dev_environment_id")
        or _text_at(summary, "dev_environment_id")
        or _text_at(environment, "dev_environment_id")
        or _text_at(environment, "environment_id")
    )
    actual_project_id = (
        _text_at(summary, "project_id")
        or _text_at(environment, "project_id")
        or _text_at(payload, "project_id")
    )
    actual_host_kind = _text_at(summary, "host_kind") or _text_at(environment, "host_kind")
    git_proofs = _git_proofs_from_evidence(
        payload,
        expected_run_ids=expected_run_id_set,
    )

    checks: dict[str, bool] = {}
    missing: list[str] = []

    def require(name: str, passed: bool) -> None:
        checks[name] = bool(passed)
        if not passed:
            missing.append(name)

    require("summary_present", bool(summary))
    require("run_binding_summary_present", bool(run_binding_summary))
    require("cloud_s0_proof_present", bool(cloud_s0_proof))
    require("bound_run_ids_present", bool(bound_run_ids))
    require("has_bound_run", run_binding_summary.get("has_bound_run") is True)
    require(
        "has_dev_slot_execution_binding",
        run_binding_summary.get("has_dev_slot_execution_binding") is True,
    )
    require(
        "has_daytona_binding",
        run_binding_summary.get("has_daytona_binding") is True,
    )
    require("receipt_ready", cloud_s0_proof.get("receipt_ready") is True)
    for key in (
        "environment_id_bound",
        "dev_slot_execution_bound",
        "daytona_bound",
        "work_product_present",
        "raw_trace_present",
        "cost_snapshot_present",
        "usage_snapshot_present",
    ):
        require(f"cloud_s0_checks.{key}", cloud_s0_checks.get(key) is True)
    require(
        "work_product_count_positive",
        _positive_int(cloud_s0_proof.get("work_product_count")),
    )
    require("trace_count_positive", _positive_int(cloud_s0_proof.get("trace_count")))
    require(
        "cost_summary_present",
        bool(_mapping_at(payload, "summary", "cloud_s0_proof", "cost_summary")),
    )
    require(
        "usage_summary_present",
        bool(_mapping_at(payload, "summary", "cloud_s0_proof", "usage_summary")),
    )
    require("git_branch_and_sha_present", bool(git_proofs))
    if expected_dev_environment_id:
        require(
            "expected_dev_environment_id",
            actual_dev_environment_id == expected_dev_environment_id,
        )
    if expected_project_id:
        require("expected_project_id", actual_project_id == expected_project_id)
    if expected_host_kind:
        require("expected_host_kind", actual_host_kind == expected_host_kind)
    if expected_run_id_set:
        require(
            "expected_run_ids_present",
            expected_run_id_set.issubset(set(bound_run_ids)),
        )

    return {
        "ok": not missing,
        "checks": checks,
        "missing": missing,
        "actual": {
            "dev_environment_id": actual_dev_environment_id,
            "project_id": actual_project_id,
            "host_kind": actual_host_kind,
            "bound_run_ids": bound_run_ids,
            "git_proofs": git_proofs,
            "work_product_count": cloud_s0_proof.get("work_product_count"),
            "trace_count": cloud_s0_proof.get("trace_count"),
        },
        "expected": {
            "dev_environment_id": expected_dev_environment_id,
            "project_id": expected_project_id,
            "host_kind": expected_host_kind,
            "run_ids": sorted(expected_run_id_set),
        },
    }


def _run_launch_kwargs(
    *,
    objective: str,
    intended_horizon_hours: str,
    providers: str,
    limit_json: str | None,
    worker_pool_id: str | None,
    runbook: str | None,
    runbook_preset: str | None,
    runbook_config_id: str | None,
    timebox_seconds: int | None,
    agent_profile: str | None,
    initial_runtime_messages: str,
    workflow: str | None,
    run_policy: str | None,
    kickoff_contract: str | None,
    resource_bindings: str | None,
    required_work_products: str,
    require_report: bool,
    ai_cache: str | None,
    primary_objective_id: str | None,
    primary_objective_kind: str | None,
    primary_parent_ref: str | None,
    primary_parent: str | None,
    effort_id: str | None,
    idempotency_key_run_create: str | None,
) -> dict[str, Any]:
    return {
        "objective": objective,
        "intended_horizon_hours": int(intended_horizon_hours),
        "providers": _json_object_array(providers, label="providers"),
        "limit": _optional_json_object(limit_json, label="limit"),
        "worker_pool_id": worker_pool_id,
        "runbook": runbook,
        "runbook_preset": runbook_preset,
        "runbook_config_id": runbook_config_id,
        "timebox_seconds": timebox_seconds,
        "agent_profile": agent_profile,
        "initial_runtime_messages": _json_object_array(
            initial_runtime_messages,
            label="initial_runtime_messages",
        ),
        "workflow": _optional_json_object(workflow, label="workflow"),
        "run_policy": _optional_json_object(run_policy, label="run_policy"),
        "kickoff_contract": _optional_json_object(
            kickoff_contract,
            label="kickoff_contract",
        ),
        "resource_bindings": _optional_json_object(
            resource_bindings,
            label="resource_bindings",
        ),
        "required_work_products": _json_object_array(
            required_work_products,
            label="required_work_products",
        ),
        "require_report": require_report,
        "ai_cache": _optional_json_object(ai_cache, label="ai_cache"),
        "primary_objective_id": primary_objective_id,
        "primary_objective_kind": primary_objective_kind,
        "primary_parent_ref": _optional_json_object(
            primary_parent_ref,
            label="primary_parent_ref",
        ),
        "primary_parent": _optional_json_object(primary_parent, label="primary_parent"),
        "effort_id": effort_id,
        "idempotency_key_run_create": idempotency_key_run_create,
    }


def _dev_environment_run_options(fn):
    fn = click.option("--api-key", envvar="SYNTH_API_KEY")(fn)
    fn = click.option("--backend-url", envvar="SYNTH_BACKEND_URL")(fn)
    fn = click.option("--idempotency-key-run-create", default=None)(fn)
    fn = click.option("--effort-id", default=None)(fn)
    fn = click.option("--primary-parent", default=None)(fn)
    fn = click.option("--primary-parent-ref", default=None)(fn)
    fn = click.option("--primary-objective-kind", default=None)(fn)
    fn = click.option("--primary-objective-id", default=None)(fn)
    fn = click.option("--ai-cache", default=None)(fn)
    fn = click.option("--require-report/--no-require-report", default=True, show_default=True)(fn)
    fn = click.option("--required-work-products", default="[]", show_default=True)(fn)
    fn = click.option("--resource-bindings", default=None)(fn)
    fn = click.option("--kickoff-contract", default=None)(fn)
    fn = click.option("--run-policy", default=None)(fn)
    fn = click.option("--workflow", default=None)(fn)
    fn = click.option("--initial-runtime-messages", default="[]", show_default=True)(fn)
    fn = click.option("--agent-profile", default=None)(fn)
    fn = click.option("--timebox-seconds", default=None, type=int)(fn)
    fn = click.option("--runbook-config-id", default=None)(fn)
    fn = click.option("--runbook-preset", default=None)(fn)
    fn = click.option("--runbook", default=None)(fn)
    fn = click.option("--worker-pool-id", default=None)(fn)
    fn = click.option("--limit-json", default=None)(fn)
    fn = click.option("--providers", default="[]", show_default=True)(fn)
    fn = click.option(
        "--intended-horizon-hours",
        required=True,
        type=click.Choice(["1", "4", "8", "24", "168"]),
    )(fn)
    fn = click.option("--objective", required=True)(fn)
    fn = click.option("--project-id", required=True)(fn)
    return click.argument("dev_environment_id")(fn)


def _client(api_key: str | None, backend_url: str | None):
    from synth_ai.core.research._legacy import ManagedResearchClient

    return ManagedResearchClient(
        api_key=_resolve_api_key(api_key),
        backend_base=_resolve_backend_url(backend_url),
    )


def _print(data: object) -> None:
    def normalize(value: object) -> object:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        return value

    click.echo(json.dumps(normalize(data), indent=2, sort_keys=True, default=str))


@click.group(name="dev-envs")
def dev_envs() -> None:
    """Manage live Managed Research dev environments."""


@dev_envs.command("list")
@click.option("--project-id", default=None)
@click.option("--limit", default=100, type=int)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def list_dev_envs(
    project_id: str | None,
    limit: int,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    client = _client(api_key, backend_url)
    _print(client.dev_environments.list(project_id=project_id, limit=limit))


@dev_envs.command("topologies")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def list_topologies(
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.topologies())


@dev_envs.command("topology")
@click.argument("topology_id")
@click.option("--version", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def get_topology(
    topology_id: str,
    version: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.topology(
            topology_id,
            version=version,
        )
    )


@dev_envs.command("seed-topology-manifest")
@click.option("--topology-id", default="synth-dev", show_default=True)
@click.option("--version", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def seed_topology_manifest(
    topology_id: str,
    version: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.seed_topology_environment(
            topology_id=topology_id,
            version=version,
        )
    )


@dev_envs.command("materialization-queue")
@click.option("--project-id", default=None)
@click.option("--host-kind", default=None)
@click.option("--backend-target", default=None)
@click.option("--worker-id", default=None)
@click.option("--include-leased", is_flag=True, default=False)
@click.option("--limit", default=100, type=int)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def materialization_queue(
    project_id: str | None,
    host_kind: str | None,
    backend_target: str | None,
    worker_id: str | None,
    include_leased: bool,
    limit: int,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.materialization_queue(
            project_id=project_id,
            host_kind=host_kind,
            backend_target=backend_target,
            worker_id=worker_id,
            include_leased=include_leased,
            limit=limit,
        )
    )


@dev_envs.command("create")
@click.option("--project-id", required=True)
@click.option("--name", required=True)
@click.option("--environment-name", required=True)
@click.option("--backend-target", default="dev", show_default=True)
@click.option("--topology-id", default="synth-dev", show_default=True)
@click.option("--topology-version", default=None)
@click.option("--environment-digest", default=None)
@click.option("--host-kind", default="daytona", show_default=True)
@click.option("--quota-class", default=None)
@click.option("--metadata", default="{}")
@click.option("--uptime-rate-microcents-per-hour", default=None, type=int)
@click.option("--billing-model-class", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def create_dev_env(
    project_id: str,
    name: str,
    environment_name: str,
    backend_target: str,
    topology_id: str,
    topology_version: str | None,
    environment_digest: str | None,
    host_kind: str,
    quota_class: str | None,
    metadata: str,
    uptime_rate_microcents_per_hour: int | None,
    billing_model_class: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    client = _client(api_key, backend_url)
    _print(
        client.dev_environments.create(
            project_id=project_id,
            name=name,
            environment_name=environment_name,
            backend_target=backend_target,
            topology_id=topology_id,
            topology_version=topology_version,
            environment_digest=environment_digest,
            host_kind=host_kind,
            quota_class=quota_class,
            metadata=_json_object(metadata, label="metadata"),
            uptime_rate_microcents_per_hour=uptime_rate_microcents_per_hour,
            billing_model_class=billing_model_class,
        )
    )


@dev_envs.command("create-from-topology")
@click.option("--project-id", required=True)
@click.option("--name", required=True)
@click.option("--backend-target", default="dev", show_default=True)
@click.option("--topology-id", default="synth-dev", show_default=True)
@click.option("--topology-version", default=None)
@click.option("--host-kind", default="daytona", show_default=True)
@click.option("--quota-class", default=None)
@click.option("--metadata", default="{}")
@click.option("--uptime-rate-microcents-per-hour", default=None, type=int)
@click.option("--billing-model-class", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def create_dev_env_from_topology(
    project_id: str,
    name: str,
    backend_target: str,
    topology_id: str,
    topology_version: str | None,
    host_kind: str,
    quota_class: str | None,
    metadata: str,
    uptime_rate_microcents_per_hour: int | None,
    billing_model_class: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    client = _client(api_key, backend_url)
    _print(
        client.dev_environments.create_from_topology(
            project_id=project_id,
            name=name,
            backend_target=backend_target,
            topology_id=topology_id,
            topology_version=topology_version,
            host_kind=host_kind,
            quota_class=quota_class,
            metadata=_json_object(metadata, label="metadata"),
            uptime_rate_microcents_per_hour=uptime_rate_microcents_per_hour,
            billing_model_class=billing_model_class,
        )
    )


@dev_envs.command("get")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def get_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.get(dev_environment_id))


@dev_envs.command("billing-preflight")
@click.argument("dev_environment_id")
@click.option("--model-class", default="value", show_default=True)
@click.option("--estimated-customer-debit-microcents", default=0, type=int)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def billing_preflight(
    dev_environment_id: str,
    model_class: str,
    estimated_customer_debit_microcents: int,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).billing.preflight_dev_environment(
            dev_environment_id,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
        )
    )


@dev_envs.command("billing-drawdown")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def billing_drawdown(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).billing.dev_environment_drawdown(dev_environment_id))


@dev_envs.command("claim-materialization")
@click.argument("dev_environment_id")
@click.option("--worker-id", required=True)
@click.option("--lease-seconds", default=None, type=int)
@click.option("--metadata", default="{}")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def claim_materialization(
    dev_environment_id: str,
    worker_id: str,
    lease_seconds: int | None,
    metadata: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.claim_materialization(
            dev_environment_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            metadata=_json_object(metadata, label="metadata"),
        )
    )


@dev_envs.command("preflight")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def preflight_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.preflight(dev_environment_id))


@dev_envs.command("run-preflight")
@_dev_environment_run_options
def run_preflight_dev_env(
    dev_environment_id: str,
    project_id: str,
    objective: str,
    intended_horizon_hours: str,
    providers: str,
    limit_json: str | None,
    worker_pool_id: str | None,
    runbook: str | None,
    runbook_preset: str | None,
    runbook_config_id: str | None,
    timebox_seconds: int | None,
    agent_profile: str | None,
    initial_runtime_messages: str,
    workflow: str | None,
    run_policy: str | None,
    kickoff_contract: str | None,
    resource_bindings: str | None,
    required_work_products: str,
    require_report: bool,
    ai_cache: str | None,
    primary_objective_id: str | None,
    primary_objective_kind: str | None,
    primary_parent_ref: str | None,
    primary_parent: str | None,
    effort_id: str | None,
    idempotency_key_run_create: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    client = _client(api_key, backend_url)
    _print(
        client.runs.launch_preflight_in_dev_environment(
            project_id=project_id,
            dev_environment_id=dev_environment_id,
            **_run_launch_kwargs(
                objective=objective,
                intended_horizon_hours=intended_horizon_hours,
                providers=providers,
                limit_json=limit_json,
                worker_pool_id=worker_pool_id,
                runbook=runbook,
                runbook_preset=runbook_preset,
                runbook_config_id=runbook_config_id,
                timebox_seconds=timebox_seconds,
                agent_profile=agent_profile,
                initial_runtime_messages=initial_runtime_messages,
                workflow=workflow,
                run_policy=run_policy,
                kickoff_contract=kickoff_contract,
                resource_bindings=resource_bindings,
                required_work_products=required_work_products,
                require_report=require_report,
                ai_cache=ai_cache,
                primary_objective_id=primary_objective_id,
                primary_objective_kind=primary_objective_kind,
                primary_parent_ref=primary_parent_ref,
                primary_parent=primary_parent,
                effort_id=effort_id,
                idempotency_key_run_create=idempotency_key_run_create,
            ),
        )
    )


@dev_envs.command("start-run")
@_dev_environment_run_options
def start_run_dev_env(
    dev_environment_id: str,
    project_id: str,
    objective: str,
    intended_horizon_hours: str,
    providers: str,
    limit_json: str | None,
    worker_pool_id: str | None,
    runbook: str | None,
    runbook_preset: str | None,
    runbook_config_id: str | None,
    timebox_seconds: int | None,
    agent_profile: str | None,
    initial_runtime_messages: str,
    workflow: str | None,
    run_policy: str | None,
    kickoff_contract: str | None,
    resource_bindings: str | None,
    required_work_products: str,
    require_report: bool,
    ai_cache: str | None,
    primary_objective_id: str | None,
    primary_objective_kind: str | None,
    primary_parent_ref: str | None,
    primary_parent: str | None,
    effort_id: str | None,
    idempotency_key_run_create: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    client = _client(api_key, backend_url)
    _print(
        client.runs.start_run_in_dev_environment(
            project_id=project_id,
            dev_environment_id=dev_environment_id,
            **_run_launch_kwargs(
                objective=objective,
                intended_horizon_hours=intended_horizon_hours,
                providers=providers,
                limit_json=limit_json,
                worker_pool_id=worker_pool_id,
                runbook=runbook,
                runbook_preset=runbook_preset,
                runbook_config_id=runbook_config_id,
                timebox_seconds=timebox_seconds,
                agent_profile=agent_profile,
                initial_runtime_messages=initial_runtime_messages,
                workflow=workflow,
                run_policy=run_policy,
                kickoff_contract=kickoff_contract,
                resource_bindings=resource_bindings,
                required_work_products=required_work_products,
                require_report=require_report,
                ai_cache=ai_cache,
                primary_objective_id=primary_objective_id,
                primary_objective_kind=primary_objective_kind,
                primary_parent_ref=primary_parent_ref,
                primary_parent=primary_parent,
                effort_id=effort_id,
                idempotency_key_run_create=idempotency_key_run_create,
            ),
        )
    )


def _metadata_option(fn):
    fn = click.option("--metadata", default="{}")(fn)
    fn = click.option("--api-key", envvar="SYNTH_API_KEY")(fn)
    fn = click.option("--backend-url", envvar="SYNTH_BACKEND_URL")(fn)
    return click.argument("dev_environment_id")(fn)


@dev_envs.command("deploy")
@_metadata_option
def deploy_dev_env(
    dev_environment_id: str,
    metadata: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.deploy(
            dev_environment_id,
            metadata=_json_object(metadata, label="metadata"),
        )
    )


@dev_envs.command("start")
@_metadata_option
def start_dev_env(
    dev_environment_id: str,
    metadata: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.start(
            dev_environment_id,
            metadata=_json_object(metadata, label="metadata"),
        )
    )


@dev_envs.command("snapshot")
@_metadata_option
def snapshot_dev_env(
    dev_environment_id: str,
    metadata: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.snapshot(
            dev_environment_id,
            metadata=_json_object(metadata, label="metadata"),
        )
    )


@dev_envs.command("materialize")
@click.argument("dev_environment_id")
@click.option(
    "--result",
    default="succeeded",
    type=click.Choice(["succeeded", "failed", "degraded"]),
    show_default=True,
)
@click.option(
    "--lifecycle-state",
    default=None,
    type=click.Choice(["deployed", "running", "stopped"]),
)
@click.option("--service-summary", default=None)
@click.option("--log-entries", default="[]")
@click.option("--receipt-refs", default="[]")
@click.option("--metadata", default="{}")
@click.option("--error", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def materialize_dev_env(
    dev_environment_id: str,
    result: str,
    lifecycle_state: str | None,
    service_summary: str | None,
    log_entries: str,
    receipt_refs: str,
    metadata: str,
    error: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.materialize(
            dev_environment_id,
            result=result,
            lifecycle_state=lifecycle_state,
            service_summary=_json_object(
                service_summary,
                label="service_summary",
            )
            if service_summary is not None
            else None,
            log_entries=_json_object_array(log_entries, label="log_entries"),
            receipt_refs=_json_object_array(receipt_refs, label="receipt_refs"),
            metadata=_json_object(metadata, label="metadata"),
            error=_json_object(error, label="error") if error is not None else None,
        )
    )


@dev_envs.command("stop")
@click.argument("dev_environment_id")
@click.option("--decision", default="retain", type=click.Choice(["retain", "destroy"]))
@click.option("--metadata", default="{}")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def stop_dev_env(
    dev_environment_id: str,
    decision: str,
    metadata: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.stop(
            dev_environment_id,
            decision=decision,
            metadata=_json_object(metadata, label="metadata"),
        )
    )


@dev_envs.command("destroy")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def destroy_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.destroy(dev_environment_id))


@dev_envs.command("services")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def services_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.services(dev_environment_id))


@dev_envs.command("attach")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def attach_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.attach(dev_environment_id))


@dev_envs.command("logs")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def logs_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.logs(dev_environment_id))


@dev_envs.command("runs")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def runs_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.runs(dev_environment_id))


@dev_envs.command("usage")
@click.argument("dev_environment_id")
@click.option("--limit", default=100, type=int)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def usage_dev_env(
    dev_environment_id: str,
    limit: int,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.usage(
            dev_environment_id,
            limit=limit,
        )
    )


@dev_envs.command("receipts")
@click.argument("dev_environment_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def receipts_dev_env(
    dev_environment_id: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(_client(api_key, backend_url).dev_environments.receipts(dev_environment_id))


@dev_envs.command("evidence")
@click.argument("dev_environment_id")
@click.option("--usage-limit", default=100, type=int)
@click.option("--skip-preflight", is_flag=True)
@click.option("--include-logs", is_flag=True)
@click.option("--skip-billing", is_flag=True)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def evidence_dev_env(
    dev_environment_id: str,
    usage_limit: int,
    skip_preflight: bool,
    include_logs: bool,
    skip_billing: bool,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.evidence(
            dev_environment_id,
            usage_limit=usage_limit,
            include_preflight=not skip_preflight,
            include_logs=include_logs,
            include_billing=not skip_billing,
        )
    )


@dev_envs.command("wait-ready")
@click.argument("dev_environment_id")
@click.option(
    "--lifecycle-state",
    "lifecycle_states",
    multiple=True,
    help="Accepted lifecycle state. Repeat to allow multiple states. Defaults to running.",
)
@click.option("--timeout", "timeout_seconds", default=1800.0, type=float, show_default=True)
@click.option(
    "--poll-interval",
    "poll_interval_seconds",
    default=10.0,
    type=float,
    show_default=True,
)
@click.option("--no-require-readiness", is_flag=True)
@click.option("--require-attachable", is_flag=True)
@click.option("--skip-preflight", is_flag=True)
@click.option("--skip-billing", is_flag=True)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL")
def wait_ready_dev_env(
    dev_environment_id: str,
    lifecycle_states: tuple[str, ...],
    timeout_seconds: float,
    poll_interval_seconds: float,
    no_require_readiness: bool,
    require_attachable: bool,
    skip_preflight: bool,
    skip_billing: bool,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _print(
        _client(api_key, backend_url).dev_environments.wait_ready(
            dev_environment_id,
            lifecycle_states=lifecycle_states or None,
            timeout=timeout_seconds,
            poll_interval=poll_interval_seconds,
            require_readiness=not no_require_readiness,
            require_attachable=require_attachable,
            include_preflight=not skip_preflight,
            include_billing=not skip_billing,
        )
    )


@dev_envs.command("verify-cloud-s0-evidence")
@click.argument("evidence_path", type=click.Path(exists=False, dir_okay=False))
@click.option("--expect-dev-environment-id", default=None)
@click.option("--expect-project-id", default=None)
@click.option(
    "--expect-run-id",
    "expect_run_ids",
    multiple=True,
    help="Run id that must appear in bound_run_ids. Repeat to require multiple.",
)
@click.option("--expect-host-kind", default="daytona", show_default=True)
@click.option(
    "--soft",
    is_flag=True,
    help="Print the verdict but exit 0 even when checks fail.",
)
def verify_cloud_s0_evidence(
    evidence_path: str,
    expect_dev_environment_id: str | None,
    expect_project_id: str | None,
    expect_run_ids: tuple[str, ...],
    expect_host_kind: str,
    soft: bool,
) -> None:
    """Verify an offline Cloud S0 evidence bundle against the proof bar.

    Reads the JSON produced by `synth-ai dev-envs evidence`, checks the
    Cloud S0 receipt spine (env binding, dev_slot_execution, Daytona,
    WorkProduct, raw trace, git branch/SHA, cost + usage snapshots), and
    exits nonzero on any missing check unless --soft is passed.
    """
    payload = _json_file(evidence_path, label="evidence file")
    verdict = _verify_cloud_s0_evidence_payload(
        payload,
        expected_dev_environment_id=(str(expect_dev_environment_id or "").strip() or None),
        expected_project_id=str(expect_project_id or "").strip() or None,
        expected_run_ids=expect_run_ids,
        expected_host_kind=str(expect_host_kind or "").strip() or None,
    )
    _print(verdict)
    if not verdict["ok"] and not soft:
        raise SystemExit(1)


__all__ = ["dev_envs"]
