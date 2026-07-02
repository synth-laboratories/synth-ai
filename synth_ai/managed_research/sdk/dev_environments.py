"""Live dev-environment namespace."""

from __future__ import annotations

import time
from typing import Any, List, Mapping

from synth_ai.managed_research.models.billing import (
    SmrBillingDrawdown,
    SmrBillingPreflight,
)
from synth_ai.managed_research.models.dev_environment_evidence import (
    DevEnvironmentEvidence,
)
from synth_ai.managed_research.models.types import (
    DevEnvironment,
    DevEnvironmentAttach,
    DevEnvironmentCollection,
    DevEnvironmentMaterializationQueue,
    DevEnvironmentMaterializationWorkItem,
    DevEnvironmentPreflight,
    DevEnvironmentTopology,
    DevEnvironmentUsage,
    Environment,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class DevEnvironmentsAPI(_ClientNamespace):
    DEFAULT_READY_LIFECYCLE_STATES = ("running",)

    @staticmethod
    def _summary_text(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @classmethod
    def _normalized_run_binding(
        cls,
        item: Mapping[str, object],
        *,
        source: str,
        fallback_dev_environment_id: str,
        fallback_project_id: str,
        fallback_host_kind: str,
        fallback_topology_id: str,
        fallback_topology_version: str | None,
    ) -> dict[str, object] | None:
        run_id = cls._summary_text(item.get("run_id"))
        if run_id is None:
            return None
        dev_environment_id = (
            cls._summary_text(item.get("dev_environment_id"))
            or cls._summary_text(item.get("environment_id"))
            or fallback_dev_environment_id
        )
        project_id = cls._summary_text(item.get("project_id")) or fallback_project_id
        host_kind = cls._summary_text(item.get("host_kind")) or fallback_host_kind
        topology_id = cls._summary_text(item.get("topology_id")) or fallback_topology_id
        topology_version = (
            cls._summary_text(item.get("topology_version")) or fallback_topology_version
        )
        launch_mode = cls._summary_text(item.get("launch_mode")) or "dev_slot_execution"
        source_refs = item.get("source_refs")
        return {
            "source": source,
            "dev_environment_id": dev_environment_id,
            "environment_id": dev_environment_id,
            "project_id": project_id,
            "run_id": run_id,
            "effort_id": cls._summary_text(item.get("effort_id")),
            "launch_mode": launch_mode,
            "host_kind": host_kind,
            "topology_id": topology_id,
            "topology_version": topology_version,
            "recorded_at": cls._summary_text(item.get("recorded_at")),
            "source_refs": dict(source_refs) if isinstance(source_refs, Mapping) else {},
        }

    @classmethod
    def _run_binding_summary(
        cls,
        *,
        environment: DevEnvironment,
        runs: DevEnvironmentCollection,
        receipts: DevEnvironmentCollection,
    ) -> dict[str, object]:
        bindings_by_run: dict[str, dict[str, object]] = {}
        source_counts: dict[str, int] = {}

        def add_binding(item: Mapping[str, object], *, source: str) -> None:
            binding = cls._normalized_run_binding(
                item,
                source=source,
                fallback_dev_environment_id=environment.dev_environment_id,
                fallback_project_id=environment.project_id,
                fallback_host_kind=environment.host_kind,
                fallback_topology_id=environment.topology_id,
                fallback_topology_version=environment.topology_version,
            )
            if binding is None:
                return
            run_id = str(binding["run_id"])
            source_counts[source] = source_counts.get(source, 0) + 1
            if run_id not in bindings_by_run:
                bindings_by_run[run_id] = binding
                return
            merged = dict(bindings_by_run[run_id])
            merged.update(
                {key: value for key, value in binding.items() if value is not None and value != ""}
            )
            existing_refs = merged.get("source_refs")
            new_refs = binding.get("source_refs")
            if isinstance(existing_refs, Mapping) and isinstance(new_refs, Mapping):
                merged["source_refs"] = {**dict(existing_refs), **dict(new_refs)}
            bindings_by_run[run_id] = merged

        for item in runs.items:
            add_binding(item, source="runs")
        for item in receipts.items:
            if str(item.get("kind") or "") == "dev_environment_run_receipt":
                add_binding(item, source="receipts")

        bindings = list(bindings_by_run.values())
        dev_environment_id = environment.dev_environment_id
        bound_bindings = [
            binding
            for binding in bindings
            if binding.get("dev_environment_id") == dev_environment_id
            or binding.get("environment_id") == dev_environment_id
        ]
        dev_slot_bindings = [
            binding
            for binding in bound_bindings
            if binding.get("launch_mode") == "dev_slot_execution"
        ]
        daytona_bindings = [
            binding for binding in bound_bindings if binding.get("host_kind") == "daytona"
        ]
        return {
            "run_bindings": bindings,
            "bound_run_ids": [str(binding["run_id"]) for binding in bound_bindings],
            "bound_run_count": len(bound_bindings),
            "dev_slot_execution_run_count": len(dev_slot_bindings),
            "daytona_run_count": len(daytona_bindings),
            "source_counts": source_counts,
            "latest_run_binding": bound_bindings[-1] if bound_bindings else None,
            "has_bound_run": bool(bound_bindings),
            "has_dev_slot_execution_binding": bool(dev_slot_bindings),
            "has_daytona_binding": bool(daytona_bindings),
        }

    @staticmethod
    def _int_summary_value(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            return int(str(value or "0"))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _cloud_s0_summary(
        cls,
        *,
        run_binding_summary: Mapping[str, object],
        environment: DevEnvironment,
        usage: DevEnvironmentUsage,
        receipts: DevEnvironmentCollection,
    ) -> dict[str, object]:
        receipt_summary = receipts.summary
        work_product_count = cls._int_summary_value(receipt_summary.get("work_product_count"))
        trace_count = cls._int_summary_value(receipt_summary.get("trace_count"))
        usage_summary = usage.summary or receipts.usage.get("summary")
        has_usage_snapshot = bool(usage_summary)
        has_cost_snapshot = bool(environment.cost_summary) or bool(
            receipts.environment.get("cost_summary")
        )
        checks = {
            "environment_id_bound": run_binding_summary.get("has_bound_run") is True,
            "dev_slot_execution_bound": (
                run_binding_summary.get("has_dev_slot_execution_binding") is True
            ),
            "daytona_bound": run_binding_summary.get("has_daytona_binding") is True,
            "work_product_present": work_product_count > 0,
            "raw_trace_present": trace_count > 0,
            "cost_snapshot_present": has_cost_snapshot,
            "usage_snapshot_present": has_usage_snapshot,
        }
        return {
            "checks": checks,
            "receipt_ready": all(checks.values()),
            "work_product_count": work_product_count,
            "trace_count": trace_count,
            "cost_summary": (
                dict(environment.cost_summary)
                if environment.cost_summary
                else dict(receipts.environment.get("cost_summary", {}))
            ),
            "usage_summary": dict(usage_summary) if isinstance(usage_summary, Mapping) else {},
            "latest_run_binding": run_binding_summary.get("latest_run_binding"),
        }

    @staticmethod
    def _evidence_summary(
        *,
        environment: DevEnvironment,
        preflight: DevEnvironmentPreflight | None,
        services: DevEnvironmentCollection,
        attach: DevEnvironmentAttach,
        runs: DevEnvironmentCollection,
        usage: DevEnvironmentUsage,
        receipts: DevEnvironmentCollection,
        billing_preflight: SmrBillingPreflight | None,
        billing_drawdown: SmrBillingDrawdown | None,
    ) -> dict[str, object]:
        readiness = environment.service_summary.get("readiness")
        if not isinstance(readiness, Mapping):
            readiness = services.summary.get("readiness")
        usage_summary = usage.summary
        receipt_summary = receipts.summary
        summary: dict[str, object] = {
            "dev_environment_id": environment.dev_environment_id,
            "project_id": environment.project_id,
            "backend_target": environment.backend_target,
            "host_kind": environment.host_kind,
            "topology_id": environment.topology_id,
            "lifecycle_state": environment.lifecycle_state,
            "readiness": dict(readiness) if isinstance(readiness, Mapping) else {},
            "preflight_ok": preflight.preflight_ok if preflight is not None else None,
            "attachable": attach.attachable,
            "operator_next_action": attach.operator_next_action,
            "service_count": len(services.items),
            "run_count": len(runs.items),
            "receipt_count": len(receipts.items),
            "usage_event_count": usage_summary.get(
                "event_count",
                receipt_summary.get("usage_event_count", 0),
            ),
            "usage_billed_amount_cents": usage_summary.get(
                "billed_amount_cents",
                receipt_summary.get("usage_billed_amount_cents", 0),
            ),
            "usage_nominal_amount_cents": usage_summary.get(
                "nominal_amount_cents",
                receipt_summary.get("usage_nominal_amount_cents", 0),
            ),
            "billing_allowed": (
                billing_preflight.allowed if billing_preflight is not None else None
            ),
            "billing_blocked": (billing_drawdown.blocked if billing_drawdown is not None else None),
            "billing_total_customer_debit_microcents": (
                billing_drawdown.total_customer_debit_microcents
                if billing_drawdown is not None
                else None
            ),
        }
        run_binding_summary = DevEnvironmentsAPI._run_binding_summary(
            environment=environment,
            runs=runs,
            receipts=receipts,
        )
        summary["run_binding_summary"] = run_binding_summary
        summary["cloud_s0_proof"] = DevEnvironmentsAPI._cloud_s0_summary(
            run_binding_summary=run_binding_summary,
            environment=environment,
            usage=usage,
            receipts=receipts,
        )
        return summary

    def topologies(self) -> List[DevEnvironmentTopology]:
        return [
            DevEnvironmentTopology.from_wire(item)
            for item in self._client.list_dev_environment_topologies()
        ]

    def topology(
        self,
        topology_id: str,
        *,
        version: str | None = None,
    ) -> DevEnvironmentTopology:
        return DevEnvironmentTopology.from_wire(
            self._client.get_dev_environment_topology(
                topology_id=topology_id,
                version=version,
            )
        )

    def seed_topology_environment(
        self,
        *,
        topology_id: str = "synth-dev",
        version: str | None = None,
    ) -> Environment:
        topology = self.topology(topology_id, version=version)
        catalog_manifest = topology.metadata.get("catalog_manifest")
        if not isinstance(catalog_manifest, Mapping):
            raise ValueError("topology metadata.catalog_manifest is required")
        manifest = catalog_manifest.get("template")
        if not isinstance(manifest, Mapping):
            raise ValueError("topology metadata.catalog_manifest.template is required")
        return self._client.environments.create(manifest=manifest)

    def list(
        self,
        *,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> List[DevEnvironment]:
        return [
            DevEnvironment.from_wire(item)
            for item in self._client.list_dev_environments(
                project_id=project_id,
                limit=limit,
            )
        ]

    def materialization_queue(
        self,
        *,
        project_id: str | None = None,
        host_kind: str | None = None,
        backend_target: str | None = None,
        worker_id: str | None = None,
        include_leased: bool | None = None,
        limit: int | None = None,
    ) -> DevEnvironmentMaterializationQueue:
        return DevEnvironmentMaterializationQueue.from_wire(
            self._client.list_dev_environment_materialization_queue(
                project_id=project_id,
                host_kind=host_kind,
                backend_target=backend_target,
                worker_id=worker_id,
                include_leased=include_leased,
                limit=limit,
            )
        )

    def create(
        self,
        *,
        project_id: str,
        name: str,
        environment_name: str,
        backend_target: str = "dev",
        topology_id: str = "synth-dev",
        topology_version: str | None = None,
        environment_digest: str | None = None,
        host_kind: str = "daytona",
        quota_class: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        uptime_rate_microcents_per_hour: int | None = None,
        billing_model_class: str | None = None,
    ) -> DevEnvironment:
        metadata_payload = self._create_metadata_with_billing(
            metadata=metadata,
            uptime_rate_microcents_per_hour=uptime_rate_microcents_per_hour,
            billing_model_class=billing_model_class,
        )
        return DevEnvironment.from_wire(
            self._client.create_dev_environment(
                project_id=project_id,
                name=name,
                environment_name=environment_name,
                backend_target=backend_target,
                topology_id=topology_id,
                topology_version=topology_version,
                environment_digest=environment_digest,
                host_kind=host_kind,
                quota_class=quota_class,
                metadata=metadata_payload,
            )
        )

    def create_from_topology(
        self,
        *,
        project_id: str,
        name: str,
        backend_target: str = "dev",
        topology_id: str = "synth-dev",
        topology_version: str | None = None,
        host_kind: str = "daytona",
        quota_class: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        uptime_rate_microcents_per_hour: int | None = None,
        billing_model_class: str | None = None,
    ) -> DevEnvironment:
        environment = self.seed_topology_environment(
            topology_id=topology_id,
            version=topology_version,
        )
        return self.create(
            project_id=project_id,
            name=name,
            environment_name=environment.name,
            backend_target=backend_target,
            topology_id=topology_id,
            topology_version=topology_version,
            environment_digest=environment.digest,
            host_kind=host_kind,
            quota_class=quota_class,
            metadata=metadata,
            uptime_rate_microcents_per_hour=uptime_rate_microcents_per_hour,
            billing_model_class=billing_model_class,
        )

    @staticmethod
    def _create_metadata_with_billing(
        *,
        metadata: Mapping[str, Any] | None,
        uptime_rate_microcents_per_hour: int | None,
        billing_model_class: str | None,
    ) -> dict[str, Any] | None:
        if (
            metadata is None
            and uptime_rate_microcents_per_hour is None
            and billing_model_class is None
        ):
            return None
        payload = dict(metadata or {})
        billing_value = payload.get("billing")
        if billing_value is not None and not isinstance(billing_value, Mapping):
            raise ValueError("metadata.billing must be an object when provided")
        billing = dict(billing_value or {})
        if uptime_rate_microcents_per_hour is not None:
            if uptime_rate_microcents_per_hour < 0:
                raise ValueError("uptime_rate_microcents_per_hour must be non-negative")
            billing["uptime_rate_microcents_per_hour"] = int(uptime_rate_microcents_per_hour)
        if billing_model_class is not None:
            model_class = str(billing_model_class or "").strip().lower()
            if model_class not in {"value", "premium"}:
                raise ValueError("billing_model_class must be 'value' or 'premium'")
            billing["model_class"] = model_class
        if billing:
            payload["billing"] = billing
        return payload

    def get(self, dev_environment_id: str) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.get_dev_environment(dev_environment_id=dev_environment_id)
        )

    def claim_materialization(
        self,
        dev_environment_id: str,
        *,
        worker_id: str,
        lease_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DevEnvironmentMaterializationWorkItem:
        return DevEnvironmentMaterializationWorkItem.from_wire(
            self._client.claim_dev_environment_materialization(
                dev_environment_id=dev_environment_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                metadata=metadata,
            )
        )

    def preflight(self, dev_environment_id: str) -> DevEnvironmentPreflight:
        return DevEnvironmentPreflight.from_wire(
            self._client.preflight_dev_environment(
                dev_environment_id=dev_environment_id,
            )
        )

    def deploy(
        self,
        dev_environment_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.deploy_dev_environment(
                dev_environment_id=dev_environment_id,
                metadata=metadata,
            )
        )

    def start(
        self,
        dev_environment_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.start_dev_environment(
                dev_environment_id=dev_environment_id,
                metadata=metadata,
            )
        )

    def stop(
        self,
        dev_environment_id: str,
        *,
        decision: str = "retain",
        metadata: Mapping[str, Any] | None = None,
    ) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.stop_dev_environment(
                dev_environment_id=dev_environment_id,
                decision=decision,
                metadata=metadata,
            )
        )

    def snapshot(
        self,
        dev_environment_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.snapshot_dev_environment(
                dev_environment_id=dev_environment_id,
                metadata=metadata,
            )
        )

    def materialize(
        self,
        dev_environment_id: str,
        *,
        result: str = "succeeded",
        lifecycle_state: str | None = None,
        service_summary: Mapping[str, Any] | None = None,
        log_entries: list[Mapping[str, Any]] | None = None,
        receipt_refs: list[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.report_dev_environment_materialization(
                dev_environment_id=dev_environment_id,
                result=result,
                lifecycle_state=lifecycle_state,
                service_summary=service_summary,
                log_entries=log_entries,
                receipt_refs=receipt_refs,
                metadata=metadata,
                error=error,
            )
        )

    def destroy(self, dev_environment_id: str) -> DevEnvironment:
        return DevEnvironment.from_wire(
            self._client.delete_dev_environment(dev_environment_id=dev_environment_id)
        )

    def services(self, dev_environment_id: str) -> DevEnvironmentCollection:
        return DevEnvironmentCollection.from_wire(
            self._client.get_dev_environment_services(
                dev_environment_id=dev_environment_id,
            ),
            key="services",
        )

    def attach(self, dev_environment_id: str) -> DevEnvironmentAttach:
        return DevEnvironmentAttach.from_wire(
            self._client.get_dev_environment_attach(
                dev_environment_id=dev_environment_id,
            )
        )

    def logs(self, dev_environment_id: str) -> DevEnvironmentCollection:
        return DevEnvironmentCollection.from_wire(
            self._client.get_dev_environment_logs(
                dev_environment_id=dev_environment_id,
            ),
            key="entries",
        )

    def runs(self, dev_environment_id: str) -> DevEnvironmentCollection:
        return DevEnvironmentCollection.from_wire(
            self._client.get_dev_environment_runs(
                dev_environment_id=dev_environment_id,
            ),
            key="runs",
        )

    def usage(
        self,
        dev_environment_id: str,
        *,
        limit: int | None = None,
    ) -> DevEnvironmentUsage:
        return DevEnvironmentUsage.from_wire(
            self._client.get_dev_environment_usage(
                dev_environment_id=dev_environment_id,
                limit=limit,
            )
        )

    def billing_preflight(
        self,
        dev_environment_id: str,
        *,
        model_class: str = "value",
        estimated_customer_debit_microcents: int = 0,
    ) -> SmrBillingPreflight:
        return self._client.billing.preflight_dev_environment(
            dev_environment_id,
            model_class=model_class,
            estimated_customer_debit_microcents=estimated_customer_debit_microcents,
        )

    def billing_drawdown(self, dev_environment_id: str) -> SmrBillingDrawdown:
        return self._client.billing.dev_environment_drawdown(dev_environment_id)

    def receipts(self, dev_environment_id: str) -> DevEnvironmentCollection:
        return DevEnvironmentCollection.from_wire(
            self._client.get_dev_environment_receipts(
                dev_environment_id=dev_environment_id,
            ),
            key="receipts",
        )

    def evidence(
        self,
        dev_environment_id: str,
        *,
        usage_limit: int | None = 100,
        include_preflight: bool = True,
        include_logs: bool = False,
        include_billing: bool = True,
    ) -> DevEnvironmentEvidence:
        environment = self.get(dev_environment_id)
        preflight = self.preflight(dev_environment_id) if include_preflight else None
        services = self.services(dev_environment_id)
        attach = self.attach(dev_environment_id)
        logs = self.logs(dev_environment_id) if include_logs else None
        runs = self.runs(dev_environment_id)
        usage = self.usage(dev_environment_id, limit=usage_limit)
        billing_preflight = self.billing_preflight(dev_environment_id) if include_billing else None
        billing_drawdown = self.billing_drawdown(dev_environment_id) if include_billing else None
        receipts = self.receipts(dev_environment_id)
        return DevEnvironmentEvidence(
            dev_environment_id=environment.dev_environment_id,
            environment=environment,
            preflight=preflight,
            services=services,
            attach=attach,
            logs=logs,
            runs=runs,
            usage=usage,
            billing_preflight=billing_preflight,
            billing_drawdown=billing_drawdown,
            receipts=receipts,
            summary=self._evidence_summary(
                environment=environment,
                preflight=preflight,
                services=services,
                attach=attach,
                runs=runs,
                usage=usage,
                receipts=receipts,
                billing_preflight=billing_preflight,
                billing_drawdown=billing_drawdown,
            ),
        )

    def wait_ready(
        self,
        dev_environment_id: str,
        *,
        lifecycle_states: tuple[str, ...] | list[str] | None = None,
        timeout: float | None = 1800.0,
        poll_interval: float = 10.0,
        require_readiness: bool = True,
        require_attachable: bool = False,
        include_preflight: bool = True,
        include_billing: bool = True,
    ) -> DevEnvironmentEvidence:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative when provided")
        targets = self._normalized_lifecycle_targets(lifecycle_states)
        deadline = time.monotonic() + timeout if timeout is not None else None
        last_evidence: DevEnvironmentEvidence | None = None
        while True:
            last_evidence = self.evidence(
                dev_environment_id,
                include_preflight=include_preflight,
                include_billing=include_billing,
            )
            if self._evidence_ready(
                last_evidence,
                lifecycle_states=targets,
                require_readiness=require_readiness,
                require_attachable=require_attachable,
            ):
                return last_evidence
            if deadline is not None and time.monotonic() >= deadline:
                reason = self._wait_ready_reason(
                    last_evidence,
                    lifecycle_states=targets,
                    require_readiness=require_readiness,
                    require_attachable=require_attachable,
                )
                raise TimeoutError(
                    f"DevEnvironment {dev_environment_id} was not ready within {timeout}s: {reason}"
                )
            time.sleep(poll_interval)

    @classmethod
    def _normalized_lifecycle_targets(
        cls,
        lifecycle_states: tuple[str, ...] | list[str] | None,
    ) -> tuple[str, ...]:
        raw_states = lifecycle_states or cls.DEFAULT_READY_LIFECYCLE_STATES
        states = tuple(state for state in (str(item or "").strip() for item in raw_states) if state)
        if not states:
            raise ValueError("at least one lifecycle state is required")
        return states

    @classmethod
    def _evidence_ready(
        cls,
        evidence: DevEnvironmentEvidence,
        *,
        lifecycle_states: tuple[str, ...],
        require_readiness: bool,
        require_attachable: bool,
    ) -> bool:
        if evidence.environment.lifecycle_state not in lifecycle_states:
            return False
        if require_readiness and not cls._readiness_ok(evidence):
            return False
        return not require_attachable or evidence.attach.attachable

    @staticmethod
    def _readiness_ok(evidence: DevEnvironmentEvidence) -> bool:
        readiness = evidence.summary.get("readiness")
        if not isinstance(readiness, Mapping):
            return False
        return readiness.get("required_ready") is True or readiness.get("ready") is True

    @classmethod
    def _wait_ready_reason(
        cls,
        evidence: DevEnvironmentEvidence,
        *,
        lifecycle_states: tuple[str, ...],
        require_readiness: bool,
        require_attachable: bool,
    ) -> str:
        summary = evidence.summary
        pieces = [
            f"lifecycle_state={evidence.environment.lifecycle_state!r}",
            f"target_lifecycle_states={list(lifecycle_states)!r}",
        ]
        if require_readiness:
            pieces.append(f"readiness={summary.get('readiness')!r}")
        if require_attachable:
            pieces.append(f"attachable={evidence.attach.attachable!r}")
        next_action = summary.get("operator_next_action")
        if next_action:
            pieces.append(f"operator_next_action={next_action!r}")
        return ", ".join(pieces)


__all__ = ["DevEnvironmentsAPI"]
