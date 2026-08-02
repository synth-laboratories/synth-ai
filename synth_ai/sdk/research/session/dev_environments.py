"""Live dev-environment namespace."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Mapping

from synth_ai.sdk.research.contracts.billing import (
    SmrBillingDrawdown,
    SmrBillingPreflight,
)
from synth_ai.sdk.research.contracts.dev_environment_evidence import (
    DevEnvironmentEvidence,
)
from synth_ai.sdk.research.contracts.types import (
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
from synth_ai.sdk.research.session._base import _ClientNamespace


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

    def _run_proofs(
        self,
        *,
        environment: DevEnvironment,
        run_binding_summary: Mapping[str, object],
    ) -> List[dict[str, object]]:
        proofs: List[dict[str, object]] = []
        for run_id in run_binding_summary.get("bound_run_ids") or ():
            run_id_text = str(run_id or "").strip()
            if not run_id_text:
                continue
            proof: dict[str, object] = {
                "project_id": environment.project_id,
                "run_id": run_id_text,
            }
            work_products = self._client.list_run_work_products(
                environment.project_id,
                run_id_text,
            )
            proof["work_products"] = work_products
            proof["work_product_count"] = len(work_products)
            proof["ready_work_product_count"] = sum(
                1
                for item in work_products
                if isinstance(item, Mapping)
                and str(item.get("status") or "").strip().lower() == "ready"
            )
            traces = self._client.get_project_run_traces(
                environment.project_id,
                run_id_text,
            )
            proof["trace_count"] = int(getattr(traces, "count", 0) or 0)
            proof["traces"] = [
                {
                    "trace_id": trace.trace_id,
                    "artifact_id": trace.artifact_id,
                    "event_count": trace.event_count,
                    "storage_uri": trace.artifact_uri,
                    "participant_session_id": trace.participant_session_id,
                    "participant_role": trace.participant_role,
                }
                for trace in getattr(traces, "traces", ()) or ()
            ]
            proofs.append(proof)
        return proofs

    @classmethod
    def _cloud_s0_summary(
        cls,
        *,
        run_binding_summary: Mapping[str, object],
        environment: DevEnvironment,
        usage: DevEnvironmentUsage,
        receipts: DevEnvironmentCollection,
        run_proofs: List[dict[str, object]],
    ) -> dict[str, object]:
        receipt_summary = receipts.summary
        hydrated_work_product_count = sum(
            cls._int_summary_value(proof.get("work_product_count")) for proof in run_proofs
        )
        hydrated_ready_work_product_count = sum(
            cls._int_summary_value(proof.get("ready_work_product_count")) for proof in run_proofs
        )
        hydrated_trace_count = sum(
            cls._int_summary_value(proof.get("trace_count")) for proof in run_proofs
        )
        work_product_count = max(
            cls._int_summary_value(receipt_summary.get("work_product_count")),
            hydrated_work_product_count,
        )
        ready_work_product_count = max(
            cls._int_summary_value(receipt_summary.get("ready_work_product_count")),
            hydrated_ready_work_product_count,
        )
        trace_count = max(
            cls._int_summary_value(receipt_summary.get("trace_count")),
            hydrated_trace_count,
        )
        usage_summary = usage.summary or receipts.usage.get("summary")
        has_usage_snapshot = bool(usage_summary)
        receipt_cost_summary = receipts.environment.get("cost_summary")
        has_cost_snapshot = bool(environment.cost_summary) or bool(receipt_cost_summary)
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
            "ready_work_product_count": ready_work_product_count,
            "trace_count": trace_count,
            "cost_summary": (
                dict(environment.cost_summary)
                if environment.cost_summary
                else dict(receipt_cost_summary)
                if isinstance(receipt_cost_summary, Mapping)
                else {}
            ),
            "usage_summary": dict(usage_summary) if isinstance(usage_summary, Mapping) else {},
            "latest_run_binding": run_binding_summary.get("latest_run_binding"),
            "run_proof_count": len(run_proofs),
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
        run_proofs: List[dict[str, object]],
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
            run_proofs=run_proofs,
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
        log_entries: List[Mapping[str, Any]] | None = None,
        receipt_refs: List[Mapping[str, Any]] | None = None,
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
        run_binding_summary = self._run_binding_summary(
            environment=environment,
            runs=runs,
            receipts=receipts,
        )
        run_proofs = self._run_proofs(
            environment=environment,
            run_binding_summary=run_binding_summary,
        )
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
            run_proofs=run_proofs,
            summary=self._evidence_summary(
                environment=environment,
                preflight=preflight,
                services=services,
                attach=attach,
                runs=runs,
                usage=usage,
                receipts=receipts,
                run_proofs=run_proofs,
                billing_preflight=billing_preflight,
                billing_drawdown=billing_drawdown,
            ),
        )

    def wait_ready(
        self,
        dev_environment_id: str,
        *,
        lifecycle_states: tuple[str, ...] | List[str] | None = None,
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
        lifecycle_states: tuple[str, ...] | List[str] | None,
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


# The Cloud S0 receipt shape is a backend contract, so reading it lives next to
# the namespace that fetches it.


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


@dataclass(frozen=True, slots=True)
class CloudS0GitProof:
    """One run receipt's git branch/SHA proof, extracted from the evidence payload."""

    run_id: str
    branch: str
    commit_sha: str
    source: str
    last_push_confirmed: object
    project_git_status: object
    run_git: object


@dataclass(frozen=True, slots=True)
class CloudS0EvidenceActuals:
    """What the evidence payload actually contained."""

    dev_environment_id: str | None
    project_id: str | None
    host_kind: str | None
    bound_run_ids: tuple[str, ...]
    git_proofs: tuple[CloudS0GitProof, ...]
    work_product_count: object
    trace_count: object


@dataclass(frozen=True, slots=True)
class CloudS0EvidenceExpectations:
    """What the caller asked the evidence to prove."""

    dev_environment_id: str | None
    project_id: str | None
    host_kind: str | None
    run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CloudS0EvidenceReport:
    """Typed verdict of :func:`verify_cloud_s0_evidence`.

    ``ok`` is True only when every required check passed; ``missing`` names the
    failed checks so operators can act without diffing ``checks`` by hand.
    """

    ok: bool
    checks: dict[str, bool]
    missing: tuple[str, ...]
    actual: CloudS0EvidenceActuals
    expected: CloudS0EvidenceExpectations


def _git_proofs_from_evidence(
    payload: dict[str, Any],
    *,
    expected_run_ids: set[str],
) -> tuple[CloudS0GitProof, ...]:
    proofs: list[CloudS0GitProof] = []
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
                CloudS0GitProof(
                    run_id=run_id,
                    branch=branch,
                    commit_sha=commit_sha,
                    source=str(run_git.get("source") or "unknown"),
                    last_push_confirmed=run_git.get("last_push_confirmed"),
                    project_git_status=source_refs.get("project_git_status"),
                    run_git=source_refs.get("run_git"),
                )
            )
    return tuple(proofs)


def verify_cloud_s0_evidence(
    payload: dict[str, Any],
    *,
    expected_dev_environment_id: str | None = None,
    expected_project_id: str | None = None,
    expected_run_ids: tuple[str, ...] = (),
    expected_host_kind: str | None = "daytona",
) -> CloudS0EvidenceReport:
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

    return CloudS0EvidenceReport(
        ok=not missing,
        checks=checks,
        missing=tuple(missing),
        actual=CloudS0EvidenceActuals(
            dev_environment_id=actual_dev_environment_id,
            project_id=actual_project_id,
            host_kind=actual_host_kind,
            bound_run_ids=tuple(bound_run_ids),
            git_proofs=git_proofs,
            work_product_count=cloud_s0_proof.get("work_product_count"),
            trace_count=cloud_s0_proof.get("trace_count"),
        ),
        expected=CloudS0EvidenceExpectations(
            dev_environment_id=expected_dev_environment_id,
            project_id=expected_project_id,
            host_kind=expected_host_kind,
            run_ids=tuple(sorted(expected_run_id_set)),
        ),
    )


__all__ = [
    "CloudS0EvidenceActuals",
    "CloudS0EvidenceExpectations",
    "CloudS0EvidenceReport",
    "CloudS0GitProof",
    "DevEnvironmentsAPI",
    "verify_cloud_s0_evidence",
]
