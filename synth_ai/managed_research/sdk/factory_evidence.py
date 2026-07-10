"""Canonical Factory evidence SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.factory_evidence import (
    ActorContainerRunBinding,
    AppliedBudgetOwnerReceipt,
    AppliedExperimentRegistrationReceipt,
    AppliedSynthWikiReceipt,
    ConfirmedProjectGitPushReceipt,
    FactoryEvidencePacket,
    FactoryEvidenceReadSet,
    FactoryEvidenceTier,
    RunCostIdentity,
    RuntimeImageIdentity,
    SourceIdentity,
    assemble_factory_evidence_packet,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class FactoryEvidenceAPI(_ClientNamespace):
    """Read and assemble Factory evidence exclusively through typed owner APIs."""

    def read_cycle(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
    ) -> FactoryEvidenceReadSet:
        factory_status = self._client.factories.status(factory_id)
        project = self._client.projects.get(project_id)
        effort = self._client.efforts.get(effort_id)
        run = self._client.runs.get(run_id, project_id=project_id)
        authority_readouts = self._client.runs.get_authority_readouts(
            run_id,
            project_id=project_id,
            include_runtime_authority=True,
        )
        work_products = tuple(
            self._client.work_products.list_artifact_backed_for_run(project_id, run_id)
        )
        cost_identity = RunCostIdentity.from_summary(self._client.runs.cost_summary(run_id))
        operator_evidence = self._client.runs.get_operator_evidence(
            project_id,
            run_id,
        )
        return FactoryEvidenceReadSet(
            factory_status=factory_status,
            project=project,
            effort=effort,
            run=run,
            authority_readouts=authority_readouts,
            work_products=work_products,
            cost_identity=cost_identity,
            operator_evidence=operator_evidence,
        )

    def assemble_cycle(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        evidence_tier: FactoryEvidenceTier | str,
        source_identity: SourceIdentity | Mapping[str, Any],
        runtime_image_identity: RuntimeImageIdentity | Mapping[str, Any],
        runtime_contract_evidence: ActorContainerRunBinding | Mapping[str, Any],
        experiment_registration: AppliedExperimentRegistrationReceipt | Mapping[str, Any],
        synth_wiki_receipt: AppliedSynthWikiReceipt | Mapping[str, Any],
        git_push_receipt: ConfirmedProjectGitPushReceipt | Mapping[str, Any],
    ) -> FactoryEvidencePacket:
        """Fetch typed owner reads and assemble one fail-closed evidence packet."""

        reads = self.read_cycle(
            factory_id=factory_id,
            project_id=project_id,
            effort_id=effort_id,
            run_id=run_id,
        )
        typed_source_identity = (
            source_identity
            if isinstance(source_identity, SourceIdentity)
            else SourceIdentity.from_wire(source_identity)
        )
        typed_runtime_image = (
            runtime_image_identity
            if isinstance(runtime_image_identity, RuntimeImageIdentity)
            else RuntimeImageIdentity.from_wire(runtime_image_identity)
        )
        typed_actor_container_binding = (
            runtime_contract_evidence
            if isinstance(runtime_contract_evidence, ActorContainerRunBinding)
            else ActorContainerRunBinding.from_wire(runtime_contract_evidence)
        )
        typed_experiment = (
            experiment_registration
            if isinstance(experiment_registration, AppliedExperimentRegistrationReceipt)
            else AppliedExperimentRegistrationReceipt.from_wire(experiment_registration)
        )
        typed_wiki = (
            synth_wiki_receipt
            if isinstance(synth_wiki_receipt, AppliedSynthWikiReceipt)
            else AppliedSynthWikiReceipt.from_wire(synth_wiki_receipt)
        )
        typed_git_push = (
            git_push_receipt
            if isinstance(git_push_receipt, ConfirmedProjectGitPushReceipt)
            else ConfirmedProjectGitPushReceipt.from_wire(git_push_receipt)
        )
        budget_owner_receipt = AppliedBudgetOwnerReceipt.from_owner_reads(
            reads.factory_status.factory,
            reads.project,
        )
        return assemble_factory_evidence_packet(
            evidence_tier=evidence_tier,
            factory=reads.factory_status.factory,
            project=reads.project,
            effort=reads.effort,
            run=reads.run,
            authority_readouts=reads.authority_readouts,
            factory_runtime=reads.factory_status.typed_runtime,
            work_products=reads.work_products,
            source_identity=typed_source_identity,
            runtime_image_identity=typed_runtime_image,
            actor_container_binding=typed_actor_container_binding,
            cost_identity=reads.cost_identity,
            experiment_registration=typed_experiment,
            synth_wiki_receipt=typed_wiki,
            git_push_receipt=typed_git_push,
            budget_owner_receipt=budget_owner_receipt,
            operator_evidence=reads.operator_evidence,
        )


__all__ = ["FactoryEvidenceAPI"]
