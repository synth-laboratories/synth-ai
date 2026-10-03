"""Prepared study/corpus declarations; validation confers no scientific authority.

See backend/notes/specifications/synth-index/study-corpus.md. The active v1
registry and public readers deliberately do not accept this prepared v2 profile.
"""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .contracts import ContributionReference, Identifier, IndexContract, require_unique
from .lifecycle import Digest, Title
from .public_corpus import CorpusPin


class ReleasedStudyAsset(IndexContract):
    """Exact released source/asset pin, requiring independent current authorization."""

    source: CorpusPin
    asset_id: Identifier
    asset_digest_sha256: Digest


class StudyTask(IndexContract):
    task_id: Identifier
    track: Literal["prompt_optimization", "harness_engineering", "sft_fbc"]
    task_card_sha256: Digest


class TaskApplicability(IndexContract):
    reference: ContributionReference
    task_id: Identifier


class StudyProtocol(IndexContract):
    release: ReleasedStudyAsset
    membership_sha256: Digest
    design_sha256: Digest


class MeasuredStudyResults(IndexContract):
    release: ReleasedStudyAsset
    membership_sha256: Digest
    design_sha256: Digest
    protocol_asset_sha256: Digest


class ProspectiveStudy(IndexContract):
    stage: Literal["prospective"]
    study_id: Identifier
    protocol: StudyProtocol


class MeasuredStudy(IndexContract):
    stage: Literal["measured"]
    study_id: Identifier
    protocol: StudyProtocol
    results: MeasuredStudyResults

    @model_validator(mode="after")
    def same_protocol_and_membership(self) -> Self:
        if self.results.protocol_asset_sha256 != self.protocol.release.asset_digest_sha256:
            raise ValueError("Results must bind the exact declared protocol asset")
        if self.results.membership_sha256 != self.protocol.membership_sha256:
            raise ValueError("Results must bind the same declared corpus membership")
        if self.results.design_sha256 != self.protocol.design_sha256:
            raise ValueError("Results must bind the same declared task/applicability design")
        if (
            self.results.release.source.reference == self.protocol.release.source.reference
            and self.results.release.asset_id == self.protocol.release.asset_id
        ):
            raise ValueError("A protocol declaration cannot also be measured results")
        return self


StudyDeclaration = Annotated[ProspectiveStudy | MeasuredStudy, Field(discriminator="stage")]


class RegisteredStudyCorpus(IndexContract):
    schema_version: Literal["synth.index.registered-corpus.v2"]
    corpus_id: Identifier
    title: Title
    source_receipt_sha256: Digest
    task_profile: Literal["synth.index.reb-three-track.v1"]
    members: tuple[CorpusPin, ...] = Field(min_length=1, max_length=256)
    tasks: tuple[StudyTask, ...] = Field(min_length=1, max_length=64)
    applicability: tuple[TaskApplicability, ...] = Field(min_length=1, max_length=2048)
    study: StudyDeclaration

    @model_validator(mode="after")
    def exact_study_bindings(self) -> Self:
        """Check consistency; see backend/notes/specifications/synth-index/study-corpus.md."""
        contribution_ids = tuple(pin.reference.contribution_id for pin in self.members)
        require_unique(contribution_ids, "corpus Contributions")
        task_ids = tuple(task.task_id for task in self.tasks)
        require_unique(task_ids, "study tasks")
        member_references = {
            (pin.reference.contribution_id, pin.reference.revision_id) for pin in self.members
        }
        bindings = tuple(
            (item.reference.contribution_id, item.reference.revision_id, item.task_id)
            for item in self.applicability
        )
        require_unique(bindings, "task applicability")
        covered_members, covered_tasks = set(), set()
        for contribution_id, revision_id, task_id in bindings:
            reference = (contribution_id, revision_id)
            if reference not in member_references or task_id not in task_ids:
                raise ValueError("Applicability must name an exact declared member and task")
            covered_members.add(reference)
            covered_tasks.add(task_id)
        if covered_members != member_references or covered_tasks != set(task_ids):
            raise ValueError("Every declared member and task requires explicit applicability")
        if self.study.protocol.membership_sha256 != study_membership_digest(self.members):
            raise ValueError("Protocol must bind the exact ordered member pins")
        if self.study.protocol.design_sha256 != study_design_digest(self.tasks, self.applicability):
            raise ValueError("Protocol must bind the exact ordered tasks and applicability")
        sources = [self.study.protocol.release.source]
        if isinstance(self.study, MeasuredStudy):
            sources.append(self.study.results.release.source)
        declared_digests = {
            (pin.reference.contribution_id, pin.reference.revision_id): pin.manifest_digest
            for pin in self.members
        }
        for pin in sources:
            reference = (pin.reference.contribution_id, pin.reference.revision_id)
            known = declared_digests.setdefault(reference, pin.manifest_digest)
            if known != pin.manifest_digest:
                raise ValueError(
                    "One exact source revision cannot have conflicting descriptor digests"
                )
        study_corpus_bytes(self)
        return self


def study_membership_digest(members: tuple[CorpusPin, ...]) -> str:
    """Bind pins; see backend/notes/specifications/synth-index/study-corpus.md."""
    value = [member.model_dump(mode="json") for member in members]
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def study_design_digest(
    tasks: tuple[StudyTask, ...], applicability: tuple[TaskApplicability, ...]
) -> str:
    """Bind design; see backend/notes/specifications/synth-index/study-corpus.md."""
    value = {
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "applicability": [item.model_dump(mode="json") for item in applicability],
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def study_corpus_bytes(corpus: RegisteredStudyCorpus) -> bytes:
    """Encode v2; see backend/notes/specifications/synth-index/study-corpus.md."""
    encoded = json.dumps(
        corpus.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > 262_144:
        raise ValueError("Study corpus exceeds the existing 262144-byte registry bound")
    return encoded


def validate_reb_launch_counts(corpus: RegisteredStudyCorpus) -> None:
    """Check counts; see backend/notes/specifications/synth-index/study-corpus.md."""
    if len(corpus.tasks) != 20 or len(corpus.members) != 100:
        raise ValueError("REB launch requires 20 tasks and 100 distinct Contributions")
