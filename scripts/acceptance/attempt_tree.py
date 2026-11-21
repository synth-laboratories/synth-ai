"""Validate and compare canonical trial ancestry exposed by the public result.

This verifies projection coherence, not retention at the record authority. Trial
record IDs are the public attempt identities; internal worker IDs are not guessed.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit unavailable state at acceptance boundaries.
from dataclasses import dataclass
from typing import Optional

from synth_ai.models import RunAttemptTreeAvailable, RunAttemptTreeUnavailable, RunRecordReference

__all__ = ["Attempt", "TreeError", "project_attempts"]


@dataclass(frozen=True, slots=True, order=True)
class Attempt:
    attempt_id: str
    parent_id: Optional[str]
    state: str


@dataclass(slots=True)
class TreeError(Exception):
    code: str
    record_id: str


def _scores_trial(cited: RunRecordReference, current: RunRecordReference) -> bool:
    """A score pins the trial revision it scored.

    The trial may be revised afterwards (the worker's outcome note lands after
    scoring), so the cited reference names the same record at a revision no
    newer than the node's current one. An equal revision must match exactly.
    """
    if (cited.source, cited.kind, cited.record_id) != (
        current.source,
        current.kind,
        current.record_id,
    ):
        return False
    cited_revision, current_revision = int(cited.revision), int(current.revision)
    if cited_revision == current_revision:
        return cited == current
    return cited_revision < current_revision


def project_attempts(
    tree: RunAttemptTreeAvailable | RunAttemptTreeUnavailable, *, run_id: str
) -> Optional[tuple[Attempt, ...]]:
    if tree.run_id != run_id:
        raise TreeError("attempt_tree_run_mismatch", "")
    if tree.state == "unavailable":
        return None
    nodes = {node.trial.record_id: node for node in tree.attempts}
    if len(nodes) != len(tree.attempts):
        raise TreeError("attempt_tree_duplicate_trial", "")
    attempts: list[Attempt] = []
    for record_id, node in nodes.items():
        if node.protocol != tree.protocol or node.experiment != tree.experiment:
            raise TreeError("attempt_tree_organizing_reference_mismatch", record_id)
        parent_id = None
        if node.parent_trial is not None:
            parent_id = node.parent_trial.record_id
            parent = nodes.get(parent_id)
            if parent is None or parent.trial != node.parent_trial:
                raise TreeError("attempt_tree_parent_reference_mismatch", record_id)
        if any(not _scores_trial(score.trial, node.trial) for score in node.scores):
            raise TreeError("attempt_tree_score_reference_mismatch", record_id)
        if any(note.log != tree.research_log for note in node.notes):
            raise TreeError("attempt_tree_note_reference_mismatch", record_id)
        attempts.append(Attempt(record_id, parent_id, node.state))
    if any(note.log != tree.research_log for note in tree.notes):
        raise TreeError("attempt_tree_note_reference_mismatch", "")
    # Linear traversal avoids recursion and quadratic work at the public bound.
    parents = {attempt.attempt_id: attempt.parent_id for attempt in attempts}
    complete: set[str] = set()
    for record_id in parents:
        path: set[str] = set()
        current: Optional[str] = record_id
        while current is not None and current not in complete:
            if current in path:
                raise TreeError("attempt_tree_cycle", current)
            path.add(current)
            current = parents[current]
        complete.update(path)
    return tuple(sorted(attempts))
