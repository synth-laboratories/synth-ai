"""Metadata for GEPA blog task app coverage.

This module centralises the set of task apps that the GEPA blog post
references so that configuration files and documentation can import the
same canonical definitions. Each entry mirrors a task app that is
available via Synth's prompt-learning backend, making it easier to keep
configs, docs, and evaluation notebooks in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class TaskAppSupport:
    """Describes a task app that the GEPA blog supports."""

    app_id: str
    display_name: str
    dataset_id: str
    description: str
    default_port: int
    tags: Sequence[str]
    metrics: Sequence[str]
    sources: Sequence[str]


SUPPORTED_TASK_APPS: tuple[TaskAppSupport, ...] = (
    TaskAppSupport(
        app_id="banking77",
        display_name="Banking77 Intent Classification",
        dataset_id="PolyAI/banking77",
        description="Classify banking customer support queries into 77 intents.",
        default_port=8102,
        tags=("classification", "intent", "nlp"),
        metrics=("accuracy",),
        sources=(
            "GEPA blog quickstart",
            "PolyAI Banking77 dataset card",
        ),
    ),
    TaskAppSupport(
        app_id="hotpotqa",
        display_name="HotpotQA Multi-Hop QA",
        dataset_id="hotpot_qa",
        description="Answer multi-hop questions with supporting facts sourced from Wikipedia passages.",
        default_port=8110,
        tags=("qa", "multi-hop", "reasoning"),
        metrics=("answer_em", "supporting_fact_f1"),
        sources=(
            "GEPA Table 1",
            "HotpotQA (Yang et al., 2018)",
        ),
    ),
    TaskAppSupport(
        app_id="ifbench",
        display_name="IFBench Instruction Following",
        dataset_id="Muennighoff/IFBench",
        description="Follow natural language instructions focusing on faithful adherence.",
        default_port=8111,
        tags=("instruction-following", "nlp"),
        metrics=("compliance", "accuracy"),
        sources=(
            "GEPA Table 1",
            "IFBench benchmark release",
        ),
    ),
    TaskAppSupport(
        app_id="hover",
        display_name="HoVer Claim Verification",
        dataset_id="hover",
        description="Determine whether Wikipedia claims are supported, refuted, or not enough info given retrieved evidence.",
        default_port=8112,
        tags=("fact-checking", "classification"),
        metrics=("label_accuracy", "evidence_f1"),
        sources=(
            "GEPA Table 1",
            "HoVer benchmark (Jiang et al., 2020)",
        ),
    ),
    TaskAppSupport(
        app_id="pupa",
        display_name="PUPA Privacy-Aware Delegation",
        dataset_id="microsoft/PUPA",
        description="Delegate actions while respecting privacy policies and extracting structured responses.",
        default_port=8113,
        tags=("delegation", "privacy", "structured-output"),
        metrics=("privacy_compliance", "task_success"),
        sources=(
            "GEPA Table 1",
            "PUPA benchmark release",
        ),
    ),
)


def list_supported_task_apps() -> Iterable[TaskAppSupport]:
    """Return iterable over supported task apps for convenience."""

    return SUPPORTED_TASK_APPS


__all__ = ["TaskAppSupport", "SUPPORTED_TASK_APPS", "list_supported_task_apps"]
