"""Evaluation functionality for task apps."""

from synth_ai.core.eval.config import EvalRunConfig, SeedSet, resolve_eval_config
from synth_ai.core.eval.runner import (
    EvalResult,
    format_eval_report,
    format_eval_table,
    run_eval,
    run_eval_direct,
    run_eval_via_backend,
    save_traces,
)
from synth_ai.core.eval.validation import validate_eval_options

__all__ = [
    "EvalRunConfig",
    "resolve_eval_config",
    "SeedSet",
    "run_eval",
    "run_eval_direct",
    "run_eval_via_backend",
    "format_eval_table",
    "format_eval_report",
    "save_traces",
    "EvalResult",
    "validate_eval_options",
]
