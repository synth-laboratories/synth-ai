"""Training API for RL, SFT, and Prompt Learning (MIPRO/GEPA).

This module provides both CLI and SDK interfaces for training jobs.

CLI Usage:
    uvx synth-ai train --type prompt_learning --config my_config.toml --poll
    uvx synth-ai train --type sft --config my_config.toml --poll

SDK Usage:
    from synth_ai.sdk.api.train import PromptLearningJob, SFTJob
    
    # Prompt Learning
    job = PromptLearningJob.from_config("my_config.toml")
    job.submit()
    result = job.poll_until_complete()
    
    # SFT
    sft_job = SFTJob.from_config("my_sft_config.toml")
    sft_job.submit()
    result = sft_job.poll_until_complete()
"""

from __future__ import annotations

from typing import Any

# Re-export high-level SDK classes
from .prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
    PromptLearningJobPoller,
)
from .sft import (
    SFTJob,
    SFTJobConfig,
)
from .context_learning import (
    ContextLearningJob,
    ContextLearningJobConfig,
)
from .graphgen import (
    GraphGenJob,
)
from .graphgen_models import (
    GraphGenJobConfig,
    GraphGenTaskSet,
)

__all__ = [
    # CLI
    "register",
    "train_command",
    # SDK - Prompt Learning
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    # SDK - SFT
    "SFTJob",
    "SFTJobConfig",
    # SDK - Context Learning
    "ContextLearningJob",
    "ContextLearningJobConfig",
    # SDK - GraphGen
    "GraphGenJob",
    "GraphGenJobConfig",
    "GraphGenTaskSet",
]


def register(cli: Any) -> None:
    """Register the train command with the CLI."""
    from synth_ai.sdk.api.train.cli import (
        register as _register,  # local import avoids circular dependency
    )

    _register(cli)


def train_command(*args: Any, **kwargs: Any) -> Any:
    """Entrypoint for the train CLI command."""
    from synth_ai.sdk.api.train.cli import (
        train_command as _train_command,  # local import avoids cycle
    )

    return _train_command(*args, **kwargs)
