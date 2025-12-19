from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

import click

from synth_ai.cli.lib.env import get_synth_and_env_keys, mask_str
from synth_ai.cli.lib.train_cfgs import find_train_cfgs_in_cwd, validate_train_cfg
from synth_ai.core.paths import print_paths_formatted

try:
    _config_module = cast(
        Any, importlib.import_module("synth_ai.core.env")
    )
    get_backend_from_env = cast(Callable[[], str], _config_module.get_backend_from_env)
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load backend configuration helpers") from exc

from synth_ai.cli.lib.env import load_env_file
from synth_ai.cli.lib.errors import format_error_message, get_required_value
from synth_ai.core.telemetry import flush_logger, log_error, log_info
from synth_ai.sdk.streaming import (
    GraphGenHandler,
    CLIHandler,
    JobStreamer,
    LossCurveHandler,
    PromptLearningHandler,
    StreamConfig,
    StreamEndpoints,
    StreamType,
)

from .builders import build_prompt_learning_payload, build_rl_payload, build_sft_payload
from .task_app import check_task_app_health
from .graphgen import GraphGenJob
from .graphgen_models import load_graphgen_taskset
from .context_learning import ContextLearningJob
from .utils import (
    TrainError,
    ensure_api_base,
    http_get,
    http_post,
    limit_jsonl_examples,
    mask_value,
    post_multipart,
    preview_json,
    sleep,
    validate_sft_jsonl,
)

# Constants for prompt learning event types
_PROMPT_LEARNING_EVENT_BEST_PROMPT = "prompt.learning.best.prompt"
_PROMPT_LEARNING_EVENT_FINAL_RESULTS = "prompt.learning.final.results"
_PROMPT_LEARNING_EVENT_VALIDATION_SCORED = "prompt.learning.validation.scored"
_PROMPT_LEARNING_EVENT_GEPA_COMPLETE = "prompt.learning.gepa.complete"
_PROMPT_LEARNING_EVENT_MIPRO_COMPLETE = "prompt.learning.mipro.complete"
_PROMPT_LEARNING_EVENT_GEPA_NEW_BEST = "prompt.learning.gepa.new_best"
_PROMPT_LEARNING_EVENT_PHASE_CHANGED = "prompt.learning.phase.changed"
_PROMPT_LEARNING_EVENT_PROGRESS = "prompt.learning.progress"
_PROMPT_LEARNING_EVENT_STREAM_CONNECTED = "prompt.learning.stream.connected"

# Constants for formatting
_MAX_TEXT_REPLACEMENTS_DISPLAY = 3  # Max number of text replacements to show in output
_RESULTS_FILE_MAX_EVENTS = 10000  # Max events to fetch for results file generation


def _format_text_replacements(obj: dict[str, Any] | None, max_display: int = _MAX_TEXT_REPLACEMENTS_DISPLAY) -> list[str]:
    """Extract and format text replacements from a candidate object.
    
    Args:
        obj: Candidate object dictionary containing text_replacements
        max_display: Maximum number of replacements to display
        
    Returns:
        List of formatted lines showing role and replacement text
    """
    lines = []
    if not obj or not isinstance(obj, dict):
        return lines
    
    text_replacements = obj.get("text_replacements", [])
    if not text_replacements or not isinstance(text_replacements, list):
        return lines
    
    for replacement in text_replacements[:max_display]:
        if isinstance(replacement, dict):
            new_text = replacement.get("new_text", "")
            role = replacement.get("apply_to_role", "system")
            if new_text:
                lines.append(f"  [{role.upper()}]: {new_text}")
                lines.append("")
    
    return lines


def _default_backend() -> str:
    """Resolve backend URL with proper production default.
    
    Priority order:
    1. BACKEND_BASE_URL env var (highest priority) - checked FIRST before any .env loading
    2. BACKEND_OVERRIDE env var
    3. get_backend_from_env() standard resolution (which may use SYNTH_BASE_URL from .env)
    
    CRITICAL: This function MUST check BACKEND_BASE_URL directly from os.getenv() 
    to ensure it's not overridden by .env file loading.
    """
    # Check explicit override first (BACKEND_BASE_URL takes absolute precedence)
    # Read directly from os.environ to avoid any dotenv interference
    explicit = os.environ.get("BACKEND_BASE_URL", "").strip()
    if explicit:
        # Return as-is, ensure_api_base() will normalize it
        return explicit
    
    # Fallback to BACKEND_OVERRIDE (also read directly from environ)
    override = os.environ.get("BACKEND_OVERRIDE", "").strip()
    if override:
        return override
    
    # Use standard resolution logic (may use SYNTH_BASE_URL from .env)
    base, _ = get_backend_from_env()
    return f"{base}/api" if not base.endswith("/api") else base


_DEFAULT_SFT_HIDDEN_EVENTS = {
    "sft.created",
    "sft.pricing.check.requested",
    "sft.pricing.check.allowed",
    "sft.stage",
    "snapshot.fetch",
    "hatchet.preflight",
    "hatchet.submission.attempt",
    "hatchet.submission.result",
    "sft.running",
    "sft.status",
    "sft.worker.alive",
    "sft.dispatch.selected",
    "sft.config.prepared",
    "sft.strategy.selected",
    "sft.training.args",
}

_DEFAULT_RL_HIDDEN_SUBSTRINGS = {"modal", "hatchet"}

_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS = {
    "prompt.learning.policy.tokens",
    "mipro.bootstrap.progress",  # Hide individual bootstrap seed scores
    "mipro.tpe.rankings",  # Hide verbose TPE rankings
    "mipro.tpe.selected",  # Hide TPE selection details
    "mipro.tpe.update",  # Hide TPE density updates
    "mipro.trial.duplicate",  # Hide duplicate trial messages
    "mipro.trial.started",  # Hide individual trial start messages (too verbose with instructions)
    "mipro.trial.minibatch",  # Hide minibatch completion (only show full eval)
    "mipro.trial.complete",  # Hide individual trial completion
    "mipro.iteration.skip_generation",  # Hide skip generation messages
    "mipro.budget.update",  # Hide verbose budget updates (progress handler shows summary)
    "mipro.instruction.proposed",  # Hide proposed instructions (shown in results/logs only)
    "gepa.transformation.proposed",  # Hide proposed transformations (shown in results/logs only)
    # Note: mipro.stage_proposer.called is shown so users know instruction generation is happening
}


def _load_toml_config(config_path: Path) -> dict[str, Any]:
    """Load TOML config file."""
    try:
        import tomli  # type: ignore[import-untyped]
    except ImportError:
        # Fallback to tomllib for Python 3.11+
        try:
            import tomllib as tomli
        except ImportError:
            return {}
    
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except Exception:
        return {}


def parse_env_file_path_from_config(config_path: Path) -> str | None:
    """Parse env_file_path from TOML config.
    
    Checks both [prompt_learning] and top-level sections.
    """
    config = _load_toml_config(config_path)
    
    # Check prompt_learning section first
    pl_section = config.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        env_file_path = pl_section.get("env_file_path")
        if env_file_path:
            return str(env_file_path)
    
    # Check top-level
    env_file_path = config.get("env_file_path")
    if env_file_path:
        return str(env_file_path)
    
    return None


def parse_results_folder(config_path: Path) -> Path:
    """Parse results_folder from TOML config and validate it exists.
    
    Checks both [prompt_learning] and top-level sections.
    Raises ClickException if missing or invalid.
    """
    config = _load_toml_config(config_path)
    
    # Check prompt_learning section first
    pl_section = config.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        results_folder = pl_section.get("results_folder")
        if results_folder:
            results_folder_str = str(results_folder).strip()
            # Resolve relative to config file's directory if path is relative
            if not Path(results_folder_str).is_absolute():
                config_dir = config_path.parent.resolve()
                results_path = (config_dir / results_folder_str).resolve()
            else:
                results_path = Path(results_folder_str).expanduser().resolve()
            
            # Validate that the folder exists or can be created
            try:
                results_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise click.ClickException(
                    f"Could not create results folder: {results_path}\n"
                    f"  Error: {e}\n"
                    f"  Config: {config_path}\n"
                    f"  TOML results_folder: {results_folder}"
                ) from e
            
            return results_path
    
    # Check top-level section
    results_folder = config.get("results_folder")
    if results_folder:
        results_folder_str = str(results_folder).strip()
        # Resolve relative to config file's directory if path is relative
        if not Path(results_folder_str).is_absolute():
            config_dir = config_path.parent.resolve()
            results_path = (config_dir / results_folder_str).resolve()
        else:
            results_path = Path(results_folder_str).expanduser().resolve()
        
        # Validate that the folder exists or can be created
        try:
            results_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise click.ClickException(
                f"Could not create results folder: {results_path}\n"
                f"  Error: {e}\n"
                f"  Config: {config_path}\n"
                f"  TOML results_folder: {results_folder}"
            ) from e
        
        return results_path
    
    # Missing - raise error
    raise click.ClickException(
        f"Missing required 'results_folder' field in TOML config: {config_path}\n"
        f"  Please add 'results_folder = \"path/to/results\"' to [prompt_learning] section or top-level.\n"
        f"  Paths can be relative (to config file directory) or absolute."
    )


def parse_display_config(config_path: Path) -> dict[str, Any]:
    """Parse [display] section from TOML config."""
    config = _load_toml_config(config_path)
    display_section = config.get("display", {})
    
    # Also extract termination_config for max limits
    termination_section = config.get("termination_config", {})
    # Also check prompt_learning.termination_config
    pl_section = config.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_termination = pl_section.get("termination_config", {})
        if isinstance(pl_termination, dict):
            # Merge with top-level termination_config (top-level takes precedence)
            termination_section = {**pl_termination, **termination_section}
    
    return {
        "local_backend": display_section.get("local_backend", False),
        "tui": display_section.get("tui", False),
        "show_curve": display_section.get("show_curve", True),
        "verbose_summary": display_section.get("verbose_summary", True),
        "show_trial_results": display_section.get("show_trial_results", True),
        "show_transformations": display_section.get("show_transformations", False),
        "show_validation": display_section.get("show_validation", True),
        "max_tokens": termination_section.get("max_tokens"),
        "max_time_seconds": termination_section.get("max_time_seconds"),
        "max_rollouts": termination_section.get("max_rollouts"),
    }


def _build_stream_components(
    stream_format: str,
    *,
    hidden_event_types: set[str] | None = None,
    hidden_event_substrings: set[str] | None = None,
) -> tuple[StreamConfig, list]:
    """Return stream configuration and handlers for the requested format."""
    if stream_format == "chart":
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "sft.progress",
                "sft.training.started",
                "sft.training.finish",
                "sft.validation.summary",
                "rl.train.step",
                "rl.train.started",
                "rl.train.completed",
                "workflow.completed",
                "workflow.failed",
            },
            metric_names={"train.loss"},
        )
        handlers = [LossCurveHandler()]
    else:
        config = StreamConfig.default()
        handlers = [
            CLIHandler(
                hidden_event_types=hidden_event_types or set(),
                hidden_event_substrings=hidden_event_substrings or set(),
            )
        ]
    return config, handlers


def _validate_openai_key_if_provider_is_openai(cfg_path: Path) -> None:
    """Validate that OPENAI_API_KEY is set if the provider is OpenAI.

    For prompt learning jobs, checks if policy.provider is 'openai' and raises
    a ClickException if OPENAI_API_KEY is not set in the environment.
    """
    cfg = _load_toml_config(cfg_path)

    # Check prompt_learning section
    pl_section = cfg.get("prompt_learning", {})
    if not isinstance(pl_section, dict):
        return

    policy = pl_section.get("policy", {})
    if not isinstance(policy, dict):
        return

    provider = policy.get("provider", "").lower()

    if provider == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not openai_key:
            raise click.ClickException(
                "OPENAI_API_KEY is required when using provider='openai'.\n"
                "Please set OPENAI_API_KEY in your .env file or environment."
            )


# Module-level logging to track import and registration
import logging as _logging  # noqa: E402
import sys  # noqa: E402

_logger = _logging.getLogger(__name__)
_logger.debug("[TRAIN_MODULE] Module synth_ai.sdk.api.train.cli imported")

@click.command("train")
@click.argument(
    "cfg_path",
    required=False,
    type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--env",
    "env_file",
    type=click.Path(exists=True, path_type=Path),
    help=".env file(s) to preload (skips selection prompt)",
)
@click.option(
    "--task-url",
    default=None,
    help="Override task app base URL (RL only)"
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(),
    default=None,
    help="Override dataset JSONL path (SFT)",
)
@click.option("--model", default=None, help="Override model identifier")
@click.option(
    "--allow-experimental",
    "allow_experimental",
    is_flag=True,
    flag_value=True,
    default=None,
    help="Allow experimental models (overrides SDK_EXPERIMENTAL env)",
)
@click.option(
    "--no-allow-experimental",
    "allow_experimental",
    is_flag=True,
    flag_value=False,
    help="Disallow experimental models (overrides SDK_EXPERIMENTAL env)",
)
@click.option("--idempotency", default=None, help="Idempotency-Key header for job creation")
@click.option("--dry-run", is_flag=True, hidden=True, help="Deprecated: no-op")
@click.option("--poll/--no-poll", default=True, help="Poll job status until terminal state")
@click.option(
    "--poll-timeout", default=3600.0, type=float, help="Maximum seconds to poll before timing out"
)
@click.option("--poll-interval", default=5.0, type=float, help="Seconds between poll attempts")
@click.option(
    "--stream-format",
    type=click.Choice(["cli", "chart"]),
    default="cli",
    show_default=True,
    help="Streaming output style (cli = line updates, chart = live loss panel)",
)
@click.option(
    "--examples",
    "examples_limit",
    type=int,
    default=None,
    help="Limit SFT training to the first N examples",
)
@click.option(
    "--backend",
    "backend_override",
    default=None,
    help="Backend base URL (e.g., http://localhost:8000). Overrides BACKEND_BASE_URL env var.",
)
@click.option(
    "--local-backend",
    is_flag=True,
    default=None,
    help="Use local backend (localhost:8000). Overrides TOML [display].local_backend",
)
@click.option(
    "--tui",
    is_flag=True,
    default=None,
    help="Enable live TUI dashboard. Overrides TOML [display].tui",
)
@click.option(
    "--show-curve",
    is_flag=True,
    default=None,
    help="Show optimization curve at end. Overrides TOML [display].show_curve",
)
@click.option(
    "--verbose-summary",
    is_flag=True,
    default=None,
    help="Show detailed final summary. Overrides TOML [display].verbose_summary",
)
@click.option(
    "--type",
    "train_type_override",
    type=click.Choice(["prompt", "rl", "sft", "adas", "context_learning"]),
    default=None,
    help="Explicitly set training type. Required for ADAS (uses JSON datasets).",
)
@click.option(
    "--rollout-budget",
    "rollout_budget",
    type=int,
    default=None,
    help="Rollout budget for ADAS optimization (default: 100)",
)
@click.option(
    "--proposer-effort",
    "proposer_effort",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Proposer effort level for ADAS (default: medium)",
)
def train_command(
    cfg_path: Path | None,
    env_file: Path | None,
    task_url: str | None,
    dataset_path: str | None,
    model: str | None,
    allow_experimental: bool | None,
    idempotency: str | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
    examples_limit: int | None,
    backend_override: str | None,
    local_backend: bool | None,
    tui: bool | None,
    show_curve: bool | None,
    verbose_summary: bool | None,
    train_type_override: str | None,
    rollout_budget: int | None,
    proposer_effort: str | None,
) -> None:

    """Interactive launcher for RL / SFT / Prompt Learning / ADAS / Context Learning jobs."""
    import traceback

    ctx: dict[str, Any] = {
        "cfg_path": str(cfg_path) if cfg_path else None,
        "poll": poll,
        "poll_timeout": poll_timeout,
        "poll_interval": poll_interval,
        "stream_format": stream_format,
        "backend_override": backend_override,
    }
    log_info("train_command invoked", ctx=ctx)

    # Wrap entire function in try-except to catch ALL exceptions
    try:
        # Log entry point IMMEDIATELY - this should always appear
        sys.stderr.write("[TRAIN_CMD] Starting train command\n")
        sys.stderr.flush()
        click.echo(f"[TRAIN_CMD] Args: cfg_path={cfg_path}, poll={poll}", err=True)
        click.echo(f"[TRAIN_CMD] Python executable: {sys.executable}", err=True)
        click.echo(f"[TRAIN_CMD] Working directory: {os.getcwd()}", err=True)
        
        try:
            load_env_file()
            click.echo("[TRAIN_CMD] Environment file loaded", err=True)
        except Exception as e:
            click.echo(f"[TRAIN_CMD] ERROR loading env file: {e}", err=True)
            traceback.print_exc(file=sys.stderr)
            raise

        # CRITICAL: Load explicit .env file BEFORE config validation to ensure BACKEND_BASE_URL is available
        if env_file and Path(env_file).exists():
            from dotenv import load_dotenv
            # Load with override=True to ensure BACKEND_BASE_URL from .env takes precedence
            load_dotenv(Path(env_file), override=True)
            click.echo(f"[TRAIN_CMD] Loaded explicit .env: {env_file}", err=True)

        # Handle ADAS specially - it uses JSON datasets, not TOML configs
        if train_type_override == "adas":
            # For ADAS, dataset_path is required and cfg_path is ignored
            if not dataset_path:
                raise click.ClickException(
                    "ADAS requires --dataset flag with path to JSON dataset file.\n"
                    "Usage: synth-ai train --type adas --dataset my_tasks.json"
                )
            train_type = "adas"
            click.echo(f"[TRAIN_CMD] ADAS mode: using dataset {dataset_path}", err=True)
        else:
            # Non-ADAS: use TOML config
            if not cfg_path:
                available_cfgs = find_train_cfgs_in_cwd()
                if len(available_cfgs) == 1:
                    train_type, cfg_path_str, _ = available_cfgs[0]
                    cfg_path = Path(cfg_path_str)
                    print(f"Automatically selected {train_type} training config at", cfg_path)
                else:
                    if len(available_cfgs) == 0:
                        print("No training config found in cwd.")
                        print("Validate your training config: synth-ai train-cfg check [CFG_PATH]")
                    else:
                        print("Multiple training configs found. Please specify which one to use:")
                        print_paths_formatted(available_cfgs)
                    print("Usage: synth-ai train --config [CFG_PATH]")
                    return None

            train_type = train_type_override or validate_train_cfg(cfg_path)
        
        synth_api_key, _ = get_synth_and_env_keys(env_file)
        
        # Resolve backend URL with priority: --backend flag > BACKEND_BASE_URL env > default
        if backend_override:
            # CLI flag takes highest precedence
            backend_base = ensure_api_base(backend_override.strip())
            click.echo(f"Backend base: {backend_base} (from --backend flag)")
        else:
            # Check BACKEND_BASE_URL AFTER loading env file
            backend_base_url_env = os.environ.get("BACKEND_BASE_URL", "").strip()
            backend_override_env = os.environ.get("BACKEND_OVERRIDE", "").strip()
            
            # Debug: Show what env vars are set
            click.echo(f"🔍 DEBUG: BACKEND_BASE_URL={backend_base_url_env or '(not set)'}", err=True)
            click.echo(f"🔍 DEBUG: BACKEND_OVERRIDE={backend_override_env or '(not set)'}", err=True)
            
            # Use _default_backend() to respect BACKEND_BASE_URL env var
            backend_raw = _default_backend()
            click.echo(f"🔍 DEBUG: _default_backend() returned: {backend_raw}", err=True)
            backend_base = ensure_api_base(backend_raw)
            
            # Assertion: Validate backend URL is what we expect
            if backend_base_url_env:
                expected_backend = ensure_api_base(backend_base_url_env)
                if backend_base != expected_backend:
                    raise click.ClickException(
                        f"Backend URL mismatch! Expected: {expected_backend}, Got: {backend_base}. "
                        f"BACKEND_BASE_URL={backend_base_url_env} but resolved to {backend_base}. "
                        f"This indicates BACKEND_BASE_URL is not being respected.\n"
                        f"💡 Solutions:\n"
                        f"   1. Add BACKEND_BASE_URL=http://localhost:8000 to your .env file\n"
                        f"   2. Use --backend http://localhost:8000 flag (requires package rebuild)\n"
                        f"   3. Set BACKEND_OVERRIDE=http://localhost:8000 in your shell\n"
                        f"   4. Set SYNTH_BACKEND_URL_OVERRIDE=local and LOCAL_BACKEND_URL=http://localhost:8000"
                    )
            
            click.echo(f"Backend base: {backend_base} (key {mask_str(synth_api_key)})")
            if backend_base_url_env:
                click.echo(f"  (from BACKEND_BASE_URL={backend_base_url_env})")

        # Skip TOML-based validation for ADAS (uses JSON datasets)
        if train_type != "adas" and cfg_path:
            _validate_openai_key_if_provider_is_openai(cfg_path)

        match train_type:
            case "prompt":
                if not cfg_path:
                    raise click.ClickException("Prompt Learning requires a TOML config file.")
                handle_prompt_learning(
                    cfg_path=cfg_path,
                    backend_base=backend_base,
                    synth_key=synth_api_key,
                    task_url_override=task_url,
                    allow_experimental=allow_experimental,
                    dry_run=dry_run,
                    poll=poll,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                    stream_format=stream_format,
                )
            case "context_learning":
                if not cfg_path:
                    raise click.ClickException(
                        "Context Learning requires a TOML config file.\n"
                        "Usage: synth-ai train --type context_learning --config my_context.toml"
                    )
                handle_context_learning(
                    cfg_path=cfg_path,
                    backend_base=backend_base,
                    synth_key=synth_api_key,
                    poll=poll,
                    stream_format=stream_format,
                )
            case "rl":
                if not cfg_path:
                    raise click.ClickException("RL requires a TOML config file.")
                handle_rl(
                    cfg_path=cfg_path,
                    backend_base=backend_base,
                    synth_key=synth_api_key,
                    task_url_override=task_url,
                    model_override=model,
                    idempotency=idempotency,
                    allow_experimental=allow_experimental,
                    dry_run=dry_run,
                    poll=poll,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                    stream_format=stream_format,
                )
            case "sft":
                if not cfg_path:
                    raise click.ClickException("SFT requires a TOML config file.")
                dataset_override_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
                handle_sft(
                    cfg_path=cfg_path,
                    backend_base=backend_base,
                    synth_key=synth_api_key,
                    dataset_override=dataset_override_path,
                    allow_experimental=allow_experimental,
                    dry_run=dry_run,
                    poll=poll,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                    stream_format=stream_format,
                    examples_limit=examples_limit,
                )
            case "adas":
                if not dataset_path:
                    raise click.ClickException("ADAS requires a dataset path.")
                adas_dataset_path = Path(dataset_path).expanduser().resolve()
                handle_adas(
                    dataset_path=adas_dataset_path,
                    backend_base=backend_base,
                    synth_key=synth_api_key,
                    policy_model=model,
                    rollout_budget=rollout_budget,
                    proposer_effort=proposer_effort,
                    poll=poll,
                    poll_timeout=poll_timeout,
                    poll_interval=poll_interval,
                    stream_format=stream_format,
                )
    except Exception as e:
        ctx["error"] = type(e).__name__
        log_error("train_command failed", ctx=ctx)
        click.echo(f"[TRAIN_CMD] FATAL ERROR: {e}", err=True)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        flush_logger()


def handle_context_learning(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    poll: bool,
    stream_format: str,
) -> None:
    """Submit and stream a Context Learning job.

    Context Learning is SSE-first; polling flags are ignored.
    """
    if not poll:
        click.echo("Note: --no-poll is ignored for context learning (SSE streaming only).")

    click.echo("\n=== Submitting Context Learning Job ===")
    try:
        job = ContextLearningJob.from_config(
            cfg_path,
            backend_url=backend_base,
            api_key=synth_key,
        )
        result = job.submit()
    except Exception as e:
        raise click.ClickException(str(e))

    click.echo("\n✓ Job created:")
    click.echo(f"  Context Learning Job ID: {result.job_id}")
    click.echo(f"  Status: {result.status}")

    click.echo("\n=== Streaming Job Progress ===")
    if stream_format == "chart":
        click.echo("Chart stream format is not supported for context learning; using CLI output.")

    try:
        final_status = job.stream_until_complete()
    except Exception as e:
        raise click.ClickException(str(e))

    status = final_status.get("status") if isinstance(final_status, dict) else "unknown"
    click.echo(f"\nFinal status: {status}")
    click.echo(preview_json(final_status, limit=600))

    if status in {"succeeded", "completed"}:
        click.echo("\n=== Best Preflight Script ===")
        try:
            best = job.download_best_script()
            if best.preflight_script:
                click.echo(best.preflight_script[:2000])
                if len(best.preflight_script) > 2000:
                    click.echo(
                        f"\n... (truncated, {len(best.preflight_script)} chars total)"
                    )
        except Exception as e:
            click.echo(f"⚠️  Could not download best script: {e}")


def _wait_for_training_file(
    backend_base: str, api_key: str, file_id: str, *, timeout: float = 10.0
) -> None:
    """Wait for training file to be visible after upload.
    
    Reduced from 120s to 10s because:
    - POST response already confirms file is uploaded
    - Backend now forces read-your-writes consistency  
    - By job creation time, replica lag has resolved
    - Quick sanity check only, not critical path
    """
    url = f"{backend_base.rstrip('/')}/files/{file_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    elapsed = 0.0
    interval = 2.0
    first_check = True
    while True:
        resp = http_get(url, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                data = {}
            status = str(
                data.get("status") or data.get("state") or data.get("storage_state") or "ready"
            ).lower()
            if first_check:
                click.echo(f"File uploaded successfully (id={file_id}, status={status})")
                first_check = False
            if status in {"ready", "uploaded", "stored", "complete"}:
                click.echo(f"✓ Training file ready (status={status})")
                return
            # Show progress for processing states
            if status in {"processing", "pending", "validating"}:
                click.echo(
                    f"  Waiting for file processing... (status={status}, {elapsed:.0f}s elapsed)"
                )
        elif resp.status_code == 404:
            # Keep polling; object may not be visible yet
            if first_check:
                click.echo(f"Waiting for file {file_id} to become visible...")
                first_check = False
        elif resp.status_code in {401, 403}:
            # Auth errors won't resolve by polling - fail immediately
            try:
                error_body = resp.json()
            except json.JSONDecodeError:
                error_body = resp.text[:400]
            click.echo("\n[ERROR] Authentication failed when checking training file:")
            click.echo(f"  URL: {url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {error_body}")
            click.echo(f"  API key: {mask_value(api_key)}")
            raise click.ClickException(
                f"Authentication error ({resp.status_code}). "
                "Check that your SYNTH_API_KEY is valid and has permission to access this organization's files."
            )
        else:
            # Other errors - show details but keep polling
            try:
                error_body = resp.json()
            except json.JSONDecodeError:
                error_body = resp.text[:400]
            click.echo(f"[WARN] Unexpected response checking file {file_id}:")
            click.echo(f"  URL: {url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {error_body}")

        if elapsed >= timeout:
            raise click.ClickException(
                f"Training file {file_id} not ready after {timeout:.0f}s (last status: {resp.status_code})"
            )
        sleep(interval)
        elapsed += interval


def handle_rl(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    task_url_override: str | None,
    model_override: str | None,
    idempotency: str | None,
    allow_experimental: bool | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
) -> None:
    ctx: dict[str, Any] = {
        "cfg_path": str(cfg_path),
        "backend_base": backend_base,
        "task_url_override": task_url_override,
        "poll": poll,
    }
    log_info("handle_rl invoked", ctx=ctx)
    overrides: dict[str, Any] = {
        "backend": backend_base,
        "task_url": task_url_override,
        "model": model_override,
    }
    build = build_rl_payload(
        config_path=cfg_path,
        task_url=task_url_override or os.environ.get("TASK_APP_URL", ""),
        overrides=overrides,
        idempotency=idempotency,
        allow_experimental=allow_experimental,
    )

    # Backend-side verification: try ALL org environment keys against /health and /task_info
    verify_url = f"{backend_base}/rl/verify_task_app"
    verify_headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
    try:
        vresp = http_post(
            verify_url, headers=verify_headers, json_body={"endpoint_base_url": build.task_url}
        )
        try:
            parsed_json = vresp.json()
        except json.JSONDecodeError:
            parsed_json = None

        if isinstance(parsed_json, Mapping):
            vjs: dict[str, Any] = dict(parsed_json)
        else:
            vjs = {
                "status": vresp.status_code,
                "text": (vresp.text or "")[:400],
            }
            if parsed_json is not None:
                vjs["body"] = parsed_json
    except Exception as _ve:
        raise click.ClickException(
            f"Task app verification call failed: {type(_ve).__name__}: {_ve}"
        ) from _ve
    if vresp.status_code is not None and vresp.status_code >= 400:
        click.echo("Task app verification error:\n" + preview_json(vjs, limit=800))
        raise click.ClickException(f"Verification failed with status {vresp.status_code}")
    if not bool(vjs.get("any_ok")):
        click.echo("Task app verification failed; no auth combination succeeded. Full report:")
        click.echo(preview_json(vjs, limit=1200))
        raise click.ClickException("Task app verification failed (auth)")
    else:
        # Print concise summary
        try:
            cands = vjs.get("candidates_first15") or []
            attempts_raw = vjs.get("attempts")
            attempts: list[Mapping[str, Any]] = (
                [a for a in attempts_raw if isinstance(a, Mapping)]
                if isinstance(attempts_raw, list)
                else []
            )
            statuses = [attempt.get("status") for attempt in attempts]
            click.echo(f"Verification OK (candidates={cands}, statuses={statuses})")
        except (KeyError, ValueError, AttributeError):
            # Parsing verification summary failed, but verification itself succeeded
            click.echo("Verification OK")

    env_key = get_required_value(
        "environment_api_key",
        env_value=os.environ.get("ENVIRONMENT_API_KEY"),
    )
    os.environ["ENVIRONMENT_API_KEY"] = env_key

    click.echo("Performing task app health check…")
    health = check_task_app_health(build.task_url, env_key)
    if not health.ok:
        click.echo(f"Task app health check failed: {health.detail}")
        raise click.ClickException("Aborting due to failing health check")
    else:
        click.echo("Task app healthy")

    create_url = f"{backend_base}/rl/jobs"
    headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
    if build.idempotency:
        headers["Idempotency-Key"] = build.idempotency

    click.echo(f"POST {create_url}")
    click.echo("Payload preview:\n" + preview_json(build.payload, limit=800))

    resp = http_post(create_url, headers=headers, json_body=build.payload)
    try:
        js = resp.json()
    except json.JSONDecodeError as e:
        click.echo(f"⚠️  Failed to parse JSON response: {e}")
        js = {"status": resp.status_code, "text": resp.text[:400]}
    click.echo(f"Response {resp.status_code}: {preview_json(js, limit=400)}")
    if resp.status_code not in (200, 201):
        raise click.ClickException("Job creation failed")
    job_id = js.get("job_id") or js.get("id")
    if not job_id:
        raise click.ClickException("Response missing job id")

    if not poll:
        click.echo(f"Created job {job_id} (polling disabled)")
        return

    click.echo("\n=== Streaming Job Progress ===")
    
    # Enable metrics for prompt learning
    if stream_format == "chart":
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "prompt.learning.progress",
                "prompt.learning.gepa.start",
                "prompt.learning.gepa.complete",
            },
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [LossCurveHandler()]
        click.echo("Using live chart (metric=gepa.transformation.mean_score)")
    else:
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [CLIHandler(hidden_event_substrings=_DEFAULT_RL_HIDDEN_SUBSTRINGS)]
    
    streamer = JobStreamer(
        base_url=backend_base,
        api_key=synth_key,
        job_id=job_id,
        endpoints=StreamEndpoints.rl(job_id),
        config=config,
        handlers=handlers,
        interval_seconds=poll_interval,
        timeout_seconds=poll_timeout,
    )
    final_status = asyncio.run(streamer.stream_until_terminal())
    click.echo(f"Final status: {final_status.get('status', 'unknown')}")
    click.echo(preview_json(final_status, limit=600))


def handle_sft(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    dataset_override: Path | None,
    allow_experimental: bool | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
    examples_limit: int | None,
) -> None:
    ctx: dict[str, Any] = {
        "cfg_path": str(cfg_path),
        "backend_base": backend_base,
        "dataset_override": str(dataset_override) if dataset_override else None,
        "poll": poll,
    }
    log_info("handle_sft invoked", ctx=ctx)
    try:
        build = build_sft_payload(
            config_path=cfg_path,
            dataset_override=dataset_override,
            allow_experimental=allow_experimental,
        )
    except TrainError as exc:
        _raise_sft_usage_error(exc)

    limited_path: Path | None = None

    try:
        if examples_limit is not None:
            limited_path = limit_jsonl_examples(build.train_file, examples_limit)
            click.echo(
                f"Using first {examples_limit} examples from {build.train_file} -> {limited_path}"
            )
            build.train_file = limited_path

        click.echo("Validating training dataset…")
        validate_sft_jsonl(build.train_file)
        if build.validation_file and build.validation_file.suffix == ".jsonl":
            click.echo("Validating validation dataset…")
            validate_sft_jsonl(build.validation_file)

        upload_url = f"{backend_base.rstrip('/')}/files"
        click.echo("\n=== Uploading Training Data ===")
        click.echo(f"Dataset: {build.train_file}")
        click.echo(f"Destination: {upload_url}")
        resp = post_multipart(
            upload_url, api_key=synth_key, file_field="file", file_path=build.train_file
        )
        js = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        if resp.status_code is not None and resp.status_code >= 400 or "id" not in js:
            click.echo("\n[ERROR] Training file upload failed:")
            click.echo(f"  URL: {upload_url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {js or resp.text[:400]}")
            click.echo(f"  File: {build.train_file}")
            raise click.ClickException(
                f"Training file upload failed with status {resp.status_code}"
            )
        train_file_id = js["id"]
        click.echo(f"✓ Training file uploaded (id={train_file_id})")
        val_file_id = None
        if build.validation_file:
            click.echo(f"Uploading validation dataset: {build.validation_file}")
            vresp = post_multipart(
                upload_url,
                api_key=synth_key,
                file_field="file",
                file_path=build.validation_file,
            )
            vjs = (
                vresp.json()
                if vresp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            if vresp.status_code is not None and vresp.status_code < 400 and "id" in vjs:
                val_file_id = vjs["id"]
                click.echo(f"✓ Validation file uploaded (id={val_file_id})")
            else:
                click.echo(
                    f"[WARN] Validation upload failed ({vresp.status_code}): {vjs or vresp.text[:200]}"
                )
        payload = dict(build.payload)
        payload["training_file_id"] = train_file_id
        if val_file_id:
            payload.setdefault("metadata", {}).setdefault("effective_config", {}).setdefault(
                "data", {}
            )["validation_files"] = [val_file_id]

        click.echo("\n=== Checking File Processing Status ===")
        try:
            _wait_for_training_file(backend_base, synth_key, train_file_id)
        except click.ClickException as exc:
            click.echo(f"[WARN] File readiness check failed: {exc}")
            click.echo("Proceeding anyway - backend will validate file during job creation...")

        click.echo("\n=== Creating Training Job ===")
        click.echo("Job payload preview:")
        click.echo(preview_json(payload, limit=800))

        create_url = f"{backend_base}/learning/jobs"
        headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
        click.echo(f"\nPOST {create_url}")
        resp = http_post(create_url, headers=headers, json_body=payload)
        js = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        if resp.status_code not in (200, 201):
            click.echo("\n[ERROR] Job creation failed:")
            click.echo(f"  URL: {create_url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {preview_json(js, limit=600)}")
            raise click.ClickException(f"Job creation failed with status {resp.status_code}")
        job_id = js.get("job_id") or js.get("id")
        if not job_id:
            raise click.ClickException("Response missing job id")
        click.echo(f"✓ Job created (id={job_id})")

        click.echo("\n=== Starting Training Job ===")
        start_url = f"{backend_base}/learning/jobs/{job_id}/start"
        click.echo(f"POST {start_url}")
        start_resp = http_post(start_url, headers=headers, json_body={})
        if start_resp.status_code not in (200, 201):
            click.echo(f"[WARN] Job start returned status {start_resp.status_code}")
        else:
            click.echo("✓ Job started")

        if not poll:
            click.echo(f"Started job {job_id} (polling disabled)")
            return

        click.echo("\n=== Streaming Job Progress ===")
        config, handlers = _build_stream_components(
            stream_format, hidden_event_types=_DEFAULT_SFT_HIDDEN_EVENTS
        )
        if stream_format == "chart":
            click.echo("Using live loss chart (metric=train.loss)")
        streamer = JobStreamer(
            base_url=backend_base,
            api_key=synth_key,
            job_id=job_id,
            endpoints=StreamEndpoints.learning(job_id),
            config=config,
            handlers=handlers,
            interval_seconds=poll_interval,
            timeout_seconds=poll_timeout,
        )
        final_status = asyncio.run(streamer.stream_until_terminal())
        status = final_status.get('status') if isinstance(final_status, dict) else 'unknown'
        click.echo(f"Final status: {status}")
        click.echo(preview_json(final_status, limit=600))
    finally:
        if limited_path is not None:
            with contextlib.suppress(OSError):
                limited_path.unlink(missing_ok=True)
            # Clean up empty parent directory if possible
            with contextlib.suppress(OSError):
                limited_path.parent.rmdir()


def handle_adas(
    *,
    dataset_path: Path,
    backend_base: str,
    synth_key: str,
    policy_model: str | None,
    rollout_budget: int | None,
    proposer_effort: str | None,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
) -> None:
    """Handle ADAS workflow optimization job creation and streaming.

    ADAS uses JSON dataset files and auto-generates task apps.
    """
    ctx: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "backend_base": backend_base,
        "poll": poll,
    }
    log_info("handle_adas invoked", ctx=ctx)

    # Load dataset
    click.echo(f"Loading ADAS dataset from: {dataset_path}")
    try:
        dataset = load_graphgen_taskset(dataset_path)
    except FileNotFoundError:
        raise click.ClickException(f"Dataset file not found: {dataset_path}")
    except ValueError as e:
        raise click.ClickException(f"Invalid ADAS dataset format: {e}")

    click.echo(f"Dataset loaded: {dataset.metadata.name}")
    click.echo(f"  Tasks: {len(dataset.tasks)}")
    click.echo(f"  Gold outputs: {len(dataset.gold_outputs)}")
    click.echo(f"  Judge mode: {dataset.judge_config.mode}")

    # Create ADAS job
    job = GraphGenJob.from_dataset(
        dataset=dataset,
        policy_model=policy_model or "gpt-4o-mini",
        rollout_budget=rollout_budget or 100,
        proposer_effort=proposer_effort or "medium",  # type: ignore
        backend_url=backend_base,
        api_key=synth_key,
        auto_start=True,
    )

    click.echo("\n=== Submitting ADAS Job ===")
    click.echo(f"Policy model: {job.config.policy_model}")
    click.echo(f"Rollout budget: {job.config.rollout_budget}")
    click.echo(f"Proposer effort: {job.config.proposer_effort}")

    try:
        result = job.submit()
    except RuntimeError as e:
        raise click.ClickException(str(e))

    click.echo(f"\n✓ Job created:")
    click.echo(f"  ADAS Job ID: {result.graphgen_job_id}")
    click.echo(f"  Status: {result.status}")

    if not poll:
        click.echo(f"\nCreated job {result.graphgen_job_id} (polling disabled)")
        return

    click.echo("\n=== Streaming Job Progress ===")

    # Build stream handlers
    if stream_format == "chart":
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [LossCurveHandler()]
        click.echo("Using live loss chart (metric=gepa.transformation.mean_score)")
    else:
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            max_events_per_poll=500,
            deduplicate=True,
        )
        handlers = [GraphGenHandler()]

    # Stream until complete
    try:
        final_status = job.stream_until_complete(
            timeout=poll_timeout,
            interval=poll_interval,
            handlers=handlers,
        )
    except TimeoutError as e:
        raise click.ClickException(str(e))

    status = final_status.get('status') if isinstance(final_status, dict) else 'unknown'
    click.echo(f"\nFinal status: {status}")
    click.echo(preview_json(final_status, limit=600))

    # Download and display best prompt if succeeded
    if status == "succeeded" or status == "completed":
        click.echo("\n=== Best Optimized Prompt ===")
        try:
            prompt = job.download_prompt()
            if prompt:
                click.echo(prompt[:2000])
                if len(prompt) > 2000:
                    click.echo(f"\n... (truncated, {len(prompt)} chars total)")
        except Exception as e:
            click.echo(f"⚠️  Could not download prompt: {e}")


def _raise_sft_usage_error(exc: TrainError) -> NoReturn:
    message = str(exc).strip()
    lower_msg = message.lower()
    context = "Preparing SFT training job payload"
    impact = "Cannot submit training job without a valid dataset path"

    if "dataset not specified" in lower_msg:
        raise click.UsageError(
            format_error_message(
                summary="Dataset path required",
                context=context,
                problem="No dataset path was provided via config or CLI",
                impact=impact,
                solutions=[
                    ("Add [job].data = \"/path/to/data.jsonl\" to the config", "Persist the dataset path in the TOML file"),
                    ("Re-run with --dataset /path/to/data.jsonl", "Override the dataset path from the CLI"),
                    ("Use an absolute path accessible from the current working directory", "Relative paths are resolved from the shell cwd"),
                ],
            )
        ) from exc

    if "dataset not found" in lower_msg:
        raise click.UsageError(
            format_error_message(
                summary="Dataset path not found",
                context=context,
                problem=message,
                impact=impact,
                solutions=[
                    ("Verify the dataset path exists on disk", "Double-check spelling and that the file hasn't moved"),
                    ("Provide an absolute path to the dataset file", "Avoid relying on relative paths that resolve incorrectly"),
                    ("Sync the dataset to this machine before running the CLI", "Remote paths must be accessible locally"),
                ],
            )
        ) from exc

    raise click.ClickException(message) from exc


def _save_verbose_log_file(
    events: list[dict[str, Any]],
    log_file: Path,
    algorithm_name: str,
    job_id: str,
    append_summary: bool = False,
) -> None:
    """Save a verbose log file with all events in chronological order, including summary.
    
    If append_summary is True, only append the summary section (events were already streamed live).
    """
    import json
    from datetime import datetime
    
    try:
        lines = []
        if not append_summary:
            # Full log file with header and all events
            lines.append("=" * 80)
            lines.append(f"{algorithm_name} PROMPT LEARNING VERBOSE LOG")
            lines.append("=" * 80)
            lines.append(f"Job ID: {job_id}")
            lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Total Events: {len(events)}")
            lines.append("=" * 80)
            lines.append("")
        
        # Sort events by timestamp if available
        def get_timestamp(event: dict[str, Any]) -> str:
            return event.get("timestamp", event.get("created_at", ""))
        
        sorted_events = sorted(events, key=get_timestamp)
        
        # Only include events if not appending summary (events were already streamed live)
        if not append_summary:
            for idx, event in enumerate(sorted_events, 1):
                if not isinstance(event, dict):
                    continue
                
                event_type = event.get("type", "unknown")
                timestamp = event.get("timestamp") or event.get("created_at", "")
                level = event.get("level", "info")
                message = event.get("message", "")
                data = event.get("data", {})
                
                lines.append(f"[{idx}] {timestamp} [{level.upper()}] {event_type}")
                if message:
                    lines.append(f"  Message: {message}")
                if data:
                    # Format data nicely (truncate very long values)
                    formatted_data = {}
                    for key, value in data.items():
                        if isinstance(value, dict | list):
                            # Convert to JSON string, truncate if too long
                            json_str = json.dumps(value, indent=2)
                            if len(json_str) > 1000:
                                json_str = json_str[:1000] + "... (truncated)"
                            formatted_data[key] = json_str
                        elif isinstance(value, str) and len(value) > 500:
                            formatted_data[key] = value[:500] + "... (truncated)"
                        else:
                            formatted_data[key] = value
                    
                    if formatted_data:
                        lines.append(f"  Data: {json.dumps(formatted_data, indent=2)}")
                lines.append("")
        
        # Add summary table and chart at the end (always included)
        if append_summary:
            lines.append("\n\n")
        lines.append("=" * 80)
        lines.append("FINAL SUMMARY")
        lines.append("=" * 80)
        
        try:
            from .summary import _generate_summary_text
            # Extract optimization curve from events
            optimization_curve = None
            trial_scores = []
            for event in sorted_events:
                if isinstance(event, dict):
                    event_type = event.get("type", "")
                    if event_type in ("prompt.learning.trial.complete", "mipro.new_incumbent"):
                        data = event.get("data", {})
                        trial_num = data.get("trial") or data.get("trial_num")
                        score = data.get("score") or data.get("minibatch_score")
                        if trial_num is not None and score is not None:
                            trial_scores.append((trial_num, score))
            
            if trial_scores:
                best_so_far = {}
                for trial_num, score in sorted(trial_scores):
                    if trial_num not in best_so_far or score > best_so_far[trial_num]:
                        best_so_far[trial_num] = score
                optimization_curve = sorted(best_so_far.items())
            
            summary_text, curve_text = _generate_summary_text(
                events=sorted_events,
                algorithm=algorithm_name.lower() if algorithm_name else None,
                optimization_curve=optimization_curve,
            )
            if summary_text:
                lines.append(summary_text)
            if curve_text:
                lines.append("")
                lines.append(curve_text)
        except Exception as e:
            lines.append(f"⚠️  Could not generate summary: {e}")
        
        lines.append("=" * 80)
        lines.append("END OF LOG")
        lines.append("=" * 80)
        
        # Write to file (append if summary-only mode)
        mode = "a" if append_summary else "w"
        with open(log_file, mode, encoding="utf-8") as f:
            if append_summary:
                f.write("\n")
            f.write("\n".join(lines))
    
    except Exception as e:
        click.echo(f"⚠️  Could not save verbose log file: {e}")


def _save_prompt_learning_results_locally(
    *,
    backend_base: str,
    api_key: str,
    job_id: str,
    config_path: Path,
    results_folder: Path,
) -> None:
    """Fetch events and generate results file locally after prompt learning completes."""
    from datetime import datetime
    
    try:
        # Fetch all events
        url = f"{backend_base}/prompt-learning/online/jobs/{job_id}/events?limit={_RESULTS_FILE_MAX_EVENTS}"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = http_get(url, headers=headers, timeout=30.0)
        
        if resp.status_code != 200:
            click.echo(f"⚠️  Could not fetch events to generate results file (status={resp.status_code})")
            return
        
        data = resp.json()
        # Handle both list response (backend) and dict response (legacy compatibility)
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            events = data.get("events", [])
            if not isinstance(events, list):
                click.echo(f"⚠️  Events field is not a list: {type(events).__name__}")
                return
        else:
            click.echo(f"⚠️  Unexpected response type: {type(data).__name__}")
            return
        
        if not events:
            return
        
        # Extract key data from events
        best_score = None
        best_prompt = None
        baseline_score = None
        attempted_candidates = []
        optimized_candidates = []
        mipro_topk_candidates = []  # Collect MIPRO top-K candidates
        proposed_instructions = []  # Collect proposed instructions from MIPRO
        proposed_transformations = []  # Collect proposed transformations from GEPA
        
        for event in events:
            if not isinstance(event, dict):
                continue  # Skip malformed events
            
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            if not isinstance(event_data, dict):
                event_data = {}  # Fallback to empty dict for safety
            
            if event_type == _PROMPT_LEARNING_EVENT_BEST_PROMPT:
                best_score = event_data.get("best_score")
                best_prompt = event_data.get("best_prompt")
            elif event_type == _PROMPT_LEARNING_EVENT_FINAL_RESULTS:
                attempted_candidates = event_data.get("attempted_candidates", [])
                optimized_candidates = event_data.get("optimized_candidates", [])
            elif event_type == _PROMPT_LEARNING_EVENT_VALIDATION_SCORED:
                # Check if this is the baseline by checking for is_baseline flag or baseline in message
                is_baseline = event_data.get("is_baseline", False)
                if not is_baseline:
                    msg = event.get("message", "")
                    is_baseline = "baseline" in msg.lower()
                if is_baseline:
                    baseline_score = event_data.get("accuracy")
            elif event_type == _PROMPT_LEARNING_EVENT_GEPA_COMPLETE and best_score is None:
                best_score = event_data.get("best_score")
            elif event_type == _PROMPT_LEARNING_EVENT_MIPRO_COMPLETE:
                # MIPRO completion event includes best_prompt and best_score
                if best_score is None:
                    best_score = event_data.get("best_score")
                if best_prompt is None:
                    best_prompt = event_data.get("best_prompt")
            elif event_type == "mipro.topk.evaluated":
                # Extract MIPRO top-K candidate data with full details
                rank = event_data.get("rank")
                train_score = event_data.get("train_score")
                test_score = event_data.get("test_score")
                if rank is not None and train_score is not None and test_score is not None:
                    # Extract full instruction text (may be multi-line)
                    instruction_text = event_data.get("instruction_text", "")
                    if not instruction_text:
                        # Try to get from instruction_lines if available
                        instruction_lines = event_data.get("instruction_lines", [])
                        if instruction_lines:
                            instruction_text = "\n".join(str(line) for line in instruction_lines)
                    
                    mipro_topk_candidates.append({
                        "rank": rank,
                        "train_score": train_score,
                        "test_score": test_score,
                        "lift_absolute": event_data.get("lift_absolute"),
                        "lift_percent": event_data.get("lift_percent"),
                        "instruction_text": instruction_text,
                        "instruction_lines": event_data.get("instruction_lines", []),
                        "demo_indices": event_data.get("demo_indices", []),
                        "stage_payloads": event_data.get("stage_payloads", {}),
                        "instruction_indices": event_data.get("instruction_indices", []),
                        "test_per_seed": event_data.get("test_per_seed", {}),
                    })
            elif event_type == "mipro.baseline.test":
                # Extract baseline test score
                if baseline_score is None:
                    baseline_score = event_data.get("test_score")
            elif event_type == "mipro.instruction.proposed":
                # Collect proposed instructions
                proposed_instructions.append({
                    "iteration": event_data.get("iteration"),
                    "stage_id": event_data.get("stage_id"),
                    "module_id": event_data.get("module_id"),
                    "instruction_id": event_data.get("instruction_id"),
                    "instruction_text": event_data.get("instruction_text", ""),
                    "instruction_lines": event_data.get("instruction_lines", []),
                    "demo_indices": event_data.get("demo_indices", []),
                    "proposal_id": event_data.get("proposal_id"),
                    "timestamp": event.get("created_at"),
                })
            elif event_type == "gepa.transformation.proposed":
                # Collect proposed transformations
                proposed_transformations.append({
                    "generation": event_data.get("generation"),
                    "mutation_type": event_data.get("mutation_type"),
                    "operator": event_data.get("operator"),
                    "transformation_id": event_data.get("transformation_id"),
                    "parent_id": event_data.get("parent_id"),
                    "transformation_text": event_data.get("transformation_text", ""),
                    "transformation_dict": event_data.get("transformation_dict", {}),
                    "mutation_params": event_data.get("mutation_params", {}),
                    "timestamp": event.get("created_at"),
                })
        
        # Check if we have any results to display (best_prompt, best_score, or candidates)
        has_results = bool(attempted_candidates or optimized_candidates or best_prompt or best_score is not None)
        if not has_results:
            return
        
        # Determine algorithm name from events
        algorithm_name = "PROMPT LEARNING"
        for event in events:
            if isinstance(event, dict):
                event_type = event.get("type", "")
                if "gepa" in event_type.lower():
                    algorithm_name = "GEPA"
                    break
                elif "mipro" in event_type.lower():
                    algorithm_name = "MIPRO"
                    break
        
        # Generate formatted report
        lines = []
        lines.append("=" * 80)
        lines.append(f"{algorithm_name} PROMPT LEARNING RESULTS")
        lines.append("=" * 80)
        lines.append(f"Job ID: {job_id}")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        if baseline_score is not None:
            lines.append(f"📊 Baseline Score: {baseline_score:.4f} ({baseline_score*100:.1f}%)")
        if best_score is not None:
            lines.append(f"🏆 Best Score:     {best_score:.4f} ({best_score*100:.1f}%)")
        if baseline_score is not None and best_score is not None:
            improvement = ((best_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0
            lines.append(f"📈 Improvement:    {improvement:+.1f}% relative ({(best_score - baseline_score)*100:+.1f} pp absolute)")
        lines.append("=" * 80)
        lines.append("")
        
        # Add best prompt if available
        if best_prompt and isinstance(best_prompt, dict):
            lines.append("🏆 BEST PROMPT")
            lines.append("-" * 80)
            sections = best_prompt.get("sections", [])
            if not isinstance(sections, list):
                sections = []
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                role = sec.get("role", "unknown")
                content = sec.get("content", "")
                lines.append(f"\n[{role.upper()}]:")
                lines.append(content)
            lines.append("")
        
        # Add optimized candidates
        if optimized_candidates and isinstance(optimized_candidates, list):
            lines.append("=" * 80)
            lines.append(f"✨ TOP OPTIMIZED CANDIDATES ({len(optimized_candidates)})")
            lines.append("=" * 80)
            lines.append("")
            
            for idx, cand in enumerate(optimized_candidates):
                if not isinstance(cand, dict):
                    continue
                candidate_score = cand.get("score") or {}
                accuracy = candidate_score.get("accuracy", 0.0)
                prompt_length = candidate_score.get("prompt_length", 0)
                payload_kind = cand.get("payload_kind", "unknown")
                
                # Try score.instance_scores first, then cand.instance_scores (explicit check)
                instance_scores = (
                    candidate_score.get('instance_scores') 
                    if 'instance_scores' in candidate_score 
                    else cand.get('instance_scores')
                )
                n_eval = len(instance_scores) if instance_scores and isinstance(instance_scores, list) else 0
                
                lines.append(f"[{idx+1}] Accuracy: {accuracy:.4f} | Length: {prompt_length} | Type: {payload_kind} | N: {n_eval}")
                lines.append("-" * 80)
                
                obj = cand.get("object")
                if obj and isinstance(obj, dict) and payload_kind == "transformation":
                    # For transformations, text_replacements are nested in data
                    data_obj = obj.get("data", {})
                    replacement_lines = _format_text_replacements(data_obj)
                    lines.extend(replacement_lines)
                lines.append("")
        
        # Add MIPRO top-K candidates
        if mipro_topk_candidates and isinstance(mipro_topk_candidates, list):
            # Sort by rank
            mipro_topk_candidates.sort(key=lambda x: x.get("rank", 999))
            lines.append("=" * 80)
            lines.append(f"🎯 TOP-K CANDIDATES ({len(mipro_topk_candidates)})")
            lines.append("=" * 80)
            lines.append("")
            
            for cand in mipro_topk_candidates:
                rank = cand.get("rank", 0)
                train_score = cand.get("train_score", 0.0)
                test_score = cand.get("test_score", 0.0)
                lift_abs = cand.get("lift_absolute")
                lift_pct = cand.get("lift_percent")
                instruction_text = cand.get("instruction_text", "")
                instruction_lines = cand.get("instruction_lines", [])
                demo_indices = cand.get("demo_indices", [])
                instruction_indices = cand.get("instruction_indices", [])
                stage_payloads = cand.get("stage_payloads", {})
                test_per_seed = cand.get("test_per_seed", {})
                
                lift_str = ""
                if lift_abs is not None and lift_pct is not None:
                    lift_str = f" | Lift: {lift_abs:+.3f} ({lift_pct:+.1f}%)"
                
                lines.append(f"[Rank {rank}] Train: {train_score:.4f} ({train_score*100:.1f}%) | Test: {test_score:.4f} ({test_score*100:.1f}%){lift_str}")
                lines.append("-" * 80)
                
                # Show full instruction text (use instruction_lines if available, otherwise instruction_text)
                if instruction_lines:
                    lines.append("Instructions:")
                    for idx, instr_line in enumerate(instruction_lines, 1):
                        lines.append(f"  {idx}. {instr_line}")
                elif instruction_text:
                    # Split multi-line instructions
                    instr_parts = instruction_text.split("\n")
                    if len(instr_parts) > 1:
                        lines.append("Instructions:")
                        for idx, part in enumerate(instr_parts, 1):
                            if part.strip():
                                lines.append(f"  {idx}. {part.strip()}")
                    else:
                        lines.append(f"Instruction: {instruction_text}")
                
                if instruction_indices:
                    lines.append(f"Instruction Indices: {instruction_indices}")
                if demo_indices:
                    lines.append(f"Demo Indices: {demo_indices}")
                
                # Show per-stage breakdown if available
                if stage_payloads:
                    lines.append("Per-stage breakdown:")
                    for stage_id, payload in stage_payloads.items():
                        if isinstance(payload, dict):
                            instr_ids = payload.get("instruction_indices", [])
                            demo_ids = payload.get("demo_indices", [])
                            module_id = payload.get("module_id", "unknown")
                            lines.append(f"  [{module_id}/{stage_id}] instr_ids={instr_ids} demo_ids={demo_ids}")
                
                # Show test per-seed scores if available
                if test_per_seed:
                    seed_scores = []
                    for seed, score in sorted(test_per_seed.items()):
                        seed_scores.append(f"{seed}: {score:.2f}")
                    if seed_scores:
                        lines.append(f"Test per-seed: {', '.join(seed_scores)}")
                
                lines.append("")
        
        # Add all proposal candidates
        if attempted_candidates and isinstance(attempted_candidates, list):
            lines.append("=" * 80)
            lines.append(f"💡 ALL PROPOSAL CANDIDATES ({len(attempted_candidates)})")
            lines.append("=" * 80)
            lines.append("")
            
            for idx, cand in enumerate(attempted_candidates):
                if not isinstance(cand, dict):
                    continue
                accuracy = cand.get('accuracy', 0.0)
                prompt_length = cand.get('prompt_length', 0)
                tool_rate = cand.get('tool_call_rate', 0.0)
                instance_scores = cand.get('instance_scores', [])
                n_eval = len(instance_scores) if instance_scores else 0
                
                lines.append(f"[{idx+1}] Accuracy: {accuracy:.4f} | Length: {prompt_length} | Tool Rate: {tool_rate:.2f} | N: {n_eval}")
                lines.append("-" * 80)
                
                obj = cand.get("object")
                if obj and isinstance(obj, dict):
                    # For proposals, text_replacements are at top level of object
                    replacement_lines = _format_text_replacements(obj)
                    lines.extend(replacement_lines)
                lines.append("")
        
        # Add proposed instructions section (MIPRO)
        if proposed_instructions and isinstance(proposed_instructions, list):
            lines.append("=" * 80)
            lines.append(f"💡 PROPOSED INSTRUCTIONS ({len(proposed_instructions)})")
            lines.append("=" * 80)
            lines.append("")
            
            for idx, instr in enumerate(proposed_instructions):
                if not isinstance(instr, dict):
                    continue
                iteration = instr.get("iteration", "?")
                stage_id = instr.get("stage_id", "?")
                module_id = instr.get("module_id", "?")
                instruction_id = instr.get("instruction_id", "?")
                instruction_text = instr.get("instruction_text", "")
                instruction_lines = instr.get("instruction_lines", [])
                demo_indices = instr.get("demo_indices", [])
                
                lines.append(f"[{idx+1}] Iteration {iteration} | Stage: {stage_id} | Module: {module_id} | ID: {instruction_id}")
                if demo_indices:
                    lines.append(f"Demo Indices: {demo_indices}")
                lines.append("-" * 80)
                
                # Show instruction text (use instruction_lines if available, otherwise instruction_text)
                if instruction_lines:
                    for line_idx, line in enumerate(instruction_lines, 1):
                        if line.strip():
                            lines.append(f"  {line_idx}. {line.strip()}")
                elif instruction_text:
                    # Split multi-line instructions
                    instr_parts = instruction_text.split("\n")
                    if len(instr_parts) > 1:
                        for line_idx, part in enumerate(instr_parts, 1):
                            if part.strip():
                                lines.append(f"  {line_idx}. {part.strip()}")
                    else:
                        lines.append(f"  {instruction_text}")
                
                lines.append("")
        
        # Add proposed transformations section (GEPA)
        if proposed_transformations and isinstance(proposed_transformations, list):
            lines.append("=" * 80)
            lines.append(f"🧬 PROPOSED TRANSFORMATIONS ({len(proposed_transformations)})")
            lines.append("=" * 80)
            lines.append("")
            
            for idx, trans in enumerate(proposed_transformations):
                if not isinstance(trans, dict):
                    continue
                generation = trans.get("generation", "?")
                mutation_type = trans.get("mutation_type", "?")
                operator = trans.get("operator", "?")
                transformation_id = trans.get("transformation_id", "?")
                parent_id = trans.get("parent_id", "?")
                transformation_text = trans.get("transformation_text", "")
                transformation_dict = trans.get("transformation_dict", {})
                
                lines.append(f"[{idx+1}] Generation {generation} | Type: {mutation_type} | Operator: {operator}")
                lines.append(f"Transformation ID: {transformation_id} | Parent ID: {parent_id}")
                lines.append("-" * 80)
                
                # Show transformation text
                if transformation_text:
                    lines.append("Transformation Text:")
                    lines.append(f"  {transformation_text}")
                
                # Show transformation dict details if available
                if transformation_dict:
                    text_replacements = transformation_dict.get("text_replacements", [])
                    if text_replacements:
                        lines.append("Text Replacements:")
                        for repl_idx, repl in enumerate(text_replacements, 1):
                            if isinstance(repl, dict):
                                apply_to = repl.get("apply_to_role", "unknown")
                                old_text = repl.get("old_text", "")[:100]
                                new_text = repl.get("new_text", "")[:200]
                                lines.append(f"  {repl_idx}. [{apply_to}]")
                                if old_text:
                                    lines.append(f"      Old: {old_text}...")
                                if new_text:
                                    lines.append(f"      New: {new_text}...")
                
                lines.append("")
        
        # Add summary table and chart before END OF REPORT
        lines.append("")
        lines.append("=" * 80)
        lines.append("FINAL SUMMARY")
        lines.append("=" * 80)
        
        # Generate summary table text (reuse summary.py logic)
        try:
            from .summary import _generate_summary_text
            # Extract optimization curve from events if available
            optimization_curve = None
            # Try to extract curve from trial events
            trial_scores = []
            for event in events:
                if isinstance(event, dict):
                    event_type = event.get("type", "")
                    if event_type in ("prompt.learning.trial.complete", "mipro.new_incumbent"):
                        data = event.get("data", {})
                        trial_num = data.get("trial") or data.get("trial_num")
                        score = data.get("score") or data.get("minibatch_score")
                        if trial_num is not None and score is not None:
                            trial_scores.append((trial_num, score))
            
            if trial_scores:
                # Build optimization curve (best score so far at each trial)
                best_so_far = {}
                for trial_num, score in sorted(trial_scores):
                    if trial_num not in best_so_far or score > best_so_far[trial_num]:
                        best_so_far[trial_num] = score
                optimization_curve = sorted(best_so_far.items())
            
            summary_text, curve_text = _generate_summary_text(
                events=events,
                algorithm=algorithm_name.lower() if algorithm_name else None,
                optimization_curve=optimization_curve,
            )
            if summary_text:
                lines.append(summary_text)
            if curve_text:
                lines.append("")
                lines.append(curve_text)
        except Exception as e:
            lines.append(f"⚠️  Could not generate summary: {e}")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Determine save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use results_folder from config (create if it doesn't exist)
        output_dir = results_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use algorithm-specific filename
        algorithm_prefix = algorithm_name.lower() if algorithm_name else "prompt_learning"
        output_file = output_dir / f"{algorithm_prefix}_results_{job_id}_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        click.echo(f"\n📄 Results saved locally to: {output_file}")
        
        # Also save verbose log file with all events (append summary if log was streamed live)
        log_file = output_dir / f"{algorithm_prefix}_log_{job_id}_{timestamp}.log"
        append_summary = log_file.exists()  # If log file exists, it was streamed live, so just append summary
        _save_verbose_log_file(events, log_file, algorithm_name, job_id, append_summary=append_summary)
        click.echo(f"📋 Verbose log saved locally to: {log_file}")
        
    except (PermissionError, OSError) as e:
        click.echo(f"⚠️  Could not save results file locally: {e}")
    except Exception as e:
        click.echo(f"⚠️  Unexpected error saving results file: {e}")


def handle_prompt_learning(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    task_url_override: str | None,
    allow_experimental: bool | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
    display_config: dict[str, Any] | None = None,
    tui: bool = False,
    show_curve: bool = True,
    verbose_summary: bool = True,
) -> None:
    """Handle prompt learning job creation (MIPRO or GEPA)."""
    ctx: dict[str, Any] = {
        "cfg_path": str(cfg_path),
        "backend_base": backend_base,
        "task_url_override": task_url_override,
        "poll": poll,
    }
    log_info("handle_prompt_learning invoked", ctx=ctx)
    env_key = get_required_value(
        "environment_api_key",
        env_value=os.environ.get("ENVIRONMENT_API_KEY"),
    )
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    
    overrides: dict[str, Any] = {
        "backend": backend_base,
        "task_url": task_url_override,
    }
    
    build = build_prompt_learning_payload(
        config_path=cfg_path,
        task_url=task_url_override,
        overrides=overrides,
        allow_experimental=allow_experimental,
    )
    
    # Assertion: Validate task app URL is reachable from backend perspective
    # If backend is localhost and task app is localhost, they should be able to communicate
    task_app_url = build.task_url or ""
    if backend_base.startswith("http://localhost") or backend_base.startswith("http://127.0.0.1"):
        if task_app_url.startswith("http://localhost") or task_app_url.startswith("http://127.0.0.1"):
            # Both are local - this should work
            pass
        else:
            click.echo(f"⚠️  WARNING: Backend is local ({backend_base}) but task app is remote ({task_app_url})")
            click.echo("   The backend may not be able to reach the task app. Consider using a tunnel or local task app.")
    
    click.echo("Performing task app health check…")
    click.echo(f"Task app URL: {build.task_url}")
    click.echo("⏳ Checking /health endpoint (timeout: 10s)...")
    health = check_task_app_health(build.task_url, env_key, timeout=10.0)
    if not health.ok:
        click.echo(f"❌ Task app health check failed: {health.detail}")
        click.echo(f"   Health status: {health.health_status}")
        click.echo(f"   Task info status: {health.task_info_status}")
        click.echo("💡 Troubleshooting:")
        click.echo("   1. Ensure the task app is running: lsof -i :8102")
        click.echo("   2. Test manually: curl -v http://127.0.0.1:8102/health")
        click.echo("   3. Check task app logs for errors")
        click.echo("   4. Restart the task app if it's hung")
        raise click.ClickException("Aborting due to failing health check")
    else:
        click.echo("Task app healthy")
    
    # Ensure backend_base has /api prefix
    if not backend_base.endswith("/api"):
        backend_base = ensure_api_base(backend_base)
    
    # Assertion: Validate backend URL before making request
    if not backend_base.startswith("http"):
        raise click.ClickException(
            f"Invalid backend URL: {backend_base}. Must start with http:// or https://"
        )
    
    create_url = f"{backend_base}/prompt-learning/online/jobs"
    headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
    
    click.echo(f"POST {create_url}")
    click.echo("Payload preview:\n" + preview_json(build.payload, limit=800))
    
    # Assertion: If using local backend, verify it's actually localhost
    if (
        os.getenv("BACKEND_BASE_URL")
        and "localhost" in os.getenv("BACKEND_BASE_URL", "").lower()
        and "localhost" not in backend_base.lower()
        and "127.0.0.1" not in backend_base
    ):
        raise click.ClickException(
            f"BACKEND_BASE_URL was set to localhost but backend_base resolved to {backend_base}. "
            f"This indicates the environment variable is not being respected."
        )
    
    # Increase timeout for job creation (can take longer due to validation checks)
    resp = http_post(create_url, headers=headers, json_body=build.payload, timeout=180.0)
    try:
        js = resp.json()
    except json.JSONDecodeError as e:
        click.echo(f"⚠️  Failed to parse JSON response: {e}")
        js = {"status": resp.status_code, "text": resp.text[:400]}
    click.echo(f"Response {resp.status_code}: {preview_json(js, limit=400)}")
    if resp.status_code not in (200, 201):
        raise click.ClickException("Job creation failed")
    job_id = js.get("job_id") or js.get("id")
    if not job_id:
        raise click.ClickException("Response missing job id")
    
    if not poll:
        click.echo(f"Created job {job_id} (polling disabled)")
        return
    
    algorithm = str(build.payload.get("algorithm") or "").lower()
    metric_names: set[str] | None = None
    if algorithm == "gepa":
        metric_names = {"gepa.transformation.mean_score"}

    chart_mode = stream_format == "chart" and algorithm == "gepa"
    if stream_format == "chart" and not chart_mode:
        click.echo("Chart streaming is only available for GEPA jobs; showing textual updates instead.")

    # Prepare log file path for real-time streaming
    results_folder = parse_results_folder(cfg_path)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_prefix = algorithm.lower() if algorithm else "prompt_learning"
    log_file = results_folder / f"{algorithm_prefix}_log_{job_id}_{timestamp}.log"
    
    # Write initial streaming message to log file if handler will be created
    if not chart_mode:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n=== Streaming Job Progress ===\n")
        except Exception:
            pass  # Continue even if log file can't be written
    
    click.echo("\n=== Streaming Job Progress ===")

    # Create appropriate handler based on algorithm
    if algorithm == "gepa":
        if chart_mode:
            config = StreamConfig(
                enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
                event_types={
                    "prompt.learning.progress",
                    "prompt.learning.gepa.start",
                    "prompt.learning.gepa.complete",
                },
                metric_names=metric_names,
            )
            handlers = [LossCurveHandler()]
            click.echo("Using live loss chart (metric=gepa.transformation.mean_score)")
        else:
            config = StreamConfig(
                enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS, StreamType.TIMELINE},
                metric_names=metric_names,
                max_events_per_poll=500,  # Capture more events per poll
                deduplicate=True,  # Still deduplicate but capture more
                # Don't filter events - show all of them
                event_types=None,  # No whitelist - show all event types
                event_types_exclude=None,  # No blacklist - show all events
                event_levels=None,  # Show all levels
            )
            # Use PromptLearningHandler for enhanced event handling
            handler = PromptLearningHandler(
                show_trial_results=display_config.get("show_trial_results", True) if display_config else True,
                show_transformations=display_config.get("show_transformations", False) if display_config else False,
                show_validation=display_config.get("show_validation", True) if display_config else True,
                max_tokens=display_config.get("max_tokens") if display_config else None,
                max_time_seconds=display_config.get("max_time_seconds") if display_config else None,
                max_rollouts=display_config.get("max_rollouts") if display_config else None,
                log_file=log_file,
            )
            handlers = [handler]
    else:
        # Use PromptLearningHandler for MIPRO (same as GEPA)
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS, StreamType.TIMELINE},
            metric_names=metric_names,
            max_events_per_poll=500,  # Capture more events per poll
            deduplicate=True,  # Still deduplicate but capture more
            # Don't filter events - show all of them
            event_types=None,  # No whitelist - show all event types
            event_types_exclude=None,  # No blacklist - show all events
            event_levels=None,  # Show all levels
        )
        handler = PromptLearningHandler(
            show_trial_results=display_config.get("show_trial_results", True) if display_config else True,
            show_transformations=display_config.get("show_transformations", False) if display_config else False,
            show_validation=display_config.get("show_validation", True) if display_config else True,
            max_tokens=display_config.get("max_tokens") if display_config else None,
            max_time_seconds=display_config.get("max_time_seconds") if display_config else None,
            max_rollouts=display_config.get("max_rollouts") if display_config else None,
            log_file=log_file,
        )
        handlers = [handler]
    
    streamer = JobStreamer(
        base_url=backend_base,
        api_key=synth_key,
        job_id=job_id,
        endpoints=StreamEndpoints.prompt_learning(job_id),
        config=config,
        handlers=handlers,
        interval_seconds=poll_interval,
        timeout_seconds=poll_timeout,
    )
    final_status = asyncio.run(streamer.stream_until_terminal())
    
    # Write final status to log file if handler has one
    if isinstance(handlers[0], PromptLearningHandler) and handlers[0]._log_file_handle:
        handlers[0]._write_log(f"Final status: {final_status.get('status', 'unknown')}")
        handlers[0]._write_log(preview_json(final_status, limit=600))
    
    click.echo(f"Final status: {final_status.get('status', 'unknown')}")
    click.echo(preview_json(final_status, limit=600))
    
    # Display final summary for GEPA/MIPRO jobs if requested
    if verbose_summary and algorithm in ("gepa", "mipro"):
        optimization_curve = None
        if isinstance(handlers[0], PromptLearningHandler):
            optimization_curve = handlers[0].optimization_curve
        
        from .summary import display_prompt_learning_summary
        # Pass log_writer if handler has one
        log_writer = None
        if isinstance(handlers[0], PromptLearningHandler) and handlers[0]._log_file_handle:
            log_writer = handlers[0]._write_log
        display_prompt_learning_summary(
            job_id=job_id,
            backend_base=backend_base,
            api_key=synth_key,
            optimization_curve=optimization_curve,
            show_curve=show_curve,
            algorithm=algorithm,
            log_writer=log_writer,
        )
    
    # Save results file locally
    # Parse and validate results_folder from config (already done above, but ensure it's available)
    if 'results_folder' not in locals():
        results_folder = parse_results_folder(cfg_path)
    
    # Close log file if handler has one (flush is already called by streamer, but ensure it's closed)
    if isinstance(handlers[0], PromptLearningHandler) and handlers[0]._log_file_handle:
        handlers[0].flush()
    
    _save_prompt_learning_results_locally(
        backend_base=backend_base,
        api_key=synth_key,
        job_id=job_id,
        config_path=cfg_path,
        results_folder=results_folder,
    )


def register(cli: click.Group) -> None:
    cli.add_command(train_command)
