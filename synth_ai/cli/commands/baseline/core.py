"""CLI command for baseline evaluation."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import click

from synth_ai.sdk.baseline.config import BaselineResults
from synth_ai.sdk.baseline.discovery import (
    BASELINE_FILE_PATTERNS,
    BaselineChoice,
    discover_baseline_files,
    load_baseline_config_from_file,
)
from synth_ai.sdk.baseline.execution import aggregate_results, run_baseline_evaluation


class BaselineGroup(click.Group):
    """Custom group that allows positional arguments (baseline_id) even when subcommands exist."""
    
    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra,
    ) -> click.Context:
        """Override make_context to store original args before Click parses them."""
        # Store original args in the context's meta
        ctx = super().make_context(info_name, args, parent, **extra)
        ctx.meta['_original_args'] = args.copy() if isinstance(args, list) else list(args)
        return ctx
    
    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve command, checking if first arg is a subcommand or baseline_id."""
        
        # Check if first arg is a known subcommand
        if args and not args[0].startswith('--'):
            first_arg = args[0]
            if first_arg in self.commands:
                # It's a known subcommand, let Click handle it normally
                cmd_name, cmd, remaining = super().resolve_command(ctx, args)
                return cmd_name, cmd, remaining
        
        # Not a subcommand - this means baseline_id is a positional argument
        # Store baseline_id in ctx for the callback to access
        if args and not args[0].startswith('--'):
            baseline_id = args[0]
            ctx.meta['baseline_id'] = baseline_id
            # Remove baseline_id from args so Click doesn't try to parse it
            remaining_args = args[1:]
            
            # Create a wrapper function that injects baseline_id into the callback
            original_callback = self.callback
            if original_callback is None:
                raise click.ClickException("Command callback is None")
            def wrapper_callback(ctx, **kwargs):
                # Inject baseline_id into kwargs
                kwargs['baseline_id'] = baseline_id
                return original_callback(ctx, **kwargs)
            
            # Create a wrapper command with the modified callback
            # Filter out baseline_id from params since we're injecting it manually
            filtered_params = [p for p in self.params if getattr(p, 'name', None) != 'baseline_id']
            wrapper_cmd = click.Command(
                name="_baseline_wrapper",  # Use a different name to avoid confusion
                callback=wrapper_callback,
                params=filtered_params,
                context_settings=self.context_settings,
            )
            return "_baseline_wrapper", wrapper_cmd, remaining_args
        
        # No args or args start with --, so no baseline_id
        # Let Click handle it normally (will invoke main callback if invoke_without_command=True)
        cmd_name, cmd, remaining = super().resolve_command(ctx, args)
        return cmd_name, cmd, remaining
    
    def invoke(self, ctx: click.Context) -> Any:
        """Invoke command, handling baseline_id as positional arg."""
        # Check if baseline_id is in ctx.params (Click might have parsed it)
        if 'baseline_id' in ctx.params and ctx.params['baseline_id']:
            baseline_id = ctx.params['baseline_id']
            # Invoke callback with baseline_id from params
            if self.callback is None:
                raise click.ClickException("Command callback is None")
            return self.callback(ctx, **ctx.params)
        
        # Manually call resolve_command with full args (including baseline_id if present)
        # Try to get the original args from ctx.meta (stored in make_context())
        full_args = ctx.meta.get('_original_args', ctx.args)
        
        # If no args, invoke callback directly (invoke_without_command=True behavior)
        if not full_args:
            if self.callback is None:
                raise click.ClickException("Command callback is None")
            return ctx.invoke(self.callback, **ctx.params)
        
        cmd_name, cmd, resolved_args = self.resolve_command(ctx, full_args)
        
        # Check if baseline_id was detected
        if 'baseline_id' in ctx.meta:
            baseline_id = ctx.meta['baseline_id']
            # Parse options from resolved_args - don't use OptionParser, just use Click's make_context
            # Create a temporary context to parse the options
            temp_ctx = self.make_context(self.name, resolved_args, parent=ctx.parent, allow_extra_args=True, allow_interspersed_args=False)
            params = temp_ctx.params.copy()
            params['baseline_id'] = baseline_id
            # Don't pass ctx explicitly - Click's @click.pass_context decorator injects it
            # Use ctx.invoke to properly call the callback with the right context
            if self.callback is None:
                raise click.ClickException("Command callback is None")
            return ctx.invoke(self.callback, **params)
        
        # Normal flow - if it's a subcommand, invoke it
        if cmd and cmd is not self and isinstance(cmd, click.Command):
            with cmd.make_context(cmd_name, resolved_args, parent=ctx) as sub_ctx:
                return cmd.invoke(sub_ctx)
        
        # No baseline_id and no subcommand - invoke callback if invoke_without_command=True
        if self.callback is None:
            raise click.ClickException("Command callback is None")
        return self.callback(ctx)


__all__ = ["command"]

def _select_baseline_interactive(choices: list[BaselineChoice]) -> Optional[str]:
    """Prompt user to select a baseline interactively."""
    if not choices:
        return None
    
    if len(choices) == 1:
        return choices[0].baseline_id
    
    click.echo("\nFound multiple baseline files:")
    for i, choice in enumerate(choices, 1):
        click.echo(f"  {i}. {choice.baseline_id} ({choice.path})")
    
    while True:
        try:
            selection = click.prompt("Select baseline", type=int)
            if 1 <= selection <= len(choices):
                return choices[selection - 1].baseline_id
            click.echo(f"Please enter a number between 1 and {len(choices)}")
        except (click.Abort, KeyboardInterrupt):
            return None

def _parse_seeds(seeds_str: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated seeds string."""
    if not seeds_str:
        return None
    
    try:
        return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    except ValueError as e:
        raise click.ClickException(f"Invalid seeds format: {seeds_str}. Expected comma-separated integers.") from e

def _parse_splits(splits_str: str) -> list[str]:
    """Parse comma-separated splits string."""
    return [s.strip() for s in splits_str.split(",") if s.strip()]

@click.group(
    "baseline",
    help="Run self-contained task evaluation using a baseline file.",
    invoke_without_command=True,
    cls=BaselineGroup,
)
@click.pass_context
# DON'T define baseline_id as an argument here - it will be consumed before resolve_command()
# @click.argument("baseline_id", type=str, required=False)
@click.option(
    "--split",
    default="train",
    help="Data split(s) to evaluate (comma-separated). Default: train",
)
@click.option(
    "--seeds",
    default=None,
    help="Comma-separated seeds to evaluate (overrides split defaults)",
)
@click.option(
    "--model",
    default=None,
    help="Model identifier (overrides default_policy_config)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (overrides default_policy_config)",
)
@click.option(
    "--policy-config",
    type=str,
    default=None,
    help="JSON string with policy config overrides",
)
@click.option(
    "--env-config",
    type=str,
    default=None,
    help="JSON string with env config overrides",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Save results to JSON file",
)
@click.option(
    "--trace-db",
    default=None,
    help="SQLite/Turso URL for storing traces (set to 'none' to disable)",
)
@click.option(
    "--concurrency",
    type=int,
    default=4,
    help="Maximum concurrent task executions",
)
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Environment file(s) to load (for API keys, etc.)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def command(
    ctx: click.Context,
    baseline_id: str | None = None,
    split: str = "train",
    seeds: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    policy_config: str | None = None,
    env_config: str | None = None,
    output: str | None = None,
    trace_db: str | None = None,
    concurrency: int = 4,
    env_file: Sequence[str] = (),
    verbose: bool = False,
) -> None:
    """Run baseline evaluation."""
    # If a subcommand was invoked, don't run the default command
    if ctx.invoked_subcommand is not None:
        return
    
    # Check if baseline_id is actually a subcommand (shouldn't happen, but handle gracefully)
    if baseline_id and isinstance(ctx.command, click.Group) and baseline_id in ctx.command.commands:
        # It's a subcommand, re-invoke with that subcommand
        subcmd = ctx.command.get_command(ctx, baseline_id)
        if subcmd:
            return ctx.invoke(subcmd, **ctx.params)
    
    # baseline_id should be parsed by Click as a positional argument
    # No need to extract from meta since resolve_command returns None for non-subcommands
    
    # Run the evaluation
    asyncio.run(
        _baseline_command_impl(
            baseline_id=baseline_id,
            split=split,
            seeds=seeds,
            model=model,
            temperature=temperature,
            policy_config_json=policy_config,
            env_config_json=env_config,
            output_path=Path(output) if output else None,
            trace_db_url=trace_db,
            concurrency=concurrency,
            env_files=env_file,
            verbose=verbose,
        )
    )

@command.command("run")
@click.argument("baseline_id", type=str, required=False)
@click.option(
    "--split",
    default="train",
    help="Data split(s) to evaluate (comma-separated). Default: train",
)
@click.option(
    "--seeds",
    default=None,
    help="Comma-separated seeds to evaluate (overrides split defaults)",
)
@click.option(
    "--model",
    default=None,
    help="Model identifier (overrides default_policy_config)",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (overrides default_policy_config)",
)
@click.option(
    "--policy-config",
    type=str,
    default=None,
    help="JSON string with policy config overrides",
)
@click.option(
    "--env-config",
    type=str,
    default=None,
    help="JSON string with env config overrides",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Save results to JSON file",
)
@click.option(
    "--trace-db",
    default=None,
    help="SQLite/Turso URL for storing traces (set to 'none' to disable)",
)
@click.option(
    "--concurrency",
    type=int,
    default=4,
    help="Maximum concurrent task executions",
)
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Environment file(s) to load (for API keys, etc.)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def run_command(
    baseline_id: str | None,
    split: str,
    seeds: str | None,
    model: str | None,
    temperature: float | None,
    policy_config: str | None,
    env_config: str | None,
    output: str | None,
    trace_db: str | None,
    concurrency: int,
    env_file: Sequence[str],
    verbose: bool,
) -> None:
    """Run baseline evaluation."""
    asyncio.run(
        _baseline_command_impl(
            baseline_id=baseline_id,
            split=split,
            seeds=seeds,
            model=model,
            temperature=temperature,
            policy_config_json=policy_config,
            env_config_json=env_config,
            output_path=Path(output) if output else None,
            trace_db_url=trace_db,
            concurrency=concurrency,
            env_files=env_file,
            verbose=verbose,
        )
    )

async def _baseline_command_impl(
    baseline_id: str | None,
    split: str,
    seeds: str | None,
    model: str | None,
    temperature: float | None,
    policy_config_json: str | None,
    env_config_json: str | None,
    output_path: Path | None,
    trace_db_url: str | None,
    concurrency: int,
    env_files: Sequence[str],
    verbose: bool,
) -> None:
    """Implementation of baseline command."""
    
    # Load environment files if provided
    if env_files:
        try:
            from dotenv import load_dotenv
            for env_file in env_files:
                load_dotenv(env_file, override=False)
        except ImportError:
            click.echo("Warning: python-dotenv not installed, skipping --env-file", err=True)
    
    # 1. Discovery
    search_roots = [Path.cwd()]
    choices = discover_baseline_files(search_roots)
    
    if not choices:
        search_dirs = [str(root) for root in search_roots]
        raise click.ClickException(
            f"❌ No baseline files found\n"
            f"   Searched in: {', '.join(search_dirs)}\n"
            f"   Patterns: {', '.join(BASELINE_FILE_PATTERNS)}\n"
            f"   Create baseline files in:\n"
            f"     - examples/baseline/*.py\n"
            f"     - **/*_baseline.py (anywhere in the tree)\n"
            f"   Example: Create examples/baseline/my_task_baseline.py\n"
            f"   See: https://docs.usesynth.ai/baseline for more info"
        )
    
    if baseline_id is None:
        selected_id = _select_baseline_interactive(choices)
        if selected_id is None:
            raise click.ClickException(
                "❌ No baseline selected\n"
                "   Run with a baseline ID: synth-ai baseline <baseline_id>\n"
                "   Or use: synth-ai baseline list to see available baselines"
            )
        baseline_id = selected_id
    
    # Find matching baseline
    matching = [c for c in choices if c.baseline_id == baseline_id]
    if not matching:
        available = sorted({c.baseline_id for c in choices})
        # Find close matches (fuzzy matching)
        close_matches = [
            bid for bid in available
            if baseline_id.lower() in bid.lower() or bid.lower() in baseline_id.lower()
        ]
        
        error_msg = (
            f"❌ Baseline '{baseline_id}' not found\n"
            f"   Available baselines ({len(available)}): {', '.join(available)}"
        )
        
        if close_matches:
            error_msg += f"\n   Did you mean: {', '.join(close_matches[:3])}?"
        
        error_msg += "\n   Use 'synth-ai baseline list' to see all baselines with details"
        
        raise click.ClickException(error_msg)
    
    choice = matching[0]
    
    # 2. Load config
    try:
        config = load_baseline_config_from_file(baseline_id, choice.path)
    except ImportError as e:
        # ImportError already has good formatting from discovery.py
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        # ValueError already has good formatting from discovery.py
        raise click.ClickException(str(e)) from e
    except Exception as e:
        error_type = type(e).__name__
        raise click.ClickException(
            f"❌ Unexpected error loading baseline '{baseline_id}'\n"
            f"   File: {choice.path}\n"
            f"   Error: {error_type}: {str(e)}\n"
            f"   Tip: Run with --verbose for more details"
        ) from e
    
    # 3. Validate split
    split_names = _parse_splits(split)
    for split_name in split_names:
        if split_name not in config.splits:
            available_splits = sorted(config.splits.keys())
            raise click.ClickException(
                f"❌ Invalid split '{split_name}' for baseline '{baseline_id}'\n"
                f"   Available splits: {', '.join(available_splits)}\n"
                f"   Use: --split {available_splits[0]}  (or comma-separated: --split {','.join(available_splits)})"
            )
    
    # 4. Determine seeds
    if seeds:
        try:
            seed_list = _parse_seeds(seeds)
            if not seed_list:
                raise click.ClickException(
                    f"❌ No valid seeds provided\n"
                    f"   Provided: '{seeds}'\n"
                    f"   Expected: comma-separated integers (e.g., '0,1,2')"
                )
        except ValueError as e:
            raise click.ClickException(
                f"❌ Invalid seeds format\n"
                f"   Provided: '{seeds}'\n"
                f"   Expected: comma-separated integers (e.g., '0,1,2' or '10,20,30')\n"
                f"   Error: {str(e)}"
            ) from e
    else:
        # Use all seeds from specified splits
        seed_list = []
        for split_name in split_names:
            seed_list.extend(config.splits[split_name].seeds)
    
    if not seed_list:
        split_info = []
        for split_name in split_names:
            num_seeds = len(config.splits[split_name].seeds)
            split_info.append(f"{split_name} ({num_seeds} seeds)")
        
        raise click.ClickException(
            f"❌ No seeds found for split(s): {', '.join(split_names)}\n"
            f"   Split details: {', '.join(split_info)}\n"
            f"   This may indicate an empty split configuration\n"
            f"   Fix: Use --seeds to specify seeds manually (e.g., --seeds 0,1,2)"
        )
    
    # 5. Merge configs
    policy_config = {**config.default_policy_config}
    if model:
        policy_config["model"] = model
    if temperature is not None:
        policy_config["temperature"] = temperature
    if policy_config_json:
        try:
            policy_overrides = json.loads(policy_config_json)
            policy_config.update(policy_overrides)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f"❌ Invalid --policy-config JSON\n"
                f"   Provided: {policy_config_json[:100]}...\n"
                f"   Error: {str(e)}\n"
                f"   Expected: Valid JSON object (e.g., '{{\"model\": \"gpt-4o\", \"temperature\": 0.7}}')"
            ) from e
    
    env_config = {**config.default_env_config}
    if env_config_json:
        try:
            env_overrides = json.loads(env_config_json)
            env_config.update(env_overrides)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f"❌ Invalid --env-config JSON\n"
                f"   Provided: {env_config_json[:100]}...\n"
                f"   Error: {str(e)}\n"
                f"   Expected: Valid JSON object (e.g., '{{\"max_steps\": 1000}}')"
            ) from e
    
    # Handle split-specific env config
    for split_name in split_names:
        split_config = config.splits[split_name]
        if split_config.metadata:
            env_config.update(split_config.metadata)
    
    # 6. Setup trace storage (if requested)
    tracer = None
    if trace_db_url and trace_db_url.lower() not in {"none", "off"}:
        from synth_ai.core.tracing_v3.session_tracer import SessionTracer
        tracer = SessionTracer(db_url=trace_db_url, auto_save=True)
        await tracer.initialize()
    
    # 7. Execute tasks
    click.echo(f"Running {len(seed_list)} tasks across {len(split_names)} split(s)...")
    click.echo(f"Model: {policy_config.get('model', 'default')}")
    click.echo(f"Concurrency: {concurrency}")
    
    start_time = time.perf_counter()
    try:
        results = await run_baseline_evaluation(
            config=config,
            seeds=seed_list,
            policy_config=policy_config,
            env_config=env_config,
            concurrency=concurrency,
        )
    except Exception as e:
        error_type = type(e).__name__
        raise click.ClickException(
            f"❌ Error running baseline evaluation\n"
            f"   Baseline: {baseline_id}\n"
            f"   Tasks: {len(seed_list)} seeds\n"
            f"   Error: {error_type}: {str(e)}\n"
            f"   Common causes:\n"
            f"     - Missing dependencies (check baseline file imports)\n"
            f"     - API key not set (check environment variables)\n"
            f"     - Model/inference configuration issues\n"
            f"   Tip: Run with --verbose for detailed error output"
        ) from e
    
    elapsed = time.perf_counter() - start_time
    
    # Store traces if requested
    if tracer:
        for result in results:
            if result.trace:
                # Store trace (simplified - would need proper trace storage logic)
                pass
    
    # 8. Aggregate results
    aggregate_metrics = aggregate_results(config, results)
    
    # 9. Create output
    baseline_results = BaselineResults(
        config=config,
        split_name=",".join(split_names),
        results=results,
        aggregate_metrics=aggregate_metrics,
        execution_time_seconds=elapsed,
        model_name=policy_config.get("model", "unknown"),
        timestamp=datetime.now().isoformat(),
    )
    
    # 10. Display summary
    click.echo("\n" + "=" * 60)
    click.echo(f"Baseline Evaluation: {config.name}")
    click.echo("=" * 60)
    click.echo(f"Split(s): {baseline_results.split_name}")
    click.echo(f"Tasks: {len(results)}")
    click.echo(f"Success: {sum(1 for r in results if r.success)}/{len(results)}")
    click.echo(f"Execution time: {elapsed:.2f}s")
    click.echo("\nAggregate Metrics:")
    for key, value in aggregate_metrics.items():
        if isinstance(value, float):
            click.echo(f"  {key}: {value:.4f}")
        else:
            click.echo(f"  {key}: {value}")
    
    # 11. Save output if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(baseline_results.to_dict(), indent=2))
        click.echo(f"\nResults saved to: {output_path}")
