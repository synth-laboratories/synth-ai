"""Eval command."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import click
from dotenv import load_dotenv


@click.command()
@click.argument("app_id", required=False)
@click.option("--model", required=False, default="")
@click.option("--config", "config_path", required=False, default="")
@click.option("--trace-db", required=False, default="")
@click.option("--metadata", multiple=True)
@click.option("--seeds", required=False, default="")
@click.option("--url", required=False, default="")
@click.option("--backend", required=False, default="")
@click.option("--env-file", required=False, default="")
@click.option("--ops", required=False, default="")
@click.option("--return-trace", is_flag=True, default=False)
@click.option("--concurrency", required=False, default="")
@click.option(
    "--seed-set", type=click.Choice(["seeds", "validation_seeds", "test_pool"]), default="seeds"
)
@click.option("--wait", is_flag=True, default=False)
@click.option("--poll", required=False, default="")
@click.option("--output", "output_path", required=False, default="")
@click.option("--traces-dir", required=False, default="")
@click.option("--output-txt", required=False, default="")
@click.option("--output-json", required=False, default="")
def eval(
    app_id: str | None,
    model: str,
    config_path: str,
    trace_db: str,
    metadata: tuple[str, ...],
    seeds: str,
    url: str,
    backend: str,
    env_file: str,
    ops: str,
    return_trace: bool,
    concurrency: str,
    seed_set: str,
    wait: bool,
    poll: str,
    output_path: str,
    traces_dir: str,
    output_txt: str,
    output_json: str,
) -> None:
    """Execute evaluation rollouts against a task app."""
    from synth_ai.core.eval import (
        format_eval_report,
        format_eval_table,
        resolve_eval_config,
        run_eval,
        save_traces,
        validate_eval_options,
    )

    config_file = Path(config_path) if config_path else None
    if config_file and not config_file.exists():
        raise click.ClickException("Eval config not found")

    options = {
        "app_id": app_id or "",
        "model": model,
        "config": config_path,
        "trace_db": trace_db,
        "metadata": list(metadata),
        "seeds": seeds,
        "url": url,
        "backend": backend,
        "env_file": env_file,
        "ops": ops,
        "return_trace": return_trace,
        "concurrency": concurrency,
        "poll": poll,
    }
    try:
        normalized = validate_eval_options(options)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if env_file:
        load_dotenv(env_file, override=False)

    output_json_path = output_path or output_json
    effective_return_trace = return_trace or bool(traces_dir)

    resolved = resolve_eval_config(
        config_path=config_file,
        cli_app_id=str(normalized.get("app_id") or "") or None,
        cli_model=str(normalized.get("model") or "") or None,
        cli_seeds=normalized.get("seeds") or None,
        cli_url=str(normalized.get("url") or "") or None,
        cli_env_file=str(normalized.get("env_file") or "") or None,
        cli_ops=normalized.get("ops") or None,
        cli_return_trace=effective_return_trace,
        cli_concurrency=normalized.get("concurrency") or None,
        cli_output_txt=Path(output_txt) if output_txt else None,
        cli_output_json=Path(output_json_path) if output_json_path else None,
        cli_backend_url=str(normalized.get("backend") or "") or None,
        cli_wait=bool(wait),
        cli_poll_interval=float(normalized.get("poll") or 0) if normalized.get("poll") else None,
        cli_traces_dir=Path(traces_dir) if traces_dir else None,
        seed_set=seed_set,
        metadata=normalized.get("metadata") or {},
    )

    if not resolved.task_app_url:
        raise click.ClickException("task_app_url is required (provide via TOML or --url)")
    if not resolved.env_name:
        raise click.ClickException("env_name is required (provide via TOML)")
    if not resolved.seeds:
        raise click.ClickException("No seeds found (provide via TOML or --seeds)")
    if not resolved.policy_config.get("model") and not model:
        raise click.ClickException("policy model is required (set in TOML or --model)")

    results = asyncio.run(run_eval(resolved))
    if results:
        table = format_eval_table(results)
        click.echo(table)

        report = format_eval_report(resolved, results)
        if resolved.output_txt:
            resolved.output_txt.write_text(report, encoding="utf-8")
            click.echo(f"\nWrote report: {resolved.output_txt}")
        if resolved.output_json:
            results_data = []
            for result in results:
                result_dict = asdict(result)
                result_dict.pop("trace", None)
                results_data.append(result_dict)
            payload = {
                "config": {
                    "app_id": resolved.app_id,
                    "task_app_url": resolved.task_app_url,
                    "env_name": resolved.env_name,
                    "policy_name": resolved.policy_name,
                    "policy_config": resolved.policy_config,
                    "seeds": resolved.seeds,
                    "ops": resolved.ops,
                    "concurrency": resolved.concurrency,
                },
                "results": results_data,
            }
            resolved.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            click.echo(f"Wrote JSON report: {resolved.output_json}")

        if traces_dir:
            saved_count = save_traces(results, traces_dir)
            click.echo(f"Saved {saved_count} traces to {traces_dir}")
