"""Run command for Harbor rollouts."""

import click
import httpx


@click.command()
@click.argument("deployment_id")
@click.option(
    "--seeds",
    "-s",
    type=int,
    default=10,
    help="Number of seeds to run (0..seeds-1)",
)
@click.option(
    "--seed",
    type=int,
    multiple=True,
    help="Specific seed(s) to run",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o-mini",
    help="Model to use for rollouts",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Timeout per rollout in seconds",
)
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    help="Synth API key",
)
@click.option(
    "--backend-url",
    envvar="SYNTH_BACKEND_URL",
    default="https://api.usesynth.ai",
    help="Synth backend URL",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def run(
    deployment_id: str,
    seeds: int,
    seed: tuple[int, ...],
    model: str,
    timeout: int,
    api_key: str | None,
    backend_url: str,
    output_json: bool,
):
    """Run rollouts on a Harbor deployment.

    Example:
        synth harbor run abc-123 --seeds 10
        synth harbor run abc-123 --seed 0 --seed 5 --seed 10
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    # Determine which seeds to run
    seed_list = list(seed) if seed else list(range(seeds))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    results = []

    click.echo(f"Running {len(seed_list)} rollout(s) on deployment {deployment_id}...")
    click.echo(f"  Model: {model}")
    click.echo(f"  Timeout: {timeout}s")
    click.echo()

    with httpx.Client(timeout=float(timeout) + 60) as client:
        for i, s in enumerate(seed_list):
            if not output_json:
                click.echo(f"  [{i + 1}/{len(seed_list)}] Seed {s}...", nl=False)

            try:
                # Build rollout request in TaskApp format
                run_id = f"harbor-cli-{deployment_id[:8]}"
                trace_id = f"{run_id}-s{s}"
                payload = {
                    "run_id": run_id,
                    "trace_correlation_id": trace_id,
                    "env": {
                        "seed": s,
                        "env_name": "harbor",
                        "config": {},
                    },
                    "policy": {
                        "config": {
                            "model": model,
                            "inference_url": "https://api.openai.com/v1",
                            "provider": "openai",
                        },
                    },
                }

                response = client.post(
                    f"{backend_url}/api/harbor/deployments/{deployment_id}/rollout",
                    json=payload,
                    headers=headers,
                )

                if response.status_code >= 400:
                    if not output_json:
                        click.echo(click.style(" FAILED", fg="red"))
                        click.echo(f"      Error: {response.text[:100]}")
                    results.append(
                        {
                            "seed": s,
                            "success": False,
                            "error": response.text,
                        }
                    )
                    continue

                data = response.json()
                reward = 0.0

                # Extract reward from response
                if "reward_info" in data:
                    reward = data["reward_info"].get("outcome_reward", 0.0)
                elif "metrics" in data:
                    reward = data["metrics"].get("reward_mean", 0.0)

                if not output_json:
                    color = "green" if reward > 0.5 else "yellow" if reward > 0 else "red"
                    click.echo(click.style(f" reward={reward:.3f}", fg=color))

                results.append(
                    {
                        "seed": s,
                        "success": True,
                        "reward": reward,
                        "response": data,
                    }
                )

            except httpx.TimeoutException:
                if not output_json:
                    click.echo(click.style(" TIMEOUT", fg="red"))
                results.append(
                    {
                        "seed": s,
                        "success": False,
                        "error": "timeout",
                    }
                )

            except Exception as e:
                if not output_json:
                    click.echo(click.style(f" ERROR: {e}", fg="red"))
                results.append(
                    {
                        "seed": s,
                        "success": False,
                        "error": str(e),
                    }
                )

    # Summary
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if output_json:
        import json

        click.echo(
            json.dumps(
                {
                    "deployment_id": deployment_id,
                    "total": len(results),
                    "successful": len(successful),
                    "failed": len(failed),
                    "mean_reward": sum(r.get("reward", 0) for r in successful) / len(successful)
                    if successful
                    else 0,
                    "results": results,
                },
                indent=2,
                default=str,
            )
        )
    else:
        click.echo()
        click.echo("Summary:")
        click.echo(f"  Total: {len(results)}")
        click.echo(f"  Successful: {click.style(str(len(successful)), fg='green')}")
        click.echo(f"  Failed: {click.style(str(len(failed)), fg='red' if failed else 'green')}")

        if successful:
            rewards = [r.get("reward", 0) for r in successful]
            click.echo(f"  Mean reward: {sum(rewards) / len(rewards):.3f}")
            click.echo(f"  Min reward: {min(rewards):.3f}")
            click.echo(f"  Max reward: {max(rewards):.3f}")
