#!/usr/bin/env python3
"""Execute the Crafter VLM GEPA demo notebook using papermill."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    """Execute the demo notebook using papermill."""
    import papermill as pm

    parser = argparse.ArgumentParser(description="Run Crafter VLM GEPA demo notebook")
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("BACKEND_BASE_URL", "https://api.usesynth.ai"),
        help="Backend URL (default: https://api.usesynth.ai)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SYNTH_API_KEY"),
        help="Synth API key (default: from SYNTH_API_KEY env var)",
    )
    parser.add_argument(
        "--policy-model",
        default="gpt-4.1-nano",
        help="VLM model for the agent (default: gpt-4.1-nano)",
    )
    parser.add_argument(
        "--verifier-model",
        default="gpt-5-nano",
        help="Model for verification (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=30,
        help="Total rollout budget (default: 30)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="Number of GEPA generations (default: 2)",
    )
    parser.add_argument(
        "--use-tunnel",
        action="store_true",
        default=True,
        help="Use cloudflared tunnels (required for production)",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Disable cloudflared tunnels (for local backend only)",
    )
    args = parser.parse_args()

    notebook_dir = Path(__file__).parent
    input_notebook = notebook_dir / "demo_prod.ipynb"
    output_notebook = notebook_dir / "demo_prod_executed.ipynb"

    use_tunnel = args.use_tunnel and not args.no_tunnel
    parameters = {
        "BACKEND_URL": args.backend_url,
        "POLICY_MODEL": args.policy_model,
        "VERIFIER_MODEL": args.verifier_model,
        "ROLLOUT_BUDGET": args.rollout_budget,
        "NUM_GENERATIONS": args.num_generations,
        "USE_TUNNEL": use_tunnel,
    }
    if args.api_key:
        parameters["API_KEY"] = args.api_key

    print(f"Running notebook: {input_notebook}")
    print(f"Backend: {args.backend_url}")
    print(f"Policy model: {args.policy_model}")
    print(f"Rollout budget: {args.rollout_budget}")
    print(f"Generations: {args.num_generations}")
    print(f"Use tunnel: {use_tunnel}")
    print()

    pm.execute_notebook(
        str(input_notebook),
        str(output_notebook),
        parameters=parameters,
        cwd=str(notebook_dir),
    )

    print(f"\nExecuted notebook saved to: {output_notebook}")
    print(f"Results saved to: {notebook_dir / 'results'}")


if __name__ == "__main__":
    main()
