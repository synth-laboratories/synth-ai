#!/usr/bin/env python3
"""
Script to run Crafter evaluation using the standardized eval framework
"""

import asyncio
from pathlib import Path

import toml
from src.synth_env.examples.crafter_classic.agent_demos.eval_framework import (
    CrafterEvalFramework,
    run_crafter_eval,
)


async def main():
    # Load configuration
    config_path = Path(__file__).parent / "eval_config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = toml.load(config_path)
    eval_config = config["evaluation"]

    models = eval_config["models"]
    difficulties = eval_config["difficulties"]
    max_turns = eval_config["max_turns"]
    n_trajectories = eval_config["trajectories_per_condition"]

    print("🎯 Crafter Multi-Action Model Comparison (Eval Framework)")
    print("=" * 60)
    print(f"Models: {', '.join(models)}")
    print(f"Difficulties: {', '.join(difficulties)}")
    print(f"Max turns: {max_turns}")
    print(f"Trajectories per condition: {n_trajectories}")
    print("=" * 60)

    # Run evaluation using the framework
    results = await run_crafter_eval(
        model_names=models,
        difficulties=difficulties,
        num_trajectories=n_trajectories,
        max_turns=max_turns,
    )

    # The framework already prints detailed reports
    print("\n🏆 Evaluation completed!")
    return results


if __name__ == "__main__":
    asyncio.run(main())
