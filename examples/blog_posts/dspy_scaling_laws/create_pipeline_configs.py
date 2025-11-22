"""Script to generate all TOML configuration files for the scaling experiment."""

from pathlib import Path
import toml

# Base configuration template
def create_gepa_config(benchmark, steps, train_seeds, val_seeds, rollout_budget=200):
    """Create GEPA configuration for a benchmark and pipeline complexity."""

    # Task app mapping
    task_app_map = {
        ("banking77", 1): "banking77",
        ("banking77", 2): "banking77-pipeline",
        ("banking77", 3): "banking77-3step",
        ("banking77", 5): "banking77-5step",
        ("heartdisease", 1): "heartdisease",
        ("heartdisease", 3): "heartdisease-3step",
        ("heartdisease", 5): "heartdisease-5step",
        ("hotpotqa", 1): "hotpotqa",
        ("hotpotqa", 3): "hotpotqa-3step",
        ("hotpotqa", 5): "hotpotqa-5step",
    }

    task_app = task_app_map.get((benchmark, steps), f"{benchmark}-{steps}step")

    config = {
        "optimizer": {
            "type": "gepa",
            "params": {
                "max_metric_calls": rollout_budget,
                "reflection_minibatch_size": 3,
                "track_stats": True,
            }
        },
        "task": {
            "name": f"{benchmark}_{steps}step_gepa",
            "task_app": task_app,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
        },
        "model": {
            "provider": "groq",
            "name": "llama-3.1-8b-instant",
        },
        "output": {
            "dir": f"results/{benchmark}/{steps}step/gepa",
        }
    }

    return config


def create_mipro_config(benchmark, steps, train_seeds, val_seeds, rollout_budget=200):
    """Create MIPRO configuration for a benchmark and pipeline complexity."""

    task_app_map = {
        ("banking77", 1): "banking77",
        ("banking77", 2): "banking77-pipeline",
        ("banking77", 3): "banking77-3step",
        ("banking77", 5): "banking77-5step",
        ("heartdisease", 1): "heartdisease",
        ("heartdisease", 3): "heartdisease-3step",
        ("heartdisease", 5): "heartdisease-5step",
        ("hotpotqa", 1): "hotpotqa",
        ("hotpotqa", 3): "hotpotqa-3step",
        ("hotpotqa", 5): "hotpotqa-5step",
    }

    task_app = task_app_map.get((benchmark, steps), f"{benchmark}-{steps}step")

    config = {
        "optimizer": {
            "type": "mipro",
            "params": {
                "num_candidates": 20,
                "num_trials": 10,
                "max_bootstrapped_demos": 10,
                "max_labeled_demos": 10,
            }
        },
        "task": {
            "name": f"{benchmark}_{steps}step_mipro",
            "task_app": task_app,
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
        },
        "model": {
            "provider": "groq",
            "name": "gpt-oss-20b",
        },
        "output": {
            "dir": f"results/{benchmark}/{steps}step/mipro",
        }
    }

    return config


def main():
    """Generate all 24 TOML configuration files."""

    # Define benchmarks and their train/val splits
    benchmark_configs = {
        "banking77": {
            "train_seeds": list(range(50)),  # 0-49
            "val_seeds": list(range(50, 250)),  # 50-249
        },
        "heartdisease": {
            "train_seeds": list(range(25)),  # 0-24
            "val_seeds": list(range(50, 150)),  # 50-149
        },
        "hotpotqa": {
            "train_seeds": list(range(25)),  # 0-24
            "val_seeds": list(range(50, 150)),  # 50-149
        },
    }

    # Pipeline complexities
    pipeline_steps = [1, 2, 3, 5]

    # Create configs directory
    base_dir = Path(__file__).parent

    for benchmark, seeds in benchmark_configs.items():
        for steps in pipeline_steps:
            # Skip 2-step for non-banking77 (not implemented)
            if steps == 2 and benchmark != "banking77":
                continue

            # Create GEPA config
            gepa_config = create_gepa_config(
                benchmark, steps,
                seeds["train_seeds"],
                seeds["val_seeds"]
            )

            gepa_path = base_dir / f"benchmarks/{benchmark}/configs/gepa_{steps}step.toml"
            gepa_path.parent.mkdir(parents=True, exist_ok=True)

            with open(gepa_path, "w") as f:
                toml.dump(gepa_config, f)

            print(f"✓ Created {gepa_path}")

            # Create MIPRO config
            mipro_config = create_mipro_config(
                benchmark, steps,
                seeds["train_seeds"],
                seeds["val_seeds"]
            )

            mipro_path = base_dir / f"benchmarks/{benchmark}/configs/mipro_{steps}step.toml"
            mipro_path.parent.mkdir(parents=True, exist_ok=True)

            with open(mipro_path, "w") as f:
                toml.dump(mipro_config, f)

            print(f"✓ Created {mipro_path}")

    print(f"\n✅ Generated all configuration files!")


if __name__ == "__main__":
    main()
