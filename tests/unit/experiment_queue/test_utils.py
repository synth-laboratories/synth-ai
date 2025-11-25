from __future__ import annotations

import json
from pathlib import Path

import tomllib

from synth_ai.cli.local.experiment_queue.config_utils import prepare_config_file
from synth_ai.cli.local.experiment_queue.results import collect_result_summary


def test_prepare_config_file_resolves_paths_and_overrides(tmp_path):
    config_path = tmp_path / "source.toml"
    config_path.write_text(
        "[prompt_learning]\nresults_folder = \"rel_results\"\nenv_file_path = \"../.env\"\n",
        encoding="utf-8",
    )

    prepared = prepare_config_file(
        config_path,
        overrides={
            "prompt_learning": {
                "termination_config": {"max_rollouts": 42},
            }
        },
    )

    with open(prepared.path, "rb") as handle:
        data = tomllib.load(handle)

    results_folder = Path(data["prompt_learning"]["results_folder"])
    assert results_folder.is_absolute()
    assert results_folder.exists()
    assert data["prompt_learning"]["termination_config"]["max_rollouts"] == 42

    env_path = Path(data["prompt_learning"]["env_file_path"])
    assert env_path.is_absolute()

    prepared.cleanup()
    assert not prepared.workdir.exists()


def test_collect_result_summary_reads_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    curve = {
        "curve": [
            {"rollout_count": 1, "performance": 0.2},
            {"rollout_count": 3, "performance": 0.6},
        ]
    }
    stats = {"total_time": 15.5, "total_rollouts": 3}

    (results_dir / "run_learning_curve.json").write_text(json.dumps(curve), encoding="utf-8")
    (results_dir / "run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (results_dir / "run_best_prompt.json").write_text("{}", encoding="utf-8")

    summary = collect_result_summary(results_dir)
    assert summary.best_score == 0.6
    assert summary.baseline_score == 0.2
    assert summary.total_rollouts == 3
    assert summary.total_time == 15.5
    assert "learning_curve_path" in summary.artifacts
    assert "stats_path" in summary.artifacts
