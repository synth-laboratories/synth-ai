from __future__ import annotations

from synth_ai.core.levers import Lever, MiproLeverSummary
from synth_ai.core.sensors import SensorFrameSummary
from synth_ai.sdk.optimization.models import PolicyOptimizationResult


def test_mipro_lever_summary_from_dict() -> None:
    summary = MiproLeverSummary.from_dict(
        {
            "prompt_lever_id": "mipro.prompt.sys",
            "candidate_lever_versions": {"baseline": 1, "cand_1": 2},
            "best_candidate_id": "cand_1",
            "baseline_candidate_id": "baseline",
            "lever_count": 2,
            "mutation_count": 1,
            "latest_version": 2,
        }
    )
    assert summary is not None
    assert summary.prompt_lever_id == "mipro.prompt.sys"
    assert summary.candidate_lever_versions["baseline"] == 1
    assert summary.best_candidate_id == "cand_1"
    assert summary.latest_version == 2


def test_sensor_frame_summary_from_dict() -> None:
    frame = SensorFrameSummary.from_dict(
        {
            "frame_id": "frame_123",
            "created_at": "2026-02-11T12:00:00Z",
            "sensor_count": 3,
            "sensor_kinds": ["reward", "resource"],
            "trace_ids": ["trace_abc"],
            "lever_versions": {"mipro.prompt.sys": 2},
        }
    )
    assert frame is not None
    assert frame.frame_id == "frame_123"
    assert frame.sensor_count == 3
    assert frame.lever_versions["mipro.prompt.sys"] == 2


def test_policy_optimization_result_typed_properties() -> None:
    data = {
        "status": "succeeded",
        "best_reward": 0.5,
        "best_candidate": {"candidate_id": "cand_1"},
        "lever_summary": {
            "prompt_lever_id": "mipro.prompt.sys",
            "candidate_lever_versions": {"baseline": 1, "cand_1": 2},
            "best_candidate_id": "cand_1",
            "baseline_candidate_id": "baseline",
            "lever_count": 2,
            "mutation_count": 1,
            "latest_version": 2,
        },
        "sensor_frames": [
            {
                "frame_id": "frame_123",
                "created_at": "2026-02-11T12:00:00Z",
                "sensor_count": 3,
                "sensor_kinds": ["reward"],
                "trace_ids": ["trace_abc"],
                "lever_versions": {"mipro.prompt.sys": 2},
            }
        ],
    }

    result = PolicyOptimizationResult.from_response("pl_test", data, algorithm="mipro")
    typed = result.lever_summary_typed
    assert typed is not None
    assert typed.prompt_lever_id == "mipro.prompt.sys"

    frames = result.sensor_frame_summaries_typed
    assert len(frames) == 1
    assert frames[0].frame_id == "frame_123"


def test_lever_from_dict_defaults_unknown_enums() -> None:
    lever = Lever.from_dict(
        {
            "lever_id": "lever_1",
            "kind": "not_a_kind",
            "scope": [{"kind": "org", "id": "org_1"}],
            "value": {"hello": "world"},
            "value_format": "not_a_format",
            "mutability": "not_a_mutability",
            "version": "2",
        }
    )
    assert lever.kind.value == "custom"
    assert lever.value_format.value == "custom"
    assert lever.mutability.value == "optimizer"
    assert lever.version == 2

