from __future__ import annotations

from synth_ai.sdk.api.train.progress.dataclasses import BaselineInfo, CandidateInfo


def test_candidate_info_instance_objectives_parsed() -> None:
    data = {
        "version_id": "cand-1",
        "objectives": {"reward": 0.9},
        "instance_objectives": [
            {"objectives": {"reward": 0.1}},
            {"objectives": {"reward": 0.2}},
        ],
    }
    candidate = CandidateInfo.from_event_data(data)
    assert candidate.objectives == {"reward": 0.9}
    assert candidate.accuracy == 0.9
    assert candidate.instance_scores == [0.1, 0.2]
    assert candidate.instance_objectives == data["instance_objectives"]


def test_baseline_info_instance_objectives_fallback() -> None:
    data = {
        "accuracy": 0.5,
        "instance_objectives": [{"objectives": {"reward": 0.5}}],
    }
    baseline = BaselineInfo.from_event_data(data)
    assert baseline.objectives == {"reward": 0.5}
    assert baseline.instance_scores == [0.5]
    assert baseline.instance_objectives == data["instance_objectives"]
