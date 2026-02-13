from __future__ import annotations

import asyncio

from synth_ai.sdk.optimization.internal import graph_evolve_streaming, prompt_learning_streaming
from synth_ai.sdk.optimization.internal.graph_evolve_builder import build_placeholder_dataset
from synth_ai.sdk.optimization.internal.graphgen import GraphEvolveJob
from synth_ai.sdk.optimization.internal.graphgen_models import (
    GraphGenJobConfig as GraphEvolveJobConfig,
)
from synth_ai.sdk.optimization.internal.prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
)


class _DummyStreamer:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def stream_until_terminal(self) -> dict:
        return self._payload


def test_prompt_learning_streaming_async(monkeypatch) -> None:
    streamer = _DummyStreamer({"status": "succeeded", "best_score": 0.9})

    monkeypatch.setattr(
        prompt_learning_streaming,
        "build_prompt_learning_streamer",
        lambda **_kwargs: streamer,
    )

    config = PromptLearningJobConfig(
        config_dict={
            "prompt_learning": {
                "algorithm": "gepa",
                "container_url": "http://example.com",
                "policy": {
                    "inference_mode": "synth_hosted",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                },
                "gepa": {
                    "evaluation": {"train_seeds": list(range(70)), "val_seeds": list(range(70, 80))},
                    "archive": {"pareto_set_size": 10},
                },
            }
        },
        backend_url="http://example.com",
        api_key="key",
        container_api_key="local",
    )
    job = PromptLearningJob(config, job_id="pl_test", skip_health_check=True)

    # Pass handlers=[] to bypass the Rust-native streaming path and use the
    # Python streamer that we monkeypatched above.
    result = asyncio.run(job.stream_until_complete_async(timeout=0.1, handlers=[]))
    assert result.succeeded
    assert result.best_reward == 0.9


def test_graph_evolve_streaming_async(monkeypatch) -> None:
    streamer = _DummyStreamer({"status": "succeeded"})

    monkeypatch.setattr(
        graph_evolve_streaming,
        "build_graph_evolve_streamer",
        lambda **_kwargs: streamer,
    )

    dataset = build_placeholder_dataset()
    config = GraphEvolveJobConfig(policy_models=["gpt-4o-mini"])
    job = GraphEvolveJob(
        dataset=dataset,
        config=config,
        backend_url="http://example.com",
        api_key="key",
        auto_start=False,
    )
    job._graph_evolve_job_id = "graph_evolve_test"

    result = asyncio.run(job.stream_until_complete_async(timeout=0.1))
    assert result["status"] == "succeeded"
