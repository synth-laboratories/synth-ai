import importlib

import pytest

from synth_ai.sdk.task.contracts import RolloutMode


def _load_grpo_module():
    from synth_ai.sdk.task.apps import registry

    registry.clear()
    return importlib.import_module("examples.task_apps.crafter.task_app.grpo_crafter")


def _load_utils_module():
    return importlib.import_module("examples.task_apps.crafter.task_app.synth_envs_hosted.utils")


def test_extract_trace_correlation_id_from_modal_url():
    utils_mod = _load_utils_module()
    extractor = getattr(utils_mod, "extract_trace_correlation_id")
    url = (
        "https://synth-labs--crafter.modal.run/v1/chat/completions"
        "?cid=trace_run-1234abcd&foo=bar"
    )

    assert extractor(url, mode=RolloutMode.RL) == "trace_run-1234abcd"


@pytest.mark.parametrize("mode", [RolloutMode.RL, "rl"])
def test_resolve_trace_correlation_id_uses_inference_url(mode):
    grpo_mod = _load_grpo_module()
    resolver = getattr(grpo_mod, "_resolve_trace_correlation_id")
    policy_cfg = {
        "model": "Qwen/Qwen3-4B",
        "inference_url": (
            "https://ta-01k8skyc1cgtrtep1avyyqhs3s-8000.modal.host"
            "?cid=trace_run-e2f1b3da"
        ),
    }

    assert resolver(policy_cfg, mode=mode) == "trace_run-e2f1b3da"
