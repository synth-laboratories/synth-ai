from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any

import pytest
import synth_ai.dspy as dspy
from synth_ai.dspy import GEPA, MIPROv2
from synth_ai.dspy.miprov2 import MIPROv2DetailedResult
from synth_ai.gepa.core.result import GEPAResult


@dataclass
class _DummySignature:
    instructions: str


class _DummyStudent:
    def __init__(self, instructions: str) -> None:
        self.signature = _DummySignature(instructions)

    def deepcopy(self) -> _DummyStudent:
        return copy.deepcopy(self)


def test_synth_ai_dspy_import_paths_expose_drop_in_optimizers() -> None:
    synth_dspy = importlib.import_module("synth_ai.dspy")
    synth_dspy_gepa = importlib.import_module("synth_ai.dspy.gepa")
    synth_teleprompt = importlib.import_module("synth_ai.dspy.teleprompt")
    synth_teleprompt_gepa = importlib.import_module("synth_ai.dspy.teleprompt.gepa")
    synth_teleprompt_mipro = importlib.import_module(
        "synth_ai.dspy.teleprompt.mipro_optimizer_v2"
    )

    assert synth_dspy.GEPA is GEPA
    assert synth_dspy.MIPROv2 is MIPROv2
    assert synth_dspy_gepa.GEPA is GEPA
    assert synth_teleprompt.GEPA is GEPA
    assert synth_teleprompt.MIPROv2 is MIPROv2
    assert synth_teleprompt_gepa.GEPA is GEPA
    assert synth_teleprompt_mipro.MIPROv2 is MIPROv2


def test_synth_ai_gepa_package_exports_optimize() -> None:
    gepa_module = importlib.import_module("synth_ai.gepa")
    assert hasattr(gepa_module, "optimize")
    assert callable(gepa_module.optimize)


def test_gepa_wrapper_compile_invokes_synth_optimize(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}
    synthetic_result = GEPAResult(
        candidates=[{"system_prompt": "optimized prompt"}],
        parents=[[]],
        val_aggregate_scores=[0.91],
        val_subscores=[{}],
        per_val_instance_best_candidates={},
        discovery_eval_counts=[10],
        total_metric_calls=10,
        seed=7,
    )

    def _fake_optimize(**kwargs: Any) -> GEPAResult:
        called.update(kwargs)
        return synthetic_result

    monkeypatch.setattr("synth_ai.dspy.gepa.synth_gepa.optimize", _fake_optimize)

    optimizer = GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 1.0,
        max_metric_calls=10,
        reflection_lm="openai/gpt-5",
        track_stats=True,
        seed=7,
    )
    student = _DummyStudent("baseline prompt")
    optimized = optimizer.compile(
        student,
        trainset=[{"input": "hello", "answer": "world"}],
    )

    assert called["task_lm"] == "openai/gpt-4.1-mini"
    assert called["max_metric_calls"] == 10
    assert called["seed_candidate"] == {"system_prompt": "baseline prompt"}
    assert called["use_merge"] is False
    assert optimized is not student
    assert optimized.signature.instructions == "optimized prompt"
    assert optimized.detailed_results is synthetic_result


def test_miprov2_wrapper_compile_invokes_synth_runner(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    class _Status:
        value = "succeeded"

    @dataclass
    class _Result:
        job_id: str = "pl_test"
        status: Any = _Status()
        best_reward: float = 0.82
        best_candidate: dict[str, Any] = None  # type: ignore[assignment]
        raw: dict[str, Any] = None  # type: ignore[assignment]
        failed: bool = False
        error: str | None = None

    synthetic_result = _Result(
        best_candidate={"messages": [{"role": "system", "pattern": "mipro optimized prompt"}]},
        raw={"status": "succeeded"},
    )

    def _fake_runner(**kwargs: Any) -> _Result:
        called.update(kwargs)
        return synthetic_result

    monkeypatch.setattr("synth_ai.dspy.miprov2._run_synth_mipro", _fake_runner)

    optimizer = MIPROv2(
        metric=lambda gold, pred: 1.0,
        task_model="openai/gpt-4.1-mini",
        prompt_model="openai/gpt-4.1-mini",
        auto="light",
    )
    student = _DummyStudent("baseline prompt")
    optimized = optimizer.compile(
        student,
        trainset=[{"input": "hello", "answer": "world"}],
        minibatch_size=1,
    )

    assert called["task_lm"] == "openai/gpt-4.1-mini"
    assert called["proposer_lm"] == "openai/gpt-4.1-mini"
    assert called["max_rollouts"] == 7
    assert called["bootstrap_train_seeds"] == [0]
    assert called["online_pool"] == [0]
    assert called["validation_seeds"] == [0]
    assert optimized.signature.instructions == "mipro optimized prompt"

    details = optimized.detailed_results
    assert isinstance(details, MIPROv2DetailedResult)
    assert details.job_id == "pl_test"
    assert details.best_score == 0.82


def test_miprov2_auto_none_accepts_num_candidates(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    class _Status:
        value = "succeeded"

    @dataclass
    class _Result:
        job_id: str = "pl_test"
        status: Any = _Status()
        best_reward: float = 0.5
        best_candidate: dict[str, Any] = None  # type: ignore[assignment]
        raw: dict[str, Any] = None  # type: ignore[assignment]
        failed: bool = False
        error: str | None = None

    def _fake_runner(**kwargs: Any) -> _Result:
        called.update(kwargs)
        return _Result(
            best_candidate={"messages": [{"role": "system", "pattern": "ok"}]},
            raw={"status": "succeeded"},
        )

    monkeypatch.setattr("synth_ai.dspy.miprov2._run_synth_mipro", _fake_runner)

    optimizer = MIPROv2(
        metric=lambda gold, pred: 1.0,
        task_model="openai/gpt-4.1-mini",
        prompt_model="openai/gpt-4.1-mini",
        auto=None,
        num_candidates=8,
    )
    student = _DummyStudent("baseline prompt")
    optimized = optimizer.compile(
        student,
        trainset=[{"input": "hello", "answer": "world"}],
        minibatch_size=1,
    )

    assert called["max_rollouts"] >= 1
    assert optimized.signature.instructions == "ok"


def test_miprov2_minibatch_false_uses_full_online_pool(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    class _Status:
        value = "succeeded"

    @dataclass
    class _Result:
        job_id: str = "pl_test"
        status: Any = _Status()
        best_reward: float = 0.5
        best_candidate: dict[str, Any] = None  # type: ignore[assignment]
        raw: dict[str, Any] = None  # type: ignore[assignment]
        failed: bool = False
        error: str | None = None

    def _fake_runner(**kwargs: Any) -> _Result:
        called.update(kwargs)
        return _Result(
            best_candidate={"messages": [{"role": "system", "pattern": "ok"}]},
            raw={"status": "succeeded"},
        )

    monkeypatch.setattr("synth_ai.dspy.miprov2._run_synth_mipro", _fake_runner)

    optimizer = MIPROv2(
        metric=lambda gold, pred: 1.0,
        task_model="openai/gpt-4.1-mini",
        prompt_model="openai/gpt-4.1-mini",
        auto="light",
    )
    student = _DummyStudent("baseline prompt")
    optimizer.compile(
        student,
        trainset=[
            {"input": "hello1", "answer": "world1"},
            {"input": "hello2", "answer": "world2"},
            {"input": "hello3", "answer": "world3"},
        ],
        minibatch=False,
        minibatch_size=1,
    )
    assert called["online_pool"] == [0, 1, 2]


def test_miprov2_minibatch_false_ignores_default_minibatch_size(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    class _Status:
        value = "succeeded"

    @dataclass
    class _Result:
        job_id: str = "pl_test"
        status: Any = _Status()
        best_reward: float = 0.5
        best_candidate: dict[str, Any] = None  # type: ignore[assignment]
        raw: dict[str, Any] = None  # type: ignore[assignment]
        failed: bool = False
        error: str | None = None

    def _fake_runner(**kwargs: Any) -> _Result:
        called.update(kwargs)
        return _Result(
            best_candidate={"messages": [{"role": "system", "pattern": "ok"}]},
            raw={"status": "succeeded"},
        )

    monkeypatch.setattr("synth_ai.dspy.miprov2._run_synth_mipro", _fake_runner)

    optimizer = MIPROv2(
        metric=lambda gold, pred: 1.0,
        task_model="openai/gpt-4.1-mini",
        prompt_model="openai/gpt-4.1-mini",
        auto="light",
    )
    student = _DummyStudent("baseline prompt")
    optimizer.compile(
        student,
        trainset=[
            {"input": "hello1", "answer": "world1"},
            {"input": "hello2", "answer": "world2"},
            {"input": "hello3", "answer": "world3"},
        ],
        minibatch=False,
    )
    assert called["online_pool"] == [0, 1, 2]


def test_miprov2_auto_budget_yields_to_explicit_num_candidates(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    class _Status:
        value = "succeeded"

    @dataclass
    class _Result:
        job_id: str = "pl_test"
        status: Any = _Status()
        best_reward: float = 0.5
        best_candidate: dict[str, Any] = None  # type: ignore[assignment]
        raw: dict[str, Any] = None  # type: ignore[assignment]
        failed: bool = False
        error: str | None = None

    def _fake_runner(**kwargs: Any) -> _Result:
        called.update(kwargs)
        return _Result(
            best_candidate={"messages": [{"role": "system", "pattern": "ok"}]},
            raw={"status": "succeeded"},
        )

    monkeypatch.setattr("synth_ai.dspy.miprov2._run_synth_mipro", _fake_runner)

    optimizer = MIPROv2(
        metric=lambda gold, pred: 1.0,
        task_model="openai/gpt-4.1-mini",
        prompt_model="openai/gpt-4.1-mini",
        auto="light",
    )
    student = _DummyStudent("baseline prompt")
    with pytest.warns(RuntimeWarning, match="overrides auto"):
        optimizer.compile(
            student,
            trainset=[{"input": "hello", "answer": "world"}],
            num_candidates=8,
            minibatch_size=1,
        )

    assert called["max_rollouts"] != 7


def test_gepa_accepts_alias_kwargs_and_ignores_unknown(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}
    synthetic_result = GEPAResult(
        candidates=[{"system_prompt": "optimized prompt"}],
        parents=[[]],
        val_aggregate_scores=[0.91],
        val_subscores=[{}],
        per_val_instance_best_candidates={},
        discovery_eval_counts=[10],
        total_metric_calls=10,
        seed=11,
    )

    def _fake_optimize(**kwargs: Any) -> GEPAResult:
        called.update(kwargs)
        return synthetic_result

    monkeypatch.setattr("synth_ai.dspy.gepa.synth_gepa.optimize", _fake_optimize)

    with pytest.warns(RuntimeWarning, match="Ignoring unsupported GEPA init argument"):
        optimizer = GEPA(
            metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 1.0,
            max_metric_calls=10,
            reflection_lm="openai/gpt-5",
            task_model="openai/gpt-4.1-mini",
            unknown_init_arg=True,
        )

    student = _DummyStudent("baseline prompt")
    with pytest.warns(RuntimeWarning, match="Ignoring unsupported GEPA compile argument"):
        optimizer.compile(
            student,
            trainset=[{"input": "hello", "answer": "world"}],
            seed=11,
            unknown_compile_arg=True,
        )

    assert called["task_lm"] == "openai/gpt-4.1-mini"
    assert called["seed"] == 11


def test_dspy_configure_sets_default_lm() -> None:
    model = dspy.LM("openai/gpt-4.1-mini", temperature=0.1)
    dspy.configure(lm=model, max_errors=3)
    assert dspy.settings.lm is model
    assert dspy.settings.max_errors == 3
