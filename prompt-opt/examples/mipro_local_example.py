"""Local prompt-opt MIPRO example (shared Rust core)."""

from __future__ import annotations

from dataclasses import dataclass

from prompt_opt.dspy.miprov2 import MIPROv2


@dataclass
class _Sig:
    instructions: str

    def with_instructions(self, text: str) -> "_Sig":
        return _Sig(instructions=text)


@dataclass
class _Predictor:
    signature: _Sig


class _Student:
    def __init__(self) -> None:
        self._predictor = _Predictor(signature=_Sig(instructions="Answer briefly."))

    def deepcopy(self) -> "_Student":
        cloned = _Student()
        cloned._predictor = _Predictor(signature=_Sig(self._predictor.signature.instructions))
        return cloned

    def named_predictors(self):
        return [("main", self._predictor)]


def _task_llm(prompt: str) -> str:
    return "paris" if "capital of france" in prompt.lower() else "unknown"


def main() -> None:
    optimizer = MIPROv2(
        metric=lambda *_args, **_kwargs: 0.0,
        task_model=_task_llm,
        backend_mode="local",
        proposer_backend="rlm",
        auto="light",
        num_candidates=8,
    )
    student = _Student()
    optimized = optimizer.compile(
        student,
        trainset=[
            {"input": "Capital of France?", "answer": "paris"},
            {"input": "Capital of Italy?", "answer": "rome"},
        ],
        valset=[
            {"input": "Capital of France?", "answer": "paris"},
            {"input": "Capital of Italy?", "answer": "rome"},
        ],
    )
    print(optimized.named_predictors()[0][1].signature.instructions)


if __name__ == "__main__":
    main()
