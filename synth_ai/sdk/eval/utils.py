from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from .job import EvalJob, EvalJobConfig, EvalResult


def mean_reward_or_zero(results: Iterable[EvalResult | None]) -> float:
    values = []
    for result in results:
        value = result.mean_reward if result and result.mean_reward is not None else 0.0
        values.append(value)
    return sum(values) / max(len(values), 1)


def run_eval_slices(
    label: str,
    instruction: str,
    seed_slices: Sequence[Sequence[int]],
    *,
    task_app_url: str,
    base_config: dict,
    policy_config: dict,
    timeout: float = 120.0,
    interval: float = 1.0,
    parallel: bool = False,
    max_workers: int | None = None,
    progress: bool = True,
    status_interval: float = 15.0,
    debug: bool = False,
) -> list[EvalResult]:
    from synth_ai.sdk.optimization.progress.handlers import EvalStatusPrinter

    def _instruction_signature(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return digest[:12]

    def _instruction_preview(text: str, limit: int = 140) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    total_seeds = sum(len(s) for s in seed_slices)
    printer: EvalStatusPrinter | None = None
    if progress:
        printer = EvalStatusPrinter(label=label, total_seeds=total_seeds, debug=debug)
        printer.log_start(total=total_seeds)
        if debug:
            preview = _instruction_preview(instruction)
            signature = _instruction_signature(instruction)
            model = policy_config.get("model")
            provider = policy_config.get("provider")
            env_name = base_config.get("env_name")
            printer.log_debug_config(
                "Eval config: "
                f"label={label} "
                f"instruction_sig={signature} "
                f"model={model} "
                f"provider={provider} "
                f"env={env_name} "
                f"seeds={total_seeds} "
                f"task_app_url={task_app_url} "
                f"instruction_preview={preview}"
            )

    jobs: list[tuple[int, EvalJob]] = []
    for index, seeds in enumerate(seed_slices, start=1):
        eval_job = EvalJob(
            EvalJobConfig(
                task_app_url=task_app_url,
                policy_config={
                    **policy_config,
                    "instruction": instruction,
                },
                seeds=list(seeds),
                **base_config,
            )
        )
        eval_job.submit()
        if debug and printer:
            signature = _instruction_signature(instruction)
            printer.log_debug_config(
                "Eval job submitted: "
                f"label={label} slice={index}/{len(seed_slices)} "
                f"job_id={eval_job.job_id} instruction_sig={signature}"
            )
        jobs.append((index, eval_job))

    def poll(entry: tuple[int, EvalJob]) -> tuple[int, EvalResult]:
        index, job = entry
        result = job.poll_until_complete(
            timeout=timeout,
            interval=interval,
            progress_label=None,  # suppress per-slice progress
            on_status=printer.handle_status if printer else None,
        )
        return index, result

    if not parallel:
        results: list[EvalResult] = []
        for entry in jobs:
            _, result = poll(entry)
            results.append(result)
    else:
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        results_by_index: dict[int, EvalResult] = {}
        worker_count = max_workers or len(jobs)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(poll, entry) for entry in jobs}
            while futures:
                done, futures = wait(futures, timeout=status_interval, return_when=FIRST_COMPLETED)
                for future in done:
                    index, result = future.result()
                    results_by_index[index] = result
                if printer and futures:
                    printer.tick(min_idle_seconds=status_interval)
        results = [results_by_index[index] for index in sorted(results_by_index)]

    if printer:
        mean_reward = mean_reward_or_zero(results)
        printer.log_terminal(status="completed", mean_reward=mean_reward)

    return results
