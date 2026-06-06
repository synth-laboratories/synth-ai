#!/usr/bin/env python3
"""Run Crafter code-policy DEO hillclimb (1 candidate) via the synth-ai Research SDK.

Edit ``CrafterDeoRunConfig`` below — project notes, worker instructions, seeds,
models, and timeboxes all live in this file. Runnable code still comes from
``crafter_runs/lane/`` (policy, sweep, hillclimb runner).

```bash
cd ~/Documents/GitHub/synth-ai
uv sync --group dev
uv run python crafter_runs/crafter_deo_run.py --use-default-slot1
# or: bash scripts/run_crafter_deo_hillclimb_1cand_slot1.sh
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from synth_ai.research import ResearchApiError, ResearchClient, ResearchWorkMode

# Import shared slot/backend helpers from the README smoke driver.
from readme_runs.readme_smoke import (
    ReadmeSmokeLaunch,
    _apply_slot_trigger_overrides,
    _ensure_evals_importable,
    _resolve_evals_root,
    build_research_client,
    resolve_readme_smoke_launch,
)

LogFn = Callable[[str], None]

SYNTH_AI_ROOT = Path(__file__).resolve().parent.parent
LANE_ROOT = Path(__file__).resolve().parent / "lane"
TASK_TOML = LANE_ROOT / "task.toml"
TASK_ID = "reportbench/crafter_code_policy_deo_hillclimb_1cand"
DEFAULT_CRAFTER_IMAGE = "synth-local-open-research-crafter:latest"


# ---------------------------------------------------------------------------
# Edit run setup here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrafterDeoRunConfig:
    """Lane text + launch knobs. Code artifacts stay under ``crafter_runs/lane/``."""

    parallel_worker_count: int = 3
    candidate_ids: tuple[str, ...] = ("attempt_1", "attempt_2", "attempt_3")

    train_seeds: tuple[int, ...] = (
        101,
        103,
        105,
        107,
        109,
        111,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
    )

    orchestrator_profile_id: str = "codex_gpt_5_4_mini_medium"
    worker_profile_id: str = "codex_gpt_5_4_mini_high"

    run_timebox_seconds: int = 3600
    poll_timebox_seconds: int = 7200
    setup_retry_timebox_seconds: int = 180
    launch_retry_timebox_seconds: int = 3600

    crafter_docker_image: str = DEFAULT_CRAFTER_IMAGE

    extra_worker_instructions: str = ""
    extra_orchestrator_instructions: str = ""

    def train_seeds_csv(self) -> str:
        return ",".join(str(seed) for seed in self.train_seeds)

    def train_seeds_cli(self) -> str:
        return f"crafter={self.train_seeds_csv()}"

    def candidate_dir(self, candidate_id: str) -> str:
        return f"candidates/crafter/{candidate_id}"

    def worker_task_key(self, candidate_id: str) -> str:
        return f"hillclimb_crafter_deo_{candidate_id}"

    def project_notes(self, *, worker_pool_id: str) -> str:
        candidate_lines = "\n".join(
            f"  - task_key `{self.worker_task_key(candidate_id)}` → "
            f"`{self.candidate_dir(candidate_id)}/heuristic_policy.py` "
            f"(task_affinity_key `crafter-{candidate_id}`)"
            for candidate_id in self.candidate_ids
        )
        extra = self.extra_orchestrator_instructions.strip()
        extra_block = f"\n\n{extra}" if extra else ""
        return f"""
REPORT BENCH - Crafter code-policy DEO (parallel worker race).

You are the orchestrator. Create the directed-effort objective and milestones,
then run a PARALLEL CANDIDATE RACE:

1. Use ONE `plan_tasks` call to assign {self.parallel_worker_count} repo tasks to
   worker_host actors on worker pool `{worker_pool_id}` AT THE SAME TIME. Do not
   assign serially. Each planned task must include:
   `task_dispatch={{"execution_owner":"worker_host","worker_pool":"{worker_pool_id}","target_kind":"repo","task_affinity_key":"crafter-<candidate_id>"}}`

   Planned tasks (copy instructions verbatim from kickoff task briefs):
{candidate_lines}

2. Let all {self.parallel_worker_count} workers run concurrently. Each worker
   owns exactly one candidate directory and must NOT publish the final
   WorkProduct.

3. Monitor worker progress and experiment/result rows. As soon as ANY worker
   reports a completed candidate eval with `symbolic_policy_score` above the
   baseline, treat that worker as the provisional winner.

4. Request reviewer verification on the FIRST improving candidate's artifacts.
   Do not wait for slower workers unless no worker has beaten baseline yet.

5. When review passes (or the best available candidate is verified), publish
   ONE final Crafter code-policy DEO WorkProduct citing the winning candidate,
   cancel or stop remaining worker tasks, and close the directed-effort outcome.

6. If multiple workers beat baseline, still submit the first improving candidate
   that completed — not the highest score among all three unless review requires
   re-picking.

Workers only produce candidate evidence; orchestrator owns final submission.{extra_block}
""".strip()

    def worker_instructions(
        self,
        *,
        dataset_ref: str,
        candidate_id: str,
        worker_index: int,
    ) -> str:
        candidate_path = f"{self.candidate_dir(candidate_id)}/heuristic_policy.py"
        extra = self.extra_worker_instructions.strip()
        extra_block = f"\n\n{extra}" if extra else ""
        return f"""Improve a pure symbolic Crafter code policy in directed-effort mode.

You are parallel worker {worker_index} of {self.parallel_worker_count}. Your ONLY
candidate directory is `{candidate_path}`. Do not create other candidate dirs.

Read the staged task files under `{dataset_ref}/` first:
`{dataset_ref}/TASK_README.md`, `{dataset_ref}/TASK_INSTRUCTIONS.md`,
`{dataset_ref}/STARTING_CONTAINER.md`, `{dataset_ref}/CRAFTER_POLICY_CONTEXT.md`,
and `{dataset_ref}/task_contract.json`.

Before running, assert the root files exist:
`workspace/run_hillclimb_symbolicbench_task.py`,
`containers/crafter/heuristic_policy.py`,
`containers/crafter/run_heuristic_sweep.py`, and `task_contract.json`.

Run baseline + your single candidate in this workspace:

```bash
mkdir -p {self.candidate_dir(candidate_id)}
cp containers/crafter/heuristic_policy.py {candidate_path}
# edit {candidate_path}
python3 workspace/run_hillclimb_symbolicbench_task.py run --output-root . --env crafter --iterations 0 --train-seeds {self.train_seeds_cli()} --strict-env
python3 workspace/run_hillclimb_symbolicbench_task.py score --output-root .
```

Use exactly these train seeds: {self.train_seeds_csv()}.

Register your non-baseline candidate as an SMR experiment (baseline_snapshot,
candidate_snapshot, protocol_snapshot, artifact_refs) and attach one aggregate
result row (`symbolic_policy_score`, higher_is_better, sample_size=20,
split_name=train).

Do NOT publish the final project WorkProduct. Report your eval_summary,
candidate path, score, and delta to the orchestrator. Stop after your candidate
is evaluated once (revise in place only if it fails to compile).{extra_block}
""".strip()

    def worker_acceptance_criteria(self, candidate_id: str) -> list[str]:
        return [
            "Baseline and the assigned non-baseline candidate are evaluated on 20 train seeds.",
            f"Candidate code exists at {self.candidate_dir(candidate_id)}/heuristic_policy.py.",
            "eval_summary.json contains baseline + this candidate with score delta recorded.",
            "SMR experiment + aggregate result rows exist for this candidate attempt.",
            "Worker does not publish the final orchestrator WorkProduct.",
        ]

    def kickoff_tasks(self, *, dataset_ref: str, worker_pool_id: str) -> list[dict[str, Any]]:
        work_product = {
            "kind": "report",
            "subtype": "code_result",
            "title": "Crafter code-policy DEO result",
            "required": True,
            "description": "Winning candidate evidence (orchestrator publishes after review).",
        }
        tasks: list[dict[str, Any]] = []
        for index, candidate_id in enumerate(self.candidate_ids, start=1):
            tasks.append(
                {
                    "task_key": self.worker_task_key(candidate_id),
                    "kind": "repo_task",
                    "worker_pool": worker_pool_id,
                    "title": f"Crafter candidate {candidate_id}",
                    "required_work_products": [work_product],
                    "instructions": self.worker_instructions(
                        dataset_ref=dataset_ref,
                        candidate_id=candidate_id,
                        worker_index=index,
                    ),
                    "acceptance_criteria": self.worker_acceptance_criteria(candidate_id),
                }
            )
        return tasks

    def task_instructions_markdown(self) -> str:
        candidate_lines = "\n".join(
            f"- `{self.candidate_dir(candidate_id)}/heuristic_policy.py` (worker {index})"
            for index, candidate_id in enumerate(self.candidate_ids, start=1)
        )
        return f"""# Task Instructions

Parallel candidate race: the orchestrator assigns {self.parallel_worker_count}
workers at once. Each worker owns one candidate path:

{candidate_lines}

The orchestrator reviews the **first improving candidate** and publishes the
final WorkProduct. Workers must not publish the final report themselves.

Read `TASK_README.md`, `STARTING_CONTAINER.md`, `CRAFTER_POLICY_CONTEXT.md`,
and `task_contract.json` first.

Train seeds: {self.train_seeds_csv()}.

```bash
python3 workspace/run_hillclimb_symbolicbench_task.py run --output-root . --env crafter --iterations 0 --train-seeds {self.train_seeds_cli()} --strict-env
python3 workspace/run_hillclimb_symbolicbench_task.py score --output-root .
```
""".strip()

    def profile_overrides(self) -> dict[str, Any]:
        return {
            "smr": {
                "roles": {
                    "orchestrator": {"profile_id": self.orchestrator_profile_id},
                    "worker": {"profile_id": self.worker_profile_id},
                }
            }
        }


DEFAULT_CONFIG = CrafterDeoRunConfig()


def _bootstrap_workspace_env() -> None:
    """Default sibling-repo paths when the driver is invoked outside the shell wrapper."""
    workspace_root = SYNTH_AI_ROOT.parent
    os.environ.setdefault("SYNTH_WORKSPACE_ROOT", str(workspace_root))
    os.environ.setdefault("EVALS_ROOT", str(workspace_root / "evals"))
    os.environ.setdefault("MANAGED_RESEARCH_ROOT", str(workspace_root / "managed-research"))
    os.environ.setdefault("REPORTBENCH_TASK_ROOT", str(LANE_ROOT.resolve()))


# ---------------------------------------------------------------------------
# Bundle + launch helpers
# ---------------------------------------------------------------------------


def crafter_runs_dir() -> Path:
    return Path(__file__).resolve().parent / "runs"


def default_output_root(*, target: str) -> Path:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.strip()) or "local"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (crafter_runs_dir() / f"{stamp}_{label}").resolve()


def _dataset_ref_from_bundle(bundle: dict[str, Any]) -> str:
    kickoff = (bundle.get("trigger_payload") or {}).get("kickoff_contract")
    if isinstance(kickoff, dict):
        ref = str(kickoff.get("kickoff_contract_ref") or "").strip()
        if ref.startswith("starting-data/"):
            return ref.rsplit("/", 1)[0]
        for item in kickoff.get("model_visible_contract_files") or ():
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            if path.startswith("starting-data/") and path.endswith("/task_contract.json"):
                return path.rsplit("/", 1)[0]
    effective = bundle.get("effective_config") or {}
    for path in (effective.get("smr") or {}).get("workspace_inputs", {}).get("files") or ():
        if not isinstance(path, dict):
            continue
        staged = str(path.get("path") or "")
        if staged == "task_contract.json":
            continue
    return "starting-data/reportbench-lane"


def apply_config_to_bundle(bundle: dict[str, Any], config: CrafterDeoRunConfig) -> None:
    """Patch kickoff contract + staged markdown from in-script config."""
    from reportbench.project_config import _render_bootstrap_task_brief

    dataset_ref = _dataset_ref_from_bundle(bundle)
    worker_pool_id = str(bundle.get("worker_pool_id") or "").strip()
    task_instructions_md = config.task_instructions_markdown()
    project_notes = config.project_notes(worker_pool_id=worker_pool_id)
    kickoff_tasks = config.kickoff_tasks(
        dataset_ref=dataset_ref,
        worker_pool_id=worker_pool_id,
    )

    trigger_payload = bundle.setdefault("trigger_payload", {})
    kickoff = trigger_payload.get("kickoff_contract")
    if not isinstance(kickoff, dict):
        raise RuntimeError("launch bundle missing trigger_payload.kickoff_contract")

    model_visible_files = [
        item
        for item in kickoff.get("model_visible_contract_files") or ()
        if isinstance(item, dict)
    ]
    kickoff["project_notes_framing"] = project_notes
    kickoff["tasks"] = kickoff_tasks
    kickoff["task_briefs"] = [
        _render_bootstrap_task_brief(
            task_payload=task_payload,
            index=index,
            resolved_worker_pool_id=worker_pool_id,
            model_visible_contract_files=model_visible_files,
        )
        for index, task_payload in enumerate(kickoff_tasks, start=1)
    ]

    runnable = bundle.get("runnable_project_request")
    if isinstance(runnable, dict):
        runnable["notes"] = project_notes

    workspace_inputs = bundle.get("workspace_inputs")
    if not isinstance(workspace_inputs, dict):
        return

    def _patch_files(files: Any) -> None:
        if not isinstance(files, list):
            return
        for item in files:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            if path.endswith("TASK_INSTRUCTIONS.md"):
                item["content"] = task_instructions_md
            if path.endswith("kickoff_contract.json"):
                item["content"] = json.dumps(kickoff, indent=2, ensure_ascii=True)

    _patch_files(workspace_inputs.get("files"))
    _patch_files(workspace_inputs.get("all_files"))

    effective = bundle.setdefault("effective_config", {})
    effective["crafter_deo_run_config"] = {
        "parallel_worker_count": config.parallel_worker_count,
        "candidate_ids": list(config.candidate_ids),
        "train_seeds": list(config.train_seeds),
        "orchestrator_profile_id": config.orchestrator_profile_id,
        "worker_profile_id": config.worker_profile_id,
    }


def build_launch_bundle(
    *,
    worker_pool_id: str,
    config: CrafterDeoRunConfig,
) -> dict[str, Any]:
    from reportbench.project_config import build_staged_reportbench_launch_bundle

    if not TASK_TOML.is_file():
        raise FileNotFoundError(f"missing lane task.toml: {TASK_TOML}")

    os.environ["REPORTBENCH_TASK_ROOT"] = str(LANE_ROOT.resolve())
    bundle = build_staged_reportbench_launch_bundle(
        task_id=str(TASK_TOML.resolve()),
        nick=worker_pool_id,
        worker_pool_id=worker_pool_id,
        overrides=config.profile_overrides(),
    )
    apply_config_to_bundle(bundle, config)
    timebox = max(config.run_timebox_seconds, 60)
    trigger_payload = bundle.setdefault("trigger_payload", {})
    trigger_payload["timebox_seconds"] = timebox
    return bundle


def _require_docker_image(image: str) -> None:
    import subprocess

    probe = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"[crafter-deo] missing Docker image: {image}\n"
            "Build open_research_crafter first (see crafter_runs/README.md)."
        )


def run_crafter_deo(
    *,
    launch: ReadmeSmokeLaunch,
    output_root: Path,
    config: CrafterDeoRunConfig = DEFAULT_CONFIG,
    research: ResearchClient | None = None,
    log: LogFn | None = None,
) -> int:
    from reportbench.readme_smoke_harness import (
        field_value,
        is_retryable_launch_backpressure,
        is_retryable_pre_run_transport_error,
        is_retryable_setup_backpressure,
        jsonish,
        launch_blocker_code,
        poll_run_until_terminal,
        setup_retry_delay_seconds,
        trigger_kwargs_from_bundle,
    )

    _log = log or (lambda message: print(message, flush=True))
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    archive_path = output_root / "workspace.tar.gz"
    log_path = output_root / "run.log"
    log_lines: list[str] = []

    def _emit(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        log_lines.append(line)
        _log(msg)

    client = (research or build_research_client(
        api_key=launch.api_key,
        base_url=launch.backend,
    )).control(timeout_seconds=120.0)

    summary: dict[str, Any] = {
        "sdk": "synth-ai",
        "driver": "crafter_runs/crafter_deo_run.py",
        "task_id": TASK_ID,
        "lane_root": str(LANE_ROOT),
        "output_root": str(output_root),
        "target": launch.target,
        "backend": launch.backend,
        "host_kind": launch.host_kind.value,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_config": json.loads(json.dumps(config.profile_overrides())),
    }
    summary["run_config"].update(
        {
            "parallel_worker_count": config.parallel_worker_count,
            "candidate_ids": list(config.candidate_ids),
            "train_seeds": list(config.train_seeds),
            "run_timebox_seconds": config.run_timebox_seconds,
            "poll_timebox_seconds": config.poll_timebox_seconds,
        }
    )

    bundle = build_launch_bundle(worker_pool_id=launch.worker_pool_id, config=config)
    worker_pool_id = str(bundle.get("worker_pool_id") or launch.worker_pool_id).strip()
    runnable_project_request = dict(bundle.get("runnable_project_request") or {})
    files = bundle.get("workspace_inputs", {}).get("files")
    if not isinstance(files, list):
        files = []
    work_mode = ResearchWorkMode(
        str((bundle.get("trigger_payload") or {}).get("work_mode") or "directed_effort")
    )
    trigger_kwargs = trigger_kwargs_from_bundle(
        host_kind=launch.host_kind,
        work_mode=work_mode,
        bundle=bundle,
        run_timebox_seconds=config.run_timebox_seconds,
    )
    source_repo_cfg = (
        bundle.get("workspace_inputs", {}).get("source_repo")
        if isinstance(bundle.get("workspace_inputs"), dict)
        else None
    )
    _apply_slot_trigger_overrides(
        trigger_kwargs,
        launch,
        source_repo_cfg=source_repo_cfg if isinstance(source_repo_cfg, dict) else None,
    )

    project_name = str(runnable_project_request.get("name") or "").strip() or (
        f"ReportBench Crafter DEO {launch.worker_pool_id}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    summary.update(
        {
            "worker_pool_id": worker_pool_id,
            "project_name": project_name,
            "workspace_input_count": len(files),
            "work_mode": work_mode.value,
        }
    )

    retention_policy = runnable_project_request.get("retention_policy")
    should_auto_archive = (
        isinstance(retention_policy, dict)
        and str(retention_policy.get("class") or "").strip().lower() == "local_ephemeral_eval"
        and str(retention_policy.get("auto_archive") or "true").strip().lower()
        not in {"false", "0", "no"}
    )

    _emit(
        f"target={launch.target} backend={launch.backend} "
        f"worker_pool={worker_pool_id} run_timebox_s={config.run_timebox_seconds} "
        f"poll_timebox_s={config.poll_timebox_seconds}"
    )
    _emit(f"lane_root={LANE_ROOT}")
    _emit(f"workspace_inputs={len(files)} project_name={project_name!r}")

    final_state = ""
    try:
        with client:
            _emit("research.projects.create_runnable_project ...")
            project = client.create_runnable_project(runnable_project_request)
            project_id = str(field_value(project, "project_id", default="") or "").strip()
            if not project_id:
                raise ResearchApiError(
                    f"create_runnable_project returned no project_id: {project}"
                )
            summary["project_id"] = project_id
            _emit(f"project_id={project_id}")

            setup_deadline = time.monotonic() + config.setup_retry_timebox_seconds
            upload_attempt = 0
            while True:
                upload_attempt += 1
                try:
                    _emit(f"research.upload_workspace_files attempt={upload_attempt} ...")
                    upload_result = client.upload_workspace_files(project_id, files)
                    break
                except ResearchApiError as exc:
                    if not is_retryable_setup_backpressure(exc):
                        raise
                    if time.monotonic() >= setup_deadline:
                        raise ResearchApiError(
                            "upload_workspace_files blocked after "
                            f"{config.setup_retry_timebox_seconds}s: {exc}"
                        ) from exc
                    delay_s = setup_retry_delay_seconds(upload_attempt)
                    _emit(f"setup backpressure upload delay_s={delay_s:.1f}")
                    time.sleep(delay_s)
            summary["upload_result"] = jsonish(upload_result)

            prepare_attempt = 0
            while True:
                prepare_attempt += 1
                try:
                    _emit(f"research.prepare_project_setup attempt={prepare_attempt} ...")
                    setup = client.prepare_project_setup(project_id)
                    break
                except ResearchApiError as exc:
                    if not is_retryable_setup_backpressure(exc):
                        raise
                    if time.monotonic() >= setup_deadline:
                        raise ResearchApiError(
                            f"prepare_project_setup blocked after "
                            f"{config.setup_retry_timebox_seconds}s: {exc}"
                        ) from exc
                    delay_s = setup_retry_delay_seconds(prepare_attempt)
                    time.sleep(delay_s)
            setup_state = str(field_value(setup, "state", default="") or "").strip().lower()
            summary["setup"] = jsonish(setup)
            if setup_state != "ready":
                raise ResearchApiError(f"setup not ready (state={setup_state!r}): {setup}")

            launch_deadline = time.monotonic() + config.launch_retry_timebox_seconds
            launch_attempt = 0
            run_id = ""
            while True:
                launch_attempt += 1
                try:
                    preflight = client.get_launch_preflight(project_id, **trigger_kwargs)
                    summary["launch_preflight"] = jsonish(preflight)
                    if not bool(field_value(preflight, "clear_to_trigger", default=False)):
                        if is_retryable_launch_backpressure(preflight):
                            if time.monotonic() >= launch_deadline:
                                raise ResearchApiError("launch preflight blocked (timeout)")
                            code = launch_blocker_code(preflight) or "launch_backpressure"
                            _emit(f"launch preflight backpressure code={code}")
                            time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                            continue
                        raise ResearchApiError(f"launch preflight blocked: {preflight}")
                    run = client.trigger_run(project_id, **trigger_kwargs)
                    run_id = str(field_value(run, "run_id", "id", default="") or "").strip()
                    if not run_id:
                        raise ResearchApiError(f"trigger_run returned no run_id: {run}")
                    break
                except ResearchApiError as exc:
                    if is_retryable_launch_backpressure(exc) and time.monotonic() < launch_deadline:
                        _emit(f"launch backpressure trigger_run: {exc}")
                        time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                        continue
                    raise
                except Exception as exc:  # noqa: BLE001
                    if (
                        is_retryable_pre_run_transport_error(exc)
                        and time.monotonic() < launch_deadline
                    ):
                        _emit(f"launch transport retry: {type(exc).__name__}: {exc}")
                        time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                        continue
                    raise

            summary["run_id"] = run_id
            summary["trigger_response"] = jsonish(run)
            _emit(f"run_id={run_id}")

            final_run = poll_run_until_terminal(
                client,
                project_id,
                run_id,
                timebox_s=config.poll_timebox_seconds,
                log=_emit,
            )
            final_state = str(field_value(final_run, "public_state", default="") or "").lower()
            summary["final_state"] = final_state
            summary["final_run"] = jsonish(final_run)
            _emit(f"final_state={final_state}")

            try:
                archive_meta = client.download_workspace_archive(project_id, archive_path)
                summary["archive_meta"] = jsonish(archive_meta)
                _emit(f"archive bytes={archive_meta.get('bytes_written')}")
            except ResearchApiError as exc:
                summary["workspace_download_error"] = str(exc)
                _emit(f"workspace download failed: {exc}")

            if should_auto_archive:
                try:
                    summary["archive_project"] = jsonish(client.archive_project(project_id))
                    _emit("project archived")
                except ResearchApiError as exc:
                    summary["archive_project_error"] = str(exc)

    except ResearchApiError as exc:
        summary["fatal_error"] = {"type": "ResearchApiError", "message": str(exc)}
        _emit(f"FATAL ResearchApiError: {exc}")
    except Exception as exc:  # noqa: BLE001
        summary["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _emit(f"FATAL {type(exc).__name__}: {exc}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"\nsummary written to: {summary_path}")
    print(f"run log:           {log_path}")
    if archive_path.exists():
        print(f"workspace archive: {archive_path} ({archive_path.stat().st_size} bytes)")

    if summary.get("fatal_error"):
        return 2
    if final_state in {"completed", "succeeded", "done"}:
        print(f"[ok] task_id={TASK_ID} final_state={final_state}")
        return 0
    print(f"[fail] task_id={TASK_ID} final_state={final_state}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-default-slot1", action="store_true")
    parser.add_argument("--slot", default=None)
    parser.add_argument("--slot-mode", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--worker-pool", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-docker-image-check", action="store_true")
    args = parser.parse_args(argv)

    _bootstrap_workspace_env()
    evals_root = _resolve_evals_root()
    _ensure_evals_importable(evals_root)

    if args.use_default_slot1:
        args.slot = args.slot or "slot1"
        args.slot_mode = args.slot_mode or "local-dockerized"

    launch = resolve_readme_smoke_launch(
        slot=args.slot,
        slot_mode=args.slot_mode,
        backend=args.backend,
        api_key=args.api_key,
        worker_pool=args.worker_pool,
        use_default_slot1=args.use_default_slot1,
    )

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(target=launch.target)
    )

    config = DEFAULT_CONFIG
    if not args.skip_docker_image_check:
        _require_docker_image(config.crafter_docker_image)

    print(f"[crafter-deo] output_root={output_root}", flush=True)
    research = build_research_client(api_key=launch.api_key, base_url=launch.backend)
    return run_crafter_deo(
        launch=launch,
        output_root=output_root,
        config=config,
        research=research,
    )


if __name__ == "__main__":
    raise SystemExit(main())
