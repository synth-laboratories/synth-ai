#!/usr/bin/env python3
"""Run Crafter code-policy DEO hillclimb via the synth-ai Research SDK.

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
import contextlib
import json
import os
import re
import tarfile
import time
import tomllib
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Import shared slot/backend helpers from the README smoke driver.
from readme_runs.inspect_actor_progress import (
    DEFAULT_PROGRESS_KINDS,
    ActorTraceState,
    _event_id,
    _event_kind,
    _fetch_transcript_events,
    _format_event_line,
    _is_tool_event_kind,
    _iter_trace_events,
    _kind_allowed,
    _load_actors,
    _participant_session_id,
)
from readme_runs.kickoff_guidance import apply_guidance_only_kickoff, kickoff_guidance_summary
from readme_runs.readme_smoke import (
    ReadmeSmokeLaunch,
    _actor_cost_usd,
    _actor_token_total,
    _apply_slot_trigger_overrides,
    _ensure_evals_importable,
    _fetch_project_git_status,
    _format_token_count,
    _format_usd,
    _is_workspace_noise_path,
    _normalize_workspace_compare_path,
    _resolve_evals_root,
    build_research_client,
    resolve_readme_smoke_launch,
)
from synth_ai.managed_research import (
    ActorImageId,
    RunLaunchRequest,
    RuntimeImage,
    RuntimeImageError,
)
from synth_ai.research import ResearchApiError, ResearchClient, ResearchWorkMode

LogFn = Callable[[str], None]

SYNTH_AI_ROOT = Path(__file__).resolve().parent.parent
LANE_ROOT = Path(__file__).resolve().parent / "lane"
TASK_TOML = LANE_ROOT / "task.toml"
TASK_ID = "reportbench/crafter_code_policy_deo_hillclimb_1cand"


# ---------------------------------------------------------------------------
# Edit run setup here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrafterDeoRunConfig:
    """Lane text + launch knobs. Code artifacts stay under ``crafter_runs/lane/``."""

    parallel_worker_count: int = 1
    candidate_ids: tuple[str, ...] = ("attempt_1",)

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
    objective_without_plan_observations: int = 5
    worker_no_evidence_token_threshold: int = 750_000
    worker_no_evidence_observations: int = 2
    worker_evidence_without_progress_token_threshold: int = 750_000
    worker_evidence_without_progress_observations: int = 2
    worker_evidence_with_progress_token_threshold: int = 750_000
    worker_evidence_with_progress_observations: int = 2

    crafter_runtime_image: RuntimeImage = field(
        default_factory=lambda: RuntimeImage.catalog(ActorImageId.OPEN_RESEARCH_CRAFTER)
    )

    extra_worker_instructions: str = (
        "Use a simple crafting ladder before inventing a new search strategy: "
        "when wood is available, place a table; after the table, make a wood "
        "pickaxe once wood is sufficient; then prefer stone pickaxe, furnace, "
        "iron pickaxe, and basic tools/walls when resources allow. Before "
        "calling set_task_state(done), complete the terminal checklist in order: "
        "verify the assigned candidate file exists at the exact path in this brief; "
        "verify eval_summary.json names the same candidate id with a non-baseline "
        "score and delta; call smr_attach_experiment_result for the aggregate "
        "symbolic_policy_score row; call smr_list_experiment_results and confirm "
        "that row is visible; verify reports/final_report.md exists; call "
        "workspace_push; send baseline-score and candidate-score create_runtime_message "
        "calls and copy the returned message ids into reports/final_report.md and "
        "reports/candidate_progression.md; call publish_report_work_product first "
        "with exact title `Crafter candidate evidence report`, then publish/update "
        "`Crafter candidate progression`; re-check list_run_work_products shows "
        "both reports ready; only then call set_task_state(done)."
    )
    extra_orchestrator_instructions: str = ""

    def train_seeds_csv(self) -> str:
        return ",".join(str(seed) for seed in self.train_seeds)

    def train_seed_count(self) -> int:
        return len(self.train_seeds)

    def train_seeds_cli(self) -> str:
        return f"crafter={self.train_seeds_csv()}"

    def candidate_dir(self, candidate_id: str) -> str:
        return f"candidates/crafter/{candidate_id}"

    def required_candidate_paths(self) -> list[str]:
        return [
            f"{self.candidate_dir(candidate_id)}/heuristic_policy.py"
            for candidate_id in self.candidate_ids
        ]

    def worker_task_key(self, candidate_id: str) -> str:
        return f"hillclimb_crafter_deo_{candidate_id}"

    def worker_required_work_products(self) -> list[dict[str, str]]:
        return [
            {
                "kind": "report",
                "title": "Crafter candidate progression",
                "description": (
                    "Live worker-published progression report with every evaluated "
                    "candidate, score, delta, path, seed count, and per-achievement "
                    "frequencies."
                ),
            },
            {
                "kind": "report",
                "title": "Crafter candidate evidence report",
                "description": (
                    "Worker-published final report citing baseline score, candidate "
                    "score, score delta, seed count, candidate path, eval_summary.json, "
                    "per-achievement frequencies, and reproduction notes."
                ),
            },
        ]

    def worker_plan_task_payload(
        self,
        *,
        dataset_ref: str,
        candidate_id: str,
        worker_index: int,
        worker_pool_id: str,
        depends_on_task_keys: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "task_key": self.worker_task_key(candidate_id),
            "kind": "repo_task",
            "depends_on_task_keys": list(depends_on_task_keys),
            "task_dispatch": {
                "execution_owner": "worker_host",
                "worker_pool": worker_pool_id,
                "target_kind": "repo",
                "task_affinity_key": f"crafter-{candidate_id}",
            },
            "status_detail": {
                "agent_goal_assignment": {
                    "review_policy": "reviewer_adjudicates_objective_progress",
                },
            },
            "input": {
                "assigned_candidate_id": candidate_id,
                "assigned_candidate_path": f"{self.candidate_dir(candidate_id)}/heuristic_policy.py",
                "instructions": self.worker_instructions(
                    dataset_ref=dataset_ref,
                    candidate_id=candidate_id,
                    worker_index=worker_index,
                ),
                "acceptance_criteria": self.worker_acceptance_criteria(candidate_id),
                "required_work_products": self.worker_required_work_products(),
                "require_workspace_push_on_done": True,
            },
        }

    def worker_plan_task_payloads(
        self,
        *,
        dataset_ref: str,
        worker_pool_id: str,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        concurrency = max(1, self.parallel_worker_count)
        for index, candidate_id in enumerate(self.candidate_ids, start=1):
            depends_on: list[str] = []
            dependency_index = index - concurrency
            if dependency_index >= 1:
                depends_on.append(self.worker_task_key(self.candidate_ids[dependency_index - 1]))
            payloads.append(
                self.worker_plan_task_payload(
                    dataset_ref=dataset_ref,
                    candidate_id=candidate_id,
                    worker_index=index,
                    worker_pool_id=worker_pool_id,
                    depends_on_task_keys=depends_on,
                )
            )
        return payloads

    def worker_plan_tasks_json(
        self,
        *,
        dataset_ref: str,
        worker_pool_id: str,
    ) -> str:
        return json.dumps(
            {
                "tasks": self.worker_plan_task_payloads(
                    dataset_ref=dataset_ref, worker_pool_id=worker_pool_id
                )
            },
            indent=2,
            ensure_ascii=True,
        )

    def git_server_collaboration_brief(self, *, role: str) -> str:
        shared = """
Git-server collaboration:
- This run shares one project git-server repo. That repo is the durable handoff
  surface between orchestrator and worker(s); local-only edits are invisible until
  pushed.
- Do not use raw `git commit` or raw `git push`. Use the `workspace_push` tool.
""".strip()
        if role == "orchestrator":
            return f"""{shared}
- Supervise workers from git-server state (repo tree, recent commits) plus SMR
  experiment/result rows — not by running worker hillclimb commands locally.
- Before closeout, confirm the winning candidate path, `eval_summary.json`, and
  ledgers are present in the pushed git-server tree.
- Confirm at least one report WorkProduct exists before requesting review or
  setting the run done."""
        if role == "worker":
            return f"""{shared}
- Edit candidate code and run hillclimb in your repo workspace, then push all
  deliverable paths to git-server with `workspace_push` before `set_task_state(done)`.
- The hillclimb runner writes `reports/final_report.md` from eval_summary.json.
  After the run, inspect that file, publish the same content as a report WorkProduct,
  push it with the artifacts, and re-check the WorkProduct list before
  `set_task_state(done)`.
- The orchestrator reviews your pushed git-server tree and experiment rows; keep
  candidate paths stable and cite the same paths in experiment artifacts."""
        return shared

    def runtime_messaging_collaboration_brief(self, *, role: str) -> str:
        shared = """
Runtime messaging (SMR message queue):
- Workers use the worker MCP `create_runtime_message` tool for durable run-scoped
  coordination. If the runtime exposes a legacy alias such as `smr_send_message` or
  `send_message`, that is acceptable too. Messages appear in the runtime message
  queue and in poll logs as `[msg]` lines.
- Prefer `mode=steer`, `topic=crafter.metrics`, and concrete metrics/paths.
""".strip()
        if role == "orchestrator":
            return f"""{shared}
- After objective bootstrap, send a short kickoff note to workers with candidate paths
  and the expectation to report baseline + candidate scores via message + git-server push.
- When a worker message reports baseline or candidate metrics, call
  `mcp__orchestrator__record_objective_progress` with those numbers.
- Before closeout, confirm at least one worker progress message matches pushed git-server
  artifacts (eval_summary.json, candidate path)."""
        if role == "worker":
            return f"""{shared}
- After baseline eval completes, send one message with baseline score and seed count.
- After your candidate eval completes, send one message with candidate id, path, score,
  and delta vs baseline.
- Prefer `create_runtime_message` with your task key in `task_key`, topic
  `crafter.metrics`, action `baseline_score` or `candidate_score`, a concise
  body, and structured JSON payload containing candidate id, score, delta, seed
  count, candidate path, and per-achievement frequencies when available.
- Then `workspace_push`, publish the report WorkProduct, verify the WorkProduct
  exists, and only then call `set_task_state(done)`."""
        return shared

    def work_product_collaboration_brief(self, *, role: str) -> str:
        if role == "orchestrator":
            return """
WorkProduct closeout:
- Worker tasks must publish a candidate evidence report WorkProduct before they
  are marked done.
- The orchestrator may publish a final closeout report too, but must not request
  review or call `set_run_state(done)` until `list_run_work_products` shows at
  least one ready report WorkProduct for the run.
- Build any final closeout report from pushed git-server evidence: winning
  candidate path, eval_summary metrics, experiment rows, and reproduction notes."""
        if role == "worker":
            return """
WorkProduct requirement:
- The hillclimb runner creates `reports/final_report.md`. Read it after the run
  and confirm it summarizes baseline score, candidate score, score delta, seed
  count, candidate path, candidate progression, per-achievement frequencies,
  and artifact paths.
- Publish live report WorkProducts as the task progresses:
  - after baseline metrics are available, publish `Crafter candidate progression`
    with baseline score, seed count, and baseline achievement frequencies;
  - after each non-baseline candidate finishes, publish/update `Crafter candidate
    progression` with all candidate rows so far, score deltas, paths, and
    per-achievement frequencies.
- Use `publish_report_work_product` for both required reports. In the terminal
  checklist after `workspace_push`, publish the acceptance-critical report first:
  - title `Crafter candidate evidence report`, mode `directed_effort`,
    report_text copied from `reports/final_report.md`;
  - title `Crafter candidate progression`, mode `directed_effort`, report_text
    copied from `reports/candidate_progression.md`.
  Pass `control_plane_task_id` set to your assigned control-plane task UUID for
  both calls.
- Push `reports/final_report.md` plus eval artifacts to git-server.
- Call `list_run_work_products` and confirm both exact titles are ready before
  `set_task_state(done)`. If either WorkProduct is not listed, do not mark done."""
        return ""

    def objective_authoring_playbook(self) -> str:
        spec = load_lane_objective_spec()
        success = self.objective_success_criteria(spec)
        deliverables = spec.get("deliverable_requirements") or []
        milestones = spec.get("suggested_milestones") or []
        success_lines = "\n".join(f"  - {line}" for line in success)
        deliverable_lines = "\n".join(f"  - {line}" for line in deliverables)
        milestone_lines = "\n".join(f"  - `{key}`" for key in milestones)
        return f"""
Objective authoring (orchestrator-owned — nothing is pre-created at launch):

Turn-0 bootstrap BEFORE `plan_tasks` (same turn if possible):
1. Inspect existing objectives:
   - `mcp__orchestrator__read_objective_verdicts` (if available), and/or
   - project MCP `smr_directed_effort_outcomes` operation=list for this project/run.
2. If no directed-effort parent exists yet, CREATE exactly one parent objective using
   the lane spec below. Prefer, in order:
   - `mcp__orchestrator__create_directed_effort_objective` (orchestrator server), else
   - project MCP `smr_directed_effort_outcomes` operation=create with kind
     `directed_effort_outcome`.
   Do not proceed to worker planning until the create call returns an objective id.
3. Publish milestone ladder with `mcp__orchestrator__write_project_milestones`
   (after the objective exists). Use these checkpoint keys (adapt titles/goals to
   Crafter hillclimb semantics):
{milestone_lines}
4. Optionally verify with `mcp__orchestrator__get_project_milestones`.

Lane directed-effort spec to copy into the create payload (paraphrase allowed; keep
measurable success criteria verbatim):
- kind: directed_effort_outcome
- title: {spec.get("title") or "Improve Crafter symbolic code policy"}
- description: {spec.get("description") or "Crafter code-policy DEO hillclimb"}
- scope: {spec.get("scope") or "Crafter HillClimbSymbolicBench branch"}
- outcome_text: {spec.get("outcome_text") or "Best accepted Crafter policy with score delta"}
- success_criteria:
{success_lines or "  - (see task.toml smr.objective.success_criteria)"}
- deliverable_requirements:
{deliverable_lines or "  - (see task.toml smr.objective.deliverable_requirements)"}

During supervision (after workers start):
- After baseline capture, first compile, first score lift, and each verified candidate,
  call `mcp__orchestrator__record_objective_progress` with concrete metrics from durable
  artifacts (eval_summary.json, experiment rows) — e.g.
  `baseline=0.2065 on {self.train_seed_count()} configured seeds`,
  `candidate attempt_1 score=0.31 delta=+0.1035`, not narrative guesses. If this
  run uses a configured seed override, state the actual seed count.
- Use claim_kind=`progress` for partial metric movement; claim_kind=`achievement` when a
  milestone threshold is satisfied.
- When linking tasks to the objective, include in planned task status_detail:
  `{{"objective_kind":"directed_effort_outcome","objective_id":"<id>"}}`.

Closeout:
- Re-read verdicts with `mcp__orchestrator__read_objective_verdicts`.
- Confirm `list_run_work_products` shows a ready report WorkProduct. If not,
  keep the run active and ask the worker to publish the required report.
- Objective review is mandatory for this lane. After a ready report WorkProduct
  exists and eval_summary.json shows either a score lift or a terminal no-lift
  result, call `mcp__reviewer_dispatch__request_objective_review` with the
  directed-effort objective id and a reason that cites the winning candidate
  path, baseline score, candidate score, score delta, and WorkProduct/report refs.
- Keep the run active while the objective reviewer records its advisory verdict.
  Poll with `mcp__orchestrator__read_objective_verdicts` until the objective
  `evaluation_state` is no longer `active` and `review_summary` is populated.
  Do not confuse the normal task reviewer with `reviewer:objective`; this gate is
  specifically the parent objective review.
- If the objective reviewer returns `satisfied` or `partial`, record any final
  achievement/progress claim needed for the milestone ladder, then
  `mcp__orchestrator__set_run_state(state="done")`. If it returns
  `needs_revision` or `failed`, keep the run active and plan repair work instead
  of closing the run.
""".strip()

    def objective_success_criteria(self, spec: Mapping[str, Any]) -> list[str]:
        configured_seed_count = self.train_seed_count()
        criteria = spec.get("success_criteria") or []
        normalized: list[str] = []
        for criterion in criteria:
            text = str(criterion)
            text = text.replace(
                "exactly 20 train seeds",
                f"exactly {configured_seed_count} configured train seeds",
            )
            normalized.append(text)
        return normalized

    def project_notes(self, *, worker_pool_id: str, dataset_ref: str) -> str:
        candidate_lines = "\n".join(
            f"  - task_key `{self.worker_task_key(candidate_id)}` → "
            f"`{self.candidate_dir(candidate_id)}/heuristic_policy.py` "
            f"(task_affinity_key `crafter-{candidate_id}`)"
            for candidate_id in self.candidate_ids
        )
        extra = self.extra_orchestrator_instructions.strip()
        extra_block = f"\n\n{extra}" if extra else ""
        plan_payload_json = json.dumps(
            self.worker_plan_task_payloads(
                dataset_ref=dataset_ref,
                worker_pool_id=worker_pool_id,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return f"""
REPORT BENCH - Crafter code-policy DEO.

Turn-0 mandatory tool sequence:
1. Create or discover exactly one directed-effort objective.
2. Write the milestone ladder for that objective.
3. Immediately call `plan_tasks` for the worker task(s) below.
4. Immediately call `set_run_state(state="active")`.

Do not end the first orchestrator turn after only writing milestones. If
`existing_tasks_count=0` and the objective exists, planning the worker task is
the next required action before any more narration or supervision.

ORCHESTRATOR PLAYBOOK (guidance-only kickoff — you must plan_tasks; no pre-seeded tasks).

{self.git_server_collaboration_brief(role="orchestrator")}

{self.runtime_messaging_collaboration_brief(role="orchestrator")}

{self.work_product_collaboration_brief(role="orchestrator")}

{self.objective_authoring_playbook()}

Bootstrap (after objective exists):
- When `existing_tasks_count=0`, call `mcp__orchestrator__plan_tasks` then
  `mcp__orchestrator__set_run_state(state="active")` in the same turn (or immediately
  after objective + milestones are created).
- Do not execute worker shell commands from this orchestrator session.

Planning contract — ONE `plan_tasks` call with {len(self.candidate_ids)} repo task(s), staged at up to {self.parallel_worker_count} worker task(s) at a time:
- worker pool `{worker_pool_id}`; execution_owner `worker_host` only.
- Plan exactly one task per candidate id. `parallel_worker_count` controls dependencies/concurrency only; it is not the total task count.
- Suggested tasks (copy each matching kickoff task brief into `input.instructions`):
{candidate_lines}
- Preferred payload: copy `kickoff_contract.plan_task_payloads` exactly into the
  `tasks` argument for `mcp__orchestrator__plan_tasks`. It contains the
  candidate-specific `input.instructions`, `input.acceptance_criteria`,
  `input.required_work_products`, and `task_dispatch` for every worker.
- The exact `tasks` argument JSON is also inlined below. Use this array verbatim;
  do not summarize it, truncate it, or regenerate it from memory:
```json
{plan_payload_json}
```
- Do not collapse the sibling tasks into one template. Before calling
  `plan_tasks`, verify every task has a different candidate id in
  `input.instructions`, a non-empty `input.acceptance_criteria`, and a
  matching `task_dispatch.task_affinity_key`.
- Exact plan payload source: kickoff_contract.plan_task_payloads and the inline
  JSON block above.
- Each planned task MUST set:
  - `task_dispatch={{"execution_owner":"worker_host","worker_pool":"{worker_pool_id}","target_kind":"repo","task_affinity_key":"crafter-<candidate_id>"}}`
  - `require_workspace_push_on_done: true` in task input
  - `required_work_products=[
      {{"kind":"report","title":"Crafter candidate progression","description":"Live worker-published progression report with every evaluated candidate, score, delta, path, seed count, and per-achievement frequencies."}},
      {{"kind":"report","title":"Crafter candidate evidence report","description":"Worker-published final report citing baseline score, candidate score, score delta, seed count, candidate path, eval_summary.json, per-achievement frequencies, and reproduction notes."}}
    ]` in task input

Supervision:
- Let each planned worker task run to one assigned candidate evaluation.
- Monitor experiment/result rows. When any worker beats baseline on
  `symbolic_policy_score`, treat it as provisional winner, but for parallel
  proof runs continue supervising until every assigned candidate has either a
  ready progression/evidence WorkProduct or an explicit terminal failure.
- Request reviewer verification only after the pushed workspace evidence,
  experiment result row, and ready report WorkProduct agree on the same
  candidate id and candidate path.
- If no worker beats baseline, leave the objective active with concrete progress
  evidence and do not publish a false-success WorkProduct.

Closeout:
- Use the worker-published report WorkProduct as the minimum closeout proof.
- Optionally publish one final Crafter code-policy DEO report WorkProduct citing
  the winning candidate if it adds information beyond the worker report.
- Close the directed-effort outcome and `set_run_state(done)`.
- If multiple workers are enabled by CLI override, compare all completed
  candidate reports and close out with the best completed non-baseline
  candidate. Do not close after the first improving candidate while other
  candidate workers are still running.

Workers produce candidate evidence and a required report WorkProduct; the
orchestrator owns final run closeout.{extra_block}
""".strip()

    def worker_planning_brief(
        self,
        *,
        dataset_ref: str,
        candidate_id: str,
        worker_index: int,
        worker_pool_id: str,
    ) -> str:
        acceptance = "\n".join(
            f"- {criterion}" for criterion in self.worker_acceptance_criteria(candidate_id)
        )
        return f"""Task brief {worker_index} (template for `plan_tasks` — copy into `input.instructions`):
- task_key: {self.worker_task_key(candidate_id)}
- kind: repo_task
- worker_pool: {worker_pool_id}
- require_workspace_push_on_done: true
- task_dispatch.execution_owner: worker_host
- task_dispatch.worker_pool: {worker_pool_id}
- task_dispatch.target_kind: repo
- task_dispatch.task_affinity_key: crafter-{candidate_id}
- input.required_work_products:
  - kind: report
    title: Crafter candidate progression
    description: Live worker-published progression report with every evaluated candidate, score, delta, path, seed count, and per-achievement frequencies.
  - kind: report
    title: Crafter candidate evidence report
    description: Worker-published final report citing baseline score, candidate score, score delta, seed count, candidate path, eval_summary.json, per-achievement frequencies, and reproduction notes.

Instructions:
{self.worker_instructions(dataset_ref=dataset_ref, candidate_id=candidate_id, worker_index=worker_index)}

Acceptance criteria:
{acceptance}
""".strip()

    def worker_planning_briefs(
        self,
        *,
        dataset_ref: str,
        worker_pool_id: str,
    ) -> list[str]:
        return [
            self.worker_planning_brief(
                dataset_ref=dataset_ref,
                candidate_id=candidate_id,
                worker_index=index,
                worker_pool_id=worker_pool_id,
            )
            for index, candidate_id in enumerate(self.candidate_ids, start=1)
        ]

    def worker_instructions(
        self,
        *,
        dataset_ref: str,
        candidate_id: str,
        worker_index: int,
    ) -> str:
        candidate_path = f"{self.candidate_dir(candidate_id)}/heuristic_policy.py"
        task_key = self.worker_task_key(candidate_id)
        extra = self.extra_worker_instructions.strip()
        extra_block = f"\n\n{extra}" if extra else ""
        return f"""Improve a pure symbolic Crafter code policy in directed-effort mode.

First-turn execution contract:
- Do not spend the worker turn planning in prose. Start with a shell action.
- Your first substantive shell action must prepare the assigned candidate path
  and run the exact baseline + candidate command below. Do not inspect runner
  internals unless that command fails and the failure output requires it.
- If the required files are missing, report the missing path and block the task.

You are candidate worker {worker_index}. This run allows at most
{self.parallel_worker_count} worker task(s) at a time. Your ONLY candidate
directory is `{candidate_path}`. Your task key is `{task_key}`.
Do not create other candidate dirs.

Operational rules:
- Use `workspace_push`; do not use raw `git commit` or raw `git push`.
- Send one `create_runtime_message` after baseline metrics and one after candidate
  metrics. Pass `task_key="{task_key}"`, `topic="crafter.metrics"`, and action
  `baseline_score` or `candidate_score`. Use a legacy `smr_send_message` or
  `send_message` alias only if that is what the tool list exposes.
- After both message calls succeed, append a `Runtime Message Evidence` section
  to `reports/final_report.md` and `reports/candidate_progression.md` with the
  returned message ids, actions, candidate id, scores, delta, seed count, and
  task key. If either message call fails or no message id is returned, retry
  once and then block instead of marking done.
- Publish a live `Crafter candidate progression` report WorkProduct after
  baseline metrics are available, then publish/update it again after candidate
  metrics are available. Include every evaluated candidate row, score, delta,
  candidate path, per-achievement frequencies from eval_summary.json, and the
  runtime message ids that reported the metrics. This live progression report
  must never delay the terminal `Crafter candidate evidence report` publication.
- Push candidate code, eval artifacts, experiment registration, and
  `reports/final_report.md` with `workspace_push` before publishing the final
  report WorkProduct.
- Publish `reports/final_report.md` with exact title
  `Crafter candidate evidence report` first after `workspace_push`; then
  publish/update `reports/candidate_progression.md` with exact title
  `Crafter candidate progression` before `set_task_state(done)`.

Read the staged task files under `{dataset_ref}/` first:
`{dataset_ref}/TASK_README.md`, `{dataset_ref}/TASK_INSTRUCTIONS.md`,
`{dataset_ref}/STARTING_CONTAINER.md`, `{dataset_ref}/CRAFTER_POLICY_CONTEXT.md`,
and `{dataset_ref}/task_contract.json`.

Before running, assert the root files exist:
`workspace/run_hillclimb_symbolicbench_task.py`,
`workspace/crafter_proof_policy.py`,
`containers/crafter/heuristic_policy.py`,
`containers/crafter/run_heuristic_sweep.py`, and `task_contract.json`.

Run baseline + your single candidate in this workspace:

```bash
mkdir -p {self.candidate_dir(candidate_id)}
cp workspace/crafter_proof_policy.py {candidate_path}
python3 workspace/run_hillclimb_symbolicbench_task.py run --output-root . --env crafter --iterations 0 --candidate-root candidates --candidate-id {candidate_id} --train-seeds {self.train_seeds_cli()} --strict-env
```

Use exactly these train seeds: {self.train_seeds_csv()}.
The proof eval normally takes minutes. Do not interrupt it merely because the
terminal is quiet; wait for the command to exit. Only stop it for a clear process
failure, an explicit external timeout, or a reproducible hang with diagnostic
evidence. If your shell tool supports a timeout, set it to at least 900 seconds.
Do not run the benchmark `score` subcommand inside the worker loop for this DEO
proof; it is a verifier gate, not the durable evidence source. Read
`artifacts/workproduct_container/eval_summary.json` directly and publish from
that file.

Register your non-baseline candidate as an SMR experiment with
`smr_propose_project_experiment` (baseline_snapshot, candidate_snapshot,
protocol_snapshot, artifact_refs). Then call `smr_attach_experiment_result` for
the aggregate candidate row. Use:
- experiment_id: the id returned by `smr_propose_project_experiment`
- metric: `symbolic_policy_score`
- metric_direction: `higher_is_better`
- value: candidate score from `artifacts/workproduct_container/eval_summary.json`
- baseline_value: baseline score from the same file
- delta: candidate score minus baseline score
- candidate_id: `crafter:{candidate_id}`
- candidate_label: `{candidate_id}`
- candidate_kind: `code_policy`
- dataset_or_task_set_id: `{TASK_ID}`
- sample_size: {self.train_seed_count()}
- seed_set: [{self.train_seeds_csv()}]
- split_name: `train`
- summary_artifact_path: `artifacts/workproduct_container/eval_summary.json`
- per_example_artifact_path:
  `artifacts/workproduct_container/experiment_results.json`
- task_ids: include your control-plane task UUID if it is present in your task
  context; otherwise include `{task_key}`
- truth_status: `observed`

After attaching, call `smr_list_experiment_results` for that experiment_id and
metric. If it returns count=0 or no row for `crafter:{candidate_id}`, do not
mark the task done; retry the attach once, then block with the missing result
row details.

Mandatory terminal checklist after the run exits 0:
1. Read `artifacts/workproduct_container/eval_summary.json` and `reports/final_report.md`.
2. Call `smr_propose_project_experiment`, then `smr_attach_experiment_result`
   exactly as specified above, then verify it with `smr_list_experiment_results`.
3. Send `create_runtime_message` with `task_key="{task_key}"` (or exposed legacy
   message alias) with baseline score, candidate score, delta, seed count, and
   `candidates/crafter/{candidate_id}/heuristic_policy.py`; capture the returned
   message ids.
4. Append a `Runtime Message Evidence` section to both
   `reports/candidate_progression.md` and `reports/final_report.md` containing
   the message ids, task key, candidate id, baseline score, candidate score,
   delta, and seed count.
5. Call `workspace_push` including candidate code, eval artifacts, experiment
   registration, `reports/candidate_progression.md`, and `reports/final_report.md`.
6. Call `publish_report_work_product` with exact title
   `Crafter candidate evidence report` using `reports/final_report.md` as
   report_text.
7. Publish/update `Crafter candidate progression` with exact title
   `Crafter candidate progression`, all evaluated candidates, score deltas,
   candidate paths, per-achievement frequencies, and runtime message ids.
8. Call `list_run_work_products`; if both exact report titles are not listed as
   ready/published, block instead of marking the task done.
9. Only after steps 1-8 succeed, call `set_task_state(done)`.

Report your eval_summary, candidate path, score, delta, accepted/rejected status,
and WorkProduct id to the orchestrator. Stop after the staged proof candidate is
evaluated once. If it fails to compile or the eval command fails, block the task.
If it compiles and evaluates but does not beat baseline, publish rejected
candidate evidence, push artifacts, publish the report WorkProduct, and mark the
task done so the orchestrator can compare every candidate.{extra_block}
""".strip()

    def worker_acceptance_criteria(self, candidate_id: str) -> list[str]:
        return [
            "Baseline and the assigned non-baseline candidate are evaluated "
            f"on the configured {self.train_seed_count()} train seeds.",
            f"Candidate code exists at {self.candidate_dir(candidate_id)}/heuristic_policy.py.",
            "eval_summary.json contains baseline + this candidate with score delta recorded.",
            "SMR experiment exists and smr_list_experiment_results returns a normalized "
            "aggregate row for metric symbolic_policy_score, candidate_id "
            f"crafter:{candidate_id}, sample size {self.train_seed_count()}, and this "
            "task before task done.",
            "reports/ exists and reports/final_report.md summarizes all evaluated candidates, per-achievement frequencies, baseline score, candidate score, delta, seed count, candidate path, and artifact paths.",
            "A live candidate progression report WorkProduct with exact title `Crafter candidate progression` is published with all evaluated candidates, per-achievement frequencies, and runtime message ids.",
            "A final report WorkProduct with exact title `Crafter candidate evidence report` is published with publish_report_work_product before task done.",
            "Candidate code and eval artifacts are pushed to git-server via workspace_push before task done.",
            "Worker sends baseline and candidate score updates via create_runtime_message "
            "or an exposed legacy message alias before task done, and records the returned message ids in reports/final_report.md.",
        ]

    def task_instructions_markdown(self) -> str:
        candidate_lines = "\n".join(
            f"- `{self.candidate_dir(candidate_id)}/heuristic_policy.py` (worker {index})"
            for index, candidate_id in enumerate(self.candidate_ids, start=1)
        )
        return f"""# Task Instructions

{self.git_server_collaboration_brief(role="worker")}

{self.runtime_messaging_collaboration_brief(role="worker")}

Candidate hillclimb: the orchestrator assigns {len(self.candidate_ids)}
total worker task(s), staged at up to {self.parallel_worker_count} at a time.
Each worker owns one candidate path:

{candidate_lines}

The orchestrator compares completed candidate reports and reviews the best
non-baseline candidate. Workers must publish a candidate evidence report
WorkProduct before marking their task done. In
multi-candidate runs, each worker must use the exact candidate path from its
task brief; the examples below use `<candidate_id>` and must not be copied as
`attempt_1` unless the assigned candidate is `attempt_1`.

Read `TASK_README.md`, `STARTING_CONTAINER.md`, `CRAFTER_POLICY_CONTEXT.md`,
and `task_contract.json` first.

Train seeds: {self.train_seeds_csv()}.

```bash
mkdir -p candidates/crafter/<candidate_id>
cp workspace/crafter_proof_policy.py candidates/crafter/<candidate_id>/heuristic_policy.py
python3 workspace/run_hillclimb_symbolicbench_task.py run --output-root . --env crafter --iterations 0 --candidate-root candidates --candidate-id <candidate_id> --train-seeds {self.train_seeds_cli()} --strict-env
```

Do not run the benchmark `score` subcommand inside the worker loop. Use
`artifacts/workproduct_container/eval_summary.json` as the scoring authority for
experiment rows, objective progress, and report WorkProduct publication.
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

PROOF_TRAIN_SEEDS: tuple[int, ...] = (101, 103, 105, 107, 109, 111, 113, 127)


def config_from_cli_args(args: argparse.Namespace) -> CrafterDeoRunConfig:
    candidate_ids = tuple(
        candidate.strip()
        for candidate in str(args.candidate_ids or "").split(",")
        if candidate.strip()
    )
    if not candidate_ids:
        candidate_ids = DEFAULT_CONFIG.candidate_ids
    parallel_worker_count = args.parallel_worker_count
    if parallel_worker_count is None:
        parallel_worker_count = DEFAULT_CONFIG.parallel_worker_count
    parallel_worker_count = max(1, int(parallel_worker_count))
    if parallel_worker_count > len(candidate_ids):
        raise SystemExit(
            "--parallel-worker-count must be less than or equal to the "
            "--candidate-ids count for this Crafter proof lane"
        )
    train_seeds = DEFAULT_CONFIG.train_seeds
    if getattr(args, "proof", False):
        train_seeds = PROOF_TRAIN_SEEDS
    elif getattr(args, "train_seeds", None):
        parsed = tuple(
            int(seed.strip()) for seed in str(args.train_seeds).split(",") if seed.strip()
        )
        if parsed:
            train_seeds = parsed
    orchestrator_profile_id = (
        str(args.orchestrator_profile).strip()
        if getattr(args, "orchestrator_profile", None)
        else DEFAULT_CONFIG.orchestrator_profile_id
    )
    worker_profile_id = (
        str(args.worker_profile).strip()
        if getattr(args, "worker_profile", None)
        else DEFAULT_CONFIG.worker_profile_id
    )
    return CrafterDeoRunConfig(
        parallel_worker_count=parallel_worker_count,
        candidate_ids=candidate_ids,
        train_seeds=train_seeds,
        orchestrator_profile_id=orchestrator_profile_id,
        worker_profile_id=worker_profile_id,
        run_timebox_seconds=(
            int(args.run_timebox_seconds)
            if getattr(args, "run_timebox_seconds", None)
            else DEFAULT_CONFIG.run_timebox_seconds
        ),
        poll_timebox_seconds=(
            int(args.poll_timebox_seconds)
            if getattr(args, "poll_timebox_seconds", None)
            else DEFAULT_CONFIG.poll_timebox_seconds
        ),
    )


def load_lane_objective_spec() -> dict[str, Any]:
    """Load ``[smr.objective]`` from the lane task.toml for orchestrator-facing spec text."""

    if not TASK_TOML.is_file():
        return {}
    raw = tomllib.loads(TASK_TOML.read_text(encoding="utf-8"))
    objective = raw.get("smr", {}).get("objective")
    return dict(objective) if isinstance(objective, dict) else {}


def _objective_field(obj: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = obj.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _summarize_objectives(objectives: list[dict[str, Any]]) -> str:
    if not objectives:
        return "objectives=0"
    parts: list[str] = []
    for row in objectives[:4]:
        objective_id = _objective_field(
            row,
            "objective_id",
            "directed_effort_outcome_id",
            "open_ended_question_id",
            "id",
        )
        short_id = objective_id[:8] if objective_id else "?"
        state = _objective_field(row, "evaluation_state", "status") or "unknown"
        title = _objective_field(row, "title")[:36]
        parts.append(f"{short_id}:{state}:{title}")
    suffix = f" (+{len(objectives) - 4} more)" if len(objectives) > 4 else ""
    return f"objectives={len(objectives)} [{'; '.join(parts)}]{suffix}"


def fetch_objective_poll_line(client: Any, project_id: str, run_id: str) -> str:
    try:
        objectives = client.list_objectives(project_id, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        return f"objectives=unavailable err={type(exc).__name__}"
    if not isinstance(objectives, list):
        return "objectives=unavailable err=invalid_list_payload"
    line = _summarize_objectives([row for row in objectives if isinstance(row, dict)])
    if not objectives:
        return line
    first = objectives[0]
    if not isinstance(first, dict):
        return line
    objective_id = _objective_field(
        first,
        "objective_id",
        "directed_effort_outcome_id",
        "open_ended_question_id",
        "id",
    )
    if not objective_id:
        return line
    kind = _objective_field(first, "objective_kind", "kind") or "directed_effort_outcome"
    try:
        progress = client.get_objective_progress(project_id, objective_id, kind=kind)
    except Exception as exc:  # noqa: BLE001
        return f"{line} progress=unavailable err={type(exc).__name__}"
    if isinstance(progress, dict):
        status = _objective_field(progress, "status", "evaluation_state", "state")
        milestone_count = progress.get("milestone_count")
        claim_count = progress.get("claim_count") or progress.get("progress_count")
        bits = [
            bit for bit in (status, f"milestones={milestone_count}", f"claims={claim_count}") if bit
        ]
        if bits:
            return f"{line} progress({'; '.join(str(b) for b in bits)})"
    return line


def fetch_objective_progress_snapshot(client: Any, project_id: str, run_id: str) -> dict[str, Any]:
    try:
        objectives = client.list_objectives(project_id, run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "line": f"objectives=unavailable err={type(exc).__name__}",
            "objectives": [],
            "progress_by_objective_id": {},
            "errors": [str(exc)],
        }
    if not isinstance(objectives, list):
        return {
            "line": "objectives=unavailable err=invalid_list_payload",
            "objectives": [],
            "progress_by_objective_id": {},
            "errors": ["invalid_list_payload"],
        }

    objective_rows = [row for row in objectives if isinstance(row, dict)]
    progress_by_objective_id: dict[str, Any] = {}
    errors: list[str] = []
    for row in objective_rows:
        objective_id = _objective_field(
            row,
            "objective_id",
            "directed_effort_outcome_id",
            "open_ended_question_id",
            "id",
        )
        if not objective_id:
            continue
        kind = _objective_field(row, "objective_kind", "kind") or "directed_effort_outcome"
        try:
            progress_by_objective_id[objective_id] = client.get_objective_progress(
                project_id,
                objective_id,
                kind=kind,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{objective_id}:{type(exc).__name__}:{exc}")

    return {
        "line": fetch_objective_poll_line(client, project_id, run_id),
        "objectives": objective_rows,
        "progress_by_objective_id": progress_by_objective_id,
        "errors": errors,
    }


def summarize_objective_review_gate(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    objectives = [row for row in snapshot.get("objectives", []) if isinstance(row, Mapping)]
    pending: list[dict[str, str]] = []
    reviewed: list[dict[str, str]] = []
    for row in objectives:
        objective_id = _objective_field(
            row,
            "objective_id",
            "directed_effort_outcome_id",
            "open_ended_question_id",
            "id",
        )
        state = (_objective_field(row, "evaluation_state", "state") or "").lower()
        review_summary = _objective_field(row, "review_summary") or ""
        entry = {
            "objective_id": objective_id,
            "evaluation_state": state,
            "review_summary": review_summary,
        }
        if state in {"satisfied", "partial", "needs_revision", "failed"} and review_summary:
            reviewed.append(entry)
        else:
            pending.append(entry)
    return {
        "required": bool(objectives),
        "complete": bool(objectives) and not pending,
        "reviewed_count": len(reviewed),
        "pending_count": len(pending),
        "pending": pending,
        "reviewed": reviewed,
    }


def summarize_objective_review_gate_line(gate: Mapping[str, Any]) -> str:
    pending = [item for item in gate.get("pending", []) if isinstance(item, Mapping)]
    if not gate.get("required"):
        return "objective_review required=false objectives=0"
    if gate.get("complete"):
        return f"objective_review complete=true reviewed={gate.get('reviewed_count', 0)}"
    parts = [
        f"{item.get('objective_id') or '?'}:{item.get('evaluation_state') or 'unknown'}"
        for item in pending[:4]
    ]
    suffix = f" (+{len(pending) - 4} more)" if len(pending) > 4 else ""
    return f"objective_review complete=false pending={len(pending)} [{'; '.join(parts)}]{suffix}"


@dataclass(frozen=True)
class _GitBranchPollState:
    name: str
    head_commit_sha: str
    summary: str
    merged_into_default: bool | None


@dataclass(frozen=True)
class _GitPollState:
    branch: str
    default_branch: str
    head_commit_sha: str
    tree_paths: frozenset[str]
    commit_count: int
    unmerged_branches: tuple[_GitBranchPollState, ...]


def _git_poll_tree_paths(git_status: Mapping[str, Any] | None) -> set[str]:
    if not git_status:
        return set()
    raw_paths = git_status.get("tree_paths")
    if not isinstance(raw_paths, list):
        return set()
    paths: set[str] = set()
    for raw_path in raw_paths:
        normalized = _normalize_workspace_compare_path(str(raw_path or ""))
        if not normalized or _is_workspace_noise_path(normalized):
            continue
        if normalized.startswith("starting-data/"):
            continue
        paths.add(normalized)
    return paths


def _git_poll_state_from_status(git_status: Mapping[str, Any] | None) -> _GitPollState | None:
    if not git_status:
        return None
    recent_commits = git_status.get("recent_commits")
    commit_count = len(recent_commits) if isinstance(recent_commits, list) else 0
    raw_unmerged = git_status.get("unmerged_branches")
    unmerged_branches: list[_GitBranchPollState] = []
    if isinstance(raw_unmerged, list):
        for raw_branch in raw_unmerged:
            if not isinstance(raw_branch, Mapping):
                continue
            name = str(raw_branch.get("name") or "").strip()
            if not name:
                continue
            raw_merged = raw_branch.get("merged_into_default")
            merged_into_default = raw_merged if isinstance(raw_merged, bool) else None
            unmerged_branches.append(
                _GitBranchPollState(
                    name=name,
                    head_commit_sha=str(raw_branch.get("head_commit_sha") or "").strip(),
                    summary=str(raw_branch.get("summary") or "").strip(),
                    merged_into_default=merged_into_default,
                )
            )
    return _GitPollState(
        branch=str(git_status.get("branch") or "").strip(),
        default_branch=str(git_status.get("default_branch") or "").strip(),
        head_commit_sha=str(git_status.get("head_commit_sha") or "").strip(),
        tree_paths=frozenset(_git_poll_tree_paths(git_status)),
        commit_count=commit_count,
        unmerged_branches=tuple(unmerged_branches),
    )


def _git_branch_ref(branch: _GitBranchPollState) -> str:
    head = branch.head_commit_sha[:12] if branch.head_commit_sha else "-"
    return f"{branch.name}@{head}"


def _git_poll_state_summary(state: _GitPollState) -> dict[str, Any]:
    return {
        "branch": state.branch,
        "default_branch": state.default_branch,
        "head_commit_sha": state.head_commit_sha,
        "tree_path_count": len(state.tree_paths),
        "commit_count": state.commit_count,
        "unmerged_branch_count": len(state.unmerged_branches),
        "unmerged_branches": [
            {
                "name": branch.name,
                "head_commit_sha": branch.head_commit_sha,
                "summary": branch.summary,
                "merged_into_default": branch.merged_into_default,
            }
            for branch in state.unmerged_branches
        ],
    }


def summarize_git_status_snapshot(git_status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    state = _git_poll_state_from_status(git_status)
    if state is None:
        return None
    recent_commits = git_status.get("recent_commits") if git_status else None
    recent_commit_summaries: list[dict[str, Any]] = []
    if isinstance(recent_commits, list):
        for commit in recent_commits[:5]:
            if not isinstance(commit, Mapping):
                continue
            recent_commit_summaries.append(
                {
                    "sha": str(commit.get("sha") or commit.get("commit_sha") or "")[:12],
                    "subject": commit.get("subject") or commit.get("message"),
                }
            )
    snapshot = _git_poll_state_summary(state)
    snapshot.update(
        {
            "tree_paths_sample": sorted(state.tree_paths)[:20],
            "tree_truncated": bool(git_status.get("tree_truncated")) if git_status else False,
            "recent_commits": recent_commit_summaries,
        }
    )
    return snapshot


def _read_archive_json(archive_path: Path, member_path: str) -> Any:
    candidates = (member_path, f"./{member_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        for candidate in candidates:
            try:
                member = archive.extractfile(candidate)
            except KeyError:
                continue
            if member is None:
                continue
            with member:
                return json.loads(member.read().decode("utf-8"))
    return None


def _crafter_candidate_result_row(record: Mapping[str, Any]) -> dict[str, Any]:
    train = record.get("train") if isinstance(record.get("train"), Mapping) else {}
    train_summary = train.get("summary") if isinstance(train.get("summary"), Mapping) else {}
    experiment_contract = (
        record.get("experiment_contract")
        if isinstance(record.get("experiment_contract"), Mapping)
        else {}
    )
    result_summary = (
        experiment_contract.get("result_summary")
        if isinstance(experiment_contract.get("result_summary"), Mapping)
        else {}
    )
    reward = train_summary.get("reward") if isinstance(train_summary.get("reward"), Mapping) else {}
    achievement_frequency = train_summary.get("achievement_frequency")
    if not isinstance(achievement_frequency, Mapping):
        achievement_frequency = {}
    return {
        "candidate_id": record.get("candidate_id"),
        "source_kind": record.get("source_kind"),
        "status": record.get("status"),
        "accepted": record.get("accepted"),
        "score": record.get("score"),
        "score_delta": record.get("score_delta"),
        "baseline_value": record.get("baseline_value"),
        "candidate_hash": record.get("candidate_hash"),
        "candidate_policy_path": record.get("candidate_policy_path"),
        "score_source": record.get("score_source"),
        "metric": result_summary.get("metric"),
        "sample_size": result_summary.get("sample_size") or train_summary.get("seed_count"),
        "achievement_count_mean": train_summary.get("achievement_count_mean"),
        "achievement_frequency": dict(achievement_frequency),
        "reward_mean": reward.get("mean"),
        "reward_median": reward.get("median"),
        "result_artifacts": record.get("result_artifacts"),
    }


def summarize_crafter_results_archive(archive_path: Path) -> dict[str, Any] | None:
    eval_summary = _read_archive_json(
        archive_path,
        "artifacts/workproduct_container/eval_summary.json",
    )
    if not isinstance(eval_summary, Mapping):
        return None
    raw_records = eval_summary.get("records")
    records = (
        [
            _crafter_candidate_result_row(record)
            for record in raw_records
            if isinstance(record, Mapping)
        ]
        if isinstance(raw_records, list)
        else []
    )
    achievement_diversity = _read_archive_json(
        archive_path,
        "artifacts/workproduct_container/achievement_diversity.json",
    )
    return {
        "schema_version": eval_summary.get("schema_version"),
        "baseline_score": eval_summary.get("baseline_score"),
        "best_candidate_id": eval_summary.get("best_candidate_id"),
        "best_score": eval_summary.get("best_score"),
        "best_score_delta": eval_summary.get("best_score_delta"),
        "best_source_kind": eval_summary.get("best_source_kind"),
        "score_source": eval_summary.get("score_source"),
        "candidate_count": eval_summary.get("candidate_count"),
        "completed_candidate_count": eval_summary.get("completed_candidate_count"),
        "completed_non_baseline_candidate_count": eval_summary.get(
            "completed_non_baseline_candidate_count"
        ),
        "records": records,
        "achievement_diversity": (
            achievement_diversity if isinstance(achievement_diversity, Mapping) else None
        ),
    }


def summarize_crafter_results_line(results: Mapping[str, Any] | None) -> str:
    if not results:
        return "results=unavailable"
    best_candidate_id = str(results.get("best_candidate_id") or "-")
    baseline = results.get("baseline_score")
    best = results.get("best_score")
    delta = results.get("best_score_delta")
    records = results.get("records")
    achievement_count = "-"
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if str(record.get("candidate_id") or "") != best_candidate_id:
                continue
            achievements = record.get("achievement_frequency")
            if isinstance(achievements, Mapping):
                achievement_count = str(len(achievements))
            break
    return (
        f"best={best_candidate_id} baseline={baseline} score={best} "
        f"delta={delta} achievements={achievement_count}"
    )


def _format_git_path_delta(
    added: list[str],
    removed: list[str],
    *,
    limit: int = 6,
) -> str:
    parts: list[str] = []
    for path in added[:limit]:
        parts.append(f"++{path}")
    remaining = max(0, limit - len(parts))
    for path in removed[:remaining]:
        parts.append(f"--{path}")
    hidden = max(0, len(added) + len(removed) - len(parts))
    if hidden:
        parts.append(f"+{hidden} more")
    return " ".join(parts)


def summarize_git_poll_line(
    git_status: Mapping[str, Any] | None,
    *,
    previous: _GitPollState | None = None,
) -> str:
    state = _git_poll_state_from_status(git_status)
    if state is None:
        return "git=unavailable"
    head = state.head_commit_sha[:12] if state.head_commit_sha else "-"
    bits = [
        f"branch={state.branch or '-'}",
        f"default={state.default_branch or '-'}",
        f"head={head}",
        f"commits={state.commit_count}",
        f"tree={len(state.tree_paths)}",
    ]
    if state.unmerged_branches:
        top = " ".join(_git_branch_ref(branch) for branch in state.unmerged_branches[:3])
        hidden = len(state.unmerged_branches) - 3
        bits.append(f"unmerged={len(state.unmerged_branches)}")
        bits.append(f"top={top}" + (f" +{hidden} more" if hidden > 0 else ""))
    else:
        bits.append("unmerged=0")
    if previous is not None:
        added = sorted(state.tree_paths - previous.tree_paths)
        removed = sorted(previous.tree_paths - state.tree_paths)
        delta = _format_git_path_delta(added, removed)
        if delta:
            bits.append(delta)
        elif state.head_commit_sha and state.head_commit_sha != previous.head_commit_sha:
            recent_commits = git_status.get("recent_commits") if git_status else None
            subject = ""
            if isinstance(recent_commits, list) and recent_commits:
                first = recent_commits[0]
                if isinstance(first, Mapping):
                    subject = str(first.get("subject") or first.get("message") or "").strip()
            if subject:
                bits.append(f'commit="{subject[:48]}"')
    return " ".join(bits)


def fetch_git_poll_line(
    client: Any,
    project_id: str,
    *,
    previous: _GitPollState | None = None,
    branch: str | None = None,
    max_tree_entries: int = 200,
) -> tuple[str, _GitPollState | None]:
    git_status = _fetch_project_git_status(
        client,
        project_id,
        branch=branch,
        max_tree_entries=max_tree_entries,
    )
    state = _git_poll_state_from_status(git_status)
    return summarize_git_poll_line(git_status, previous=previous), state


def build_git_poll_callback(
    client: Any,
    project_id: str,
    *,
    log: LogFn,
    min_interval_s: float = 15.0,
) -> tuple[Callable[[Any], None], Callable[[], list[dict[str, Any]]]]:
    last_poll = 0.0
    last_state: _GitPollState | None = None
    last_line = ""
    snapshots: list[dict[str, Any]] = []

    def on_snapshot(_snapshot: Any) -> None:
        nonlocal last_poll, last_state, last_line
        now = time.monotonic()
        if now - last_poll < min_interval_s:
            return
        last_poll = now
        try:
            line, state = fetch_git_poll_line(client, project_id, previous=last_state)
        except Exception as exc:  # noqa: BLE001
            line = f"git=unavailable err={type(exc).__name__}"
            state = None
        if line == last_line:
            return
        last_line = line
        if state is not None:
            last_state = state
        log(f"[git] {line}")
        snapshots.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "line": line,
                "state": _git_poll_state_summary(state) if state is not None else None,
            }
        )

    return on_snapshot, lambda: list(snapshots)


def _clip_message_text(value: Any, *, limit: int = 96) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _format_runtime_message_row(row: Mapping[str, Any]) -> str:
    sender = str(row.get("sender") or "?").strip() or "?"
    target = str(row.get("target") or "broadcast").strip() or "broadcast"
    status = str(row.get("status") or "unknown").strip() or "unknown"
    mode = str(row.get("mode") or "").strip()
    mode_bit = f" mode={mode}" if mode else ""
    body = _clip_message_text(row.get("body"))
    if body:
        return f'{sender}→{target}:{status}{mode_bit} "{body}"'
    return f"{sender}→{target}:{status}{mode_bit}"


def summarize_runtime_messages_poll_line(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "total=0"
    by_status: dict[str, int] = {}
    for row in messages:
        status = str(row.get("status") or "unknown").strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    status_bits = " ".join(f"{key}={value}" for key, value in sorted(by_status.items()))
    return f"total={len(messages)} {status_bits}"


def fetch_runtime_messages(
    client: Any,
    project_id: str,
    run_id: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(source: str, candidates: Any) -> None:
        if not isinstance(candidates, list):
            return
        for row in candidates:
            if not isinstance(row, dict):
                continue
            message_id = str(
                row.get("message_id") or row.get("id") or row.get("runtime_message_id") or ""
            ).strip()
            dedupe_key = message_id or f"{source}:{len(rows)}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            item = dict(row)
            item.setdefault("message_source", source)
            rows.append(item)

    with contextlib.suppress(Exception):
        add(
            "runtime_messages",
            client.list_project_run_runtime_messages(project_id, run_id, limit=limit),
        )
    with contextlib.suppress(Exception):
        add("manderqueue", client.list_messages(run_id, project_id=project_id, limit=limit))
    return rows


def fetch_message_poll_line(
    client: Any,
    project_id: str,
    run_id: str,
) -> str:
    try:
        messages = fetch_runtime_messages(client, project_id, run_id)
    except Exception as exc:  # noqa: BLE001
        return f"messages=unavailable err={type(exc).__name__}"
    return summarize_runtime_messages_poll_line(messages)


def seed_driver_runtime_message(
    client: Any,
    *,
    project_id: str,
    run_id: str,
    config: CrafterDeoRunConfig,
) -> dict[str, Any]:
    body = (
        "Crafter DEO proof driver checkpoint: run triggered with "
        f"{', '.join(config.candidate_ids)} on {config.train_seed_count()} proof seeds. "
        "Expected durable artifacts are reports/final_report.md, a ready report "
        "WorkProduct, pushed workspace evidence, and candidate metrics."
    )
    message = client.send_message(
        run_id,
        project_id=project_id,
        intent="queue",
        audience={"kind": "run"},
        body=body,
        payload={
            "source": "crafter_deo_run_driver",
            "project_id": project_id,
            "run_id": run_id,
            "candidate_ids": list(config.candidate_ids),
            "seed_count": config.train_seed_count(),
            "required_artifacts": [
                "reports/final_report.md",
                "artifacts/workproduct_container/eval_summary.json",
                *config.required_candidate_paths(),
            ],
        },
        message_kind="runtime_message",
        fallback_policy="block",
    )
    reflected_messages: list[dict[str, Any]] = []
    for attempt in range(1, 7):
        reflected_messages = fetch_runtime_messages(client, project_id, run_id)
        if reflected_messages:
            break
        if attempt < 6:
            time.sleep(0.5)
    if not reflected_messages:
        raise ResearchApiError(
            "driver runtime message was accepted but did not appear in "
            "list_messages/list_project_run_runtime_messages"
        )
    return {
        "message": message,
        "reflected_count": len(reflected_messages),
        "poll_line": summarize_runtime_messages_poll_line(reflected_messages),
    }


def build_message_poll_callback(
    client: Any,
    project_id: str,
    run_id: str,
    *,
    log: LogFn,
    min_interval_s: float = 15.0,
) -> tuple[Callable[[Any], None], Callable[[], list[dict[str, Any]]]]:
    last_poll = 0.0
    seen_message_ids: set[str] = set()
    last_summary = ""
    snapshots: list[dict[str, Any]] = []
    try:
        bootstrap_messages = fetch_runtime_messages(client, project_id, run_id)
        for row in bootstrap_messages:
            message_id = str(row.get("message_id") or "").strip()
            if message_id:
                seen_message_ids.add(message_id)
        last_summary = summarize_runtime_messages_poll_line(bootstrap_messages)
    except Exception:
        last_summary = ""

    def on_snapshot(_snapshot: Any) -> None:
        nonlocal last_poll, last_summary
        now = time.monotonic()
        if now - last_poll < min_interval_s:
            return
        last_poll = now
        try:
            messages = fetch_runtime_messages(client, project_id, run_id)
        except Exception as exc:  # noqa: BLE001
            line = f"messages=unavailable err={type(exc).__name__}"
            if line != last_summary:
                last_summary = line
                log(f"[msg] {line}")
                snapshots.append({"at": datetime.now(UTC).isoformat(), "line": line})
            return

        for row in sorted(messages, key=lambda item: int(item.get("seq") or 0)):
            message_id = str(row.get("message_id") or "").strip()
            if not message_id or message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
            line = _format_runtime_message_row(row)
            log(f"[msg] new {line}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": line,
                    "message_id": message_id,
                }
            )

        summary = summarize_runtime_messages_poll_line(messages)
        if summary != last_summary:
            last_summary = summary
            log(f"[msg] {summary}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": summary,
                    "kind": "summary",
                }
            )

    return on_snapshot, lambda: list(snapshots)


def _format_work_product_row(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or row.get("name") or "?").strip() or "?"
    status = str(row.get("status") or row.get("state") or "unknown").strip() or "unknown"
    kind = str(row.get("kind") or row.get("work_product_kind") or "").strip()
    kind_bit = f" kind={kind}" if kind else ""
    work_product_id = str(row.get("work_product_id") or row.get("id") or "").strip()
    id_bit = f" id={work_product_id[:8]}" if work_product_id else ""
    return f"{title}:{status}{kind_bit}{id_bit}"


def summarize_work_products_poll_line(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "total=0"
    by_status: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or row.get("state") or "unknown").strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    status_bits = " ".join(f"{key}={value}" for key, value in sorted(by_status.items()))
    return f"total={len(rows)} {status_bits}"


def fetch_work_products_poll_line(client: Any, project_id: str, run_id: str) -> str:
    try:
        rows = client.list_run_work_products(project_id, run_id)
    except Exception as exc:  # noqa: BLE001
        return f"work_products=unavailable err={type(exc).__name__}"
    if not isinstance(rows, list):
        return "work_products=unavailable err=invalid_list_payload"
    return summarize_work_products_poll_line([row for row in rows if isinstance(row, dict)])


def fetch_work_products_snapshot(client: Any, project_id: str, run_id: str) -> dict[str, Any]:
    try:
        rows = client.list_run_work_products(project_id, run_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "line": f"work_products=unavailable err={type(exc).__name__}",
            "work_products": [],
            "errors": [str(exc)],
        }
    if not isinstance(rows, list):
        return {
            "line": "work_products=unavailable err=invalid_list_payload",
            "work_products": [],
            "errors": ["invalid_list_payload"],
        }
    work_products = [row for row in rows if isinstance(row, dict)]
    return {
        "line": summarize_work_products_poll_line(work_products),
        "work_products": work_products,
        "errors": [],
    }


def build_work_product_poll_callback(
    client: Any,
    project_id: str,
    run_id: str,
    *,
    log: LogFn,
    min_interval_s: float = 15.0,
) -> tuple[Callable[[Any], None], Callable[[], list[dict[str, Any]]]]:
    last_poll = 0.0
    seen_work_product_ids: set[str] = set()
    last_summary = ""
    snapshots: list[dict[str, Any]] = []

    def on_snapshot(_snapshot: Any) -> None:
        nonlocal last_poll, last_summary
        now = time.monotonic()
        if now - last_poll < min_interval_s:
            return
        last_poll = now
        try:
            rows = client.list_run_work_products(project_id, run_id)
        except Exception as exc:  # noqa: BLE001
            line = f"work_products=unavailable err={type(exc).__name__}"
            if line != last_summary:
                last_summary = line
                log(f"[wp] {line}")
                snapshots.append({"at": datetime.now(UTC).isoformat(), "line": line})
            return
        if not isinstance(rows, list):
            return
        for row in sorted(rows, key=lambda item: str(item.get("created_at") or "")):
            if not isinstance(row, dict):
                continue
            work_product_id = str(row.get("work_product_id") or row.get("id") or "").strip()
            if not work_product_id or work_product_id in seen_work_product_ids:
                continue
            seen_work_product_ids.add(work_product_id)
            line = _format_work_product_row(row)
            log(f"[wp] new {line}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": line,
                    "work_product_id": work_product_id,
                }
            )
        summary = summarize_work_products_poll_line([row for row in rows if isinstance(row, dict)])
        if summary != last_summary:
            last_summary = summary
            log(f"[wp] {summary}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": summary,
                    "kind": "summary",
                }
            )

    return on_snapshot, lambda: list(snapshots)


def _usage_row_as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    row_dict = getattr(row, "__dict__", None)
    if isinstance(row_dict, dict):
        return dict(row_dict)
    return {}


def _usage_actor_key(row: Mapping[str, Any]) -> str:
    actor_id = str(row.get("actor_id") or "").strip()
    if actor_id:
        return actor_id
    worker_id = str(row.get("worker_id") or "").strip()
    if worker_id:
        return worker_id
    role = str(row.get("participant_role") or "unknown").strip() or "unknown"
    return role


def _usage_signature(row: Mapping[str, Any]) -> tuple[int, int, int]:
    event_count = int(row.get("event_count") or 0)
    token_total = _actor_token_total(dict(row)) or 0
    billed_cents = int(row.get("billed_amount_cents") or 0)
    return event_count, token_total, billed_cents


def _usage_model_label(row: Mapping[str, Any]) -> str:
    by_model = row.get("by_model")
    if isinstance(by_model, dict):
        candidates = [
            (str(key), float(value))
            for key, value in by_model.items()
            if str(key).strip()
            and str(key).strip().lower() != "unknown"
            and isinstance(value, (int, float))
        ]
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]
    return ""


def _format_actor_usage_update(row: Mapping[str, Any]) -> str:
    role = str(row.get("participant_role") or "?").strip() or "?"
    actor_id = str(row.get("actor_id") or "?").strip()
    actor_short = actor_id[:8] if actor_id else "?"
    events = int(row.get("event_count") or 0)
    tokens = _format_token_count(_actor_token_total(dict(row)))
    cost = _format_usd(_actor_cost_usd(dict(row)))
    model = _usage_model_label(row)
    model_bit = f" model={model}" if model else ""
    task_key = str(row.get("task_key") or "").strip()
    task_bit = f" task={task_key[:24]}" if task_key else ""
    return f"{role} {actor_short}{task_bit} events={events} tokens={tokens} cost={cost}{model_bit}"


def summarize_usage_poll_line(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "actors=0 events=0 tokens=0 cost=$0.0000"
    total_events = 0
    total_tokens = 0
    total_cost = 0.0
    for row in rows:
        total_events += int(row.get("event_count") or 0)
        token_total = _actor_token_total(row)
        if isinstance(token_total, int):
            total_tokens += token_total
        cost = _actor_cost_usd(row)
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    return (
        f"actors={len(rows)} events={total_events} "
        f"tokens={_format_token_count(total_tokens)} cost={_format_usd(total_cost)}"
    )


def fetch_usage_poll_lines(
    client: Any,
    project_id: str,
    run_id: str,
    previous: Mapping[str, tuple[int, int, int]],
) -> tuple[list[str], str, dict[str, tuple[int, int, int]]]:
    try:
        usage = client.get_project_run_actor_usage(project_id, run_id)
        actor_rows = getattr(usage, "actors", ()) or ()
    except Exception as exc:  # noqa: BLE001
        return [], f"usage=unavailable err={type(exc).__name__}", dict(previous)

    rows = [_usage_row_as_dict(row) for row in actor_rows]
    rows = [row for row in rows if row]
    new_lines: list[str] = []
    next_previous: dict[str, tuple[int, int, int]] = dict(previous)
    for row in sorted(
        rows, key=lambda item: (_usage_actor_key(item), str(item.get("actor_id") or ""))
    ):
        actor_key = _usage_actor_key(row)
        signature = _usage_signature(row)
        prior = previous.get(actor_key)
        if prior == signature:
            next_previous[actor_key] = signature
            continue
        next_previous[actor_key] = signature
        if prior is None and signature == (0, 0, 0):
            continue
        new_lines.append(_format_actor_usage_update(row))

    return new_lines, summarize_usage_poll_line(rows), next_previous


def build_usage_poll_callback(
    client: Any,
    project_id: str,
    run_id: str,
    *,
    log: LogFn,
    min_interval_s: float = 15.0,
) -> tuple[Callable[[Any], None], Callable[[], list[dict[str, Any]]]]:
    last_poll = 0.0
    last_summary = ""
    previous_signatures: dict[str, tuple[int, int, int]] = {}
    snapshots: list[dict[str, Any]] = []

    def on_snapshot(_snapshot: Any) -> None:
        nonlocal last_poll, last_summary, previous_signatures
        now = time.monotonic()
        if now - last_poll < min_interval_s:
            return
        last_poll = now
        new_lines, summary, previous_signatures = fetch_usage_poll_lines(
            client,
            project_id,
            run_id,
            previous_signatures,
        )
        for line in new_lines:
            log(f"[usage] {line}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": line,
                    "kind": "actor",
                }
            )
        if summary != last_summary:
            last_summary = summary
            log(f"[usage] {summary}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": summary,
                    "kind": "summary",
                }
            )

    return on_snapshot, lambda: list(snapshots)


def summarize_actor_trace_counts(actor_states: Mapping[str, ActorTraceState]) -> str:
    tool_total = sum(state.tool_calls for state in actor_states.values())
    reasoning_total = sum(state.reasoning_events for state in actor_states.values())
    transcript_total = sum(state.transcript_events for state in actor_states.values())
    return (
        f"tracked={len(actor_states)} transcript={transcript_total} "
        f"tools={tool_total} reasoning={reasoning_total}"
    )


def fetch_actor_trace_poll_lines(
    client: Any,
    project_id: str,
    run_id: str,
    actor_states: dict[str, ActorTraceState],
    *,
    trace_limit: int = 100,
    transcript_view: str = "debug",
    allowed_kinds: frozenset[str] = DEFAULT_PROGRESS_KINDS,
) -> tuple[list[str], str]:
    from readme_runs.smr_slot_client import actor_trace_key  # noqa: PLC0415

    new_lines: list[str] = []
    try:
        usage = client.get_project_run_actor_usage(project_id, run_id)
        usage_by_actor = {str(row.actor_id): row for row in getattr(usage, "actors", ()) or ()}
        actors = _load_actors(client, project_id, run_id)
    except Exception as exc:  # noqa: BLE001
        return [], f"actors=unavailable err={type(exc).__name__}"

    for actor in actors:
        actor_id = str(actor.get("actor_id") or "").strip()
        trace_key = actor_trace_key(dict(actor))
        session_id = _participant_session_id(actor, usage_by_actor)
        state_key = actor_id or trace_key or session_id or "unknown"
        state = actor_states.setdefault(
            state_key,
            ActorTraceState(
                actor_key=trace_key or actor_id,
                actor_id=actor_id,
                participant_role=str(actor.get("participant_role") or ""),
                participant_session_id=session_id,
                actor_state=str(actor.get("state") or ""),
            ),
        )
        state.actor_id = actor_id or state.actor_id
        state.participant_role = str(actor.get("participant_role") or state.participant_role)
        state.actor_state = str(actor.get("state") or state.actor_state)
        if session_id:
            state.participant_session_id = session_id
        if not session_id:
            continue

        trace_payload = _fetch_transcript_events(
            client,
            run_id=run_id,
            participant_session_id=session_id,
            state=state,
            trace_limit=trace_limit,
            transcript_view=transcript_view,
        )
        for event in _iter_trace_events(trace_payload):
            event_id = _event_id(event)
            kind = _event_kind(event)
            if event_id and event_id in state.seen_event_ids:
                continue
            if not _kind_allowed(kind, allowed_kinds):
                if event_id:
                    state.seen_event_ids.add(event_id)
                continue
            if event_id:
                state.seen_event_ids.add(event_id)
            state.transcript_events += 1
            if _is_tool_event_kind(kind):
                state.tool_calls += 1
            if kind == "reasoning.summary":
                state.reasoning_events += 1
            if kind == "token.usage":
                payload = event.get("payload")
                if isinstance(payload, Mapping):
                    total = payload.get("total_tokens") or payload.get("tokens")
                    if isinstance(total, int) and not isinstance(total, bool):
                        state.token_usage_total = max(state.token_usage_total, total)
                        state.token_usage_events += 1
            role = str(event.get("participant_role") or state.participant_role or "?")
            new_lines.append(
                _format_event_line(role=role, actor_id=state.actor_id or trace_key, event=event)
            )

    return new_lines, summarize_actor_trace_counts(actor_states)


def build_actor_trace_poll_callback(
    client: Any,
    project_id: str,
    run_id: str,
    *,
    log: LogFn,
    min_interval_s: float = 10.0,
    trace_limit: int = 100,
    transcript_view: str = "debug",
) -> tuple[Callable[[Any], None], Callable[[], list[dict[str, Any]]]]:
    actor_states: dict[str, ActorTraceState] = {}
    last_poll = 0.0
    last_summary = ""
    snapshots: list[dict[str, Any]] = []

    def on_snapshot(_snapshot: Any) -> None:
        nonlocal last_poll, last_summary
        now = time.monotonic()
        if now - last_poll < min_interval_s:
            return
        last_poll = now
        try:
            new_lines, summary = fetch_actor_trace_poll_lines(
                client,
                project_id,
                run_id,
                actor_states,
                trace_limit=trace_limit,
                transcript_view=transcript_view,
            )
        except Exception as exc:  # noqa: BLE001
            summary = f"actors=unavailable err={type(exc).__name__}"
            new_lines = []
        for line in new_lines:
            log(f"[trace] {line}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": line,
                    "kind": "event",
                }
            )
        if summary != last_summary:
            last_summary = summary
            log(f"[actor] {summary}")
            snapshots.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "line": summary,
                    "kind": "summary",
                }
            )

    def actor_state_snapshot() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in actor_states.values():
            rows.append(
                {
                    "actor_key": state.actor_key,
                    "actor_id": state.actor_id,
                    "participant_role": state.participant_role,
                    "participant_session_id": state.participant_session_id,
                    "actor_state": state.actor_state,
                    "tool_calls": state.tool_calls,
                    "reasoning_events": state.reasoning_events,
                    "transcript_events": state.transcript_events,
                    "token_usage_total": state.token_usage_total,
                    "token_usage_events": state.token_usage_events,
                }
            )
        return rows

    on_snapshot.actor_state_snapshot = actor_state_snapshot
    return on_snapshot, lambda: list(snapshots)


def build_objective_poll_callback(
    client: Any,
    project_id: str,
    run_id: str,
    *,
    log: LogFn,
    min_interval_s: float = 30.0,
) -> tuple[Callable[[Any], None], Callable[[], list[dict[str, Any]]]]:
    last_poll = 0.0
    snapshots: list[dict[str, Any]] = []

    def on_snapshot(_snapshot: Any) -> None:
        nonlocal last_poll
        now = time.monotonic()
        if now - last_poll < min_interval_s:
            return
        last_poll = now
        line = fetch_objective_poll_line(client, project_id, run_id)
        log(f"[objective] {line}")
        snapshots.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "line": line,
            }
        )

    return on_snapshot, lambda: list(snapshots)


def _strip_launch_primary_parent(bundle: dict[str, Any]) -> None:
    """Crafter DEO expects the orchestrator to author objectives — not launch precreate."""

    trigger_payload = bundle.get("trigger_payload")
    if isinstance(trigger_payload, dict):
        trigger_payload.pop("primary_parent", None)
        trigger_payload.pop("primary_parent_ref", None)
    effective = bundle.get("effective_config")
    if isinstance(effective, dict):
        smr = effective.get("smr")
        if isinstance(smr, dict):
            smr.pop("primary_parent", None)


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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (crafter_runs_dir() / f"{stamp}_{label}").resolve()


def _slot_runtime_status_url(launch: ReadmeSmokeLaunch) -> str:
    """Resolve the local slot runtime status URL from the synth-dev slot contract."""

    if launch.slot_contract is None:
        return ""
    network = getattr(launch.slot_contract, "network", None)
    surfaces = getattr(network, "surfaces", None)
    if isinstance(surfaces, Mapping):
        runtime_surface = (surfaces.get("surfaces") or {}).get("runtime_mcp")
        if isinstance(runtime_surface, Mapping):
            for endpoint in runtime_surface.get("endpoints") or ():
                if not isinstance(endpoint, Mapping):
                    continue
                if endpoint.get("local") is True:
                    url = str(endpoint.get("url") or "").strip()
                    if url:
                        return f"{url.rstrip('/')}/smr/runtime/status"
    task_env = getattr(launch.slot_contract, "task_env", None)
    if isinstance(task_env, Mapping):
        for key in ("SMR_MCP_HOST_URL", "SMR_RUNTIME_MCP_PUBLIC_BASE_URL"):
            url = str(task_env.get(key) or "").strip()
            if url:
                return f"{url.rstrip('/')}/smr/runtime/status"
    return ""


def _fetch_slot_runtime_status(status_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(status_url) as response:  # noqa: S310 - local slot URL.
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {"payload": payload}


def _slot_activity_blockers(status_payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if status_payload.get("busy") is True:
        blockers.append("busy=true")
    active_run_count = int(status_payload.get("active_run_count") or 0)
    if active_run_count > 0:
        blockers.append(f"active_run_count={active_run_count}")
    runtime_activity = status_payload.get("runtime_activity")
    if isinstance(runtime_activity, Mapping):
        activity_count = int(runtime_activity.get("active_runtime_activity_count") or 0)
        if activity_count > 0:
            blockers.append(f"active_runtime_activity_count={activity_count}")
        live_run_ids = [
            str(run_id)
            for run_id in runtime_activity.get("local_docker_live_run_ids") or ()
            if str(run_id).strip()
        ]
        if live_run_ids:
            blockers.append(f"local_docker_live_run_ids={','.join(live_run_ids)}")
        live_containers = [
            str(name)
            for name in runtime_activity.get("local_docker_live_container_names") or ()
            if str(name).strip()
        ]
        if live_containers:
            blockers.append(f"local_docker_live_containers={len(live_containers)}")
    queue_diagnostics = status_payload.get("queue_diagnostics")
    if isinstance(queue_diagnostics, Mapping):
        queue_active_run_count = int(queue_diagnostics.get("active_run_count") or 0)
        if queue_active_run_count > 0:
            blockers.append(f"queue.active_run_count={queue_active_run_count}")
        live_run_count = int(queue_diagnostics.get("local_docker_live_run_count") or 0)
        if live_run_count > 0:
            blockers.append(f"queue.local_docker_live_run_count={live_run_count}")
        if queue_diagnostics.get("has_runtime_activity") is True:
            blockers.append("queue.has_runtime_activity=true")
        start_queue = queue_diagnostics.get("participant_start_queue")
        if isinstance(start_queue, Mapping):
            capacity = start_queue.get("local_docker_capacity")
            if isinstance(capacity, Mapping):
                running_container_count = int(capacity.get("running_container_count") or 0)
                if running_container_count > 0:
                    blockers.append(
                        f"queue.participant_running_container_count={running_container_count}"
                    )
    return blockers


def _slot_preflight(launch: ReadmeSmokeLaunch) -> dict[str, Any]:
    if not launch.slot_id:
        return {"source": "not_applicable", "slot_id": None, "clear": True}
    status_url = _slot_runtime_status_url(launch)
    if not status_url:
        return {
            "source": "slot_contract",
            "slot_id": launch.slot_id,
            "worker_pool_id": launch.worker_pool_id,
            "clear": False,
            "blockers": ["missing_runtime_status_url"],
        }
    status_payload = _fetch_slot_runtime_status(status_url)
    blockers = _slot_activity_blockers(status_payload)
    return {
        "source": "runtime_status",
        "slot_id": launch.slot_id,
        "worker_pool_id": launch.worker_pool_id,
        "runtime_status_url": status_url,
        "clear": not blockers,
        "blockers": blockers,
        "status": {
            "ok": status_payload.get("ok"),
            "phase": status_payload.get("phase"),
            "ready": status_payload.get("ready"),
            "busy": status_payload.get("busy"),
            "active_run_count": status_payload.get("active_run_count"),
            "runtime_activity": status_payload.get("runtime_activity"),
            "queue_diagnostics": status_payload.get("queue_diagnostics"),
        },
    }


def _write_run_identity(output_root: Path, payload: Mapping[str, Any]) -> None:
    output_root.joinpath("run_identity.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    run_id = str(payload.get("run_id") or "").strip()
    if run_id:
        output_root.joinpath("RUN_ID").write_text(f"{run_id}\n", encoding="utf-8")


def _link_run_identity(output_root: Path, run_id: str) -> dict[str, Any]:
    run_id = run_id.strip()
    if not run_id:
        return {"linked": False, "reason": "missing_run_id"}
    link_root = crafter_runs_dir() / "by-run-id"
    link_root.mkdir(parents=True, exist_ok=True)
    link_path = link_root / run_id
    try:
        if link_path.is_symlink():
            current = Path(os.readlink(link_path))
            if not current.is_absolute():
                current = (link_path.parent / current).resolve()
            if current == output_root.resolve():
                return {"linked": True, "path": str(link_path)}
            return {
                "linked": False,
                "path": str(link_path),
                "reason": f"existing_symlink_points_to={current}",
            }
        if link_path.exists():
            return {
                "linked": False,
                "path": str(link_path),
                "reason": "path_exists_not_symlink",
            }
        os.symlink(os.path.relpath(output_root, link_path.parent), link_path)
        return {"linked": True, "path": str(link_path)}
    except OSError as exc:
        return {
            "linked": False,
            "path": str(link_path),
            "reason": f"{type(exc).__name__}: {exc}",
        }


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


def apply_config_to_bundle(bundle: dict[str, Any], config: CrafterDeoRunConfig) -> dict[str, Any]:
    """Patch guidance-only kickoff contract + staged markdown from in-script config."""
    dataset_ref = _dataset_ref_from_bundle(bundle)
    worker_pool_id = str(bundle.get("worker_pool_id") or "").strip()
    kickoff = apply_guidance_only_kickoff(
        bundle,
        project_notes=config.project_notes(
            worker_pool_id=worker_pool_id,
            dataset_ref=dataset_ref,
        ),
        task_briefs=config.worker_planning_briefs(
            dataset_ref=dataset_ref,
            worker_pool_id=worker_pool_id,
        ),
        plan_task_payloads=config.worker_plan_task_payloads(
            dataset_ref=dataset_ref,
            worker_pool_id=worker_pool_id,
        ),
        task_instructions_md=config.task_instructions_markdown(),
    )

    effective = bundle.setdefault("effective_config", {})
    effective["crafter_deo_run_config"] = {
        "parallel_worker_count": config.parallel_worker_count,
        "candidate_ids": list(config.candidate_ids),
        "train_seeds": list(config.train_seeds),
        "orchestrator_profile_id": config.orchestrator_profile_id,
        "worker_profile_id": config.worker_profile_id,
        "objective_without_plan_observations": config.objective_without_plan_observations,
        "worker_no_evidence_token_threshold": config.worker_no_evidence_token_threshold,
        "worker_no_evidence_observations": config.worker_no_evidence_observations,
        "worker_evidence_without_progress_token_threshold": (
            config.worker_evidence_without_progress_token_threshold
        ),
        "worker_evidence_without_progress_observations": (
            config.worker_evidence_without_progress_observations
        ),
        "worker_evidence_with_progress_token_threshold": (
            config.worker_evidence_with_progress_token_threshold
        ),
        "worker_evidence_with_progress_observations": (
            config.worker_evidence_with_progress_observations
        ),
        **kickoff_guidance_summary(kickoff),
    }
    return kickoff


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
    _strip_launch_primary_parent(bundle)
    timebox = max(config.run_timebox_seconds, 60)
    trigger_payload = bundle.setdefault("trigger_payload", {})
    trigger_payload["timebox_seconds"] = timebox
    return bundle


def build_launch_request(
    trigger_kwargs: dict[str, Any],
    *,
    runtime_image: RuntimeImage | None = None,
) -> RunLaunchRequest:
    kwargs = dict(trigger_kwargs)
    if runtime_image is not None:
        kwargs["runtime_image"] = runtime_image
    return RunLaunchRequest.from_client_kwargs(kwargs)


def _is_retryable_crafter_launch_blocker(value: Any) -> bool:
    from reportbench.readme_smoke_harness import (
        is_retryable_launch_backpressure,
        launch_blocker_code,
    )

    if is_retryable_launch_backpressure(value):
        return True
    code = launch_blocker_code(value)
    # Local slot1 runs often hit transient Codex quota windows; keep polling until reset.
    return code == "codex_pool_no_usable_accounts"


def _launch_blocker_summary(value: Any) -> str:
    from reportbench.readme_smoke_harness import launch_blocker_code

    code = launch_blocker_code(value)
    if code:
        return code
    text = str(value)
    return text[:500] + ("..." if len(text) > 500 else "")


def _is_retryable_crafter_setup_error(exc: BaseException) -> bool:
    from reportbench.readme_smoke_harness import (
        is_retryable_pre_run_transport_error,
        is_retryable_setup_backpressure,
    )

    if is_retryable_setup_backpressure(exc):
        return True
    if is_retryable_pre_run_transport_error(exc):
        return True
    failure_class = getattr(exc, "failure_class", None)
    if failure_class == "db_connection_closed":
        return True
    text = str(exc).lower()
    return "db_connection_closed" in text or "connection is closed" in text


def _is_retryable_crafter_transport_error(exc: BaseException) -> bool:
    from reportbench.readme_smoke_harness import is_retryable_pre_run_transport_error

    return is_retryable_pre_run_transport_error(exc)


def _require_runtime_image(image: RuntimeImage) -> None:
    try:
        image.require_local_docker_image()
    except RuntimeImageError as exc:
        raise SystemExit(
            f"[crafter-deo] {exc}\nBuild open_research_crafter first (see crafter_runs/README.md)."
        ) from exc


def _execution_has_intent(execution: Any, intent_name: str) -> bool:
    needle = intent_name.lower()
    for event in getattr(execution, "events", ()) or ():
        title = str(getattr(event, "title", "") or "").lower()
        summary = str(getattr(event, "summary", "") or "").lower()
        if needle in title or needle in summary:
            return True
    return False


def _execution_has_durable_evidence(execution: Any) -> bool:
    if getattr(execution, "work_products", None):
        return True
    evidence_needles = (
        "record_objective_progress",
        "workspace_push",
        "workspace push",
        "workspace pushed",
        "work_product",
        "work product",
        "publish_report_work_product",
        "smr_attach_experiment_result",
        "experiment result",
    )
    for event in getattr(execution, "events", ()) or ():
        text = f"{getattr(event, 'title', '')} {getattr(event, 'summary', '')}".lower()
        if any(needle in text for needle in evidence_needles):
            return True
    return False


def _actor_state(value: Any) -> str:
    return str(getattr(value, "state", "") or "").strip().lower()


def _actor_role(value: Any) -> str:
    return str(getattr(value, "role", "") or "").strip().lower()


def _actor_is_terminal(value: Any) -> bool:
    return _actor_state(value) in {
        "done",
        "completed",
        "failed",
        "blocked",
        "stopped",
        "canceled",
        "cancelled",
    }


def _actor_snapshot_state(row: Mapping[str, Any]) -> str:
    return str(row.get("actor_state") or "").strip().lower()


def _actor_snapshot_is_terminal(row: Mapping[str, Any]) -> bool:
    return _actor_snapshot_state(row) in {
        "done",
        "completed",
        "failed",
        "blocked",
        "stopped",
        "canceled",
        "cancelled",
    }


def _worker_actor_snapshot_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    worker_rows = [
        row for row in rows if str(row.get("participant_role") or "").strip().lower() == "worker"
    ]
    live_rows = [row for row in worker_rows if not _actor_snapshot_is_terminal(row)]
    return len(worker_rows), len(live_rows)


def _non_bootstrap_orchestrator_actors(execution: Any) -> list[Any]:
    actors: list[Any] = []
    for actor in getattr(execution, "actors", ()) or ():
        if _actor_role(actor) != "orchestrator":
            continue
        actor_id = str(getattr(actor, "actor_id", "") or "").strip()
        if actor_id == "orchestrator:main":
            continue
        actors.append(actor)
    return actors


def _objective_progress_totals(snapshot: Mapping[str, Any]) -> tuple[int, int]:
    progress_total = 0
    achievement_total = 0
    progress_by_id = snapshot.get("progress_by_objective_id")
    if not isinstance(progress_by_id, Mapping):
        return progress_total, achievement_total
    for row in progress_by_id.values():
        if not isinstance(row, Mapping):
            continue
        progress_total += int(row.get("progress_count") or row.get("claim_count") or 0)
        achievement_total += int(row.get("achievement_count") or 0)
    return progress_total, achievement_total


def _objective_has_milestones(snapshot: Mapping[str, Any]) -> bool:
    progress_by_id = snapshot.get("progress_by_objective_id")
    if not isinstance(progress_by_id, Mapping):
        return False
    for row in progress_by_id.values():
        if isinstance(row, Mapping) and int(row.get("milestone_count") or 0) > 0:
            return True
    return False


@dataclass
class CrafterRunLivenessGuard:
    client: Any
    project_id: str
    run_id: str
    log: LogFn
    objective_without_plan_observations: int
    worker_no_evidence_token_threshold: int
    worker_no_evidence_observations: int
    worker_evidence_without_progress_token_threshold: int
    worker_evidence_without_progress_observations: int
    worker_evidence_with_progress_token_threshold: int
    worker_evidence_with_progress_observations: int
    actor_state_snapshot: Callable[[], list[dict[str, Any]]]
    no_plan_observations: int = 0
    no_evidence_observations: int = 0
    evidence_without_progress_observations: int = 0
    evidence_with_progress_observations: int = 0
    stop_requested: bool = False
    stop_suppressed: bool = False
    stop_reason: str = ""
    stop_receipt: dict[str, Any] | None = None
    stop_suppression_detail: dict[str, Any] | None = None
    no_evidence_warning_logged: bool = False
    evidence_without_progress_warning_logged: bool = False
    evidence_with_progress_warning_logged: bool = False
    observations: list[dict[str, Any]] = field(default_factory=list)

    def _max_worker_tokens(self) -> int:
        max_tokens = 0
        for row in self.actor_state_snapshot():
            if str(row.get("participant_role") or "").strip().lower() != "worker":
                continue
            total = int(row.get("token_usage_total") or 0)
            max_tokens = max(max_tokens, total)
        return max_tokens

    def _request_stop(self, reason: str, detail: Mapping[str, Any]) -> None:
        if self.stop_requested or self.stop_suppressed:
            return
        self.stop_suppressed = True
        self.stop_reason = reason
        self.stop_suppression_detail = dict(detail)
        self.log(f"[guard] stop suppressed reason={reason} detail={self.stop_suppression_detail}")

    def on_snapshot(self, _snapshot: Any) -> None:
        if self.stop_requested or self.stop_suppressed:
            return
        try:
            execution = self.client.get_run_execution(
                self.project_id,
                self.run_id,
                view="summary",
                event_limit=200,
                actor_limit=30,
                task_limit=30,
                work_product_limit=30,
            )
            objective_snapshot = fetch_objective_progress_snapshot(
                self.client,
                self.project_id,
                self.run_id,
            )
        except Exception as exc:  # noqa: BLE001
            self.observations.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return

        tasks = list(getattr(execution, "tasks", ()) or ())
        orchestrator_actors = _non_bootstrap_orchestrator_actors(execution)
        live_orchestrator_actors = [
            actor for actor in orchestrator_actors if not _actor_is_terminal(actor)
        ]
        reviewer_actors = [
            actor
            for actor in getattr(execution, "actors", ()) or ()
            if _actor_role(actor) == "reviewer"
        ]
        live_reviewer_actors = [actor for actor in reviewer_actors if not _actor_is_terminal(actor)]
        task_states = {
            str(getattr(task, "public_task_state", "") or getattr(task, "task_state", "") or "")
            .strip()
            .lower()
            for task in tasks
        }
        has_plan_tasks = _execution_has_intent(execution, "plan_tasks")
        objective_count = len(objective_snapshot.get("objectives") or [])
        has_milestones = _objective_has_milestones(objective_snapshot)
        progress_total, achievement_total = _objective_progress_totals(objective_snapshot)
        has_durable_evidence = _execution_has_durable_evidence(execution)
        actor_snapshot_rows = self.actor_state_snapshot()
        worker_actor_count, live_worker_actor_count = _worker_actor_snapshot_counts(
            actor_snapshot_rows
        )
        max_worker_tokens = self._max_worker_tokens()
        observation = {
            "at": datetime.now(UTC).isoformat(),
            "objective_count": objective_count,
            "has_milestones": has_milestones,
            "task_count": len(tasks),
            "task_states": sorted(state for state in task_states if state),
            "has_plan_tasks": has_plan_tasks,
            "orchestrator_actor_count": len(orchestrator_actors),
            "live_orchestrator_actor_count": len(live_orchestrator_actors),
            "orchestrator_actor_states": sorted(
                _actor_state(actor) for actor in orchestrator_actors
            ),
            "reviewer_actor_count": len(reviewer_actors),
            "live_reviewer_actor_count": len(live_reviewer_actors),
            "reviewer_actor_states": sorted(_actor_state(actor) for actor in reviewer_actors),
            "worker_actor_count": worker_actor_count,
            "live_worker_actor_count": live_worker_actor_count,
            "progress_count": progress_total,
            "achievement_count": achievement_total,
            "has_durable_evidence": has_durable_evidence,
            "max_worker_tokens": max_worker_tokens,
        }
        self.observations.append(observation)
        self.observations = self.observations[-12:]

        if (
            objective_count > 0
            and has_milestones
            and not tasks
            and not has_plan_tasks
            and orchestrator_actors
            and not live_orchestrator_actors
        ):
            self.no_plan_observations += 1
        else:
            self.no_plan_observations = 0
        if self.no_plan_observations >= self.objective_without_plan_observations:
            self._request_stop(
                "objective_without_plan_tasks",
                {
                    **observation,
                    "observations": self.no_plan_observations,
                },
            )
            return

        nonterminal_task_states = task_states - {
            "done",
            "failed",
            "blocked",
            "stopped",
            "canceled",
            "cancelled",
        }
        if "review_required" in task_states or reviewer_actors:
            self.evidence_without_progress_observations = 0
            self.evidence_with_progress_observations = 0
            return
        if (
            tasks
            and nonterminal_task_states
            and not has_durable_evidence
            and progress_total == 0
            and achievement_total == 0
            and max_worker_tokens >= self.worker_no_evidence_token_threshold
        ):
            self.no_evidence_observations += 1
        else:
            self.no_evidence_observations = 0
        if (
            self.no_evidence_observations >= self.worker_no_evidence_observations
            and not self.no_evidence_warning_logged
        ):
            self.no_evidence_warning_logged = True
            detail = {
                **observation,
                "observations": self.no_evidence_observations,
                "token_threshold": self.worker_no_evidence_token_threshold,
            }
            self.log(f"[guard] observe reason=worker_without_durable_evidence detail={detail}")

        if (
            tasks
            and nonterminal_task_states
            and has_durable_evidence
            and progress_total == 0
            and achievement_total == 0
            and orchestrator_actors
            and not live_orchestrator_actors
            and live_worker_actor_count > 0
        ):
            self.evidence_without_progress_observations = 0
            if not self.evidence_without_progress_warning_logged:
                self.evidence_without_progress_warning_logged = True
                self.log(
                    "[guard] observe reason=evidence_without_objective_progress "
                    f"detail={observation}"
                )
            return
        if (
            tasks
            and nonterminal_task_states
            and has_durable_evidence
            and progress_total == 0
            and achievement_total == 0
            and orchestrator_actors
            and not live_orchestrator_actors
            and worker_actor_count > 0
            and live_worker_actor_count == 0
        ):
            self.evidence_without_progress_observations += 1
        else:
            self.evidence_without_progress_observations = 0
        if (
            self.evidence_without_progress_observations
            >= self.worker_evidence_without_progress_observations
        ):
            self._request_stop(
                "evidence_without_objective_progress",
                {
                    **observation,
                    "observations": self.evidence_without_progress_observations,
                },
            )
            return

        if (
            tasks
            and nonterminal_task_states
            and has_durable_evidence
            and (progress_total > 0 or achievement_total > 0)
            and orchestrator_actors
            and not live_orchestrator_actors
            and live_worker_actor_count > 0
        ):
            self.evidence_with_progress_observations = 0
            if not self.evidence_with_progress_warning_logged:
                self.evidence_with_progress_warning_logged = True
                self.log(
                    "[guard] observe reason=evidence_with_progress_nonterminal_task "
                    f"detail={observation}"
                )
            return
        if (
            tasks
            and nonterminal_task_states
            and has_durable_evidence
            and (progress_total > 0 or achievement_total > 0)
            and orchestrator_actors
            and not live_orchestrator_actors
            and worker_actor_count > 0
            and live_worker_actor_count == 0
        ):
            self.evidence_with_progress_observations += 1
        else:
            self.evidence_with_progress_observations = 0
        if (
            self.evidence_with_progress_observations
            >= self.worker_evidence_with_progress_observations
        ):
            self._request_stop(
                "evidence_with_progress_without_terminal_task",
                {
                    **observation,
                    "observations": self.evidence_with_progress_observations,
                },
            )

    def summary(self) -> dict[str, Any]:
        return {
            "stop_requested": self.stop_requested,
            "stop_suppressed": self.stop_suppressed,
            "stop_reason": self.stop_reason or None,
            "stop_receipt": self.stop_receipt,
            "stop_suppression_detail": self.stop_suppression_detail,
            "no_plan_observations": self.no_plan_observations,
            "no_evidence_observations": self.no_evidence_observations,
            "no_evidence_warning_logged": self.no_evidence_warning_logged,
            "evidence_without_progress_warning_logged": (
                self.evidence_without_progress_warning_logged
            ),
            "evidence_without_progress_observations": (self.evidence_without_progress_observations),
            "evidence_with_progress_warning_logged": self.evidence_with_progress_warning_logged,
            "evidence_with_progress_observations": self.evidence_with_progress_observations,
            "recent_observations": list(self.observations),
        }


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
        line = f"[{datetime.now(UTC).isoformat()}] {msg}"
        log_lines.append(line)
        _log(msg)

    client = (
        research
        or build_research_client(
            api_key=launch.api_key,
            base_url=launch.backend,
        )
    ).control(timeout_seconds=120.0)

    summary: dict[str, Any] = {
        "sdk": "synth-ai",
        "driver": "crafter_runs/crafter_deo_run.py",
        "task_id": TASK_ID,
        "lane_root": str(LANE_ROOT),
        "output_root": str(output_root),
        "target": launch.target,
        "backend": launch.backend,
        "host_kind": launch.host_kind.value,
        "started_at": datetime.now(UTC).isoformat(),
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
    initial_runtime_messages = list(trigger_kwargs.get("initial_runtime_messages") or [])
    initial_runtime_messages.append(
        {
            "body": (
                "Crafter DEO proof kickoff: worker must evaluate "
                f"{', '.join(config.candidate_ids)} on {config.train_seed_count()} "
                "configured seeds, publish reports/final_report.md as a report "
                "WorkProduct, push workspace artifacts, and report baseline plus "
                "candidate metrics before done."
            ),
            "mode": "queue",
            "payload": {
                "source": "crafter_deo_run_driver",
                "candidate_ids": list(config.candidate_ids),
                "seed_count": config.train_seed_count(),
                "required_artifacts": [
                    "reports/final_report.md",
                    "artifacts/workproduct_container/eval_summary.json",
                    *config.required_candidate_paths(),
                ],
            },
        }
    )
    trigger_kwargs["initial_runtime_messages"] = initial_runtime_messages
    launch_request = build_launch_request(
        trigger_kwargs,
        runtime_image=config.crafter_runtime_image,
    )

    project_name = str(runnable_project_request.get("name") or "").strip() or (
        f"ReportBench Crafter DEO {launch.worker_pool_id}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    kickoff_meta = (bundle.get("effective_config") or {}).get("crafter_deo_run_config") or {}
    summary.update(
        {
            "worker_pool_id": worker_pool_id,
            "project_name": project_name,
            "workspace_input_count": len(files),
            "work_mode": work_mode.value,
            "kickoff_guidance_mode": kickoff_meta.get("kickoff_guidance_mode"),
            "kickoff_tasks_count": kickoff_meta.get("kickoff_tasks_count"),
            "task_brief_count": kickoff_meta.get("task_brief_count"),
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
        preflight = _slot_preflight(launch)
        summary["slot_preflight"] = preflight
        blockers = [
            str(blocker) for blocker in preflight.get("blockers") or () if str(blocker).strip()
        ]
        _emit(
            "slot_preflight "
            f"source={preflight.get('source')} slot={preflight.get('slot_id')} "
            f"clear={preflight.get('clear')} blockers={blockers}"
        )
        if blockers:
            raise ResearchApiError(
                "slot preflight blocked active local work: " + "; ".join(blockers)
            )

        with client:
            _emit("research.projects.create_runnable_project ...")
            project = client.create_runnable_project(runnable_project_request)
            project_id = str(field_value(project, "project_id", default="") or "").strip()
            if not project_id:
                raise ResearchApiError(f"create_runnable_project returned no project_id: {project}")
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
                    if not _is_retryable_crafter_setup_error(exc):
                        raise
                    if time.monotonic() >= setup_deadline:
                        raise ResearchApiError(
                            "upload_workspace_files blocked after "
                            f"{config.setup_retry_timebox_seconds}s: {exc}"
                        ) from exc
                    delay_s = setup_retry_delay_seconds(upload_attempt)
                    _emit(
                        f"setup retry upload attempt={upload_attempt} "
                        f"failure_class={getattr(exc, 'failure_class', None)!r} "
                        f"delay_s={delay_s:.1f}"
                    )
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
                    if not _is_retryable_crafter_setup_error(exc):
                        raise
                    if time.monotonic() >= setup_deadline:
                        raise ResearchApiError(
                            f"prepare_project_setup blocked after "
                            f"{config.setup_retry_timebox_seconds}s: {exc}"
                        ) from exc
                    delay_s = setup_retry_delay_seconds(prepare_attempt)
                    _emit(
                        f"setup retry prepare attempt={prepare_attempt} "
                        f"failure_class={getattr(exc, 'failure_class', None)!r} "
                        f"delay_s={delay_s:.1f}"
                    )
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
                    preflight = client.get_launch_preflight(
                        project_id,
                        request=launch_request,
                    )
                    summary["launch_preflight"] = jsonish(preflight)
                    if not bool(field_value(preflight, "clear_to_trigger", default=False)):
                        if _is_retryable_crafter_launch_blocker(preflight):
                            if time.monotonic() >= launch_deadline:
                                raise ResearchApiError("launch preflight blocked (timeout)")
                            code = launch_blocker_code(preflight) or "launch_backpressure"
                            _emit(f"launch preflight backpressure code={code}")
                            time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                            continue
                        raise ResearchApiError(
                            f"launch preflight blocked: {_launch_blocker_summary(preflight)}"
                        )
                    run = client.trigger_run(project_id, request=launch_request)
                    run_id = str(field_value(run, "run_id", "id", default="") or "").strip()
                    if not run_id:
                        raise ResearchApiError(f"trigger_run returned no run_id: {run}")
                    break
                except ResearchApiError as exc:
                    if (
                        _is_retryable_crafter_launch_blocker(exc)
                        and time.monotonic() < launch_deadline
                    ):
                        _emit(f"launch backpressure trigger_run: {_launch_blocker_summary(exc)}")
                        time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                        continue
                    raise
                except Exception as exc:  # noqa: BLE001
                    if (
                        _is_retryable_crafter_transport_error(exc)
                        and time.monotonic() < launch_deadline
                    ):
                        _emit(f"launch transport retry: {type(exc).__name__}: {exc}")
                        time.sleep(min(30.0, 2.0 + launch_attempt * 1.5))
                        continue
                    raise

            summary["run_id"] = run_id
            summary["trigger_response"] = jsonish(run)
            identity = {
                "task_id": TASK_ID,
                "target": launch.target,
                "slot_id": launch.slot_id,
                "slot_mode": launch.slot_mode,
                "worker_pool_id": worker_pool_id,
                "backend": launch.backend,
                "project_id": project_id,
                "run_id": run_id,
                "output_root": str(output_root),
                "created_at": datetime.now(UTC).isoformat(),
            }
            _write_run_identity(output_root, identity)
            summary["run_identity"] = identity
            summary["run_identity_link"] = _link_run_identity(output_root, run_id)
            _emit(f"run_id={run_id}")
            _emit(f"run_identity={output_root / 'run_identity.json'}")
            driver_runtime_message = seed_driver_runtime_message(
                client,
                project_id=project_id,
                run_id=run_id,
                config=config,
            )
            summary["driver_runtime_message"] = jsonish(driver_runtime_message)
            _emit(f"[msg] driver_seeded {driver_runtime_message['poll_line']}")
            post_trigger_objectives = fetch_objective_poll_line(client, project_id, run_id)
            _emit(f"[objective] post_trigger {post_trigger_objectives}")
            summary["objective_post_trigger"] = post_trigger_objectives
            post_trigger_git_status = _fetch_project_git_status(client, project_id)
            post_trigger_git = summarize_git_poll_line(post_trigger_git_status)
            _emit(f"[git] post_trigger {post_trigger_git}")
            summary["git_post_trigger"] = post_trigger_git
            summary["git_post_trigger_status"] = jsonish(
                summarize_git_status_snapshot(post_trigger_git_status)
            )
            post_trigger_messages = fetch_message_poll_line(client, project_id, run_id)
            _emit(f"[msg] post_trigger {post_trigger_messages}")
            summary["message_post_trigger"] = post_trigger_messages
            post_trigger_actor_states: dict[str, ActorTraceState] = {}
            _post_trigger_trace_lines, post_trigger_actor_summary = fetch_actor_trace_poll_lines(
                client,
                project_id,
                run_id,
                post_trigger_actor_states,
            )
            for line in _post_trigger_trace_lines:
                _emit(f"[trace] {line}")
            _emit(f"[actor] post_trigger {post_trigger_actor_summary}")
            summary["actor_post_trigger"] = post_trigger_actor_summary
            post_trigger_wp = fetch_work_products_poll_line(client, project_id, run_id)
            _emit(f"[wp] post_trigger {post_trigger_wp}")
            summary["work_product_post_trigger"] = post_trigger_wp
            _post_trigger_usage_lines, post_trigger_usage_summary, _ = fetch_usage_poll_lines(
                client,
                project_id,
                run_id,
                {},
            )
            for line in _post_trigger_usage_lines:
                _emit(f"[usage] {line}")
            _emit(f"[usage] post_trigger {post_trigger_usage_summary}")
            summary["usage_post_trigger"] = post_trigger_usage_summary

            on_objective_snapshot, objective_snapshots = build_objective_poll_callback(
                client,
                project_id,
                run_id,
                log=_emit,
            )
            on_git_snapshot, git_snapshots = build_git_poll_callback(
                client,
                project_id,
                log=_emit,
            )
            on_message_snapshot, message_snapshots = build_message_poll_callback(
                client,
                project_id,
                run_id,
                log=_emit,
            )
            on_actor_trace_snapshot, actor_trace_snapshots = build_actor_trace_poll_callback(
                client,
                project_id,
                run_id,
                log=_emit,
            )
            on_work_product_snapshot, work_product_snapshots = build_work_product_poll_callback(
                client,
                project_id,
                run_id,
                log=_emit,
            )
            on_usage_snapshot, usage_snapshots = build_usage_poll_callback(
                client,
                project_id,
                run_id,
                log=_emit,
            )
            actor_state_snapshot = getattr(
                on_actor_trace_snapshot,
                "actor_state_snapshot",
                lambda: [],
            )
            liveness_guard = CrafterRunLivenessGuard(
                client=client,
                project_id=project_id,
                run_id=run_id,
                log=_emit,
                objective_without_plan_observations=config.objective_without_plan_observations,
                worker_no_evidence_token_threshold=config.worker_no_evidence_token_threshold,
                worker_no_evidence_observations=config.worker_no_evidence_observations,
                worker_evidence_without_progress_token_threshold=(
                    config.worker_evidence_without_progress_token_threshold
                ),
                worker_evidence_without_progress_observations=(
                    config.worker_evidence_without_progress_observations
                ),
                worker_evidence_with_progress_token_threshold=(
                    config.worker_evidence_with_progress_token_threshold
                ),
                worker_evidence_with_progress_observations=(
                    config.worker_evidence_with_progress_observations
                ),
                actor_state_snapshot=actor_state_snapshot,
            )

            def on_poll_snapshot(snapshot: Any) -> None:
                on_objective_snapshot(snapshot)
                on_git_snapshot(snapshot)
                on_message_snapshot(snapshot)
                on_actor_trace_snapshot(snapshot)
                on_work_product_snapshot(snapshot)
                on_usage_snapshot(snapshot)
                liveness_guard.on_snapshot(snapshot)

            final_run = poll_run_until_terminal(
                client,
                project_id,
                run_id,
                timebox_s=config.poll_timebox_seconds,
                log=_emit,
                on_snapshot=on_poll_snapshot,
            )
            final_state = str(field_value(final_run, "public_state", default="") or "").lower()
            summary["final_state"] = final_state
            summary["final_run"] = jsonish(final_run)
            summary["objective_poll_snapshots"] = objective_snapshots()
            summary["git_poll_snapshots"] = git_snapshots()
            summary["message_poll_snapshots"] = message_snapshots()
            summary["actor_trace_poll_snapshots"] = actor_trace_snapshots()
            summary["work_product_poll_snapshots"] = work_product_snapshots()
            summary["usage_poll_snapshots"] = usage_snapshots()
            summary["liveness_guard"] = liveness_guard.summary()
            final_objective_snapshot = fetch_objective_progress_snapshot(
                client,
                project_id,
                run_id,
            )
            final_objective_line = str(
                final_objective_snapshot.get("line")
                or fetch_objective_poll_line(client, project_id, run_id)
            )
            _emit(f"[objective] final {final_objective_line}")
            summary["objective_final"] = final_objective_line
            summary["objective_final_snapshot"] = jsonish(final_objective_snapshot)
            objective_review_gate = summarize_objective_review_gate(final_objective_snapshot)
            objective_review_gate_line = summarize_objective_review_gate_line(objective_review_gate)
            _emit(f"[objective] {objective_review_gate_line}")
            summary["objective_review_gate"] = jsonish(objective_review_gate)
            terminal_success_state = final_state in {"completed", "succeeded", "done"}
            if terminal_success_state and not bool(objective_review_gate.get("complete")):
                summary.setdefault("verification_failures", []).append(
                    {
                        "code": "objective_review_incomplete",
                        "detail": objective_review_gate_line,
                    }
                )
            final_git_status = _fetch_project_git_status(client, project_id)
            final_git_state = _git_poll_state_from_status(final_git_status)
            final_git_line = summarize_git_poll_line(final_git_status)
            _emit(f"[git] final {final_git_line}")
            summary["git_final"] = final_git_line
            summary["git_final_status"] = jsonish(summarize_git_status_snapshot(final_git_status))
            git_unmerged_branch_statuses: list[dict[str, Any]] = []
            if final_git_state is not None:
                for branch_state in final_git_state.unmerged_branches[:5]:
                    branch_git_status = _fetch_project_git_status(
                        client,
                        project_id,
                        branch=branch_state.name,
                        max_tree_entries=500,
                        max_commits=10,
                        max_unmerged_branches=5,
                    )
                    branch_status_summary = summarize_git_status_snapshot(branch_git_status)
                    if branch_status_summary is None:
                        continue
                    git_unmerged_branch_statuses.append(branch_status_summary)
                    branch_line = summarize_git_poll_line(branch_git_status)
                    _emit(f"[git] branch {branch_state.name} {branch_line}")
            summary["git_unmerged_branch_statuses"] = jsonish(git_unmerged_branch_statuses)
            final_message_line = fetch_message_poll_line(client, project_id, run_id)
            _emit(f"[msg] final {final_message_line}")
            summary["message_final"] = final_message_line
            _final_actor_states: dict[str, ActorTraceState] = {}
            final_actor_lines, final_actor_summary = fetch_actor_trace_poll_lines(
                client,
                project_id,
                run_id,
                _final_actor_states,
            )
            for line in final_actor_lines:
                _emit(f"[trace] {line}")
            _emit(f"[actor] final {final_actor_summary}")
            summary["actor_final"] = final_actor_summary
            final_wp_snapshot = fetch_work_products_snapshot(client, project_id, run_id)
            final_wp_line = str(
                final_wp_snapshot.get("line")
                or fetch_work_products_poll_line(client, project_id, run_id)
            )
            _emit(f"[wp] final {final_wp_line}")
            summary["work_product_final"] = final_wp_line
            summary["work_products_final"] = final_wp_line
            summary["work_products_final_snapshot"] = jsonish(final_wp_snapshot)
            _final_usage_lines, final_usage_summary, _ = fetch_usage_poll_lines(
                client,
                project_id,
                run_id,
                {},
            )
            for line in _final_usage_lines:
                _emit(f"[usage] {line}")
            _emit(f"[usage] final {final_usage_summary}")
            summary["usage_final"] = final_usage_summary
            _emit(f"final_state={final_state}")

            try:
                archive_meta = client.download_workspace_archive(project_id, archive_path)
                summary["archive_meta"] = jsonish(archive_meta)
                archive_commit = str(archive_meta.get("commit_sha") or "").strip()
                if archive_commit:
                    summary["git_archive_commit"] = archive_commit
                    matched_branches = [
                        str(branch_status.get("branch") or "")
                        for branch_status in git_unmerged_branch_statuses
                        if str(branch_status.get("head_commit_sha") or "").strip() == archive_commit
                    ]
                    if matched_branches:
                        summary["git_archive_commit_matched_branches"] = matched_branches
                    prior_git_final = str(summary.get("git_final") or "").strip()
                    archive_match = (
                        f" archive_branch={matched_branches[0]}" if matched_branches else ""
                    )
                    summary["git_final"] = (
                        f"{prior_git_final} archive_commit={archive_commit}{archive_match}".strip()
                    )
                    _emit(f"[git] archive_commit={archive_commit}")
                _emit(f"archive bytes={archive_meta.get('bytes_written')}")
                crafter_results = summarize_crafter_results_archive(archive_path)
                if crafter_results is not None:
                    summary["crafter_results_final"] = crafter_results
                    _emit(f"[result] final {summarize_crafter_results_line(crafter_results)}")
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

    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"\nsummary written to: {summary_path}")
    print(f"run log:           {log_path}")
    if archive_path.exists():
        print(f"workspace archive: {archive_path} ({archive_path.stat().st_size} bytes)")

    if summary.get("fatal_error"):
        return 2
    verification_failures = summary.get("verification_failures")
    if isinstance(verification_failures, list) and verification_failures:
        print(f"[fail] task_id={TASK_ID} verification_failures={len(verification_failures)}")
        return 1
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
    parser.add_argument(
        "--candidate-ids",
        default=",".join(DEFAULT_CONFIG.candidate_ids),
        help="Comma-separated candidate ids to plan; default is the 1-candidate proof lane.",
    )
    parser.add_argument("--parallel-worker-count", type=int, default=None)
    parser.add_argument("--skip-docker-image-check", action="store_true")
    parser.add_argument(
        "--proof",
        action="store_true",
        help="Use 8 train seeds for faster local proof (verifier gates still expect 20).",
    )
    parser.add_argument(
        "--train-seeds",
        default=None,
        help="Comma-separated train seed override (default: 20 seeds; --proof uses 8).",
    )
    parser.add_argument("--orchestrator-profile", default=None)
    parser.add_argument("--worker-profile", default=None)
    parser.add_argument("--run-timebox-seconds", type=int, default=None)
    parser.add_argument("--poll-timebox-seconds", type=int, default=None)
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

    config = config_from_cli_args(args)
    if not args.skip_docker_image_check:
        _require_runtime_image(config.crafter_runtime_image)

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
