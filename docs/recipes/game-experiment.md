# Game experiment

**Prerequisites:** Install an exact published release with
`uv add "synth-ai[research]==<version>"`, then set `SYNTH_AI_VERSION` to that
same version. Set `SYNTH_API_KEY`, `SYNTH_GAME_PROJECT_ID`,
`SYNTH_GAMEBENCH_SCORER_BASE`, and `SYNTH_GAMEBENCH_SCORER_TOKEN`. The prepared
project must publish one UTF-8 candidate artifact whose metadata has
`role="gamebench_candidate"` and a complete `gamebench_score_identity` object
for `GameBenchCandidateScoreRequest`. It must also have a positive backend-owned
per-run cost cap and a blocking server wallclock limit of at most 30 minutes.

**Duration:** About 10 minutes of setup; both managed-run and scorer waits are
bounded to 30 minutes, and the run itself must expose a blocking server limit no
greater than that. **Cost:** Game rollouts and model calls are billable. The
example fails early without the backend-owned per-run cap and prints actual
drawdown after the run.

```python
import html
import os
from importlib.metadata import version

from synth_ai import SynthClient
from synth_ai.gamebench import GameBenchCandidateScoreRequest, GameBenchClient
from synth_ai.research import ResearchWorkMode

expected_version = os.environ["SYNTH_AI_VERSION"]
if version("synth-ai") != expected_version:
    raise RuntimeError("installed synth-ai does not match SYNTH_AI_VERSION")

project_id = os.environ["SYNTH_GAME_PROJECT_ID"]
research = SynthClient().research
scorer = GameBenchClient(
    scorer_token=os.environ["SYNTH_GAMEBENCH_SCORER_TOKEN"],
    scorer_base=os.environ["SYNTH_GAMEBENCH_SCORER_BASE"],
    timeout_seconds=30,
)
work_mode = ResearchWorkMode.DIRECTED_EFFORT


def close_clients():
    errors = []
    for name, close in (("gamebench", scorer.close), ("research", research.close)):
        try:
            close()
        except BaseException as error:
            error.add_note(f"failed to close {name} client")
            errors.append(error)
    return errors

handle = None
run_terminal = False
score_job_id = None
score_terminal = False
scorer_cleaned = False
try:
    economics = research.economics.project(project_id)
    run_cap_cents = next(
        (
            economics.budgets.get(key)
            for key in (
                "run_usd_cents",
                "per_run_usd_cents",
                "per_run_max_usd_cents",
            )
            if economics.budgets.get(key) is not None
        ),
        None,
    )
    if (
        isinstance(run_cap_cents, bool)
        or not isinstance(run_cap_cents, int)
        or run_cap_cents <= 0
    ):
        raise RuntimeError("project has no positive backend-owned per-run cost cap")

    plan = research.economics.plan()
    preflight = research.runs.check_preflight(project_id, work_mode=work_mode)
    clear_to_trigger = preflight.get("clear_to_trigger", preflight.get("allowed"))
    if plan.blocked or clear_to_trigger is not True:
        raise RuntimeError(
            f"experiment denied: {plan.blocked_detail}; {preflight.get('blockers')}"
        )

    handle = research.runs.create(
        project_id,
        objective="Produce the prepared game candidate for its frozen score suite.",
        work_mode=work_mode,
    )
    limits = handle.resource_limits()
    wallclock = next(
        (item for item in limits.items if item.metric == "wallclock_seconds"),
        None,
    )
    if (
        wallclock is None
        or not wallclock.blocks_at_limit
        or wallclock.limit_value is None
        or not 0 < wallclock.limit_value <= 1800
    ):
        raise RuntimeError("run has no blocking server wallclock limit <= 1800 seconds")

    research.runs.wait(project_id, handle.run_id, timeout=1800, raise_if_failed=True)
    run_terminal = True
    manifest = handle.artifacts.manifest.get()
    candidates = [
        item
        for item in manifest.artifacts
        if item.metadata.get("role") == "gamebench_candidate"
    ]
    if len(candidates) != 1:
        raise RuntimeError("run must publish exactly one gamebench_candidate artifact")
    candidate = candidates[0]
    candidate_bytes = handle.artifacts.content.get(candidate.artifact_id, as_text=False)
    if not isinstance(candidate_bytes, bytes):
        raise TypeError("candidate artifact content must be bytes")
    identity = candidate.metadata.get("gamebench_score_identity")
    if not isinstance(identity, dict):
        raise TypeError("candidate metadata must contain gamebench_score_identity")
    request = GameBenchCandidateScoreRequest(
        **identity,
        candidate_bytes=candidate_bytes,
    )
    work_product_ids = {
        item.work_product_id for item in handle.work_products.list()
    }
    if (
        request.project_id != project_id
        or request.run_id != handle.run_id
        or request.work_product_id not in work_product_ids
    ):
        raise RuntimeError("score identity does not belong to the managed run")
    if request.timeout_seconds > 1800:
        raise RuntimeError("score request timeout exceeds the 1800-second bound")

    submission = scorer.scorers.submit_candidate(request)
    score_job_id = submission.job_id
    score_result = scorer.scorers.wait(
        score_job_id,
        timeout=request.timeout_seconds,
        poll_interval=min(5, request.timeout_seconds),
    )
    score_terminal = True
    score_succeeded = score_result.status == "succeeded"
    cleanup_receipt = scorer.scorers.cleanup(score_job_id)
    scorer_cleaned = True
    if not score_succeeded:
        raise RuntimeError(
            f"score job ended with {score_result.status}: {score_result.failure}"
        )

    score_json = score_result.model_dump_json(indent=2)
    visual_receipt = research.visuals.publish(
        handle.run_id,
        title="GameBench score",
        html=f"<html><body><pre>{html.escape(score_json)}</pre></body></html>",
        visual_kind="gamebench_score",
        source_run_ids=(handle.run_id,),
        metadata={"score_job_id": score_job_id},
    )
    visual_id = str(visual_receipt.get("visual_id") or "").strip()
    if not visual_id:
        raise RuntimeError("visual publication returned no visual_id")
    visual_html = research.visuals.get_content(visual_id, as_text=True)

    print(score_result)
    print(visual_receipt)
    print(visual_html)
    print(manifest)
    print(cleanup_receipt)
    print(run_cap_cents, wallclock)
    print(research.economics.run_drawdown(handle.run_id))
except BaseException as original:
    errors = [original]
    if score_job_id is not None and not scorer_cleaned:
        if not score_terminal:
            try:
                scorer.scorers.cancel(score_job_id)
            except BaseException as error:
                error.add_note("failed to cancel non-terminal score job")
                errors.append(error)
        try:
            scorer.scorers.cleanup(score_job_id)
        except BaseException as error:
            error.add_note("failed to clean score job")
            errors.append(error)
    if handle is not None and not run_terminal:
        try:
            research.runs.stop(handle.run_id, project_id=project_id)
        except BaseException as error:
            error.add_note("failed to stop non-terminal managed run")
            errors.append(error)
    errors.extend(close_clients())
    if len(errors) == 1:
        raise
    raise BaseExceptionGroup("game experiment failed during cleanup", errors)
else:
    close_errors = close_clients()
    if close_errors:
        raise BaseExceptionGroup("game experiment succeeded but close failed", close_errors)
```

**Output:** `ResearchArtifactManifest`, the retrieved candidate `bytes`,
identity-bound `GameBenchCandidateScoreResult`, published visual
`dict[str, object]` plus retrieved HTML `str`, validated
`GameBenchCandidateScoreCleanupReceipt`, `SmrResourceLimit`, the exact integer
cost cap, and `ResearchBillingDrawdown`. **Recovery:** A non-terminal scorer
failure is cancelled and cleaned before re-raising; a pre-terminal managed-run
failure requests stop. Fix the frozen identity, candidate, or project authority
before retrying. **Cleanup:** Success requires the typed zero-resource scorer
cleanup receipt. The managed run is terminal, the visual is retained as output,
and the reusable prepared project is intentionally retained.
