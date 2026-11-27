---
name: swarms-supervise
description: Supervise and steer active Cloud Swarms runs — send messages, decide approvals, update live limits with optimistic revisions, pause/resume/cancel, and continue from terminal runs.
---

# Supervising Active Cloud Swarms Runs

Use these workflows to inspect, steer, and control runs while they execute or to branch follow-up work once terminal.

## 1. Observe Progress

- `swarms_run_get` (`run_id`): current lifecycle state, queue standing, and summary.
- `swarms_run_events` (`run_id`, `cursor`, `limit`): ordered run event stream.
- `swarms_run_tasks` (`run_id`, `cursor`, `limit`): delegated worker tasks.
- `swarms_run_sandboxes` / `swarms_run_allocation` (`run_id`): active sandboxes and role allocation.

## 2. Steer with Run Messages

- Read thread: `swarms_run_messages` (`run_id`) or `swarms_run_message` (`run_id`, `message_id`).
- Send guidance: `swarms_run_message_send` (`run_id`, `body: {kind, text}`, `idempotency_key`). Requires `swarms:control`.
- Refusals:
  - `run_message_admission_pending` (503, retryable): run admission has not settled yet; retry after `Retry-After` with the **same** `idempotency_key`.
  - `run_message_queue_full` (409): wait for pending messages to drain before retrying.
  - `run_not_accepting_messages` (409): run is terminal or has no active manager queue.

## 3. Decide Pending Approvals

1. Call `swarms_run_approvals_list` (`run_id`, optional `status` filter) to inspect pending approval gates and their current revision.
2. Call `swarms_run_approval_decide` (`run_id`, `approval_id`, `body: {decision, reason?, expected_revision?}`, `idempotency_key`). Requires `swarms:control`.

## 4. Update Live Run Limits

Limits use optimistic concurrency via `revision`:
1. Read current state with `swarms_run_limits` (`run_id`) and capture `revision`.
2. Call `swarms_run_limits_update` (`run_id`, `body: {expected_revision, limits, resume?, reason?}`, `idempotency_key`). Requires `swarms:run`.
3. Recovery:
   - `limits_revision_conflict`: re-read `swarms_run_limits` and resubmit with the fresh `revision` and a new `idempotency_key`.
   - `limit_above_ceiling`: lower the requested value to or below the org/run ceiling.
   - `limit_still_reached`: when resuming a limit-paused run, raise the reached limit first (or set `resume: true` alongside the higher limit).

## 5. Pause, Resume, and Cancel

All control mutations require `swarms:control` and a fresh `idempotency_key`:
- `swarms_run_pause` (`run_id`, `idempotency_key`): pause active execution.
- `swarms_run_resume` (`run_id`, `idempotency_key`): resume a paused run (ensure any reached limit has been raised).
- `swarms_run_cancel` (`run_id`, `idempotency_key`): terminate the run; collect already-published outputs via `swarms_run_outputs`.

## 6. Continue From a Completed Run

To start follow-up work building on a finished run, pass `continue_from` in `swarms_run_preflight` and `swarms_run_launch`:
- The parent run must be terminal (`continue_from_not_terminal` otherwise).
- If retained parent inputs are unavailable (`continue_from_inputs_unavailable`), launch with explicit fresh `inputs`.
