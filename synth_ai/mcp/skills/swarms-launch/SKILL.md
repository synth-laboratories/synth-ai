---
name: swarms-launch
description: Launch a Cloud Swarms run end-to-end — inspect launch choices, preflight, upload inputs, launch with launch_menu_revision and Idempotency-Key, wait, and collect results or outputs.
---

# Launching Cloud Swarms Runs

Follow this sequence to launch a run safely without wasted retries or stale-menu refusals.

## 1. Inspect Current Choices and Affordability

Call `swarms_launch_choices` (optional filters: `profile`, `maximum_charge`, `wall_time_seconds`, `parallel_workers`, `domicile_requirement`).

Retain:
- `menu_revision`: pass this as `launch_menu_revision` on preflight and launch.
- `affordability`, `roles`, `inference_profiles`, `sandbox_profiles`, and `exclusions`.

For budget floors, actor sizing, and staleness recovery, read `skill://swarms-launch/references/budget-and-choices.md`.

## 2. Upload Input Files (Optional)

- Small inline files: `swarms_upload_file` (`body: {name, media_type, content}`, `idempotency_key`).
- Larger files (up to 16 MiB): `swarms_upload_create` (`body: {name, media_type, size_bytes, digest_sha256}`, `idempotency_key`), upload bytes to the returned PUT URL, then call `swarms_upload_finalize` (`upload_id`, `idempotency_key`).
- Local stdio MCP only: when started with `--upload-root`, `swarms_upload_directory` uploads the granted directory with per-file retry custody.
- Reference returned `file_id` values in `body.inputs` when launching.

## 3. Preflight Before Spending

Call `swarms_run_preflight` with the exact `body` you intend to launch (`message`, `limits`, plus optional `profile`, `role_assignments`, `launch_menu_revision`, `inputs`, `guidance`, `resources`, `routing`, `domicile_requirement`, `labels`, `continue_from`).

- Check `admissible == true`.
- If `admissible == false`, fix every code in `refusals` before calling `swarms_run_launch`.

## 4. Launch with `launch_menu_revision` and `Idempotency-Key`

Call `swarms_run_launch` with:
- `idempotency_key`: unique caller-chosen key (1-128 ASCII chars, maps to `Idempotency-Key`).
- `body.message`: clear goal and acceptance criteria.
- `body.limits`: at minimum `maximum_charge` (`{amount, currency: "USD"}`) and `wall_time_seconds`; optionally `parallel_workers`.
- `body.launch_menu_revision`: the `menu_revision` from `swarms_launch_choices`.

Idempotency discipline:
- Replay the **same** `idempotency_key` only after an uncertain transport failure with the identical body.
- Never reuse a key with a modified body (`idempotency_key_reused`).
- After any server refusal (4xx), fix the cause and mint a **new** `idempotency_key`.

## 5. Wait and Collect Outputs

1. Poll with `swarms_run_wait` (`run_id`, optional `timeout_seconds`). A timeout outcome means the run is still active — wait again or inspect via `swarms_run_get` / `swarms_run_events`.
2. Once terminal, call `swarms_run_result` (`run_id`) for `status`, `answer`, `answer_output`, `accounting`, `charge`, `lineage`, and `attempt_tree`.
3. Enumerate files with `swarms_run_outputs` (`run_id`). Read bounded output bytes with `swarms_run_output_content` pinning `run_id`, `output_id`, `revision_id`, and `digest_sha256`.
4. If the run failed, inspect `swarms_run_report` (`run_id`).
