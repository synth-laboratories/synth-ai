# Changelog

## Unreleased

- Contract 1.47.0 (synth-backend `958f195b`): cursor paging and filters on
  notebook artifact, experiment and research-log lists (SYN-6066), and run
  readback `notebook_inheritance` (SYN-6068).
  - SDK: `artifacts.list_artifacts` gains keyword-only `kind`, `origin`, `q`,
    `limit` and `cursor`; `experiments.list_experiments` gains `q`, `limit` and
    `cursor`; `research_logs.search_entries` gains `q` (alias of `query`),
    `limit` and `cursor`. Pages carry `next_cursor` and `has_more`. Text
    filters match literally, including `%` and `_`. The server now returns at
    most 20 items per page by default (max 100).
  - CLI: `synth artifacts list --kind --origin --q`, `synth experiments list --q`
    and `synth research-log search --q --author-kind`, each with `--limit` and
    `--cursor`.
  - MCP: `swarms_artifacts_list`, `swarms_experiments_list` and
    `swarms_research_log_search` forward the new arguments.
  - Models: `RunSnapshot.notebook_inheritance` (`RunNotebookInheritance`,
    `RunInheritedRecord`).
- Contract 1.47.0 (synth-backend `3223ec2d`, 93 MCP tools): notebook discovery
  and the Human Artifactory across SDK, CLI and MCP.
  - SDK: `client.notebooks` (`list_notebooks`, `iter_notebooks`) and
    `client.artifactory` (`create_record`, `list_records`, `resolve_preference`,
    `get_record`, `revise_record`), sync and async, with typed
    `ArtifactoryPreference`, `ArtifactoryHypothesis` and `ArtifactoryFinding`
    payloads; the record category follows the payload type.
  - CLI: `synth notebooks list` and `synth artifactory list|get|preference|create|revise`
    with readable tables; `--file` payloads are validated before sending
    (`ArtifactoryPayloadError`, `artifactory_payload_invalid`).
  - MCP: `swarms_notebooks_list` and `swarms_artifactory_*`;
    `swarms_artifactory_record_create` defaults `origin` to `agent`.
  - Typed errors: `ArtifactoryOriginRequiresHumanCredentialError` (403) and
    `ArtifactoryRevisionConflictError` (409).
- Synth Style compliance and modularization across SDK (`synth_ai`), CLI, and MCP:
  split large CLI and helper modules below 800 lines, validate HTTP origins,
  idempotency keys, non-empty path IDs, and `--wait --timeout` before file uploads,
  sanitize validation errors with `hide_input_in_errors=True`, and align
  `synth-backend@1e3d493b` / `9b9a2f7d` (`1.43.0`) acceptance pins.

This checkout tracks trunk changes since the last published release; it does not attest publication of unreleased changes or installed service acceptance. The pinned Core contract revision and digest are recorded in [contracts/SOURCE.toml](contracts/SOURCE.toml).

## 0.42.0 — 2026-11-24

Published to PyPI as `synth-ai==0.42.0` (pinned to Core contract 1.43.0 in [contracts/SOURCE.toml](contracts/SOURCE.toml)).

Changes below describe the Swarms release through `0.42.0`. Historical releases remain in Git history.

- Contract 1.43.0 (Core `0a0d61d3`): Track B Research & Artifacts across SDK, CLI,
  and MCP (83 MCP tools, 82 public OpenAPI operations):
  - SDK: `client.artifacts` (`ArtifactsClient` / `AsyncArtifacts`),
    `client.experiments` (`ExperimentsClient` / `AsyncExperiments`),
    `client.research_logs` (`ResearchLogsClient` / `AsyncResearchLogs`), and
    `client.swarms.review(run_id)`.
  - CLI: `synth artifacts`, `synth experiments`, `synth research-log`, and
    `synth inspect RUN --view review`.
  - MCP: `swarms_artifact_*`, `swarms_experiment_*`, `swarms_research_log_*`,
    and `swarms_run_review`.
- Contract 1.41.0–1.42.0: `parallel_workers` supports `1..32` workers per run
  and metered usage records surface `index_searches` when reported by Core
  (`/v1/index` routes stay gated with `index_not_available` until launch).
- Contract 1.40.0 (Core c17b5fd6): revoke your own MCP OAuth grant with an API key
  (SYN-5737): SDK `client.account.revoke_oauth_grant(grant_id)`, CLI
  `synth grants revoke GRANT`, MCP tool `account_oauth_grant_revoke` (`account:write`).
- Message sends retry Core's 503 `run_message_admission_pending` (SYN-5744): the
  SDK, CLI and MCP resend under the same Idempotency-Key, waiting Retry-After
  (1 s when absent, capped at 60 s) within a 120 s budget. Exhaustion raises
  `UpstreamUnavailableError` with `attempts` and `elapsed_seconds`.
- Diagnostics: `retry_suppressed` events with a typed `RetrySuppressed` when the
  deadline or an invalid `Retry-After` stops a retry (SYN-5724).
- Contract 1.39.0 (Core e9dc3a36, introduced at 1dfeed39): run messages
  (SYN-5745) and `continue_from` (SYN-5748).
  - SDK: `swarms.send_message`, `messages`/`iter_messages`, `message`,
    `follow_messages` and `wait_for_reply` (a local timeout never cancels the
    message or the run). `follow()` yields typed `run.message.*` events.
    `MessageKind`, `ManagerMessageState` and `MessageDisposition` are generated
    StrEnums; a discarded body is `BodyUnavailable(reason)`.
  - Typed refusals: `message_too_large`, `run_message_rate_limited` (Retry-After
    only when retryable), `run_not_accepting_messages`, `run_message_queue_full`,
    `messaging_unavailable`, `continue_from_not_found`,
    `continue_from_not_terminal` and `continue_from_inputs_unavailable`.
  - CLI: `synth message RUN TEXT [--kind ...]`, `synth messages RUN [--follow]`,
    `synth run --continue-from RUN`. MCP: `swarms_run_message_send`,
    `swarms_run_messages`, `swarms_run_message`, and `continue_from` on
    `swarms_run_launch`. `RunLineage` appears on the result and report.
- Contracts 1.37.0 and 1.38.0 (Core 033ca45a): `RunResult` now carries `run_id`,
  `status` (the terminal outcome: `completed`, `partial`, `failed` or
  `cancelled`) and `accounting` (the final ledger-backed `RunCharge`, always
  `final`, or `None` when Core has no accounting reader) through
  `swarms.result`, `synth result` (`--json`; human output adds `status` and
  `accounting` lines on stderr) and the `swarms_run_result` MCP tool (SYN-5637).
  A 422 `run_budget_below_call_bound` launch refusal's `Problem` carries
  `minimum_budget_micros` and `budget_floor` (`BudgetFloor`).
- Contract 1.36.1 (SYN-5667; 1.36.1 adds the public names to the contract and MCP tool descriptions): a `swarms_v2` pin takes `params.max_tasks`
  (1..4096) and `params.max_repairs_per_task` (0..32) in the SDK
  (`OptimizerParams`), the CLI (`--max-tasks`, `--max-repairs-per-task`, with
  `--optimizer swarms_v2@VERSION`) and the MCP launch tool. Public names are
  Swarms (`swarms_v2`, default) and Fork-and-Flush (`fork_flush`, experimental,
  never default); `topology_name()` and `synth families` print them.
- Contract 1.35.1: `synth login --device` and `auth.login_device` sign in on a
  machine without a browser through the OAuth device authorization grant (SYN-5680).
- Contract 1.35.0: `account.redeem_promo`, `synth promo redeem` and the
  `account_promo_redeem` MCP tool redeem a promotional code (SYN-5752). A trial
  campaign's redemption carries `trial` (plan and fixed start/end).
- Typed synchronous and asynchronous Swarms clients, CLI and MCP tools share the
  public Core contract. Files and a message launch a run; result reads preserve
  its answer, exact charge, output references and canonical attempt tree.
- The [CLI migration guide](docs/cli-migration-0.31-0.32.md) lists the grouped
  commands removed in 0.32, their replacements, and the corresponding commands
  in 0.41.
- Named `local`, `staging` and `prod` profiles select the target explicitly. The
  customer default is production; credentials remain bound to their host.
- Retryable 429 and 503 responses honor `Retry-After` with bounded jittered
  retries. Nonretryable refusals surface immediately. Queued runs expose their
  position, estimate and expiry while waiting for admission.
- Special Free requests, self-serve billing, run-limit updates and per-role
  launch choices use the same typed public API.
- Contract 1.34.0 includes queued run standing and result.md references. Source and fixture qualification do
  not establish acceptance of a deployed cohort.

- Nov 16 admits `swarms_v2` only. `fork_flush` stays a reserved public enum;
  pinned requests receive a typed refusal and experimental entries never default.
