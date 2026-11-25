# Changelog

## 0.42.0 — source candidate

This checkout does not attest publication or installed acceptance. The exact Core
revision and digest are recorded in [contracts/SOURCE.toml](contracts/SOURCE.toml).

Changes here describe the new Swarms product. Historical releases remain in Git history.

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
