# CLI migration: 0.31 to 0.32

Version 0.32 replaced the grouped, operation-oriented CLI with task-oriented
commands. This is a breaking change: upgrade the client and update scripts
before invoking the new commands. The table describes the **0.31.0 to 0.32.0**
transition; later versions can add commands or change their options. See the
[current CLI reference](cli.md) for the version you have installed.

The Python examples below use `client` as an open `SynthClient`. An SDK method
is listed where 0.32 has no equivalent CLI command; it is not a shell command.
API keys and legal acceptance can also be managed in the signed-in Console.

## Swarms

| 0.31 command | 0.32 replacement |
| --- | --- |
| `synth swarms launch MESSAGE --budget USD` | `synth run MESSAGE --budget USD` (returns after launch unless `--wait` is supplied). |
| `synth swarms run MESSAGE --budget USD` | `synth run MESSAGE --budget USD --wait`. |
| `synth swarms status RUN` | `synth status RUN`. |
| `synth swarms report RUN` | `synth report RUN`. |
| `synth swarms sandboxes RUN` | `client.swarms.sandboxes(RUN)`. |
| `synth swarms list` | `synth ls`. |
| `synth swarms events RUN` | `synth logs RUN` (use `--cursor` or `--limit` as needed). |
| `synth swarms follow RUN` | `synth logs RUN -f`. |
| `synth swarms attach RUN` | `synth logs RUN -f` for event following. |
| `synth swarms watch RUN` | `synth wait RUN` to wait for completion; use `synth logs RUN -f` to follow events. |
| `synth swarms wait RUN` | `synth wait RUN`. |
| `synth swarms cancel RUN` | `synth cancel RUN`. |
| `synth swarms pause RUN` | `synth pause RUN`. |
| `synth swarms resume RUN` | `synth resume RUN`. |
| `synth swarms program RUN` | `client.swarms.program(RUN)`. |
| `synth swarms configuration RUN` | `client.swarms.configuration(RUN)`. |
| `synth swarms tasks RUN` | `synth tasks RUN`. |
| `synth swarms outputs RUN` | `synth outputs RUN`. |
| `synth swarms output RUN OUTPUT` | `client.swarms.output(RUN, OUTPUT)` for the output record. |
| `synth swarms output-content RUN OUTPUT` | `synth download RUN OUTPUT -o DIR` for verified bytes, or `client.swarms.output_content(RUN, OUTPUT)`. |
| `synth swarms download RUN OUTPUT` | `synth download RUN OUTPUT -o DIR`. |
| `synth swarms families` | `client.swarms.families()`. |
| `synth swarms profiles` | `synth profiles`. |
| `synth swarms upload PATH` | `synth upload PATH`. |
| `synth swarms upload-text` | `client.swarms.upload_text(...)`. |
| `synth swarms uploads-prepare` | `client.swarms.prepare_upload(...)`; normally use `synth upload PATH` for the complete transfer. |
| `synth swarms uploads-get UPLOAD` | `client.swarms.upload(UPLOAD)`. |
| `synth swarms uploads-finalize UPLOAD` | `client.swarms.finalize_upload(UPLOAD, ...)`. |
| `synth swarms files` | `client.swarms.files()`. |
| `synth swarms file FILE` | `client.swarms.file(FILE)`. |
| `synth swarms labels RUN` | `client.swarms.update_labels(RUN, ..., expected_revision=...)`. |

## Account, billing and usage

| 0.31 command | 0.32 replacement |
| --- | --- |
| `synth account get` | `synth whoami`. |
| `synth account accept-terms` | Accept terms in the signed-in Console. |
| `synth api-keys list`, `mint`, `update-grants`, `rotate`, `revoke` | Manage keys in the signed-in Console; the SDK's `client.account.api_keys` also exposes these operations where authorized. |
| `synth billing balance` | `synth billing`. |
| `synth billing history` | `client.billing.history()`. |
| `synth billing statements` | `client.billing.statements()`. |
| `synth billing plans` | `client.billing.plans()`. |
| `synth billing checkout` | `client.billing.checkout(...)` or the signed-in Console. |
| `synth billing checkout-status ID` | `client.billing.checkout_status(ID)`. |
| `synth usage summary` | `synth usage`. |
| `synth usage records` | `synth usage RUN` for a run; `client.account.usage_records(...)` for other filters. |
| `synth usage export` | `client.account.export_usage(...)`. |
| `synth usage export-status ID` | `client.account.usage_export(ID)`. |
| `synth limits list` | `synth limits`. |
| `synth auth status` | `synth whoami` for the account; `synth login` and `synth logout` remain available. |

## Script and option changes

- In 0.31, `--json` on generated operation commands supplied the request body.
  In 0.32, `--json` requests JSON **output**. Do not pass a request body there.
- `synth run` takes a required goal string. `--file PATH` attaches a file; it
  does not replace the goal. Use `--wait` for launch-and-wait, `--wall` for the
  run's wall limit, and `--timeout` for the client's wait timeout.
- The 0.31 launch/run `--task`, `--until`, provider/model selection flags and
  explicit `--idempotency-key` have no equivalent CLI flags in 0.32. Use the
  corresponding typed SDK operation when that control is required. Do not
  assume `--timeout` sets the server's run deadline.
- `synth logs -f` resumes from a local stream checkpoint. The old
  `--last-event-id` flag is not a 0.32 CLI option; `--cursor` is available for
  a paged log read. See [CLI event checkpoints](cli.md#event-checkpoint).
- Human-readable output is now the default. For scripts requiring a record, use
  the new command with `--json` and handle its documented exit codes. Some old
  generated commands have no CLI counterpart; use their typed SDK method.
