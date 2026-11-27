# `synth` command line

`synth` (also installed as `synth-ai`) is a thin layer over the typed client.
Every command accepts `--json`; the global `--base-url` selects the API.
For the 0.31 grouped commands, see the [0.31 to 0.32 migration](cli-migration-0.31-0.32.md).

| Step | Command | SDK record under `--json` |
| --- | --- | --- |
| Sign in | `synth login [--no-browser \| --device]` | `AuthStatus` |
| | `synth whoami` | `Account` |
| | `synth logout` | `AuthStatus` |
| Discover | `synth profiles [--research-task T]` | `RunProfileCatalog` |
| | `synth choices [--profile P] [--domicile D] [--parallel-workers N] [--wall SECONDS] [--budget 2.00]` | `LaunchChoices` |
| Inputs | `synth upload PATH...` | array of `File` |
| Launch | `synth run "goal" --budget 2.00 [--profile P] [--research-task T] [--file PATH ...] [--wall SECONDS (1..86400)] [--parallel-workers N (1..32)] [--label k=v ...] [--assign ACTOR.inference\|sandbox\|option=ID ... --menu-revision R] [--planner-decides ACTOR.inference\|sandbox ...] [--idempotency-key KEY] [--wait [--timeout S]]` | `RunSnapshot` (final snapshot with `--wait`; human output ends with the answer text when there is one, then the charge line) |
| Watch | `synth status RUN` | `RunSnapshot` |
| | `synth wait RUN [--timeout S]` | `RunSnapshot` |
| | `synth logs RUN [-f] [--limit N] [--cursor C]` | `EventPage`; with `-f`, one `RunEvent` per line |
| | `synth tasks RUN [--limit N] [--cursor C]` | `RunTaskPage` |
| | `synth ls [--state S] [--limit N] [--cursor C]` | `RunPage` |
| Steer | `synth pause RUN`, `synth resume RUN` | `RunControlReceipt` |
| | `synth cancel RUN` | `RunSnapshot` |
| Collect | `synth outputs RUN [--limit N] [--cursor C]` | `OutputPage` |
| | `synth download RUN [OUTPUT] [-o DIR]` | array of `{output, path}` |
| | `synth report RUN` | `RunReport` |
| | `synth allocation RUN` | `RunAllocation` |
| | `synth result RUN` | `RunResult` as `{run_id, status, accounting, run, state, answer, answer_output, objective_answer, charge, attempt_tree, result_file}`; human output is the answer text (a text output, or an objective answer's text), then status, accounting and charge lines on stderr |
| Account | `synth usage` | `UsageSummary` |
| | `synth usage RUN` | `UsageRecordPage` |
| | `synth billing` | `Balance` |
| | `synth limits` | `LimitList` |

`synth run` requires a positional goal, `--goal-file PATH` (alias `--task-file PATH`,
or `-` for stdin), or piped stdin; the UTF-8 contents become the launch message.
File and stdin input avoid operating-system argument (`E2BIG`) limits and receive
the same bounded message validation. `--file` uploads
each file (prepare, transfer, finalize in one step) and attaches its reference.
The default `--timeout` for `wait` and `run --wait` is 3600 seconds.

For a launch whose response was lost, reuse the same `--idempotency-key` with
exactly the same goal, flags and file contents. The CLI derives stable upload
keys from it and preserves file order. A changed request under the same key is
refused. If an upload grant has expired, recover the retained upload through the
SDK's `resume_upload` with the `upload_id` and `idempotency_key` printed by
the CLI before retrying; do not issue a new run blindly.

`synth result RUN` reads a finished run with `swarms.result`: it prints the
answer (the last available `text/*` output, digest-verified, or else the latest
objective answer's text) or, when there is none, the run, state and output count.
When Core reports the run's charge, stderr gets one line, `charge  AMOUNT USD (final)`
or `(not final)`; usage still settling can raise a charge that is not final.
`--json` carries it as `charge`. It never waits; on an unfinished run it
exits 1 with `result_unavailable`. See [errors.md](errors.md) for every code.

`synth choices` reads the per-role menu a launch with the same profile,
domicile, worker count and wall time admits: each declared actor with its
default inference and sandbox profile, the selectable inference and sandbox
profiles with prices, and the exclusions with a cause (`locality`,
`not_executable`, or `affordability` when `--budget` is given). Choose with
`synth run ... --menu-revision R --assign worker.inference=luna-high
--assign worker.sandbox=8vcpu-16gib-us --assign worker.option=sbo1_...`
(an option is required only where the menu says so) and
`--planner-decides worker_2.sandbox` to leave a worker's choice to the planner.
Unnamed actors keep their defaults. A launch against a menu that has since
changed is refused with `launch_menu_revision_stale` and the current revision.

`synth allocation RUN` shows each actor's launch choice (explicit or default),
every frozen placement including the planner's replacements, usage and compute
charge, and whether the run is final.

`synth login --no-browser` prints the authorization URL to stderr and reads the
redirect URL that the browser reached (`http://127.0.0.1:PORT/callback?...`) from
stdin. The state and issuer are verified as in the browser flow.

`synth login --device` is for a machine with no browser at all (RFC 8628). It prints
the console's device page and a code such as `BCDF-GHJK` to stderr, then polls until
someone signed in to the console enters the code and approves it, denies it, or it
expires after ten minutes. The secret device code is never printed.

The resource commands below call the same public SDK methods. Mutations remain
subject to the server's scopes and signed-in-session requirements.

| Resource | Commands |
| --- | --- |
| Files | `synth files list`, `get FILE`, `upload-text TEXT --name NAME` |
| Upload grants | `synth uploads prepare --body PATH`, `get UPLOAD`, `finalize UPLOAD` |
| Keys | `synth keys list`, `mint --body PATH`, `update KEY --body PATH`, `rotate KEY`, `revoke KEY` |
| Organization | `synth organization get`, `usage-summary`, `usage-records`, `usage-export --since TIME --until TIME`, `usage-export-status EXPORT`, `limits`, `accept-terms --version VERSION --accepted-at TIME` |
| Billing | `synth billing balance`, `plans`, `history`, `statements`, `checkout --body PATH`, `checkout-status CHECKOUT`, `subscription`, `cancel-subscription`, `resume-subscription`, `preview-plan PLAN`, `change-plan --body PATH`, `payment-method`, `auto-reload`, `set-auto-reload --body PATH`, `spend-alerts`, `set-spend-alerts --body PATH` |
| Plan requests | `synth plan request --use-case TEXT [--link URL ...]`, `synth plan requests`, `synth plan withdraw REQUEST`. The use case is sent to Synth for owner review and is never printed or logged; a denial cooldown refuses with `plan_request_cooldown` and prints the wait in seconds (`retry_after_seconds` in the `--json` error document) |
| Run details | `synth inspect RUN --view program\|configuration\|sandboxes\|output\|wait-update\|review`, `synth families` |
| Run labels | `synth labels RUN --body PATH` |
| Pinned bytes | `synth content RUN OUTPUT --out PATH --revision REVISION --sha256 DIGEST` |
| Artifacts | `synth artifacts create\|get\|update\|content\|list\|share\|revoke-share` |
| Experiments | `synth experiments create\|list\|get\|revise` |
| Research logs | `synth research-log init\|append\|search\|correct\|export` |

`--body` reads a JSON file validated against the generated request type. Resource
commands print the canonical response JSON; a nullable response prints `null`.
`content` verifies the immutable revision and SHA-256 before saving bytes, then
prints the local destination path as a JSON string. Use each command's `--help`
for its pagination, filter and idempotency options.

## Output

Without `--json`, output is short aligned text. With `--json`, stdout carries
exactly one JSON document: the SDK record serialised with
`model_dump_json(exclude_unset=True)`. `logs -f --json` is the one stream: one
`RunEvent` document per line. Errors go to stderr: `code: detail`, or the typed
problem JSON under `--json`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Client or usage error, including transport failures |
| 2 | `wait` (or `run --wait`) reached its timeout; the run is still running |
| 3 | The run finished `failed` or `cancelled` (`wait`, `run --wait`) |
| 4 | Server refusal (typed problem from the API) |
| 5 | The run paused (`wait`, `run --wait`) |

The constants are `EXIT_*` in `synth_ai/cli.py`.

## Event checkpoint

`synth logs RUN -f` stores its `StreamCheckpoint` at
`~/.config/synth/streams/RUN.json` (mode 0600, directory 0700). Set
`SYNTH_STATE_DIR` to use another directory. The checkpoint advances only after
an event has been written to stdout, so a restarted `logs -f` resumes after the
last printed event without repeating it.

### Directory uploads

`synth upload --directory ./bundle --idempotency-key bundle-1 --json` snapshots
one directory with the SDK's bounded binary-safe traversal and preserves relative
names. Keep the key when retrying unchanged inputs. A partial failure returns
exit 1 and a JSON receipt on stderr containing committed files, the failed name,
and both retry keys. Committed files are retained; the batch is not atomic.
