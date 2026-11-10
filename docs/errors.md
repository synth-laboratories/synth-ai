# Errors

Every refusal and local failure carries a stable `code`. Branch on the code, not
the message. The [error guide](#error-guide) below says what each one means and
what to do.

Catch `SwarmsAPIError` for HTTP refusals. Its `problem` preserves the RFC 9457
fields `type`, `title`, `status`, `detail`, `instance` and the Synth extensions
`code`, `service`, `stage`, `upstream_status`, `cause`, `outcome`.
`request_id` carries the response request identifier.

Status-specific subclasses cover invalid requests, authentication, payment,
forbidden access, missing resources, conflicts, rate limits and upstream/server
failures. Unknown codes remain intact. Malformed responses and transport failures
have distinct classes; a mutation transport failure has outcome `unknown`.

`RunFailedError` means a closed run failed. `WaitTimeoutError` means polling
expired; it does not cancel the run. The CLI prints one JSON refusal to stderr.

## Admission codes

| Code | HTTP | Error class |
| --- | --- | --- |
| `provider_model_unavailable` | 422 | `ProviderPolicyError` |
| `provider_preset_unavailable` | 422 | `ProviderPolicyError` |
| `provider_policy_invalid` | 422 | `ProviderPolicyError` |
| `provider_policy_mismatch` | 422 | `ProviderPolicyError` |
| `provider_model_mismatch` | 422 | `ProviderPolicyError` |
| `provider_preset_not_allowed` | 403 | `ProviderPresetNotAllowedError` |
| `resource_request_invalid` | 422 | `ResourceAdmissionError` |
| `resource_limits_required` | 422 | `ResourceAdmissionError` |
| `grant_not_widenable` | 422 | `ResourceAdmissionError` |
| `launch_budget_below_actor_floor` | 422 | `LaunchBudgetBelowActorFloorError` |
| `project_not_found` | 404 | `NotFoundError` |
| `mutation_identity_required` | 422 | `InvalidRequestError` |
| `gateway_url_missing` | 503 | `UpstreamUnavailableError` |
| `codex_pool_unavailable` | 503 | `UpstreamUnavailableError` |
| `resource_authority_unavailable` | 503 | `UpstreamUnavailableError` |
| `plugin_not_qualified` | 409 | `ConflictError` |
| `gateway_issue_malformed` | 502 | `UpstreamUnavailableError` |

`launch_budget_below_actor_floor` (an `InvalidRequestError`): `maximum_charge`
cannot fund the planner, the reviewer and every one of `limits.parallel_workers`
workers at once (each parallel worker holds an equal slice of the budget). The
Problem `detail` states the minimum in micro-USD. Raise `budget` or lower
`parallel_workers` (`--parallel-workers`).

## Upload codes

| Code | HTTP | Error class |
| --- | --- | --- |
| `upload_format_unsupported` | 422 | `UploadFormatUnsupportedError` |

`upload_format_unsupported` (an `InvalidRequestError`): the media type is not
an accepted upload format. Accepted formats are `text/plain`, `text/markdown`
and `application/json` (`supported_formats` on the error). The SDK raises it
before any request from `upload_file`, `upload_text`, `prepare_upload` and the
stdio MCP upload tool (as an `isError` Problem), and maps the server's own 422
to the same class. The CLI prints it like any refusal and exits 4. Convert the
file to one of the accepted formats. Any other invalid stdio MCP tool input is
`tool_arguments_invalid` (422, `InvalidRequestError`).

All are `SwarmsAPIError` subclasses; the original Problem fields are preserved.
The specialized admission classes also inherit their HTTP-status class.

## Error guide

One line per code: what it means, then what the caller should do.

### Launch and run

- `invalid_launch_request` (422): the launch body is invalid, for example the
  message is missing or empty; `cause` names the field. Fix the request and
  launch again with a new idempotency key.
- `idempotency_key_reused` (422): the key was already used with a different
  request body. Use a new key for a new request.
- `run_pool_below_actor_floor`: the run ended because its pool fell below the
  minimum number of actors it needs. Launch again, with a larger budget or a
  profile whose pool is available (`synth profiles`).
- `run_deadline`: the run stopped at its wall-time limit. Collect the outputs
  it published; launch again with a larger `wall_time_seconds` (`--wall`).
- `org_suspended` (403): the organization is suspended; launches and other
  mutations are refused. Contact support; retrying will not help.
- `run_not_found` (404) and other `NotFoundError` codes: the ID is wrong or
  belongs to another organization. Check the ID and the key's organization.

### Client version

- `client_upgrade_required` (426, `UpgradeRequiredError`): the API no longer
  serves this synth-ai version. Each SDK release is a full cutover with no
  support window. Run `pip install -U synth-ai`. The error carries
  `minimum_version` (the oldest version served) and `client_version` (the
  version this client sent); `retryable` is false and the SDK never retries
  it. The CLI prints the problem, then a one-line upgrade hint.

### Idempotency replay

Mutations take an `Idempotency-Key`. Reusing a key with the same request replays
the recorded result, including a recorded refusal: a refused launch stays
refused under its key. To retry a refused launch, fix the cause and send it with
a new key. A mutation whose transport failed has outcome `unknown`; retry it
with the same key, which never runs it twice.

### Rate limits

- `rate_limit_exceeded` (429, `RateLimitedError`): the API key exceeded 600
  requests per minute, or the organization exceeded 3000 per minute across its
  keys. Wait for `Retry-After` seconds (also `RateLimit-Reset`) and retry; the
  SDK retries safe reads within its retry policy. Spread work across time, not
  across keys.
- `rate_limiter_unavailable` (503): the limiter could not decide. Retry with
  backoff.

### Event streams

- `event_owner_unavailable` (503): the run's event feed moved to another owner,
  for example during a deploy. `wait()`, `follow()`, `synth wait` and
  `synth logs -f` reconnect from their checkpoint without repeating events.
  Since contract 1.4.0 the server marks in-stream refusals with `retryable` and
  `last_event_id`; the SDK reconnects only when `retryable` is true, and treats
  this exact code and status as retryable when an older server omits the field.
- `event_stream_invalid` (local, `ResponseContractError`): a stream frame did
  not match the contract. Upgrade the SDK; report it if the latest release fails.
- `event_stream_timeout` (local, `TransportFailureError`): the stream outlasted
  its timeout. Resume with the error's `checkpoint`.

### Local SDK errors

- `wait_timeout` (`WaitTimeoutError`): the wait expired; the run keeps running.
  Wait again or check `status`.
- `run_failed` (`RunFailedError`): the run finished `failed` (only with
  `raise_on_failure=True`). Read `report()` and outputs.
- `result_unavailable` (`ResultUnavailableError`): `result()` was called before
  the run finished. `wait()` first.
- `run_history_unavailable` (`RunHistoryUnavailableError`): the run's event
  history is no longer retained. Read the snapshot and outputs instead.
- `content_integrity_failed` (`DataIntegrityError`): downloaded bytes did not
  match the recorded digest or size bound. Retry the download; never use the bytes.
- `pagination_invalid` (`PaginationError`): the server returned an inconsistent
  page or cursor. Restart the listing from the first page.
- `upload_incomplete` (`UploadTransferError`) and `upload_transfer_failed`: the
  file did not finish uploading. Resume with `resume_upload(upload_id, ...)`
  and the same idempotency key.
- `output_transfer_failed`: an output download failed in transit. Retry it.
- `client_closed` (`ClientClosedError`): the client was used after `close()`.
  Create a new client.

### CLI

- `usage_error` (exit 1): bad arguments. See `synth COMMAND --help`.
- `client_error` (exit 1): a local failure such as an unreadable file. Local SDK
  errors print their own code instead (for example `result_unavailable`).
- `wait_timeout` (exit 2): the run is still running. Run `synth wait RUN` again.
- Exit 3: the run finished `failed` or `cancelled`. Run `synth report RUN`.
- Exit 4: a server refusal; stderr carries `code: detail` (the problem JSON
  under `--json`). Use the codes above.
