# Errors

HTTP refusals and SDK transport failures carry a stable `code`. Branch on the code, not
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

| Code | HTTP | Error class | Next action |
| --- | --- | --- | --- |
| `provider_model_unavailable` | 422 | `ProviderPolicyError` | Choose a currently offered model from the run profile. |
| `provider_preset_unavailable` | 422 | `ProviderPolicyError` | Select an available preset from the profile menu. |
| `provider_policy_invalid` | 422 | `ProviderPolicyError` | Correct the provider policy fields named in the Problem. |
| `provider_policy_mismatch` | 422 | `ProviderPolicyError` | Refresh the profile menu and use its current policy. |
| `provider_model_mismatch` | 422 | `ProviderPolicyError` | Choose a model offered by the selected provider and preset. |
| `routing_constraints_invalid` | 422 | `ProviderPolicyError` | Correct the routing constraints in the request. |
| `routing_constraints_unsatisfiable` | 422 | `ProviderPolicyError` | Choose constraints that the current provider menu can satisfy. |
| `provider_preset_not_allowed` | 403 | `ProviderPresetNotAllowedError` | Choose a preset admitted for your organization. |
| `resource_request_invalid` | 422 | `ResourceAdmissionError` | Correct the invalid resource fields named in the Problem. |
| `resource_limits_required` | 422 | `ResourceAdmissionError` | Supply the required resource limits. |
| `resource_routing_unverified` | 422 | `ResourceAdmissionError` | Select a resource route with verified admission. |
| `grant_not_widenable` | 422 | `ResourceAdmissionError` | Keep the request within the parent grant or ask its owner for a new grant. |
| `launch_budget_below_actor_floor` | 422 | `LaunchBudgetBelowActorFloorError` | Raise the budget or reduce parallel workers. |
| `project_not_found` | 404 | `NotFoundError` | Check the project ID and the credential’s organization. |
| `mutation_identity_required` | 422 | `InvalidRequestError` | Supply an idempotency key before retrying the request. |
| `gateway_url_missing` | 503 | `UpstreamUnavailableError` | Give support the request ID so they can check Gateway configuration. |
| `codex_pool_unavailable` | 503 | `UpstreamUnavailableError` | Check the available profiles and select one with an available pool. |
| `resource_authority_unavailable` | 503 | `UpstreamUnavailableError` | Use the Problem’s retry guidance, or share the request ID with support. |
| `plugin_not_qualified` | 409 | `ConflictError` | Choose a profile with a currently qualified runtime. |
| `gateway_issue_malformed` | 502 | `UpstreamUnavailableError` | Share the request ID with support so they can inspect Gateway’s response. |

`launch_budget_below_actor_floor` (an `InvalidRequestError`): `maximum_charge`
cannot fund the planner, the reviewer and every one of `limits.parallel_workers`
workers at once (each parallel worker holds an equal slice of the budget). The
Problem `detail` states the minimum in micro-USD. Raise `budget` or lower
`parallel_workers` (`--parallel-workers`).

## Continuation codes

| Code | HTTP | Error class | Next action |
| --- | --- | --- | --- |
| `continue_from_not_found` | 404 | `ContinueFromNotFoundError` | Check the parent run ID and its organization. |
| `continue_from_not_terminal` | 409 | `ContinueFromNotTerminalError` | Wait for the parent run to finish, then launch with a new key. |
| `continue_from_inputs_unavailable` | 409 | `ContinueFromInputsUnavailableError` | Launch with fresh explicit inputs if you can reconstruct them. |

These refusals leave the parent unchanged and do not create a child run. The
current source refuses missing retained input producers explicitly; the
launch argument alone does not establish successful continuation.

## Message codes

| Code | HTTP | Error class | Next action |
| --- | --- | --- | --- |
| `message_too_large` | 422 | `MessageTooLargeError` | Shorten the message and submit it with a new key. |
| `run_message_rate_limited` | 429 | `RunMessageRateLimitedError` | Follow the Problem’s retry guidance and wait for the rate window to reset. |
| `run_not_accepting_messages` | 409 | `RunNotAcceptingMessagesError` | Check the run state and use a run with an admitted manager queue. |
| `run_message_queue_full` | 409 | `RunMessageQueueFullError` | Wait for existing messages to drain and follow the Problem’s retry guidance. |
| `messaging_unavailable` | 503 | `MessagingUnavailableError` | Check that the run has a visible thread and consult the Problem’s retry guidance. |

Message retries follow the Problem's `retryable` field. A `Retry-After` header
does not authorize retrying a non-retryable refusal. `MessageReplyTimeoutError`
is a local observation timeout and leaves the run and message unchanged.

## Upload codes

| Code | HTTP | Error class | Next action |
| --- | --- | --- | --- |
| `upload_format_unsupported` | 422 | `UploadFormatUnsupportedError` | Convert the file to an accepted media type. |

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

## Run control codes

| Code | Error class | Next action |
| --- | --- | --- |
| `limits_revision_conflict` | `LimitsRevisionConflictError` | Read the current limits and resubmit the desired change with the current revision. |
| `limit_above_ceiling` | `LimitAboveCeilingError` | Choose a value at or below the allowed ceiling. |
| `limit_still_reached` | `LimitStillReachedError` | Raise the reached limit before resuming the run. |
| `run_expired` | `RunExpiredError` | Collect existing outputs and launch a new run if you still need work. |
| `spend_entitlement_exceeded` | `SpendEntitlementExceededError` | Check available spend and ask an organization admin to adjust the entitlement if needed. |

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

### Authentication, authorization and service errors

- `unauthenticated` (401): the request supplies no credential. Sign in or
  configure an API key for the intended organization.
- `api_key_invalid` (401): the API key was rejected. Invalid, expired, rotated-out
  and revoked keys receive the same code. Configure a valid key.
- `authentication_required` (401): the operation requires valid authentication.
- `insufficient_scope` (403): the credential does not grant the operation's
  scope. Use a credential with the required grant or ask an organization admin.
- `launch_requires_sdk_or_mcp` (403): a signed-in browser session cannot launch
  a run directly. Use an API key through the SDK, CLI or MCP.
- `invalid_request` (422): the request does not match the operation contract.
  Inspect the problem's detail and correct the request before trying again.
- `resource_not_found` (404): the requested resource is unavailable to this
  caller. Check the identifier and the credential's organization.
- `terms_version_unknown` (400): the requested terms version is not recognized.
  Refresh the Console and use the currently presented terms.
- `service_unavailable` (503): a required service cannot handle the request.
  Retry with backoff when safe. For a mutation with an uncertain outcome, reuse
  its idempotency key rather than creating a second operation.
- `internal_error` (500): the server encountered an unexpected error. Retain
  the request ID and contact support if the problem persists. Reconcile an
  uncertain mutation using its original idempotency key before attempting a new one.

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
  the run finished (Core's 409 `result_unavailable`; the error carries a fresh
  snapshot). `wait()` first.
- `result_answer_too_large`, `result_answer_not_text` (`ConflictError`, 409):
  the answer output is over 1 MiB or not UTF-8 text. Fetch it with
  `download_output`.
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


## Capacity and queue refusals

- `run_queue_full` (429): the admission queue has no room. A server Problem with
  `retryable: true` permits a bounded retry after `Retry-After` plus jitter.
- `overloaded` (503): the service cannot admit this request. Retry only when
  the server marks it retryable; the SDK shares that decision with CLI and MCP.
- Quota or spend exhaustion marked `retryable: false` surfaces immediately,
  preserving the server's code, detail and named limit fields. An omitted
  retryable member also refuses automatic replay.

Automatic retries cover safe reads and contract-authorized writes with a stable
idempotency key and a proven `not_accepted` outcome. Unknown mutation outcomes
never retry automatically. `RetryPolicy` bounds attempts, jitter and the overall
deadline; a server delay outside the remaining budget surfaces the refusal.

### Journey refusal cases

- `optimizer_topology_not_allowed`: this environment does not admit the topology.
  Select an admitted topology from its current catalog.
- `optimizer_topology_unavailable`: the topology has no usable served definition.
  Choose an available topology from the current catalog.
- `optimizer_version_unavailable`: the requested topology version is not served.
  Use the version published by the current catalog.
- `launch_request_too_large`: the serialized launch exceeds the server's limit.
  Shorten the message or upload long material as files referenced by the launch.
- `index_not_available` (404): Index requests are not served by the Swarms API in
  this release. Core answers it for `/v1/index` routes and the CLI prints it for
  `synth index ...` without sending a request. Launch a run with files and a
  message instead; retrying will not help.
- `route_not_found`: the requested API route is not served. Check the origin and
  current contract; retired products have no replacement route at that URL.
- `billing_admin_required` (403): the operation requires an organization admin or owner.
  Ask an authorized admin to perform it; retrying the same credential will not help.

### Usage and billing

Each billing and usage operation in the public contract lists its refusals.

Request and lookup:

- `invalid_filter` (422): a usage filter, such as a time, cannot be parsed. Fix
  the query parameter named in `detail`.
- `usage_window_invalid` (422): `until` is not after `since`, or the window is
  longer than 366 days. Narrow the window.
- `invalid_cursor` (422): the page cursor is not one this listing issued. Start
  again from the first page.
- `invalid_limit` (422): the page size is out of range. Use a smaller `limit`.
- `invalid_history_filter` (422): a billing history parameter is invalid;
  `detail` names it.
- `invalid_id` (422): the checkout ID is malformed. Use the ID the checkout
  returned.
- `checkout_not_found` (404): no checkout with that ID belongs to this org.
- `checkout_request_invalid` (422) and `malformed_request` (422): the checkout
  body is invalid. Send either a plan or a credit amount within the stated bounds.
- `credit_pack_unknown` (422): the credit pack does not exist. Pick one from
  the current price list.
- `plan_not_purchasable` (422): the plan cannot be bought by checkout.
- `plan_unknown` (400): the plan ID does not exist. Use a listed plan.
- `spend_alerts_invalid` (400): a spend alert threshold is invalid. Fix it and
  save again.
- `auto_reload_invalid` (400): an auto-reload field is out of bounds; `field`
  names it and `reason` `below_threshold` means the amount is below the
  threshold.
- `auto_reload_disclosure_required` (422): `disclosure_version` is missing or
  unknown. Show the current disclosure and send its version.

Permission and setup:

- `browser_session_required` (403): an API key or OAuth grant tried to turn
  auto-reload on. Turn it on from a signed-in browser session.
- `payment_method_required` (409): no card is saved. Add one, then retry.
- `billing_customer_not_found` (404): the org has no Stripe customer yet. Buy
  credits or a plan first.
- `config_missing` (503): the service is missing billing configuration; `detail`
  names it. Retrying will not help until the operator fixes it.
- `stripe_price_missing` (503): a Stripe price is not configured; `detail` names
  the lookup key. Same as `config_missing`.
- `portal_configuration_unsafe` (503): the payment-method portal would allow
  more than payment methods and invoices, so the session is refused. Report it.

Subscription state:

- `subscription_already_active` (409): the org already holds a subscription
  that Stripe has not ended. Change the plan instead of buying a second one.
- `subscription_not_found` (404): the org has no subscription.
- `subscription_not_self_serve` (409): the subscription is not managed through
  Stripe self-serve. Contact support to change it.
- `subscription_not_cancelable` (409): the subscription is `incomplete` or
  `incomplete_expired`. There is nothing to cancel.
- `subscription_not_resumable` (409): no cancellation is pending, or the
  subscription already ended.
- `subscription_cancel_pending` (409): a cancellation is pending. Resume the
  subscription before changing the plan.
- `subscription_not_changeable` (409): the subscription is `past_due`,
  `unpaid`, `paused` or `incomplete`. Update the card first.
- `subscription_plan_unchanged` (409): the requested plan is the current plan.
- `subscription_binding_conflict` (409): the org's stored subscription does not
  match Stripe's. Do not retry; report it with the `request_id`.
- `subscription_change_payment_failed` (402): the payment for the switch was
  declined. The plan is unchanged and the switch is final; paying its voided
  invoice later applies nothing. Fix the card and request the change again.
- `idempotency_in_progress` (409): the same Idempotency-Key is still being
  processed. Retry after `Retry-After` with the same key.

Stripe failures (502, `upstream_kind: stripe`):

- `stripe_checkout_failed` (502), `stripe_customer_read_failed` (502),
  `stripe_payment_method_read_failed` (502),
  `stripe_invoice_preview_failed` (502),
  `stripe_subscription_read_failed` (502) and
  `stripe_portal_session_failed` (502): Stripe did not answer. Retry later;
  nothing changed.
- `stripe_portal_configuration_read_failed` (502) and
  `stripe_portal_configuration_create_failed` (502): Stripe could not read or
  create the payment-method portal configuration. Retry later.
- `stripe_subscription_cancel_failed` (502),
  `stripe_subscription_resume_failed` (502) and
  `stripe_subscription_change_failed` (502): Stripe refused the
  change. `upstream_status` and `cause` carry Stripe's status and code. Retry
  with the same Idempotency-Key.

These entries describe server Problems. The convenience launch method reports
invalid generated fields as `LaunchValidationError` (`invalid_launch_request`,
service `synth-ai`, outcome `not_accepted`) before any HTTP request. It also
inherits `ValueError`. Direct construction of generated models can still raise
`pydantic.ValidationError`; neither is a server refusal receipt.
