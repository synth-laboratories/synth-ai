# Errors

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
| `invalid_inference_session` | 422 | `InvalidInferenceSessionError` |
| `provider_model_unavailable` | 422 | `ProviderPolicyError` |
| `provider_preset_unavailable` | 422 | `ProviderPolicyError` |
| `provider_policy_invalid` | 422 | `ProviderPolicyError` |
| `provider_policy_mismatch` | 422 | `ProviderPolicyError` |
| `provider_model_mismatch` | 422 | `ProviderPolicyError` |
| `provider_preset_not_allowed` | 403 | `ProviderPresetNotAllowedError` |
| `resource_request_invalid` | 422 | `ResourceAdmissionError` |
| `resource_limits_required` | 422 | `ResourceAdmissionError` |
| `grant_not_widenable` | 422 | `ResourceAdmissionError` |
| `project_not_found` | 404 | `NotFoundError` |
| `mutation_identity_required` | 422 | `InvalidRequestError` |
| `gateway_url_missing` | 503 | `UpstreamUnavailableError` |
| `codex_pool_unavailable` | 503 | `UpstreamUnavailableError` |
| `resource_authority_unavailable` | 503 | `UpstreamUnavailableError` |
| `plugin_not_qualified` | 409 | `ConflictError` |
| `gateway_issue_malformed` | 502 | `UpstreamUnavailableError` |

All are `SwarmsAPIError` subclasses; the original Problem fields are preserved.
The specialized admission classes also inherit their HTTP-status class.
