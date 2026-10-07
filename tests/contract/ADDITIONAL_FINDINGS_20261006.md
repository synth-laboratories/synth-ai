# Additional discrepancy findings — 2026-10-06

Seven new findings beyond the original consolidated audit, reproduced offline.
Source dev heads rechecked against GitHub: synth-ai `67bc4c9f`, Forge `69815acd`.
No product changes, slot claims, provider/model calls, CI wiring or spend.

**35 known-red assertions; 12 passing controls.** Every red message names EX-01
through EX-07. Tests keep the original first-class error requirements: stable
operation/code identity, explicit retryability, and parsing causes at the edge.

| ID | Severity | Confirmed behavior | Source | Acceptance selector |
|---|---|---|---|---|
| EX-01 | Medium | Explicit `retryable:false` 409/429/503 is still attempted three times, even though the raised typed error preserves false. Both GET and idempotent POST, sync and async, reproduce. Status fallback overrides the declared machine signal. | synth-ai `synth_ai/core/http/retry.py:94` | `tests/contract/test_transport_extension_law.py::test_explicit_refusal_is_attempted_once__EX01` |
| EX-02 | Low | `IDEMPOTENCY-KEY` and mixed-case equivalents are valid HTTP headers but lose their identity in retry admission. Unsafe requests with these spellings cannot use their intended idempotent retry path. | synth-ai `synth_ai/core/http/retry.py:65` | `tests/contract/test_transport_extension_law.py::test_idempotency_header_is_case_insensitive__EX02` |
| EX-03 | Low | NaN retry delays, infinite max delay, fractional attempts and boolean attempts pass constructor validation. These are not well-defined finite scheduling limits; a NaN max produces a NaN computed delay. | synth-ai `synth_ai/core/http/retry.py:21` | `tests/contract/test_transport_extension_law.py::test_retry_configuration_is_finite_and_typed__EX03` |
| EX-04 | Medium | HTTP 302 without Location is returned as successful JSON/bytes. `response.is_error` excludes redirects, so no typed boundary error is raised. Sync and async reproduce. | synth-ai `synth_ai/core/http/transport.py:394,449`; `async_transport.py:87,145` | `tests/contract/test_transport_extension_law.py::test_redirect_without_location_is_not_success__EX04` |
| EX-05 | Medium | Empty HTTP 200 declaring application/json becomes a fabricated `{}` rather than a typed contract failure. This is malformed JSON, unlike an explicit no-content operation. Both transports reproduce. | synth-ai `synth_ai/core/http/transport.py:396`; `async_transport.py:89` | `tests/contract/test_transport_extension_law.py::test_empty_json_success_is_contract_failure__EX05` |
| EX-06 | Medium | Scientific measurement value/uncertainty accepts true/false and converts them into 1.0/0.0, despite declaring number/null in the producer schema. Both fields and both booleans reproduce. | Forge `packages/forge/records.py:61` | `tests/test_measurement_extension_law.py::test_boolean_is_not_numeric_measurement__EX06` |
| EX-07 | Medium | NaN, Infinity and -Infinity in response JSON pass the strict core decoder as floating-point values. Such values cannot be serialized by the scientific canonical encoder, which uses allow_nan=False. Both transports reproduce. | synth-ai `synth_ai/core/http/transport.py:51` | `tests/contract/test_transport_extension_law.py::test_nonfinite_json_number_is_contract_failure__EX07` |

## Positive controls and ruled-out suspicion

- Declared retryable transient failures recover on attempt two, sync and async.
- The HTTP Retry-After parser already rejects NaN, infinity and negative values;
  that suspected defect is ruled out and preserved as a passing test.
- Valid zero-delay/one-attempt retry limits remain accepted.
- Finite JSON, null, boolean flags and the literal string "NaN" remain accepted.
- Scientific null, negative, zero and positive finite measurements round-trip.

These distinguish invalid numeric tokens and scientific boolean coercion from
legitimate JSON flags, numeric-looking strings, or retained negative results.

## Run and review

Use the existing backend test interpreter, with the SDK worktree on PYTHONPATH,
and explicitly select `test_transport_extension_law.py` plus
`test_transport_extension_contracts.py`. For Forge put its packages,
services/forge and app on PYTHONPATH and select both measurement extension files.
The SDK socket guard denies real network calls; all HTTP requests use
httpx.MockTransport. Retry delays in transport probes are zero and attempts
are capped at three. The tests never sleep on non-finite configuration.

Ruff E/F/I and formatting pass. Raw logs and JUnit remain under
`artifacts/discrepancy-tests-20261006/extension/`; full extension selectors and
outcomes are committed in `extension_test_results.json`. Product fixes must
follow these tests, not weaken their assertions or reclassify permanent failures
as transient. See backend tigerstyle.md and SDK core_research_migration.md.
