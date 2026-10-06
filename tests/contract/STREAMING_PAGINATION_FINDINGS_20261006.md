# Streaming, MCP and pagination findings — 2026-10-06

Thirteen additional confirmed defects, EX-08 through EX-20. These extend the
original audit and EX-01–EX-07; no product fixes are included.

Source: synth-ai dev `67bc4c9ff7e4385dbf1b4fb759430bbe07b67ea9`, rechecked through
GitHub SSH. New tests: **52 known-red assertions and 18 passing controls**.
All 13 finding IDs appear in the red failure messages. Source fixtures use
httpx.MockTransport and the socket guard; no services, credentials, provider
calls, sleeps, slot claims, CI or spend.

SSE cases were checked against the primary
[WHATWG parsing/interpretation rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream),
including the response media-type processing rule. MCP/retry/page guarantees are
explicit in `testing/specifications/sdk/core_research_migration.md`, section
“October 5 stable boundary qualification.” Report severity describes the
consumer failure, not a claim that deployed servers currently emit bad frames.

## SSE findings

Source module: `synth_ai/core/http/streaming.py`, shared by both transports;
response metadata checks belong to `core/http/transport.py` and its async peer.
Each row is reproduced through both real SDK transports over fake HTTP responses.

| ID | Severity | Reproduction | Source | Test in `tests/contract/test_streaming_extension_law.py` |
|---|---|---|---|---|
| EX-08 | Medium | Invalid retry fields raise or become numeric directives instead of being ignored. | streaming.py:47 | `test_invalid_retry_field_is_ignored__EX08` |
| EX-09 | Medium | NUL-containing IDs replace the previous replay cursor. | streaming.py:45 | `test_nul_event_id_preserves_previous_id__EX09` |
| EX-10 | Low | Empty event names replace the default message route. | streaming.py:41,59 | `test_empty_event_type_uses_message__EX10` |
| EX-11 | Medium | EOF dispatches unfinished frames without the final blank line. | streaming.py:77,88 | `test_eof_does_not_dispatch_unterminated_event__EX11` |
| EX-12 | Medium | A leading UTF-8 BOM drops the first event. | streaming.py:32 | `test_leading_bom_does_not_drop_first_event__EX12` |
| EX-13 | Medium | Missing or incorrect response media types are decoded as SSE. | transport.py:525; async_transport.py:199 | `test_wrong_stream_media_type_is_typed_failure__EX13` |

## Closed boundaries and pagination

| ID | Severity | Confirmed behavior | Source | Test in `tests/contract/test_boundary_extension_law.py` |
|---|---|---|---|---|
| EX-14 | Medium | Both exported registry.call_tool and public ResearchMcpServer.call_tool invoke an effect-counting handler when declared required keys are absent or unknown keys violate additionalProperties=false. This proves the shared gate fails before handlers, independently of whichever argument checks an individual handler happens to implement. | mcp/research/registry.py:313; server.py:496 | `test_closed_mcp_arguments_refuse_before_handler__EX14` |
| EX-15 | Medium | Different identities in duplicate case-insensitive headers, header versus body, or two body idempotency fields are silently resolved by first-match precedence. Replay admission should refuse ambiguity. | core/http/retry.py:65 | `test_ambiguous_idempotency_identity_refuses__EX15` |
| EX-16 | Medium | Missing/null/object/string item arrays become successful empty pages, hiding malformed evidence as “no data.” | sdk/pagination.py:25 | `test_invalid_items_do_not_become_empty_page__EX16` |
| EX-17 | Medium | Explicit terminal next_cursor=null falls back to an old cursor. A full terminal Projects/Swarm page also invents another cursor and has_more=true from its last item. | sdk/pagination.py:29; sdk/research/projects.py:70; swarms.py:83 | `test_terminal_null_is_not_previous_cursor__EX17`; `test_full_terminal_page_does_not_invent_continuation__EX17` |
| EX-18 | Medium | String/numeric/null has_more is silently converted with bool(); the string “false” becomes true. | sdk/pagination.py:31 | `test_has_more_is_strict_boolean__EX18` |
| EX-19 | Low | Opaque cursors are stripped by both core and SDK parsers; SDK also converts numeric/boolean/object cursors into invented strings. | core/contracts/pagination.py:24; sdk/pagination.py:30 | `test_opaque_cursor_is_preserved_exactly__EX19`; `test_nonstring_cursor_is_not_fabricated__EX19` |
| EX-20 | Medium | has_more=true with no next cursor, and has_more=false with a next cursor, are accepted despite contradictory continuation. | sdk/pagination.py:29–32 | `test_contradictory_continuation_refuses__EX20` |

## Passing controls and validation

Valid LF/CR/CRLF streams preserve multiline data, ASCII retry values, explicit
and inherited cursor IDs, custom event names, empty data and explicit cursor
reset. Comment-only/unknown-field blocks emit nothing. Correct SSE content types
with charset parameters remain valid.

Valid closed tool arguments reach the handler exactly once through both entry
points. Equal duplicate idempotency identities remain valid. Bare legacy arrays,
array-valued items/data and consistent terminal/continuation pages round-trip.
These controls rule out incidental fixture failures and preserve compatibility.

Ruff E/F/I plus formatting pass. Exact node/status evidence is committed in
`streaming_boundary_test_results.json`; raw logs and JUnit are retained under
`artifacts/discrepancy-tests-20261006/streaming/`.

Run the existing backend test interpreter with this SDK worktree on PYTHONPATH
and explicitly select these four files:

```sh
python -m pytest tests/contract/test_streaming_extension_law.py \
  tests/contract/test_streaming_extension_contracts.py \
  tests/contract/test_boundary_extension_law.py \
  tests/contract/test_boundary_extension_contracts.py -q --tb=short
```

No expected-failure markers hide the red laws. Wrong SSE response media types
must fail with a first-class ContractMismatchError retaining the operation and
non-retryable status. Invalid optional SSE fields must follow their documented
ignore rule while retaining valid data. Product fixes must preserve these
assertions and the passing controls.
