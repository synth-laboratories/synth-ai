# Synth Python SDK, CLI & MCP Style Guide

This guide defines the engineering standards for `synth-ai` across the public Python SDK, the `synth` CLI, the stdio/hosted MCP surface, packaging metadata, and local editor/static-analysis tooling. Core's pinned OpenAPI document (`contracts/synth-v1.openapi.yaml`) remains the wire authority; see [protocols.md](protocols.md) for wire compatibility, event streaming, and recovery semantics.

## 1. Public API Ergonomics

- **Resource-oriented namespaces:** Group operations under typed clients (`client.swarms`, `client.account`, `client.account.api_keys`, `client.billing`). Never attach raw HTTP helpers or untyped escape hatches to public clients.
- **Named, typed parameters only:** Public functions and constructors must declare every parameter explicitly (`SYN002` forbids `**kwargs` bags). Contextual or optional knobs follow `*` as keyword-only parameters.
- **No silent fallbacks or implicit coercion:** Reject unknown enum strings, invalid budgets, or missing required fields at the boundary before any network request. Never guess defaults or silently drop unrecognized fields.
- **Explicit financial & mutation boundaries:**
  - Monetary budgets accept `str | Decimal` (never binary `float`).
  - Mutating operations accept an explicit `idempotency_key: str | None` (or required `idempotency_key: str` where Core mandates it, such as `send_message`). Retries preserve identical payload bytes, request ID, and idempotency key.
- **Module hygiene:** Every new Python module starts with `from __future__ import annotations`, declares `__all__` at public boundaries, avoids parent-relative imports (`TID252`), and keeps `print()` out of library modules (`T20` is enforced on `synth_ai/` except `cli.py`).

## 2. Sync & Async Parity

- **`AsyncSynthClient` is the single implementation authority.** All I/O and orchestration live in `synth_ai/swarms.py`, `synth_ai/account.py`, and `synth_ai/billing.py`.
- **`SynthClient` delegates are generated, never hand-maintained.** After any change to an async public signature or docstring, run:

  ```sh
  uv run python scripts/gen_sync.py
  ```

  `scripts/gen_sync.py --check` and `tests/test_sync_docstrings.py` fail closed if `synth_ai/_sync.py` drifts from the async surface in signature, return type, or docstring.
- **Blocking bridge:** `SynthClient` executes async coroutines and async generators through `synth_ai._blocking.Runner` on a dedicated event-loop thread. Never nest `asyncio.run()` inside library methods.
- **Context-managed lifecycle:** Both `SynthClient` and `AsyncSynthClient` support context management (`with` / `async with`); using a client after `close()` raises `ClientClosedError` (`client_closed`).

## 3. Strict, Frozen Pydantic Boundaries

- **Generated models are strict and frozen (`SYN004`):** Wire records inherit from `synth_ai._wire.WireModel` (`strict=True`, `frozen=True`). Booleans, numeric strings, and floats are never coerced into `int` limits; field mutation raises `ValidationError`.
- **No `Any` (`SYN001`) and no opaque `dict[str, object]` maps (`SYN003`):** Use generated models, `StrEnum` values, or recursive `pydantic.JsonValue` exclusively at intentionally open JSON boundaries.
- **No validation bypasses (`SYN005`):** Do not call `model_construct`, legacy `construct`, `model_copy(update=...)`, or `strict=False` on SDK models.
- **Static & consumer type contracts:**
  - `scripts/lint_standards.py` parses ASTs without importing `synth_ai` and enforces `SYN001`–`SYN005` across both handwritten and generated code.
  - `scripts/check_type_contracts.py` runs Astral `ty` against `typing_tests/accepted.py` (positive `assert_type` checks) and `typing_tests/rejected.py` (exact expected diagnostic locations and rule codes).

## 4. Typed `SynthError` Hierarchy & Error Codes

- **Branch on `error.code`, never prose.** Every SDK error inherits from `synth_ai.errors.SynthError` and exposes a stable `kind: ErrorKind` and machine-readable `code: str`.
- **Server refusals (`SwarmsAPIError`):** Preserve the full RFC 9457 `Problem` (`type`, `title`, `status`, `detail`, `instance`, `code`, `service`, `stage`, `outcome`, `retryable`, extensions), `request_id`, `idempotency_key`, and `retry_after_seconds`. Status-specific and code-specific subclasses (`_BY_CODE`, `_BY_STATUS`) map refusals deterministically without discarding unknown codes.
- **Local launch validation (`LaunchValidationError`):** All contract-invalid local launch inputs (oversize/blank/NUL message, invalid budget, invalid `domicile_requirement`, profile/`research_task` mismatch, missing `launch_menu_revision`, invalid `optimizer`, or Pydantic `ValidationError` inside `prepare_launch`/`launch`/`run`) raise `LaunchValidationError` (`InvalidRequestError` + `ValueError`, `code="invalid_launch_request"`, `status=422`, `service="synth-ai"`, `outcome="not_accepted"`) before any HTTP call. Never leak raw `pydantic.ValidationError` from high-level launch helpers.
- **Documentation parity:** Every Problem example and operation refusal in `contracts/synth-v1.openapi.yaml`, and every code in `_BY_CODE`, must have a matching entry and recovery action in [errors.md](errors.md) (verified by `tests/test_error_guide.py`).

## 5. CLI Flags, JSON & Exit-Code Contracts

- **Thin task-verb layer:** `synth` (`synth-ai`) delegates exclusively to `AsyncSynthClient` (verified by `scripts/check_surface_parity.py`). Never duplicate transport or policy logic in `synth_ai/cli.py`.
- **Goal input without OS `E2BIG`:** `synth run` accepts a positional `goal`, `--goal-file PATH` (alias `--task-file PATH`), `--goal-file -` (stdin), or piped stdin. Stdin and file reads are bounded to `MESSAGE_MAX_BYTES + 1` bytes so `>1 MiB` goals never fail with OS `E2BIG` and are refused before any upload or HTTP request.
- **Structured output:**
  - With `--json`, stdout carries exactly one JSON document (`model_dump_json(exclude_unset=True)`), except `logs -f --json` and `messages --follow --json` which emit one JSON object per line.
  - Errors always go to stderr: `code: one-line detail` in human mode, or a single JSON object carrying `code` (plus `retry_after_seconds` when present) under `--json`.
- **Stable exit codes (`synth_ai/cli.py`):**

  | Exit code | Constant | Meaning |
  | --- | --- | --- |
  | `0` | `EXIT_OK` | Command succeeded |
  | `1` | `EXIT_CLIENT_ERROR` | Usage error, local validation refusal, unreadable file, or transport failure |
  | `2` | `EXIT_WAIT_TIMEOUT` | `wait` / `run --wait` timed out; run remains active |
  | `3` | `EXIT_RUN_UNSUCCESSFUL` | Run finished `failed` or `cancelled` |
  | `4` | `EXIT_SERVER_REFUSAL` | Typed server Problem refusal |
  | `5` | `EXIT_RUN_PAUSED` | Observed run entered `paused` |

## 6. MCP Tool Schema & `isError` Conventions

- **One shared catalog:** `synth_ai/mcp/tools.json.gz` is generated from `contracts/mcp-tools.json` so stdio (`synth-ai-mcp`) and hosted `/mcp` expose identical tool names, scopes, and JSON Schemas.
- **Typed `isError` refusals:** The stdio server registers `server.call_tool(validate_input=False)` and validates arguments itself so schema violations return `CallToolResult(isError=True)` containing a typed Problem JSON (`upload_format_unsupported` or `tool_arguments_invalid`, status 422) rather than unstructured library text.
- **Retry guidance without HTTP headers:** `synth_ai.mcp.server.refusal()` attaches `retry_after_seconds` beside the Problem fields when the server returned a `Retry-After` header.
- **Hosted `/mcp` bearer auth:** Streamable-HTTP callers (`HostedEndpoint` in `scripts/acceptance/mcp_transport.py`) require an `http://` or `https://` URL and a non-empty opaque bearer token without whitespace or control characters, sent as `Authorization: Bearer <token>`. Missing/rejected tokens fail closed with a typed transport refusal.

## 7. PyPI Metadata & Packaging Standards

- **Required `[project.urls]`:** `pyproject.toml` must declare HTTPS URLs for `Homepage`, `Documentation`, `Repository`, `Changelog`, and `Issues`.
- **No relative links in `README.md`:** All links in `README.md` must be absolute `https://` URLs (or same-document `#anchors`) so rendered PyPI pages have zero broken links.
- **Fail-closed package qualification:** `make package-check` builds the wheel and sdist (`uv build`), validates metadata (`twine check dist/*`), and runs `scripts/check_package.py`, which enforces:
  - `[project.urls]` presence and `https://` scheme in both `pyproject.toml` and wheel `METADATA`.
  - Zero relative markdown links in both `README.md` and wheel `METADATA`.
  - PEP 561 `synth_ai/py.typed` marker.
  - Size budgets (`< 150 KiB` wheel, `< 480 KiB` package source).
  - Isolated fresh-venv installation for both `synth-ai` and `synth-ai[mcp]`.

## 8. Canonical Quality Gate & Local Zed LSP Setup

### One Canonical `make check`

`Makefile`, `lefthook.yml` (pre-push hook), and `scripts/qualify_release.py` all run the same five static checks:

```sh
make check
```

This executes:

1. `uv run --locked --extra mcp ruff check synth_ai scripts tests examples typing_tests`
2. `uv run --locked --extra mcp ruff format --check synth_ai scripts tests examples typing_tests`
3. `uv run --locked --extra mcp ty check` (covers `synth_ai`, `scripts`, `examples`, `tests`, and `typing_tests/accepted.py` with `all = "error"`)
4. `uv run --locked --extra mcp python scripts/lint_standards.py`
5. `uv run --locked --extra mcp python scripts/check_type_contracts.py`

Install the pre-push hook once per clone:

```sh
lefthook install
```

### Zed Editor Setup (Astral `ruff` + `ty`)

The repository ships [.zed/settings.json](../.zed/settings.json), which configures Zed to:

- Enable Astral `ty` and `ruff` for Python while disabling `basedpyright`/`pyright`.
- Run Ruff import sorting, autofixes, and formatting on save (`lineLength = 100`).
- Point directly to `.venv/bin/ty` and `.venv/bin/ruff` so editor diagnostics match the exact lockfile-pinned versions in `uv.lock`.

Setup in a fresh checkout:

```sh
uv sync --locked --extra mcp
```

If you open files outside a synced `.venv`, you can also install the tools globally for Zed fallback:

```sh
uv tool install ruff ty
```
