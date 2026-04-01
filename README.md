# Synth AI

<!-- CI release pins: PyPI-0.9.11-orange synth-ai==0.9.11 -->

[![image](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![image](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![image](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

CLI and MCP server for [Synth Managed Research](https://docs.usesynth.ai/managed-research/intro).

## MCP

Hosted endpoint: `https://api.usesynth.ai/mcp`

Managed Research tools include **project usage** (`smr_get_usage` with `meter_quantities`; `smr_get_usage_overview` adds optional per-run spend), **run spend** (`smr_get_run_usage`: cost, billed/charged cents, `quantity_by_meter_kind`), **provider API keys** (`smr_set_provider_key`, `smr_provider_key_status` for OpenAI/Anthropic/RunPod/Modal/Tinker, etc.), **execution JSON** (`smr_merge_project_execution` for compute/runtime blobs under `project.execution`), and lane prefs (`smr_set_execution_preferences`).

Local server (requires `SYNTH_API_KEY`):

```bash
synth-ai-mcp-managed-research
```

**Codex**

```bash
codex mcp add managed-research --url https://api.usesynth.ai/mcp
```

**Claude Code**

```bash
claude mcp add --transport http managed-research https://api.usesynth.ai/mcp
```

## Documentation

[docs.usesynth.ai/managed-research](https://docs.usesynth.ai/managed-research/quickstart)

## Local development

```bash
uv sync --group dev
uv run ruff check .
# ty: use the same command as `.github/workflows/ci.yml` → job `type-check` (Lefthook runs it on staged files).
uv run ty check
```

Use **`uv run`** for Python tools (not bare **`python`** / **`python3`**). **Ruff** lints `synth_ai/`; **ty** type-checks `synth_ai/` (`[tool.ty.src]` in `pyproject.toml`). A plain `ty check` may be stricter than CI; match CI when debugging PR failures.

Optional: install [Lefthook](https://github.com/evilmartians/lefthook) and run **`lefthook install`** to run **`uv run ruff check`** and **`uv run ty check`** on staged `.py` files (see `lefthook.yml`).
