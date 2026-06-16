# Changelog

All notable changes to the `synth-ai` package are documented here.

## 0.11.2 — 2026-06-16

### Added

- **`SynthClient().research`** — public Managed Research + Research Factory namespace (`ResearchClient`).
- MCP entrypoint **`synth-ai-managed-research-mcp`** (stdio) alongside hosted HTTP MCP at `https://api.usesynth.ai/mcp`.
- Optional extra **`synth-ai[research]`** for Research/MCP dependencies.

### Changed

- **Canonical install:** `pip install "synth-ai[research]==0.11.2"` (or `uv add "synth-ai[research]"`).
- **Canonical Python entry:** `from synth_ai import SynthClient` → `client.research`.
- MCP tool enumeration and SDK namespaces reconciled to the shipped backend OpenAPI contract.

### Removed (breaking vs 0.11.1)

The following **public SDK/MCP surfaces were removed** because they are not part of the launch contract. Use the replacements below.

| Removed area | Replacement |
| --- | --- |
| Objective / milestone / experiment CRUD helpers | Objectives are **runtime-managed** during runs; inspect with `smr_get_run_work_graph`, `smr_objectives` (read), run observability |
| Project economics convenience wrappers on deprecated paths | **`smr_get_project_usage`**, **`smr_get_run_usage`**, resource-limit progress tools, billing entitlements |
| Connect-git / legacy GitHub setup SDK modules | GitHub OAuth via MCP `smr_setup_github_*` where enabled; attach repos via console or `smr_work_repos_*` |
| Standalone **`managed-research`** PyPI package as the primary story | **`synth-ai[research]`** only; legacy `ManagedResearchClient` import remains under `synth_ai` for advanced/low-level callers |
| Data-factory trigger/publish MCP tools (`smr_trigger_data_factory`, …) | Factory programs via hosted control plane + documented factory MCP tools |

### Migration (0.11.1 → 0.11.2)

```bash
uv add "synth-ai[research]==0.11.2"
export SYNTH_API_KEY="sk_..."
```

```python
# Before (0.11.1 docs / standalone package)
# from managed_research import ManagedResearchClient
# client = ManagedResearchClient()

# After
from synth_ai import SynthClient

research = SynthClient().research
projects = research.projects.list()
run = research.runs.start(
    "Inspect the repo and leave evidence.",
    host_kind="daytona",
    work_mode="directed_effort",
    providers=[{"provider": "openrouter"}],
    runbook="lite",
)
```

- **MCP stdio:** `uv tool install "synth-ai[research]"` then run `synth-ai-managed-research-mcp`.
- **Errors:** launch denials return typed exceptions / MCP tool errors with `402` (insufficient credits) and `429` (limits). See [Preflight and Errors](https://docs.usesynth.ai/managed-research/preflight-and-errors).
