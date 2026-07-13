# Research SDK reference (hero surface)

Canonical entry: `SynthClient().research`

## Namespaces

| Namespace | Methods / nested |
| --- | --- |
| `research.factories.tag` | `sessions.create`, `sessions.messages.send`, `scopes.get_default` |
| `research.limits` | `get()` |
| `research.secrets` | `list`, `create`, `delete` |
| `research.projects` | `create`, `list`, `get`, `update`, `archive` |
| `research.projects.setup` | `get`, `prepare`, onboarding helpers |
| `research.projects.workspace` | `get`, `upload`, `upload_directory`, `download`, `inputs` |
| `research.projects.repos` | `attach` |
| `research.projects.git` | `get`, `connect` |
| `research.projects.objectives` | `list` |
| `research.projects.milestones` | `list` |
| `research.runs` | `create`, `check_preflight`, `get`, `open`, `state`, `list`, `wait`, lifecycle verbs |
| Run session readouts | `usage`, `progress`, `snapshots`, `events`, `tasks`, `message_queue`, `work_products`, `artifacts`, `results`, `logs`, `orchestrator`, `trained_models`, `workspace`, `transcript` |

## CLI smoke

```bash
synth-ai research limits get
synth-ai research tag smoke
synth-ai research smoke
```

## Eval harness

Eval code imports `evals/reportbench/synth_client.py`:

- `build_hero_research()` → `SynthClient().research`
- `build_eval_sdk_client()` → `ReportbenchHeroClient` (hero namespaces; default `EVAL_USE_HERO_SDK=1`)
- `build_reportbench_client()` in `reportbench/hero_driver.py` for typed ReportBench driver

## MCP

Hosted MCP exposes `smr_*` tools plus `research_*` aliases (same handlers).
