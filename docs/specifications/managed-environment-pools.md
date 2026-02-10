# Managed Environment Pools

Managed pools provide a "data in, optimized pipeline out" experience: users upload
files (or point to a GitHub repo / S3 bucket), an LLM agent analyzes the data, and
the system assembles a fully configured environment pool ready for GEPA/Eval.

**Owner:** `rust_backend/src/managed_env_pool/`
**Agent execution:** `sandbox-agent` (wraps OpenCode CLI into HTTP API)
**Pool primitives:** `rhodes-core` crate

## Architecture

```
User (SDK / CLI / Dashboard)
        |
        | file upload / GitHub URL / S3 config
        v
+---------------------------+
|   Synth Backend (Python)  |   auth, billing, org scoping
+------------+--------------+
             |
             v
+--------------------------------------------------+
|  rust_backend/src/managed_env_pool/              |
|                                                  |
|  1. Data Ingestion                               |
|     FileIngestor | GitIngestor | S3Ingestor      |
|     -> DataManifest (inventory, hashes, metadata) |
|                                                  |
|  2. Assembly Engine                              |
|     sandbox-agent HTTP -> OpenCode -> Claude LLM |
|     Fallback: opencode CLI -> heuristic          |
|     -> AssemblyResult (env type, files, config)  |
|                                                  |
|  3. Pool Provisioner                             |
|     PoolConfig + generated files -> PoolRegistry |
+--------------------------------------------------+
             |
             v
+--------------------------------------------------+
|  rhodes-core (imported crate)                    |
|  PoolConfig, PoolRegistry, SandboxAgentClient    |
+--------------------------------------------------+
```

## Data Flow

1. **Ingest** -- user provides data via upload, GitHub URL, or S3 reference.
   A `DataManifest` is produced (file listing with sizes, SHA256 hashes, MIME types).

2. **Assemble** -- the assembly engine sends the manifest plus sample file
   contents to an LLM agent (Claude via OpenCode via sandbox-agent). The agent
   selects an environment type (harbor, openenv, browser, archipelago) and
   generates configuration files (task.toml, Dockerfile, localapi.py, etc.).

3. **Provision** -- the assembly output is converted to a `PoolConfig`, generated
   files are stored in blob storage, and the pool is registered with the pool
   registry.

## API Endpoints

All endpoints are org-scoped via API key (`x-user-api-key` header).

### Data Sources

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/managed-pools/uploads` | Generate presigned upload URL |
| POST | `/v1/managed-pools/data-sources` | Create data source (multipart upload, GitHub, or S3) |
| GET | `/v1/managed-pools/data-sources` | List data sources for org |
| GET | `/v1/managed-pools/data-sources/:id` | Get data source details |
| PUT | `/v1/managed-pools/data-sources/:id` | Update data source (re-upload or change git ref) |
| DELETE | `/v1/managed-pools/data-sources/:id` | Delete data source |
| GET | `/v1/managed-pools/data-sources/:id/manifest` | Get parsed data manifest |
| POST | `/v1/managed-pools/data-sources/:id/refresh` | Re-fetch from GitHub/S3 and regenerate manifest |

### Assembly

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/managed-pools/assemble` | Start assembly from data source |
| GET | `/v1/managed-pools/assembly/:id` | Get assembly status and result |
| GET | `/v1/managed-pools/assembly/:id/generated` | Get generated files from completed assembly |
| GET | `/v1/managed-pools/assembly/:id/events` | SSE stream of assembly lifecycle events |

**Assembly request body:**
```json
{
  "data_source_id": "ds_...",
  "exclusion_patterns": ["*.log", "node_modules/**"],
  "target_hint": "openenv",
  "agent_model": "claude-sonnet-4-20250514"
}
```

### Managed Pools

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/managed-pools` | Create pool from completed assembly |
| GET | `/v1/managed-pools` | List pools for org |
| GET | `/v1/managed-pools/:id` | Get pool details |
| DELETE | `/v1/managed-pools/:id` | Delete pool |
| GET | `/v1/managed-pools/:id/status` | Get pool status |
| GET | `/v1/managed-pools/:id/generated` | Get generated files |
| PUT | `/v1/managed-pools/:id/exclusions` | Update file exclusion patterns |
| POST | `/v1/managed-pools/:id/re-assemble` | Trigger re-assembly |

## Assembly Engine

The assembly engine has a 3-tier fallback:

1. **sandbox-agent HTTP** (preferred) -- creates a session on the sandbox-agent
   server, sends the prompt to OpenCode which calls the Claude LLM, polls
   universal-schema events via HTTP, collects text deltas, and parses the
   structured JSON output.

2. **opencode CLI** -- spawns OpenCode as a subprocess with `--format json`,
   pipes the prompt via stdin, and parses JSONL events from stdout.

3. **heuristic** -- pattern-matches on file names and content to guess the
   environment type and generate scaffold files without an LLM.

### sandbox-agent Integration

The sandbox-agent is a Rust HTTP server that wraps coding agents (Claude Code,
OpenCode, Codex) into a unified session-based API with universal event schema.

**Session lifecycle (from rust_backend perspective):**

```
1. POST /v1/sessions/{id}          -- create session (agent=opencode, model, permissionMode)
2. POST /v1/sessions/{id}/messages -- send assembly prompt
3. GET  /v1/sessions/{id}/events   -- poll universal events (offset pagination)
4. POST /v1/sessions/{id}/permissions/{pid}/reply -- auto-approve permissions
5. POST /v1/sessions/{id}/terminate -- cleanup
```

**Event processing:** The engine collects `item.delta` events for text content
and watches for `session.idle` or `session.ended` to detect completion.
`permission.requested` events are auto-approved with "always".

**OpenCode SSE specifics:** OpenCode v1.1.18+ wraps SSE events in a
`{"directory":"...","payload":{...}}` structure. The sandbox-agent unwraps the
`payload` before converting to universal schema. A typed dispatch function
(`opencode_typed_event`) routes events by the `type` field to avoid issues with
the generated `#[serde(untagged)]` Event enum where greedy variants swallow
events.

### Agent Output Format

The agent is prompted to output structured JSON:

```json
{
  "environment_type": "openenv",
  "reasoning": "This is a text classification task...",
  "generated_files": {
    "task.toml": "...",
    "localapi.py": "...",
    "Dockerfile": "..."
  },
  "pool_config": {
    "pool_id": "...",
    "type": "openenv",
    "concurrency": 4
  },
  "task_definitions": [...]
}
```

The parser is lenient: `pool_config` and `task_definitions` are optional with
fallback defaults. The JSON extractor finds the last valid JSON block in the
agent output to avoid capturing prompt echo text.

## Environment Types

| Type | Pool Backend | Use Case | Required Files |
|------|-------------|----------|----------------|
| `openenv` | OpenEnv | Classification, RL, sequential decision tasks | `localapi.py`, `task.toml`, `Dockerfile` |
| `harbor` | Sandbox | Code execution with test suites | `task.toml`, `tests/test.sh`, `instruction.md` |
| `browser` | Browser/Kernel | Web automation tasks | `browser_task.yaml` |
| `archipelago` | Archipelago | Multi-step tasks with MCP tools | `mcp_config.json` |

## Feature Flags

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MANAGED_POOLS` | `true` | Enable managed pools feature |
| `MANAGED_POOLS_GITHUB` | `false` | Enable GitHub data sources |
| `MANAGED_POOLS_S3` | `false` | Enable S3 data sources |
| `MANAGED_POOLS_AUTO_REASSEMBLY` | `false` | Auto-reassemble on data/exclusion changes |
| `MANAGED_POOLS_SANDBOX_AGENT_URL` | (none) | sandbox-agent server URL (enables agent path) |
| `MANAGED_POOLS_ASSEMBLY_AGENT` | `opencode` | Agent to use for assembly |
| `MANAGED_POOLS_ASSEMBLY_MODEL` | (none) | Default model for assembly agent |
| `MANAGED_POOLS_SANDBOX_PERMISSION_MODE` | `bypass` | Permission mode for agent sessions |
| `MANAGED_POOLS_ASSEMBLY_TIMEOUT_SEC` | `600` | Assembly timeout in seconds |

## Assembly Events (SSE)

Events streamed via `GET /v1/managed-pools/assembly/:id/events`:

| Event Type | Description |
|-----------|-------------|
| `assembly.started` | Assembly job created |
| `assembly.agent_spawned` | Agent session created |
| `assembly.agent_failed` | Agent path failed (falling back) |
| `assembly.files_generated` | Generated files stored |
| `assembly.completed` | Assembly finished successfully |
| `assembly.failed` | Assembly failed with error |

## Directory Structure

```
rust_backend/src/managed_env_pool/
├── mod.rs              # ManagedPoolService, feature flags, orchestration
├── types.rs            # DataSource, AssemblyRecord, ManagedPool, etc.
├── upload.rs           # Presigned upload URL generation
├── api/
│   ├── mod.rs
│   ├── routes.rs       # 21 Axum route handlers
│   ├── requests.rs     # Request DTOs
│   └── responses.rs    # Response DTOs
├── assembly/
│   ├── mod.rs
│   ├── engine.rs       # 3-tier assembly engine (agent -> CLI -> heuristic)
│   ├── prompts.rs      # LLM prompt template
│   ├── parser.rs       # Lenient JSON parser for agent output
│   └── generators/     # Heuristic fallback generators
│       ├── openenv.rs  # OpenEnv localapi.py generation
│       ├── harbor.rs   # Harbor scaffold
│       ├── browser.rs  # Browser scaffold
│       └── archipelago.rs
├── ingestion/
│   ├── mod.rs
│   ├── file.rs         # File/archive ingestor
│   ├── github.rs       # Git clone ingestor
│   ├── s3.rs           # S3 bucket ingestor
│   └── manifest.rs     # DataManifest builder
├── provisioner/
│   ├── mod.rs
│   └── storage.rs      # Generated file blob storage
└── store/
    ├── mod.rs
    └── postgres.rs     # DB persistence for data sources, assemblies, pools
```

## MVP: Banking77 Flow

1. User uploads `test.csv` + `categories.json` (Banking77 dataset)
2. System ingests files and produces `DataManifest`
3. User calls `POST /v1/managed-pools/assemble`
4. Assembly engine sends manifest + file samples to Claude via sandbox-agent
5. Agent determines: environment_type = openenv, generates localapi.py with
   text classification logic, Dockerfile, task.toml
6. Pool is provisioned as PoolType::Openenv with concurrency=4
7. User runs GEPA against the managed pool
