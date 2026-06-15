# Synth SDK Logic Specification

## 1. Canonical Front Door

### 1.1 Required entry points

The SDK MUST expose exactly two canonical entry points:

- `synth_ai.SynthClient` (sync)
- `synth_ai.AsyncSynthClient` (async)

Both clients MUST share the same namespace topology.

### 1.2 Namespace topology

Both clients expose the following first-class namespaces:

- `optimization`
- `inference`
- `graphs`
- `verifiers`
- `pools`
- `tunnels`
- `container`

### 1.3 Construction rules

- `api_key` is required, explicitly or via `SYNTH_API_KEY`.
- `base_url` defaults to Synth backend base when omitted.
- `timeout` applies to network operations.

## 2. Optimization Model

### 2.1 Resources

`optimization` is split into three primitives:

- `systems`
- `offline`
- `online`

### 2.2 Systems

`systems` manages canonical policy-optimization system resources.

Required methods:

- `create`
- `get`
- `list`

### 2.3 Offline jobs

`offline` manages batch optimization jobs.

Required methods:

- `create`
- `get`
- `list`

Job instance behavior includes:

- `status`
- `events`
- `artifacts`
- state transitions via `pause`/`resume`/`cancel`.

### 2.4 Online sessions

`online` manages reward-driven online optimization sessions.

Required methods:

- `create`
- `get`
- `list`

Session instance behavior includes:

- `submit_reward`
- `events`
- state transitions via `pause`/`resume`/`cancel`.

## 3. Inference Model

### 3.1 Chat completions

`inference.chat.completions.create` MUST accept OpenAI-compatible inputs:

- `model`
- `messages`
- standard generation controls (`temperature`, `max_tokens`, etc.)

The request/response model SHOULD remain OpenAI-compatible to support polyglot clients.

### 3.2 Inference jobs

`inference.jobs` covers async environment-backed jobs.

Required methods:

- `create`
- `create_from_request`
- `create_from_path`
- `get`
- `list_artifacts`
- `download_artifact`

## 4. First-Class Containers, Pools, Tunnels

### 4.1 Containers

`container` is a composed namespace with:

- `hosted`: hosted container CRUD
- `local`: in-process/local container helpers
- `create`/`connect`: direct local container entry points

### 4.2 Container pools

`pools` is first-class and MUST include structured subclients:

- `uploads`
- `data_sources`
- `assemblies`
- `rollouts`
- `tasks`
- `metrics`
- `agent_rollouts`
- `skills`

It MUST provide target-specific templates:

- `pools.harbor`
- `pools.openenv`
- `pools.horizons`
- `pools.arbitrary`

Each target adapter MUST support assembling from data source and reassembly.

### 4.3 Tunnels

`tunnels` is first-class and MUST include:

- `tunnels.open`
- `tunnels.open_for_app`
- `tunnels.synth` specialized `SynthTunnel`

Backend variants:

- `SynthTunnel`
- `CloudflareQuickTunnel`
- `CloudflareManagedTunnel`
- `CloudflareManagedLease`
- `Localhost`

`SynthTunnel` MUST expose `worker_token` semantics for container auth in relay mode.

## 5. Sync/Async Coherence

### 5.1 Behavioral parity

Sync and async clients MUST maintain parity across canonical resources.

### 5.2 Naming parity

Method names and argument shape MUST remain equivalent between sync and async variants, except coroutine semantics.

## 6. Contract Invariants

- Canonical docs and examples MUST use `SynthClient` / `AsyncSynthClient`.
- Canonical docs MUST treat `container_url` as the primary container endpoint field.
- Reward payloads use canonical reward fields (`reward_info.*`), not legacy aliases.
- New SDK symbols SHOULD land in canonical namespaces before legacy wrappers.

## 7. Non-Goals

- This file does not define backend implementation internals.
- This file does not describe deprecated, compatibility-only wrappers.
