# Synth HTTP + OpenAI Compatibility Specification

## 1. Purpose

This spec defines the HTTP contract used by the canonical SDK and the OpenAI-compatible chat surface.

Machine-readable contract: `openapi/synth-api-v1.yaml`.

## 2. Authentication

All authenticated endpoints accept bearer auth:

- `Authorization: Bearer <SYNTH_API_KEY>`

Some clients also send `x-api-key` for compatibility; servers should tolerate both when configured.

## 3. API Areas

### 3.1 Optimization (`/v1/*`)

Canonical resources:

- `/v1/policy-optimization/systems`
- `/v1/offline/jobs`
- `/v1/online/sessions`

These are the default targets for `SynthClient.optimization`.

### 3.2 Inference

OpenAI-compatible chat completions:

- `POST /api/inference/v1/chat/completions`

Contract behavior:

- Request shape mirrors OpenAI Chat Completions.
- Response shape mirrors OpenAI Chat Completions.
- Model IDs are normalized by Synth routing.

Inference jobs:

- `POST /api/inference/jobs`
- `GET /api/inference/jobs/{job_id}`
- `GET /inference/jobs/{job_id}/artifacts/{artifact_id}`

### 3.3 Graphs and verifiers

- `POST /api/graphs/completions`
- `GET /graph-evolve/graphs`

Verifier requests in the SDK are layered on top of graph completions with verifier-shaped inputs.

### 3.4 Containers

Hosted container resources:

- `/api/v1/containers`

### 3.5 Container pools

Pools and rollout control plane:

- `/v1/pools/*`
- `/v1/rollouts/*`

Includes uploads, data sources, assemblies, pools, rollouts, tasks, metrics, and SSE event streams.

### 3.6 SynthTunnel lease API

- `POST /api/v1/synthtunnel/leases`
- `DELETE /api/v1/synthtunnel/leases/{lease_id}`

## 4. Streaming

SSE/event stream endpoints return `text/event-stream` and are consumed as line-delimited `data:` payloads:

- `/v1/offline/jobs/{job_id}/events`
- `/v1/online/sessions/{session_id}/events`
- `/v1/pools/assemblies/{assembly_id}/events`
- `/v1/pools/{pool_id}/rollouts/{rollout_id}/events`

## 5. Compatibility Policy

- Canonical docs must refer to the endpoints in `openapi/synth-api-v1.yaml`.
- OpenAI-compatible schema support is maintained on Synth inference routes.
- Legacy path aliases are outside this spec unless explicitly promoted.
