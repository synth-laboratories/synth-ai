---
name: container-pools-overview
description: Use when driving Rhodes-backed container pools through the public synth-ai API instead of inventing custom Docker or rollout flows.
---

Use the public synth-ai pools surface as the source of truth.

- Front door: `SynthClient.pools` and `synth_ai.sdk.container_pools.ContainerPoolsClient`
- Auth: `SYNTH_API_KEY`, or explicit `api_key=...`
- Base URL: use `BACKEND_URL_BASE` and `normalize_backend_base`

Public pools coverage lives under `/v1/pools/*`, including uploads, data sources,
assemblies, pool CRUD, rollouts, tasks, and metrics.

Use the existing `ContainerPoolsClient` namespaces instead of rebuilding them:
`uploads`, `data_sources`, `assemblies`, `rollouts`, `agent_rollouts`, `tasks`,
`harbor`, `openenv`, `horizons`, and `arbitrary`.

Choose the template helper matching the runtime contract: `client.harbor`,
`client.openenv`, `client.horizons`, or `client.arbitrary`.

Only send canonical rollout fields. Use `CANONICAL_ROLLOUT_REQUEST_KEYS` and
validate requests with `validate_pool_rollout_request(...)`. Keep extensions
aligned with `backend/rhodes/contracts.py`.

Hosted containers live under `SynthClient.container.hosted`; container pools
live under `SynthClient.pools`. They are different operations.
