---
name: container-pools-harbor-rollouts
description: Use when a task needs Harbor-backed pools and rollouts through Rhodes, especially Codex-backed Harbor execution.
---

Harbor-backed pools are a Rhodes execution path accessed through the public
pools API.

- Create or update the pool through `/v1/pools`.
- Create the rollout through `/v1/pools/{pool_id}/rollouts`.
- Inspect outcome through rollout summary, usage, artifacts, and events.

Harbor pool specs commonly include Dockerfile or assembly-derived build
material, `context_tar_base64`, an entrypoint, and `harbor_agent`.

For Codex-backed Harbor tasks, set `harbor_agent.name = "codex"`. Model names
may use forms such as `openai/gpt-5.4-nano`; the in-container harness removes
the provider prefix when required.

Backend-managed auth injection is the normal path. Never embed PATs or raw
OpenAI keys in checked-in pool JSON or work around missing backend auth by
hardcoding secrets into rollout payloads.

Use Harbor guidance only for Harbor-backed pools. For openenv, horizons, or
arbitrary contracts, use the corresponding template helper.

Canonical references:

- `synth_ai.sdk.container_pools.ContainerPoolsClient`
- `synth_ai.sdk.container_pools.PoolTarget`
- `evals/containers/harbor/`
