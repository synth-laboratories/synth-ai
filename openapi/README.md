# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport)
  - `container-contract-v1.yaml` (container rollout/health/info contract)
- Version: OpenAPI 3.1

## Canonical Product APIs

- Optimization systems (`/v1/systems/*`)
- Offline optimization / GEPA / MIPRO (`/v1/offline/jobs/*`)
- Online optimization / continual learning (`/v1|v2/online/sessions/*`, events support JSON polling + SSE via `Accept: text/event-stream`)
- Runtime compatibility map (`/v2/runtime/compatibility`) for logical runtime aliases
- Runtime queue controls (`/v2/runtime/systems/{system_id}/queue/*` and session aliases under `/v2/runtime/sessions/{session_id}/queue/*`) for queue contract GET/PATCH, trial/rollout CRUD, and lease orchestration
- Runtime container checkpoint controls (`/v2/runtime/containers/{container_id}/rollouts/{rollout_id}/checkpoint/{dump|restore}`) for long-horizon container checkpoint orchestration
- Optimizer observability query APIs (`/v1|v2/optimizer/events`, `/v1|v2/failures/query`, admin variants under `/v1|v2/admin/*`) for run-scoped Victoria-backed debugging and replay filters
- Hosted containers (`/api/v1/containers/*`)
- Container pools and pool-scoped rollouts (`/v1/pools/*`)
- Container runtime endpoints:
  - Core: `/health`, `/task_info`, `/rollouts` (with `/rollout` compatibility alias)
  - Long-horizon/multi-agent: `/rollouts/{rollout_id}/checkpoints*`, `/rollouts/{rollout_id}/resume`, `/rollouts/{rollout_id}/actors*`
  - Validation: `/candidates/validate` (with `/validate-candidate` compatibility alias)

## Research Track (2026-02-28)

The following surfaces have been moved to `research/feb28_2026/` for ownership and documentation. Routes remain live for compatibility but are not primary user-facing APIs. See `research/feb28_2026/` for full contracts.

- Inference APIs (`/api/inference/v1/chat/completions`, `/api/inference/jobs*`) -- marked `x-research-track`
- Graph APIs (`/api/graphs/completions`, `/graph-evolve/graphs`) -- marked `x-research-track`
- Global rollouts (`/v1/rollouts/*`) -- marked `x-research-track`, prefer pool-scoped rollouts

## Internal (de-exposed from user docs)

- SynthTunnel leases (`/api/v1/synthtunnel/leases*`) -- marked `x-internal`. Use SDK tunnel abstractions (`SynthClient().tunnels.*`) instead of calling directly.

## Codegen examples

```bash
# TypeScript
npx @openapitools/openapi-generator-cli generate \
  -i openapi/synth-api-v1.yaml \
  -g typescript-fetch \
  -o generated/synth-ts

# Python
openapi-generator-cli generate \
  -i openapi/synth-api-v1.yaml \
  -g python \
  -o generated/synth-py
```
